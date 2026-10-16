"""Interpret e2e measurement artifacts and build the stable result contract.

Operational recovery, recipe classification and environment-derived policy are
provided by the entrypoint; importing this module does not initialize a run.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Callable

from .reports import _existing_report_path

SCHEMA_VERSION = 2
# The published speedup must agree with the baseline/final pair in result.json.
SPEEDUP_SELF_CONSISTENCY_TOL = 1e-3
WORKLOAD_KIND_AGENTX = "agentx_trace_replay"
WORKLOAD_KIND_SYNTHETIC = "synthetic_isl_osl"
WORKLOAD_KIND_UNKNOWN = "unknown"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    """num/den rounded to 4dp, or None when either side is missing/non-positive."""
    try:
        n, d = float(num or 0.0), float(den or 0.0)
    except (TypeError, ValueError):
        return None
    return round(n / d, 4) if (n > 0 and d > 0) else None


def read_orchestrator_baseline_lifecycle(h: dict) -> tuple[float, str]:
    """Read Hyperloom's baseline anchor AND its thermal state from ``state.json``.

    Hyperloom does NOT keep a cold and a hot baseline side by side. Its double-run
    baseline runs one full warmup round, DISCARDS it, and anchors on the round
    measured against the now-hot server — so ``state.baseline_tput`` (the
    leaderboard denominator, forwarded to us as ``handoff.raw_baseline_tput``) IS
    the hot number. There is no ``baseline_hot_tput`` key anywhere in Hyperloom;
    reading one only ever returned nothing.

    What varies is whether the double run happened at all, and Hyperloom records
    that in two fields written by the same writeback that promotes the anchor:

    * ``baseline_warm_runtime_sec`` — the measure round's wall-clock. Set only on
      the double-run path and explicitly zeroed when a later baseline lands
      without one, so ``> 0`` is positive evidence that a warmup round preceded
      the anchor.
    * ``baseline_measure_round_dropped`` — True when the budget could not fund
      the hot pass and the session had to keep the COLD figure as its anchor.

    ``state.json`` lives at the SESSION dir (an ancestor of ``exp_root``); probe a
    couple of levels up.

    Returns:
        ``(hot_tput, lifecycle)``. ``lifecycle`` is one of ``hot_measure_round``,
        ``cold_single_round`` or ``unknown``, and ``hot_tput`` is 0.0 for anything
        but the first — so the hot-to-hot alignment metrics degrade to None rather
        than quietly dividing by a cold denominator. A standalone run with no
        orchestrator gets ``(0.0, "unknown")``.
    """
    exp_root = str(h.get("exp_root") or "").strip()
    if not exp_root:
        return 0.0, "unknown"
    p = Path(exp_root)
    for cand in (p / "state.json", p.parent / "state.json",
                 p.parent.parent / "state.json"):
        st = _read_json(cand)
        if not st:
            continue
        base = st.get("baseline") if isinstance(st.get("baseline"), dict) else {}
        tput = _positive_finite_float(
            st.get("baseline_tput") or base.get("baseline_tput")
        )
        if tput <= 0.0:
            continue
        warm_sec = _positive_finite_float(
            st.get("baseline_warm_runtime_sec")
            or base.get("baseline_warm_runtime_sec")
        )
        dropped = bool(
            st.get("baseline_measure_round_dropped")
            or base.get("baseline_measure_round_dropped")
        )
        if warm_sec > 0.0 and not dropped:
            return tput, "hot_measure_round"
        return 0.0, "cold_single_round" if dropped else "unknown"
    return 0.0, "unknown"


def read_orchestrator_hot_baseline(h: dict) -> float:
    """Hyperloom's baseline anchor, but only when it is a HOT measure round.

    Thin wrapper over :func:`read_orchestrator_baseline_lifecycle`; see there for
    why the hot number is ``baseline_tput`` and not a separate key.
    """
    return read_orchestrator_baseline_lifecycle(h)[0]


def _wf_best_accepted_delta_pct(wf: dict) -> float:
    """Largest positive ``e2e_delta_pct`` claimed by an accepted head/kernel.

    The workflow return carries the heads/kernels it ACCEPTED (each with the
    measured same-session A/B ``e2e_delta_pct``). This is the ground-truth signal
    that a real, parity-checked win exists — independent of whatever final
    throughput/speedup the return also reports. Returns 0.0 when nothing claims a
    positive gain.
    """
    best = 0.0
    for item in (wf.get("accepted_heads") or []) + (wf.get("accepted_kernels") or []):
        if not isinstance(item, dict):
            continue
        try:
            d = float(item.get("e2e_delta_pct") or 0.0)
        except (TypeError, ValueError):
            d = 0.0
        if d > best:
            best = d
    return best


def _best_ledger_win(wf: dict) -> dict | None:
    """Largest positive-``e2e_delta_pct`` direction the run actually MEASURED.

    Read from the workflow return's in-run experience ledger
    (``state.history.ledger`` — the Architect's per-direction record). This is the
    honest source for the winning op's identity when the accepted lists were
    dropped. Returns ``None`` when the ledger is absent or holds no positive
    direction (never fabricates a winner).
    """
    ledger = (((wf.get("state") or {}).get("history") or {}).get("ledger") or [])
    best: dict | None = None
    best_delta = 0.0
    for entry in ledger:
        if not isinstance(entry, dict):
            continue
        if not (entry.get("direction") or entry.get("short_name")):
            continue
        try:
            delta = float(entry.get("e2e_delta_pct") or 0.0)
        except (TypeError, ValueError):
            delta = 0.0
        if delta > best_delta:
            best, best_delta = entry, delta
    return best


def _state_op_names(wf: dict, queue: str) -> set[str]:
    """``short_name`` set of the return's ``state.<queue>`` (headQueue|kernelQueue)."""
    names: set[str] = set()
    for op in ((wf.get("state") or {}).get(queue) or []):
        if isinstance(op, dict) and op.get("short_name"):
            names.add(str(op["short_name"]))
    return names


def _divergence_pct(measured: Any, reference: Any) -> float | None:
    """Return percentage divergence from a positive reference."""
    try:
        measured_value = float(measured)
        reference_value = float(reference)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(measured_value)
        or not math.isfinite(reference_value)
        or measured_value <= 0.0
        or reference_value <= 0.0
    ):
        return None
    return round(
        100.0 * (measured_value - reference_value) / reference_value,
        2,
    )


def _positive_finite_float(value: Any) -> float:
    """Normalize external numeric input for strict JSON serialization."""
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(normalized) or normalized <= 0.0:
        return 0.0
    return normalized


def _build_baseline_alignment(
    same_config_divergence_pct: float | None,
    recipe_aligned: bool = True,
    same_config_reference_status: str = "",
    *,
    warning_threshold_pct: float,
) -> dict[str, Any]:
    """Classify cross-harness alignment using only the same-config metric.

    ``recipe_aligned`` says whether this run launched its servers through the
    orchestrator's own launch script. When it did not, the two harnesses served
    DIFFERENT stacks (the orchestrator's script owns the platform kernel preset,
    ``--trust-remote-code`` and the gpu-mem-util default), so a divergence is
    evidence about the launch recipe, not about the box or the bench client.
    Saying that in the status keeps the number from being read as "GEAK
    measured slow".

    ``same_config_reference_status`` is the orchestrator's own verdict on the
    same-config reference it forwarded (handoff ``same_config_reference_status``).
    A bare ``"unavailable"`` reads as "GEAK failed to compute it"; when the
    orchestrator shipped ``orchestrator_best_tput_same_config: 0.0`` with an
    ``unverified`` status, the number was never measured upstream and no amount
    of GEAK-side work can produce it. Naming that in the status is the
    difference between a fixable gap and a silent dead end.
    """
    reference_status = str(same_config_reference_status or "").strip().lower()
    if same_config_divergence_pct is None:
        status = (
            "unavailable_reference_unverified"
            if reference_status and reference_status != "verified"
            else "unavailable"
        )
    elif abs(same_config_divergence_pct) > warning_threshold_pct:
        status = "warning" if recipe_aligned else "warning_recipe_unaligned"
    else:
        status = "aligned"
    return {
        "status": status,
        "primary_metric": "current_best_same_config_divergence_pct",
        "divergence_pct": same_config_divergence_pct,
        "warning_threshold_pct": warning_threshold_pct,
        "raw_session_divergence_is_measurement_signal": False,
        "recipe_aligned_with_orchestrator": recipe_aligned,
        # Verbatim from the handoff ("" on orchestrator-less runs), so a reader
        # can tell WHOSE side the missing reference is on.
        "same_config_reference_status": reference_status or None,
    }


_HANDOFF_IDENTITY_KEYS = (
    # Newer handoffs can name the upstream observation explicitly. Keep the
    # historical names too: they are opaque hashes in older handoffs, but can
    # still be matched if Magpie's launch evidence records the same hash.
    "same_config_observed_identity",
    "observed_server_identity",
    "same_config_reference_observed_identity",
    "same_config_reference_identity",
    "same_config_expected_identity",
)
_SERVER_ARGS_IDENTITY_KEYS = (
    "model_path",
    "tokenizer_path",
    "served_model_name",
    "tp_size",
    "dp_size",
    "mem_fraction_static",
    "context_length",
    "chunked_prefill_size",
    "quantization",
    "dtype",
    "kv_cache_dtype",
    "attention_backend",
    "prefill_attention_backend",
    "decode_attention_backend",
    "disable_radix_cache",
    "trust_remote_code",
)


def _balanced_server_args(text: str) -> str:
    """Return one complete ``ServerArgs(...)`` expression from log text."""
    start = text.find("ServerArgs(")
    if start < 0:
        return ""
    depth, quote, escaped = 0, "", False
    for index, char in enumerate(text[start:], start):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _parse_sglang_server_args(text: str) -> dict[str, Any]:
    """Extract stable, observable launch fields from SGLang's startup record."""
    expression = _balanced_server_args(text)
    if not expression:
        return {}
    try:
        node = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(node, ast.Call):
        return {}
    observed: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg not in _SERVER_ARGS_IDENTITY_KEYS:
            continue
        try:
            value = ast.literal_eval(keyword.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            observed[keyword.arg] = value
    return observed


def _read_server_identity_evidence(log_path: Path) -> tuple[dict[str, Any], str]:
    """Read direct launch identity evidence without loading an unbounded log."""
    try:
        with log_path.open(encoding="utf-8", errors="ignore") as handle:
            lines = iter(handle)
            for line in lines:
                if "ServerArgs(" not in line:
                    continue
                # SGLang normally emits one line. Permit a wrapped startup
                # record too, but cap it so a malformed log cannot grow memory.
                record = line
                while not _balanced_server_args(record) and len(record) < 65536:
                    try:
                        record += next(lines)
                    except StopIteration:
                        break
                server_args = _parse_sglang_server_args(record)
                if server_args:
                    return {"backend": "sglang", "server_args": server_args}, ""
    except OSError:
        return {}, ""

    # A Magpie recipe can log its immutable identity directly. This is useful
    # for backends other than SGLang and for opaque upstream digest identities.
    try:
        with log_path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = re.search(
                    r"\b(?:observed_)?(?:server_|launch_)?identity\s*[:=]\s*"
                    r"['\"]?([A-Za-z0-9][A-Za-z0-9:._-]*)",
                    line,
                )
                if match:
                    return {"launch_identity": match.group(1)}, ""
    except OSError:
        pass
    return {}, ""


def _handoff_observed_identity(handoff: dict) -> tuple[Any, str]:
    """Return the upstream identity exactly as supplied, never synthesized."""
    for key in _HANDOFF_IDENTITY_KEYS:
        value = handoff.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), key
        if isinstance(value, dict) and value:
            return value, key
    baseline_spec = handoff.get("baseline_env_spec")
    if isinstance(baseline_spec, dict):
        value = baseline_spec.get("launch_identity")
        if isinstance(value, str) and value.strip():
            return value.strip(), "baseline_env_spec.launch_identity"
    return None, ""


def _identity_value_equal(expected: Any, observed: Any) -> bool | None:
    """Compare an upstream identity to observed launch facts conservatively."""
    if isinstance(expected, str):
        if not isinstance(observed, dict) or not observed.get("launch_identity"):
            return None
        return expected == observed["launch_identity"]
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        return None
    expected_token = expected.get("launch_identity") or expected.get("identity")
    if isinstance(expected_token, str):
        observed_token = observed.get("launch_identity")
        if not observed_token:
            return None
        return expected_token == observed_token
    expected_args = expected.get("server_args", expected)
    observed_args = observed.get("server_args", observed)
    if not isinstance(expected_args, dict) or not isinstance(observed_args, dict):
        return None
    # An upstream observation may intentionally include only a subset. Missing
    # observed fields are a verification failure, never silently ignored.
    comparable_keys = [
        key
        for key in expected_args
        if key
        not in {"backend", "launch_identity", "identity", "source", "evidence_paths"}
    ]
    if not comparable_keys:
        return None
    if any(key not in observed_args for key in comparable_keys):
        return None
    for key, expected_value in expected_args.items():
        if key not in comparable_keys:
            continue
        if observed_args[key] != expected_value:
            return False
    expected_backend = expected.get("backend")
    observed_backend = observed.get("backend")
    return not expected_backend or expected_backend == observed_backend


def _server_identity_alignment(handoff: dict, eval_dir: Path) -> dict[str, Any]:
    """Compare upstream identity with actual Setup-server launch evidence."""
    expected, expected_source = _handoff_observed_identity(handoff)
    candidates = [
        eval_dir / "baseline" / "server.log",
        eval_dir / "baseline" / "magpie_launch.log",
    ]
    evidence_paths = [str(path) for path in candidates if path.is_file()]
    observed: dict[str, Any] = {}
    for path in candidates:
        observed, _ = _read_server_identity_evidence(path)
        if observed:
            break

    if expected is None:
        status = "unavailable"
    elif not observed:
        status = "unverified"
    else:
        equal = _identity_value_equal(expected, observed)
        status = "unverified" if equal is None else "matched" if equal else "mismatched"
    return {
        "expected": expected,
        "expected_source": expected_source or None,
        "observed": observed or None,
        "status": status,
        "evidence_paths": evidence_paths,
    }


def _build_handoff_alignment(
    setup_same_config_divergence_pct: float | None,
    *,
    recipe_aligned: bool,
    server_identity: dict[str, Any],
    same_config_reference_status: str = "",
    warning_threshold_pct: float,
) -> dict[str, Any]:
    """Classify the Setup-to-upstream comparison independently of Validate."""
    metric = _build_baseline_alignment(
        setup_same_config_divergence_pct,
        recipe_aligned,
        same_config_reference_status,
        warning_threshold_pct=warning_threshold_pct,
    )
    identity_status = server_identity.get("status")
    if metric["status"].startswith("unavailable"):
        status = metric["status"]
    elif identity_status == "mismatched":
        status = "identity_mismatch"
    elif identity_status != "matched":
        # Older handoffs lack an upstream observed identity. A close throughput
        # number alone is not proof that the same server was launched.
        status = "unverified"
    else:
        status = metric["status"]
    return {
        **metric,
        "status": status,
        "metric_status": metric["status"],
        "primary_metric": "setup_seed_same_config_divergence_pct",
        "server_identity": server_identity,
    }


# Kernel-selection lines a serving stack prints at startup. Substring matching
# is backend-agnostic on purpose: the goal is not to enumerate backends but to
# record WHICH kernels the engine actually chose, so a launch that silently fell
# back to a slower stack is visible in result.json instead of only in a
# thousand-line server log.
_STACK_SIGNAL_PATTERNS = (
    "Final IR op priority",
    "for Fp8LinearMethod",
    "Fp8 MoE backend",
    "ttention backend",  # "Attention backend" / "attention backend"
    "server_args=ServerArgs(",
)
_STACK_SIGNAL_MAX_PICKS = 8


def _serving_stack_signals(log_path: Path) -> dict[str, Any]:
    """Summarize the kernel stack a server actually came up with.

    Returns ``{}`` when the log is unreadable (leg never ran / was cleaned), so
    the caller can carry the field harmlessly on a standalone run.
    """
    picks: list[str] = []
    aiter_mentions = 0
    try:
        # Stream line-by-line: server.log can reach GBs, and the old
        # read_text()+splitlines()+text.lower() held three full O(n) copies at
        # once. Iterating the handle keeps at most one line resident (O(1)).
        with log_path.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                # Whole-log count (semantics unchanged vs the old text.lower().count).
                aiter_mentions += line.lower().count("aiter")
                # Check the cap BEFORE appending so _STACK_SIGNAL_MAX_PICKS is a
                # real bound (the old code appended then checked, so switching the
                # break to a continue would have let picks grow without limit).
                if len(picks) < _STACK_SIGNAL_MAX_PICKS and any(
                    pattern in line for pattern in _STACK_SIGNAL_PATTERNS
                ):
                    # Drop the pid / timestamp / module prefix so the pick is legible.
                    picks.append(line.rsplit("] ", 1)[-1].strip()[:220])
    except OSError:
        return {}
    return {
        # A raw count, not a verdict: the accelerated-kernel stack is chatty at
        # startup, so a near-zero count is the signature of a launch that never
        # enabled it. Observed on one 122B session: 5499 mentions on the
        # orchestrator's server vs 4 on GEAK's, for the identical config.
        "aiter_mentions": aiter_mentions,
        "kernel_picks": picks,
    }


def _cold_penalty_pct(cold: Any, hot: Any) -> float | None:
    """How far a leg's cold round fell below that same leg's hot median, in %.

    Negative is the expected direction (the cold round pays JIT / graph-capture
    / page-cache costs). Comparing the two legs' penalties is what reveals
    whether their cold rounds were measured in comparable thermal states.
    """
    cold_value, hot_value = _positive_finite_float(cold), _positive_finite_float(hot)
    if cold_value <= 0.0 or hot_value <= 0.0:
        return None
    return round((cold_value - hot_value) / hot_value * 100.0, 3)


def _overlay_has_loadable_code(path: Path) -> bool:
    """True when ``path`` is an overlay that actually installs something.

    Finalize creates ``final/overlay`` unconditionally and drops a marker file
    into it when nothing was accepted, so directory existence proves nothing.
    The test is the mechanism's own contract: the overlay is activated by its
    ROOT on ``PYTHONPATH``, and the only thing the interpreter runs from there
    is ``<overlay>/sitecustomize.py`` (``e2e_workflow/scripts/overlay_setup.py``
    — only the FIRST sitecustomize on the path is imported, so two overlay dirs
    do not compound). So:

      * no ``sitecustomize.py`` at the ROOT -> inert, whatever else is inside.
        Neither a loose ``.py`` module nor a loadable ``cand_*`` SUBTREE counts:
        the consumer is handed this path, not the child.
      * a manifest that parses and names nothing -> the config-only overlay,
        which imports cleanly and installs zero kernels, so advertising it books
        a pure config win as a kernel win.
      * a manifest that does not parse -> we cannot claim it installs anything.

    Deliberately the same predicate the consumer applies (Hyperloom's
    ``_geak_overlay_is_loadable``): declaring an overlay it will refuse to load
    costs a revalidation, and an empty string tells it the truth.
    """
    if not path.is_dir() or not (path / "sitecustomize.py").is_file():
        return False
    manifest = path / "_overlay_manifest.json"
    if not manifest.is_file():
        # No manifest is the pre-manifest overlay shape: absence of evidence is
        # not evidence of an empty overlay, and the sitecustomize still runs.
        return True
    try:
        spec = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(spec, dict):
        return False
    return bool(spec.get("modules") or spec.get("rebinds") or spec.get("captures") or spec.get("installers"))


def _patch_has_hunks(path: Path) -> bool:
    """True when ``path`` is a unified diff that would actually change something.

    ``final_patch.diff`` is written even when there is nothing to patch — the
    header/provenance preamble alone runs to over a kilobyte — so a size check
    passes on a patch that applies zero edits. A unified diff changes a file
    only through an ``@@`` hunk, so that marker is the real test.
    """
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(line.startswith("@@") for line in text.splitlines())


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Shell control characters. A value carrying one of these is not a value: it is
# a fragment of the launch script that leaked into the assignment string.
_ENV_VALUE_SHELL_CHARS = ";&|<>()`"


def _parse_env_assignments(env: str) -> tuple[dict, list]:
    """Split an ``A=1 B=2`` string into ``({k: v}, [unparsable fragments])``.

    ``accepted_config.env`` is a shell string assembled from launch-script
    lines, so syntax travels with it and a consumer's naive ``s.split()`` +
    ``split("=")`` turns ``)``, ``true;`` and ``sglang;`` into environment
    variables — which is exactly how ``EXTRA_ENV=").", RUN_EVAL="true;",
    BACKEND="sglang;"`` reached a downstream rebench.

    So GEAK does the split once and publishes the result: a key must be a real
    identifier, a value must be free of shell control characters, and a trailing
    ``;`` (a statement separator the line-joining left behind) is stripped first.
    Anything that still fails goes to the reject list rather than being dropped,
    so a consumer can see the string was lossy instead of trusting a map that
    quietly lost a variable.
    """
    ok: dict[str, str] = {}
    rejected: list[str] = []
    text = str(env or "").strip()
    if not text:
        return ok, rejected
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()

    def _pair(piece: str) -> tuple[str, str] | None:
        key, sep, value = piece.partition("=")
        if (
            sep
            and _ENV_KEY_RE.match(key)
            and not any(c in value for c in _ENV_VALUE_SHELL_CHARS)
        ):
            return key, value
        return None

    for token in tokens:
        if not token.strip():
            continue
        # One token can hold several assignments joined by ``;`` — a
        # launch-script line that never got re-split. Take the whole token only
        # if EVERY piece of it is a well-formed assignment, so a half-parsed
        # fragment is quarantined whole instead of contributing half a truth.
        pieces = [p for p in token.split(";") if p.strip()]
        pairs = [_pair(p) for p in pieces]
        if pairs and all(p is not None for p in pairs):
            ok.update(dict(pairs))  # type: ignore[arg-type]
        else:
            rejected.append(token)
    return ok, rejected


def _accepted_config_with_env_map(config: dict) -> dict:
    """``accepted_config`` plus a structured, validated ``env_map``.

    ``env`` is kept verbatim (it is what the run actually exported, and older
    consumers read it), but ``env_map`` is the form a consumer should use:
    already split, already validated. ``env_unparsed`` is present only when
    something was dropped, which is the signal that the shell string is
    malformed at the source.
    """
    out = dict(config)
    env_map, rejected = _parse_env_assignments(out.get("env"))
    out["env_map"] = env_map
    if rejected:
        out["env_unparsed"] = rejected
    return out


def _material_overlay_path(eval_dir: Path, wf: dict) -> str:
    """Path to a real overlay, or ``""`` when this run produced none.

    Advertising a path that holds nothing (or does not exist at all) makes a
    caller believe there is something to reuse; an empty string tells it the
    truth. The path recorded on the return wins when it still resolves,
    otherwise we look for the product under the eval_dir we were handed.
    """
    recorded = str(wf.get("final_overlay") or "").strip()
    candidates: list[Path] = [Path(recorded)] if recorded else []
    candidates += [eval_dir / "final" / "overlay", eval_dir / "final"]
    for candidate in candidates:
        if _overlay_has_loadable_code(candidate):
            return str(candidate)
    return ""


def _material_patch_path(eval_dir: Path, wf: dict) -> str:
    """Path to a patch that carries at least one hunk, or ``""``."""
    recorded = str(wf.get("final_patch") or "").strip()
    candidates: list[Path] = [Path(recorded)] if recorded else []
    candidates.append(eval_dir / "final" / "final_patch.diff")
    for candidate in candidates:
        if _patch_has_hunks(candidate):
            return str(candidate)
    return ""


def _same_session_baseline(
    base_leg_summary: dict, validation: dict
) -> tuple[float, str]:
    """The baseline leg measured in the SAME Validate session as the final leg.

    ``baseline/bench_summary.json`` is minted during Setup, often hours before
    the final leg runs, and the box moves under us in between (we have observed
    -4%..+15% drift within a single session). Dividing a Validate-time final by
    a Setup-time baseline therefore reports drift as optimization. The Validate
    phase re-measures the unpatched engine right next to the patched one, so
    that pair — and only that pair — is a valid A/B.

    Preference order, most to least direct:
      1. ``validation/base/bench_summary.json`` — the re-measured base leg.
      2. Director's ``base_block.throughput_tok_s_median`` — same leg, as the
         Director recorded it (used when the summary file was not kept).
      3. Director's ``drift_corrected_baseline_tok_s`` — the Director's own
         drift correction, whose key name is stable across schema versions
         (unlike ``baseline_throughput_tok_s``, which means the Setup baseline
         in some versions and the corrected one in others).

    Returns ``(0.0, "")`` when no same-session leg exists, which leaves the
    caller on its previous baseline.
    """
    value = _positive_finite_float(base_leg_summary.get("throughput_tok_s_median"))
    if value > 0.0:
        return value, "validation_base_bench_summary"
    base_block = validation.get("base_block") or {}
    value = _positive_finite_float(base_block.get("throughput_tok_s_median"))
    if value > 0.0:
        return value, "director_base_block"
    value = _positive_finite_float(validation.get("drift_corrected_baseline_tok_s"))
    if value > 0.0:
        return value, "director_drift_corrected"
    return 0.0, ""


def handoff_acceptance(h: dict) -> dict | None:
    """The caller's KEEP rule for the axis it handed GEAK; None leaves GEAK's own verdict.

    Hyperloom publishes one in an AgentX ``workload_spec`` graded on interactivity: the
    ``objective`` must gain at least ``min_gain_pct`` over the reference, while each guard in
    ``guard_max_drop_pct`` -- named by basis, like the objective -- may fall no further below
    the reference than its band. A block that does not parse is ignored with a warning rather
    than half-applied: GEAK's own verdict stands, and the caller re-measures what it keeps.
    """
    spec = h.get("workload_spec")
    block = spec.get("acceptance") if isinstance(spec, dict) else None
    if block is None:
        return None
    try:
        objective = str(block["objective"]).strip()
        min_gain_pct = float(block["min_gain_pct"])
        guards = {
            str(basis).strip(): float(band)
            for basis, band in (block.get("guard_max_drop_pct") or {}).items()
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        problem = repr(exc)
    else:
        if objective and all(math.isfinite(v) for v in (min_gain_pct, *guards.values())):
            return {
                "objective": objective,
                "min_gain_pct": min_gain_pct,
                "guard_max_drop_pct": guards,
            }
        problem = "no objective, or a band that is not a finite number"
    sys.stderr.write(f"!!! workload_spec.acceptance ignored ({problem}): {block!r}\n")
    return None


def acceptance_verdict(
    rule: dict,
    *,
    measured_basis: str,
    baseline: float,
    final: float,
    reference: dict,
    candidate: dict,
) -> dict:
    """Apply the caller's KEEP rule to the pair this run publishes.

    Hyperloom's AgentX verdict, step for step: ``(final - baseline) / baseline * 100`` on the
    objective must reach ``min_gain_pct``, and each guard must hold ``candidate >= reference *
    (1 - max_drop_pct / 100)``. Guards are read from the ``guard_<basis>_median`` fields of the
    two legs' bench summaries, and one missing from either leg fails: a guard nobody measured
    is not one the candidate held. Hyperloom also refuses rounds of unequal length or with more
    failed requests and applies its latency budget; the summaries carry neither, so Hyperloom's
    re-measurement keeps the last word.
    """
    objective = rule["objective"]
    gain_pct = (final - baseline) / baseline * 100.0 if baseline > 0.0 and final > 0.0 else None
    reasons = []
    if measured_basis != objective:
        reasons.append(f"measured {measured_basis or 'an unnamed axis'}, not {objective}")
    elif gain_pct is None:
        reasons.append(f"no {objective} pair to compare")
    elif gain_pct < rule["min_gain_pct"]:
        reasons.append(f"{objective} gained {gain_pct:.3f}%, below {rule['min_gain_pct']}%")
    guards = {}
    for basis, max_drop_pct in rule["guard_max_drop_pct"].items():
        ref = _positive_finite_float(reference.get(f"guard_{basis}_median"))
        cand = _positive_finite_float(candidate.get(f"guard_{basis}_median"))
        measured = ref > 0.0 and cand > 0.0
        holds = measured and cand >= ref * (1.0 - max_drop_pct / 100.0)
        guards[basis] = {
            "reference": ref or None,
            "candidate": cand or None,
            "max_drop_pct": max_drop_pct,
            "holds": holds,
        }
        if not measured:
            reasons.append(f"guard {basis} was not measured on both legs")
        elif not holds:
            reasons.append(f"guard {basis} fell {(1.0 - cand / ref) * 100.0:.3f}%, "
                           f"past {max_drop_pct}%")
    return {
        "objective": objective,
        "measured_basis": measured_basis,
        "gain_pct": round(gain_pct, 3) if gain_pct is not None else None,
        "min_gain_pct": rule["min_gain_pct"],
        "guards": guards,
        "keep": not reasons,
        "reasons": reasons,
    }


def normalize_result(
    h: dict,
    wf: dict,
    *,
    recover_intermediate: Callable[[Path], dict | None],
    classify_workload: Callable[[dict], dict],
    warning_threshold_pct: float,
) -> dict:
    """Normalize saved artifacts using the entrypoint's recovery and workload policy."""
    eval_dir = Path(wf["eval_dir"])
    validation = _read_json(eval_dir / "director_e2e_validation.json")
    baseline_summary = _read_json(eval_dir / "baseline" / "bench_summary.json")
    final_summary = _read_json(eval_dir / "validation" / "final" / "bench_summary.json")
    # The unpatched leg re-measured during Validate, next to the final leg.
    base_leg_summary = _read_json(eval_dir / "validation" / "base" / "bench_summary.json")

    # Recover an accepted intermediate A/B only when the live return has no
    # final measurement. A measured no-gain final keeps Validate's verdict.
    wf_speedup_raw = float(wf.get("throughput_speedup") or validation.get("throughput_speedup") or 1.0)
    wf_final_raw = float(
        wf.get("final_throughput_tok_s")
        or validation.get("director_verified_throughput_tok_s")
        or 0.0
    )
    live_accepted_win = (
        not wf.get("recovered_from_disk")
        and _wf_best_accepted_delta_pct(wf) > 0.0
    )
    intermediate_win_not_confirmed = (
        live_accepted_win and wf_final_raw > 0.0 and wf_speedup_raw <= 1.0
    )
    if live_accepted_win and wf_final_raw <= 0.0:
        recovered = recover_intermediate(eval_dir)
        if recovered is not None and float(recovered.get("throughput_speedup") or 0.0) > 1.0:
            merged = dict(wf)
            for k in ("throughput_speedup", "baseline_throughput_tok_s",
                      "final_throughput_tok_s", "output_parity", "ttft_ms", "tpot_ms"):
                if recovered.get(k) is not None:
                    merged[k] = recovered[k]
            # The recovered win's own overlay (C7); "" means config-only, in
            # which case whatever the return already carried still stands.
            if recovered.get("final_overlay"):
                merged["final_overlay"] = recovered["final_overlay"]
            merged["recovered_intermediate"] = True   # provenance -> disk_intermediate_win
            merged["validate_final_missing"] = True
            wf = merged
            # The Director json (the crashed bench) is kept, not erased: it is
            # the evidence for WHY we fell back here, and validation_evidence
            # reports it. Every number above is already sourced from wf first.

    speedup = float(wf.get("throughput_speedup") or validation.get("throughput_speedup") or 1.0)

    # Same rule as report_path: never advertise a path that does not exist.
    _launch_default = eval_dir / "final" / "final_launch.sh"
    final_launch = (
        wf.get("final_launch_script")
        or validation.get("final_launch_script")
        or (str(_launch_default) if _launch_default.is_file() else "")
    )
    workload = h.get("workload") or {"isl": 1024, "osl": 1024, "conc": 64}

    # Provenance of the numbers below, so Hyperloom can gauge confidence:
    #   workflow_return        — the canonical schema-validated artifact / scraped
    #                            return (full Validate-arbitrated result).
    #   disk_director_validation — rebuilt from director_e2e_validation.json.
    #   disk_intermediate_win  — best accepted integrate A/B (no final Validate).
    #   disk_no_gain_synthesis — baseline measured, nothing accepted (do-no-harm).
    checkpoint_level = str(wf.get("recovered_e2e_checkpoint_level") or "")
    tuning_recovery_source = str(wf.get("recovered_tuning_source") or "")
    if checkpoint_level:
        result_source = f"disk_e2e_checkpoint_{checkpoint_level}"
    elif tuning_recovery_source:
        result_source = f"disk_tuning_skillset_{tuning_recovery_source}_provisional"
    elif wf.get("recovered_tuning_legacy"):
        result_source = "disk_tuning_skillset_legacy_provisional"
    elif wf.get("recovered_no_gain"):
        result_source = "disk_no_gain_synthesis"
    elif wf.get("recovered_intermediate"):
        # disk_stack_provisional — salvaged from candidates the integrator gated
        # "stack" (carry forward to compound), i.e. none of them cleared its bar
        # for a standalone win and no Director ever arbitrated the combination.
        result_source = (
            "disk_stack_provisional" if wf.get("recovered_stack_provisional")
            else "disk_intermediate_win"
        )
    elif wf.get("recovered_from_disk"):
        result_source = "disk_director_validation"
    else:
        result_source = "workflow_return"

    # A validated live win with empty accepted lists may recover its attribution
    # from the measured ledger. Preserve populated lists and label the backfill.
    if (
        result_source == "workflow_return"
        and speedup > 1.0
        and (validation.get("validation_status") == "validated_win"
             or wf.get("validation_status") == "validated_win")
        and not (wf.get("accepted_kernels") or wf.get("accepted_heads"))
    ):
        _win = _best_ledger_win(wf)
        if _win is not None:
            _name = str(_win.get("direction") or _win.get("short_name") or "")
            _entry = {
                "short_name": _name,
                "e2e_delta_pct": _win.get("e2e_delta_pct"),
                "isolated": _win.get("isolated_speedup"),
                "backend": "geak",
                "accepted_via": "director_override",
                "note": _win.get("lesson"),
            }
            _kernels = _state_op_names(wf, "kernelQueue")
            _heads = _state_op_names(wf, "headQueue")
            wf = dict(wf)   # don't mutate the caller's return object
            if _name and _name in _kernels and _name not in _heads:
                wf["accepted_kernels"] = [_entry]
            else:
                wf["accepted_heads"] = [_entry]
            wf["attribution_backfilled"] = True

    # The validation base/final pair is the optimization denominator. It is
    # deliberately NOT the cross-handoff denominator: Validate may run long
    # after Setup and the box's movement between them is measurement drift, not
    # a difference between Hyperloom and GEAK.
    same_session_base, baseline_basis_source = 0.0, ""
    if not (wf.get("recovered_intermediate") or wf.get("recovered_no_gain")):
        same_session_base, baseline_basis_source = _same_session_baseline(
            base_leg_summary, validation
        )
    setup_baseline = _positive_finite_float(
        baseline_summary.get("throughput_tok_s_median")
        or baseline_summary.get("output_throughput_tok_s_median")
        # The canonical workflow return's baseline is the Setup seed. This
        # fallback keeps disk-recovered/older runs comparable when their Setup
        # summary was not retained; it never takes a Validate-time value.
        or wf.get("baseline_throughput_tok_s")
    )
    geak_baseline = _positive_finite_float(
        same_session_base
        or wf.get("baseline_throughput_tok_s")
        or validation.get("baseline_throughput_tok_s")
        or 0.0
    )
    if not baseline_basis_source and geak_baseline > 0.0:
        baseline_basis_source = (
            "recovered_intermediate_ab"
            if wf.get("recovered_intermediate")
            else "no_gain_synthesis"
            if wf.get("recovered_no_gain")
            # A baseline rebuilt from a disk Director artifact (no more direct
            # same-session source) is NOT a live Setup measurement; label its
            # provenance for what it is so the A/B basis is auditable, mirroring
            # result_source == "disk_director_validation" above.
            else "disk_director_validation_baseline"
            if wf.get("recovered_from_disk")
            else "setup_baseline"
        )
    geak_final = _positive_finite_float(
        wf.get("final_throughput_tok_s")
        or validation.get("director_verified_throughput_tok_s")
        or 0.0
    )
    orch_baseline = _positive_finite_float(h.get("raw_baseline_tput"))
    # Orchestrator throughput measured on the SAME config GEAK seeds with
    # (the upstream orchestrator's current-best config). When present it isolates
    # the pure cross-harness measurement residue from the config gain baked into
    # the raw-baseline comparison. It remains unavailable for older handoffs.
    orch_same_cfg = _positive_finite_float(
        h.get("orchestrator_best_tput_same_config")
    )
    # The orchestrator's verdict on the reference above. "verified" means it
    # really re-measured GEAK's seed config; anything else means the 0.0 is an
    # absence, not a measurement.
    orch_same_cfg_status = str(h.get("same_config_reference_status") or "").strip().lower()
    # Cross-handoff alignment is fixed to the Setup seed measurement. Never
    # replace it with the late Validate base leg: doing so turns intra-session
    # drift into a verdict about the handoff.
    raw_session_divergence_pct = _divergence_pct(setup_baseline, orch_baseline)
    same_config_divergence_pct = _divergence_pct(setup_baseline, orch_same_cfg)
    # Only a Validate-time base leg can measure drift. ``geak_baseline`` may
    # legitimately fall back to Setup (or an intermediate A/B) for the
    # optimization result, and calling that a drift measurement would invent a
    # zero-percent observation.
    validation_base_for_drift = _positive_finite_float(same_session_base)

    # ── workload comparability gate ───────────────────────────────────────────
    # Everything that divides a GEAK measurement by an ORCHESTRATOR measurement
    # is only defined when both measured the same workload. When they did not,
    # publish None rather than a number: a suppressed field forces a reviewer to
    # look, whereas a plausible-looking ratio invites the exact false claim this
    # guard exists to prevent. The raw inputs on both sides stay published for
    # audit, and every WITHIN-GEAK ratio (hot_geak_speedup, cold_geak_speedup,
    # baseline_drift_pct, the cold penalties) is untouched -- those compare two
    # GEAK legs measured on the same workload and remain valid regardless.
    workload_comparability = classify_workload(h)
    cross_harness_ok = bool(workload_comparability["comparable"])
    if not cross_harness_ok:
        raw_session_divergence_pct = None
        same_config_divergence_pct = None

    # ── serving-stack provenance ─────────────────────────────────────────────
    # WHO launched the server, and WHAT kernels it selected. A cross-harness
    # comparison only means something when both sides served the same stack, and
    # the biggest way that breaks is silent: the orchestrator's launch script
    # exports the platform kernel preset and GEAK's own adapter does not, so the
    # identical config serves measurably slower. Recording the launcher next to
    # the kernels it produced makes that visible in the interface file.
    launcher = os.environ.get("BENCH_LAUNCHER", "native")
    recipe_aligned = launcher == "magpie"
    serving_stack = {
        "launcher": launcher,
        "launch_script": os.environ.get("MAGPIE_LAUNCH_SCRIPT", ""),
        "launch_script_source": os.environ.get("MAGPIE_LAUNCH_SCRIPT_SOURCE", ""),
        "recipe_aligned_with_orchestrator": recipe_aligned,
        "baseline": _serving_stack_signals(eval_dir / "baseline" / "server.log"),
        "validation_base": _serving_stack_signals(
            eval_dir / "validation" / "base" / "server.log"
        ),
    }
    server_identity = _server_identity_alignment(h, eval_dir)
    handoff_alignment = _build_handoff_alignment(
        same_config_divergence_pct,
        recipe_aligned=recipe_aligned,
        server_identity=server_identity,
        same_config_reference_status=orch_same_cfg_status,
        warning_threshold_pct=warning_threshold_pct,
    )
    baseline_alignment = _build_baseline_alignment(
        same_config_divergence_pct, recipe_aligned, orch_same_cfg_status,
        warning_threshold_pct=warning_threshold_pct,
    )
    setup_evidence_path = eval_dir / "baseline" / "bench_summary.json"
    validation_evidence_path = eval_dir / "validation" / "base" / "bench_summary.json"
    measurement_drift_pct = _divergence_pct(
        validation_base_for_drift, setup_baseline
    )
    measurement_drift = {
        "status": (
            "measured"
            if (setup_baseline > 0.0 and validation_base_for_drift > 0.0)
            else "unavailable"
        ),
        "setup_baseline_tok_s": setup_baseline or None,
        "validation_base_tok_s": validation_base_for_drift or None,
        "validation_base_source": (
            baseline_basis_source if validation_base_for_drift > 0.0 else None
        ),
        "drift_pct": measurement_drift_pct,
        "evidence_paths": [
            str(path)
            for path in (setup_evidence_path, validation_evidence_path)
            if path.is_file()
        ],
    }
    baseline_basis = {
        # GEAK's own measured baseline (Hyperloom-accepted config = fair engagement baseline; gating uses this).
        "geak_measured_baseline_tok_s": geak_baseline or None,
        # Which leg the denominator above came from, so a reviewer can tell a
        # same-session A/B from a Setup-time comparison at a glance.
        "baseline_basis_source": baseline_basis_source or None,
        # The Setup-time baseline, kept for audit even when it is no longer the
        # denominator, plus how far the box moved between Setup and Validate.
        # A large drift means the Setup number was never a valid denominator.
        "setup_baseline_tok_s": setup_baseline or None,
        # Legacy field: a Setup-only result historically rendered as 0.0.
        # New consumers should use measurement_drift, whose unavailable status
        # distinguishes that absence from an actual zero-drift remeasurement.
        "baseline_drift_pct": (
            measurement_drift_pct
            if measurement_drift_pct is not None
            else 0.0
            if (geak_baseline > 0.0 and geak_baseline == setup_baseline)
            else None
        ),
        # Hyperloom's own measured baseline forwarded in the handoff (the orchestrator reference).
        "orchestrator_baseline_tok_s": orch_baseline or None,
        # Audit-only comparison with the RAW session baseline. This includes
        # accepted upstream config gain and is not a measurement-drift signal.
        "raw_session_baseline_divergence_pct": raw_session_divergence_pct,
        # PURE cross-harness measurement residue: GEAK baseline vs the
        # orchestrator's throughput on the SAME (accepted) config. Both sides
        # run the identical config, so this isolates client/protocol/warm-cold
        # differences from the config gain. This is the primary alignment metric.
        "current_best_same_config_divergence_pct": same_config_divergence_pct,
        # Backward-compatible alias consumed by existing orchestrators.
        "measurement_divergence_pct": same_config_divergence_pct,
        "orchestrator_best_tput_same_config": orch_same_cfg or None,
        # Gain measured against the ORCHESTRATOR baseline (what Hyperloom sees end-to-end).
        "gain_vs_orchestrator_baseline": (
            round(geak_final / orch_baseline, 4)
            if (cross_harness_ok and geak_final > 0 and orch_baseline > 0) else None
        ),
        # Why the cross-harness fields above hold numbers or None. Always
        # published, so "comparable" is an asserted fact rather than an absence.
        "workload_comparability": workload_comparability,
        # Measurement-protocol provenance so the comparison is self-describing.
        "bench_client": os.environ.get("BENCH_CLIENT", "native"),
        "bench_protocol": h.get("bench_protocol") or {},
        "baseline_config": {
            "accepted_flags": h.get("accepted_flags", "") or "",
            "accepted_env": h.get("accepted_env", "") or "",
        },
    }

    # Cold rounds are diagnostic; the published final always uses the hot median.
    # Cross-harness hot ratios require evidence that the upstream anchor is hot.
    orch_state_hot_baseline, orch_baseline_lifecycle = read_orchestrator_baseline_lifecycle(h)
    # Prefer the handoff's anchor over the one re-read from state.json so
    # hot_speedup stays the SAME pairing Hyperloom promotes: a re-baseline that
    # lands after our handoff was minted moves state.json but not the handoff,
    # and silently reporting the newer number would make the two ratios in this
    # block disagree for no visible reason. state.json only supplies the verdict
    # on what the anchor is, plus the value when the handoff carries none.
    orch_hot_baseline = (
        (orch_baseline or orch_state_hot_baseline)
        if orch_baseline_lifecycle == "hot_measure_round"
        else 0.0
    )
    geak_cold_final = final_summary.get("cold_output_throughput_tok_s")
    # Prefer the cold baseline measured beside the final. Label the Setup
    # fallback because its earlier cold state may differ from Validate's.
    if base_leg_summary.get("cold_output_throughput_tok_s") is not None:
        cold_baseline_summary = base_leg_summary
        cold_pairing = "same_session"
    else:
        cold_baseline_summary = baseline_summary
        cold_pairing = "setup_vs_validate"
    geak_cold_baseline = cold_baseline_summary.get("cold_output_throughput_tok_s")
    if geak_cold_baseline is None:
        cold_pairing = None
    alignment_metrics = {
        "geak_hot_final_tok_s": geak_final or None,
        "geak_hot_baseline_tok_s": geak_baseline or None,
        "geak_cold_final_tok_s": geak_cold_final,
        "geak_cold_baseline_tok_s": geak_cold_baseline,
        # == handoff.raw_baseline_tput (the leaderboard anchor). Kept under the
        # historical "cold" key for the consumers that already read it, but the
        # anchor is hot whenever orchestrator_baseline_lifecycle says so.
        "orchestrator_cold_baseline_tok_s": orch_baseline or None,
        "orchestrator_baseline_tok_s": orch_baseline or None,
        # hot_measure_round | cold_single_round | unknown — read off Hyperloom's
        # state.json, so it says what the anchor above ACTUALLY is.
        "orchestrator_baseline_lifecycle": orch_baseline_lifecycle,
        # The same anchor, exposed only when it is provably a hot measure round.
        "orchestrator_hot_baseline_tok_s": orch_hot_baseline or None,
        # The cross-harness ratios here are gated on workload comparability; the
        # within-GEAK ones beside them are not, because both of their legs
        # were measured by the same client on the same workload.
        "hot_speedup": (
            _safe_ratio(geak_final, orch_hot_baseline) if cross_harness_ok else None
        ),
        # The raw upstream baseline predates its accepted config gains. Only
        # hot_geak_speedup isolates GEAK's contribution from those gains.
        "hot_speedup_denominator": "raw_session_baseline",
        "hot_speedup_includes_orchestrator_config_gain": True,
        # The same hot-to-hot ratio against the orchestrator's throughput on
        # GEAK's OWN seed config -- the only pairing in this block whose
        # numerator and denominator share a config. None when the orchestrator
        # never verified that reference. Still divides a GEAK measurement by an
        # orchestrator one, so it is gated on comparability like hot_speedup.
        "hot_speedup_same_config": (
            _safe_ratio(geak_final, orch_same_cfg) if cross_harness_ok else None
        ),
        "hot_geak_speedup": _safe_ratio(geak_final, geak_baseline),
        "cold_speedup": (
            _safe_ratio(geak_cold_final, orch_baseline) if cross_harness_ok else None
        ),
        "cold_geak_speedup": _safe_ratio(geak_cold_final, geak_cold_baseline),
        # Which two rounds cold_geak_speedup divides: "same_session" is a valid
        # A/B, "setup_vs_validate" is a comparison across thermal states and the
        # ratio should be read as such (or ignored).
        "cold_pairing": cold_pairing,
        # How much each leg's cold round lost against its OWN hot median. These
        # tell you whether the cold rounds are comparable at all: similar
        # penalties mean both paid a similar cache-fill cost, while a large
        # baseline penalty next to a small final one is the signature of a
        # "cold" round that ran on an already-warm box.
        "cold_penalty_pct_baseline": _cold_penalty_pct(
            geak_cold_baseline, cold_baseline_summary.get("throughput_tok_s_median")
        ),
        "cold_penalty_pct_final": _cold_penalty_pct(
            geak_cold_final, final_summary.get("throughput_tok_s_median")
        ),
        "final_basis": "hot",
    }

    # Keep the reported speedup consistent with the final/baseline pair.
    speedup_basis = "workflow_return"
    speedup_as_returned = speedup
    if geak_baseline > 0.0 and geak_final > 0.0:
        pair_ratio = geak_final / geak_baseline
        if abs(pair_ratio - speedup) > SPEEDUP_SELF_CONSISTENCY_TOL:
            speedup = round(pair_ratio, 6)
            speedup_basis = "final_over_baseline"
    status = "ok" if speedup > 1.0 else "no_gain"
    alignment_metrics["speedup_basis"] = speedup_basis
    # Only populated when we had to override, so its presence is the signal.
    alignment_metrics["speedup_as_returned"] = (
        speedup_as_returned if speedup_basis == "final_over_baseline" else None
    )

    # ── validation evidence ──────────────────────────────────────────────────
    # A speedup number on its own cannot be judged: 1.01x is a solid win on a
    # bench that repeats within 0.2% and pure noise on one that swings 3%. This
    # block carries what a reader needs to decide — the arbitration verdict, the
    # run-to-run spread of each leg, the Director's noise band, and whether the
    # delta clears them. Reported, never enforced: nothing here changes status.
    # The one exception is ``acceptance``, added below only when the handoff states
    # the caller's KEEP rule.
    delta_pct = round((speedup - 1.0) * 100.0, 3)
    # The metric-neutral spread first: the output_*-named alias is only written on the output
    # basis, so on any other axis it would leave both legs' scatter unknown.
    base_spread_pct = _positive_finite_float(
        base_leg_summary.get("throughput_tok_s_spread_pct")
        or base_leg_summary.get("output_throughput_tok_s_spread_pct")
        or (validation.get("base_block") or {}).get("spread_pct")
        or baseline_summary.get("throughput_tok_s_spread_pct")
        or baseline_summary.get("output_throughput_tok_s_spread_pct")
    )
    final_spread_pct = _positive_finite_float(
        final_summary.get("throughput_tok_s_spread_pct")
        or final_summary.get("output_throughput_tok_s_spread_pct")
        or (validation.get("final_block") or {}).get("spread_pct")
    )
    noise_band_pct = _positive_finite_float(validation.get("noise_band_pct"))
    # A delta smaller than the benches' own scatter, or than the band the
    # Director declared, is not measurable with the data we have.
    significance_threshold_pct = max(noise_band_pct, base_spread_pct, final_spread_pct)
    validation_evidence = {
        "validation_status": (
            validation.get("validation_status")
            or wf.get("validation_status")
            or None
        ),
        "speedup_basis": speedup_basis,
        "delta_pct": delta_pct,
        "noise_band_pct": noise_band_pct or None,
        "baseline_spread_pct": base_spread_pct or None,
        "final_spread_pct": final_spread_pct or None,
        "significance_threshold_pct": significance_threshold_pct or None,
        "delta_exceeds_noise": (
            abs(delta_pct) > significance_threshold_pct
            if significance_threshold_pct > 0.0 else None
        ),
        # Stricter than the threshold above: do the two legs' spread intervals
        # (median +/- spread/2) stay apart? Overlapping intervals mean a single
        # pair of runs could have produced either ordering. None unless BOTH
        # legs reported a spread — one unknown side cannot be assumed tight.
        "spreads_non_overlapping": (
            geak_baseline * (1.0 + base_spread_pct / 200.0)
            < geak_final * (1.0 - final_spread_pct / 200.0)
            if (geak_baseline > 0.0 and geak_final > 0.0
                and base_spread_pct > 0.0 and final_spread_pct > 0.0) else None
        ),
        "beats_orchestrator_same_config": (
            geak_final > orch_same_cfg if orch_same_cfg > 0.0 else None
        ),
        # An intermediate A/B claimed a win that the arbitrated re-check did not
        # confirm. We keep Validate's verdict; this
        # flag is how the disagreement stays visible instead of disappearing.
        "intermediate_win_not_confirmed": intermediate_win_not_confirmed or None,
        # The Validate bench produced no final at all and the number above came
        # from the best accepted intermediate A/B on disk.
        "validate_final_missing": bool(wf.get("validate_final_missing")) or None,
        # How a disk-salvaged headline was chosen out of the candidate pool
        # (see _recover_best_intermediate_win). Absent for an arbitrated result.
        "recovery": wf.get("recovery_evidence") or None,
    }

    # Measurement basis: read back from the bench_summary.json that actually produced these numbers
    # (bench_summarize.py records the throughput or interactivity axis E2E_METRIC selected),
    # so the label never lies about the basis. Falls back to output when neither summary carries it.
    # See run_e2e.md alignment table.
    metric_basis = (
        final_summary.get("metric_basis")
        or wf.get("metric_basis")
        or baseline_summary.get("metric_basis")
        or "aggregate_output_tok_s"
    )

    # ── the caller's KEEP rule ───────────────────────────────────────────────
    # When the handoff states the rule its own verdict applies (Hyperloom's AgentX
    # interactivity grading), "ok" means that rule keeps the published pair, not merely
    # that the final beat the baseline. Guards come from the two legs the pair was read
    # from; a pair with no such legs (a recovered win) cannot show its guards held.
    acceptance = handoff_acceptance(h)
    if acceptance is not None:
        verdict = acceptance_verdict(
            acceptance,
            measured_basis=metric_basis,
            baseline=geak_baseline,
            final=geak_final,
            reference={
                "validation_base_bench_summary": base_leg_summary,
                "setup_baseline": baseline_summary,
            }.get(baseline_basis_source, {}),
            candidate={} if wf.get("recovered_intermediate") else final_summary,
        )
        validation_evidence["acceptance"] = verdict
        if status == "ok" and not verdict["keep"]:
            status = "no_gain"

    # Retained correctness seed identity only; never infer it from the current stack or handoff.
    retained_state = wf.get("state")
    retained_state = retained_state if isinstance(retained_state, dict) else {}
    correctness_baseline = {}
    for key in ("flags", "env", "overlay"):
        value = retained_state.get(f"baseline_{key}")
        correctness_baseline[key] = value if isinstance(value, str) else None
    correctness_baseline["known"] = all(value is not None for value in correctness_baseline.values())

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "result_source": result_source,
        "eval_dir": str(eval_dir),
        "baseline_throughput_tok_s": geak_baseline,
        "final_throughput_tok_s": geak_final,
        "final_throughput_basis": "hot",
        "throughput_speedup": speedup,
        "output_parity": wf.get("output_parity") or validation.get("output_parity") or "unknown",
        # Latency measurement protocol (median ms), aligned field names with Hyperloom. Prefer the
        # value carried on the workflow return / recovered win (e.g. the accepted
        # A/B's candidate leg), then the same-session final/baseline summaries.
        "ttft_ms": wf.get("ttft_ms") or final_summary.get("ttft_ms_median") or baseline_summary.get("ttft_ms_median"),
        "tpot_ms": wf.get("tpot_ms") or final_summary.get("tpot_ms_median") or baseline_summary.get("tpot_ms_median"),
        # Sweep-reuse handles (see interface/run_e2e.md).
        "final_launch_script": final_launch,
        "bench_script": str(eval_dir / "bench_e2e.sh"),
        # Empty string == this run produced no reusable artifact of that kind.
        # Never a path to a directory/diff that holds nothing (see
        # _material_overlay_path / _material_patch_path).
        "final_patch": _material_patch_path(eval_dir, wf),
        "final_overlay": _material_overlay_path(eval_dir, wf),
        "metric_basis": metric_basis,
        # Which bench client measured these numbers. "inferencex" => identical
        # client to Hyperloom/Magpie (benchmark_serving.py); "native" => the
        # backend's own client (small cross-harness differences may remain).
        "bench_client": os.environ.get("BENCH_CLIENT", "native"),
        # Measurement protocol forwarded from the handoff, surfaced at TOP LEVEL
        # (not only inside baseline_basis) so a sweep/validated reuse can pin the
        # SAME num_prompts / random_range_ratio / num_warmups / seed the headline
        # result was measured with. Empty {} when running standalone (no handoff),
        # in which case the reuse path keeps bench_e2e.sh's per-conc defaults.
        "bench_protocol": h.get("bench_protocol") or {},
        # The kernels are only extracted/validated at this single workload point;
        # the caller must redo parity on out-of-regime sweep points.
        "validated_regimes": [workload],
        # What the kernel phase actually did (req: report must carry this).
        "accepted_kernels": wf.get("accepted_kernels") or [],
        "accepted_heads": wf.get("accepted_heads") or [],
        "accepted_config": _accepted_config_with_env_map(wf.get("accepted_config") or {}),
        # Self-describing baseline measurement-protocol + Hyperloom cross-check (see baseline_basis above).
        "baseline_basis": baseline_basis,
        "correctness_baseline": correctness_baseline,
        # Reliability classification is independent of the optimization status.
        "baseline_alignment": baseline_alignment,
        # Cross-handoff Setup-to-upstream comparison. Unlike the compatibility
        # baseline_alignment alias above, this requires launch evidence to
        # declare the server identity verified.
        "handoff_alignment": handoff_alignment,
        # Intra-session movement between Setup and Validate only. It explains
        # denominator changes but never changes handoff_alignment's verdict.
        "measurement_drift": measurement_drift,
        # WHO launched the servers these numbers were measured on, and which
        # kernels those servers selected. This is what tells a reviewer whether
        # baseline_alignment's divergence is a measurement signal at all.
        "serving_stack": serving_stack,
        # Actual server identity inferred from the Setup launch, compared with
        # the upstream observation when the handoff carried one.
        "server_identity": server_identity,
        # Whether the reported delta is distinguishable from measurement noise,
        # and what the arbitration actually concluded (see above). Audit only.
        "validation_evidence": validation_evidence,
        # Cold/hot speedup cross-checks (double-check only; see alignment_metrics above).
        # Does NOT change the promoted final_throughput_tok_s / throughput_speedup.
        "alignment_metrics": alignment_metrics,
        # Exposed for backfill; _enrich_kb_return also carries it to the salvage writer's input.
        "comparability": workload_comparability,
        # Preserve the declaration and the workflow's observed shape for KB backfill/recovery.
        "workload": {
            **(h.get("workload_spec") or {}), **workload,
            **(wf.get("workload") or {}),
            **({
                "isl": round(float(baseline_summary["observed_isl"])),
                "osl": round(float(baseline_summary["observed_osl"])),
                "shape_provenance": "agentx_measured_this_run",
            } if (h.get("workload_spec") or {}).get("kind") == WORKLOAD_KIND_AGENTX
            and _positive_finite_float(baseline_summary.get("observed_isl"))
            and _positive_finite_float(baseline_summary.get("observed_osl")) else {}),
        },
        # Never advertise a report that is not on disk: the old unconditional
        # fallback handed the caller a path to a file that was never written.
        # _emit fills this in with the synthesized report if the workflow died first.
        "report_path": wf.get("report_path") or _existing_report_path(eval_dir),
    }
    # ADDITIVE ONLY. Appended after the dict above is complete so it is self-evident at review time that
    # no existing key is touched, and omitted entirely when the phase did not run.
    tuning_section = _tuning_skillset_section(wf, eval_dir)
    if tuning_section is not None:
        result["tuning_skillset"] = tuning_section
    return result


def _tuning_skillset_section(wf: dict, eval_dir: Path) -> dict | None:
    """Build the ADDITIVE ``tuning_skillset`` block for result.json.

    Contract: this is the ONLY thing the tuning phase adds to result.json. Every pre-existing key keeps
    its name, type and meaning — downstream consumers that do not know about tuning are unaffected, and
    a run without the phase produces a byte-identical result.json (we return None and the key is omitted).

    The tuning gain is ALREADY inside the headline ``final_throughput_tok_s`` / ``throughput_speedup``:
    the phase runs mid-pipeline and its accepted config is inherited by every later measurement. This
    block does not restate the headline, it ATTRIBUTES part of it — which is the question the headline
    cannot answer on its own.

    ``reaches_production_via`` is the load-bearing field. The tuning win is usually a data artifact
    (a config table read from inside a library's package dir, plus a cache that must be dropped), which
    cannot travel in the PYTHONPATH overlay. Finalize folds it into the same ``final_patch`` /
    ``final_launch_script`` the caller already uses, so no new deployment path is introduced — but a
    caller that reproduces the bundle by hand needs to know the deploy step exists.
    """
    t = wf.get("tuning_skillset")
    if not isinstance(t, dict) or not t.get("enabled"):
        return None
    if not t.get("ran"):
        # Enabled but never executed (e.g. a phase-scoped invocation that skipped it). Record that
        # plainly rather than implying a measured no-win.
        return {
            "phase": "TuningSkillset",
            "ran": False,
            "gate": t.get("gate") or "not_run",
            "explanation": "The standalone tuning-skillset phase was enabled but did not run in this invocation.",
        }

    gate = t.get("gate") or "unknown"
    accepted = gate == "accepted"
    delta = t.get("tuning_delta_pct") or 0.0
    share = t.get("share_of_total_gain_pct")

    if accepted:
        explanation = (
            f"The standalone tuning-skillset phase ran before the head-kernel track and measured its own "
            f"interleaved pre/post A/B on the serving config accepted at that point: "
            f"{t.get('pre_tune_throughput_tok_s')} -> {t.get('post_tune_throughput_tok_s')} tok/s "
            f"({delta:+.2f}%). Engagement was verified, so the tuned artifacts were folded into the "
            f"accepted config and every later phase was measured on top of them. This delta is part of "
            f"the headline throughput_speedup, not additional to it"
            + (f"; it accounts for ~{share}% of the run's total gain." if share is not None
               else " (the run had no net gain to apportion).")
        )
    else:
        explanation = (
            f"The standalone tuning-skillset phase ran before the head-kernel track and did not bank a "
            f"win (gate={gate}). Nothing from it was folded into the accepted config, so the headline "
            f"result is unaffected by it. Reason: {t.get('reason') or t.get('summary') or 'not stated'}."
        )

    section: dict[str, Any] = {
        "phase": "TuningSkillset",
        "ran": True,
        "gate": gate,
        "explanation": explanation,
        "mode": t.get("mode") or "",
        "skills_used": t.get("skills_used") or [],
        "ops_tuned": t.get("ops_tuned") or [],
        # Attribution — measured by the phase itself, NOT re-derived from the run baseline.
        "pre_tune_throughput_tok_s": t.get("pre_tune_throughput_tok_s"),
        "post_tune_throughput_tok_s": t.get("post_tune_throughput_tok_s"),
        "tuning_delta_pct": t.get("tuning_delta_pct"),
        "tuning_speedup": t.get("tuning_speedup"),
        "share_of_total_gain_pct": share,
        "noise_floor_pct": t.get("noise_floor_pct"),
        "ab_interleaved": t.get("ab_interleaved"),
        "ab_complete": t.get("ab_complete"),
        "correctness_gate": t.get("correctness_gate"),
        "engagement_verified": t.get("engagement_verified"),
        "engagement_evidence": t.get("engagement_evidence") or "",
        "report_path": t.get("report_path") or str(eval_dir / "tuning" / "tuning_report.md"),
    }

    if t.get("recovered"):
        section["recovery"] = {
            "source": t.get("recovery_source") or "unknown",
            "provisional": True,
        }

    if accepted:
        section["artifacts"] = t.get("artifacts") or []
        section["apply_env"] = t.get("apply_env") or ""
        section["apply_flags"] = t.get("apply_flags") or ""
        section["cache_invalidation"] = t.get("cache_invalidation") or []
        # Data paths the deploy owns inside an installed package tree. Surfaced because a consumer
        # diffing the image against a pristine one would otherwise read these as contamination.
        section["live_tree_files"] = t.get("live_tree_files") or []
        # Non-empty when tuning also needed a code change to make the artifact bind (a routing switch).
        # It is merged into the run's accepted overlay, so it ships via the existing final_overlay key.
        section["apply_overlay"] = t.get("apply_overlay") or ""
        section["deploy_bundle"] = t.get("deploy_bundle") or str(eval_dir / "tuning" / "deploy")
        section["in_final_bundle"] = t.get("in_final_bundle")
        section["final_bundle_engagement_recheck"] = t.get("final_bundle_engagement_recheck") or ""
        section["reaches_production_via"] = {
            "note": (
                "Tuned artifacts are DATA (config tables a library reads from its own package dir, plus "
                "caches that must be invalidated), so they do not travel in final_overlay, which is a "
                "PYTHONPATH overlay for code. They ship through the SAME two handles as everything else: "
                "the tuning diff is concatenated into final_patch, and final_launch_script runs the "
                "bundle's idempotent deploy.sh before starting the server. Reusing final_launch_script "
                "needs no extra steps; applying final_patch by hand also requires the cache_invalidation "
                "commands. When apply_overlay is non-empty the tuning ALSO needed a code change to make "
                "the artifact bind; that half is merged into final_overlay and needs both to be applied."
            ),
            "final_patch_includes_tuning": t.get("in_final_bundle"),
            "final_launch_runs_deploy": t.get("in_final_bundle"),
            "deploy_script": (
                str(Path(t["deploy_bundle"]) / "deploy.sh") if t.get("deploy_bundle")
                else str(eval_dir / "final" / "tuning" / "deploy.sh")
            ),
        }
    return section
