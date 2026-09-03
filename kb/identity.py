#!/usr/bin/env python3
"""Canonical ids for both workflows, and the fallback ladders that make a miss recoverable.

One file rather than two because the failure this guards against is silent. The `geak` scheme is
CLIENT-DEFINED: the service declares no dimensions for it, does EXACT canonical-id lookup only
(`POST /v1/kb/search` answers `search_unsupported`), and a GET that misses by one segment returns a
plain 404 that is indistinguishable from "nothing was ever recorded here". The API doc says what
that costs, in as many words:

    同一个 kernel 用不同维度个数或顺序描述,会落在不同的 rollup 上,服务端无法把这种情况和
    「两个不同的 kernel」区分开,症状是历史凭空消失。

A reader and a writer that disagree by one segment therefore do not raise — the run just cold
starts and nobody finds out. So both sides of both workflows build their address here, and nowhere
else.

Two schemes, distinguished by the domain segment right after `geak`:

    kernel   geak:kernel:{gfx}:{kernel_name}:{backend}:rocm[:{version}]
    e2e      geak:e2e:{model}:{gfx}:{framework}:{version}:{precision}[:tp_N[:isl_N:osl_N:conc_N]]

Ordering is by how badly a mismatch hurts, most damaging first, because a canonical id can only be
truncated from the RIGHT. That is what makes the tail droppable and the head mandatory:

  * `gfx` leads because an arch mismatch is the nastiest failure mode available. A gfx942-tuned
    patch compiles clean on gfx950 and is simply slower — no error, just a wasted round.
  * `kernel_name` and `backend` next: get either wrong and the patch does not apply at all, which
    at least fails loudly.
  * the ROCm version last, because 7.2 -> 7.3 usually keeps a patch applicable. It is the one
    dimension worth being able to drop.

`producer` is deliberately NOT a dimension, unlike upstream's `kernel:` scheme. It has exactly one
value on our side, and it is not lost by leaving it out: the service stamps it on every record and
the artifact prefix is `kb/<producer>/<session_id>/`. Keeping it would only cost a segment. The
consequence to accept is that dropping the leading segments no longer yields a native `kernel:` id
verbatim, so widening the credential later means running a conversion rather than a prefix rewrite.

THE LADDER IS WRITTEN, NOT COMPUTED. The service does no prefix aggregation — "rollup" there means
the session index under one exact canonical id (`rollup 没有自己的 uuid,直接用 canonical_id
寻址`), and no endpoint takes a prefix or depth. A coarse rung answers a read only if somebody
wrote it. So `canonical_ids()` returns every rung, most specific first, and writers must publish to
ALL of them unconditionally. Publishing conditionally is worse than not publishing: the coarse page
becomes a biased subset of runs and ranks worse than an empty one.

The rungs share one session id, fingerprinted from the MOST SPECIFIC rung, so a measurement is
recognisably one thing at every address it appears. That sharing does NOT make the extra rungs free
on the wire: artifacts live under `kb/<producer>/<session_id>/` and the storage really is shared,
but the file MANIFEST is per (canonical_id, session_id), so a rung that skips put_files reports
file_count 0 and downloads a bundle with no patch in it. Every rung uploads. What the shared id buys
is that the bytes land on the same keys, not that they are sent once.

Stdlib only, like its siblings, so a lane agent can reach it over Bash.
"""

import hashlib
import re

SCHEME = "geak"
KERNEL_DOMAIN = "kernel"
E2E_DOMAIN = "e2e"

# kernel-side framework is `rocm` for triton, hip AND ck. Upstream means "the package that owns the
# source being patched" (vllm, sglang); our kernels are standalone extractions, so that reading has
# nothing to say. What actually moves the stack is the container image, and all three languages ship
# in the same one, so its ROCm version is the single number that says whether two speedups were
# measured on the same thing. It is kept as a literal segment rather than dropped so the ladder has
# somewhere to stop: `...:ck:rocm` is a real address, `...:ck` would be a bare prefix.
KERNEL_FRAMEWORK = "rocm"
UNKNOWN_VERSION = "unspecified"   # framework known, version not observed. Never guessed.
UNKNOWN = "unknown"

_DISALLOWED = re.compile(r"[^a-z0-9._+-]+")
_LEADING = re.compile(r"^[^a-z0-9_]+")
_UNSAFE_IN_SESSION = re.compile(r"[^A-Za-z0-9._-]+")
_NAME_BUDGET = 48         # upstream _NAME_BUDGET
_FINGERPRINT_LEN = 12     # upstream _FINGERPRINT_LEN

# The service's own check, mirrored so a bad segment fails here instead of as a 400 after upload.
# Leading underscores are explicitly legal — Triton kernels are routinely named `_attn_fwd`.
SEGMENT_RE = re.compile(r"^[a-z0-9_][a-z0-9._+-]*$")


class IdentityError(ValueError):
    """A dimension that cannot be a canonical-id segment."""


def segment(value, fallback: str) -> str:
    """Fold a free-form value into one dimension, byte-for-byte as upstream's segment()."""
    folded = _DISALLOWED.sub("-", str(value or "").strip().lower())
    folded = _LEADING.sub("", folded).strip("-")
    if not folded:
        folded = fallback
    return folded.encode("ascii", "ignore").decode("ascii")[:256] or fallback


def counted(prefix: str, value) -> str:
    """`tp_8`, `isl_1024` — a self-describing numeric segment, or "" when there is no number.

    Bare numbers would be legal and unreadable: four of them in a row at the tail of an id (and of
    the mirrored directory path) leaves nothing to check a mis-ordered write against. The prefix
    costs nothing — there is no search to confuse — and makes a truncated id say where it was cut.
    Returning "" rather than a placeholder is what lets the caller drop the whole workload rung: a
    run that did not record its shape must not file itself under `isl_unknown` and strand its
    result on a page no reader will construct.
    """
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    return "%s_%d" % (prefix, number)


def check(canonical_id: str) -> str:
    """Reject an id the service would reject, naming the offending segment."""
    parts = str(canonical_id or "").split(":")
    if len(parts) < 2:
        raise IdentityError("canonical id needs at least a scheme and one dimension: %r"
                            % (canonical_id,))
    for index, part in enumerate(parts):
        if not SEGMENT_RE.fullmatch(part):
            raise IdentityError("canonical_id segment %d=%r is not a valid slug component"
                                % (index, part))
    return canonical_id


# -- kernel scheme ------------------------------------------------------------------------------


def kernel_identity(gfx: str, kernel_name: str, backend: str, rocm_version: str = "") -> dict:
    """The four dimensions of a kernel address, already folded.

    `rocm_version` is cut to `<major>.<minor>` on purpose. detect_stack() reads whatever
    /opt/rocm/.info/version says, which is a full build string on some images (`7.2.0-98765`),
    while the recovered backlog only knows `7.2`. Keyed verbatim those land on different identities
    and a warm start stops seeing half its own history over a patch release. The exact string still
    travels in the record's value, so only the address is coarse.
    """
    return {
        "gpu": segment(gfx, UNKNOWN),
        "kernel_name": segment(kernel_name, UNKNOWN),
        "backend": segment(backend, UNKNOWN),
        "framework": KERNEL_FRAMEWORK,
        "framework_version": _short_version(rocm_version),
    }


def _short_version(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return UNKNOWN_VERSION
    m = re.match(r"\s*(\d+(?:\.\d+)?)", text)
    return segment(m.group(1) if m else text, UNKNOWN_VERSION)


# `0.5.15.post1.dev20260723+g6c9fd0adc5` -> `0.5.15`. Serving stacks are installed from git in these
# images, so `sglang.__version__` carries a PEP 440 dev/local suffix that changes on every rebuild of
# the SAME release. Keyed verbatim, a rebuilt wheel opens a fresh set of pages and the previous run's
# result becomes unreachable — not degraded, invisible, because the lookup is exact and a miss is a
# plain 404. A long opaque string is also one nobody transcribes correctly: a hand-filed record that
# dropped the trailing `+g<hash>` was already enough to hide a real result from the next run on the
# very SAME build. All three rungs missed at once, because framework_version sits in `base` and no
# rung drops it. Rebuild drift and transcription drift are the same failure; both disappear once the
# address is the release.
#
# Three components, not two: `0.5.15` and `0.5.17` are different SGLang releases with different
# kernels, so cutting to `0.5` would merge results that genuinely do not transfer. The exact build
# string still travels in the record's value, so only the address is coarse — same bargain as
# kernel_identity's ROCm cut, one component wider.
_RELEASE_VERSION = re.compile(r"\s*v?(\d+(?:\.\d+){0,2})")


def _release_version(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return UNKNOWN_VERSION
    m = _RELEASE_VERSION.match(text)
    return segment(m.group(1) if m else text, UNKNOWN_VERSION)


def kernel_canonical_ids(identity: dict):
    """Both rungs, most specific first.

    Rung 2 drops the ROCm version and nothing else. There is no third rung dropping `rocm` itself:
    the segment is a constant on the kernel side, so a page without it would hold the same records
    as rung 2 under a different name — a second copy of one thing, which is the exact confusion the
    doc warns about.

    An entry whose ROCm was never observed keys as `unspecified` and still gets both rungs, so the
    coarse page sees it even though the exact page is one nobody will look up.
    """
    head = [SCHEME, KERNEL_DOMAIN, identity["gpu"], identity["kernel_name"],
            identity["backend"], identity["framework"]]
    return [check(":".join(head + [identity["framework_version"]])), check(":".join(head))]


# -- e2e scheme ---------------------------------------------------------------------------------


def e2e_identity(model: str, gfx: str, framework: str, framework_version: str, precision: str,
                 tp=None, isl=None, osl=None, conc=None) -> dict:
    """The e2e address: what is being served, on what, at what shape.

    `framework` here is the SERVING stack (vllm / sglang) — the opposite of the kernel scheme's
    `framework`, which is always `rocm`. The two workflows genuinely key on different things: an
    extracted kernel is standalone and only its compile stack matters, while an e2e result is a
    statement about a server and is worthless without knowing which one. Note this also collides
    with e2e_workflow.js's own `args.backend`, which names the serving adapter, whereas the kernel
    scheme's `backend` dimension names the kernel language. Nothing shares a variable across the
    two, and this is why.

    `tp` sits AFTER precision and before the workload shape so the ladder can drop the measured
    point while keeping the deployment config. Anything unparseable folds to "" and simply removes
    the rung that would have carried it.

    `framework_version` is cut to `<major>.<minor>.<patch>` — see _release_version. It is the one
    dimension here that a rebuild can change without anything about the deployment changing, and
    unlike the workload dims it cannot be dropped by a coarser rung.
    """
    return {
        "model": segment(model, UNKNOWN),
        "gpu": segment(gfx, UNKNOWN),
        "framework": segment(framework, UNKNOWN),
        "framework_version": _release_version(framework_version),
        "precision": segment(precision, UNKNOWN),
        "tp": counted("tp", tp),
        "isl": counted("isl", isl),
        "osl": counted("osl", osl),
        "conc": counted("conc", conc),
    }


def e2e_canonical_ids(identity: dict):
    """Up to three rungs, most specific first: exact workload, TP config, model.

    The last rung is TP-agnostic on purpose, and it is not just a fallback — it is the only page
    that can answer "how many ways should I shard this", because that question needs TP4 and TP8
    ranked against each other rather than filed apart. The middle rung answers "given TP=8, how do
    I configure it". Rungs that would need a dimension the run did not record are omitted, never
    filled with a placeholder.
    """
    base = [SCHEME, E2E_DOMAIN, identity["model"], identity["gpu"], identity["framework"],
            identity["framework_version"], identity["precision"]]
    rungs = []
    tp, isl, osl, conc = identity["tp"], identity["isl"], identity["osl"], identity["conc"]
    if tp and isl and osl and conc:
        rungs.append(":".join(base + [tp, isl, osl, conc]))
    if tp:
        rungs.append(":".join(base + [tp]))
    rungs.append(":".join(base))
    return [check(r) for r in rungs]


# -- session id ---------------------------------------------------------------------------------


def session_id(exact_canonical_id: str, name: str, digest: str, producer: str = SCHEME) -> str:
    """`<producer>-<name>-<identity fp>-<content digest>`, upstream's shape.

    Fingerprinted from the MOST SPECIFIC rung and reused verbatim on the coarser ones. Upstream
    includes the identity fingerprint because artifacts are partitioned by session id alone, so an
    id repeated across two identities makes them share an artifact path. Here that sharing is the
    point — the rungs are one measurement filed at several addresses and the bytes are identical —
    but it only stays safe while the fingerprint comes from a rung that is unique per measurement.
    Fingerprinting each rung separately would instead give the same patch three unrelated ids, and
    the coarse pages would stop being reproductions of the exact one.
    """
    legible = _UNSAFE_IN_SESSION.sub("-", str(name or "")).strip("-.")
    legible = legible[:_NAME_BUDGET].strip("-.") or UNKNOWN
    fp = hashlib.sha256(str(exact_canonical_id or "").encode()).hexdigest()[:_FINGERPRINT_LEN]
    port = _UNSAFE_IN_SESSION.sub("", str(digest or ""))[:_FINGERPRINT_LEN]
    return ("%s-%s-%s-%s" % (segment(producer, SCHEME), legible, fp, port)).strip("-")


__all__ = ["E2E_DOMAIN", "IdentityError", "KERNEL_DOMAIN", "KERNEL_FRAMEWORK", "SCHEME",
           "SEGMENT_RE", "UNKNOWN", "UNKNOWN_VERSION", "check", "counted", "e2e_canonical_ids",
           "e2e_identity", "kernel_canonical_ids", "kernel_identity", "segment", "session_id"]
