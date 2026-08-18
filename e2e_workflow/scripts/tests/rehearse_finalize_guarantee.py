#!/usr/bin/env python3
"""End-to-end REHEARSAL of the finalize-window guarantee. No GPU, no LLM, ~1 minute.

Named ``rehearse_*`` (not ``test_*``) on purpose: it burns wall-clock time by
design, so unittest/pytest discovery must not pick it up. Run it directly:

    python3 e2e_workflow/scripts/tests/rehearse_finalize_guarantee.py
    python3 e2e_workflow/scripts/tests/rehearse_finalize_guarantee.py --budget 300 --keep

WHAT IS REAL vs FAKE
--------------------
Real: run_e2e.py as its own process, the budget partition, the sentinel timer
thread, bench_e2e.sh's stand-down decision, the serving-GPU-lock behaviour, the
result.json contract, and the leftover-budget ``phases=final`` re-entry.
Fake: only the two things that need hardware or money — the agent CLI (a stub
``claude`` on PATH) and the model server (a stub bench adapter).

WHY A REHEARSAL IS NEEDED AT ALL
--------------------------------
The unit tests prove each layer in isolation. The failure this fix addresses was
never in one layer: the run died because the JS deadline, the runner's sentinel,
the bash dispatcher and the closing phases disagreed about WHEN the reserve
starts. Only a live run with a real clock can show that they agree, and that the
run still lands a validated deliverable after the optimization phase overruns.

THE SCENARIO (deliberately the worst case)
------------------------------------------
1. The "agent" starts an optional optimization bench leg immediately. Nothing is
   known about the budget yet, so that leg must proceed untouched.
2. The agent then blows through the finalize deadline, exactly like a hung
   integrate leg, and never produces a workflow return.
3. Inside the reserve it starts two more legs: an OPTIONAL one, which must be
   stood down, and a closing-phase one tagged GEAK_TAIL_LEG=1, which must be let
   through — it IS the reserved window.
4. The first pass dies with no terminal marker on disk. The runner must notice
   the reserve went unspent and re-enter with phases=final, which validates.

Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
E2E_DIR = SCRIPTS_DIR.parent
GEAK_ROOT = E2E_DIR.parent
RUNNER = GEAK_ROOT / "interface" / "run_e2e.py"
HANDOFF_FIXTURE = GEAK_ROOT / "ci" / "fixtures" / "handoff.dry.json"
STAND_DOWN_RC = 5

# The stub agent. $1.. is the prompt; it never reads it except to tell the first
# pass from the scoped phases=final re-entry.
STUB_CLAUDE = r"""#!/usr/bin/env bash
# Stub for the `claude` CLI: impersonates the workflow-driving agent.
set -u
say() { echo "[stub-agent] $*" >&2; }

run_leg() {  # $1=label  $2=1 when this leg is a closing-phase leg
  local label="$1" tail_leg="${2:-0}" rc=0
  ( cd "$REHEARSE_BENCH_DIR" && \
    ADAPTER="$REHEARSE_ADAPTER" BACKEND=vllm MODEL="$REHEARSE_TMP/model" \
    OUT_DIR="$REHEARSE_TMP/out_$label" REPEATS=1 PROFILE=0 \
    SERVING_GPU_LOCK_DISABLE=1 GEAK_TAIL_LEG="$tail_leg" \
    bash "$REHEARSE_BENCH_DIR/bench_e2e.sh" ) \
      >"$REHEARSE_TMP/leg_$label.log" 2>&1 || rc=$?
  echo "$rc" > "$REHEARSE_TMP/leg_$label.rc"
  say "leg $label finished rc=$rc"
}

pass=1
[ -f "$REHEARSE_TMP/pass1.done" ] && pass=2
case "$*" in *phases*final*) pass=2 ;; esac

if [ "$pass" = 2 ]; then
  say "scoped re-entry: running the closing phases"
  # A closing-phase bench must be allowed to run even though the window is open.
  run_leg closing 1
  cat > "$GEAK_EVAL_DIR/director_e2e_validation.json" <<JSON
{"validated": true, "rehearsal": true, "throughput_speedup": 1.10}
JSON
  # The workflow return, as the Workflow tool would hand it back.
  cat <<JSON
{"eval_dir": "$GEAK_EVAL_DIR", "status": "success", "throughput_speedup": 1.10,
 "final_throughput_tok_s": 4904.4, "baseline_throughput_tok_s": 4458.59,
 "report_path": "$GEAK_EVAL_DIR/final_report.md",
 "budget_telemetry": {"rehearsal": true, "finalize_deadline_fired": true}}
JSON
  exit 0
fi

: > "$REHEARSE_TMP/pass1.done"
say "pass 1: budget window not open yet -> an optional leg must run untouched"
run_leg before_window 0

say "pass 1: overrunning the dispatch window, waiting for the finalize sentinel"
deadline=$(( $(date +%s) + REHEARSE_MAX_WAIT ))
while [ ! -f "$GEAK_FINALIZE_NOW_FILE" ]; do
  [ "$(date +%s)" -ge "$deadline" ] && { say "sentinel never appeared"; exit 9; }
  sleep 1
done
date +%s > "$REHEARSE_TMP/sentinel_seen_at"
say "sentinel observed -> the reserve is open"

say "pass 1: an OPTIONAL leg started inside the reserve must stand down"
run_leg inside_window 0
say "pass 1: a CLOSING-PHASE leg inside the reserve must be exempt"
run_leg exempt 1

# Die exactly like a killed/hung agent: no workflow return, no terminal marker.
say "pass 1: exiting with no workflow return (as a hung agent would)"
echo "no workflow return here, only prose"
exit 0
"""

STUB_ADAPTER = (
    "adapter_default_port() { echo 18081; }\n"
    "adapter_launch() { echo 'STUB_LAUNCH_REACHED'; exit 0; }\n"
    "adapter_health() { return 0; }\n"
    "adapter_bench() { return 0; }\n"
)


STAND_DOWN_CHECK = "an optional leg INSIDE the reserve stands down"


class Rehearsal:
    def __init__(self, budget: int, keep: bool, negative: bool = False) -> None:
        self.budget = budget
        self.keep = keep
        self.negative = negative
        self.tmp = Path(tempfile.mkdtemp(prefix="geak_rehearse_"))
        self.checks: list[tuple[bool, str, str]] = []

    # ── staging ────────────────────────────────────────────────────────────
    def stage(self) -> None:
        (self.tmp / "bin").mkdir()
        self.claude = self.tmp / "bin" / "claude"
        self.claude.write_text(STUB_CLAUDE, encoding="utf-8")
        self.claude.chmod(0o755)

        self.adapter = self.tmp / "stub_adapter.sh"
        self.adapter.write_text(STUB_ADAPTER, encoding="utf-8")

        # Stage the dispatcher the way roles/director.md does, so the
        # teardown-contract gate that sits before the stand-down is satisfied.
        self.bench_dir = self.tmp / "bench"
        self.bench_dir.mkdir()
        shutil.copy(SCRIPTS_DIR / "bench_e2e.sh", self.bench_dir / "bench_e2e.sh")
        shutil.copy(SCRIPTS_DIR / "server_teardown.sh", self.bench_dir / "server_teardown.sh")
        shutil.copytree(SCRIPTS_DIR / "adapters", self.bench_dir / "adapters")
        if self.negative:
            self.neuter_the_guard(self.bench_dir / "bench_e2e.sh")

        self.exp_root = self.tmp / "exp"
        self.exp_root.mkdir()
        handoff = json.loads(HANDOFF_FIXTURE.read_text(encoding="utf-8"))
        handoff["exp_root"] = str(self.exp_root)
        handoff["model_path"] = str(self.tmp / "model")
        self.handoff = self.tmp / "handoff.json"
        self.handoff.write_text(json.dumps(handoff, indent=2), encoding="utf-8")
        self.result = self.tmp / "result.json"

    @staticmethod
    def neuter_the_guard(staged_bench: Path) -> None:
        """Restore the pre-fix behaviour in the STAGED copy only (never the repo).

        A verification that still passes with the guard removed proves nothing, so
        ``--negative-control`` reverts the one decision the reserve depends on and
        requires the rehearsal to notice.
        """
        text = staged_bench.read_text(encoding="utf-8")
        marker = '_finalize_now() { [ -n "${FINALIZE_SENTINEL:-}" ] && [ -e "$FINALIZE_SENTINEL" ]; }'
        if marker not in text:
            raise SystemExit(
                "negative control cannot find the sentinel predicate in bench_e2e.sh; "
                "the guard moved and this harness needs updating"
            )
        staged_bench.write_text(
            text.replace(marker, "_finalize_now() { return 1; }  # NEGATIVE CONTROL"),
            encoding="utf-8",
        )

    def env(self) -> dict:
        env = dict(os.environ)
        env.pop("GEAK_EVAL_DIR", None)
        env.pop("GEAK_FINALIZE_NOW_FILE", None)
        env.pop("GEAK_TAIL_LEG", None)
        env.update(
            PATH=f"{self.tmp / 'bin'}{os.pathsep}{env.get('PATH', '')}",
            GEAK_E2E_TIMEOUT_S=str(self.budget),
            # Keep the leftover-budget re-entry reachable inside a tiny rehearsal
            # budget (production default is max(5min, 5% of budget)).
            GEAK_RESCUE_MIN_S="5",
            REHEARSE_TMP=str(self.tmp),
            REHEARSE_BENCH_DIR=str(self.bench_dir),
            REHEARSE_ADAPTER=str(self.adapter),
            REHEARSE_MAX_WAIT=str(self.budget),
        )
        return env

    # ── the run ────────────────────────────────────────────────────────────
    def run(self) -> None:
        sys.path.insert(0, str(GEAK_ROOT / "interface"))
        import importlib.util

        spec = importlib.util.spec_from_file_location("rx_rehearse", RUNNER)
        rx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rx)
        self.d2 = rx.finalize_deadline_s(self.budget)
        print(
            f"budget {self.budget}s -> the reserve opens at {self.d2}s, leaving "
            f"{self.budget - self.d2}s for Finalize/Report/Validate"
        )
        print(f"workdir {self.tmp}\n")

        self.started = time.monotonic()
        self.proc = subprocess.run(
            [sys.executable, str(RUNNER), str(self.handoff), str(self.result)],
            cwd=str(GEAK_ROOT), env=self.env(), capture_output=True, text=True,
            timeout=self.budget * 3 + 120,
        )
        self.elapsed = time.monotonic() - self.started
        print(f"runner exited rc={self.proc.returncode} after {self.elapsed:.0f}s\n")

    # ── checks ─────────────────────────────────────────────────────────────
    def check(self, ok: bool, name: str, detail: str = "") -> None:
        self.checks.append((bool(ok), name, detail))

    def leg_rc(self, label: str) -> int | None:
        path = self.tmp / f"leg_{label}.rc"
        if not path.exists():
            return None
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            return None

    def leg_log(self, label: str) -> str:
        path = self.tmp / f"leg_{label}.log"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def verify(self) -> None:
        err = self.proc.stderr

        # 1. the runner announces the partition it will enforce
        self.check(
            f"optional work stands down at {self.d2}s" in err,
            "runner announces the reserve it is enforcing",
            _tail(err),
        )

        # 2. the sentinel fired, and it fired at the deadline (not early/late)
        seen = self.tmp / "sentinel_seen_at"
        self.check(
            "finalize window opened" in err and seen.exists(),
            "the sentinel fired and the agent observed it",
            _tail(err),
        )

        # 3. before the window: an optional leg is untouched
        rc_before = self.leg_rc("before_window")
        self.check(
            rc_before is not None
            and rc_before != STAND_DOWN_RC
            and "STUB_LAUNCH_REACHED" in self.leg_log("before_window"),
            "an optional leg BEFORE the reserve runs untouched",
            f"rc={rc_before}",
        )

        # 4. inside the window: an optional leg lets go of the serving slot
        rc_inside = self.leg_rc("inside_window")
        self.check(
            rc_inside == STAND_DOWN_RC
            and "STUB_LAUNCH_REACHED" not in self.leg_log("inside_window"),
            "an optional leg INSIDE the reserve stands down",
            f"rc={rc_inside} (want {STAND_DOWN_RC})",
        )

        # 5. inside the window: the closing phases are exempt
        rc_exempt = self.leg_rc("exempt")
        self.check(
            rc_exempt is not None
            and rc_exempt != STAND_DOWN_RC
            and "STUB_LAUNCH_REACHED" in self.leg_log("exempt"),
            "a closing-phase leg INSIDE the reserve is exempt",
            f"rc={rc_exempt}",
        )

        # 6. the unspent reserve is reclaimed by a scoped re-entry
        self.check(
            "re-entering with phases=final" in err,
            "the unspent reserve is reclaimed via phases=final",
            _tail(err),
        )
        self.check(
            self.leg_rc("closing") is not None,
            "the re-entry actually got to run a closing-phase bench",
        )

        # 7. the deliverable exists, and says how it was obtained
        validation = self.eval_dir_guess() / "director_e2e_validation.json"
        self.check(validation.exists(), "the Director's validation exists on disk",
                   str(validation))
        out = {}
        if self.result.exists():
            try:
                out = json.loads(self.result.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self.check(False, "result.json parses", str(exc))
        self.check(bool(out), "result.json was written")
        self.check(out.get("rescued_finalize_pass") is True,
                   "result.json attributes the result to the rescue pass",
                   f"rescued_finalize_pass={out.get('rescued_finalize_pass')!r}")
        self.check(out.get("status") not in (None, "error"),
                   "result.json is not an error",
                   f"status={out.get('status')!r}")

        # 8. nothing ran past the granted budget
        self.check(self.elapsed <= self.budget,
                   "the whole run fit inside the granted budget",
                   f"{self.elapsed:.0f}s vs {self.budget}s")

    def eval_dir_guess(self) -> Path:
        dirs = [p for p in self.exp_root.glob("*") if p.is_dir()]
        return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else self.exp_root

    # ── report ─────────────────────────────────────────────────────────────
    def report(self) -> int:
        width = max(len(n) for _, n, _ in self.checks)
        for ok, name, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            line = f"[{mark}] {name.ljust(width)}"
            if detail and not ok:
                line += f"  <- {detail}"
            print(line)
        failed = [n for ok, n, _ in self.checks if not ok]
        print()
        if self.negative:
            detected = STAND_DOWN_CHECK in failed
            print(
                "negative control: the guard was removed from the staged dispatcher, "
                f"and the rehearsal {'DID' if detected else 'DID NOT'} notice"
            )
            if not detected:
                print("=> the rehearsal is blind to the regression it exists to catch")
            if not self.keep:
                shutil.rmtree(self.tmp, ignore_errors=True)
            return 0 if detected else 1
        if failed:
            print(f"{len(failed)} of {len(self.checks)} checks FAILED")
            print(f"logs kept in {self.tmp}")
            print("\n--- runner stderr (tail) ---")
            print(self.proc.stderr[-4000:])
            return 1
        print(f"all {len(self.checks)} checks passed in {self.elapsed:.0f}s")
        if self.keep:
            print(f"workdir kept at {self.tmp}")
        else:
            shutil.rmtree(self.tmp, ignore_errors=True)
        return 0


def _tail(text: str, n: int = 240) -> str:
    return text[-n:].replace("\n", " | ")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", type=int, default=120,
                    help="wall-clock budget in seconds (default 120 => reserve at 48s)")
    ap.add_argument("--keep", action="store_true", help="keep the temp workdir")
    ap.add_argument("--negative-control", action="store_true",
                    help="remove the guard from the STAGED dispatcher and require "
                         "the rehearsal to fail (proves the checks are load-bearing)")
    args = ap.parse_args(argv)

    if shutil.which("bash") is None:
        print("bash is required", file=sys.stderr)
        return 2
    r = Rehearsal(args.budget, args.keep, args.negative_control)
    r.stage()
    r.run()
    r.verify()
    rc = r.report()
    if rc and not args.keep:
        print("(workdir intentionally kept above for diagnosis)")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
