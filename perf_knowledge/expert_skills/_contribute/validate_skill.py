#!/usr/bin/env python3
"""validate_skill.py — gate an expert skill before it can land as `validated`.

Two-sided gate (see expert_skills/README.md):
  1. EFFICACY   — on a matching model/shape the skill achieves >= its `expects` (isolated or e2e).
  2. DO-NO-HARM — integrating it does not regress a control scenario / non-trigger run.

This tool does NOT itself spin up a server or a GPU. The actual measurement is produced by the real
harness (kernel_workflow for scope:kernel, e2e_workflow for scope:e2e) so numbers stay honest and
on-box. validate_skill.py has three modes:

  --static            schema + operator alignment + required sections + links. No GPU. (CI default.)
  --emit-plan         print the exact Workflow invocation to measure this skill (by scope).
  --record ...        check supplied measured numbers against `expects`, then stamp the skill's
                      validation block (status: validated|failed) and reindex.

Examples:
  python _contribute/validate_skill.py flydsl_fp8_gemm_playbook --static
  python _contribute/validate_skill.py flydsl_fp8_gemm_playbook --emit-plan --model /models/Qwen3.5-27B-FP8
  python _contribute/validate_skill.py flydsl_fp8_gemm_playbook --record \
      --artifact /path/eval_dir --e2e-pct 2.1 --parity pass --gpu gfx942/MI300X --model Qwen3.5-27B-FP8
"""
import argparse, os, re, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKILLS_DIR = os.path.join(ROOT, "skills")
CAP_INDEX = os.path.normpath(os.path.join(ROOT, "..", "index", "capability_index.yaml"))
GEAK = os.path.normpath(os.path.join(ROOT, "..", ".."))
REQUIRED_SECTIONS = ["When to use", "Mechanism", "Procedure", "Do-no-harm notes", "Sources"]
FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


def load(skill_id):
    path = os.path.join(SKILLS_DIR, skill_id, "skill.md")
    if not os.path.exists(path):
        sys.exit(f"ERROR: no such skill: {path}")
    txt = open(path).read()
    m = FM_RE.match(txt)
    if not m:
        sys.exit(f"ERROR: {path}: no YAML frontmatter")
    return path, yaml.safe_load(m.group(1)), m.group(2), txt


def static_check(fm, body):
    errs = []
    for k in ("id", "scope", "match", "expects"):
        if k not in fm:
            errs.append(f"missing frontmatter key: {k}")
    if fm.get("scope") not in ("kernel", "e2e"):
        errs.append(f"scope must be kernel|e2e (got {fm.get('scope')!r})")
    op = (fm.get("match") or {}).get("operator")
    if os.path.exists(CAP_INDEX):
        ops = {c["operator"] for c in (yaml.safe_load(open(CAP_INDEX)).get("candidates") or [])}
        if op not in ops:
            errs.append(f"match.operator '{op}' not in capability_index.yaml")
    exp = fm.get("expects") or {}
    if fm.get("scope") == "kernel" and "isolated_speedup_min" not in exp:
        errs.append("kernel scope needs expects.isolated_speedup_min")
    if fm.get("scope") == "e2e" and "e2e_delta_min_pct" not in exp:
        errs.append("e2e scope needs expects.e2e_delta_min_pct")
    for sec in REQUIRED_SECTIONS:
        if f"## {sec}" not in body:
            errs.append(f"missing body section: ## {sec}")
        elif not _section_filled(body, sec):
            errs.append(f"body section '## {sec}' is empty / placeholder only")
    return errs


def _section_filled(body, sec):
    m = re.search(rf"## {re.escape(sec)}\n(.*?)(?=\n## |\Z)", body, re.S)
    if not m:
        return False
    content = re.sub(r"<!--.*?-->", "", m.group(1), flags=re.S).strip()
    return len(content) > 0


def emit_plan(skill_id, fm, args):
    scope = fm.get("scope")
    if scope == "e2e":
        model = args.model or "<MODEL_PATH>"
        print("# EFFICACY (e2e_workflow, Director same-session A/B):")
        print(f"Workflow scriptPath={GEAK}/e2e_workflow/e2e_workflow.js args:")
        print(f"  model_path={model} workflow_dir={GEAK}/e2e_workflow use_expert_skills=true")
        print(f"  task='reproduce expert_skill:{skill_id} on the matching head; gate by e2e A/B'")
        print("# DO-NO-HARM (control model that does NOT match the selector must stay within noise band):")
        print(f"  model_path=<CONTROL_MODEL> use_expert_skills=true  # expect |e2e delta| < noise band")
    else:
        print("# EFFICACY (kernel_workflow, isolated A/B vs the immutable oracle):")
        print(f"Workflow scriptPath={GEAK}/kernel_workflow/kernel_workflow.js args:")
        print(f"  kernel_path=<OP_TASK_DIR> workflow_dir={GEAK}/kernel_workflow use_expert_skills=true")
        print(f"  target_language={(fm.get('match') or {}).get('to_backend') or 'triton'}")
        print(f"  task='reproduce expert_skill:{skill_id}; beat oracle, hold parity'")
    print("\nThen stamp the result with:  validate_skill.py", skill_id,
          "--record --artifact <eval_dir> ...")


def record(path, fm, body, txt, args):
    exp = fm.get("expects") or {}
    scope = fm.get("scope")
    ok, reasons = True, []
    if args.parity and args.parity != "pass" and exp.get("parity", "required") == "required":
        ok = False; reasons.append(f"parity={args.parity} but required")
    if scope == "kernel":
        need = float(exp.get("isolated_speedup_min", 1.0))
        got = args.isolated
        if got is None:
            sys.exit("ERROR: --isolated required for kernel scope")
        if got < need:
            ok = False; reasons.append(f"isolated {got} < expects {need}")
    else:
        need = float(exp.get("e2e_delta_min_pct", 0.0))
        got = args.e2e_pct
        if got is None:
            sys.exit("ERROR: --e2e-pct required for e2e scope")
        if got < need:
            ok = False; reasons.append(f"e2e +{got}% < expects +{need}%")
    if not args.artifact:
        sys.exit("ERROR: --artifact <eval_dir> required to record a result")

    fm.setdefault("validation", {})
    fm["validation"]["status"] = "validated" if ok else "failed"
    fm["validation"]["last_verified"] = args.date or ""
    fm["validation"]["gpu"] = args.gpu or ""
    fm["validation"]["model"] = args.model or ""
    fm["validation"]["measured"] = {
        "isolated": args.isolated if args.isolated is not None else "",
        "e2e_pct": args.e2e_pct if args.e2e_pct is not None else "",
        "parity": args.parity or "",
    }
    fm["validation"]["artifact"] = args.artifact
    with open(path, "w") as f:
        f.write("---\n")
        f.write(yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=100))
        f.write("---\n")
        f.write(body)
    print(f"recorded: status={fm['validation']['status']}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    os.system(f"python3 {os.path.join(HERE, 'scaffold.py')} --reindex >/dev/null 2>&1")
    if not ok:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("skill_id")
    p.add_argument("--static", action="store_true")
    p.add_argument("--emit-plan", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--artifact", default=""); p.add_argument("--gpu", default="")
    p.add_argument("--model", default=""); p.add_argument("--date", default="")
    p.add_argument("--isolated", type=float, default=None)
    p.add_argument("--e2e-pct", dest="e2e_pct", type=float, default=None)
    p.add_argument("--parity", default="")
    a = p.parse_args()
    path, fm, body, txt = load(a.skill_id)

    if a.emit_plan:
        return emit_plan(a.skill_id, fm, a)
    if a.record:
        return record(path, fm, body, txt, a)
    # default / --static
    errs = static_check(fm, body)
    if errs:
        print("STATIC FAIL:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"STATIC OK: {a.skill_id} (scope={fm.get('scope')}, operator={fm['match']['operator']}). "
          f"Run with --emit-plan to get the on-box measurement command.")


if __name__ == "__main__":
    main()
