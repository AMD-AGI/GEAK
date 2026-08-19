#!/usr/bin/env python3
"""CI check: every field an agent is ASKED to return must be READ by something.

Why this exists
---------------
The orchestrator declares response schemas (`const X_SCHEMA = obj({...})`) that agents are contractually
required to fill in. A declared-but-never-read field is worse than a missing one: the agent spends
tokens computing it, the reviewer sees it in the JSON and assumes it gated something, and the pipeline
proceeds on a check that does not exist. `synthesized` was exactly this — declared in EXTRACT_OP_SCHEMA,
asked for in the role prompt, written truthfully by the extractor, and consumed by no line of code, so a
fabricated baseline sailed through the head gate.

This check is deliberately syntactic and conservative: it flags a field only when NO plausible read of
that name occurs anywhere outside a schema literal. It cannot prove a field is used correctly; it can
prove a field is used nowhere, which is the failure mode above.

Usage:
  python3 check_schema_consumption.py [--file e2e_workflow.js] [--json] [--list]
  exit 0 = every declared field has a reader; exit 1 = at least one does not.

Waivers: add a field name to ALLOWED_UNCONSUMED below WITH a reason. A waiver is a statement that the
field is documentation for the agent, not an input to a decision.

Stdlib only.
"""
import argparse
import json
import os
import re
import sys

DEFAULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e2e_workflow.js")

# name -> why it is legitimately write-only.
ALLOWED_UNCONSUMED = {
    "notes": "free-text rationale carried into the report/ledger for humans, not a gate input",
    "smoke": "required evidence string the agent must produce; read by humans reviewing the ledger",
    "why": "human-facing justification",
    "note": "human-facing justification",
    "summary": "human-facing narrative",
    "rationale": "human-facing narrative",
    "evidence": "human-facing narrative",
    "risk": "human-facing narrative",

    # --- artifact PATHS: written so a human (or a later run) can open the file. Nothing branches on
    # them; the orchestrator addresses those artifacts by its own EVAL_DIR convention, not by the path
    # the agent reports. If one ever becomes an input to a decision, delete its waiver.
    "baseline_summary_path": "artifact path for the reader; the orchestrator uses its own EVAL_DIR path",
    "bench_script": "artifact path for reproduction; the orchestrator re-derives the bench invocation",
    "profile_topN_md": "human-readable twin of profile_topN.json, which IS consumed",
    "strategy_path": "artifact path for the reader; the strategy object itself is consumed",

    # --- telemetry that lands in the report/ledger. Reported, not gated.
    "baseline_spread_pct": "run-quality telemetry surfaced in the report; the gate uses the A/B medians",
    "total_gpu_time_ms": "profile telemetry; ranking uses pct_gpu_time, which IS consumed",
    "throughput_speedup_vs_baseline": "sweep telemetry; the sweep winner is chosen on absolute tok/s",
    "trials": "sweep search log kept for the report",
    "workload": "the setup's echo of the workload config it was given; the orchestrator holds the source of truth",
    "regime_summary": "prose summary of the regime split for the report; the regime object is consumed",
    "order_of_work": "the strategist's suggested ordering, superseded by the orchestrator's own head queue",
    "drop_list": "advisory 'do not pursue' list for the report; heads are dropped by measured gates",
    "accepted_config": "config bundle passed through to the report writer verbatim",
    "playbook_appended": "acknowledgement that the experience curator wrote its file",
    "per_backend": "full bake-off table kept for the report; the winner fields are consumed",
    "recommend_tier_c": "advisory routing hint; the actual route is decided by winner_kind + admission",
    "winner_editable": "duplicate of winner_kind=='patch_candidate', which is what routes",
    "arbitration_note": "the Director's prose when report and validation disagree",
    "build": "extraction fact recorded for the surgeon's build step, which reads the task meta.json directly",
    "regimes_captured": "recorded in the ledger; regime coverage is enforced inside the frozen unittest",
}


def find_schema_blocks(src):
    """-> [(schema_name, body_text, start, end)] for `const NAME_SCHEMA = obj({ ... })` declarations."""
    out = []
    for m in re.finditer(r"const\s+([A-Za-z0-9_]*SCHEMA)\s*=\s*obj\(\s*\{", src):
        i = m.end() - 1                       # at the '{'
        depth, j = 0, i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append((m.group(1), src[i + 1:j], m.start(), j + 1))
    return out


def _strip_comments(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    return re.sub(r"//[^\n]*", " ", s)


def fields_of(body):
    """Top-level property names of a schema body, ignoring nested object/array literals.

    Walks the text tracking brace/bracket depth so `properties` of an inline nested obj({...}) are not
    mistaken for fields of the outer schema — those are checked as part of their own nested read.
    """
    body = _strip_comments(body)
    names, depth, i, tok_start = [], 0, 0, 0
    while i < len(body):
        c = body[i]
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        elif c == ":" and depth == 0:
            seg = body[tok_start:i].strip().strip(",").strip()
            m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", seg)
            if m:
                names.append(m.group(1))
            tok_start = i + 1
        elif c == "," and depth == 0:
            tok_start = i + 1
        i += 1
    return names


def nested_field_names(body):
    """Every property name at ANY depth — used to catch nested declarations too."""
    body = _strip_comments(body)
    return set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?:\{|arr|obj\()", body))


def consumption_sites(code, field):
    """Plausible reads of `field` in `code` (which already has schema literals removed)."""
    pats = [
        rf"\.{re.escape(field)}\b",                      # o.field
        rf"\[\s*['\"]{re.escape(field)}['\"]\s*\]",      # o['field']
        rf"\b{re.escape(field)}\s*[,}}]",                # const { field } = o   /  { field, x }
        rf"['\"]{re.escape(field)}['\"]",                # 'field' as a key/lookup string
    ]
    return sum(len(re.findall(p, code)) for p in pats)


def check(path):
    with open(path) as fh:
        src = fh.read()
    blocks = find_schema_blocks(src)
    if not blocks:
        return {"error": f"no `const *SCHEMA = obj({{...}})` declarations found in {path}"}, 1

    # Code = the file with every schema literal cut out, so a field's own declaration is not
    # mistaken for a read of it.
    code, last = [], 0
    for _, _, s, e in sorted(blocks, key=lambda b: b[2]):
        code.append(src[last:s])
        last = e
    code.append(src[last:])
    code = _strip_comments("".join(code))

    declared, findings = {}, []
    for name, body, _, _ in blocks:
        for f in fields_of(body) + sorted(nested_field_names(body)):
            declared.setdefault(f, set()).add(name)

    for f in sorted(declared):
        if f in ("type", "properties", "required", "items", "enum", "additionalProperties",
                 "description", "default", "format"):
            continue                                     # JSON-Schema vocabulary, not a payload field
        n = consumption_sites(code, f)
        if n == 0:
            findings.append({"field": f, "schemas": sorted(declared[f]),
                             "waived": f in ALLOWED_UNCONSUMED,
                             "waiver_reason": ALLOWED_UNCONSUMED.get(f, "")})
    unwaived = [x for x in findings if not x["waived"]]
    return {"file": os.path.relpath(path), "num_schemas": len(blocks),
            "num_fields": len(declared), "unconsumed": findings,
            "num_unconsumed_unwaived": len(unwaived)}, (1 if unwaived else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true", help="also print every declared field")
    a = ap.parse_args()

    rep, rc = check(os.path.abspath(a.file))
    if a.json:
        print(json.dumps(rep, indent=2))
        return rc
    if "error" in rep:
        print("ERROR: " + rep["error"])
        return rc
    print(f"{rep['file']}: {rep['num_schemas']} schemas, {rep['num_fields']} declared fields")
    for x in rep["unconsumed"]:
        tag = "WAIVED " if x["waived"] else "UNREAD "
        print(f"  {tag} {x['field']:<32} declared in {', '.join(x['schemas'])}"
              + (f"   ({x['waiver_reason']})" if x["waived"] else ""))
    if rc:
        print(f"\nFAIL: {rep['num_unconsumed_unwaived']} field(s) are requested from an agent and read "
              f"by nothing. Either consume them or add a waiver with a reason in "
              f"check_schema_consumption.ALLOWED_UNCONSUMED.")
    else:
        print("\nOK: every declared field has a reader (or a documented waiver).")
    return rc


if __name__ == "__main__":
    sys.exit(main())
