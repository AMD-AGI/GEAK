#!/usr/bin/env python3
"""Fusion priors: the known-fusion cards in knowledge/learned, and the recall
gate that makes a run ACCOUNT for each of them.

Why this exists
---------------
On DSR1 the same model, the same trace and the same roles produced a different
fusion set on different runs. Nothing was broken -- the analyst simply
rediscovered the opportunity space from scratch each time, and rediscovery is
not deterministic. A fusion that landed +11.8% e2e in one run was never even
proposed in the next, and there was no artifact anywhere that said so, because
"never proposed" leaves no trace.

The fix is the same shape as every other gate in this pipeline: give the phase a
DENOMINATOR. `knowledge/learned/INDEX.md` already records the fusions we have
measured. Each such card is a PRIOR the run must dispose of explicitly -- carry
it as a candidate, or say in one line why it does not apply here. A prior that
is silently absent is exactly the run-to-run instability the user reported.

What a prior is NOT
-------------------
Not a mandate, and not a filter. Per `knowledge/learned/README.md` the KB is
ADD-only: a card may add a candidate to try, never remove one the profile found,
and never substitute for measurement. `not_applicable` with an honest reason is
a perfectly good disposition -- the gate is about the SENTENCE existing, not
about the answer being yes.
"""

import os
import re

# INDEX.md groups cards under `## <group>` headings. Only this group is a fusion
# prior; the rest of the KB (GEMM/attention routing) is read by other roles.
FUSION_GROUP = "kernel fusion"

# `type: method` cards state how to RUN the phase (give it a denominator, take
# both traces), not which fusion to try. They are worth reading and they are
# indexed in the same group, but demanding a per-candidate disposition for them
# would be nonsense, so they are advisory-only.
DISPOSABLE_TYPES = ("routing", "lever")

DISPOSITIONS = ("candidate", "already_engaged", "not_applicable")

# A reason has to carry information. These are the phrasings that were used on
# DSR1 to mean "I did not look", and they are exactly what the gate is for.
_EMPTY_REASONS = {
    "", "-", "n/a", "na", "none", "no", "skip", "skipped", "not attempted",
    "not applicable", "tbd", "todo", "unknown",
}

_LINE_CARD = re.compile(r"\(([A-Za-z0-9._-]+\.md)\)\s*$")
_FRONT_FIELD = re.compile(r"^([a-z_]+):\s*(.*)$")


def _card_front_matter(path):
    """Return the card's frontmatter dict, or {} if it has none/unreadable."""
    try:
        with open(path) as fh:
            text = fh.read()
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    body = text.split("---", 2)
    if len(body) < 3:
        return {}
    front = {}
    for line in body[1].splitlines():
        match = _FRONT_FIELD.match(line.strip())
        if match:
            front[match.group(1)] = match.group(2).strip()
    return front


def _title(path):
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("# "):
                    return line[2:].strip()
    except OSError:
        pass
    return os.path.basename(path)


def load_priors(index_path, group=FUSION_GROUP):
    """Parse INDEX.md and return the fusion cards, in index order.

    Each entry: {slug, path, title, key, type, confidence, index_line,
    disposable}. `disposable` is False for method cards -- they are surfaced to
    the reader but never demand a disposition.
    """
    index_path = os.path.abspath(index_path)
    cards_dir = os.path.dirname(index_path)
    try:
        with open(index_path) as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    priors = []
    in_group = False
    for line in lines:
        if line.startswith("## "):
            in_group = line[3:].strip().lower() == group.lower()
            continue
        if not in_group or not line.strip().startswith("- "):
            continue
        match = _LINE_CARD.search(line.strip())
        if not match:
            continue
        slug = match.group(1)[:-3]
        path = os.path.join(cards_dir, match.group(1))
        front = _card_front_matter(path)
        card_type = (front.get("type") or "").strip().lower()
        priors.append({
            "slug": slug,
            "path": path,
            "title": _title(path),
            "key": front.get("key", ""),
            "type": card_type,
            "confidence": front.get("confidence", ""),
            "last_seen": front.get("last_seen", ""),
            "index_line": line.strip()[2:].strip(),
            # An unreadable / frontmatter-less card is treated as disposable:
            # failing OPEN here would let a card drop out of the denominator by
            # being malformed, which is the failure mode this gate exists for.
            "disposable": card_type in DISPOSABLE_TYPES or not card_type,
        })
    return priors


def _bad_reason(reason):
    return str(reason or "").strip().lower() in _EMPTY_REASONS


def check_dispositions(priors, payload, candidate_ids=None):
    """Gate `payload['prior_dispositions']` against the disposable priors.

    Returns (errors, rows). `rows` covers EVERY prior (method cards included, as
    advisory) so the report can render the full known-fusion list with what this
    run did about each one.
    """
    errors = []
    declared = payload.get("prior_dispositions")
    disposable = [p for p in priors if p["disposable"]]
    if not priors:
        return errors, []
    if not disposable:
        # Only method cards in the group: nothing to dispose of, but they still
        # belong in the report so the reader sees the whole known-fusion list.
        declared = declared if isinstance(declared, list) else []
    elif declared is None:
        errors.append(
            "prior_dispositions missing: %d known fusion card(s) in "
            "knowledge/learned/INDEX.md need an explicit disposition "
            "(candidate | already_engaged | not_applicable). A prior that is "
            "silently absent is why the fusion set changes run to run."
            % len(disposable))
        declared = []
    elif not isinstance(declared, list):
        errors.append("prior_dispositions must be a list")
        declared = []

    by_slug = {}
    for entry in declared:
        if not isinstance(entry, dict):
            errors.append("prior_dispositions entries must be objects")
            continue
        slug = str(entry.get("card") or entry.get("slug") or "").strip()
        if slug.endswith(".md"):
            slug = slug[:-3]
        if not slug:
            errors.append("prior_dispositions entry missing `card`")
            continue
        by_slug[slug] = entry

    known = {p["slug"] for p in priors}
    for slug in sorted(set(by_slug) - known):
        errors.append(
            "prior_dispositions cites unknown card %r (not a `## %s` line in "
            "INDEX.md)" % (slug, FUSION_GROUP))

    rows = []
    for prior in priors:
        entry = by_slug.get(prior["slug"])
        row = dict(prior)
        row["disposition"] = None
        row["reason"] = ""
        row["candidate_id"] = ""
        if entry is not None:
            disposition = str(entry.get("disposition") or "").strip()
            row["disposition"] = disposition
            row["reason"] = str(entry.get("reason") or "").strip()
            row["candidate_id"] = str(entry.get("candidate_id") or "").strip()
        if not prior["disposable"]:
            rows.append(row)
            continue
        if entry is None:
            errors.append(
                "known fusion prior %s (%s) has no disposition -- carry it as a "
                "candidate or say why it does not apply to this run"
                % (prior["slug"], prior["title"]))
            rows.append(row)
            continue
        if row["disposition"] not in DISPOSITIONS:
            errors.append(
                "prior %s: disposition %r must be one of %s"
                % (prior["slug"], row["disposition"], "|".join(DISPOSITIONS)))
        elif row["disposition"] == "candidate":
            if not row["candidate_id"]:
                errors.append(
                    "prior %s: disposition=candidate needs the candidate_id it "
                    "became" % prior["slug"])
            elif candidate_ids is not None and row["candidate_id"] not in candidate_ids:
                errors.append(
                    "prior %s: candidate_id %r is not in this run's candidates"
                    % (prior["slug"], row["candidate_id"]))
        elif row["disposition"] == "already_engaged":
            evidence = entry.get("engagement_evidence")
            if not isinstance(evidence, dict):
                errors.append(
                    "prior %s: disposition=already_engaged needs structured "
                    "engagement_evidence; a Kernel name somewhere in the "
                    "layer is not proof that the intended seam is engaged"
                    % prior["slug"])
            else:
                scope = str(evidence.get("ownership_scope") or "")
                if scope in ("", "layer", "phase_layer_wrapper"):
                    errors.append(
                        "prior %s: already_engaged ownership_scope %r is too "
                        "broad; provide semantic_region or kernel_exact "
                        "ownership" % (prior["slug"], scope))
                if not evidence.get("kernel"):
                    errors.append(
                        "prior %s: already_engaged engagement_evidence needs "
                        "the engaged kernel" % prior["slug"])
                if evidence.get("removed_kernel_count") is None:
                    errors.append(
                        "prior %s: already_engaged engagement_evidence needs "
                        "removed_kernel_count for the helper(s) the fusion "
                        "claims to eliminate" % prior["slug"])
            if _bad_reason(row["reason"]):
                errors.append(
                    "prior %s: disposition=already_engaged needs a real reason"
                    % prior["slug"])
        elif _bad_reason(row["reason"]):
            errors.append(
                "prior %s: disposition=%s needs a real reason (%r says nothing "
                "-- name the gfx/regime/op that does not match)"
                % (prior["slug"], row["disposition"], row["reason"]))
        rows.append(row)
    return errors, rows


def summarise(rows):
    """Counts for the report/index line."""
    disposable = [r for r in rows if r["disposable"]]
    counts = {d: 0 for d in DISPOSITIONS}
    missing = 0
    for row in disposable:
        if row["disposition"] in counts:
            counts[row["disposition"]] += 1
        else:
            missing += 1
    return {
        "priors_total": len(disposable),
        "advisory_total": len(rows) - len(disposable),
        "carried": counts["candidate"],
        "already_engaged": counts["already_engaged"],
        "not_applicable": counts["not_applicable"],
        "undisposed": missing,
    }


def render_markdown(rows, summary=None):
    """The 已知融合先验 section for the Phase 2.1 report."""
    if not rows:
        return ""
    summary = summary or summarise(rows)
    out = ["## 已知融合先验（knowledge/learned）\n"]
    out.append(
        "这些是**以前测过**的融合。它们是 ADD-only 的提示：只增加候选，"
        "不删除本次 profile 找到的任何候选，也不替代实测。"
        "每条都必须有一个交代——`candidate` 带上它变成了哪条候选，"
        "`already_engaged`/`not_applicable` 带上理由。"
        "**先验被默默漏掉，正是同一份 trace 每次跑出不同融合集的原因。**\n")
    # Only paint the count red when there is something red about it -- a
    # permanent 🔴 in the header trains the reader to stop seeing it.
    out.append("先验 %d 条：已带入 %d / 已在生效 %d / 不适用 %d / %s无交代 %d\n"
               % (summary["priors_total"], summary["carried"],
                  summary["already_engaged"], summary["not_applicable"],
                  "🔴 " if summary["undisposed"] else "", summary["undisposed"]))
    out.append("| 先验卡片 | 置信度 | 本次交代 | 对应候选 / 理由 |")
    out.append("|---|:--:|:--:|---|")
    for row in rows:
        if not row["disposable"]:
            disposition = "参考（method 卡）"
            detail = "读它，不需要交代"
        elif row["disposition"] is None:
            disposition = "🔴 无交代"
            detail = "——"
        else:
            disposition = row["disposition"]
            detail = row["candidate_id"] or row["reason"] or "——"
        out.append("| %s | %s | %s | %s |"
                   % (row["title"] or row["slug"], row["confidence"] or "—",
                      disposition, detail))
    out.append("")
    return "\n".join(out)
