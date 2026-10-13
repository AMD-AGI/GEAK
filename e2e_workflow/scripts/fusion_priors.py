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

import json
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


# --- measured priors for the deterministic screen ----------------------------
# The disposition gate above makes an analyst ACCOUNT for each card. The screen
# (fusion_screen.py) needs something narrower: a machine-readable "this seam
# was measured to win (or lose) X% e2e on this model family / gfx", so its
# second ranking pass is a genuinely different signal from the first (measured
# evidence, not the same Score recomputed), so a measured win cannot silently
# fall out of the final list, and so a measured regression is not re-proposed
# as if it were new. The store is append-only (every measurement is kept) and
# ranking reads the median of the most recent measurements.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PRIORS_INDEX = os.path.abspath(os.path.join(
    _SCRIPT_DIR, "..", "knowledge", "learned", "INDEX.md"))
DEFAULT_MEASURED_PRIORS = os.path.abspath(os.path.join(
    _SCRIPT_DIR, "..", "knowledge", "fusion", "measured_priors.json"))

# "+6.046% e2e", "+1.583% output tok/s". The first signed gain that is
# explicitly an e2e / output-throughput number; kernel-time deltas do not count.
_GAIN_RE = re.compile(r"\+\s*(\d+(?:\.\d+)?)\s*%\s*(?:e2e|output)", re.I)
_FAMILY_RE = re.compile(
    r"(qwen|llama|deepseek|mistral|mixtral|gemma|glm|kimi|minimax|phi|gpt[-_ ]?oss)"
    r"[-_ ]?(\d+(?:[._]\d+)?)?", re.I)
_GFX_RE = re.compile(r"\bgfx\d{3}[a-z]?\b", re.I)
_PAREN_RE = re.compile(r"\(([^()]*)\)")


def parse_gain_pct(*texts):
    """First measured e2e gain (percent) in the given texts, or None."""
    for text in texts:
        match = _GAIN_RE.search(str(text or ""))
        if match:
            return float(match.group(1))
    return None


def model_family(text):
    """Normalised model family ("qwen35" for Qwen3.5 / Qwen3_5, "qwen3" for Qwen3).

    The version digits are part of the family: a Qwen3 dense measurement is
    not evidence for a Qwen3.5 GDN/MoE hybrid.
    """
    match = _FAMILY_RE.search(str(text or ""))
    if not match:
        return ""
    name = re.sub(r"[^a-z]", "", match.group(1).lower())
    version = re.sub(r"[^0-9]", "", match.group(2) or "")
    return name + version


def gfx_targets(text):
    return sorted({m.lower() for m in _GFX_RE.findall(str(text or ""))})


SCHEMA_VERSION = 2
POLARITIES = ("positive", "negative")
EFFECTIVE_WINDOW = 3
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _v1_to_v2(entry):
    """One schema-1 prior (a single measured_e2e_gain_pct) as a schema-2 record."""
    protocol = dict(entry.get("workload") or {})
    for key in ("gpu", "tp"):
        if entry.get(key) is not None:
            protocol.setdefault(key, entry[key])
    date = ""
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(entry.get("source") or ""))
    if match:
        date = match.group(0)
    gain = entry.get("measured_e2e_gain_pct")
    record = {
        "prior_id": str(entry.get("prior_id") or entry.get("title") or ""),
        "title": entry.get("title") or "",
        "model_family": entry.get("model_family") or entry.get("model") or "",
        "model": entry.get("model") or "",
        "gfx": entry.get("gfx") or "",
        "framework": entry.get("framework") or "",
        "polarity": entry.get("polarity") or ("negative" if (gain or 0) < 0 else "positive"),
        "seam": entry.get("seam") or {},
        "measurements": [],
    }
    if gain is not None:
        record["measurements"].append({
            "gain_pct": float(gain), "date": date or "1970-01-01", "protocol": protocol,
            "source": entry.get("source") or "", "evidence": entry.get("evidence")})
    return record


def _read_store(path):
    """The raw prior store as schema 2 ({} skeleton when the file is absent)."""
    try:
        with open(path) as fh:
            data = json.load(fh)
    except OSError:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("%s: prior store must be a JSON object" % path)
    priors = data.get("priors") or []
    if int(data.get("schema_version") or 1) < SCHEMA_VERSION:
        priors = [_v1_to_v2(entry) for entry in priors if isinstance(entry, dict)]
    out = dict(data)
    out["schema_version"] = SCHEMA_VERSION
    out["priors"] = priors
    return out


def _measurements(prior):
    out = []
    for item in (prior or {}).get("measurements") or []:
        if not isinstance(item, dict):
            continue
        try:
            gain = float(item.get("gain_pct"))
        except (TypeError, ValueError):
            continue
        out.append(dict(item, gain_pct=gain))
    if not out and (prior or {}).get("measured_e2e_gain_pct") is not None:
        out.append({"gain_pct": float(prior["measured_e2e_gain_pct"]), "date": "",
                    "protocol": {}, "source": prior.get("source") or "", "evidence": None})
    # Stable by date: same-day measurements keep their append order.
    return sorted(out, key=lambda item: str(item.get("date") or ""))


def effective_gain(prior):
    """Median of the last EFFECTIVE_WINDOW measurements by date, or None.

    One lucky A/B should not decide a ranking forever, and a seam that has been
    re-measured should be judged on its recent behaviour, so the store keeps
    every measurement and ranking reads a short recent median.
    """
    values = [item["gain_pct"] for item in _measurements(prior)][-EFFECTIVE_WINDOW:]
    return _median(values)


def load_measured_priors(path=DEFAULT_MEASURED_PRIORS):
    """knowledge/fusion/measured_priors.json -> normalised prior dicts.

    Reads schema 2 (measurements list, polarity) and schema 1 (a single
    measured_e2e_gain_pct). `measured_e2e_gain_pct` of the result is the
    effective gain (median of the last three measurements by date).
    """
    try:
        data = _read_store(path)
    except (OSError, ValueError):
        return []
    out = []
    for entry in data.get("priors") or []:
        if not isinstance(entry, dict):
            continue
        seam = entry.get("seam") or {}
        gain = effective_gain(entry)
        if gain is None or not seam:
            continue
        measurements = _measurements(entry)
        latest = measurements[-1] if measurements else {}
        polarity = str(entry.get("polarity") or "positive").lower()
        if polarity not in POLARITIES:
            polarity = "positive"
        out.append({
            "prior_id": str(entry.get("prior_id") or entry.get("title")),
            "title": entry.get("title") or "",
            "origin": "measured_priors",
            "source": latest.get("source") or entry.get("source") or "",
            "model": entry.get("model") or "",
            "model_family": model_family(entry.get("model_family") or entry.get("model")),
            "gfx": [str(entry["gfx"]).lower()] if entry.get("gfx") else [],
            "framework": entry.get("framework") or "",
            "polarity": polarity,
            "measured_e2e_gain_pct": float(gain),
            "measurement_count": len(measurements),
            "latest_measurement_date": latest.get("date") or "",
            "phases": list(seam.get("phases") or entry.get("phases") or []),
            "seam": seam,
        })
    return out


def _validate_measurement(measurement):
    if not isinstance(measurement, dict):
        raise ValueError("measurement must be an object")
    try:
        gain = float(measurement.get("gain_pct"))
    except (TypeError, ValueError):
        raise ValueError("measurement.gain_pct must be a number")
    date = str(measurement.get("date") or "")
    if not _DATE_RE.match(date):
        raise ValueError("measurement.date must be YYYY-MM-DD, got %r" % date)
    protocol = measurement.get("protocol")
    if protocol is not None and not isinstance(protocol, dict):
        raise ValueError("measurement.protocol must be an object")
    evidence = measurement.get("evidence")
    if evidence is not None and not isinstance(evidence, str):
        raise ValueError("measurement.evidence must be a string or null")
    out = dict(measurement)
    out.update(gain_pct=gain, date=date, protocol=protocol or {},
               source=str(measurement.get("source") or ""), evidence=evidence)
    return out


def _atomic_write_json(path, data):
    import tempfile
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".measured_priors.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class _StoreLock(object):
    """Advisory lock on a sidecar file so concurrent appenders serialise."""

    def __init__(self, path):
        path = os.path.abspath(path)
        self.path = os.path.join(os.path.dirname(path), "." + os.path.basename(path) + ".lock")
        self.handle = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.handle = open(self.path, "a")
        try:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        return self

    def __exit__(self, *exc):
        try:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        self.handle.close()


_NEW_PRIOR_FIELDS = ("title", "model_family", "model", "gfx", "framework", "seam")


def append_measurement(path, prior_id_or_seam_record, measurement, polarity):
    """Append one measurement to the prior store; returns the updated prior record.

    `prior_id_or_seam_record` is an existing prior_id, or a dict describing a
    seam ({prior_id, title, model_family, model, gfx, framework, seam}); a dict
    whose prior_id is not in the store creates a new prior. The store is
    append-only: measurements are never edited or removed, an identical
    measurement (same date, gain, source, evidence) is not appended twice, and
    the file is rewritten atomically under a lock. A prior's `polarity` is that
    of its most recent disposition; the full history stays in `measurements`.
    """
    polarity = str(polarity or "").lower()
    if polarity not in POLARITIES:
        raise ValueError("polarity must be one of %s" % "|".join(POLARITIES))
    record = _validate_measurement(measurement)
    if isinstance(prior_id_or_seam_record, dict):
        seam_record = dict(prior_id_or_seam_record)
        prior_id = str(seam_record.get("prior_id") or "").strip()
        if not prior_id:
            raise ValueError("a new seam record needs a prior_id")
    else:
        seam_record = None
        prior_id = str(prior_id_or_seam_record or "").strip()
        if not prior_id:
            raise ValueError("prior_id must be non-empty")
    with _StoreLock(path):
        data = _read_store(path)
        data.setdefault("artifact", "fusion_measured_priors")
        priors = data["priors"]
        target = next((p for p in priors if str(p.get("prior_id")) == prior_id), None)
        if target is None:
            if seam_record is None:
                raise KeyError("unknown prior_id %r and no seam record to create it" % prior_id)
            if not (seam_record.get("seam") or {}):
                raise ValueError("new prior %r needs a seam descriptor" % prior_id)
            target = {"prior_id": prior_id}
            for key in _NEW_PRIOR_FIELDS:
                target[key] = seam_record.get(key) if seam_record.get(key) is not None else ""
            target["polarity"] = polarity
            target["measurements"] = []
            priors.append(target)
        target.setdefault("measurements", [])
        key = (record["date"], record["gain_pct"], record["source"], record["evidence"])
        if not any((m.get("date"), float(m.get("gain_pct", 0.0)), m.get("source") or "",
                    m.get("evidence")) == key for m in target["measurements"]):
            target["measurements"].append(record)
        target["polarity"] = polarity
        _atomic_write_json(path, data)
        return dict(target)


def learned_measured_priors(index_path=DEFAULT_PRIORS_INDEX):
    """Fusion lever cards in knowledge/learned that carry a measured e2e gain.

    A card has no structured seam, so its seam is the op-tag set of the fusion
    it names (the parenthetical of its `key`, else its title), tagged with the
    catalog's token vocabulary; the screen matches it by tag overlap.
    """
    try:
        import fusion_catalog
    except Exception:  # noqa: BLE001 - the screen still runs on measured_priors.json
        return []
    out = []
    for card in load_priors(index_path):
        if card["type"] == "method":
            continue
        front = _card_front_matter(card["path"])
        gain = parse_gain_pct(front.get("effect"), card["index_line"])
        if gain is None:
            continue
        key = card.get("key") or ""
        paren = _PAREN_RE.search(key)
        seam_text = paren.group(1) if paren else card["title"]
        if not paren and "·" in seam_text:
            parts = [part.strip() for part in seam_text.split("·")]
            seam_text = parts[1] if len(parts) > 1 else parts[0]
        tags = set(fusion_catalog._op_tags(seam_text)) - {"gemm_prologue", "gemm_epilogue"}
        if len(tags) < 2:
            continue
        # Family / gfx come from what the card is ABOUT (key, slug, title, the
        # index line's leading [context] bracket), not from cautions that
        # mention other models further down the line.
        bracket = re.match(r"^\[([^\]]*)\]", card["index_line"])
        context = " ".join((key, card["slug"].replace("-", " "), card["title"],
                            bracket.group(1) if bracket else ""))
        family = ""
        for text in (key, card["slug"].replace("-", " "), card["title"],
                     bracket.group(1) if bracket else ""):
            family = model_family(text)
            if family:
                break
        out.append({
            "prior_id": "learned:" + card["slug"],
            "title": card["title"],
            "origin": "knowledge/learned",
            "source": card["path"],
            "model": "",
            "model_family": family,
            "gfx": gfx_targets(context),
            "measured_e2e_gain_pct": gain,
            "polarity": "positive",
            "measurement_count": 1,
            "framework": "",
            # Which phase the card measured (its key / context bracket says
            # "decode" or "prefill"); a tag-only seam must match it there.
            "phases": [phase for phase in ("prefill", "decode")
                       if re.search(r"\b%s\b" % phase, " ".join((key, bracket.group(1) if bracket else "")), re.I)],
            "seam": {"tags": sorted(tags)},
        })
    return out
