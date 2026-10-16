"""Shared synthetic fixtures for the selection golden tests (stdlib only).

Every builder takes ``api``, a dict of the functions under test, so the same
fixtures drive both the pre-port code (to generate the goldens) and the port.
``port_api()`` builds the dict for the port. This module defines no tests.

The fixtures model one campaign: a real ``ResearchStore`` journal of campaign
revisions, planner evidence groups with every artifact the final check reads,
and synthetic choices. Files outside the store (the campaign clock and the
planner receipts) live in an in-memory map under fake absolute paths, so the
recommendation outputs do not depend on the temporary directory.
"""

import copy
from datetime import datetime as _datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

NS = 1_000_000_000
JOB = "a" * 64
EPOCH = "epoch-1"
CAMPAIGN = "campaign-1"
BINDING = {"job_id": JOB, "measurement_epoch_id": EPOCH}
EXPECTED = {"campaign_id": CAMPAIGN, "job_id": JOB, "measurement_epoch_id": EPOCH}
START_NS = 1_000 * NS
START_UNIX = 1_700_000_000
WALL = 8 * 3600
DEADLINE = START_UNIX + WALL
LIMITS = {"wall_seconds": WALL, "evaluations": 10}
RESERVED = {"wall_seconds": 600, "evaluations": 0}
NOTE = {"claim": "claim", "scope": "scope", "next_test": "next"}
H4 = START_NS + 4 * 3600 * NS
H8 = START_NS + 8 * 3600 * NS


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(canonical(value))


CLOCK = {"started_monotonic_ns": START_NS, "wall_seconds": WALL, "started_unix": START_UNIX,
         "deadline_ns": START_NS + WALL * NS, "boot_id": "fixture-boot"}
CLOCK_RAW = canonical(CLOCK)
CLOCK_REF = {"path": "/fixture/campaign/clock.json", "sha256": sha(CLOCK_RAW)}
BACKEND_REF = {"path": "/fixture/campaign/backend.json", "sha256": "b" * 64}


def strict_decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs)


def outcome(call):
    """Run ``call``; return its value or ``{"error": class name, "message": text}``."""
    try:
        return {"value": call()}
    except Exception as exc:  # The class name and message are part of the golden.
        return {"error": type(exc).__name__, "message": str(exc)}


def normalize(value, root=None, *, rename_hashes=False, drop=()):
    """Replace ``root`` with <ROOT>; optionally rename 64-hex strings in order of first appearance."""
    names = {}

    def text(item):
        if root is not None:
            item = item.replace(str(root), "<ROOT>")
        if rename_hashes:
            item = re.sub(r"[0-9a-f]{64}", lambda m: names.setdefault(m.group(0), "<H%d>" % len(names)), item)
        return item

    def walk(item):
        if isinstance(item, dict):
            return {text(str(key)): walk(item[key]) for key in sorted(item) if key not in drop}
        if isinstance(item, (list, tuple)):
            return [walk(child) for child in item]
        if isinstance(item, str):
            return text(item)
        if isinstance(item, bytes):
            return {"bytes": text(item.decode("utf-8", "replace"))}
        return item
    return walk(value)


def golden_digest(value):
    return digest(value)


class FixedDatetime(_datetime):
    @classmethod
    def now(cls, tz=None):
        return _datetime(2026, 10, 16, 12, 0, 0, tzinfo=tz or timezone.utc)


class Ticker:
    """Fake monotonic clock: each sample returns ``now`` and advances it by 1 ns."""

    def __init__(self, now):
        self.now = now

    def __call__(self):
        value = self.now
        self.now += 1
        return value


def scratch():
    return Path(tempfile.mkdtemp(prefix="geak-selection-")).resolve()


def remove(root):
    for path, dirs, files in os.walk(root):
        for name in dirs + files:
            item = os.path.join(path, name)
            if not os.path.islink(item):
                os.chmod(item, os.stat(item).st_mode | stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
    shutil.rmtree(root)


def make_choice(candidate_id, source_id, attempt, index):
    choice = {"schema": "geak-native-recommendation-choice-v1", "binding": dict(BINDING, campaign_id=CAMPAIGN),
              "candidate_id": candidate_id, "source_id": source_id, "attempt_id": attempt,
              "history_index": index, "recorded_configuration": {"status": "recorded_prepared_bytes",
                                                                 "attempt": attempt}}
    choice["choice_handle"] = digest(choice)
    return choice


class Campaign:
    """One campaign journal in a real store, built through ``api``."""

    def __init__(self, api, root):
        self.api = api
        self.root = Path(root)
        workspace = self.root / "workspace"
        workspace.mkdir(parents=True)
        api["store_module"].datetime = FixedDatetime
        self.store = api["ResearchStore"](self.root / "store", allowed_epochs=[(JOB, EPOCH)],
                                          source_paths={JOB: ["kernel.py"]})
        sources = []
        for body in (b"def kernel():\n    return 0\n", b"def kernel():\n    return 1\n",
                     b"def kernel():\n    return 2\n"):
            (workspace / "kernel.py").write_bytes(body)
            sources.append(self.store.capture_source(str(workspace), ["kernel.py"]))
        self.sources = sources
        self.candidates = [self.store.candidate(JOB, EPOCH, source) for source in sources]
        self.base = self.candidates[0]
        self.choices = [make_choice(self.candidates[1], sources[1], "attempt-1", 0),
                        make_choice(self.candidates[2], sources[2], "attempt-2", 1)]
        self.external = {CLOCK_REF["path"]: CLOCK_RAW}
        self.revision = 0
        self.sequence = 0
        self.state = {"binding": dict(BINDING), "limits": dict(LIMITS), "deadline": DEADLINE,
                      "phase": "idle", "pending": None, "opportunities": [], "history": [],
                      "recommendation_events": [], "recommendation_head": None}
        self.evidence = []

    # -- callbacks -----------------------------------------------------------

    def read_ref(self, ref):
        path = ref["path"]
        if path in self.external:
            return self.external[path]
        if path.startswith(str(self.store.root) + "/"):
            return Path(path).read_bytes()
        raise FileNotFoundError("fixture has no file " + path)

    def authenticate(self, context, choice):
        known = {canonical(row) for row in self.choices}
        if canonical(choice) not in known:
            raise ValueError("unknown fixture choice")
        return copy.deepcopy(choice)

    def frozen_authenticate(self, frozen, context, choice):
        return self.authenticate(context, choice)

    # -- journal -------------------------------------------------------------

    def commit(self, state, direction=CAMPAIGN):
        note = dict(NOTE, evolution=copy.deepcopy(state))
        self.store.checkpoint(direction, self.base, note=note, expected_revision=self.revision)
        self.revision += 1
        self.state = copy.deepcopy(state)
        return self.revision

    def begin(self):
        return self.commit(self.state)

    def snapshot(self):
        return {"campaign_id": CAMPAIGN, "revision": self.revision, "state": copy.deepcopy(self.state)}

    def plan(self, directive, *, proposal=None, wrapped=True, cost=30.0):
        """Commit one planner turn: a pending revision, then its settled revision."""
        api, index = self.api, self.sequence
        operation = "plan-%d" % (index + 1)
        context = {"campaign_id": CAMPAIGN, "revision": self.revision, "state": copy.deepcopy(self.state)}
        opened = START_UNIX + 10 + 100 * index
        started = START_NS + (opened - START_UNIX + 1) * NS
        row = dict(BINDING, opportunity_id=operation, direction_id=CAMPAIGN, island_id="campaign",
                   engineer_id="planner", question="final pick", opened_at=opened, review_at=opened + 3600,
                   requested=dict(RESERVED), reserved=dict(RESERVED), receipts=[], settled=False,
                   completed_sequence=None, progress_complete=False)
        pending = copy.deepcopy(self.state)
        pending.update(phase="planning", pending={"operation_id": operation, "context": context},
                       opportunities=pending["opportunities"] + [row])
        self.commit(pending)

        if wrapped:
            response = {"proposal": proposal, "recommendation": directive}
            effective = directive
        else:
            response, effective = proposal, {"op": "keep"}
        line = json.dumps(response, sort_keys=True).encode()
        text = b"Planner notes about the campaign.\n" + line + b"\n"
        inner_raw = canonical(proposal) if wrapped else line
        intent = {"campaign_id": CAMPAIGN, "binding": dict(BINDING), "context_sha256": digest(context),
                  "operation_id": operation, "invocation_id": operation,
                  "admission": {"clock_ref": CLOCK_REF, "backend_ref": BACKEND_REF,
                                "model_config": {"workflow_backend": "claude"}},
                  "started_monotonic_ns": started, "deadline_monotonic_ns": started + 600 * NS,
                  "reservation": dict(RESERVED)}
        receipt = {"schema": "geak-standalone-model-receipt-v1", "invocation_id": operation,
                   "clock_ref": CLOCK_REF, "model_backend_ref": BACKEND_REF,
                   "requested_model": "claude-opus-5", "label": "planner", "original_native_timeout_ms": 600000,
                   "settlement": {"settled": True}, "status": "completed", "returncode": 0,
                   "elapsed_seconds": 12.5,
                   "observed": {"identity_valid": True, "reported_models": ["claude-opus-5"],
                                "reported_tools": ["Bash"], "terminal": "turn.completed", "protocol_errors": [],
                                "report_text": text.decode()}}
        receipt_raw = canonical(receipt)
        receipt_ref = {"path": "/fixture/campaign/receipts/%s.json" % operation, "sha256": sha(receipt_raw)}
        self.external[receipt_ref["path"]] = receipt_raw
        record = api["make_record"](context=context, intent=intent, clock_ref=CLOCK_REF, clock=CLOCK,
                                    receipt_ref=receipt_ref, response_raw=text, proposal_raw=inner_raw,
                                    directive=effective, choices=self.choices, accepted_ns=started + 5 * NS)
        costs = {"wall_seconds": cost, "evaluations": 0}
        planner = {"operation_id": operation, "settled": True, "costs": costs, "proposal": proposal,
                   "host_recommendation": record}
        artifacts = {
            "planner.json": canonical(planner), "planner_context.json": canonical(context),
            "context.json": canonical(context), "intent.json": canonical(intent),
            "recommendation-binding.json": canonical(record),
            "recommendation-acceptance.json": canonical({
                "schema": "geak-member-recommendation-acceptance-v1", "operation_id": operation,
                "binding_sha256": digest(record), "durable_before_monotonic_ns": started + 6 * NS}),
            "recommendation-choices.json": canonical(self.choices),
            "request.json": canonical({"schema": "geak-standalone-model-request-v1", "invocation_id": operation,
                                       "model": "claude-opus-5", "label": "planner", "timeout_ms": 600000}),
            "native/receipt.json": receipt_raw,
            "native-result.json": canonical({"schema": "geak-standalone-model-result-v1",
                                             "invocation_id": operation, "receipt_ref": receipt_ref,
                                             "settled": True, "text": text.decode(), "elapsed_seconds": 12.5}),
            "response.txt": text, "raw_proposal.json": inner_raw, "proposal.json": canonical(proposal),
            "completion.json": canonical({"wall_seconds": max(cost, 30.0)})}
        evidence = self.store.archive_evidence(self.base, artifacts)
        self.evidence.append(evidence)

        settled = copy.deepcopy(pending)
        settled_row = settled["opportunities"][-1]
        settled_row.update(settled=True, completed_sequence=index, progress_complete=True,
                           receipts=[{"receipt_id": evidence, "costs": costs}])
        api["commit_record"](settled, record, planner_ref={"evidence_id": evidence, "artifact": "planner.json"},
                             context=context, commit_revision=self.revision + 1, in_budget=True)
        settled.update(pending=None, phase="stopped" if proposal is None else "planned")
        self.sequence += 1
        self.commit(settled)
        return record


def set_directive(choice, reason="lowest native time"):
    return {"op": "set", "choice_handle": choice["choice_handle"], "reason": reason}


CLEAR = {"op": "clear", "reason": "no program beats ORIGINAL"}


def campaign_scenarios(api, root):
    """Name -> built Campaign for each recommendation chain shape."""
    def build(name, steps):
        campaign = Campaign(api, root / name)
        campaign.begin()
        for step in steps:
            step(campaign)
        return campaign

    def pick(index):
        return lambda c: c.plan(set_directive(c.choices[index]), proposal={"action": "stop"})

    def clear(c):
        c.plan(CLEAR, proposal=None)

    def keep(c):
        c.plan(None, proposal={"action": "probe"}, wrapped=False)

    def rejected(c):
        c.plan({"op": "set", "choice_handle": "f" * 64, "reason": "not offered"}, proposal={"action": "stop"})

    def drop_events(c):
        state = copy.deepcopy(c.state)
        state.update(recommendation_events=[], recommendation_head=None)
        c.commit(state)

    def revive_first(c):
        state = copy.deepcopy(c.state)
        state["recommendation_head"] = state["recommendation_events"][0]["record_ref"]
        c.commit(state)

    return {
        "empty": build("empty", []),
        "set": build("set", [pick(0)]),
        "set_set": build("set_set", [pick(0), pick(1)]),
        "set_clear": build("set_clear", [pick(0), clear]),
        "set_keep": build("set_keep", [pick(1), keep]),
        "set_rejected": build("set_rejected", [pick(0), rejected]),
        "clear_only": build("clear_only", [clear]),
        "dropped_events": build("dropped_events", [pick(0), drop_events]),
        "revived_head": build("revived_head", [pick(0), clear, revive_first]),
    }


def validate(api, campaign, snapshot=None, **overrides):
    arguments = dict(expected_binding=EXPECTED, clock_ref=CLOCK_REF, read_ref=campaign.read_ref,
                     authenticate_choice=campaign.authenticate)
    arguments.update(overrides)
    return outcome(lambda: api["validate"](campaign.store, snapshot or campaign.snapshot(), **arguments))


# ---------------------------------------------------------------- recommendation and final outputs

RESPONSES = [
    b'{"proposal": {"a": 1}, "recommendation": {"op": "keep"}}',
    b"null",
    b"[1, 2]",
    b'Some prose.\n{"proposal": null, "recommendation": {"op": "clear", "reason": "r"}}\n  \n',
    b'Intro\n```json\n{"x": 1}\n```\n',
    b'Intro\n```json\n{"x": 1}\n```\ntrailing',
    b'Inline {"x": 1} in prose\n',
    b'See `code` then\n{"x": 2}\n',
    b'Unclosed `code\n{"x": 2}\n',
    b'~~~\n{"x": 1}\n~~~\n',
    b'["a"]\n{"x": 1}\n',
    b'{"x": 1, "x": 2}',
    b'Text\n{"x": NaN}\n',
    b'Text\n{"x": 1} tail\n',
    b'Text\n{"x": [1, 2}\n',
    b"no json at all",
    b"[" * 3000 + b"]" * 3000,
    "café\n{\"k\": \"é\"}\n".encode(),
]

DIRECTIVES = [
    {"op": "keep"}, {"op": "keep", "reason": "x"}, {"op": "clear", "reason": "ok"}, {"op": "clear", "reason": " "},
    {"op": "clear", "reason": "x" * 2001}, {"op": "set", "reason": "fast", "choice_handle": "h1"},
    {"op": "set", "reason": "fast", "choice_handle": "missing"}, {"op": "set", "reason": "fast"},
    {"op": "drop", "reason": "x"}, None, "keep", {"op": "set", "reason": "r", "choice_handle": 7},
]


def recommendation_outputs(api):
    out = {}
    for index, raw in enumerate(RESPONSES):
        out["extract-%02d" % index] = outcome(
            lambda raw=raw: api["extract"](raw, decode=strict_decode).decode("utf-8"))
    out["extract-text"] = outcome(lambda: api["extract"]("text", decode=strict_decode))
    for index, value in enumerate([{"a": 1}, None, {"proposal": 1, "recommendation": {"op": "keep"}},
                                   {"proposal": 1}, {"recommendation": 2, "x": 3}]):
        out["split-%d" % index] = outcome(lambda value=value: list(api["split_response"](value)))
    choices = [{"choice_handle": "h1", "n": 1}, {"choice_handle": "h2", "n": 2}]
    for index, directive in enumerate(DIRECTIVES):
        out["decision-%02d" % index] = outcome(lambda directive=directive: api["decision"](directive, choices))
    out["decision-duplicate"] = outcome(lambda: api["decision"]({"op": "keep"}, choices + choices[:1]))
    return out


def final_outputs(api, root):
    campaigns = campaign_scenarios(api, root)
    out = {}
    for name, campaign in sorted(campaigns.items()):
        out["final-" + name] = validate(api, campaign)
        out["view-" + name] = outcome(lambda c=campaign: api["recommendation_view"](c.store, c.snapshot()))
    chain = campaigns["set_clear"]
    out["final-prefix-revision-3"] = validate(api, chain, snapshot={
        "campaign_id": CAMPAIGN, "revision": 3,
        "state": chain.store.get_revision(CAMPAIGN, 3)["note"]["evolution"]})
    out["final-stale-snapshot"] = validate(api, chain, snapshot={
        "campaign_id": CAMPAIGN, "revision": chain.revision,
        "state": chain.store.get_revision(CAMPAIGN, 3)["note"]["evolution"]})
    out["final-wrong-campaign"] = validate(api, chain, expected_binding=dict(EXPECTED, campaign_id="other"))
    out["final-bad-binding"] = validate(api, chain, expected_binding=dict(BINDING))
    out["final-choice-refused"] = validate(
        api, campaigns["set"], authenticate_choice=lambda context, choice: dict(choice, extra=1))
    out["final-missing-file"] = validate(
        api, campaigns["set"], read_ref=lambda ref: (_ for _ in ()).throw(FileNotFoundError("gone")))
    other_clock = dict(CLOCK, wall_seconds=WALL - 1)
    other_raw = canonical(other_clock)
    other_ref = {"path": "/fixture/campaign/other-clock.json", "sha256": sha(other_raw)}
    campaigns["set"].external[other_ref["path"]] = other_raw
    out["final-other-clock"] = validate(api, campaigns["set"], clock_ref=other_ref)
    return normalize(out, root)


def commit_record_outputs(api, root):
    """commit_record on synthetic states (no store)."""
    campaign = Campaign(api, root / "commit")
    campaign.begin()
    record = campaign.plan(set_directive(campaign.choices[0]), proposal={"action": "stop"})
    out = {"record": record, "state": campaign.state}
    context = {"campaign_id": CAMPAIGN, "revision": 1, "state": campaign.store.get_revision(CAMPAIGN, 1)
               ["note"]["evolution"]}
    pending = campaign.store.get_revision(CAMPAIGN, 2)["note"]["evolution"]
    for name, in_budget in (("in", True), ("out", False)):
        state = copy.deepcopy(pending)
        result = outcome(lambda state=state: api["commit_record"](
            state, record, planner_ref={"evidence_id": "e" * 64, "artifact": "planner.json"},
            context=context, commit_revision=3, in_budget=in_budget))
        out["commit-" + name] = {"result": result, "state": state}
    state = copy.deepcopy(pending)
    state["recommendation_head"] = {"evidence_id": "d" * 64, "artifact": "recommendation-binding.json"}
    out["commit-wrong-head"] = outcome(lambda: api["commit_record"](
        state, record, planner_ref={"evidence_id": "e" * 64, "artifact": "planner.json"},
        context=context, commit_revision=3, in_budget=True))
    late = dict(record, accepted_monotonic_ns=record["planner_deadline_monotonic_ns"])
    out["commit-late"] = outcome(lambda: api["commit_record"](
        copy.deepcopy(pending), late, planner_ref={"evidence_id": "e" * 64, "artifact": "planner.json"},
        context=context, commit_revision=3, in_budget=True))
    out["make-record-outside"] = outcome(lambda: api["make_record"](
        context=context, intent={"campaign_id": CAMPAIGN, "binding": BINDING, "operation_id": "x",
                                 "admission": {"clock_ref": CLOCK_REF}, "started_monotonic_ns": START_NS - 1,
                                 "deadline_monotonic_ns": START_NS + NS},
        clock_ref=CLOCK_REF, clock=CLOCK, receipt_ref=None, response_raw=b"r", proposal_raw=b"p",
        directive={"op": "clear", "reason": "x"}, choices=[], accepted_ns=START_NS))
    out["make-record-float-time"] = outcome(lambda: api["make_record"](
        context=context, intent={}, clock_ref=CLOCK_REF, clock=CLOCK, receipt_ref=None, response_raw=b"r",
        proposal_raw=b"p", directive=None, choices=[], accepted_ns=1.0))
    return normalize(out, root)


# ---------------------------------------------------------------- horizon seals

def _seal_view(result):
    seal, ref = result
    return {"seal": seal, "seal_ref": ref}


def frontier_outputs(api, root):
    out = {}
    out["cutoffs"] = outcome(lambda: api["horizon_cutoffs"](CLOCK, ["H4", "H8", "END"]))
    out["cutoffs-float-wall"] = outcome(lambda: api["horizon_cutoffs"](
        dict(CLOCK, wall_seconds=1.001, deadline_ns=START_NS + 1_001_000_000), ["END"]))
    out["cutoffs-bad-wall"] = outcome(lambda: api["horizon_cutoffs"](dict(CLOCK, wall_seconds=WALL + 1), ["END"]))
    out["cutoffs-duplicate"] = outcome(lambda: api["horizon_cutoffs"](CLOCK, ["H4", "H4"]))
    out["cutoffs-unknown"] = outcome(lambda: api["horizon_cutoffs"](CLOCK, ["H5"]))

    # Main timeline: a pick before H4, a clear after H4, seals at H4 and H8.
    campaign = Campaign(api, root / "frontier-main")
    ticker = Ticker(START_NS + 10)
    frontier = api["register"](campaign.store, ticker, expected_binding=EXPECTED, clock_ref=CLOCK_REF,
                               clock_raw=CLOCK_RAW, horizons=("H4", "H8"))
    campaign.begin()
    campaign.plan(set_directive(campaign.choices[0]), proposal={"action": "stop"})
    ticker.now = H4 - 100
    out["main-seal-early"] = outcome(lambda: _seal_view(frontier.seal_horizon(
        campaign.store, "H4", read_ref=campaign.read_ref, authenticate_choice=campaign.frozen_authenticate)))
    ticker.now = H4 + 10
    campaign.plan(CLEAR, proposal=None)
    sealed = outcome(lambda: frontier.seal_horizon(
        campaign.store, "H4", read_ref=campaign.read_ref, authenticate_choice=campaign.frozen_authenticate))
    out["main-seal-h4"] = {"value": _seal_view(sealed["value"])} if "value" in sealed else sealed
    out["main-reseal-h4"] = outcome(lambda: _seal_view(frontier.seal_horizon(
        campaign.store, "H4", read_ref=campaign.read_ref, authenticate_choice=campaign.frozen_authenticate)))
    if "value" in sealed:
        seal_ref = sealed["value"][1]
        out["main-recheck-h4"] = outcome(lambda: frontier.authenticate_existing_horizon(
            campaign.store, "H4", seal_ref, read_ref=campaign.read_ref,
            authenticate_choice=campaign.frozen_authenticate)[2:])
    ticker.now = H8 + 10
    out["main-seal-h8"] = outcome(lambda: _seal_view(frontier.seal_horizon(
        campaign.store, "H8", read_ref=campaign.read_ref, authenticate_choice=campaign.frozen_authenticate)))
    out["main-files"] = sorted(path.name for path in (campaign.store.root / "commit-frontier" / "seals").iterdir())

    # A commit whose interval crosses H4 leaves the horizon unresolved.
    straddle = Campaign(api, root / "frontier-straddle")
    ticker = Ticker(START_NS + 10)
    frontier = api["register"](straddle.store, ticker, expected_binding=EXPECTED, clock_ref=CLOCK_REF,
                               clock_raw=CLOCK_RAW, horizons=("H4",))
    straddle.begin()
    ticker.now = H4 - 1
    straddle.commit(straddle.state)
    out["straddle-seal-h4"] = outcome(lambda: _seal_view(frontier.seal_horizon(
        straddle.store, "H4", read_ref=straddle.read_ref, authenticate_choice=straddle.frozen_authenticate)))

    # No campaign revision before H4: missing with its reason, never a substitute.
    empty = Campaign(api, root / "frontier-empty")
    ticker = Ticker(START_NS + 10)
    frontier = api["register"](empty.store, ticker, expected_binding=EXPECTED, clock_ref=CLOCK_REF,
                               clock_raw=CLOCK_RAW, horizons=("H4", "H8"))
    empty.store.checkpoint("other-direction", empty.base, note=dict(NOTE), expected_revision=0)
    ticker.now = H4 + 5
    empty.begin()
    sealed = outcome(lambda: frontier.seal_horizon(
        empty.store, "H4", read_ref=empty.read_ref, authenticate_choice=empty.frozen_authenticate))
    out["empty-seal-h4"] = {"value": _seal_view(sealed["value"])} if "value" in sealed else sealed
    if "value" in sealed:
        out["empty-recheck-h4"] = outcome(lambda: frontier.authenticate_existing_horizon(
            empty.store, "H4", sealed["value"][1], read_ref=empty.read_ref,
            authenticate_choice=empty.frozen_authenticate)[2:])

    # A refused choice witness is unresolved, not a fallback.
    refused = Campaign(api, root / "frontier-refused")
    ticker = Ticker(START_NS + 10)
    frontier = api["register"](refused.store, ticker, expected_binding=EXPECTED, clock_ref=CLOCK_REF,
                               clock_raw=CLOCK_RAW, horizons=("H4",))
    refused.begin()
    refused.plan(set_directive(refused.choices[1]), proposal={"action": "stop"})
    ticker.now = H4
    out["refused-seal-h4"] = outcome(lambda: _seal_view(frontier.seal_horizon(
        refused.store, "H4", read_ref=refused.read_ref,
        authenticate_choice=lambda frozen, context, choice: dict(choice, changed=True))))
    out["register-late"] = outcome(lambda: api["register"](
        Campaign(api, root / "frontier-late").store, Ticker(H4), expected_binding=EXPECTED,
        clock_ref=CLOCK_REF, clock_raw=CLOCK_RAW, horizons=("H4",)).contract)
    return normalize(out, root, rename_hashes=True)


def frontier_files(api, root):
    """Run the frontier scenarios under ``root``; return {relative path: sha256} of every file written."""
    frontier_outputs(api, root)
    return {str(path.relative_to(root)): sha(path.read_bytes())
            for path in sorted(root.rglob("*")) if path.is_file()}


# ---------------------------------------------------------------- choices

class EpisodeFixture:
    """Settled native episodes in a campaign history, with a fake episode reader."""

    def __init__(self, api, root, variants):
        self.campaign = campaign = Campaign(api, root)
        host = campaign.root / "host"
        host.mkdir()
        self.episodes = {}
        history = []
        for index, variant in enumerate(variants):
            attempt = "attempt-%d" % index
            candidate = campaign.candidates[1 + index % 2]
            source = campaign.sources[1 + index % 2]
            refs = {}
            for name in ("entry_runner", "entry_input", "collection_config", "collector", "entry_provenance",
                         "driver", "clock", "request"):
                path = host / ("%s-%d.json" % (name, index))
                body = canonical({"name": name, "attempt": attempt})
                path.write_bytes(body)
                refs[name] = {"path": str(path), "sha256": sha(body)}
            config_body = canonical({"entry_input_ref": refs["entry_input"], "backend_config": {
                "probe_config": {"turn_id": attempt, "job_id": JOB, "measurement_epoch_id": EPOCH},
                "direction_id": "direction-1", "clock_ref": refs["clock"]}})
            (host / ("config-%d.json" % index)).write_bytes(config_body)
            config_ref = {"path": str(host / ("config-%d.json" % index)), "sha256": sha(config_body)}
            backend_body = canonical({"driver_ref": refs["driver"], "config_ref": config_ref})
            (host / ("backend-%d.json" % index)).write_bytes(backend_body)
            prepared = {"entry_runner_ref": refs["entry_runner"], "entry_input_ref": refs["entry_input"],
                        "backend_binding_ref": {"path": str(host / ("backend-%d.json" % index)),
                                                "sha256": sha(backend_body)},
                        "collection_config_ref": refs["collection_config"], "collector_ref": refs["collector"],
                        "entry_provenance_ref": refs["entry_provenance"]}
            evidence = campaign.store.archive_evidence(candidate, {
                "packet.json": canonical({"attempt": attempt}), "native_transport.json": b"{}",
                "collection.json": b"{}", "raw_manifest.json": b"{}", "feedback.json": b"{}",
                "native_prepared.json": canonical(prepared)})

            def ref(name, evidence=evidence):
                return {"evidence_id": evidence, "artifact": name}
            row = {"attempt_id": attempt, "settled": True, "candidate_id": candidate,
                   "packet_ref": ref("packet.json"), "transport_ref": ref("native_transport.json"),
                   "collection_ref": ref("collection.json")}
            history.append(row)
            times = {"case-a": 1.0 + index, "case-b": 2.0 + index}
            trial = {"verified": {"source_id": source, "per_case_times_ms": times, "turn_id": attempt,
                                  "operation_id": "op-%d" % index},
                     "error": None, "request": {"request_ref": refs["request"]}, "feedback_ref": ref("feedback.json")}
            selected = {"source_id": source, "correctness": "native_pass_observed",
                        "feedback_refs": [trial["feedback_ref"]], "timings": [times],
                        "verified_turn_ids": [attempt]}
            material = {"host_provenance": {"packet_ref": row["packet_ref"], "transport_ref": row["transport_ref"],
                                            "collection_ref": row["collection_ref"],
                                            "raw_manifest_ref": ref("raw_manifest.json"),
                                            "runtime_refs": {key: prepared[key] for key in (
                                                "entry_runner_ref", "collector_ref", "entry_provenance_ref")},
                                            "candidate_roles": {"selected": candidate}},
                        "harvest": {"status": "best_correct_retained", "selected": selected,
                                    "binding": dict(BINDING, turn_id=attempt)},
                        "explanation_check": {"binding_valid": True}, "native_entry_check": {"binding_valid": True},
                        "native_trials": [trial]}
            outer = {"attempt_id": attempt, "settlement": {"settled": True}, "process_reaped": True,
                     "request": {"job": {"job_id": JOB}, "measurement_epoch_id": EPOCH,
                                 "direction_id": "direction-1"}}
            self.episodes[attempt] = _variant(variant, outer, material)
        history.append({"note": "a planner row without an attempt"})
        state = copy.deepcopy(campaign.state)
        state["history"] = history
        campaign.begin()
        campaign.commit(state)

    def episode(self, snapshot, row):
        result = self.episodes[row["attempt_id"]]
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)


def _variant(kind, outer, material):
    """Apply one ineligibility to an otherwise valid episode."""
    selected = material["harvest"]["selected"]
    trial = material["native_trials"][0]
    if kind == "valid":
        pass
    elif kind == "partial_coverage":  # The witness covers fewer trials than were checked.
        material["native_trials"].append(copy.deepcopy(trial))
        material["native_trials"][1]["verified"]["operation_id"] = "op-extra"
    elif kind == "nonzero_exit":
        trial["error"] = {"returncode": 1}
    elif kind == "id_mismatch":
        material["harvest"]["binding"]["turn_id"] = "attempt-other"
    elif kind == "source_mismatch":
        selected["source_id"] = "c" * 64
    elif kind == "not_selected":
        material["harvest"]["selected"] = None
        del material["host_provenance"]["candidate_roles"]["selected"]
    elif kind == "episode_refused":
        return ValueError("episode check refused the settlement")
    elif kind == "episode_oserror":
        return OSError("episode store unreadable")
    else:
        raise AssertionError(kind)
    return outer, material


def choices_outputs(api, root):
    out = {}
    good = EpisodeFixture(api, root / "choices-good", ["valid", "not_selected", "valid"])
    snapshot = good.campaign.snapshot()
    listed = outcome(lambda: api["eligible_choices"](good.campaign.store, snapshot, good.episode))
    out["eligible"] = listed
    if "value" in listed:
        first = listed["value"][0]
        out["authenticate"] = outcome(lambda: api["authenticate_choice"](
            good.campaign.store, snapshot, first, good.episode))
        out["authenticate-tampered"] = outcome(lambda: api["authenticate_choice"](
            good.campaign.store, snapshot, dict(first, attempt_id="attempt-9"), good.episode))
        rehandled = dict(first, history_index=2)
        rehandled["choice_handle"] = digest({k: v for k, v in rehandled.items() if k != "choice_handle"})
        out["authenticate-other-index"] = outcome(lambda: api["authenticate_choice"](
            good.campaign.store, snapshot, rehandled, good.episode))
        out["references"] = outcome(lambda: api["choice_references"](first))
        directory = good.campaign.root / "checkpoints"
        out["checkpoint"] = outcome(lambda: api["checkpoint_choices"](
            good.campaign.store, snapshot, str(directory), good.episode))
        out["checkpoint-again"] = outcome(lambda: api["checkpoint_choices"](
            good.campaign.store, snapshot, str(directory), good.episode))
        (directory / ".pending-1").write_bytes(b"partial")
        (directory / "garbage.json").write_bytes(b"{")
        out["reauthenticate"] = outcome(lambda: api["reauthenticate_checkpoints"](
            good.campaign.store, snapshot, str(directory), good.episode))
    out["stale-snapshot"] = outcome(lambda: api["eligible_choices"](
        good.campaign.store, dict(snapshot, revision=1), good.episode))
    mixed = EpisodeFixture(api, root / "choices-mixed",
                           ["valid", "partial_coverage", "nonzero_exit", "id_mismatch", "source_mismatch",
                            "episode_refused", "not_selected"])
    snapshot = mixed.campaign.snapshot()
    out["mixed-eligible"] = outcome(lambda: api["eligible_choices"](mixed.campaign.store, snapshot, mixed.episode))
    out["mixed-partition"] = outcome(lambda: api["partition_choices"](mixed.campaign.store, snapshot, mixed.episode))
    fatal = EpisodeFixture(api, root / "choices-fatal", ["valid", "episode_oserror"])
    out["fatal-partition"] = outcome(lambda: api["partition_choices"](
        fatal.campaign.store, fatal.campaign.snapshot(), fatal.episode))
    references = out.pop("references", None)
    out = normalize(out, root, rename_hashes=True)
    if references is not None:
        out["references"] = _order_free(references, root)
    return out


def _order_free(result, root):
    """The reference list is sorted by object path, and object names depend on the
    scratch root, so compare it as a set plus a flag that it was path-sorted."""
    if "value" not in result:
        return normalize(result, root)
    paths = [ref["path"] for ref in result["value"]]
    refs = normalize(result["value"], root, rename_hashes=True)
    return {"sorted_by_path": paths == sorted(paths), "refs": sorted(refs, key=canonical)}


# ---------------------------------------------------------------- population selection

CASES = ["case-a", "case-b"]


def _population_contract(api, root, members):
    contract = {"schema": "geak-population-native-selector-contract-v1", "population_id": "population-1",
                "operator": "operator-1", "job_id": JOB, "horizon": "H4",
                "selection_seal_path": str(root / "selection.json"), "benchmark_case_ids": list(CASES),
                "benchmark_report_ids": {"case-a": "report-a", "case-b": "report-b"},
                "correctness_case_ids": CASES + ["case-c"],
                "benchmark_contract_ref": {"path": "/fixture/benchmark.json", "sha256": "1" * 64},
                "native_contract_sha256": "2" * 64,
                "evaluator_ref": {"path": "/fixture/evaluator.py", "sha256": "3" * 64},
                "timing_policy": {"warmup": 2}, "phases": ["correctness", "benchmark"],
                "runtime_image": "image-1", "runtime_config_digest": "config-1",
                "device_policy": {"kind": "registered_homogeneous_devices_selection_heuristic",
                                  "physical_devices": ["gpu0", "gpu1", "gpu2"],
                                  "homogeneity_ref": {"path": "/fixture/devices.json", "sha256": "4" * 64}},
                "members": members, "implementation_ref": api["implementation_ref"](),
                "known_missing_policy": "exclude_authenticated_clear_or_no_set",
                "unresolved_policy": "unresolved_population"}
    raw = canonical(contract)
    return contract, raw, {"path": str(root / "contract.json"), "sha256": sha(raw)}


def _member(index):
    return {"member_id": "m%d" % index, "campaign_id": "campaign-%d" % index, "measurement_epoch_id": EPOCH,
            "horizon_contract_ref": {"path": "/fixture/m%d/CONTRACT.json" % index, "sha256": ("%d" % index) * 64},
            "physical_devices": ["gpu%d" % (index % 3)]}


def _selected_seal(member, source_id, times_list, operations=None):
    operations = operations or ["op-%d" % i for i in range(len(times_list))]
    trials = [{"verified": {"source_id": source_id, "per_case_times_ms": times, "operation_id": operation,
                            "turn_id": "t", "measurement_valid": True},
               "error": None, "request": {"request_ref": {"path": "/fixture/r.json", "sha256": "5" * 64}},
               "feedback_ref": {"evidence_id": "6" * 64, "artifact": "f.json"}}
              for times, operation in zip(times_list, operations)]
    choice = {"binding": {"campaign_id": member["campaign_id"], "job_id": JOB,
                          "measurement_epoch_id": member["measurement_epoch_id"]},
              "source_id": source_id, "candidate_id": "7" * 63 + member["member_id"][-1],
              "source_manifest_ref": {"path": "/fixture/objects/" + source_id, "sha256": source_id},
              "source_body_refs": [{"source_path": "kernel.py", "executable": 0,
                                    "body_ref": {"path": "/fixture/objects/body", "sha256": "8" * 64},
                                    "availability": "recorded"},
                                   {"source_path": "notes.txt", "executable": 0, "body_ref": None,
                                    "availability": "absent_in_source_manifest"}],
              "selected_witness": {"role": "selected", "native_trials": trials,
                                   "harvest_selected": {"timings": [t["verified"]["per_case_times_ms"]
                                                                    for t in trials]}}}
    seal = {"schema": "geak-committed-horizon-endpoint-v1", "contract_ref": member["horizon_contract_ref"],
            "horizon": "H4", "status": "selected", "snapshot": {"campaign_id": member["campaign_id"]},
            "recommendation": {"status": "selected", "choice": choice, "record": {"disposition": "set"},
                               "head": {"evidence_id": "9" * 64, "artifact": "recommendation-binding.json"}}}
    return seal, trials


def _missing_seal(member, kind):
    seal = {"schema": "geak-committed-horizon-endpoint-v1", "contract_ref": member["horizon_contract_ref"],
            "horizon": "H4", "status": "missing"}
    if kind == "clear":
        seal.update(snapshot={"campaign_id": member["campaign_id"]}, recommendation={
            "status": "missing", "choice": None, "head": {"evidence_id": "9" * 64,
                                                         "artifact": "recommendation-binding.json"},
            "record": {"disposition": "clear"}})
    elif kind == "no_set":
        seal.update(snapshot={"campaign_id": member["campaign_id"]}, recommendation={
            "status": "missing", "choice": None, "head": None, "record": None})
    else:
        seal.update(recommendation=None, missing_reason="no_campaign_revision_before_horizon")
    return seal, []


def _population_run(api, root, name, plans):
    """plans: member index -> (seal, trials) or an Exception to raise from the callback."""
    directory = root / name
    directory.mkdir()
    members = [_member(index) for index in sorted(plans)]
    contract, raw, contract_ref = _population_contract(api, directory, members)
    material, refs = {}, {}
    for index, plan in plans.items():
        member = _member(index)
        if isinstance(plan, Exception):
            seal_raw = canonical({"placeholder": index})
            material[member["member_id"]] = plan
        else:
            seal, trials = plan
            seal_raw = canonical(seal)
            comparison = api["expected_comparison"](contract, member)
            material[member["member_id"]] = {"horizon_bytes": seal_raw, "trials": [
                {"native_trial": trial, "comparison": comparison} for trial in trials]}
        refs[member["member_id"]] = {"path": "/fixture/%s/H4.json" % member["member_id"], "sha256": sha(seal_raw)}

    def authenticate(ref, member, contract):
        result = material[member["member_id"]]
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)
    return outcome(lambda: api["select_and_seal"](contract_ref, raw, refs, contract["selection_seal_path"],
                                                  authenticate_member=authenticate))


def population_outputs(api, root):
    a, b, c = "a1" * 32, "b2" * 32, "c3" * 32
    m = {i: _member(i) for i in range(1, 5)}
    runs = {
        "selected": {1: _selected_seal(m[1], a, [{"case-a": 2.0, "case-b": 2.0}, {"case-a": 4.0, "case-b": 1.0}]),
                     2: _selected_seal(m[2], b, [{"case-a": 1.0, "case-b": 2.0}]),
                     3: _missing_seal(m[3], "clear")},
        "tie": {1: _selected_seal(m[1], b, [{"case-a": 1.0, "case-b": 1.0}]),
                2: _selected_seal(m[2], a, [{"case-a": 1.0, "case-b": 1.0}])},
        "duplicates": {1: _selected_seal(m[1], c, [{"case-a": 3.0, "case-b": 3.0}] * 2, ["op-x", "op-x"]),
                       2: _missing_seal(m[2], "no_set")},
        "all-missing": {1: _missing_seal(m[1], "clear"), 2: _missing_seal(m[2], "no_campaign"),
                        3: _missing_seal(m[3], "no_set")},
        "one-unresolved": {1: _selected_seal(m[1], a, [{"case-a": 1.0, "case-b": 1.0}]),
                           2: ({"schema": "geak-committed-horizon-endpoint-v1",
                                "contract_ref": m[2]["horizon_contract_ref"], "horizon": "H4",
                                "status": "unresolved", "recommendation": None}, [])},
        "callback-error": {1: _selected_seal(m[1], a, [{"case-a": 1.0, "case-b": 1.0}]),
                           2: OSError("seal unreadable")},
    }
    subset_seal, subset_trials = _selected_seal(m[2], b, [{"case-a": 1.0, "case-b": 1.0},
                                                          {"case-a": 2.0, "case-b": 2.0}])
    runs["trial-subset"] = {1: _selected_seal(m[1], a, [{"case-a": 5.0, "case-b": 5.0}]),
                            2: (subset_seal, subset_trials[:1])}
    out = {name: _population_run(api, root, name, plans) for name, plans in sorted(runs.items())}
    out["one-member"] = _population_run(api, root, "one-member", {1: runs["tie"][1]})
    return normalize(out, root, rename_hashes=True,
                     drop=("created_utc", "implementation_ref", "metric_source_ref"))


def population_files(api, root):
    """Run the population scenarios under ``root`` with a fixed clock; return
    {relative path: seal text} with the three values that name a different file
    (implementation_ref, metric_source_ref, selector_contract_ref sha256)
    replaced by placeholders, byte for byte. Everything else stays exact."""
    module = api["population_module"]
    module.datetime = type("FixedDatetimeModule", (), {"datetime": FixedDatetime, "timezone": timezone})
    population_outputs(api, root)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            seal = json.loads(raw)
            for value, name in ((canonical(seal["implementation_ref"]), b"<IMPLEMENTATION>"),
                                (canonical(seal["metric_source_ref"]), b"<METRIC_SOURCE>"),
                                (seal["selector_contract_ref"]["sha256"].encode(), b"<CONTRACT>")):
                raw = raw.replace(value, name)
            files[str(path.relative_to(root))] = raw.decode()
    return files


# ---------------------------------------------------------------- the port's api

def port_api():
    from geak_evo.archive import store
    from geak_evo.selection import choices, final, frontier, population, recommendation

    def register(research_store, ticker, **kwargs):
        return frontier.CommitFrontier.register(research_store, clock=lambda: (CLOCK["boot_id"], ticker()),
                                                **kwargs)
    return {"extract": recommendation.extract_structured_response, "split_response": recommendation.split_response,
            "decision": recommendation.decision, "make_record": recommendation.make_record,
            "commit_record": recommendation.commit_record,
            "recommendation_view": recommendation.recommendation_view,
            "validate": final.validate_final_recommendation, "ResearchStore": store.ResearchStore,
            "store_module": store, "horizon_cutoffs": frontier.horizon_cutoffs, "register": register,
            "eligible_choices": choices.eligible_choices, "authenticate_choice": choices.authenticate_choice,
            "partition_choices": choices.partition_choices, "choice_references": choices.choice_references,
            "checkpoint_choices": choices.checkpoint_choices,
            "reauthenticate_checkpoints": choices.reauthenticate_checkpoints,
            "select_and_seal": population.select_and_seal, "implementation_ref": population.implementation_ref,
            "expected_comparison": population.expected_comparison, "population_module": population}


def original_api():
    """The same dict for the pre-port modules (import only with their own PYTHONPATH)."""
    import structured_response
    import native_recommendation
    import population_selector
    from geak_evo import committed_frontier, member_recommendation, recommendation_validation, research_state

    def register(research_store, ticker, **kwargs):
        committed_frontier.sample_clock = lambda: (CLOCK["boot_id"], ticker())
        return committed_frontier.CommitFrontier.register(research_store, **kwargs)
    return {"extract": structured_response.extract_structured_response,
            "split_response": member_recommendation.split_response, "decision": member_recommendation.decision,
            "make_record": member_recommendation.make_record, "commit_record": member_recommendation.commit_record,
            "recommendation_view": member_recommendation.recommendation_view,
            "validate": recommendation_validation.validate_final_recommendation,
            "ResearchStore": research_state.ResearchStore, "store_module": research_state,
            "horizon_cutoffs": committed_frontier.horizon_cutoffs, "register": register,
            "eligible_choices": native_recommendation.eligible_choices,
            "authenticate_choice": native_recommendation.authenticate_choice,
            "partition_choices": native_recommendation.partition_choices,
            "choice_references": native_recommendation.choice_references,
            "checkpoint_choices": native_recommendation.checkpoint_choices,
            "reauthenticate_checkpoints": native_recommendation.reauthenticate_checkpoints,
            "select_and_seal": population_selector.select_and_seal,
            "implementation_ref": population_selector.implementation_ref,
            "expected_comparison": population_selector.expected_comparison,
            "population_module": population_selector}


class PortScratch:
    """Mix into a TestCase: ``self.api`` is the port, ``self.root`` a scratch dir."""

    def setUp(self):
        self.api = port_api()
        self.saved_datetime = self.api["store_module"].datetime
        self.root = scratch()

    def tearDown(self):
        self.api["store_module"].datetime = self.saved_datetime
        remove(self.root)

    def frontier_campaign(self, name, horizons=("H4", "H8")):
        """A new campaign with a frontier registered on a fake clock; returns (campaign, ticker, frontier)."""
        campaign = Campaign(self.api, self.root / name)
        ticker = Ticker(START_NS + 10)
        frontier = self.api["register"](campaign.store, ticker, expected_binding=EXPECTED, clock_ref=CLOCK_REF,
                                        clock_raw=CLOCK_RAW, horizons=horizons)
        campaign.begin()
        return campaign, ticker, frontier

    def seal(self, campaign, frontier, horizon):
        seal, _ = frontier.seal_horizon(campaign.store, horizon, read_ref=campaign.read_ref,
                                        authenticate_choice=campaign.frozen_authenticate)
        return seal


SUITES = {"recommendation": lambda api, root: recommendation_outputs(api),
          "commit_record": commit_record_outputs, "final": final_outputs, "frontier": frontier_outputs,
          "choices": choices_outputs, "population": population_outputs}


def suite_outputs(api, suite):
    module = api["store_module"]
    saved, root = module.datetime, scratch()
    try:
        return SUITES[suite](api, root)
    finally:
        module.datetime = saved
        remove(root)
