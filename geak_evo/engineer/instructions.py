"""Render native tool instructions from the exact admitted operation values.

The producer and root publisher share this text recipe. It only formats supplied
values; callers retain their own file, source, scope and descriptor validation.
"""

from pathlib import Path
import shlex

from .._util import require
from ..archive.policy import native_capability_context


def tool_addendum(*, config, admission, request, operation, turn_deadline_ns,
                  native_only, search_policy, common_tool_addendum):
    """Return deterministic instructions without reading files or launching work."""
    probe = config["backend_config"]["probe_config"]
    client = Path(config["imports"]["host_root"]).parent / "client/client.py"
    context = str(operation / "replies" / "client-context.json")
    command = ["python3", "-I", "-B", str(client), "--context", context]
    command += ["--profile", "native", "--request-id", "YOUR_UNIQUE_REQUEST_ID",
                "--timeout-seconds", str(probe["native_envelope_seconds"])]
    # Do not inherit a prior arm's inline alternative index, old source labels,
    # operation IDs, clock claims or command examples through its template.
    text = (
        "Read the exact outer assignment and READ_INDEX.json from INSIGHTS, the idea contract, "
        "the supplied operator brief, task/tool definitions and corrections before experiments. "
        "PARENTS.json retains only host-admitted source/history packets. Raw explanations remain hypotheses.\n"
        "READ_INDEX maps original exact file refs to readable copies. native_evidence_index.json maps original "
        "native report/artifact paths to evidence_id/artifact pairs; the copied evidence envelope maps those "
        "pairs to blob hashes, which READ_INDEX resolves. To open a source, find its source_id in "
        "READ_INDEX[].original_ref.sha256 and open that row's readable_ref, checking the bytes' hash. The manifest's own hash is "
        "its identity; do not search for a source_id field inside it. Resolve non-null files[].blob_id "
        "values through the same index; null records an absent file. If a required hash is absent, report that "
        "the complete source was not delivered in this assignment; do not reconstruct a partial program "
        "or infer that the program never existed.\n"
        "Starting and fixed patch-base source: " + request["tip"]["source_id"] + ".\n"
        "The unchanged original campaign clock bounds this opportunity. Native turn deadline (boot monotonic ns): "
        + str(turn_deadline_ns) + ". The host refuses a native operation that cannot fit its complete "
        + str(probe["native_envelope_seconds"]) + " second envelope. The defensive per-turn request ceiling is "
        + str(probe["max_calls_per_turn"]) + "; it is not a requested call count. All elapsed time and cleanup count.\n"
        "Publish one source at a time with a short foreground Bash submit command and an explicit unique "
        "request ID. Preserve the returned JSON publication receipt: phase=submitted and "
        "publication_durable=true establish durable publication, not host admission or evaluation. The receipt "
        "binds source/session/turn/request/profile identities and request_ref. Retain its canonical "
        "request_sha256 for status; request_ref.sha256 instead hashes the request file bytes.\n"
        "The host supplies routing and source paths through its read-only context file; do not repeat "
        "session/turn IDs or add routing overrides.\n"
        "Native compile/correctness/benchmark submit:\nPYTHONDONTWRITEBYTECODE=1 "
        + shlex.join(command[:-2] + ["--action", "submit"]) + "\n"
        "Then inspect that request with one-shot status, replacing the request ID and canonical hash below "
        "with the publication receipt's exact values:\nPYTHONDONTWRITEBYTECODE=1 "
        + shlex.join(["python3", "-I", "-B", str(client), "--action", "status",
            "--context", context,
            "--request-id", "YOUR_UNIQUE_REQUEST_ID", "--request-sha256",
            "CANONICAL_HASH_FROM_PUBLICATION_RECEIPT"]) + "\n"
        "Normal status calls after submit must include the receipt's --request-sha256. Omit that assertion "
        "only to inspect an ID never submitted. Status reads the original request and checked host evidence; "
        "it does not publish, capture current source or spend a native budget. Exit zero means an observation "
        "was read, not that evaluation succeeded. Inspect the JSON fields. submitted means publication only; "
        "queued means host reservation was saved; running is a worker-entry observation, not proof of "
        "current liveness or GPU occupancy. These records are observations, not heartbeats.\n"
        "Submit binds a source ID but does not freeze the workspace. Keep submitted source unchanged until "
        "status returns phase=terminal and settled=true with matching request/source/session/turn/profile "
        "feedback. Read that feedback's error, correctness and measurement results; positive settlement "
        "alone does not establish a successful evaluation. Terminal admission failure has settled=null; "
        "terminal unsettled feedback has settled=false. Neither permits another attempt. If publication or "
        "completion is uncertain, inspect the same ID; never resubmit it or generate another ID to recover "
        "the uncertain operation. A lost receipt or unpinned not_submitted observation does not authorize "
        "retry. Host reconciliation remains the existing host responsibility.\n"
    )
    if not native_only:
        text += (
            "For exact-source compiler/IR/ISA/profiler evidence, after the prior request positively settles, "
            "use the submit command above with a new unique request ID and --profile "
            + config["hardware_profile"] + " and use its own receipt for status. Admitted diagnostic profiles: "
            + ", ".join(admission["diagnostic"]["profiles"]) + ". Diagnostics are not timing-score replacements. "
            "Keep actual source, case, method, units and runtime bindings. Physical HBM traffic, sustainable "
            "bandwidth, achieved residency and overhead remain unknown without applicable evidence.\n"
        )
    else:
        text += (
            "Admitted evaluation profile: native. Read this operation's settled feedback artifact_index for exact "
            "retained paths, sizes, and SHA-256 values, including emitted compiler streams and any eligible output "
            "artifacts. Compiler caches and task-tree contents are not collected by this native route; complete "
            "compiler IR, ISA, executed-codeobject capture, and profiler diagnostics are not established by this capability.\n")
    text += (
        "Preserve best_patch.diff, unfinished_patch.diff when developing work differs, report.md and idea_selection.json; return one final engineer JSON object for host persistence; save slower faithful "
        "sources separately from selected code and the developing tip. Existing measurement/custody/reward "
        "flags are unchanged. No endpoint or evolutionary efficacy follows from development timing.\n")
    if not native_only:
        text += "\n\nHardware result access and coverage\n\nThe settled native feedback contains an artifact_index with exact paths, byte\ncounts and SHA-256 values. Follow the returned references inside this operation's\nreadable reply directory. Small records are inline. A larger raw archive can be\nlisted with transport=owned_request_file; that is an ordinary immutable tar/gzip\nfile published into the same reply directory. It is not missing evidence and does\nnot require another profiling request. The existing host and collection reader\nverify its size/hash and preserve the original member manifest.\n\nInspect raw/hardware/artifact-index.json first. It identifies the archive and\nits member manifest, or the separate compressed-file records. The member manifest\nrecords the original member names, sizes and hashes.\nUse the exact returned archive path and member name to read a selected member,\nfor example with tar -tzf ARCHIVE or tar -xOzf ARCHIVE EXACT_MEMBER. Quote paths,\ndo not execute artifact contents, and do not extract unvalidated paths over the\nworkspace. Verify the selected member against its recorded original size/hash\nbefore using its content. The complete archive hash applies to compressed bytes;\nthe original member hash applies to that member's decoded bytes. Successful file\npublication and positive cleanup do not turn a failed diagnostic into a pass.\n\nFor the corrected Compute result, inspect faithful_summary and the referenced\nraw call/code evidence. selected_call_isa_evidence_available means every selected\ncall's joined callback has validated entry assembly. selected_call_isa_complete\nremains false: this collector does not establish gap-free function reconstruction.\nselected_call_isa_extent_by_pass records each pass's exact coverage. A\nbounded_entry_region has a verified descriptor/entry and address interval, but\nthe true function extent and assignment of every instruction to that function\nremain unknown. A positive ELF symbol size is distinct from verified whole-body\ncoverage. Vendor disassembly does not recover compiler IR. Returned JIT IR, when\npresent, has its own source/specialization binding.\n\nRead the evidence fields actually supplied in this assignment. When present,\nCOHORT_HARDWARE_CALIBRATIONS links prior calibration records with their own\nsource, runtime, physical-device and method limits; those records do not supply\ncurrent-source diagnostics. HARDWARE_EVIDENCE_SKILL, when present, links the\ninvestigation guidance. HARDWARE_EVIDENCE_VIEW, when present, supplies navigation;\nan empty raw_refs list supplies no current-source diagnostic. Use\nCURRENT_ORIGINAL_HARDWARE_EVIDENCE only when that field is actually supplied,\nand follow its exact references for common ORIGINAL observations.\n\nFor every observation, verify source, case, runtime, physical-device and method\napplicability. Current-source diagnostics can supersede prior observations after\na rewrite; a prior result is not evidence for a changed source without that\ncheck. Tool availability, successful collection, complete attribution and\nvalidated physical interpretation are separate. Missing physical HBM traffic,\nachieved residency or workload roofline headroom remains unknown. Keep profiled\ndurations separate from the native unprofiled score.\n"
    require(type(common_tool_addendum) is str, "host common tool addendum must be text", ValueError)
    text += native_capability_context(search_policy) + common_tool_addendum
    return text
