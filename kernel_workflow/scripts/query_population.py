#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read committed population checkpoints without a model call or measurement.

Usage: query_population.py --root EVAL/population [--lineage ID ...]
                           [--base-tree TREE | --base-source-id SHA256]
                           [--checkpoints-only]

The dispatcher's manifest is trusted configuration. Checkpoint commands and
claimed native scores are never authoritative. Native evidence is replayed by
the manifest's result-reader argv against the retained base and patch. Legacy Git
indexes used to check source trees live in temporary directories; this command
does not update population files, pointers, or lineage incumbents.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path


LINEAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
TREE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
CHECKPOINT = re.compile(r"r(0|[1-9][0-9]*)\.json\Z")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def same_score(a, b):
    return positive(a) and positive(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def says(value, word):
    return isinstance(value, str) and value.strip().lower().startswith(word)


def read_json(path):
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=lambda value: need(False, "nonfinite JSON number: " + value))
    need(isinstance(value, dict), "expected a JSON object")
    return value, raw


def contained(path, directory):
    return path.resolve().is_relative_to(directory.resolve())


def run(argv, *, timeout=30, cwd=None, env=None):
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=timeout, check=False)
    need(result.returncode == 0,
         f"command failed ({result.returncode}): {result.stderr.decode(errors='replace')[-2000:].strip()}")
    return result.stdout


class SourceTrees:
    """Use retained legacy Git objects with a private, disposable index."""

    def __init__(self, temporary):
        self.temporary = Path(temporary)
        self.repositories = {}

    def git(self, repository, base, *args):
        env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        env.update(GIT_OPTIONAL_LOCKS="0", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
        return run(["git", "--git-dir", str(repository / ".git"), "--work-tree", str(base),
                    "-c", "core.filemode=true", *args], cwd=base, env=env).decode().strip()

    def base(self, base):
        base = base.resolve(strict=True)
        need(base.is_dir(), "checkpoint base is not a directory")
        if base not in self.repositories:
            repository = self.temporary / str(len(self.repositories))
            run(["git", "init", "-q", str(repository)], env={key: value for key, value in os.environ.items()
                                                          if not key.startswith("GIT_")})
            objects = (base / ".git" / "objects").resolve(strict=True)
            (repository / ".git/objects/info/alternates").write_text(str(objects) + "\n")
            shutil.copy2(base / ".git/index", repository / ".git/index")
            tree = self.git(repository, base, "write-tree")
            self.git(repository, base, "diff-files", "--quiet", "--exit-code")
            need(not self.git(repository, base, "ls-files", "--others", "--exclude-standard"),
                 "untracked files added to retained legacy base")
            self.repositories[base] = (repository, tree)
        return self.repositories[base][1]

    def patched(self, base, patch):
        tree = self.base(base)
        repository = self.repositories[base.resolve()][0]
        self.git(repository, base, "read-tree", tree)
        if patch.stat().st_size:
            self.git(repository, base, "apply", "--cached", "--", str(patch))
        return self.git(repository, base, "write-tree")


def reader(manifest, lane, reference, workspace, patch, *, final=False, timeout=30):
    key = "external_final_result_command" if final and lane.get("external_final_result_command") else "external_result_command"
    argv = lane.get(key)
    need(isinstance(argv, list) and argv and all(isinstance(v, str) and v for v in argv),
         "lineage has no configured native result reader")
    # Runtime identity/source arguments belong to this query, not an argv prefix.
    reserved = ("--workspace", "--patch", "--candidate-id", "--evaluation-id", "--captured", "--capture-only")
    need(not any(v == flag or v.startswith(flag + "=") for v in argv for flag in reserved),
         "configured result reader contains runtime source or identity arguments")
    need(isinstance(reference, dict) and all(isinstance(reference.get(k), str) and reference[k]
         for k in ("candidate_id", "evaluation_id")), "missing exact native evaluation identity")
    expected_task = manifest.get("task_sha256")
    if expected_task is not None:
        need(isinstance(expected_task, str) and SHA256.fullmatch(expected_task), "invalid manifest task SHA256")
        need(digest((Path(manifest["kernel_path"]) / "task.json").read_bytes()) == expected_task,
             "manifest task bytes changed")
        if "--task-root" in argv:
            need(argv.count("--task-root") == 1 and argv.index("--task-root") + 1 < len(argv),
                 "ambiguous configured task root")
            task = Path(argv[argv.index("--task-root") + 1]) / "task.json"
            need(digest(task.read_bytes()) == expected_task, "reader task differs from manifest task")
    output = run([*argv, "--candidate-id", reference["candidate_id"],
                  "--evaluation-id", reference["evaluation_id"], "--workspace", str(workspace),
                  *(["--patch", str(patch)] if patch is not None else [])], timeout=timeout)
    observed = json.loads(output, parse_constant=lambda value: need(False, "nonfinite reader number: " + value))
    need(isinstance(observed, dict), "reader did not return an object")
    objective = manifest["objective"]
    need(observed.get("external_measurement_valid") is True and observed.get("measurement_valid") is True,
         "native measurement is invalid")
    need(observed.get("objective") == objective and observed.get("primary_regime") == objective["primary_regime"],
         "native objective differs from population objective")
    need(observed.get("external_evaluation") == reference, "native reader returned another evaluation")
    need(says(observed.get("correctness"), "pass") and says(observed.get("status"), "verified"),
         "native correctness is not verified")
    need(positive(observed.get("primary_speedup")) and positive(observed.get("development_geomean_speedup")),
         "native reader returned an invalid score")
    need(isinstance(observed.get("source_id"), str) and SHA256.fullmatch(observed["source_id"]),
         "native reader returned no source identity")
    need(type(observed.get("native_reproducible_gain")) is bool
         and type(observed.get("native_reproducible_regression")) is bool, "native qualification flags missing")
    need(not observed["native_reproducible_regression"], "native reproducible regression")
    if patch is not None:
        need(observed.get("patch_sha256") == digest(patch.read_bytes()), "native reader patch hash differs")
    return observed


def legacy_score(evidence, objective):
    need(isinstance(evidence, dict) and says(evidence.get("correctness"), "pass")
         and (says(evidence.get("status"), "verified") or says(evidence.get("status"), "regression")),
         "legacy verifier did not pass source correctness")
    weighted = evidence.get("verified_weighted", evidence.get("speedup_weighted"))
    geomean = evidence.get("verified_geomean", evidence.get("speedup_geomean"))
    score = weighted if objective["kind"] == "time_weighted" and positive(weighted) else geomean
    need(positive(score), "legacy verifier has no finite positive score")
    return score


def optional_design_genome(value, checkpoint, evidence, *, native):
    """Associate bounded author claims with checked source, without scoring them."""
    try:
        need(isinstance(value, dict), "design genome is not an object")
        text = value.get("text")
        need(isinstance(text, str) and text.strip() and len(text.encode("utf-8")) <= 4096,
             "design genome text is empty or oversized")
        base_commit = value.get("base_commit")
        need(isinstance(base_commit, str) and TREE.fullmatch(base_commit)
             and base_commit == checkpoint.get("base_commit"), "design genome base differs")
        need(value.get("patch_sha256") == checkpoint["patch_ref"]["sha256"], "design genome patch differs")
        result = {"kind": "author_claims", "text": text, "base_commit": base_commit,
                  "patch_sha256": checkpoint["patch_ref"]["sha256"]}
        if native:
            for key in ("source_id", "external_evaluation"):
                need(key not in value or value[key] == evidence[key], "design genome " + key + " differs")
                result[key] = evidence[key]
        return result
    except (KeyError, TypeError, ValueError, UnicodeError):
        # Optional narrative can be unavailable without weakening source proof
        # or disqualifying an otherwise importable committed checkpoint.
        return None


def result_for_checkpoint(checkpoint, lane, evidence, score, gain):
    delivery = checkpoint["delivery_patch_ref"]
    result = {"mode": "optimize", "final_speedup": score, "primary_speedup": score,
              "outcome": "improved" if gain else "no_improvement",
              "validation_status": "accepted" if gain else "no_improvement",
              "final_patch": delivery["path"] if gain else "",
              "final_patch_sha256": delivery["sha256"] if gain else None,
              "eval_dir": lane["eval_dir"],
              "rounds": checkpoint["round"], "objective": checkpoint["metric"],
              "population_checkpoint": True}
    if checkpoint["metric"]["kind"] == "external_primary":
        result.update(evidence)
        result["final_selection"] = {**evidence["external_evaluation"], "source_id": evidence["source_id"]}
    return result


def check_checkpoint(path, root, lane, manifest, trees, expected_base, expected_source, timeout):
    record = {"checkpoint_path": str(path), "checkpoint_sha256": None, "checkpoint": None,
              "status": "invalid", "evidence": None}
    candidate = {"lineage_id": lane["id"], "kind": "checkpoint", "round": int(CHECKPOINT.fullmatch(path.name)[1]),
                 "status": "disqualified", "checkpoint_path": str(path)}
    try:
        need(contained(path, root / lane["id"]), "checkpoint file is outside lineage publication directory")
        checkpoint, raw = read_json(path)
        record.update(checkpoint={key: value for key, value in checkpoint.items() if key != "design_genome"},
                      checkpoint_sha256=digest(raw))
        need(checkpoint.get("schema") == "geak.population.checkpoint/1", "unknown checkpoint schema")
        need(checkpoint.get("lineage_id") == lane["id"] and checkpoint.get("optimizer") == lane["optimizer"],
             "checkpoint lineage or optimizer differs from manifest")
        need(type(checkpoint.get("round")) is int and checkpoint["round"] == candidate["round"],
             "checkpoint round differs from filename")
        need(checkpoint.get("verified") is True, "checkpoint is unverified")
        need(checkpoint.get("metric") == manifest["objective"], "checkpoint objective differs from population objective")
        native = manifest["objective"]["kind"] == "external_primary"
        base = Path(checkpoint["base"])
        need(base.is_absolute() and contained(base, Path(lane["eval_dir"])), "checkpoint base is outside lineage eval directory")
        base_tree = checkpoint.get("base_tree")
        need(isinstance(base_tree, str) and TREE.fullmatch(base_tree), "invalid checkpoint base tree")
        need(expected_base is None or expected_base == base_tree, "checkpoint base tree is incompatible with requested cohort")
        if native:
            need(isinstance(checkpoint.get("base_source_id"), str) and SHA256.fullmatch(checkpoint["base_source_id"]),
                 "native checkpoint has no base source identity")
            need(expected_source is None or checkpoint["base_source_id"] == expected_source,
                 "checkpoint base source is incompatible with requested cohort")
        else:
            need(expected_source is None, "base source identity filter requires native checkpoints")
            need(trees.base(base) == base_tree, "retained base tree differs from checkpoint base tree")
        ref = checkpoint.get("patch_ref")
        need(isinstance(ref, dict) and isinstance(ref.get("sha256"), str) and SHA256.fullmatch(ref["sha256"]),
             "invalid checkpoint patch reference")
        patch = Path(ref["path"])
        need(patch.is_absolute() and contained(patch, root / lane["id"]), "checkpoint patch is outside lineage publication directory")
        need(ref.get("base_tree") == base_tree and ref.get("base_commit") == checkpoint.get("base_commit"),
             "checkpoint patch base differs")
        need(digest(patch.read_bytes()) == ref["sha256"], "checkpoint patch hash differs")
        if not native:
            need(trees.patched(base, patch) == checkpoint.get("source_tree"), "checkpoint patch does not reconstruct committed source tree")
        qualification = "native_source_bound" if native else "legacy_verifier"
        need(checkpoint.get("qualification") == qualification, "checkpoint qualification differs from configured objective")
        if native:
            evidence = reader(manifest, lane, checkpoint.get("external_evaluation"), base, patch, timeout=timeout)
            need(checkpoint["base_source_id"] == evidence.get("workspace_source_id"),
                 "checkpoint base source identity differs from native reader")
            need(checkpoint.get("source_id") == evidence["source_id"], "checkpoint source identity differs from native reader")
            score = evidence["primary_speedup"]
            gain = evidence["native_reproducible_gain"] and score > 1
            record["status"] = "qualified" if gain else "no_improvement"
        else:
            evidence = checkpoint.get("evidence")
            score = legacy_score(evidence, manifest["objective"])
            gain = score > 1
            record["status"] = "legacy_verified"
        need(same_score(checkpoint.get("speedup"), score), "claimed checkpoint score differs from checked evidence")
        need(path.read_bytes() == raw and digest(patch.read_bytes()) == ref["sha256"], "checkpoint changed during query")
        record["evidence"] = evidence
        candidate.update(status="eligible" if gain else "no_gain", qualification=qualification, speedup=score,
                         base_tree=base_tree, patch_ref=ref, external_evaluation=checkpoint.get("external_evaluation"),
                         base_source_id=checkpoint.get("base_source_id"), source_id=evidence.get("source_id"),
                         source_status=record["status"], source_base=str(base))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        record.update(status="invalid", evidence=None, reason=str(error))
        candidate["reason"] = str(error)
        return record, candidate
    genome = optional_design_genome(checkpoint.get("design_genome"), checkpoint, evidence, native=native)
    if genome is not None:
        record["design_genome"] = genome
    if not native:
        # A legacy verifier's statement is retained with its exact base+patch.
        # It cannot authenticate an independent ORIGINAL-relative delivery.
        # Only an actual legacy final result may be returned as a public final.
        candidate.update(status="checkpoint_only",
                         reason="legacy checkpoint retains base-relative source; public delivery requires a lineage final result")
        return record, candidate
    # A peer-relative patch is useful for deliberate imports. Public delivery
    # additionally needs the publisher's immutable ORIGINAL-relative patch.
    try:
        delivery = checkpoint.get("delivery_patch_ref")
        need(isinstance(delivery, dict) and delivery.get("base") == "ORIGINAL"
             and isinstance(delivery.get("sha256"), str) and SHA256.fullmatch(delivery["sha256"]),
             "checkpoint has no original-relative delivery patch")
        delivery_path = Path(delivery["path"])
        need(delivery_path.is_absolute() and contained(delivery_path, root / lane["id"]),
             "delivery patch is outside lineage publication directory")
        need(digest(delivery_path.read_bytes()) == delivery["sha256"], "delivery patch hash differs")
        delivered = reader(manifest, lane, checkpoint["external_evaluation"],
                           Path(manifest["kernel_path"]), delivery_path, timeout=timeout)
        need(delivered["source_id"] == evidence["source_id"] and same_score(delivered["primary_speedup"], score),
             "delivery differs from checked checkpoint source or score")
        evidence = delivered
        need(digest(delivery_path.read_bytes()) == delivery["sha256"] and path.read_bytes() == raw,
             "checkpoint delivery changed during query")
        candidate.update(delivery_patch_ref=delivery,
                         result=result_for_checkpoint(checkpoint, lane, evidence, score, gain))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        candidate.update(status="disqualified", reason="delivery: " + str(error))
    return record, candidate


def check_final(path, lane, manifest, timeout):
    candidate = {"lineage_id": lane["id"], "kind": "final", "round": 0,
                 "status": "disqualified", "result_path": str(path)}
    try:
        result, raw = read_json(path)
        candidate["round"] = result.get("rounds", 0) if type(result.get("rounds", 0)) is int else 0
        need(result.get("validation_status") in ("accepted", "no_improvement"), "final validation did not pass")
        native = manifest["objective"]["kind"] == "external_primary"
        patch_name = result.get("final_patch") or str(Path(lane["eval_dir"]) / "host_final_patch.diff")
        patch = Path(patch_name)
        need(patch.is_absolute() and contained(patch, Path(lane["eval_dir"])), "final patch is outside lineage eval directory")
        qualification = "native_source_bound" if native else "legacy_verifier"
        if native:
            # The final measurement uses the original task base, even when a
            # lineage's development archive starts from an accepted warm start.
            observed = reader(manifest, lane, result.get("external_evaluation"),
                              Path(manifest["kernel_path"]), patch, final=True, timeout=timeout)
            selection = result.get("final_selection")
            need(isinstance(selection, dict) and all(selection.get(key) == result["external_evaluation"][key]
                 for key in ("candidate_id", "evaluation_id")) and selection.get("source_id") == observed["source_id"],
                 "final selection differs from checked source or evaluation identity")
            score = observed["primary_speedup"]
            gain = observed["native_reproducible_gain"] and score > 1 and result["validation_status"] == "accepted"
            need(same_score(result.get("final_speedup"), score), "claimed final score differs from checked evidence")
            result = {**result, **observed, "final_speedup": score,
                      "outcome": "improved" if gain else "no_improvement",
                      "validation_status": "accepted" if gain else "no_improvement"}
        else:
            score = result.get("final_speedup")
            need(positive(score), "final verifier has no finite positive score")
            gain = score > 1 and result["validation_status"] == "accepted"
            need(not gain or bool(result.get("final_patch")), "improved final has no patch")
        patch_ref = None
        if native or gain:
            patch_hash = digest(patch.read_bytes())
            need(not result.get("final_patch_sha256") or result["final_patch_sha256"] == patch_hash,
                 "final patch hash differs")
            patch_ref = {"path": str(patch), "sha256": patch_hash}
        need(path.read_bytes() == raw, "final result changed during query")
        # Final-result narratives have no supported association contract. Do
        # not let arbitrary fields bypass the checkpoint-only sanitizer.
        result.pop("design_genome", None)
        candidate.update(status="eligible" if gain else "no_gain", qualification=qualification,
                         speedup=score, patch_ref=patch_ref,
                         delivery_patch_ref={**patch_ref, "base": "ORIGINAL"} if patch_ref else None,
                         source_id=result.get("source_id"),
                         external_evaluation=result.get("external_evaluation"), result=result)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        candidate["reason"] = str(error)
    return candidate


def select(candidates):
    eligible = [row for row in candidates if row["status"] == "eligible"]
    available = eligible or [row for row in candidates if row["status"] == "no_gain"]
    return min(available, key=lambda row: (-row["speedup"], row["lineage_id"], row["round"], row["kind"])) if available else None


def query(root, *, lineages=(), base_tree=None, base_source_id=None, checkpoints_only=False, reader_timeout=30):
    root = Path(root).resolve(strict=True)
    manifest, _ = read_json(root / "manifest.json")
    need(manifest.get("schema") == "geak.population/1", "unknown population manifest schema")
    objective = manifest.get("objective")
    need(isinstance(objective, dict) and objective.get("kind") in ("external_primary", "geomean", "time_weighted"),
         "population manifest has no supported objective")
    if objective["kind"] == "external_primary":
        need(set(objective) == {"kind", "primary_regime"} and isinstance(objective["primary_regime"], str)
             and objective["primary_regime"], "invalid external population objective")
    else:
        need(set(objective) == {"kind"}, "invalid legacy population objective")
    configured = manifest.get("lineages")
    need(isinstance(configured, list) and configured, "population manifest has no lineages")
    ids = [lane.get("id") for lane in configured if isinstance(lane, dict)]
    need(len(ids) == len(configured) and all(isinstance(id_, str) and LINEAGE.fullmatch(id_) for id_ in ids)
         and len(ids) == len(set(ids)), "invalid or duplicate manifest lineage IDs")
    need(not set(lineages) - set(ids), "requested lineage is absent from manifest")
    expected_base = base_tree or manifest.get("base_tree")
    need(expected_base is None or isinstance(expected_base, str) and TREE.fullmatch(expected_base), "invalid requested base tree")
    need(base_source_id is None or isinstance(base_source_id, str) and SHA256.fullmatch(base_source_id),
         "invalid requested base source identity")
    records, candidates, dispositions = [], [], []
    with tempfile.TemporaryDirectory(prefix="geak-population-query-") as temporary:
        trees = SourceTrees(temporary)
        for lane in sorted(configured, key=lambda item: item["id"]):
            if lineages and lane["id"] not in lineages:
                continue
            disposition = {"id": lane["id"], "status": "pending", "errors": []}
            dispositions.append(disposition)
            try:
                directory = root / lane["id"]
                need(contained(directory, root), "lineage publication directory is outside population root")
                paths = sorted((path for path in directory.glob("r*.json") if CHECKPOINT.fullmatch(path.name)),
                               key=lambda path: int(CHECKPOINT.fullmatch(path.name)[1]))
                for path in paths:
                    record, candidate = check_checkpoint(path, root, lane, manifest, trees, expected_base, base_source_id, reader_timeout)
                    records.append(record)
                    candidates.append(candidate)
                pointer = directory / "latest.json"
                if pointer.exists():
                    try:
                        value, _ = read_json(pointer)
                        target = Path(value["checkpoint"])
                        need(value.get("schema") == "geak.population.pointer/1" and value.get("lineage_id") == lane["id"]
                             and contained(target, directory) and target in paths
                             and digest(target.read_bytes()) == value.get("sha256"), "invalid latest checkpoint pointer")
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        disposition["errors"].append("latest pointer: " + str(error))
                final = Path(lane["eval_dir"]) / "population_final_result.json"
                if not checkpoints_only and final.exists():
                    candidates.append(check_final(final, lane, manifest, reader_timeout))
                own = [row for row in candidates if row["lineage_id"] == lane["id"]]
                disposition["status"] = ("available" if any(row["status"] in ("eligible", "no_gain") for row in own)
                    else "checkpoint_only" if any(row["status"] == "checkpoint_only" for row in own)
                    else "disqualified" if own else "pending")
            except (OSError, ValueError, KeyError, TypeError) as error:
                disposition.update(status="error")
                disposition["errors"].append(str(error))
    groups = {}
    for candidate in candidates:
        if candidate["kind"] == "checkpoint" and candidate["status"] in ("eligible", "no_gain"):
            groups.setdefault(candidate["base_tree"], []).append(candidate)
    compatible = objective["kind"] == "external_primary" or len(groups) <= 1
    selectable = candidates if compatible else [row for row in candidates if row["kind"] == "final"]
    finalists = []
    for lane in dispositions:
        finalist = select([row for row in candidates if row["lineage_id"] == lane["id"]])
        if finalist is not None:
            finalists.append(finalist)
    sources = [row for row in candidates if row.get("source_status") in ("qualified", "no_improvement", "legacy_verified")]
    gains = [row for row in sources if row["source_status"] == "qualified"
             or row["source_status"] == "legacy_verified" and row["speedup"] > 1]
    best_source = min(gains or sources, key=lambda row: (-row["speedup"], row["lineage_id"], row["round"])) if sources else None
    if best_source:
        best_source = {key: best_source.get(key) for key in ("lineage_id", "round", "qualification", "speedup",
            "source_status", "source_base", "base_tree", "base_source_id", "source_id", "patch_ref",
            "external_evaluation", "checkpoint_path")}
    return {"schema": "geak.population.query/1", "manifest": str(root / "manifest.json"), "objective": objective,
            "base_tree": expected_base or (next(iter(groups)) if len(groups) == 1 else None),
            "base_tree_basis": "requested" if base_tree else "manifest" if manifest.get("base_tree") else "checked_checkpoints",
            "base_source_id": base_source_id,
            "compatible": compatible,
            "selection_note": None if compatible else "incompatible checkpoint base trees; only common-task final results are compared",
            "selected": select(selectable), "finalists": finalists, "best_available_source": best_source,
            "candidates": candidates, "records": records, "lineages": dispositions,
            "compatible_groups": [{"base_tree": tree, "selected": select(rows), "checkpoint_count": len(rows)}
                                  for tree, rows in sorted(groups.items())]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lineage", action="append", default=[])
    parser.add_argument("--base-tree")
    parser.add_argument("--base-source-id")
    parser.add_argument("--checkpoints-only", action="store_true")
    parser.add_argument("--reader-timeout", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        need(math.isfinite(args.reader_timeout) and args.reader_timeout > 0, "reader timeout must be positive")
        result = query(args.root, lineages=args.lineage, base_tree=args.base_tree,
                       base_source_id=args.base_source_id, checkpoints_only=args.checkpoints_only, reader_timeout=args.reader_timeout)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps({"schema": "geak.population.query/1", "selected": None, "error": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
