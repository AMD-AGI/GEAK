#!/usr/bin/env python3
"""validate_skill.py — gate an expert skill before it can land as `validated`.

Two-sided gate (see expert_skills/README.md):
  1. EFFICACY   — on a matching model/shape the skill achieves >= its `expects` (isolated or e2e).
  2. DO-NO-HARM — integrating it does not regress a control scenario / non-trigger run.

This tool does NOT itself spin up a server or a GPU. The actual measurement is produced by the real
harness (kernel_workflow for scope:kernel, e2e_workflow for scope:e2e) so numbers stay honest and
on-box. validate_skill.py has three modes:

  --static            schema + operator alignment + required sections + links. No GPU. (CI default.)
  --emit-bundle       emit path-independent component digests for a strict RunContract.
  --emit-plan         print the exact Workflow invocation to measure this skill (by scope).
  --record ...        legacy validation-v1 result recorder. Validation-v2 transfers remain
                      experimental until exact-candidate hardware evidence is promoted.

Examples:
  python _contribute/validate_skill.py flydsl_fp8_gemm_playbook --static
  python _contribute/validate_skill.py flydsl_fp8_gemm_playbook --emit-plan --model /models/Qwen3.5-27B-FP8
  python _contribute/validate_skill.py flydsl_fp8_gemm_playbook --record \
      --artifact /path/eval_dir --e2e-pct 2.1 --parity pass --gpu gfx942/MI300X --model Qwen3.5-27B-FP8
"""
import argparse, hashlib, json, os, re, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKILLS_DIR = os.path.join(ROOT, "skills")
CAP_INDEX = os.path.normpath(os.path.join(ROOT, "..", "index", "capability_index.yaml"))
GEAK = os.path.normpath(os.path.join(ROOT, "..", ".."))
REQUIRED_SECTIONS = ["When to use", "Mechanism", "Procedure", "Do-no-harm notes", "Sources"]
FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)
EMBEDDED_TAGS = {
    "planner_extension": "expert-skill-planner-extension",
    "contract": "expert-skill-contract",
    "runtime_validation": "expert-skill-runtime-python",
}
V2_REQUIRED_CANDIDATE_GATES = {
    "independent_structure", "device_jit", "path_activation_all_ranks",
    "launch_shape", "direct_accuracy", "graph_liveness", "residency",
    "distributed_memory_order", "artifact_distinctness",
    "paired_performance", "do_no_harm",
}


def load(skill_id):
    path = os.path.join(SKILLS_DIR, skill_id, "skill.md")
    if not os.path.exists(path):
        sys.exit(f"ERROR: no such skill: {path}")
    txt = open(path).read()
    m = FM_RE.match(txt)
    if not m:
        sys.exit(f"ERROR: {path}: no YAML frontmatter")
    return path, yaml.safe_load(m.group(1)), m.group(2), txt


def _skill_file(skill_path, fm, key):
    value = str(fm.get(key) or "").strip()
    if not value:
        return None
    if os.path.isabs(value) or ".." in value.split(os.sep):
        raise ValueError(f"{key} must be relative to the skill directory")
    return os.path.join(os.path.dirname(skill_path), value)


def load_validation(skill_path, fm):
    path = _skill_file(skill_path, fm, "validation_file")
    if path is None:
        return None, fm.get("validation") or {}
    if not os.path.isfile(path):
        return path, {}
    data = yaml.safe_load(open(path)) or {}
    return path, data if isinstance(data, dict) else {}


def embedded_component_names(fm):
    values = fm.get("embedded_components") or []
    return {str(value) for value in values} if isinstance(values, list) else set()


def load_embedded_component(skill_path, fm, label):
    if label not in embedded_component_names(fm):
        return None
    tag = EMBEDDED_TAGS.get(label)
    if not tag:
        raise ValueError(f"unsupported embedded component: {label}")
    text = open(skill_path).read()
    matches = re.findall(
        rf"```{re.escape(tag)}[^\n]*\n(.*?)\n```",
        text,
        flags=re.S,
    )
    if len(matches) != 1:
        raise ValueError(
            f"skill.md must contain exactly one fenced {tag} block"
        )
    if label == "runtime_validation":
        return matches[0]
    data = yaml.safe_load(matches[0])
    if not isinstance(data, dict):
        raise ValueError(f"embedded {label} must contain a YAML object")
    return data


def load_component(skill_path, fm, file_key, label):
    path = _skill_file(skill_path, fm, file_key)
    if path:
        if not os.path.isfile(path):
            raise ValueError(f"{file_key} does not exist: {fm.get(file_key)!r}")
        if label == "runtime_validation":
            return open(path).read(), path
        return yaml.safe_load(open(path)) or {}, path
    return load_embedded_component(skill_path, fm, label), skill_path


def validation_metadata(validation):
    data = validation if isinstance(validation, dict) else {}
    status = str(data.get("status") or "draft")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    requested_auto = usage.get("auto_apply")
    auto_apply = (
        status == "validated"
        and (
            requested_auto is True
            if data.get("schema_version") == "expert-skill-validation-v2"
            else (True if requested_auto is None else requested_auto is True)
        )
    )
    modes = usage.get("explicit_pin_modes") or []
    return {
        "validation_status": status,
        "reference_evidence_status": str(
            (data.get("reference_evidence") or {}).get("status") or ""
        ),
        "constraint_validation_status": str(
            (data.get("constraint_validation") or {}).get("status") or ""
        ),
        "auto_apply": auto_apply,
        "explicit_pin_modes": [str(value) for value in modes],
    }


def validation_errors(fm, validation, base_dir=None):
    data = validation if isinstance(validation, dict) else {}
    schema = str(data.get("schema_version") or "expert-skill-validation-v1")
    status = str(data.get("status") or "")
    allowed = {"draft", "experimental", "validated", "stale", "failed", "mega_only"}
    errors = []
    if schema not in {"expert-skill-validation-v1", "expert-skill-validation-v2"}:
        return [f"validation_file uses unsupported schema {schema!r}"]
    if status not in allowed:
        errors.append("validation_file status is invalid")
    if schema == "expert-skill-validation-v1":
        return errors

    reference = data.get("reference_evidence")
    constraints = data.get("constraint_validation")
    policy = data.get("candidate_validation_policy")
    usage = data.get("usage")
    if status != "draft" and (
        not isinstance(reference, dict) or reference.get("status") != "measured"
    ):
        errors.append("v2 reference_evidence.status must be measured")
    if not isinstance(constraints, dict) or constraints.get("status") not in {
        "draft", "static_validated", "hardware_validated", "failed",
    }:
        errors.append("v2 constraint_validation.status is invalid")
        constraints = {}
    if status != "draft" and (
        not isinstance(policy, dict) or not policy.get("required_gates")
    ):
        errors.append("v2 candidate_validation_policy.required_gates must be non-empty")
    if not isinstance(usage, dict):
        errors.append("v2 usage must be an object")
        usage = {}
    modes = usage.get("explicit_pin_modes") or []
    if not isinstance(modes, list) or set(modes) - {"authoring", "candidate_validation"}:
        errors.append("v2 usage.explicit_pin_modes is invalid")
    if status != "validated" and usage.get("auto_apply") is True:
        errors.append("only a validated v2 skill may auto_apply")
    if status == "experimental" and not modes:
        errors.append("experimental v2 skill requires an explicit pin mode")
    if status == "validated":
        if constraints.get("status") != "hardware_validated":
            errors.append("validated v2 skill requires hardware_validated constraints")
        if constraints.get("hardware_verified") is not True:
            errors.append("validated v2 skill requires hardware_verified=true")
        rules = constraints.get("rules") or {}
        if not isinstance(rules, dict) or not rules:
            errors.append("validated v2 skill requires non-empty rule evidence")
            rules = {}
        evidence = data.get("candidate_evidence")
        if not isinstance(evidence, dict):
            errors.append("validated v2 skill requires exact candidate_evidence")
            evidence = {}
        for field, pattern in (
            ("candidate_head", r"[0-9a-f]{40}"),
            ("candidate_tree_sha256", r"[0-9a-f]{64}"),
            ("checker_sha256", r"[0-9a-f]{64}"),
            ("manifest_sha256", r"[0-9a-f]{64}"),
        ):
            if not re.fullmatch(pattern, str(evidence.get(field) or "")):
                errors.append(f"validated v2 candidate_evidence.{field} is invalid")
        if not str(evidence.get("manifest_uri") or ""):
            errors.append("validated v2 candidate_evidence.manifest_uri is required")
        for field in (
            "contract_sha256", "planner_extension_sha256", "checker_sha256"
        ):
            if str(evidence.get(field) or "") != str(
                (constraints.get("subject") or {}).get(field) or ""
            ):
                errors.append(
                    f"validated v2 candidate_evidence.{field} must match constraint subject"
                )
        gate_results = evidence.get("gates")
        required_gates = (policy or {}).get("required_gates") or []
        missing_required = V2_REQUIRED_CANDIDATE_GATES - set(required_gates)
        if missing_required:
            errors.append(
                "validated v2 policy omits mandatory gates: "
                + ", ".join(sorted(missing_required))
            )
        if not isinstance(gate_results, dict):
            errors.append("validated v2 candidate_evidence.gates must be an object")
            gate_results = {}
        for gate in required_gates:
            if gate_results.get(gate) not in (True, "pass"):
                errors.append(f"validated v2 candidate gate is not passing: {gate}")
        manifest_uri = str(evidence.get("manifest_uri") or "")
        manifest = {}
        if not base_dir:
            errors.append("validated v2 manifest verification requires a skill directory")
        elif (
            os.path.isabs(manifest_uri)
            or ".." in manifest_uri.replace("\\", "/").split("/")
        ):
            errors.append("validated v2 manifest_uri must be skill-relative")
        else:
            manifest_path = os.path.join(base_dir, manifest_uri)
            if not os.path.isfile(manifest_path):
                errors.append("validated v2 manifest_uri does not exist")
            else:
                raw_manifest = open(manifest_path, "rb").read()
                if hashlib.sha256(raw_manifest).hexdigest() != evidence.get("manifest_sha256"):
                    errors.append("validated v2 manifest_sha256 does not match manifest bytes")
                try:
                    manifest = json.loads(raw_manifest)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    errors.append("validated v2 candidate manifest is not valid JSON")
        if manifest:
            if manifest.get("schema_version") != "expert-skill-candidate-evidence-v1":
                errors.append("validated v2 candidate manifest schema is invalid")
            for field in (
                "candidate_head", "candidate_tree_sha256", "contract_sha256",
                "planner_extension_sha256", "checker_sha256",
            ):
                if manifest.get(field) != evidence.get(field):
                    errors.append(
                        f"validated v2 candidate manifest {field} mismatch"
                    )
            if manifest.get("skill_revision") != fm.get("revision"):
                errors.append("validated v2 candidate manifest skill revision mismatch")
            manifest_gates = manifest.get("gates")
            raw_evidence = manifest.get("raw_evidence")
            if not isinstance(manifest_gates, dict) or not isinstance(raw_evidence, dict):
                errors.append("validated v2 manifest gates/raw_evidence are required")
            else:
                for gate in required_gates:
                    row = raw_evidence.get(gate)
                    if manifest_gates.get(gate) not in (True, "pass"):
                        errors.append(
                            f"validated v2 manifest lacks raw passing evidence: {gate}"
                        )
                        continue
                    if not isinstance(row, dict):
                        errors.append(
                            f"validated v2 raw evidence is not structured: {gate}"
                        )
                        continue
                    uri = str(row.get("artifact_uri") or "")
                    digest = str(row.get("sha256") or "")
                    if (
                        not base_dir or not uri
                        or os.path.isabs(uri)
                        or ".." in uri.replace("\\", "/").split("/")
                    ):
                        errors.append(
                            f"validated v2 raw evidence path is invalid: {gate}"
                        )
                        continue
                    artifact_path = os.path.join(base_dir, uri)
                    if not os.path.isfile(artifact_path):
                        errors.append(
                            f"validated v2 raw evidence artifact is missing: {gate}"
                        )
                        continue
                    raw_artifact = open(artifact_path, "rb").read()
                    if hashlib.sha256(raw_artifact).hexdigest() != digest:
                        errors.append(
                            f"validated v2 raw evidence hash mismatch: {gate}"
                        )
                        continue
                    try:
                        artifact = json.loads(raw_artifact)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        errors.append(
                            f"validated v2 raw evidence is not JSON: {gate}"
                        )
                        continue
                    if not (
                        artifact.get("schema_version")
                        == "expert-skill-gate-evidence-v1"
                        and artifact.get("gate") == gate
                        and artifact.get("status") in (True, "pass")
                        and artifact.get("candidate_head")
                        == evidence.get("candidate_head")
                        and artifact.get("candidate_tree_sha256")
                        == evidence.get("candidate_tree_sha256")
                    ):
                        errors.append(
                            f"validated v2 raw evidence content mismatch: {gate}"
                        )
        for rule_id, rule in rules.items():
            repair = rule.get("positive_repair") if isinstance(rule, dict) else None
            if not isinstance(repair, dict) or repair.get("status") != "validated":
                errors.append(
                    f"validated v2 skill has pending positive repair: {rule_id}"
                )
            elif repair.get("candidate_head") != evidence.get("candidate_head"):
                errors.append(
                    f"validated v2 rule evidence is not bound to candidate: {rule_id}"
                )
            elif repair.get("candidate_tree_sha256") != evidence.get(
                "candidate_tree_sha256"
            ):
                errors.append(
                    f"validated v2 rule tree evidence is not bound to candidate: {rule_id}"
                )
    return errors


def _canonical_component_sha256(path):
    if not path or not os.path.isfile(path):
        return ""
    if os.path.splitext(path)[1].lower() in (".yaml", ".yml"):
        data = yaml.safe_load(open(path))
        payload = yaml.safe_dump(data, sort_keys=True).encode()
    else:
        payload = open(path, "rb").read()
    return hashlib.sha256(payload).hexdigest()


def _canonical_value_sha256(value):
    if isinstance(value, (dict, list)):
        payload = yaml.safe_dump(value, sort_keys=True).encode()
    else:
        payload = str(value).encode()
    return hashlib.sha256(payload).hexdigest()


def skill_bundle_identity(skill_path, fm):
    """Return a path-independent identity for every Skill knowledge component."""
    components = {"skill": _canonical_component_sha256(skill_path)}
    for key, label in (
        ("playbook_file", "playbook"),
        ("planner_extension_file", "planner_extension"),
        ("contract_file", "contract"),
        ("validation_file", "validation"),
        ("runtime_validation_file", "runtime_validation"),
    ):
        try:
            path = _skill_file(skill_path, fm, key)
        except ValueError:
            path = None
        if path:
            components[label] = _canonical_component_sha256(path)
        elif label in embedded_component_names(fm):
            components[label] = _canonical_value_sha256(
                load_embedded_component(skill_path, fm, label)
            )
    if "validation" not in components and isinstance(fm.get("validation"), dict):
        components["validation"] = _canonical_value_sha256(fm["validation"])
    manifest = {
        "schema_version": "expert-skill-bundle-v1",
        "skill_id": str(fm.get("id") or ""),
        "revision": str(fm.get("revision") or ""),
        "components": components,
    }
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    _, validation = load_validation(skill_path, fm)
    return {
        **manifest,
        "bundle_sha256": hashlib.sha256(canonical).hexdigest(),
        "planner_extension_sha256": components.get("planner_extension", ""),
        "contract_sha256": components.get("contract", ""),
        **validation_metadata(validation),
    }


def planner_extension_errors(skill_path, fm):
    """Validate the generic envelope without interpreting operator-specific values."""
    relative = str(fm.get("planner_extension_file") or "").strip()
    embedded = "planner_extension" in embedded_component_names(fm)
    if not relative and not embedded:
        return []
    try:
        if embedded:
            extension = load_embedded_component(
                skill_path, fm, "planner_extension"
            )
            path = skill_path
        else:
            path = _skill_file(skill_path, fm, "planner_extension_file")
            extension = yaml.safe_load(open(path)) if path else {}
    except ValueError as exc:
        return [str(exc)]
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot parse planner_extension_file: {exc}"]
    if not embedded and (not path or not os.path.isfile(path)):
        return [f"planner_extension_file does not exist: {relative!r}"]
    if not isinstance(extension, dict):
        return ["planner_extension_file must contain a YAML object"]

    errs = []
    if extension.get("schema_version") != "expert-skill-planner-extension-v1":
        errs.append(
            "planner_extension_file must use schema_version "
            "expert-skill-planner-extension-v1"
        )
    if extension.get("skill_id") != fm.get("id"):
        errs.append("planner_extension_file skill_id must match skill.md id")
    if str(extension.get("revision") or "") != str(fm.get("revision") or ""):
        errs.append("planner_extension_file revision must match skill.md revision")
    if extension.get("plan_version") != "mega-plan-v2":
        errs.append("planner_extension_file plan_version must be mega-plan-v2")

    bindings = extension.get("ir_bindings")
    allowed_bindings = {
        "target", "target_by_bucket", "work_domains", "regions", "buffers", "counters", "queues",
        "events", "abi", "resources", "schedule", "compiler_constraints",
        "evidence_requirements", "known_unknowns",
    }
    if not isinstance(bindings, dict) or not bindings:
        errs.append("planner_extension_file ir_bindings must be a non-empty object")
    elif set(bindings) - allowed_bindings:
        errs.append(
            "planner_extension_file has non-PlanIR ir_bindings: " +
            ", ".join(sorted(set(bindings) - allowed_bindings))
        )
    binding_ids = {}
    if isinstance(bindings, dict):
        for collection in ("work_domains", "regions", "buffers", "counters", "queues", "events"):
            values = bindings.get(collection) or []
            if not isinstance(values, list):
                errs.append(f"planner ir_bindings.{collection} must be a list")
                continue
            present = [
                str(item.get("id") or "") for item in values if isinstance(item, dict)
            ]
            if len(present) != len(values) or any(not item for item in present):
                errs.append(f"every planner ir_bindings.{collection} entry needs an id")
            elif len(present) != len(set(present)):
                errs.append(f"planner ir_bindings.{collection} ids must be unique")
            binding_ids[collection] = set(present)
        targets = {"target": bindings.get("target") or {}}
        by_bucket = bindings.get("target_by_bucket") or {}
        if not isinstance(by_bucket, dict):
            errs.append("planner ir_bindings.target_by_bucket must map a capacity to a target")
            by_bucket = {}
        for bucket, bucket_target in by_bucket.items():
            if not str(bucket).isdigit() or not isinstance(bucket_target, dict):
                errs.append(
                    f"planner ir_bindings.target_by_bucket[{bucket!r}] must be a target "
                    "keyed by a capacity"
                )
                continue
            targets[f"target_by_bucket[{bucket}]"] = bucket_target
        for name, target in targets.items():
            if not isinstance(target, dict):
                continue
            for field, collection in (
                ("required_regions", "regions"),
                ("required_queues", "queues"),
            ):
                values = target.get(field) or []
                unknown = set(map(str, values)) - binding_ids.get(collection, set())
                if unknown:
                    errs.append(
                        f"planner ir_bindings.{name} names unknown {field}: " +
                        ", ".join(sorted(unknown))
                    )

    templates = extension.get("candidate_templates")
    if not isinstance(templates, list) or not templates:
        errs.append("planner_extension_file candidate_templates must be a non-empty list")
    else:
        ids = [str(item.get("id") or "") for item in templates if isinstance(item, dict)]
        if len(ids) != len(templates) or any(not item for item in ids):
            errs.append("every planner candidate template needs an id")
        elif len(ids) != len(set(ids)):
            errs.append("planner candidate template ids must be unique")
        for item in templates:
            if not isinstance(item, dict):
                continue
            topology = item.get("target_topology")
            if not isinstance(topology, dict) or not (
                isinstance(topology.get("launch_count"), int) and
                not isinstance(topology.get("launch_count"), bool) and
                topology["launch_count"] > 0
            ):
                errs.append(
                    f"candidate template {item.get('id') or '?'} needs a positive "
                    "target_topology.launch_count"
                )
                continue
            for field, collection in (
                ("included_regions", "regions"),
                ("included_queues", "queues"),
            ):
                values = topology.get(field) or []
                if not isinstance(values, list):
                    errs.append(
                        f"candidate template {item.get('id') or '?'} {field} must be a list"
                    )
                    continue
                unknown = set(map(str, values)) - binding_ids.get(collection, set())
                if unknown:
                    errs.append(
                        f"candidate template {item.get('id') or '?'} names unknown {field}: " +
                        ", ".join(sorted(unknown))
                    )

    routes = extension.get("failure_routes")
    if not isinstance(routes, list) or not routes:
        errs.append("planner_extension_file failure_routes must be a non-empty list")
    else:
        route_ids = []
        known_check_ids = set()
        try:
            contract, _ = load_component(
                skill_path, fm, "contract_file", "contract"
            )
            contract = contract or {}
            known_check_ids.update(
                str(item.get("id")) for item in (contract.get("checks") or [])
                if isinstance(item, dict) and item.get("id")
            )
            known_check_ids.update(
                str(item.get("id"))
                for item in ((contract.get("plan") or {}).get("assertions") or [])
                if isinstance(item, dict) and item.get("id")
            )
        except (OSError, yaml.YAMLError, ValueError):
            known_check_ids = set()
        for route in routes:
            if not isinstance(route, dict):
                errs.append("every planner failure route must be an object")
                continue
            route_id = str(route.get("id") or "")
            route_ids.append(route_id)
            match = route.get("match")
            if not route_id:
                errs.append("every planner failure route needs an id")
            if not isinstance(match, dict) or not (
                match.get("check_ids") or match.get("categories")
            ):
                errs.append(f"failure route {route_id or '?'} needs check_ids or categories")
            if not isinstance(route.get("repair_intent"), str) or not route["repair_intent"].strip():
                errs.append(f"failure route {route_id or '?'} needs repair_intent")
            check_ids = match.get("check_ids") if isinstance(match, dict) else []
            categories = match.get("categories") if isinstance(match, dict) else []
            if check_ids and not isinstance(check_ids, list):
                errs.append(f"failure route {route_id or '?'} check_ids must be a list")
                check_ids = []
            if categories and not isinstance(categories, list):
                errs.append(f"failure route {route_id or '?'} categories must be a list")
            unknown = set(map(str, check_ids or [])) - known_check_ids
            if known_check_ids and unknown:
                errs.append(
                    f"failure route {route_id or '?'} names unknown contract checks: " +
                    ", ".join(sorted(unknown))
                )
        if all(route_ids) and len(route_ids) != len(set(route_ids)):
            errs.append("planner failure route ids must be unique")

    forbidden_keys = {
        "baseline_path", "candidate_path", "oracle_path", "reference_path",
        "run_id", "candidate_head",
    }

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key) in forbidden_keys:
                    errs.append(
                        f"planner_extension_file must not contain environment identity key {key}"
                    )
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str) and os.path.isabs(value):
            errs.append(
                "planner_extension_file must not contain absolute machine paths"
            )

    walk(extension)
    return errs


def static_check(skill_path, fm, body):
    errs = []
    for k in ("id", "scope", "match", "expects"):
        if k not in fm:
            errs.append(f"missing frontmatter key: {k}")
    if fm.get("scope") not in ("kernel", "e2e"):
        errs.append(f"scope must be kernel|e2e (got {fm.get('scope')!r})")
    op = (fm.get("match") or {}).get("operator")
    if os.path.exists(CAP_INDEX):
        ops = {c["operator"] for c in (yaml.safe_load(open(CAP_INDEX)).get("candidates") or [])}
        # `*` is the wildcard the schema already uses for arch_class: a skill whose applicability
        # is decided by the STATE of the source rather than by what the kernel computes cannot name
        # one operator, and forcing it to pick one makes the selector decline the matches it should
        # take. A list is accepted for the middle ground.
        for one in ([op] if isinstance(op, str) else list(op or [])):
            if one != "*" and one not in ops:
                errs.append(f"match.operator '{one}' not in capability_index.yaml")
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
    for key in (
        "playbook_file", "planner_extension_file", "contract_file", "validation_file",
        "runtime_validation_file",
    ):
        if not fm.get(key):
            continue
        try:
            referenced = _skill_file(skill_path, fm, key)
        except ValueError as exc:
            errs.append(str(exc))
            continue
        if not referenced or not os.path.isfile(referenced):
            errs.append(f"{key} does not exist: {fm.get(key)!r}")
    runtime_file = None
    try:
        runtime_file = _skill_file(skill_path, fm, "runtime_validation_file")
    except ValueError:
        pass
    if runtime_file and os.path.isfile(runtime_file):
        try:
            compile(open(runtime_file).read(), runtime_file, "exec")
        except SyntaxError as exc:
            errs.append(f"runtime_validation_file is invalid Python: {exc}")
    for label in embedded_component_names(fm):
        try:
            component = load_embedded_component(skill_path, fm, label)
            if label == "runtime_validation":
                compile(component, f"{skill_path}#{EMBEDDED_TAGS[label]}", "exec")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errs.append(str(exc))
        except SyntaxError as exc:
            errs.append(f"embedded runtime_validation is invalid Python: {exc}")
    contract_path = None
    contract_embedded = "contract" in embedded_component_names(fm)
    contract = {}
    try:
        if contract_embedded:
            contract = load_embedded_component(skill_path, fm, "contract")
            contract_path = skill_path
        else:
            contract_path = _skill_file(skill_path, fm, "contract_file")
    except ValueError:
        pass
    if contract_path and os.path.isfile(contract_path):
        try:
            if not contract_embedded:
                contract = yaml.safe_load(open(contract_path)) or {}
            if contract.get("schema_version") != "expert-skill-contract-v1":
                errs.append("contract_file must use schema_version expert-skill-contract-v1")
            if contract.get("skill_id") != fm.get("id"):
                errs.append("contract_file skill_id must match skill.md id")
            if str(contract.get("revision") or "") != str(fm.get("revision") or ""):
                errs.append("contract_file revision must match skill.md revision")
            if not contract.get("checks"):
                errs.append("contract_file checks must be non-empty")
        except (OSError, yaml.YAMLError) as exc:
            errs.append(f"cannot parse contract_file: {exc}")
    playbook_path = None
    try:
        playbook_path = _skill_file(skill_path, fm, "playbook_file")
    except ValueError:
        pass
    if playbook_path and os.path.isfile(playbook_path):
        try:
            match = FM_RE.match(open(playbook_path).read())
            if not match:
                errs.append("playbook_file must have YAML frontmatter")
            else:
                playbook = yaml.safe_load(match.group(1)) or {}
                if str(playbook.get("revision") or "") != str(fm.get("revision") or ""):
                    errs.append("playbook_file revision must match skill.md revision")
        except (OSError, yaml.YAMLError) as exc:
            errs.append(f"cannot parse playbook_file: {exc}")
    errs.extend(planner_extension_errors(skill_path, fm))
    try:
        validation_path, validation = load_validation(skill_path, fm)
    except ValueError as exc:
        errs.append(str(exc))
        validation_path, validation = None, {}
    if validation_path or validation:
        validation_base = os.path.dirname(validation_path or skill_path)
        if fm.get("validation_schema") and str(
            validation.get("schema_version") or ""
        ) != str(fm.get("validation_schema")):
            errs.append("validation_file schema must match skill.md validation_schema")
        # Inline frontmatter validation is part of skill.md itself; only a separate
        # validation file has to name the skill and revision it belongs to.
        if validation_path:
            if validation.get("skill_id") != fm.get("id"):
                errs.append("validation_file skill_id must match skill.md id")
            if str(validation.get("revision") or "") != str(fm.get("revision") or ""):
                errs.append("validation_file revision must match skill.md revision")
        errs.extend(
            validation_errors(fm, validation, validation_base)
        )
        if validation.get("schema_version") == "expert-skill-validation-v2":
            reference_subject = (
                (validation.get("reference_evidence") or {}).get("subject") or {}
            )
            pins = contract.get("pins") if isinstance(contract, dict) else {}
            for evidence_key, pin_key in (
                ("revision", "reference_revision"),
                ("tree_sha256", "reference_tree_sha256"),
                ("baseline_tree_sha256", "baseline_tree_sha256"),
            ):
                if str(reference_subject.get(evidence_key) or "") != str(
                    (pins or {}).get(pin_key) or ""
                ):
                    errs.append(
                        f"v2 reference subject {evidence_key} must match contract pin {pin_key}"
                    )
            constraint_subject = (
                (validation.get("constraint_validation") or {}).get("subject") or {}
            )
            if str(constraint_subject.get("skill_revision") or "") != str(
                fm.get("revision") or ""
            ):
                errs.append("v2 constraint subject skill_revision must match skill.md")
            for evidence_key, component_key in (
                ("contract_sha256", "contract_file"),
                ("planner_extension_sha256", "planner_extension_file"),
            ):
                label = (
                    "contract" if component_key == "contract_file"
                    else "planner_extension"
                )
                component, component_path = load_component(
                    skill_path, fm, component_key, label
                )
                digest = (
                    _canonical_value_sha256(component)
                    if label in embedded_component_names(fm)
                    else _canonical_component_sha256(component_path)
                )
                if str(constraint_subject.get(evidence_key) or "") != digest:
                    errs.append(
                        f"v2 constraint subject {evidence_key} must match current component"
                    )
            checker_path = os.path.join(
                GEAK, "kernel_workflow", "tools", "expert_skill_contract.py"
            )
            if str(constraint_subject.get("checker_sha256") or "") != (
                _canonical_component_sha256(checker_path)
            ):
                errs.append(
                    "v2 constraint subject checker_sha256 must match current checker"
                )
    return errs


def _section_filled(body, sec):
    m = re.search(rf"## {re.escape(sec)}\n(.*?)(?=\n## |\Z)", body, re.S)
    if not m:
        return False
    content = re.sub(r"<!--.*?-->", "", m.group(1), flags=re.S).strip()
    return len(content) > 0


def emit_plan(skill_path, skill_id, fm, args):
    identity = skill_bundle_identity(skill_path, fm)
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
        match = fm.get("match") or {}
        dst = match.get("to_backend") or ""
        srcs = match.get("from_backend") or []
        srcs = [srcs] if isinstance(srcs, str) else list(srcs)
        cross_backend_srcs = [src for src in srcs if src and dst and src != dst]
        print("# EFFICACY (kernel_workflow, isolated A/B vs the immutable oracle):")
        print(f"Workflow scriptPath={GEAK}/kernel_workflow/kernel_workflow.js args:")
        extras = [
            f"expert_skill_id={skill_id}",
            f"expert_skill_revision={fm.get('revision') or ''}",
            f"expert_skill_bundle_sha256={identity['bundle_sha256']}",
            f"expert_skill_validation_status={identity['validation_status']}",
        ]
        if identity["validation_status"] == "experimental":
            extras.extend([
                "mode=mega",
                "expert_skill_usage=authoring",
                "mega_structural_only=true",
            ])
        for key in (
            "playbook_file", "planner_extension_file", "contract_file", "validation_file",
            "runtime_validation_file",
        ):
            if fm.get(key):
                arg = {
                    "runtime_validation_file": "graph_contract_tool",
                }.get(key, f"expert_skill_{key.replace('_file', '')}")
                extras.append(
                    f"{arg}={os.path.join(SKILLS_DIR, skill_id, str(fm[key]))}"
                )
        embedded = embedded_component_names(fm)
        if embedded:
            extras.extend([
                f"expert_skill_playbook={skill_path}",
                f"expert_skill_planner_extension={skill_path}",
                f"expert_skill_contract={skill_path}",
                f"expert_skill_validation={skill_path}",
            ])
        if "runtime_validation" in embedded:
            extras.append(
                "graph_contract_tool="
                + os.path.join(
                    GEAK, "kernel_workflow", "tools", "expert_skill_runtime.py"
                )
            )
        if identity["planner_extension_sha256"]:
            extras.extend([
                "require_expert_skill_bundle_identity=true",
                "expert_skill_planner_extension_sha256=" +
                identity["planner_extension_sha256"],
            ])
        if identity["contract_sha256"]:
            extras.append(
                "expert_skill_contract_sha256=" + identity["contract_sha256"]
            )
        print(f"  kernel_path=<OP_TASK_DIR> workflow_dir={GEAK}/kernel_workflow "
              f"use_expert_skills=true {' '.join(extras)}")
        if identity["validation_status"] != "experimental":
            # kernel_workflow reads target_language ONLY on the mode=author branch, so a port has to
            # ask for that branch or the run silently measures the untouched source language.
            if cross_backend_srcs:
                print("  # Cross-backend author arm:")
                print(f"  mode=author target_language={dst}"
                      f"   # port from {'|'.join(cross_backend_srcs)};"
                      " target_language is inert without mode=author")
            if not cross_backend_srcs or dst in srcs:
                if cross_backend_srcs:
                    print("  # Existing target-backend arm (measure separately):")
                print("  mode=optimize   # in-place optimization; no target_language needed")
        print(f"  task='reproduce expert_skill:{skill_id}; beat oracle, hold parity'")
        print("# DO-NO-HARM (control op that does NOT match the selector must stay within noise band):")
        print("  kernel_path=<CONTROL_OP_TASK_DIR> use_expert_skills=true  # expect no regression")
    if identity["validation_status"] == "experimental":
        print(
            "\nAfter structural authoring, reuse the exact identity/path arguments "
            "with mode=mega, expert_skill_usage=candidate_validation, "
            "mega_structural_only=false, require_expert_skill_contract=true, and "
            "the required runtime gates. Do not use --record for a v2 transfer."
        )
    else:
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

    validation_path, validation = load_validation(path, fm)
    if (
        fm.get("validation_schema") == "expert-skill-validation-v2"
        or validation.get("schema_version") == "expert-skill-validation-v2"
    ):
        sys.exit(
            "ERROR: --record cannot validate a v2 transfer; promote only from an "
            "exact-candidate finalist manifest"
        )
    validation["status"] = "validated" if ok else "failed"
    validation["last_verified"] = args.date or ""
    validation["gpu"] = args.gpu or ""
    validation["model"] = args.model or ""
    validation["measured"] = {
        "isolated": args.isolated if args.isolated is not None else "",
        "e2e_pct": args.e2e_pct if args.e2e_pct is not None else "",
        "parity": args.parity or "",
    }
    validation["artifact"] = args.artifact
    if validation_path:
        validation.setdefault("schema_version", "expert-skill-validation-v1")
        validation.setdefault("skill_id", fm.get("id"))
        validation.setdefault("revision", fm.get("revision", "v1"))
        with open(validation_path, "w") as f:
            yaml.safe_dump(validation, f, sort_keys=False, allow_unicode=True, width=100)
    else:
        fm["validation"] = validation
        with open(path, "w") as f:
            f.write("---\n")
            f.write(yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=100))
            f.write("---\n")
            f.write(body)
    print(f"recorded: status={validation['status']}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    os.system(f"python3 {os.path.join(HERE, 'scaffold.py')} --reindex >/dev/null 2>&1")
    if not ok:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("skill_id")
    p.add_argument("--static", action="store_true")
    p.add_argument("--emit-plan", action="store_true")
    p.add_argument("--emit-bundle", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--artifact", default=""); p.add_argument("--gpu", default="")
    p.add_argument("--model", default=""); p.add_argument("--date", default="")
    p.add_argument("--isolated", type=float, default=None)
    p.add_argument("--e2e-pct", dest="e2e_pct", type=float, default=None)
    p.add_argument("--parity", default="")
    a = p.parse_args()
    path, fm, body, txt = load(a.skill_id)

    if a.emit_bundle:
        errs = static_check(path, fm, body)
        if errs:
            print("STATIC FAIL:")
            for error in errs:
                print("  -", error)
            sys.exit(1)
        print(json.dumps(skill_bundle_identity(path, fm), sort_keys=True, indent=2))
        return
    if a.emit_plan:
        return emit_plan(path, a.skill_id, fm, a)
    if a.record:
        return record(path, fm, body, txt, a)
    # default / --static
    errs = static_check(path, fm, body)
    if errs:
        print("STATIC FAIL:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"STATIC OK: {a.skill_id} (scope={fm.get('scope')}, operator={fm['match']['operator']}). "
          f"Run with --emit-plan to get the on-box measurement command.")


if __name__ == "__main__":
    main()
