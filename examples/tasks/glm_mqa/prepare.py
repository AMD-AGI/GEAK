"""Stage only ORIGINAL MQA and the unchanged evaluator; no optimized input."""

import argparse
import json
from pathlib import Path

from geak_evo._util import canonical_json, sha256_hex
from geak_evo.evaluate.external import project_external_task, validate_external_task
from geak_evo.evaluate.jobs import _read

ORIGINAL = "baseline_src/fp8_mqa_logits_seam.py"
CANDIDATE = "kernel_src/fp8_mqa_logits_seam.py"
PINS = {
    ORIGINAL: "60bef8f5845f2eeb3d248dfd193e7fc892746fb13bad3f18c2acb1b8dd621c5e",
    "meta.json": "f3cd3635cfe3ddb2f26f03e4625ae445cbf1e3fcd4d40a1fc419ba4b208e5bd3",
    "ut_main.py": "a51d6e50555595fc5899f510a0a26926d811e820141d090741ed47776a5db44f",
    "harness_lib.py": "22f86cf37876e216b37726884fbe6bca37692fca5dad825c9dfb37e06b183c56",
    "weight_lib.py": "e249a9429e9c9066e88565f0022bffb4a37ffc88c17061a7d3869b062f638e36",
}


def prepare(source, output, image, config_digest):
    contents = {name: _read(source / name) for name in PINS}
    if {name: sha256_hex(raw) for name, raw in contents.items()} != PINS:
        raise ValueError("ORIGINAL task or protected evaluator differs from pinned source")
    contents[CANDIDATE] = contents[ORIGINAL]
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in contents.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    meta = json.loads(contents["meta.json"])
    contract = project_external_task(meta,
        source={"repository": "external:agent-kernel-arena/share/exp_1013",
                "revision": sha256_hex(canonical_json(PINS)), "task_selector": "glm52_dsa_gluon_mqa_logits_task"},
        runtime={"image": image, "config_digest": config_digest},
        artifacts={name: sha256_hex(raw) for name, raw in contents.items()},
        original_source_map={CANDIDATE: ORIGINAL}, primary_regime="prefill_extend")
    validate_external_task(contract, output)
    (output / "task.json").write_bytes(canonical_json(contract)+b"\n")
    (output / "objective.json").write_bytes(canonical_json({"kind": "external_primary", "primary_regime": "prefill_extend"})+b"\n")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config-digest", required=True)
    args = parser.parse_args()
    prepare(args.source, args.out, args.image, args.config_digest)


if __name__ == "__main__":
    main()
