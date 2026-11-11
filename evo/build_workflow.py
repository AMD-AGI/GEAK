from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASE_COMMIT = "a48ac8bef9aeab299ba5ac9cb44164d997e94524"
BASE_SHA256 = "6ab6132593eb9675df7a37b62fab428f61df82247a17f97ed55652e331e78b83"
DISPATCH = """if (A.optimizer != null && typeof A.optimizer !== 'string') {
  throw new Error('optimizer must be greedy or evo (a string)');
}
const OPTIMIZER = String(A.optimizer || '').trim().toLowerCase();
if (OPTIMIZER && !['greedy', 'evo'].includes(OPTIMIZER)) {
  throw new Error('optimizer must be greedy or evo');
}
if (OPTIMIZER === 'evo') {
  const kernelDir = String(A.workflow_dir || '').replace(/\\/+$/, '');
  if (!kernelDir) throw new Error('args.workflow_dir is required');
  return await workflow({ scriptPath: `${kernelDir}/../evo/kernel_lane.js` }, A);
}
"""


def replace_once(source: str, before: str, after: str) -> str:
    count = source.count(before)
    if count != 1:
        raise ValueError(f"Expected exactly one EVO seam, found {count}: {before[:100]!r}")
    return source.replace(before, after, 1)


def build(root: Path = ROOT) -> str:
    source = (root / "kernel_workflow/kernel_lane.js").read_text()
    source = replace_once(source, DISPATCH, "")
    if hashlib.sha256(source.encode()).hexdigest() != BASE_SHA256:
        raise ValueError(f"Main lane changed from reviewed {BASE_COMMIT}; review and repin the EVO seams")
    extension = (root / "evo/extension.js").read_text()
    replacements = [
        ("name: 'kernel-lane',", "name: 'kernel-lane-evo',"),
        ("const KERNEL_PATH_ORIG = A.kernel_path;", extension + "\nconst KERNEL_PATH_ORIG = A.kernel_path;"),
        ("const ENG_SCHEMA = obj({", "PLAN_SCHEMA.properties.directions.items.properties.parent_id = { type: 'string' };\n"
         "PLAN_SCHEMA.properties.directions.items.properties.donor_ids = { type: 'array', items: { type: 'string' } };\n\n"
         "const ENG_SCHEMA = obj({"),
        ("let plannerStopReason = '';", "await evoInitialize();\n\nlet plannerStopReason = '';"),
        ("const planInputs = (forbidStop) => ({", "const planInputs = (forbidStop) => ({\n    EVOLUTION: evoPlannerInput(),"),
        ("    (d) => {\n      const isDeep", "    async (d) => {\n"
         "      try {\n        d.evo_prepared = await evoPrepare(d);\n      } catch (error) {\n"
         "        const message = String(error && error.message || error);\n"
         "        EVO_ERRORS.push({ round, operation: 'prepare', id: d.id, message });\n"
         "        return { d, eng: null, evo_prepare_failed: true };\n      }\n      const isDeep"),
        ("First create YOUR private workspace, then optimize.\n", "Your selected-parent workspace is already prepared. Do not recreate it.\n"),
        ("        SPECIALTY: d.specialty,", "        EVOLUTION: evoEngineerInput(d),\n        SPECIALTY: d.specialty,"),
        ("    (prev) => {\n      const { d, eng }", "    async (prev) => {\n      const { d, eng }"),
        ("      const { d, eng } = prev;", "      const { d, eng } = prev;\n"
         "      if (prev.evo_prepare_failed) return { d, eng, ver: null };"),
        ("      const patch = `${d.out_dir}/best_patch.diff`;", "      let patch = `${d.out_dir}/best_patch.diff`;"),
        ("      const recovered = !eng || eng.status === 'failed';", "      const custody = await evoCheckPatch(d, patch);\n"
         "      if (!custody) return { d, eng, ver: null };\n      patch = custody.patch_object_path;\n"
         "      const recovered = !eng || eng.status === 'failed';"),
        ("  const clean = results.filter(Boolean);", "  const clean = results.filter(Boolean);\n  await evoRecordBatch(clean);"),
        ("    const integPrim = integrate && integrate.best ? primSpeedup({", "    const integrateCustody = integrate && integrate.best && integrate.best.patch_file\n"
         "      ? await evoCheckPatch({ id: `r${round}_integrated` }, integrate.best.patch_file) : null;\n"
         "    const integPrim = integrate && integrate.best ? primSpeedup({"),
        ("    if (integrate && integrate.conclusion === 'improved' && integrate.best &&", "    if (integrate && integrateCustody && integrate.conclusion === 'improved' && integrate.best &&"),
        ("per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,", "per_case: integrate.best.per_case || [], patch: integrateCustody.patch_object_path,"),
        ("return {\n  mode: MODE,", "return {\n  evolution: { archive: evoArchivePath, errors: EVO_ERRORS },\n  mode: MODE,"),
    ]
    for before, after in replacements:
        source = replace_once(source, before, after)
    start = source.index("Your selected-parent workspace is already prepared. Do not recreate it.\n")
    stop = source.index("${readLine} If KK_OPERATOR", start)
    source = source[:start] + "Your selected-parent workspace is already prepared. Do not recreate it.\n" + source[stop:]
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description="Build only the opt-in EVO worker from reviewed GEAK main")
    parser.add_argument("--check", action="store_true")
    options = parser.parse_args()
    expected = build()
    target = ROOT / "evo/kernel_lane.js"
    if options.check:
        if not target.is_file() or target.read_text() != expected:
            parser.error("evo/kernel_lane.js is stale; run python3 evo/build_workflow.py")
    else:
        target.write_text(expected)


if __name__ == "__main__":
    main()
