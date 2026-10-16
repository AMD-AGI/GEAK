// Trusted host adapter. The supplied backend owns one admitted model lifecycle,
// native tool pumping, authenticated conversation identity and positive cleanup.
import { readFile, realpath, open } from 'node:fs/promises';
import { join, isAbsolute } from 'node:path';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { createRuntime } from '../../../interface/runtime/engine/run_workflow.mjs';
import { makeHostBackend } from './host_backend.mjs';

const here = new URL('.', import.meta.url);
const need = (value, message) => { if (!value) throw new Error(message); };
const sha = (value) => createHash('sha256').update(value).digest('hex');

export async function runOneEngineer(input, backend) {
  need(input?.protocol === 'geak-one-engineer-v1', 'Wrong one-engineer input');
  need(backend && typeof backend.runAgent === 'function', 'Admitted backend required');
  need(['claude', 'codex'].includes(backend.name), 'Explicit admitted workflow backend required');
  const provenance = JSON.parse(await readFile(new URL('entry-provenance.json', here), 'utf8'));
  const entry = new URL('one_engineer.js', here);
  need(sha(await readFile(entry)) === provenance.entry_sha256, 'Generated entry changed');
  need(isAbsolute(input.workflow_dir) && await realpath(input.workflow_dir) === input.workflow_dir,
    'Canonical amended workflow reference directory required');
  for (const row of provenance.common_reference_overrides) {
    need(sha(await readFile(join(input.workflow_dir, row.path))) === row.sha256, 'Loaded amended role/knowledge reference changed');
  }
  let calls = 0;
  let transportResult = null;
  const runtime = createRuntime({
    backend: {
      name: backend.name,
      async runAgent(call) {
        need(++calls === 1, 'An investigation entry may invoke exactly one engineer');
        transportResult = await backend.runAgent({ ...call, cwd: input.workspace,
          investigation: {
            idea_id: input.idea_id, patch_base_source_id: input.patch_base_source_id,
            starting_source_id: input.starting_source_id,
            initial_workspace: input.initial_workspace, workspace: input.workspace,
            output_dir: input.output_dir, source_paths: input.source_paths,
            continuation: input.continuation,
          } });
        need(transportResult && transportResult.settled === true && typeof transportResult.text === 'string',
          'Missing settled actual engineer response');
        need((transportResult.driver_exit_code === undefined || transportResult.driver_exit_code === 0)
          && !transportResult.driver_signal, 'Host driver did not exit normally');
        return transportResult;
      },
    },
    concurrency: 1,
    schemaRetries: 0,
    // Native runtime passes this bound to the host backend; the backend must
    // also enforce the original study/idea deadlines and remaining allowance.
    agentTimeoutMs: input.timeout_ms,
    log: () => {},
  });
  need(Number.isSafeInteger(input.timeout_ms) && input.timeout_ms > 0, 'Explicit admitted work bound required');
  let result;
  try {
    result = await runtime.runScript(fileURLToPath(entry), { ...input, runtime_schema_retries: 0 }, 0);
  } catch (error) {
    // A malformed native return must not discard a saved best patch or its
    // settled transport/native evidence. Harvest remains a separate host step.
    if (calls === 1 && transportResult) {
      return { status: 'native_return_unavailable', native_result: null,
        transport_result: transportResult, error: String(error),
        entry_sha256: provenance.entry_sha256, model_invocations: calls };
    }
    throw error;
  }
  need(calls === 1 && runtime.state.spawned === 1, 'Not exactly one native engineer call');
  return { status: 'native_return_parsed', native_result: result, transport_result: transportResult,
    entry_sha256: provenance.entry_sha256, model_invocations: calls };
}

// Launch only with a separately prepared, reviewed host binding. The Python host
// owns admission, original clocks, evaluator construction, tools and settlement.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const options = process.argv.slice(2);
  need(options.length === 6 && options[0] === '--input' && options[2] === '--backend' && options[4] === '--output',
    'Usage: node run_one_engineer.mjs --input ABS_JSON --backend ABS_JSON --output NEW_ABS_JSON');
  const input = JSON.parse(await readFile(options[1], 'utf8'));
  const binding = JSON.parse(await readFile(options[3], 'utf8'));
  need(options[5].startsWith('/'), 'Absolute result path required');
  const output = await open(options[5], 'wx', 0o600);
  try {
    const result = await runOneEngineer(input, await makeHostBackend(binding, input));
    await output.writeFile(JSON.stringify(result, null, 2) + '\n');
    await output.sync();
    process.stdout.write(JSON.stringify({ status: result.status, output: options[5], model_invocations: result.model_invocations }) + '\n');
  } finally { await output.close(); }
}
