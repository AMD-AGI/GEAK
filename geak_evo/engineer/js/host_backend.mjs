// One subprocess call into the admitted existing Python host; no timeout race.
import { readFile, realpath, lstat } from 'node:fs/promises';
import { isAbsolute } from 'node:path';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';

const need = (value, message) => { if (!value) throw new Error(message); };
async function checked(ref) {
  need(ref && isAbsolute(ref.path) && /^[0-9a-f]{64}$/.test(ref.sha256), 'Pinned absolute host file required');
  need(await realpath(ref.path) === ref.path && (await lstat(ref.path)).isFile(), 'Canonical host file required');
  const raw = await readFile(ref.path);
  need(createHash('sha256').update(raw).digest('hex') === ref.sha256, 'Pinned host file changed');
  return raw;
}

function sanitizedHostError(value) {
  const fields = ['schema', 'status', 'error_kind', 'receipt_ref', 'text',
    'retry_permitted', 'execution_may_have_started'];
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || Object.keys(value).length !== fields.length
      || !fields.every((field) => Object.hasOwn(value, field))
      || value.schema !== 'geak-idea-engineer-host-error-v1'
      || typeof value.error_kind !== 'string' || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(value.error_kind)
      || value.receipt_ref !== null || value.text !== null || value.retry_permitted !== false
      || !((value.status === 'preflight_failed' && value.execution_may_have_started === false)
        || (value.status === 'receipt_unavailable' && value.execution_may_have_started === true))) return null;
  return { schema: value.schema, status: value.status, error_kind: value.error_kind,
    receipt_ref: value.receipt_ref, text: value.text, retry_permitted: value.retry_permitted,
    execution_may_have_started: value.execution_may_have_started };
}

export async function makeHostBackend(binding, input) {
  need(input && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$/.test(input.turn_id), 'Root-bound turn ID required');
  const config = JSON.parse(await checked(binding.config_ref));
  const model = config.backend_config?.model_config;
  need(model && ['claude', 'codex'].includes(model.workflow_backend),
    'Explicit admitted workflow backend required');
  const route = JSON.parse(await checked(model.model_route_ref));
  const routeBackend = route.schema === 'geak-codex-responses-route-v2' ? route.backend
    : route.schema === 'geak-native-model-route-v1' ? route.workflow_backend : null;
  need(typeof route.requested_model === 'string' && route.requested_model.length > 0
    && typeof route.requested_reasoning_effort === 'string' && route.requested_reasoning_effort.length > 0
    && routeBackend === model.workflow_backend
    && (route.schema !== 'geak-codex-responses-route-v2' || routeBackend === 'codex')
    && (route.schema !== 'geak-native-model-route-v1' || routeBackend === 'claude')
    && (route.schema !== 'geak-codex-responses-route-v2' || !Object.hasOwn(route, 'workflow_backend'))
    && (route.schema !== 'geak-native-model-route-v1' || !Object.hasOwn(route, 'backend')),
    'Explicit admitted model and reasoning effort required');
  let used = false;
  return { name: model.workflow_backend, async runAgent(call) {
    need(!used, 'Host mapping cannot replay an engineer invocation');
    used = true;
    await checked(binding.driver_ref);
    await checked(binding.config_ref);
    await checked(model.model_route_ref);
    need(call.cwd === input.workspace && call.timeoutMs === input.timeout_ms,
      'Native call differs from bound workspace/work allowance');
    need(call.model == null || call.model === route.requested_model, 'Native call changes the admitted model');
    const request = { schema: 'geak-standalone-model-request-v1', invocation_id: input.turn_id,
      prompt: call.prompt, label: call.label, cwd: call.cwd, model: route.requested_model,
      timeout_ms: call.timeoutMs };
    const result = await new Promise((resolve, reject) => {
      const child = spawn('python3', ['-I', '-B', binding.driver_ref.path, '--config', binding.config_ref.path,
        '--reviewed-config-sha256', binding.config_ref.sha256],
        { cwd: input.workspace, env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1' },
          stdio: ['pipe', 'pipe', 'pipe'], shell: false });
      const stdout = [], stderr = [];
      let inputError = null;
      child.stdout.on('data', (chunk) => stdout.push(chunk));
      child.stderr.on('data', (chunk) => stderr.push(chunk));
      child.stdin.on('error', (error) => { inputError = String(error); });
      child.on('error', reject);
      child.on('close', (code, signal) => {
        const raw = Buffer.concat(stdout).toString('utf8');
        let value;
        try { value = JSON.parse(raw); }
        catch { reject(new Error('Host driver returned no valid result; inspect its durable attempt receipt before any further invocation')); return; }
        if (value?.schema !== 'geak-standalone-model-result-v1' || value.invocation_id !== input.turn_id || !value.receipt_ref) {
          const hostError = sanitizedHostError(value);
          // Existing outer stderr retains this caller-correlated failure, never a receipt.
          if (hostError) process.stderr.write(JSON.stringify({ caller_invocation_id: input.turn_id, host_error: hostError }) + '\n');
          reject(new Error('Host driver result identity differs; no retry is permitted')); return;
        }
        resolve({ ...value, driver_exit_code: code, driver_signal: signal,
          driver_stderr: Buffer.concat(stderr).toString('utf8'), driver_input_error: inputError });
      });
      child.stdin.end(JSON.stringify(request));
    });
    // Receipt-bearing failures are retained by runOneEngineer for later harvest;
    // only the Python host may decide whether CPU/native cleanup is settled.
    return result;
  } };
}
