import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, writeFile, cp, symlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { test } from 'node:test';
import { makeHostBackend } from '../../geak_evo/engineer/js/host_backend.mjs';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const source = join(repo, 'geak_evo/engineer/js');
const sha = bytes => createHash('sha256').update(bytes).digest('hex');

test('host binding keeps the existing strict Responses route unchanged', async () => {
  const root = await mkdtemp(join(tmpdir(), 'geak-host-route-'));
  try {
    const route = { schema: 'geak-codex-responses-route-v2', backend: 'codex', wire_api: 'responses',
      requested_model: 'gpt-6-astra', requested_reasoning_effort: 'xhigh',
      base_url: 'https://synthetic.invalid/v1', api_key_env: 'SYNTHETIC_API_KEY',
      header_env: { 'X-Synthetic': 'SYNTHETIC_HEADER' }, reported_deployment_model: 'gpt-6-astra',
      qualification_ref: { path: '/synthetic/qualification', sha256: '0'.repeat(64) },
      retry_policy: 'codex_defaults' };
    async function save(name, value) {
      const raw = JSON.stringify(value);
      const path = join(root, name);
      await writeFile(path, raw);
      return { path, sha256: sha(raw) };
    }
    const routeRef = await save('route.json', route);
    const binding = { driver_ref: await save('driver.json', { synthetic: true }),
      config_ref: await save('config.json', { backend_config: { model_config: {
        workflow_backend: 'codex', model_route_ref: routeRef } } }) };
    const input = { turn_id: 'turn-1', workspace: root, timeout_ms: 1000 };
    const backend = await makeHostBackend(binding, input);
    assert.equal(backend.name, 'codex');
    assert.deepEqual(JSON.parse(await readFile(routeRef.path, 'utf8')), route);
    await assert.rejects(backend.runAgent({ cwd: root, timeoutMs: 1000, model: 'another-model' }),
      /changes the admitted model/);
    await assert.rejects(backend.runAgent({ cwd: root, timeoutMs: 1000 }), /cannot replay/);
    const tampered = await makeHostBackend(binding, input);
    await writeFile(routeRef.path, JSON.stringify({ ...route, requested_reasoning_effort: 'high' }));
    await assert.rejects(tampered.runAgent({ cwd: root, timeoutMs: 1000 }), /Pinned host file changed/);
    for (const [name, swapped, backendName] of [
      ['codex', { ...route, backend: 'claude' }, 'claude'],
      ['claude', { schema: 'geak-native-model-route-v1', workflow_backend: 'codex',
        requested_model: 'gpt-6-astra', requested_reasoning_effort: 'xhigh' }, 'codex'],
    ]) {
      const ref = await save(name + '-swapped-route.json', swapped);
      const config = await save(name + '-swapped-config.json', { backend_config: { model_config: {
        workflow_backend: backendName, model_route_ref: ref } } });
      await assert.rejects(makeHostBackend({ ...binding, config_ref: config }, input),
        /Explicit admitted model and reasoning effort/);
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});

// Exercise the real entry and runtime with a small, explicitly pinned workflow.
// Only the backend is fake; it cannot launch a process or contact a model.
test('one engineer call retains its result and never retries a bad return', async () => {
  const root = await mkdtemp(join(tmpdir(), 'geak-engineer-entry-'));
  try {
    const entry = join(root, 'geak_evo/engineer/js');
    await mkdir(entry, { recursive: true });
    await symlink(join(repo, 'interface'), join(root, 'interface'));
    for (const name of ['run_one_engineer.mjs', 'host_backend.mjs', 'one_engineer.js']) {
      await cp(join(source, name), join(entry, name));
    }
    const workflow = join(root, 'workflow');
    await mkdir(join(workflow, 'roles'), { recursive: true });
    const role = 'Synthetic role for an offline entry test.\n';
    await writeFile(join(workflow, 'roles/engineer.md'), role);
    await writeFile(join(entry, 'entry-provenance.json'), JSON.stringify({
      entry_sha256: sha(await readFile(join(entry, 'one_engineer.js'))),
      common_reference_overrides: [{ path: 'roles/engineer.md', sha256: sha(role) }],
    }));
    const { runOneEngineer } = await import(pathToFileURL(join(entry, 'run_one_engineer.mjs')));
    const input = {
      protocol: 'geak-one-engineer-v1', idea_id: 'idea-1', workspace: root,
      initial_workspace: root, output_dir: root, workflow_dir: workflow,
      commandment: 'Preserve correctness.', codebase_context: 'Offline fixture.',
      idea_contract_path: join(root, 'idea.json'), tool_addendum: 'No tools.',
      host_workspace_prepared: true, patch_base_source_id: 'a'.repeat(64),
      starting_source_id: 'b'.repeat(64), parents: [], source_paths: ['kernel.py'],
      continuation: { mode: 'fresh' }, direction: {
        specialty: 'compute', title: 'Offline test', prompt: 'Inspect the source.',
      }, timeout_ms: 1200,
    };
    const valid = { status: 'complete', strategy: 'No change.', speedup_geomean: 1,
      measurement_valid: false };
    for (const [name, transport, expected] of [
      ['valid', { settled: true, text: JSON.stringify(valid) }, 'native_return_parsed'],
      ['malformed', { settled: true, text: 'invalid JSON' }, 'native_return_unavailable'],
      ['unsettled', { settled: false, text: JSON.stringify(valid) }, 'native_return_unavailable'],
      ['driver failed', { settled: true, text: JSON.stringify(valid), driver_exit_code: 1 },
        'native_return_unavailable'],
    ]) {
      let calls = 0;
      const result = await runOneEngineer(input, { name: 'codex', async runAgent(call) {
        calls++;
        assert.equal(call.cwd, root);
        assert.equal(call.timeoutMs, input.timeout_ms);
        assert.equal(call.investigation.starting_source_id, input.starting_source_id);
        return transport;
      } });
      assert.equal(calls, 1, name);
      assert.equal(result.model_invocations, 1, name);
      assert.equal(result.status, expected, name);
      assert.deepEqual(result.transport_result, transport, name);
      if (expected === 'native_return_unavailable') assert.equal(result.native_result, null);
      else assert.deepEqual(result.native_result.eng, valid);
    }
    const prompts = [];
    for (const instruction of [undefined, 'off', 'before-trial-v1']) {
      await runOneEngineer({ ...input, report_update_instruction: instruction }, {
        name: 'codex', async runAgent(call) {
          prompts.push(call.prompt);
          return { settled: true, text: JSON.stringify(valid) };
        },
      });
    }
    assert.equal(prompts[0], prompts[1], 'disabled report guidance adds no prompt text');
    const reportInstruction = `Before each trial, update ${input.output_dir}/report.md with the change, why it may help, the observation that would support it, relevant source/result references, and the next unresolved test. Keep it concise; use a temporary file and rename. Missing notes do not block evaluation.\n`;
    assert.ok(prompts[2].includes(reportInstruction));
    assert.equal(prompts[2].replace(reportInstruction, ''), prompts[0]);
    const parent = { candidate_id: 'c'.repeat(64), source_id: 'd'.repeat(64), role: 'working_parent',
      source_files: [], reviewed_outcomes: [], evidence: [], current_claim_context: { claim_states: [] } };
    const hub = { ...parent, reviewed_outcomes: Array.from({ length: 80 }, (_, i) =>
      ({ attempt_id: `attempt-${i}`, review: `Review ${i}: ` + 'evidence '.repeat(200) })) };
    for (const parents of [[parent], [hub, { ...parent, candidate_id: 'e'.repeat(64),
      source_id: 'f'.repeat(64), role: 'agent_selected_semantic_donor' }]]) {
      const raw = JSON.stringify(parents);
      const ref = { path: join(root, 'PARENTS.json'), sha256: sha(raw) };
      await writeFile(ref.path, raw);
      let prompt;
      await runOneEngineer({ ...input, parents, insights: [ref] }, {
        name: 'codex', async runAgent(call) {
          prompt = call.prompt;
          return { settled: true, text: JSON.stringify(valid) };
        },
      });
      const start = prompt.indexOf('Parent implementation/evidence packets');
      const text = prompt.slice(prompt.indexOf('\n', start) + 1,
        prompt.indexOf('\n\n## Admitted local tools', start));
      assert.ok(Buffer.byteLength(text) <= 24 * 1024);
      const delivered = JSON.parse(text);
      assert.deepEqual(delivered.map(p => p.candidate_id), parents.map(p => p.candidate_id));
      if (parents.length === 1) assert.deepEqual(delivered, parents);
      else {
        assert.deepEqual(delivered[1], parents[1], 'small donor stays complete');
        assert.ok(delivered[0].reviewed_outcomes.length > 0);
        assert.equal(delivered[0].reviewed_outcomes.at(-1).attempt_id, 'attempt-79');
        assert.deepEqual(delivered[0].inline_cap.full_packets_ref, ref);
        assert.equal(delivered[0].inline_cap.full_packet, 'PARENTS.json[0]');
      }
      assert.equal(sha(await readFile(ref.path)), ref.sha256);
      assert.deepEqual(JSON.parse(await readFile(ref.path, 'utf8')), parents);
    }
    let calls = 0;
    await assert.rejects(runOneEngineer(input, { name: 'codex', async runAgent() {
      calls++;
      throw new Error('backend stopped');
    } }), /backend stopped/);
    assert.equal(calls, 1);

    await writeFile(join(workflow, 'roles/engineer.md'), 'Changed after pinning.');
    calls = 0;
    await assert.rejects(runOneEngineer(input, { name: 'codex', async runAgent() { calls++; } }),
      /Loaded amended role\/knowledge reference changed/);
    assert.equal(calls, 0, 'changed reference must fail before a model call');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
