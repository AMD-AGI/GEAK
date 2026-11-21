import test from 'node:test';
import assert from 'node:assert/strict';

import {
  isAgentTimeout, isTransportFailure, transportBackoffMs,
} from '../lib/agent_failure.mjs';

const spawnResult = (overrides = {}) => ({
  ok: false, nTools: 0, resultText: '', err: '', ...overrides,
});

test('the observed API outage is classified as transport', () => {
  // Verbatim from the run that aborted: four workers, 0 tool calls, ~190s each.
  assert.equal(isTransportFailure(spawnResult({
    resultText: 'API Error: Unable to connect to API (ConnectionRefused)',
  }), null), true);
});

test('connection and availability failures on either channel are transport', () => {
  const cases = [
    { err: 'connect ECONNREFUSED 127.0.0.1:8080', nTools: 7 },
    { err: 'Error: socket hang up', nTools: 3 },
    { resultText: 'API Error: 429 Too Many Requests' },
    { resultText: 'Error: 503 Service Unavailable' },
    { resultText: 'API Error: 529 overloaded_error' },
    { resultText: 'getaddrinfo ENOTFOUND api.example.com' },
  ];
  for (const c of cases) {
    assert.equal(isTransportFailure(spawnResult(c), null), true, JSON.stringify(c));
  }
});

test('a reachable agent that broke the output contract is not transport', () => {
  // It must get the re-prompt, not a backoff: the prompt is what needs fixing.
  assert.equal(isTransportFailure(spawnResult({
    ok: true, nTools: 12, resultText: 'I finished the work; here is a summary.',
  }), null), false);
});

test('transport strings inside a healthy session do not fake an outage', () => {
  // A healthy agent quoting a log line must not spend the backoff budget. This is
  // why the raw event stream (tool output, file contents) is never consulted.
  assert.equal(isTransportFailure(spawnResult({
    ok: true, nTools: 12,
    resultText: 'the kernel log showed 503 retries and a rate limit note',
  }), null), false);
});

test('a parsed result is never an outage, however it reads', () => {
  assert.equal(isTransportFailure(spawnResult({
    resultText: 'API Error: Unable to connect to API (ConnectionRefused)',
  }), { status: 'ok' }), false);
});

test('a silent session with no usable output is treated as transport only on evidence', () => {
  // Zero tool calls alone is not enough — an agent can legitimately answer without
  // touching a tool. It still needs a transport string to be charged as an outage.
  assert.equal(isTransportFailure(spawnResult({
    ok: true, nTools: 0, resultText: 'Nothing to change here.',
  }), null), false);
});

test('shim wall-clock timeout is classified separately from transport', () => {
  const result = spawnResult({
    nTools: 48,
    err: 'benchmark still running\n[shim] TIMEOUT',
  });
  assert.equal(isAgentTimeout(result), true);
  assert.equal(isTransportFailure(result, null), false);
});

test('ordinary contract failures are not timeouts', () => {
  assert.equal(isAgentTimeout(spawnResult({
    ok: true, nTools: 12, resultText: 'finished without JSON',
  })), false);
});

test('backoff doubles from the base and honours the cap', () => {
  assert.equal(transportBackoffMs(1, 20000, 300000), 20000);
  assert.equal(transportBackoffMs(2, 20000, 300000), 40000);
  assert.equal(transportBackoffMs(4, 20000, 300000), 160000);
  assert.equal(transportBackoffMs(5, 20000, 300000), 300000);
  assert.equal(transportBackoffMs(9, 20000, 300000), 300000);
});

test('six default retries wait out a multi-minute outage without unbounded delay', () => {
  const total = [1, 2, 3, 4, 5, 6]
    .reduce((sum, n) => sum + transportBackoffMs(n, 20000, 300000), 0);
  assert.equal(total, 20000 + 40000 + 80000 + 160000 + 300000 + 300000);
  assert.ok(total / 60000 > 10, 'should ride out at least ten minutes');
  assert.ok(total / 60000 < 20, 'but must not stall a step for a third of an hour');
});
