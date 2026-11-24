import assert from 'node:assert/strict';
import test from 'node:test';

import { finalAnswer } from '../lib/agent_contract.mjs';

test('a JSON object with every required key is the final answer', () => {
  assert.deepEqual(
    finalAnswer('{"status":"ok","probe":"a"}', ['status', 'probe']),
    { status: 'ok', probe: 'a' });
  assert.deepEqual(finalAnswer('  {"status":"ok"}\n', ['status']), { status: 'ok' });
});

test('a single code fence around the object is accepted', () => {
  assert.deepEqual(
    finalAnswer('```json\n{"status":"ok","probe":"a"}\n```', ['status', 'probe']),
    { status: 'ok', probe: 'a' });
  assert.deepEqual(finalAnswer('```\n{"status":"ok"}\n```', ['status']), { status: 'ok' });
});

test('progress text is not an answer', () => {
  assert.equal(finalAnswer('still waiting for the background task', ['status']), null);
  assert.equal(finalAnswer('', ['status']), null);
});

test('JSON embedded in prose is not an answer', () => {
  assert.equal(finalAnswer(
    'Draft so far: {"status":"ok","verified_geomean":1.0} - will confirm',
    ['status', 'verified_geomean']), null);
  assert.equal(finalAnswer(
    '{"status":"ok","verified_geomean":1.0}\nStill checking the last repeat.',
    ['status', 'verified_geomean']), null);
});

test('a missing required key disqualifies the object', () => {
  assert.equal(finalAnswer('{"status":"ok"}', ['status', 'verified_geomean']), null);
});

test('only a plain object qualifies', () => {
  for (const text of ['[{"status":"ok"}]', 'null', '42', '"ok"', 'true']) {
    assert.equal(finalAnswer(text, ['status']), null, text);
  }
});

test('with no required keys any JSON object qualifies', () => {
  assert.deepEqual(finalAnswer('{}', []), {});
  assert.deepEqual(finalAnswer('{"note":"done"}'), { note: 'done' });
});

test('non-string input is not an answer', () => {
  for (const value of [undefined, null, 7, { status: 'ok' }]) {
    assert.equal(finalAnswer(value, ['status']), null);
  }
});
