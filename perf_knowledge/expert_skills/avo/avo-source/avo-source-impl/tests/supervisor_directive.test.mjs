import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeSupervisorDirective } from '../lib/supervisor_directive.mjs';

test('fills missing prompt/expected on directions', () => {
  const { directive, repairs } = normalizeSupervisorDirective({
    diagnosis: 'Stall on P1',
    mark_failed: [],
    new_directions: [
      { name: 'coalesce', next: true },
      { name: 'tune', expected: '1.05x' },
    ],
  }, { step: 7, failed: [] });
  assert.equal(directive.new_directions.length, 2);
  assert.match(directive.new_directions[0].prompt, /concrete change/i);
  assert.ok(directive.new_directions[0].expected.length > 0);
  assert.match(directive.new_directions[1].prompt, /1\.05x|Deliver/i);
  assert.ok(repairs.length >= 1);
});

test('null raw yields synthetic directive', () => {
  const { directive } = normalizeSupervisorDirective(null, { step: 3 });
  assert.ok(directive.diagnosis);
  assert.equal(directive.new_directions.length, 1);
  assert.ok(directive.new_directions[0].name.includes('escape'));
});

test('converge only when strictly true', () => {
  const a = normalizeSupervisorDirective({ diagnosis: 'x', new_directions: [{ name: 'a', prompt: 'p', expected: 'e' }], converge: 'yes' }, { step: 1 });
  assert.equal(a.directive.converge, false);
  const b = normalizeSupervisorDirective({ diagnosis: 'x', new_directions: [{ name: 'a', prompt: 'p', expected: 'e' }], converge: true }, { step: 1 });
  assert.equal(b.directive.converge, true);
});
