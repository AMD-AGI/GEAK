import test from 'node:test';
import assert from 'node:assert/strict';
import {
  asmDirectionViolation,
  constrainLanguageDirections,
  constrainSpecialists,
  validateLanguagePatchScope,
} from '../lib/language_guard.mjs';

test('ASM specialist dispatch never includes host_runtime', () => {
  const specialists = constrainSpecialists(
    ['host_runtime', 'memory', 'compute', 'host_runtime'], 'asm');
  assert.equal(specialists.length, 4);
  assert.equal(specialists.includes('host_runtime'), false);
});

test('ASM directions drop source and host-runtime work then backfill safely', () => {
  const guarded = constrainLanguageDirections([
    {
      name: 'graph replay',
      specialty: 'host_runtime',
      prompt: 'Add HIP graph capture in the Python wrapper.',
    },
    {
      name: 'source rewrite',
      specialty: 'algorithm',
      prompt: 'Make a source-level HIP change.',
      focus_files: ['kernel.cpp'],
    },
    {
      name: 'vmem schedule',
      specialty: 'memory',
      prompt: 'Reorder VMEM instructions in kernel.s.',
      focus_files: ['kernel.s'],
    },
  ], { languageMode: 'asm', count: 3, step: 5 });

  assert.equal(guarded.directions.length, 3);
  assert.equal(guarded.directions[0].name, 'vmem schedule');
  assert.ok(guarded.repairs.some((repair) => repair.includes('graph replay')));
  assert.ok(guarded.repairs.some((repair) => repair.includes('source rewrite')));
  for (const direction of guarded.directions) {
    assert.equal(asmDirectionViolation(direction), '');
    assert.notEqual(direction.specialty, 'host_runtime');
  }
});

test('non-ASM directions are unchanged', () => {
  const input = [{ name: 'wrapper', specialty: 'host_runtime', prompt: 'graph replay' }];
  const guarded = constrainLanguageDirections(input, {
    languageMode: 'source', count: 3,
  });
  assert.deepEqual(guarded.directions, input);
  assert.deepEqual(guarded.repairs, []);
});

test('ASM patch scope permits only kernel.s and window.s', () => {
  const allowed = validateLanguagePatchScope([
    '--- a/kernel.s',
    '+++ b/kernel.s',
    '--- a/subdir/window.s',
    '+++ b/subdir/window.s',
  ].join('\n'), 'asm');
  assert.equal(allowed.ok, true);

  const mixed = validateLanguagePatchScope([
    '--- a/kernel.s',
    '+++ b/kernel.s',
    '--- a/launcher.cpp',
    '+++ b/launcher.cpp',
  ].join('\n'), 'asm');
  assert.equal(mixed.ok, false);
  assert.deepEqual(mixed.violations, ['launcher.cpp']);

  assert.equal(validateLanguagePatchScope('binary patch', 'asm').ok, false);
});
