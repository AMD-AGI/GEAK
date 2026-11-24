import { basename } from 'node:path';

const ASM_SPECIALTIES = ['memory', 'compute', 'algorithm'];
const ASM_FALLBACKS = [
  {
    stem: 'dependency_schedule',
    specialty: 'compute',
    expected: 'reduce instruction stalls without changing semantics',
    prompt: 'Inspect kernel.s/window.s dependencies, waitcnt placement, and independent instruction ordering. Make one assembly-only scheduling change and measure it.',
  },
  {
    stem: 'register_pressure',
    specialty: 'compute',
    expected: 'reduce VGPR/SGPR pressure or spill risk',
    prompt: 'Inspect kernel.s/window.s live ranges and temporary registers. Make one assembly-only register-pressure change without changing the ABI.',
  },
  {
    stem: 'memory_pipeline',
    specialty: 'memory',
    expected: 'improve VMEM/LDS issue efficiency',
    prompt: 'Inspect kernel.s/window.s VMEM/LDS ordering, reuse, and bank behavior. Make one assembly-only memory-pipeline change.',
  },
  {
    stem: 'control_flow',
    specialty: 'algorithm',
    expected: 'remove avoidable assembly control-flow overhead',
    prompt: 'Inspect kernel.s/window.s branches, masks, and loop control. Make one assembly-only control-flow change while preserving exact semantics.',
  },
];

const ASM_FORBIDDEN_INTENT = [
  /\bhost[_ -]?runtime\b/i,
  /\bhost[- ]side\b/i,
  /\bsource[- ]level\b/i,
  /\bsource\s+(?:code|file|kernel|implementation|edit|change|rewrite)\b/i,
  /\b(?:hip|c\+\+|python)\s+(?:source|wrapper|launcher|runtime|binding)\b/i,
  /\b(?:wrapper|binding|launcher)\s+(?:edit|change|optimization|overhead)\b/i,
  /\bkernel[- ]launch\s+(?:overhead|path|collapse|fusion)\b/i,
  /\blaunch(?:er)?\s+overhead\b/i,
  /\bdispatch\s+collapse\b/i,
  /\b(?:graph|stream)\s+(?:capture|replay)\b/i,
  /\b(?:pybind|cmake|setup\.py|torch\s+extension)\b/i,
  /\.(?:cpp|cc|cxx|cu|hip|py)\b/i,
  /\bctypes\b/i,
  /\bdata_ptr\b/i,
];

function directionText(direction) {
  return [
    direction && direction.name,
    direction && direction.title,
    direction && direction.expected,
    direction && direction.prompt,
  ].filter(Boolean).join(' ');
}

export function asmDirectionViolation(direction) {
  if (!direction || typeof direction !== 'object') return 'missing direction';
  if (String(direction.specialty || '').toLowerCase() === 'host_runtime') {
    return 'host_runtime specialty is forbidden in ASM mode';
  }
  const focusFiles = Array.isArray(direction.focus_files) ? direction.focus_files : [];
  const invalidFocus = focusFiles.find(
    (path) => !['kernel.s', 'window.s'].includes(basename(String(path))));
  if (invalidFocus) return `focus file is outside ASM edit scope: ${invalidFocus}`;
  const text = directionText(direction);
  const forbidden = ASM_FORBIDDEN_INTENT.find((pattern) => pattern.test(text));
  return forbidden ? `source/host-runtime intent matched ${forbidden}` : '';
}

export function constrainSpecialists(specialists, languageMode) {
  if (String(languageMode).toLowerCase() !== 'asm') return [...(specialists || [])];
  const allowed = (specialists || []).filter(
    (specialty) => String(specialty).toLowerCase() !== 'host_runtime');
  const target = Math.max(1, (specialists || []).length);
  for (let i = 0; allowed.length < target; i++) {
    allowed.push(ASM_SPECIALTIES[i % ASM_SPECIALTIES.length]);
  }
  return allowed.slice(0, target);
}

export function constrainLanguageDirections(directions, opts = {}) {
  const languageMode = String(opts.languageMode || '').toLowerCase();
  const input = Array.isArray(directions) ? directions : [];
  if (languageMode !== 'asm') {
    return { directions: [...input], repairs: [] };
  }

  const repairs = [];
  const failed = new Set((opts.failed || []).map(String));
  const targetCount = Math.max(1, Number(opts.count) || input.length || 1);
  const accepted = [];
  const names = new Set();
  for (const direction of input) {
    const violation = asmDirectionViolation(direction);
    const name = String(direction && (direction.name || direction.title) || '').trim();
    if (violation || !name || failed.has(name) || names.has(name)) {
      repairs.push(`dropped ASM-incompatible direction "${name || '(unnamed)'}": ${
        violation || (failed.has(name) ? 'already failed' : 'missing/duplicate name')}`);
      continue;
    }
    names.add(name);
    accepted.push({ ...direction });
    if (accepted.length >= targetCount) break;
  }

  const step = Number(opts.step) || 0;
  for (let i = 0; accepted.length < targetCount; i++) {
    const fallback = ASM_FALLBACKS[(step + i) % ASM_FALLBACKS.length];
    let name = `asm_${fallback.stem}_s${step}_${i}`;
    while (names.has(name) || failed.has(name)) name += '_next';
    names.add(name);
    accepted.push({
      name,
      specialty: fallback.specialty,
      expected: fallback.expected,
      prompt: fallback.prompt,
      focus_files: ['kernel.s', 'window.s'],
    });
    repairs.push(`injected ASM-only fallback direction "${name}"`);
  }
  return { directions: accepted, repairs };
}

export function validateLanguagePatchScope(patchText, languageMode) {
  if (String(languageMode).toLowerCase() !== 'asm') {
    return { ok: true, files: [], violations: [] };
  }
  const files = new Set();
  for (const line of String(patchText || '').split(/\r?\n/)) {
    let match = line.match(/^diff --git a\/(\S+) b\/(\S+)$/);
    if (match) {
      files.add(match[1]);
      files.add(match[2]);
      continue;
    }
    match = line.match(/^(?:---|\+\+\+) (.+?)(?:\t.*)?$/);
    if (match && match[1] !== '/dev/null') {
      files.add(match[1].replace(/^[ab]\//, ''));
      continue;
    }
    match = line.match(/^rename (?:from|to) (.+)$/);
    if (match) files.add(match[1]);
  }
  const violations = [...files].filter(
    (path) => !['kernel.s', 'window.s'].includes(basename(path)));
  if (files.size === 0) violations.push('(patch has no inspectable file headers)');
  return { ok: violations.length === 0, files: [...files], violations };
}
