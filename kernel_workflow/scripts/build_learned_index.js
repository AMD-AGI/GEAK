#!/usr/bin/env node
/**
 * build_learned_index.js — regenerate knowledge/learned/INDEX.md FROM the cards.
 *
 * INDEX.md is a GENERATED file: every line is derived from one card's discovery frontmatter
 * (`name`, `description`, `keywords`, `kernels`, `platforms`, `kernel_class`, `regime`, `confidence`).
 * Nobody hand-edits it — parallel lanes used to race on an append-to-INDEX step and silently drop each
 * other's line, whereas a regen republishes whatever cards are on disk.
 *
 * SINK-AGNOSTIC: the directory is an argument, so one generator serves every `learned/` folder in the
 * repo. Referenced in place, never copied. Pure node (fs/path only) — no deps, no network, no GPU.
 *
 *   node build_learned_index.js                                          # this workflow's learned/
 *   node build_learned_index.js e2e_workflow/knowledge/learned           # any other sink
 *   node build_learned_index.js <learned_dir> --check                    # exit 1 if INDEX.md is stale
 */
'use strict';
const fs = require('fs');
const path = require('path');

const CAP = 40;                       // README's budget: INDEX.md holds at most this many card lines.
const SKIP = new Set(['INDEX.md', 'README.md', '_archive.md']);

// Cards that describe a cross-cutting technique rather than one op sort last — a reader scanning for
// their own kernel_class should hit the op groups first.
const LAST_GROUPS = ['method', 'other'];

// The sink's owning workflow ("<workflow>/knowledge/learned"), so an e2e index does not tell the
// reader to rebuild the kernel one.
const owner = (dir) => {
  const parts = path.resolve(dir).split(path.sep);
  const i = parts.lastIndexOf('knowledge');
  return i > 0 ? parts[i - 1] : parts[parts.length - 1];
};
const regenCmd = (dir) => 'node kernel_workflow/scripts/build_learned_index.js' +
  (owner(dir) === 'kernel_workflow' ? '' : ` ${owner(dir)}/knowledge/learned`);

const HEADER = (dir) => `# Learned — index of distilled ${owner(dir)} experience cards

<!-- GENERATED FILE — do not hand-edit. Regenerate with:
       ${regenCmd(dir)}
     Every line below is derived from one card's discovery frontmatter. To change a line, edit the
     card's \`description\`/\`keywords\`/\`confidence\` and regenerate. -->

Open the cards matching your run as **additional, advisory priors** — they only ADD candidate levers to
try, never remove any and never replace measurement. The frozen-baseline isolated A/B + oracle parity is
always the judge (see \`README.md\`). **Cap: ≤${CAP} card lines.** Confidence (a hint strength, not
authority): ★ noise/unverified · ★★ single non-overlap or ≥2 consistent · ★★★ ≥2 non-overlap.

Effects are **ratios or percent deltas only, never wall-clock or absolute throughput** — those vary box
to box and stay in the run's \`EVAL_DIR\` (see \`README.md\` → "Content rules").

**How to use this file: READ it, then open the 0–3 cards that look relevant.** Each line carries the
card's own description, the kernel symbols it was measured on, and its keywords — enough to judge
relevance without opening anything. Match on *meaning*, not on an exact string: a card written for
\`split-k on skinny-M GEMM\` is worth opening for a tall-K GEMM too. If nothing matches, that is a real
answer — plan cold, exactly as this workflow does without any KB.

(Every line here is derived from a card's discovery header, so a card is still self-describing if you
open it directly. A \`grep\` for an exact kernel symbol works as a shortcut, but it is not the lookup
path — it matches strings, and the thing you are looking for is a *concept*.)
`;

function parseFrontmatter(text) {
  if (!text.startsWith('---')) return null;
  const end = text.indexOf('\n---', 3);
  if (end === -1) return null;
  const out = {};
  for (const raw of text.slice(3, end).split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const i = line.indexOf(':');
    if (i === -1) continue;
    const k = line.slice(0, i).trim();
    let v = line.slice(i + 1).trim();
    if (v.startsWith('[') && v.endsWith(']')) {
      out[k] = v.slice(1, -1).split(',').map(s => s.trim().replace(/^["']|["']$/g, '')).filter(Boolean);
    } else {
      out[k] = v.replace(/^["']|["']$/g, '');
    }
  }
  return out;
}

function stars(c) { const m = String(c || '').match(/★+/); return m ? m[0] : '★'; }

// Keyword hygiene against vocabulary drift ("split-k" / "split_k" / "splitk" / "Split K"), which stops
// sibling cards from clustering. normalize() fixes mechanical spelling; fold() is a stricter key used
// ONLY to DETECT near-duplicates that survived it — never to auto-merge, since "gemm"/"gemms" are the
// same word but "mfma"/"mfmas" might not be.
const normalize = (s) => String(s).toLowerCase().trim()
  .replace(/[\s_]+/g, '-').replace(/-{2,}/g, '-').replace(/^-|-$/g, '');
const fold = (s) => normalize(s).replace(/-/g, '').replace(/s$/, '');

function collect(dir) {
  return fs.readdirSync(dir)
    .filter(f => f.endsWith('.md') && !SKIP.has(f))
    .sort()
    .map((f) => {
      const fm = parseFrontmatter(fs.readFileSync(path.join(dir, f), 'utf8'));
      if (!fm) return null;   // no frontmatter = not indexable; skipped, not guessed at
      const arr = (v) => (Array.isArray(v) ? v : (v ? [String(v)] : []));
      return {
        file: f,
        name: fm.name || f.replace(/\.md$/, ''),
        description: fm.description || fm.effect || '(no description)',
        keywords: [...new Set(arr(fm.keywords).map(normalize).filter(Boolean))],
        kernels: arr(fm.kernels),
        platforms: arr(fm.platforms),
        kernel_class: fm.kernel_class || 'other',
        regime: fm.regime || '',
        confidence: stars(fm.confidence),
        lifecycle: fm.lifecycle || 'active',
      };
    })
    .filter(Boolean)
    // An archived/retired card keeps its file (the refuting source matters) but leaves the index.
    .filter(c => c.lifecycle === 'active');
}

function render(cards, dir) {
  const groups = new Map();
  for (const c of cards) {
    if (!groups.has(c.kernel_class)) groups.set(c.kernel_class, []);
    groups.get(c.kernel_class).push(c);
  }
  const names = [...groups.keys()].sort((a, b) => {
    const ra = LAST_GROUPS.indexOf(a), rb = LAST_GROUPS.indexOf(b);
    if (ra !== rb) return (ra === -1 ? -1 : ra) - (rb === -1 ? -1 : rb);
    return a.localeCompare(b);
  });

  let body = '';
  for (const g of names) {
    const rows = groups.get(g).sort((a, b) =>
      (b.confidence.length - a.confidence.length) || a.name.localeCompare(b.name));
    body += `\n## ${g}\n`;
    for (const c of rows) {
      const scope = [c.platforms.join('/'), c.regime].filter(Boolean).join(' · ');
      body += `- ${scope ? `[${scope}] ` : ''}${c.description} ${c.confidence} — (${c.file})\n`;
      const tags = [
        c.kernels.length ? `kernels: ${c.kernels.join(', ')}` : '',
        c.keywords.length ? `kw: ${c.keywords.join(', ')}` : '',
      ].filter(Boolean).join(' · ');
      if (tags) body += `  - ${tags}\n`;
    }
  }
  if (!cards.length) body = '\n## (no cards yet)\n';

  // Vocabulary appendix: the controlled word list a curator picks from before coining a synonym.
  const counts = new Map();
  for (const c of cards) for (const k of c.keywords) counts.set(k, (counts.get(k) || 0) + 1);
  const vocab = [...counts.entries()].sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]));

  if (vocab.length) {
    body += '\n## keyword vocabulary (generated — REUSE these before coining a new term)\n' +
      vocab.map(([k, n]) => `${k}${n > 1 ? `(${n})` : ''}`).join(' · ') + '\n';

    const byFold = new Map();
    for (const [k] of vocab) {
      const f = fold(k);
      if (!byFold.has(f)) byFold.set(f, []);
      byFold.get(f).push(k);
    }
    const dupes = [...byFold.values()].filter(v => v.length > 1);
    if (dupes.length) {
      body += '\n> ⚠ **Near-duplicate keywords** — same concept, different spelling. Pick one, edit the\n' +
        '> cards, regenerate:\n' +
        dupes.map(v => `> - ${v.join(' / ')}\n`).join('');
    }
  }

  let footer = '';
  if (cards.length > CAP) {
    footer = `\n> ⚠ **Over the ${CAP}-card cap (${cards.length}).** Evict the lowest \`confidence ×\n` +
      `> freshness\` card to \`_archive.md\` (★★★ is never auto-evicted), then regenerate.\n`;
  }
  return HEADER(dir) + body + footer;
}

function main(argv) {
  const args = argv.filter(a => a !== '--check');
  const check = argv.includes('--check');
  const dir = path.resolve(args[0] || path.join(__dirname, '..', 'knowledge', 'learned'));
  if (!fs.existsSync(dir)) { console.error(`no such learned dir: ${dir}`); return 2; }

  const cards = collect(dir);
  const next = render(cards, dir);
  const out = path.join(dir, 'INDEX.md');
  const prev = fs.existsSync(out) ? fs.readFileSync(out, 'utf8') : '';

  if (check) {
    if (prev === next) { console.log(`INDEX.md up to date (${cards.length} cards).`); return 0; }
    console.error('INDEX.md is stale — run: node kernel_workflow/scripts/build_learned_index.js');
    return 1;
  }
  if (prev !== next) fs.writeFileSync(out, next);
  console.log(`INDEX.md ${prev === next ? 'unchanged' : 'written'} (${cards.length} cards).`);
  return 0;
}

if (require.main === module) process.exit(main(process.argv.slice(2)));
module.exports = { parseFrontmatter, collect, render, main };
