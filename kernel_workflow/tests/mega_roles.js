// What a mode=mega agent reads for a role: the shared role file with its mode=mega fragment applied
// (roles/_fragments/<role>_mega.md: each `##` section replaces the same-named section of the role file,
// new sections are added at the end, and "Opening (mode=mega)" replaces the text above the first `##`).
// The mega planner is tech_lead's analyze/plan_round and the MoE analysis is profile_engineer's
// analyze_profile, so the old mega role names map to those fragments.
'use strict';
const fs = require('fs');
const path = require('path');

const ROLES = path.join(__dirname, '..', 'roles');
const read = (p) => fs.readFileSync(p, 'utf8');
const fragment = (role) => path.join(ROLES, '_fragments', `${role}_mega.md`);

// [[heading or null, text]] split on `## ` lines outside code fences.
function sections(text) {
  const out = [];
  let head = null;
  let cur = [];
  let fence = false;
  for (const line of text.split(/(?<=\n)/)) {
    if (line.startsWith('```')) fence = !fence;
    if (!fence && line.startsWith('## ')) {
      out.push([head, cur.join('')]);
      head = line.trim();
      cur = [];
    }
    cur.push(line);
  }
  out.push([head, cur.join('')]);
  return out;
}

function applied(role) {
  const shared = sections(read(path.join(ROLES, `${role}.md`)));
  if (!fs.existsSync(fragment(role))) return shared.map(([, t]) => t).join('');
  const frag = sections(read(fragment(role))).filter(([h]) => h !== null);   // drop the fragment's own header
  const byHead = new Map(frag);
  const used = new Set();
  const merged = shared.map(([h, t]) => {
    const key = h === null ? '## Opening (mode=mega)' : h;
    if (byHead.has(key)) { used.add(key); return byHead.get(key); }
    return t;
  });
  for (const [h, t] of frag) if (!used.has(h)) merged.push(t);
  return merged.join('');
}

module.exports = (name) => {
  const role = String(name).replace(/\.md$/, '');
  if (role === 'mega_search_lead') return read(fragment('tech_lead'));
  if (role === 'analysis_engineer') return read(fragment('profile_engineer'));
  return applied(role);
};
module.exports.fragment = fragment;
module.exports.sections = sections;
