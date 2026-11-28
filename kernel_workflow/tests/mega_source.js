// The mode=mega backend section of kernel_lane.js (its args, helpers, run state and phase hooks). The mega
// suites read this, not the whole file, so a pattern can only match mega code; the lane phases that call
// the hooks are in the rest of the file (`.full()`).
'use strict';
const fs = require('fs');
const path = require('path');

const full = () => fs.readFileSync(path.join(__dirname, '..', 'kernel_lane.js'), 'utf8');
// The mega schemas use shorthands (STR, NUM, ...); the suites match the literal schema text, so the
// shorthands are expanded back here.
const SHORTHANDS = [
  ['ANYOBJS', "{ type: 'array', items: { type: 'object', additionalProperties: true } }"],
  ['ANYOBJ', "{ type: 'object', additionalProperties: true }"],
  ['STRS', "{ type: 'array', items: { type: 'string' } }"],
  ['STR_N', "{ type: ['string', 'null'] }"], ['NUM_N', "{ type: ['number', 'null'] }"],
  ['STR', "{ type: 'string' }"], ['NUM', "{ type: 'number' }"], ['BOOL', "{ type: 'boolean' }"],
];
const expand = (text) => SHORTHANDS.reduce(
  (t, [k, v]) => t.replace(new RegExp(`(?<=[:,\\[(] *)\\b${k}\\b(?! *=)`, 'g'), v), text);
const raw = () => {
  const src = full();
  const i = src.indexOf('\n// mode=mega backend:');
  const j = src.indexOf('\n// end of mode=mega backend\n');
  if (i < 0 || j < i) throw new Error('mode=mega backend section not found in kernel_lane.js');
  return src.slice(i + 1, j);
};
module.exports = () => expand(raw());
module.exports.raw = raw;
module.exports.full = full;
