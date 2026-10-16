// ---------------------------------------------------------------------------
// WARM-START (the experience KB). Before the optimize loop, search the machine-produced KB —
// addressed by canonical id on both planes — for the top-3 best patches for THIS (kernel, language,
// gfx), validate each through the SAME verify_engineer gate, and adopt the first that passes; after
// write this run's own win back.
//   on (default)      | read + validate top-3, ADOPT the first that passes.
//   reference         | read top-3 as prose only, never auto-apply.
//   return_after_read | adopt then RETURN before the optimize loop.
//   off/false/none, a STATE_DIR resume, or no arch => cold start (byte-identical to pre-feature).
const WARM_START = String(A.warm_start != null ? A.warm_start : 'on').trim().toLowerCase() || 'on';
let WARM_START_ON = WARM_START !== 'off' && WARM_START !== 'false' && WARM_START !== 'none';
if (SKILLS_ISOLATE) {
  KERNEL_KNOWLEDGE_DIR = '';
  USE_LEARNED_READ = false;
  WARM_START_ON = false;
  USE_EXPERT_SKILLS = false;
}
const WARM_START_REF_ONLY = WARM_START === 'reference';
const WARM_START_RETURN_AFTER = WARM_START === 'return_after_read';
const KB_ARTIFACTS_DIR = String(A.kb_artifacts_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/kb_artifacts' : '')).replace(/\/+$/, '');
const EXPERIENCE_STORE = `${WORKFLOW_DIR}/scripts/experience_store.py`;
// Every candidate costs a full on-box verify to reject, so a recorded near-tie is not worth reading.
const WARM_START_MIN_SPEEDUP = Number.isFinite(parseFloat(A.warm_start_min_speedup))
  ? parseFloat(A.warm_start_min_speedup) : 1.05;
// How hard the resolver may work to map this run's kernel name onto a stored page. The name is
// layout-derived (`fused_moe_kernel` standalone, `fused_moe_kernel_task` from an e2e head extraction),
// so `exact` alone would make the head path miss its own history; `fuzzy` also accepts an op_kind.
const WARM_START_MATCH = ['exact', 'normalized', 'fuzzy'].includes(String(A.warm_start_match || '').trim())
  ? String(A.warm_start_match).trim() : 'fuzzy';
// Which PLANE the experience is READ from. Same phases, same schemas, same verify gate either way
// — only the command string differs, because the store subcommands were built to print the same
// JSON as the directory ones.
//   store (default)  a KB Store on disk, in the shape the service uses, keyed by canonical id.
//   local            the kb_artifacts/ tree, keyed by slug. The original scheme, kept as an escape
//                    hatch for a box whose store is unusable.
//
// `store` is the default because it is the SAME addressing the service uses. Two schemes side by
// side drifted — `resolve` and `resolve-remote` assembled their filter chains separately, which is
// how the keyed read lost the `--include-retired` the slug read always had. Nothing moves: the tree
// is still written (`write-remote --plane both`) and `sync-local` carries the backlog across before
// each read; the tree stays the assembly buffer and the surface a curation pass edits.
const KB_MODE = String(A.kb_mode || 'store').trim().toLowerCase() === 'local' ? 'local' : 'store';
const KB_STORE_DIR = String(A.kb_store_dir ||
  (KB_ARTIFACTS_DIR ? KB_ARTIFACTS_DIR.replace(/\/[^/]*$/, '') + '/kb_store_local' : '')).replace(/\/+$/, '');
// The key carries a rocm <major>.<minor>; on a box without /opt/rocm the measured stack is empty and
// every record would file under `unspecified`, which splits one kernel's history across two keys.
const KB_FRAMEWORK_VERSION = String(A.kb_framework_version || '').trim();
const KB_VERSION_FLAG = KB_FRAMEWORK_VERSION ? ` --framework-version ${JSON.stringify(KB_FRAMEWORK_VERSION)}` : '';
const KB_ROOT_OK = KB_MODE === 'store' ? !!KB_STORE_DIR : !!KB_ARTIFACTS_DIR;
// Whether this run may also talk to the KB Store SERVICE, on top of whichever local plane KB_MODE
// selected. `auto` (default) uses it when credentials are present and silently does not when they
// are not; `off` restores the directory-only behaviour byte for byte.
const KB_REMOTE = String(A.kb_remote || 'auto').trim().toLowerCase() === 'off' ? 'off' : 'auto';
// The credentials are NOT visible from this process. They live in the user's profile, and the
// shells these commands run in are non-interactive, so `process.env.KB_STORE_TOKEN` is empty here
// even on a box where the service is configured. Deciding the plane in JS would therefore mean
// deciding it wrong. Instead every emitted command exports its own credentials from the profile's
// two sources and then branches on what it actually got — the test happens where the answer is
// knowable. The token is read from a 0600 file into a variable, never passed in argv, because
// `ps` is world-readable on this box.
// The gateway's internal AMD CA is not in a stock container trust store, so a KB command run inside
// one fails TLS (the old workaround was `curl -k`). DETECT then heal: only when the caller has NOT
// already established trust (SSL_CERT_FILE unset) do we point urllib/requests/curl/node at the first
// readable AMD-root bundle we find. Most callers here run OUTSIDE the warm-start launcher (which sets
// these at `docker run`), so this self-heal is what keeps their KB reachable. Path-only, no CA
// content; overridable with KB_CA_BUNDLE; a no-op when SSL_CERT_FILE is already set or no bundle is
// readable (so CI and already-trusting images are byte-identical). DNS (the host has none
// in-container) is a launch concern, handled with `docker run --add-host`.
// GEAK_KB_STORE_* wins over the bare KB_STORE_* (so a host that also runs Hyperloom, which carries
// its own KB_STORE_*, can point GEAK at a distinct store); re-exported under the bare names every
// downstream reader here branches on.
const KB_ENV_PRELUDE =
  'export KB_STORE_URL="${GEAK_KB_STORE_URL:-${KB_STORE_URL:-https://global.primus-safe.amd.com/knowledge-base}}"; ' +
  'export KB_STORE_TOKEN="${GEAK_KB_STORE_TOKEN:-${KB_STORE_TOKEN:-$(cat ~/.geak_kb_token 2>/dev/null)}}"; ' +
  'if [ -z "${SSL_CERT_FILE:-}" ]; then for _ca in "${KB_CA_BUNDLE:-}" ' +
  '/shared_nfs/hyperloom/ca/amd-ca-combined.pem "$HOME/amd-extra-ca-bundle.pem"; do ' +
  '[ -n "$_ca" ] && [ -r "$_ca" ] && { export SSL_CERT_FILE="$_ca" REQUESTS_CA_BUNDLE="$_ca" ' +
  'CURL_CA_BUNDLE="$_ca" NODE_EXTRA_CA_CERTS="$_ca"; break; }; done; fi; ';
// Writing in store mode records BOTH planes in one call, so it needs both roots: the directory tree
// stays the source of truth a curation pass edits, and the store is derived from it.
const KB_WRITE_OK = KB_ROOT_OK && !!KB_ARTIFACTS_DIR;
const kebab = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);

