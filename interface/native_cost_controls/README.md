# Explicit registered-prefix cache wrapper

This opt-in Linux utility wraps an existing noninteractive command and adds one
five-minute cache marker when an Anthropic Messages request begins with an
explicitly registered tool prefix. It preserves the other request-body bytes,
command arguments and provider responses. Unsupported request layouts pass
through with their original body bytes. It adds no provider retries.

Nothing enables the wrapper automatically. Run it from a GEAK checkout:

```bash
python -m interface.native_cost_controls.run_registered \
  --catalog /absolute/path/current-tools.json EXPECTED_SHA256 \
  -- python interface/run_e2e.py handoff.json result.json
```

Replace `EXPECTED_SHA256` with the catalog's lowercase SHA-256 digest and use the
normal arguments for the wrapped command. Multiple `--catalog PATH SHA256` pairs
are allowed; the longest matching registered prefix controls the decision.
Only `ANTHROPIC_BASE_URL` changes in the child environment. The existing GEAK SDK
session settings, session recording and workflow remain in their normal runner.
No helper emulation or alternate kernel runner is installed.

## Catalog contract

A catalog is a UTF-8 JSON array containing the exact ordered leading client-tool
definitions, including their complete schemas. It must contain distinct names,
no cache markers and no `StructuredOutput` entry. A nonempty catalog and its
external expected hash are mandatory. Invalid or changed catalog bytes prevent
the child from starting. The policy never learns a catalog from the first
provider request.

Produce and verify the catalog for the actual CLI/SDK, entrypoint, source
revision, tool filters and permissions before using it. Record those identities
with the catalog. A hash proves which supplied bytes were used; it does not prove
that they came from the intended workflow. This contribution ships no captured
profile or catalog generator, and qualifies no historical catalog for current
GEAK. Missing or changed prefixes simply receive no additional cache marker.

The conservative request guard declines existing tool markers, four or more
existing markers, unsupported TTL/layout/output-format controls, server-side
compaction and unknown content-block types. These requests retain their original
body bytes. Provider caching eligibility also depends on the request's earlier
content and the provider's policy; marker insertion alone proves neither a cache
hit nor billed savings.

## Runtime boundary

Use Python 3.10 or newer on Linux with `/proc/self/task/<pid>/children`, pidfds
and child-subreaper support. Some Linux kernels omit the proc children interface;
the wrapper rejects those hosts before launching a command.
The wrapper must begin as its own process with no existing children or extra
threads and the default `SIGCHLD` handler. It is for noninteractive commands that
honor `ANTHROPIC_BASE_URL`; terminal job control is unsupported. It forwards
termination signals and terminates/reaps its owned remaining descendants,
including detached sessions, before closing the proxy.

Only direct HTTP(S) Messages transport using system TLS trust is supported.
Alternate provider modes, HTTP proxy variables, custom CA/certificate settings
and Unix-socket transport are rejected before child launch. Do not remove
required environment settings to force this utility to run; retain the original
command when its transport is outside this boundary. The loopback listener and
child command must share a network namespace. Wrapping a command that delegates
provider requests to remote or isolated workers is not qualified. This wrapper
does not provide container isolation or control an existing Ray cluster.

The proxy does not log request/response bodies, credentials or command arguments.
Its in-memory decisions contain only marker-applied booleans and fixed reason
codes. The original child retains its own normal output and logging behavior.

## Validation scope

The accompanying tests use synthetic bytes, local loopback HTTP and owned child
processes. Lifecycle tests declare the proc-children requirement and skip on
unsupported kernels; a skipped lifecycle test is not runtime qualification.
Run the extracted suites with:

```bash
python -m pytest -q \
  interface/test_native_prefix_marker.py \
  interface/test_registered_native_cache.py \
  interface/test_cache_proxy.py \
  interface/test_run_registered_cache.py
```

They cover byte-preserving insertion and rejection, registered-prefix identity,
HTTP forwarding and streaming, and wrapper lifecycle/error paths. The narrow
port retains the corresponding runtime from GEAK #489 without its SDK/helper
bundle. These tests do not establish current native catalog compatibility,
completed workflow performance, GPU correctness or actual provider savings.
