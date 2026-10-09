#!/usr/bin/env python3
"""Tiny CORS-enabled static server for Perfetto trace files.

Binds 127.0.0.1 only (reach it from a laptop via `ssh -L 8099:127.0.0.1:8099 user@host`).
CORS + OPTIONS-preflight headers let https://ui.perfetto.dev fetch a trace via ?url=,
though the robust path is: open http://localhost:8099/ -> download -> drag into
ui.perfetto.dev (avoids the HTTPS->HTTP mixed-content block).

GEAK shared kernel tool (kernel_workflow/scripts/kernel_tools/); the Gluon pack's scripts/ copy is a
shim. A developer convenience: it serves files and never touches the GPU.

Usage: python3 serve_traces.py [PORT] [DIRECTORY]
       python3 serve_traces.py --selftest
"""
import functools
import http.server
import os
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(200)
        self.send_header("Allow", "GET, HEAD, OPTIONS")
        self.end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


def make_server(port, directory):
    return http.server.ThreadingHTTPServer(
        ("127.0.0.1", port), functools.partial(Handler, directory=directory))


def _selftest():
    import tempfile
    import threading
    import urllib.request
    ok = True
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "t.json"), "w") as fh:
            fh.write('{"traceEvents": []}')
        httpd = make_server(0, td)                       # ephemeral port, loopback only
        Handler.log_message = lambda *a, **k: None       # quiet
        th = threading.Thread(target=httpd.serve_forever, daemon=True)
        th.start()
        try:
            port = httpd.server_address[1]
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/t.json", timeout=10) as r:
                ok &= r.status == 200 and b"traceEvents" in r.read()
                ok &= r.headers.get("Access-Control-Allow-Origin") == "*"
            req = urllib.request.Request(f"http://127.0.0.1:{port}/t.json", method="OPTIONS")
            with urllib.request.urlopen(req, timeout=10) as r:
                ok &= r.status == 200 and "OPTIONS" in (r.headers.get("Allow") or "")
            ok &= httpd.server_address[0] == "127.0.0.1"
        finally:
            httpd.shutdown()
            httpd.server_close()
    print("[serve_traces] SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    if "--selftest" in argv:
        return _selftest()
    port = int(argv[0]) if argv else 8099
    directory = os.path.abspath(argv[1] if len(argv) > 1 else ".")
    httpd = make_server(port, directory)
    print(f"serving {directory} at http://127.0.0.1:{port}/")
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
