import ssl, certifi, pathlib

for f in pathlib.Path("/usr/local/share/ca-certificates").glob("*.crt"):
    open(certifi.where(), "a").write(f.read_text())

_orig = ssl.create_default_context
def _patched(*a, **k):
    ctx = _orig(*a, **k)
    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    return ctx
ssl.create_default_context = _patched
