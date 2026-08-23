# Credentials and trust for every KB command, in ONE place. Sourced, never executed.
#
# Two programs need this: e2e_workflow.js, which emits KB commands as bash for its agents, and
# interface/run_e2e.py, which writes the record itself when the workflow died before its own writer
# ran. A second copy of these six lines in Python is a copy of the token path and the CA fallback
# list — the two things whose drift is silent (a write that 401s, or one that fails TLS) and whose
# only symptom is a run whose result never reached the KB. So: one file, sourced by both.
#
# The service token is NOT present in a non-interactive shell (it lives in ~/.bashrc, which such a
# shell never sources), so we export it here from the 0600 file. It is never passed in argv:
# /proc is world-readable on this box, and the service has no revocation story for a leaked key.
#
# The gateway's internal AMD CA is not in a stock container trust store, so a KB command run inside
# one fails TLS (the old workaround was `curl -k`). DETECT then heal: only when the caller has NOT
# already established trust (SSL_CERT_FILE unset) do we point urllib/requests/curl/node at the first
# readable AMD-root bundle we find. Path-only, no CA content; overridable with KB_CA_BUNDLE; a no-op
# when SSL_CERT_FILE is already set or no bundle is readable (so CI and already-trusting images are
# byte-identical). DNS (the host has none in-container) is a launch concern, handled with
# `docker run --add-host`.
export KB_STORE_URL="${KB_STORE_URL:-https://global.primus-safe.amd.com/knowledge-base}"
export KB_STORE_TOKEN="${KB_STORE_TOKEN:-$(cat ~/.geak_kb_token 2>/dev/null)}"
if [ -z "${SSL_CERT_FILE:-}" ]; then
  for _ca in "${KB_CA_BUNDLE:-}" /shared_nfs/hyperloom/ca/amd-ca-combined.pem \
             "$HOME/amd-extra-ca-bundle.pem"; do
    [ -n "$_ca" ] && [ -r "$_ca" ] && {
      export SSL_CERT_FILE="$_ca" REQUESTS_CA_BUNDLE="$_ca" \
             CURL_CA_BUNDLE="$_ca" NODE_EXTRA_CA_CERTS="$_ca"
      break
    }
  done
fi
