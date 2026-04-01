#!/bin/bash
# SSL Certificate Diagnostic Script
# Run this on the problematic pod to collect all SSL-related info

OUTPUT="/shared_nfs/xiaofei/ssl_diag.log"
echo "=== SSL Diagnostic Report ===" > "$OUTPUT"
echo "Date: $(date)" >> "$OUTPUT"
echo "Hostname: $(hostname)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# 1. Check if cert file exists and its content
echo "=== 1. Cert file check ===" >> "$OUTPUT"
ls -la /shared_nfs/xiaofei/amd-issuing-ca.crt >> "$OUTPUT" 2>&1
file /shared_nfs/xiaofei/amd-issuing-ca.crt >> "$OUTPUT" 2>&1
openssl x509 -in /shared_nfs/xiaofei/amd-issuing-ca.crt -noout -subject -issuer -dates >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 2. Check system CA store
echo "=== 2. System CA store ===" >> "$OUTPUT"
which update-ca-certificates >> "$OUTPUT" 2>&1
echo "exit: $?" >> "$OUTPUT"
ls -la /usr/local/share/ca-certificates/ >> "$OUTPUT" 2>&1
ls -la /etc/ssl/certs/ca-certificates.crt >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 3. Try copying cert and updating CA store
echo "=== 3. Copy cert and update-ca-certificates ===" >> "$OUTPUT"
cp /shared_nfs/xiaofei/amd-issuing-ca.crt /usr/local/share/ca-certificates/ 2>&1 | tee -a "$OUTPUT"
update-ca-certificates >> "$OUTPUT" 2>&1
echo "update-ca-certificates exit: $?" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# 4. Verify cert is in system bundle
echo "=== 4. Check cert in system bundle ===" >> "$OUTPUT"
grep -c "AMD" /etc/ssl/certs/ca-certificates.crt >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 5. Test system-level SSL (curl)
echo "=== 5. curl test to LLM gateway ===" >> "$OUTPUT"
curl -v --max-time 10 https://project1.tw325.primus-safe.amd.com/llm-gateway/v1/models >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"
echo "curl exit: $?" >> "$OUTPUT"
echo "" >> "$OUTPUT"

# 6. Python SSL info
echo "=== 6. Python SSL paths ===" >> "$OUTPUT"
python3 -c "
import ssl
print('openssl version:', ssl.OPENSSL_VERSION)
paths = ssl.get_default_verify_paths()
print('cafile:', paths.cafile)
print('capath:', paths.capath)
print('openssl_cafile:', paths.openssl_cafile)
print('openssl_capath:', paths.openssl_capath)
print('openssl_cafile_env:', paths.openssl_cafile_env)
print('openssl_capath_env:', paths.openssl_capath_env)
" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 7. Check certifi
echo "=== 7. Certifi info ===" >> "$OUTPUT"
python3 -c "
import certifi
bundle = certifi.where()
print('certifi bundle:', bundle)
import os
print('bundle size:', os.path.getsize(bundle))
# Check if AMD cert is already in certifi bundle
with open(bundle) as f:
    content = f.read()
    print('AMD cert in certifi:', 'AMD' in content)
" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 8. Append cert to certifi and test
echo "=== 8. Append cert to certifi bundle ===" >> "$OUTPUT"
python3 -c "
import certifi
bundle = certifi.where()
print('certifi bundle:', bundle)
with open('/shared_nfs/xiaofei/amd-issuing-ca.crt') as f:
    cert_content = f.read()
with open(bundle, 'a') as f:
    f.write('\n')
    f.write(cert_content)
print('Appended cert to certifi bundle')
# Verify
with open(bundle) as f:
    content = f.read()
    print('AMD cert in certifi now:', 'AMD' in content)
print('new bundle size:', len(content))
" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 9. Test Python SSL after certifi patch
echo "=== 9. Python SSL test after certifi patch ===" >> "$OUTPUT"
python3 -c "
import ssl
import urllib.request
try:
    r = urllib.request.urlopen('https://project1.tw325.primus-safe.amd.com/llm-gateway/v1/models', timeout=10)
    print('urllib OK, status:', r.status)
except Exception as e:
    print('urllib FAILED:', e)
" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 10. Test httpx after certifi patch
echo "=== 10. httpx test after certifi patch ===" >> "$OUTPUT"
python3 -c "
import httpx
try:
    r = httpx.get('https://project1.tw325.primus-safe.amd.com/llm-gateway/v1/models', timeout=10)
    print('httpx OK, status:', r.status_code)
except Exception as e:
    print('httpx FAILED:', type(e).__name__, e)
" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 11. Test httpx with explicit verify=False (control test)
echo "=== 11. httpx verify=False (control test) ===" >> "$OUTPUT"
python3 -c "
import httpx
try:
    r = httpx.get('https://project1.tw325.primus-safe.amd.com/llm-gateway/v1/models', timeout=10, verify=False)
    print('httpx verify=False OK, status:', r.status_code)
except Exception as e:
    print('httpx verify=False FAILED:', type(e).__name__, e)
" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 12. Test httpx with explicit system CA bundle
echo "=== 12. httpx with system CA bundle ===" >> "$OUTPUT"
python3 -c "
import httpx
try:
    r = httpx.get('https://project1.tw325.primus-safe.amd.com/llm-gateway/v1/models', timeout=10, verify='/etc/ssl/certs/ca-certificates.crt')
    print('httpx system-ca OK, status:', r.status_code)
except Exception as e:
    print('httpx system-ca FAILED:', type(e).__name__, e)
" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 13. Test with SSL_CERT_FILE env var
echo "=== 13. httpx with SSL_CERT_FILE env ===" >> "$OUTPUT"
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt python3 -c "
import httpx
try:
    r = httpx.get('https://project1.tw325.primus-safe.amd.com/llm-gateway/v1/models', timeout=10)
    print('httpx SSL_CERT_FILE OK, status:', r.status_code)
except Exception as e:
    print('httpx SSL_CERT_FILE FAILED:', type(e).__name__, e)
" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 14. Check httpx SSL context details
echo "=== 14. httpx SSL context internals ===" >> "$OUTPUT"
python3 -c "
import httpx._config
import ssl
import certifi
ctx = ssl.create_default_context()
print('default context ca stats:', ctx.get_ca_certs().__len__(), 'certs')
ctx2 = ssl.create_default_context(cafile=certifi.where())
print('certifi context ca stats:', ctx2.get_ca_certs().__len__(), 'certs')
# Check what httpx actually uses
print('httpx version:', httpx.__version__)
" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

# 15. pip/certifi versions
echo "=== 15. Package versions ===" >> "$OUTPUT"
pip list 2>/dev/null | grep -iE "certifi|httpx|httpcore|openai|litellm|ssl" >> "$OUTPUT" 2>&1
echo "" >> "$OUTPUT"

echo "=== DONE ===" >> "$OUTPUT"
echo "Results written to $OUTPUT"
