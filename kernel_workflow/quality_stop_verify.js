// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// This self-contained block is copied into kernel_lane.js. Native Workflow
// exposes BigInt but no imports, WebCrypto, TextEncoder, Buffer, or atob.
// SHA-256 follows FIPS 180-4 sections 4.1.2, 4.2.2, 5.1.1, and 6.2.2.
// RSA verification follows RFC 8017 sections 8.2.2 and 9.2 (SHA-256 only).
// No private-key operation occurs in the Workflow VM.
// BEGIN QUALITY STOP VERIFIER
function qualitySha256Ascii(text) {
  if (typeof text !== 'string' || text.length > 1048576 || /[^\x00-\x7f]/.test(text)) throw new Error('invalid_signed_payload');
  const bytes = Array.from(text, c => c.charCodeAt(0));
  const length = bytes.length * 8;
  bytes.push(128);
  while (bytes.length % 64 !== 56) bytes.push(0);
  bytes.push(0, 0, 0, 0, length >>> 24, (length >>> 16) & 255, (length >>> 8) & 255, length & 255);
  const constants = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
  let state = [0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
  const rotate = (x, n) => (x >>> n) | (x << (32 - n));
  for (let offset = 0; offset < bytes.length; offset += 64) {
    const words = [];
    for (let i = 0; i < 16; i++) {
      const j = offset + 4 * i;
      words[i] = ((bytes[j] << 24) | (bytes[j + 1] << 16) | (bytes[j + 2] << 8) | bytes[j + 3]) >>> 0;
    }
    for (let i = 16; i < 64; i++) {
      const x = words[i - 15], y = words[i - 2];
      words[i] = (words[i - 16] + (rotate(x, 7) ^ rotate(x, 18) ^ (x >>> 3)) + words[i - 7]
        + (rotate(y, 17) ^ rotate(y, 19) ^ (y >>> 10))) >>> 0;
    }
    let [a,b,c,d,e,f,g,h] = state;
    for (let i = 0; i < 64; i++) {
      const first = (h + (rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25)) + ((e & f) ^ (~e & g))
        + constants[i] + words[i]) >>> 0;
      const second = ((rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      h=g; g=f; f=e; e=(d+first)>>>0; d=c; c=b; b=a; a=(first+second)>>>0;
    }
    state = state.map((value, i) => (value + [a,b,c,d,e,f,g,h][i]) >>> 0);
  }
  return state.map(value => value.toString(16).padStart(8, '0')).join('');
}

function qualityBase64(text, url) {
  if (typeof text !== 'string') throw new Error('invalid_signature_encoding');
  if (url) {
    if (!/^[A-Za-z0-9_-]+$/.test(text) || text.length % 4 === 1) throw new Error('invalid_public_key_encoding');
    text = text.replace(/-/g, '+').replace(/_/g, '/');
    while (text.length % 4) text += '=';
  }
  if (!/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(text)) throw new Error('invalid_signature_encoding');
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let accumulator = 0, bits = 0;
  const result = [];
  for (const character of text.replace(/=+$/, '')) {
    accumulator = (accumulator << 6) | alphabet.indexOf(character);
    bits += 6;
    if (bits >= 8) { bits -= 8; result.push((accumulator >>> bits) & 255); }
  }
  if ((accumulator & ((1 << bits) - 1)) !== 0) throw new Error('noncanonical_signature_encoding');
  return result;
}

function qualityVerify(envelope, config, task) {
  try {
    if (!envelope || Object.keys(envelope).sort().join(',') !== 'payload,signature') return null;
    const key = config.public_key;
    if (!key || key.kty !== 'RSA' || key.alg !== 'RS256' || key.e !== 'AQAB') return null;
    const modulus = qualityBase64(key.n, true), signature = qualityBase64(envelope.signature, false);
    if (modulus.length !== 256 || signature.length !== 256 || modulus[0] < 128) return null;
    const integer = bytes => BigInt('0x' + bytes.map(value => value.toString(16).padStart(2, '0')).join(''));
    const n = integer(modulus);
    let base = integer(signature), exponent = 65537n, result = 1n;
    if (base >= n) return null;
    while (exponent > 0n) {
      if (exponent & 1n) result = (result * base) % n;
      base = (base * base) % n;
      exponent >>= 1n;
    }
    const digestInfo = '3031300d060960864801650304020105000420' + qualitySha256Ascii(envelope.payload);
    const expected = '0001' + 'ff'.repeat(256 - 3 - digestInfo.length / 2) + '00' + digestInfo;
    if (result.toString(16).padStart(512, '0') !== expected) return null;
    const value = JSON.parse(envelope.payload);
    if (value.protocol !== 'geak-fixed-floor-stop-v2' || value.task !== task
        || typeof value.certified !== 'boolean' || typeof value.qualifying !== 'boolean'
        || typeof value.issued_at !== 'number' || !Number.isFinite(value.issued_at)
        || !value.native_binding || Object.keys(value.native_binding).sort().join(',')
          !== 'agent_id,root_task,root_tool,run_id,session_id') return null;
    return value;
  } catch (_) { return null; }
}
// END QUALITY STOP VERIFIER
