#!/usr/bin/env python3
"""Weekly web-ingest detector (plan Part 2.4) — widens the candidate set, never judges.

Reads watchlist.yaml, detects what changed since the last run, and writes
  kb_inbox/<date>/candidates.yaml
listing every new/changed source as a CANDIDATE. It does NOT touch the KB: the kb_curator agent
(index/roles/kb_curator.md) turns candidates into a reviewed PR, and CI opens that PR for a human. By
hard rule every ingested item is `lifecycle: candidate` + `verified_on: null` until an on-box run
promotes it via _promote.py — web ingestion only enlarges the candidate pool (ADD-only; measurement is
always the judge).

Change detection is stateful: `kb_inbox/_state.json` remembers the last commit SHA / content-hash /
known-URL set per source, so each run reports only the delta. Network access and a `GITHUB_TOKEN` make
git/blog/search sources live; without them the run degrades to a clean no-op (writes an empty
candidates file with a `notes:` explaining what was skipped) so CI never hard-fails on connectivity.

Usage:
  _ingest_web.py                      # live detect, write kb_inbox/<today>/candidates.yaml
  _ingest_web.py --offline            # structural run: validate watchlist, emit empty candidates
  _ingest_web.py --date 2026-08-10    # pin the output-dir date (tests / reproducibility)
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WATCHLIST = os.path.join(HERE, "watchlist.yaml")
INBOX = os.path.join(REPO, "kb_inbox")
STATE = os.path.join(INBOX, "_state.json")

try:
    import yaml
except Exception:
    yaml = None

UA = {"User-Agent": "geak-kb-ingest/1.0"}


def load_watchlist(path):
    if not os.path.isfile(path):
        return {"sources": []}
    text = open(path, encoding="utf-8").read()
    if yaml is not None:
        try:
            d = yaml.safe_load(text) or {}
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {"sources": []}


def load_state():
    try:
        return json.loads(open(STATE, encoding="utf-8").read())
    except Exception:
        return {}


def _get(url, token=None, accept=None, timeout=20):
    req = urllib.request.Request(url, headers=dict(UA))
    if accept:
        req.add_header("Accept", accept)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - trusted watchlist URLs
        return r.read()


def _sha256(b):
    return "sha256:" + hashlib.sha256(b).hexdigest()


def detect_git(src, state, token):
    """New commits touching any watched path since the last-seen SHA (GitHub API)."""
    repo, watch = src.get("repo"), src.get("watch") or [None]
    out, newest = [], state.get("last_sha", {}) if isinstance(state.get("last_sha"), dict) else {}
    for path in watch:
        api = f"https://api.github.com/repos/{repo}/commits?per_page=10"
        if path:
            api += f"&path={path}"
        try:
            data = json.loads(_get(api, token=token, accept="application/vnd.github+json"))
        except (urllib.error.URLError, ValueError, OSError) as e:
            return out, {**state, "error": str(e)}
        if not isinstance(data, list) or not data:
            continue
        last = state.get("last_sha", {}).get(path or "") if isinstance(state.get("last_sha"), dict) else None
        for c in data:
            sha = c.get("sha")
            if sha == last:
                break
            out.append({"source": src["id"], "type": "git", "repo": repo, "path": path,
                        "ref": sha, "title": (c.get("commit", {}).get("message", "") or "").splitlines()[0][:200],
                        "url": c.get("html_url", "")})
        newest[path or ""] = data[0].get("sha")
    return out, {"last_sha": newest}


def detect_hash(src, state, token, urls):
    """URL/content-hash drift for blog_index / doc / url_set sources."""
    out, hashes = [], dict(state.get("hashes", {}))
    for url in urls:
        try:
            body = _get(url, token=token)
        except (urllib.error.URLError, OSError) as e:
            out.append({"source": src["id"], "type": src.get("type"), "url": url,
                        "status": "unreachable", "note": str(e)})
            continue
        h = _sha256(body)
        if hashes.get(url) and hashes[url] != h:
            out.append({"source": src["id"], "type": src.get("type"), "url": url,
                        "status": "changed", "old": hashes[url], "new": h})
        elif not hashes.get(url):
            out.append({"source": src["id"], "type": src.get("type"), "url": url,
                        "status": "new", "new": h})
        hashes[url] = h
    return out, {"hashes": hashes}


def _dump_yaml(d):
    if yaml is not None:
        return yaml.safe_dump(d, sort_keys=False, allow_unicode=True)
    return json.dumps(d, indent=2, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="skip all network; validate + emit empty candidates")
    ap.add_argument("--date", default=None, help="output dir date YYYY-MM-DD (default: today)")
    ap.add_argument("--watchlist", default=WATCHLIST)
    ap.add_argument("--write-state", action="store_true", help="persist detection state for next run's delta")
    a = ap.parse_args(argv)

    wl = load_watchlist(a.watchlist)
    sources = wl.get("sources") or []
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    date = a.date or time.strftime("%Y-%m-%d")
    state = load_state()

    candidates, notes, new_state = [], [], dict(state)
    if a.offline:
        notes.append("offline mode: no network polling performed; structural run only.")
    else:
        for src in sources:
            sid, stype = src.get("id"), src.get("type")
            prev = state.get(sid, {})
            try:
                if stype == "git":
                    found, st = detect_git(src, prev, token)
                elif stype in ("blog_index", "doc", "url_set"):
                    urls = src.get("urls") or []
                    if stype == "url_set":
                        notes.append(f"{sid}: url_set drift needs sources_index.md URL expansion (skipped this run).")
                        found, st = [], prev
                    else:
                        found, st = detect_hash(src, prev, token, urls)
                elif stype == "search":
                    notes.append(f"{sid}: search source '{src.get('query','')}' — arXiv polling not run in this build.")
                    found, st = [], prev
                else:
                    notes.append(f"{sid}: unknown type '{stype}', skipped.")
                    found, st = [], prev
            except Exception as e:  # never hard-fail the weekly job on one bad source
                notes.append(f"{sid}: error {e}")
                found, st = [], prev
            candidates.extend(found)
            new_state[sid] = st

    out_dir = os.path.join(INBOX, date)
    os.makedirs(out_dir, exist_ok=True)
    doc = {
        "generated": date,
        "watchlist": os.path.relpath(a.watchlist, REPO),
        "count": len(candidates),
        # Every candidate carries the invariant lifecycle up front so the curator/PR can't forget it.
        "lifecycle_default": "candidate",
        "verified_on_default": None,
        "notes": notes,
        "candidates": candidates,
    }
    out_path = os.path.join(out_dir, "candidates.yaml")
    open(out_path, "w", encoding="utf-8").write(_dump_yaml(doc))

    if a.write_state and not a.offline:
        os.makedirs(INBOX, exist_ok=True)
        tmp = STATE + ".tmp"
        open(tmp, "w", encoding="utf-8").write(json.dumps(new_state, indent=2))
        os.replace(tmp, STATE)

    print(f"_ingest_web: {len(candidates)} candidate(s) -> {os.path.relpath(out_path, REPO)}"
          + (f"; {len(notes)} note(s)" if notes else ""))
    for n in notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
