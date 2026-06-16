#!/usr/bin/env python3
"""Aggregate every '## Sources' URL across the KB into index/sources_index.md.
Run from perf_knowledge/: python3 index/_gen_sources.py"""
import os, re, glob, collections
KK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
urls = collections.defaultdict(set)   # domain -> {url}
nfiles = 0
for f in glob.glob(os.path.join(KK, "**", "*.md"), recursive=True):
    if os.path.basename(f) == "sources_index.md": continue
    t = open(f, encoding="utf-8").read()
    nfiles += 1
    for u in re.findall(r'https?://[^\s)\]<>"]+', t):
        u = u.rstrip('.,;')
        dom = re.sub(r'^https?://(www\.)?', '', u).split('/')[0]
        urls[dom].add(u)
alln = sum(len(v) for v in urls.values())
out = ["---","title: Consolidated source index","kind: reference","updated: 2026-06-09","---","",
       f"# Sources index — {alln} unique URLs across {nfiles} docs","",
       "Auto-generated union of every `## Sources` / inline URL (run `index/_gen_sources.py`). Each doc keeps its own inline `## Sources`.",""]
for dom in sorted(urls):
    out.append(f"## {dom}")
    for u in sorted(urls[dom]): out.append(f"- {u}")
    out.append("")
open(os.path.join(KK,"index","sources_index.md"),"w",encoding="utf-8").write("\n".join(out)+"\n")
print(f"OK: {alln} urls / {nfiles} docs -> index/sources_index.md")
