#!/usr/bin/env python3
"""Print output_throughput / TTFT / TPOT for a set of benchmark result directories, one
line each, so arms can be compared without opening five JSON files.

    python3 analysis/show_results.py results/base*_w*_* results/final_w*_*

Every arm table in FINDINGS.md section 6 was assembled from this. It reads
inferencex_result.json under each directory given (or treats the argument as a glob for the
JSON itself), which is the file the task defines as the source of the key metric.
"""
import json,sys,glob,os
rows=[]
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d,'inferencex_result.json'))) or sorted(glob.glob(d)):
        j=json.load(open(f))
        rows.append((os.path.basename(os.path.dirname(f)), j['output_throughput'], j['mean_ttft_ms'], j['mean_tpot_ms'], j.get('total_token_throughput')))
for r in rows: print('%-46s %9.3f  ttft %8.1f  tpot %6.2f  tot %9.1f'%r)
