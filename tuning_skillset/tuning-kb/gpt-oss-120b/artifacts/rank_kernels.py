import gzip, json, sys, collections
path = sys.argv[1]
with gzip.open(path,'rt') as f: tr = json.load(f)
ev=[e for e in tr['traceEvents'] if e.get('cat')=='kernel']
tot=collections.Counter(); cnt=collections.Counter()
tmin=min(e['ts'] for e in ev); tmax=max(e['ts']+e['dur'] for e in ev)
for e in ev:
    tot[e['name']]+=e['dur']; cnt[e['name']]+=1
span=tmax-tmin
allk=sum(tot.values())
print(f"span={span/1000:.1f} ms  sum_kernel={allk/1000:.1f} ms  nkernels={len(ev)}  unique={len(tot)}")
print(f"{'us_total':>10} {'%':>6} {'count':>6} {'us_each':>8}  name")
for n,v in tot.most_common(40):
    print(f"{v:10.0f} {100*v/allk:6.2f} {cnt[n]:6d} {v/cnt[n]:8.2f}  {n[:110]}")
