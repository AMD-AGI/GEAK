import gzip, json, sys, collections, statistics
path=sys.argv[1]
with gzip.open(path,'rt') as f: tr=json.load(f)
ev=sorted([e for e in tr['traceEvents'] if e.get('cat')=='kernel'], key=lambda e:e['ts'])
ar=[e for e in ev if 'cross_device_reduce' in e['name']]
d=sorted(e['dur'] for e in ar)
print(f"allreduce n={len(d)} max={d[-1]:.0f} p999={d[int(.999*len(d))]:.1f} p99={d[int(.99*len(d))]:.1f} p50={d[len(d)//2]:.2f} min={d[0]:.2f} sum={sum(d)/1000:.1f}ms")
# drop the leading skew outlier(s): find events > 1000us
out=[e for e in ev if e['dur']>1000]
print("kernels >1ms:", [(e['name'][:40], round(e['dur'],1), round((e['ts']-ev[0]['ts'])/1000,1)) for e in out])
t0=max(e['ts']+e['dur'] for e in out) if out else ev[0]['ts']
rest=[e for e in ev if e['ts']>=t0]
span=max(e['ts']+e['dur'] for e in rest)-min(e['ts'] for e in rest)
tot=collections.Counter(); cnt=collections.Counter()
for e in rest: tot[e['name']]+=e['dur']; cnt[e['name']]+=1
busy=sum(tot.values())
nsteps=32
print(f"\npost-skew span={span/1000:.2f} ms  busy={busy/1000:.2f} ms  gap(exposed)={100*(1-busy/span):.2f}%  n={len(rest)}")
print(f"per decode step: span={span/nsteps:.0f} us  busy={busy/nsteps:.0f} us")
print(f"{'us/step':>9} {'%busy':>6} {'n/step':>7} {'us_each':>8}  name")
for n,v in tot.most_common(22):
    print(f"{v/nsteps:9.1f} {100*v/busy:6.2f} {cnt[n]/nsteps:7.1f} {v/cnt[n]:8.2f}  {n[:95]}")
