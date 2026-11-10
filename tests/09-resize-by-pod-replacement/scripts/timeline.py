#!/usr/bin/env python3
"""Summarize a loadgen.py record file in fixed time buckets, with event markers.

    timeline.py <records.jsonl> <events.txt> [bucket-seconds]

events.txt has one "<epoch-seconds> <text>" per line. A request is counted in
the bucket where it ended.
"""
import json
import statistics
import sys

recs = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
events = []
for line in open(sys.argv[2]):
    if line.strip():
        t, text = line.split(" ", 1)
        events.append((float(t), text.strip()))
bucket = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
t0 = min(r["start"] for r in recs)
t1 = max(r["end"] for r in recs)

print(f"{'t(s)':>6} {'ok':>4} {'err':>4} {'tok/s':>7} {'TTFT avg':>9} {'TTFT max':>9} {'ITL avg ms':>10}  events")
t = t0
while t < t1:
    b = [r for r in recs if t <= r["end"] < t + bucket]
    ok = [r for r in b if not r["err"]]
    err = [r for r in b if r["err"]]
    toks = sum(r["chunks"] for r in ok)
    ttft = [r["ttft"] for r in ok if r["ttft"] is not None]
    itl = [(r["end"] - r["start"] - r["ttft"]) / (r["chunks"] - 1) * 1000 for r in ok if r["ttft"] and r["chunks"] > 1]
    ev = "; ".join(f"+{e - t0:.0f}s {text}" for e, text in events if t <= e < t + bucket)
    print(f"{t - t0:6.0f} {len(ok):4d} {len(err):4d} {toks / bucket:7.0f} "
          f"{(statistics.mean(ttft) if ttft else float('nan')):9.2f} {(max(ttft) if ttft else float('nan')):9.2f} "
          f"{(statistics.mean(itl) if itl else float('nan')):10.1f}  {ev}")
    t += bucket

errs = [r for r in recs if r["err"]]
print(f"\ntotal requests {len(recs)}, errors {len(errs)}")
kinds = {}
for r in errs:
    k = r["err"][:90]
    kinds[k] = kinds.get(k, 0) + 1
for k, n in sorted(kinds.items(), key=lambda x: -x[1])[:6]:
    print(f"  {n:4d} x {k}")
if errs:
    print(f"first error at +{min(r['end'] for r in errs) - t0:.1f}s, last at +{max(r['end'] for r in errs) - t0:.1f}s")
