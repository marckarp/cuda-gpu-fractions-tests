# Test 09: resizing prefill and decode by replacing pods, under load

PoC for changing the prefill/decode SM split on one GPU by killing a pod and
bringing up another, to find what Dynamo has to add. Follows test 08, which showed
that the driver refuses a live change.

- [`RESULTS.md`](RESULTS.md): findings and the list of Dynamo gaps.
- [`CHANGELOG.md`](CHANGELOG.md): every change and failure.

## Scenarios

| Run | What | Script |
|---|---|---|
| Split benchmark | AIPerf at 60/35 and 35/60 | `run-benchmark.sh` from the PoC folder |
| R1 | Kill, then start, for prefill and decode, under load | `scripts/r1-kill-then-start.sh` |
| R2 | Standby decode (shadow mode), normal delete | `scripts/r2-shadow-swap.sh` |
| R3 | Standby decode, hard kill | `scripts/r3-shadow-swap-forcekill.sh` |
| R4 | Standby prefill | `scripts/r4-prefill-shadow-swap.sh` |

## Reproduce

Needs test 08's files in `/opt/t08` on the node (its `21-gms-disagg-1gpu.yaml` is
the template) and this folder's scripts in `/opt/t09`.

```bash
# plain layout: GMS (weights), frontend, prefill 60%, decode 35%
kubectl apply -f /opt/t08/21a-gms.yaml -f /opt/t09/frontend.yaml
python3 mkpod.py prefill prefill-a 0.6  | kubectl apply -f -
python3 mkpod.py decode  decode-a  0.35 | kubectl apply -f -
bash r1.sh

# shadow layout: GMS (weights + kv_cache), prefill 45%, decode 35% active, decode 15% standby
python3 mkshadow.py gms | kubectl apply -f -
python3 mkpod.py prefill prefill-s 0.45        | kubectl apply -f -
python3 mkshadow.py decode decode-s0 0.35 0    | kubectl apply -f -   # wait until it serves
python3 mkshadow.py decode decode-s1 0.15 1    | kubectl apply -f -   # wait for "waiting for lock"
bash r2.sh
```

Empty `/opt/dyn-discovery` only while the frontend is down.
