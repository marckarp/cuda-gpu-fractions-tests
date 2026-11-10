#!/usr/bin/env bash
# Test 09 R4. Shadow swap for the prefill role, under load.
# Start state: prefill-s 0.45 (plain, holds no lock), decode-s4 0.35 active, 15% free (5% of it goes to the gms-prefill-kv pod).
#  1. add prefill-p0 (shadow, 10%): the lock is free, so it becomes a second active prefill
#  2. delete prefill-s: prefill-p0 keeps serving (add-then-remove, needs the spare 15%)
#  3. add prefill-p1 (shadow, 45%): parks as standby
#  4. delete prefill-p0: prefill-p1 takes over
cd /opt/t09
EV=/opt/t09/r4.events; : > $EV
ev() { echo "$(date +%s.%N) $*" >> $EV; echo "[$(date +%T)] $*"; }
waitlog() { for i in $(seq 400); do kubectl logs "$1" </dev/null 2>/dev/null | grep -q "$2" && return 0; kubectl get pod "$1" --no-headers </dev/null 2>/dev/null | grep -qE "Error|Failed" && return 1; sleep 0.25; done; return 1; }

python3 mkshadow.py prefill prefill-p0 0.10 0 > prefill-p0.yaml
python3 mkshadow.py prefill prefill-p1 0.45 1 > prefill-p1.yaml
kubectl apply -f prefill-p0.yaml </dev/null >/dev/null
waitlog prefill-p0 "failover_state engine=0 -> active" || { echo "prefill-p0 did not become active"; kubectl logs prefill-p0 --tail=15 </dev/null | cut -c1-240; exit 1; }
sleep 5
nohup python3 loadgen.py /opt/t09/r4.jsonl 16 230 </dev/null >/opt/t09/r4.loadgen.log 2>&1 &
ev "load started (16 clients); prefill-s 45% and prefill-p0 10% both active"
sleep 40
ev "delete prefill-s (45%)"
kubectl delete pod prefill-s --wait=true </dev/null >/dev/null
ev "prefill-s gone; create standby prefill-p1 (45%)"
kubectl apply -f prefill-p1.yaml </dev/null >/dev/null
waitlog prefill-p1 "waiting for lock" && ev "prefill-p1 parked"
sleep 20
ev "delete prefill-p0 (active, 10%)"
kubectl delete pod prefill-p0 --wait=false </dev/null >/dev/null
waitlog prefill-p1 "failover_state engine=1 -> active" && ev "prefill-p1 active (45%)"
wait
ev "load finished"
python3 timeline.py /opt/t09/r4.jsonl $EV 5
kubectl get pods --no-headers </dev/null | cut -c1-70
