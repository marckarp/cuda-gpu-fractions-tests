#!/usr/bin/env bash
# Test 09 R2. Run on the box with the shadow layout up: gms (weights + kv_cache),
# prefill-s 0.45, decode-s0 0.35 holding the lock (active), decode-s1 0.15 parked.
# Under continuous load: delete the active decode so the standby takes over at its
# own share, then start a new standby at 0.35 and switch again.
cd /opt/t09
EV=/opt/t09/r2.events; : > $EV
ev() { echo "$(date +%s.%N) $*" >> $EV; echo "[$(date +%T)] $*"; }
waitlog() { for i in $(seq 240); do kubectl logs "$1" </dev/null 2>/dev/null | grep -q "$2" && return 0; sleep 0.5; done; return 1; }

python3 mkshadow.py decode decode-s2 0.35 2 > decode-s2.yaml
nohup python3 loadgen.py /opt/t09/r2.jsonl 16 290 </dev/null >/opt/t09/r2.loadgen.log 2>&1 &
ev "load started (16 clients), active decode-s0 35%"
sleep 45
ev "delete decode-s0 (active, 35%)"
kubectl delete pod decode-s0 --wait=false </dev/null >/dev/null
waitlog decode-s1 "failover_state engine=1 -> active" && ev "decode-s1 active (15%)"
kubectl wait --for=delete pod/decode-s0 --timeout=60s </dev/null >/dev/null 2>&1; ev "decode-s0 pod gone"
sleep 20
ev "create standby decode-s2 (35%)"
kubectl apply -f decode-s2.yaml </dev/null >/dev/null
waitlog decode-s2 "waiting for lock" && ev "decode-s2 parked"
sleep 20
ev "delete decode-s1 (active, 15%)"
kubectl delete pod decode-s1 --wait=false </dev/null >/dev/null
waitlog decode-s2 "failover_state engine=2 -> active" && ev "decode-s2 active (35%)"
wait
ev "load finished"
python3 timeline.py /opt/t09/r2.jsonl $EV 5
echo "## wake timing from the engines' own logs"
for p in decode-s1 decode-s2; do kubectl logs $p --timestamps </dev/null 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "Lock acquired|failover_state" | cut -c1-140; done
kubectl get pods --no-headers </dev/null | cut -c1-70
