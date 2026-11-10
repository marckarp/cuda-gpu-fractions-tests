#!/usr/bin/env bash
# Test 09 R3. Same as R2 but the active engine is removed with --grace-period=0
# --force (no wait for a clean shutdown), to measure the shortest gap, and the
# engine that takes over has its log saved to explain any stall after takeover.
# Start state: prefill-s 0.45, decode-s2 0.35 active.
cd /opt/t09
EV=/opt/t09/r3.events; : > $EV
ev() { echo "$(date +%s.%N) $*" >> $EV; echo "[$(date +%T)] $*"; }
waitlog() { for i in $(seq 400); do kubectl logs "$1" </dev/null 2>/dev/null | grep -q "$2" && return 0; sleep 0.25; done; return 1; }

python3 mkshadow.py decode decode-s3 0.15 3 > decode-s3.yaml
python3 mkshadow.py decode decode-s4 0.35 4 > decode-s4.yaml
nohup python3 loadgen.py /opt/t09/r3.jsonl 16 300 </dev/null >/opt/t09/r3.loadgen.log 2>&1 &
ev "load started (16 clients), active decode-s2 35%"
kubectl apply -f decode-s3.yaml </dev/null >/dev/null; ev "create standby decode-s3 (15%)"
waitlog decode-s3 "waiting for lock" && ev "decode-s3 parked"
sleep 20
ev "force-kill decode-s2 (active, 35%)"
kubectl delete pod decode-s2 --grace-period=0 --force </dev/null >/dev/null 2>&1
waitlog decode-s3 "failover_state engine=3 -> active" && ev "decode-s3 active (15%)"
sleep 45
kubectl logs decode-s3 --timestamps </dev/null > /opt/t09/r3-decode-s3.log 2>&1
kubectl logs prefill-s --timestamps --since=90s </dev/null > /opt/t09/r3-prefill-during-swap.log 2>&1
kubectl apply -f decode-s4.yaml </dev/null >/dev/null; ev "create standby decode-s4 (35%)"
waitlog decode-s4 "waiting for lock" && ev "decode-s4 parked"
sleep 15
ev "force-kill decode-s3 (active, 15%)"
kubectl delete pod decode-s3 --grace-period=0 --force </dev/null >/dev/null 2>&1
waitlog decode-s4 "failover_state engine=4 -> active" && ev "decode-s4 active (35%)"
wait
ev "load finished"
python3 timeline.py /opt/t09/r3.jsonl $EV 5
kubectl get pods --no-headers </dev/null | cut -c1-70
