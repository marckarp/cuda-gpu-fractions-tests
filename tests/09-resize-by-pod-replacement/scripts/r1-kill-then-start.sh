#!/usr/bin/env bash
# Test 09 R1. Run on the box with split A up (prefill-a 0.6, decode-a 0.35, gms 0.05).
# Under continuous load, resize to prefill 0.35 / decode 0.60 by deleting each pod
# and creating its replacement (kill, then start). Prefill first, because the
# decode cannot grow until prefill has given up its share.
cd /opt/t09
EV=/opt/t09/r1.events; : > $EV
ev() { echo "$(date +%s.%N) $*" >> $EV; echo "[$(date +%T)] $*"; }
ready() { # pod, log pattern
  for i in $(seq 200); do kubectl logs "$1" </dev/null 2>/dev/null | grep -q "$2" && return 0; sleep 1; done; return 1; }

echo "## does KAI let SM requests exceed 100%? (60+35+5 booked, asking for 50 more)"
sed -e 's/name: live-probe/name: overbook/' -e 's/gpu-compute.request: "0.3"/gpu-compute.request: "0.5"/' /opt/t08/10-live-probe.yaml | kubectl apply -f - </dev/null >/dev/null
sleep 15; kubectl get pod overbook --no-headers </dev/null
kubectl describe pod overbook </dev/null | grep -iE "warning|unschedul|not enough|insufficient" | head -3 | cut -c1-260
kubectl delete pod overbook --wait=true --grace-period=1 </dev/null >/dev/null 2>&1

python3 mkpod.py prefill prefill-b 0.35 > prefill-b.yaml
python3 mkpod.py decode decode-b 0.6 > decode-b.yaml
nohup python3 loadgen.py /opt/t09/r1.jsonl 16 330 </dev/null >/opt/t09/r1.loadgen.log 2>&1 &
ev "load started (16 clients)"
sleep 45
ev "delete prefill-a (60%)"
kubectl delete pod prefill-a --wait=true </dev/null >/dev/null
ev "prefill-a gone; create prefill-b (35%)"
kubectl apply -f prefill-b.yaml </dev/null >/dev/null
ready prefill-b "dynamo.prefill.generate" && ev "prefill-b ready"
sleep 20
ev "delete decode-a (35%)"
kubectl delete pod decode-a --wait=true </dev/null >/dev/null
ev "decode-a gone; create decode-b (60%)"
kubectl apply -f decode-b.yaml </dev/null >/dev/null
ready decode-b "dynamo.backend.generate" && ev "decode-b ready"
wait
ev "load finished"
python3 timeline.py /opt/t09/r1.jsonl $EV 10
