#!/usr/bin/env bash
# Run on the box. One DGD, one frontend, N workers started together, all serving
# Qwen/Qwen3-0.6B under different names (m01..mNN), each a small GPU fraction.
# Reports how many of the N models the frontend lists, 1, 2.5 and 4 minutes after
# every pod is Running, and any model-registration errors in the frontend log.
#   n-models.sh <N> <label>
N=${1:?N}; L=${2:?label}
cd /opt/t10
workers=""
# NS_PER_MODEL=1 gives each worker its own Dynamo namespace (needed on Dynamo 1.3.0,
# where two models cannot register in one namespace). IMAGE selects the runtime image.
for i in $(seq -w 1 $N); do
  ns=-; [ -n "$NS_PER_MODEL" ] && ns=default-nm-m$i
  workers="$workers m$i:Qwen/Qwen3-0.6B:0.05:4500:$ns:m$i"
done
python3 mkdgd.py nm Frontend $workers > nm-$L.yaml
kubectl apply -f nm-$L.yaml </dev/null >/dev/null
T0=$(date +%s)
for i in $(seq 120); do
  r=$(kubectl get pods --no-headers </dev/null 2>/dev/null | grep -c "1/1 *Running")
  [ "$r" = "$((N + 1))" ] && break
  kubectl get pods --no-headers </dev/null 2>/dev/null | grep -q CrashLoop && { echo "workers crash-looping, stopping"; kubectl delete dynamographdeployment nm </dev/null >/dev/null 2>&1; exit 1; }
  sleep 5
done
echo "## N=$N ($L): $r of $((N + 1)) pods Running after $(( $(date +%s) - T0 )) s"
ip=$(kubectl get svc nm-frontend -o jsonpath='{.spec.clusterIP}' </dev/null)
for wait in 60 90 90; do
  sleep $wait
  listed=$(curl -s -m 10 http://$ip:8000/v1/models </dev/null | python3 -c "import json,sys; print(' '.join(sorted(m['id'] for m in json.load(sys.stdin)['data'])))" 2>/dev/null)
  n=$(echo $listed | wc -w)
  missing=$(for i in $(seq -w 1 $N); do echo " $listed " | grep -q " m$i " || printf "m$i "; done)
  echo "   +$(( $(date +%s) - T0 )) s: frontend lists $n of $N; missing: ${missing:-none}"
done
fe=$(kubectl get pods --no-headers </dev/null | grep frontend | cut -d' ' -f1)
echo "## frontend log: model registration errors"
kubectl logs $fe </dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -iE "Error adding model|Lock acquisition|No such file|Failed to download" | cut -c1-330 | head -6
echo "   error lines: $(kubectl logs $fe </dev/null | grep -ciE 'Error adding model')"
echo "## one request to the first and last model"
for m in m$(seq -w 1 $N | head -1) m$(seq -w 1 $N | tail -1); do
  curl -s -m 60 http://$ip:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word.\"}],\"max_tokens\":8}" </dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('   $m ->', 'OK' if 'choices' in d else str(d)[:120])" 2>/dev/null || echo "   $m -> no JSON"
done
kubectl logs $fe </dev/null > fe-$L.log 2>&1
kubectl delete dynamographdeployment nm --wait=true </dev/null >/dev/null 2>&1
for i in $(seq 60); do [ $(kubectl get pods --no-headers </dev/null 2>/dev/null | wc -l) = 0 ] && break; sleep 3; done
