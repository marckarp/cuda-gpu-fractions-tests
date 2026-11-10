#!/usr/bin/env bash
# Test 10. Run on the box. Shows the state of everything in the default namespace
# and, for every frontend Service, which models it lists and whether each of the
# two models answers through it.
#   check.sh [seconds to wait for pods to be Ready, default 240]
WAIT=${1:-240}
MODELS="Qwen/Qwen3-4B Qwen/Qwen3-0.6B"
end=$(( $(date +%s) + WAIT ))
while [ $(date +%s) -lt $end ]; do
  n=$(kubectl get pods --no-headers </dev/null 2>/dev/null | grep -v -E "1/1 +Running|Completed" | wc -l)
  [ "$n" = "0" ] && break; sleep 5
done
sleep 8
echo "## DGDs";  kubectl get dynamographdeployments </dev/null 2>&1 | cut -c1-120
echo "## pods";  kubectl get pods --no-headers </dev/null | cut -c1-90
echo "## each pod's Dynamo namespace as the operator set it"
for p in $(kubectl get pods --no-headers </dev/null | grep Running | cut -d' ' -f1); do
  echo "$p: $(kubectl exec $p -- sh -c 'echo DYN_NAMESPACE=$DYN_NAMESPACE DYN_NAMESPACE_PREFIX=$DYN_NAMESPACE_PREFIX DYN_DISCOVERY_BACKEND=$DYN_DISCOVERY_BACKEND' </dev/null 2>/dev/null)"
done
echo "## frontends"
for svc in $(kubectl get svc --no-headers </dev/null | grep -i frontend | cut -d' ' -f1); do
  ip=$(kubectl get svc $svc -o jsonpath='{.spec.clusterIP}' </dev/null)
  listed=$(curl -s -m 10 http://$ip:8000/v1/models </dev/null | python3 -c "import json,sys; print(sorted(m['id'] for m in json.load(sys.stdin)['data']))" 2>/dev/null)
  echo "--- $svc ($ip) lists: $listed"
  for m in $MODELS; do
    r=$(curl -s -m 60 http://$ip:8000/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word.\"}],\"max_tokens\":8,\"chat_template_kwargs\":{\"enable_thinking\":false}}" </dev/null)
    echo "    $m -> $(echo "$r" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print('OK, served by model', d['model']) if 'choices' in d else print('FAIL:', str(d)[:110])
except Exception as e: print('FAIL: no JSON')")"
  done
done
echo "## pods not Running, and why"
for p in $(kubectl get pods --no-headers </dev/null | grep -v -E "1/1 +Running|Completed" | cut -d' ' -f1); do
  echo "--- $p"; kubectl logs $p --tail=200 </dev/null 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -iE "cannot register|already registered|error|Unschedul" | tail -3 | cut -c1-300
  kubectl get events --field-selector involvedObject.name=$p --no-headers </dev/null 2>/dev/null | grep -i warn | tail -2 | cut -c1-260
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader </dev/null
