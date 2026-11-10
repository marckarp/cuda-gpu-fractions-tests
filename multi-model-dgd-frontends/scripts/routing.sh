#!/usr/bin/env bash
# Run on the box. Proves each request reaches the engine of the model it
# names: through every frontend Service, send 12 requests for Qwen3-4B and 7 for
# Qwen3-0.6B, mixed and 8 at a time, and compare with the per-engine counter
# vllm:request_success_total read from each worker pod before and after.
# Also sends one request for a model that does not exist.
count() { # pod IP -> finished requests on that engine
  curl -s -m 10 http://$1:9090/metrics </dev/null | awk '/^vllm:request_success_total/ {s+=$NF} END {print s+0}'
}
one() { # frontend IP, model -> http status
  curl -s -m 90 -o /dev/null -w "%{http_code}\n" http://$1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word.\"}],\"max_tokens\":8,\"chat_template_kwargs\":{\"enable_thinking\":false}}" </dev/null
}
export -f one
workers=$(kubectl get pods -o wide --no-headers </dev/null | grep Running | grep -v -i frontend | awk '{print $1"="$6}')
for svc in $(kubectl get svc --no-headers </dev/null | grep -i frontend | cut -d' ' -f1); do
  ip=$(kubectl get svc $svc -o jsonpath='{.spec.clusterIP}' </dev/null)
  echo "--- through $svc"
  declare -A before
  for w in $workers; do before[${w%%=*}]=$(count ${w##*=}); done
  { for i in $(seq 12); do echo "Qwen/Qwen3-4B"; done; for i in $(seq 7); do echo "Qwen/Qwen3-0.6B"; done; } | shuf \
    | xargs -P 8 -I{} bash -c "one $ip {}" | sort | uniq -c | sed 's/^/    http status counts: /'
  echo "    unknown model -> http $(one $ip Qwen/Does-Not-Exist)"
  sleep 2
  for w in $workers; do
    d=$(( $(count ${w##*=}) - ${before[${w%%=*}]} ))
    case ${w%%=*} in *qwen4b*) want=12;; *qwen06b*) want=7;; *) want=0;; esac
    [ "$d" = "$want" ] && v=PASS || v=FAIL
    echo "    ${w%%=*}: +$d requests (expected +$want) $v"
  done
done
