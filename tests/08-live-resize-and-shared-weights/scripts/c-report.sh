#!/usr/bin/env bash
# Test 08 C/E. Run on the box after prefill + decode are up. Prints the evidence:
# startup times, GPU memory (total and per process), what each engine reports
# for weights and KV cache, and one served request.
T0=${1:-0}   # epoch seconds when `kubectl apply` was issued (0 = skip timings)
for p in prefill decode; do
  ready=$(kubectl logs $p --timestamps </dev/null 2>/dev/null | grep -m1 -E "dynamo\.(prefill|backend)\.generate" | cut -d' ' -f1)
  [ -n "$ready" ] && [ "$T0" != 0 ] && echo "$p ready after $(( $(date -d "$ready" +%s) - T0 )) s"
done
echo "## GPU memory"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader </dev/null
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader </dev/null
echo "## what the engines report"
for p in prefill decode; do
  echo "--- $p"
  kubectl logs $p </dev/null 2>/dev/null | grep -iE "GMS\]|imported|weights took|Model loading took|GPU KV cache size|Available KV cache memory|Maximum concurrency|lock" | cut -c1-230 | tail -12
done
kubectl get pod gms </dev/null >/dev/null 2>&1 && { echo "--- gms"; kubectl logs gms </dev/null | grep -iE "alloc|commit|session|lock|bytes" | tail -8 | cut -c1-230; }
echo "## one request"
curl -s -m 60 localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":24,"chat_template_kwargs":{"enable_thinking":false}}' </dev/null | cut -c1-400
echo
echo "## KV transfer"; kubectl logs decode </dev/null 2>/dev/null | grep "KV Transfer metrics" | tail -1 | cut -c1-300
