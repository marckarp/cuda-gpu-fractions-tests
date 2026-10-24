#!/bin/sh
# Run the fixed AIPerf workload against whatever is serving on localhost:8000.
# Usage: ./run-benchmark.sh <label>      (results land in /opt/aiperf-results/<label>.log on the node)
set -e
LABEL="${1:?label}"
kubectl exec aiperf -- sh -c "cd /results && rm -rf $LABEL && aiperf profile \
  --model Qwen/Qwen3-4B --url http://localhost:8000 --endpoint-type chat --streaming \
  --concurrency 32 --request-count 192 --warmup-request-count 16 \
  --prompt-input-tokens-mean 4000 --prompt-input-tokens-stddev 0 \
  --output-tokens-mean 128 --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true --extra-inputs min_tokens:128 --random-seed 7 \
  --artifact-dir /results/$LABEL > /results/$LABEL.log 2>&1; \
  grep -E 'Time to First Token|Inter Token Latency|Request Latency \(|Output Token Throughput \(|Request Throughput' /results/$LABEL.log"
