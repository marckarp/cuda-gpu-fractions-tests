# PoC: disaggregated serving of a small model on ONE GPU with GPU fractions

**Claim proven:** prefill and decode running as two GPU fractions of a single A100
serve the same throughput as the usual aggregated deployment on that GPU, with
inter-token latency nearly halved.

| Qwen3-4B, A100-40GB, 4k-token prompts, 128 output, concurrency 32 | Output tok/s | ITL avg / p99 | TTFT avg |
|---|---|---|---|
| Aggregated (one engine, whole GPU) | 374 | 77 / 83 ms | 1.0 s |
| **Prefill + decode as fractions (60% / 40% of SMs), same GPU** | **377** | **43 / 45 ms** | 4.7 s |

TTFT is the trade-off of giving prefill 60% of the SMs at this load; shift the split
to move it. AISimulate (whole GPUs only) predicts "aggregated wins per GPU" for this
case; on one GPU the fraction layout matches it while fixing the ITL tail.

## Why this works, in plain terms

- Aggregated serving runs prefill and decode in the same forward pass, so every
  long prompt stretches decode's step and jitters inter-token latency.
- Disaggregated serving separates them, but normally costs a second GPU and a
  network hop for the KV cache.
- GPU fractions give each engine its own SM share and memory cap on **one** GPU.
  The KV handoff becomes a copy inside the card (NIXL over CUDA IPC, ~9 GB/s
  measured), so you get the isolation without the second GPU or the network.

## The two things that had to be fixed to get here

1. **Identical PIDs in two containers break CUDA IPC.** Each container has its own
   PID namespace, and vLLM's EngineCore lands on the same PID (348) in both. CUDA
   IPC handles carry the exporter's PID; when the importer has the same PID the
   open fails (`cuIpcOpenMemHandle: invalid device context`) and UCX silently falls
   back to TCP at ~175 MB/s, about 50x slower, with no error at normal log levels.
   Fix in this PoC: the decode container forks 200 throwaway processes before
   `exec`, so its EngineCore gets a different PID. (Same deployment, only this
   change: one 560 MB transfer went from 3.3 s to 0.015 s.) The proper fix belongs
   in the CUDA/MPS or UCX layer; see `../tests/07-small-model-agg-vs-frac-disagg/RESULTS.md`.
2. **Memory headroom for the IPC mapping.** Under MPS the mapping of the peer's
   memory counts against the importer's cap. Decode at 0.88 utilization failed at
   transfer post; 0.75 works. Rule of thumb: leave headroom at least the size of the
   peer's KV arena.

Everything else is stock: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0`, no vLLM
patches, Dynamo defaults (file discovery, TCP request plane, ZMQ events).

## Prerequisites

- One node with 1 or more A100-40GB (tested on Brev `test-cuda-gpu-fractions`,
  2x A100), driver 615.71.09 / CUDA 13.4, microk8s.
- The GPU fraction stack from this repo's [`setup/`](../setup/): GPU Operator,
  KAI-Scheduler and kai-gpu-fractioning at the pinned fork commits (PR #147),
  verified with [test 01](../tests/01-smoke-half-gpu/RESULTS.md).
- Host directories: `sudo mkdir -p /opt/hf-cache /opt/dyn-discovery && sudo chmod 777 /opt/hf-cache /opt/dyn-discovery`
- The runtime image pulled once: `sudo microk8s ctr --namespace k8s.io images pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0`
- Egress to Hugging Face and PyPI (model download, `aiperf` install).

## Steps

All commands run on the node. Manifests are in [`manifests/`](manifests/).

1. **Model cache and load generator** (once). Downloads Qwen3-4B into `/opt/hf-cache`
   and starts an AIPerf pod on the host network.
   ```bash
   kubectl apply -f manifests/00-helpers.yaml
   kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/dl-qwen3-4b --timeout=15m
   kubectl logs aiperf | grep AIPERF_READY
   ```
2. **Baseline: aggregated on one whole GPU.**
   ```bash
   kubectl apply -f manifests/10-agg-1gpu.yaml
   until curl -s localhost:8000/v1/models | grep -q Qwen3-4B; do sleep 5; done
   ./run-benchmark.sh A-agg
   kubectl delete pod agg --wait=true
   ```
3. **Fraction disaggregation on the same GPU.** Both workers are KAI fractions
   (`gpu-compute.request` 0.6 / 0.4, memory 14000Mi / 25000Mi, `sm-sharing`), with
   the PID shift on decode and the headroom utilizations.
   ```bash
   kubectl apply -f manifests/20-frac-disagg-1gpu.yaml
   # both on the same GPU?
   for p in prefill decode; do kubectl exec $p -- env | grep -E "NVIDIA_VISIBLE_DEVICES|THREAD_PERCENTAGE"; done
   # different EngineCore PIDs? (this is the fix)
   for p in prefill decode; do echo "$p: $(kubectl exec $p -- pgrep -f EngineCore | head -1)"; done
   until kubectl logs prefill | grep -q "dynamo.prefill.generate" && kubectl logs decode | grep -q "dynamo.backend.generate"; do sleep 5; done
   ./run-benchmark.sh B-frac
   ```
4. **Check the transfer path is healthy** before trusting the numbers. The decode
   worker logs KV transfer metrics every 30 s; you want tens of ms per transfer and
   GB/s, not seconds and MB/s:
   ```bash
   kubectl logs decode | grep "KV Transfer metrics" | tail -1
   # healthy: Avg xfer time (ms)=~60, Throughput (MB/s)=~9000, Avg MB per transfer=~565
   ```
   If you see seconds per transfer, the PID collision is back: compare the two
   `pgrep -f EngineCore` values above.
5. **Compare** `A-agg` and `B-frac` output: Output Token Throughput, Inter Token
   Latency p99, Time to First Token.

## What to try next

- Rerun both without `--enforce-eager` (CUDA graphs on) for realistic decode numbers.
- Sweep the SM split (70/30, 50/50) and two prefill fractions to trade TTFT against ITL.
- Measure HBM contention directly: decode ITL while prefill is saturated.
- Larger models (8B, 32B) where decode is more clearly bandwidth-bound.

## Known gaps to raise

- CUDA IPC across containers with equal PIDs (CUDA/MPS or UCX): silent TCP fallback.
- KAI does not allow two fractional containers in one pod yet; one pod would also
  avoid the PID collision.
- The IPC mapping counting against the importer's MPS memory cap needs a documented
  sizing rule or a shared KV pool design.
