# PoC: prefill and decode as GPU fractions of ONE GPU

## Result

Qwen3-4B on one A100-40GB, 4k-token prompts, 128 output tokens, 32 concurrent requests:

| Layout | GPUs | Output tok/s | tok/s per GPU | ITL avg / p99 | TTFT avg |
|---|---|---|---|---|---|
| Aggregated: one engine, whole GPU | 1 | 374 | 374 | 77 / 83 ms | 1.0 s |
| **Disaggregated: prefill + decode as two fractions of the same GPU** | 1 | **377** | **377** | **43 / 45 ms** | 4.7 s |
| Disaggregated: prefill on one whole GPU, decode on the other (NVLink) | 2 | 586 | 293 | 44 / 48 ms | 0.9 s |

On one GPU: same throughput as aggregated, inter-token latency nearly halved. TTFT
is higher because prefill only gets 60% of the SMs here; change the split to trade
one for the other. Against real two-GPU disaggregation, the fraction layout delivers
29% more tokens per GPU with the same ITL; the two-GPU layout wins on TTFT because
prefill has a whole GPU, and gives 1.56x total throughput for 2x hardware.

## The idea in one paragraph

In aggregated serving, prefill and decode share one engine, so every long prompt
stretches decode's step and jitters token latency. Disaggregation separates them,
but normally costs a second GPU and a network hop for the KV cache. GPU fractions
give each engine its own SM share and memory cap on **one** GPU, so decode is
protected from prefill, and the KV handoff is a copy inside the card (CUDA IPC,
~9 GB/s measured) instead of a network transfer.

## What we deployed

Two separate pods on the same A100 (separate pods so prefill and decode can scale
and restart independently), each a KAI GPU fraction:

| Pod | SMs | Memory cap | vLLM `--gpu-memory-utilization` |
|---|---|---|---|
| prefill | 60% | 14000 MiB | 0.80 |
| decode | 40% | 25000 MiB | 0.75 |

Plus a frontend pod and an AIPerf pod. Stock `vllm-runtime:1.5.0` image, Dynamo
defaults, no code patches. Manifests in [`manifests/`](manifests/).

## The problem we hit, and the workaround

**Symptom.** Every KV transfer from prefill to decode took 3 to 20 seconds instead
of milliseconds. Disaggregation was 10x slower than aggregated. No error was logged.

**Cause: both engines had the same PID.** PIDs are numbered per container, starting
at 1. The two pods run the same image with the same command, so they start the same
processes in the same order, and vLLM's engine process ends up as PID 348 in both.
A CUDA IPC handle records the exporter's PID (348). When decode (also 348) opens it,
CUDA thinks the handle is its own and refuses. UCX then quietly falls back to
copying over TCP at ~175 MB/s. The PID namespace is not part of the check.

**Workaround 1: give decode a different PID.** The decode container forks 200
throwaway processes before starting the engine, so its engine becomes PID 549.
This is the `command:` of the decode pod in
[`manifests/20-frac-disagg-1gpu.yaml`](manifests/20-frac-disagg-1gpu.yaml):

```yaml
command: ["sh", "-c", "for i in $(seq 200); do /bin/true; done; exec python3 -m dynamo.vllm \"$@\"", "--"]
```

Same deployment, only this change: one 560 MB transfer went from 3.3 s to 0.015 s.
It is a hack. The right number depends on the image and launch path, so always
check that the two engines differ: `kubectl exec <pod> -- pgrep -f EngineCore`.

**Workaround 2: leave memory headroom.** Fractions are not two virtual GPUs. They
are two processes on the same physical GPU, and MPS enforces two caps per process:
an SM share and a memory limit. For decode to read prefill's KV cache it must map
prefill's memory into its own address space (that is what CUDA IPC is), and MPS
counts that mapped memory against **decode's** cap even though the bytes belong to
prefill. Decode at 0.88 utilization (about 22 GB of its own) plus about 3 GB of
mapped prefill KV goes over its 25 GB cap, and the transfer fails. Decode at 0.75
(about 19 GB) plus the mapping fits. Prefill at 0.80 and decode at 0.75 work. Rule
of thumb: leave headroom at least the size of the peer's KV cache. This rule is
inferred from the failures and successes observed, not from MPS documentation; it
is item 3 in the list below for the MPS owners to confirm.

## Steps to reproduce

Prerequisites: a node with an A100-40GB, driver 615 / CUDA 13.4, microk8s, and the
GPU fraction stack from [`../setup/`](../setup/) (GPU Operator, KAI-Scheduler and
kai-gpu-fractioning at the pinned fork commits). Then:

```bash
sudo mkdir -p /opt/hf-cache /opt/dyn-discovery && sudo chmod 777 /opt/hf-cache /opt/dyn-discovery
sudo microk8s ctr --namespace k8s.io images pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0
```

1. Model download and load generator (once):
   ```bash
   kubectl apply -f manifests/00-helpers.yaml
   kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/dl-qwen3-4b --timeout=15m
   kubectl logs aiperf | grep AIPERF_READY
   ```
2. Baseline, aggregated on one whole GPU:
   ```bash
   kubectl apply -f manifests/10-agg-1gpu.yaml
   until curl -s localhost:8000/v1/models | grep -q Qwen3-4B; do sleep 5; done
   ./run-benchmark.sh A-agg
   kubectl delete pod agg --wait=true
   ```
3. Disaggregated on fractions of the same GPU:
   ```bash
   kubectl apply -f manifests/20-frac-disagg-1gpu.yaml
   until kubectl logs prefill | grep -q "dynamo.prefill.generate" && kubectl logs decode | grep -q "dynamo.backend.generate"; do sleep 5; done
   # same GPU, different fractions:
   for p in prefill decode; do kubectl exec $p -- env | grep -E "NVIDIA_VISIBLE_DEVICES|THREAD_PERCENTAGE"; done
   # different engine PIDs (the workaround):
   for p in prefill decode; do echo "$p: $(kubectl exec $p -- pgrep -f EngineCore | head -1)"; done
   ./run-benchmark.sh B-frac
   ```
4. Check the transfer path is healthy before trusting the numbers:
   ```bash
   kubectl logs decode | grep "KV Transfer metrics" | tail -1
   # healthy: Avg xfer time (ms) around 60, Throughput (MB/s) around 9000
   # broken:  seconds per transfer, MB/s in the hundreds  -> PIDs collided
   ```
5. Compare `A-agg` and `B-frac`: Output Token Throughput, Inter Token Latency p99,
   Time to First Token. Both logs are in `/opt/aiperf-results/` on the node.

## What has to be fixed to support this without hacks

| # | Problem | Owner | Fixed means |
|---|---|---|---|
| 1 | CUDA IPC fails when the two containers' engines have the same PID; UCX falls back to TCP silently. | CUDA driver / MPS, or UCX `cuda_ipc` as a workaround | Equal PIDs in different PID namespaces are treated as different processes; or UCX detects the case and errors out instead of falling back. |
| 2 | The fallback to TCP is silent. | UCX, NIXL, vLLM connector | A warning on unexpected transport, and the transport name in vLLM's KV transfer metrics. |
| 3 | The IPC mapping is charged against the importer's MPS memory cap. | MPS / fractioning docs, Dynamo operator | Stop charging it twice, or document the headroom rule and have the operator reserve it. |
| 4 | **Across two GPUs on one node**, each pod sees only its own GPU, so CUDA IPC over NVLink is impossible and UCX falls back to TCP (15 MB/s). Making both GPUs visible to both pods (privileged, `NVIDIA_VISIBLE_DEVICES=all`, engine pinned by `CUDA_VISIBLE_DEVICES` order) gave 30 GB/s over NVLink. | Dynamo operator / device plugin | Same-node disaggregated workers need the peer GPU visible (or an RDMA NIC); the operator should set this up, or document that NVLink P/D on Kubernetes requires it. Not needed for the one-GPU fraction layout. |

Prefill and decode stay in separate pods so they scale independently; one pod would
avoid problem 1 but defeats the purpose. Longer term, a KV pool owned once and
referenced by both engines (the pattern Dynamo's GMS uses for weights) removes
problems 1 and 3 together.

Evidence for everything above, including every dead end: [`../tests/07-small-model-agg-vs-frac-disagg/RESULTS.md`](../tests/07-small-model-agg-vs-frac-disagg/RESULTS.md).

## Caveats

One run per configuration; all engines ran with `--enforce-eager` (copied from
Dynamo's disagg example), so absolute decode numbers are pessimistic for both
layouts; 4B model only; HBM contention between the fractions is folded into the
numbers, not measured separately.
