# 06: Dynamo disaggregated serving (prefill + decode) on ONE GPU via fractions

**PASS, 2026-10-22.** A Dynamo 1.5 prefill worker and decode worker ran as two
fractional pods on the same A100, served requests end to end, and the KV cache
moved between them over NIXL at on-device copy speed (CUDA IPC, ~214 GB/s).
First time a model ran on this box.

## Setup

- Image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0`, model `Qwen/Qwen3-0.6B`.
- Operator-less: three plain pods ([`disagg-fractions.yaml`](disagg-fractions.yaml)).
  Discovery `file` backend over a hostPath (`DYN_FILE_KV=/opt/dyn-discovery`),
  request plane TCP, event plane ZMQ (the 1.5 defaults). No etcd, no NATS,
  no Dynamo operator installed.
- Fractions (same shape as test 05):

| Pod | Annotations | Got (from pod env) |
|---|---|---|
| prefill | `gpu-compute.request: "0.7"`, memory 10000Mi | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70`, `PINNED_DEVICE_MEM_LIMIT=0=10000M` |
| decode | `gpu-compute.request: "0.3"`, memory 28000Mi | `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30`, `PINNED_DEVICE_MEM_LIMIT=0=28000M` |

Both pods reported the same `NVIDIA_VISIBLE_DEVICES` UUID
(`GPU-4fb800e6-a00c-3c44-4c8b-081bd97d3216`): KAI packed them onto one GPU.

## Evidence

Frontend discovered the decode worker, then the prefill worker, through the
file backend; the ZMQ event channel was discovered the same way:

```
Discovery Added event received ... transport: Zmq { endpoint: "tcp://10.1.22.230:38037" }
Committed discovered model group model_name="Qwen/Qwen3-0.6B" ... members=1
```

Decode instantiated NIXL (`Backend UCX was instantiated`) and, on the first
request, checked compatibility with prefill and planned the transfer:

```
NIXL compatibility check passed
Transfer plan: TransferTopology(tp_ratio=1, num_kv_heads=8, local_tp=1, remote_tp=1, ...)
```

Two requests, both handled by the prefill pod (`request completed` x2 in its
log), then decoded by the decode pod. Decode's KV transfer metrics:

| Request | Prompt tokens | KV moved | Transfer time | Throughput |
|---|---|---|---|---|
| 1 (short) | ~20 | 3.5 MB | 17.0 ms | 205 MB/s (first-connection setup dominates) |
| 2 (long) | 4,406 | 483 MB | 2.26 ms | **214,096 MB/s** |

483 MB in 2.26 ms is only possible as a device-to-device copy on the same GPU.
UCX selected its CUDA IPC path between the two MPS clients; nothing went
through host memory or the network.

## Findings

- Dynamo 1.5 disagg works unchanged on fractions. Nothing in Dynamo knows about
  MPS; the two engines just see one GPU with a memory cap and an SM cap.
- NIXL between two fractions of one GPU is CUDA IPC at on-device speed. The
  KV transfer cost of disaggregation, the main tax it pays across GPUs or
  nodes, is effectively gone in this layout.
- `--gpu-memory-utilization` is relative to the MPS memory cap, not the
  physical card: CUDA reports the cap as the device total. Prefill at 0.20
  failed with 0.42 GiB of KV room (`0.2 x 10000M - weights`); 0.80 of the cap
  works. Size the fraction, then set utilization against it.
- No operator, etcd, or NATS is needed for a single-node bring-up: file
  discovery on a hostPath plus the TCP/ZMQ defaults is enough.

## What this does NOT prove

- No performance comparison yet. Whether fractioned P+D on one GPU beats
  aggregated serving on the same GPU (chunked prefill), or how it compares to
  disagg across two whole GPUs, is unmeasured. Baseline manifests are checked in
  ([`agg-1gpu.yaml`](agg-1gpu.yaml), [`disagg-2gpu.yaml`](disagg-2gpu.yaml))
  for an AIPerf run at matched load.
- HBM bandwidth contention between the two fractions is not measured. Memory
  bandwidth is not partitioned by MPS, so a busy prefill will still slow
  decode steps; how much is the open question.
- Two requests only; no concurrency, no noisy-neighbor test.
- The transfer counter is per decode process and resets on restart; read the
  latest `KV Transfer metrics` line, which is logged every 30 s.

## Reproduce

```bash
kubectl apply -f disagg-fractions.yaml
kubectl get pods -l app=dyn-frac -o wide
# same GPU?
for p in prefill decode; do kubectl exec $p -- env | grep -E "NVIDIA_VISIBLE_DEVICES|CUDA_MPS"; done
# wait for both workers to register, then
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hi"}],"max_tokens":20}'
kubectl logs decode | grep "KV Transfer metrics" | tail -1
```
