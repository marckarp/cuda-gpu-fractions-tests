# 03: half of two GPUs per pod (symmetric multi-GPU)

**PASS, 2026-10-22.** Two pods, each with `gpu-fraction: "0.5"` and
`gpu-fraction-num-devices: "2"`, both ran on our 2x A100-40GB box. Each pod got
half of each GPU: 54 of 108 SMs and 19968 MiB, on both GPUs. A third identical
pod was correctly refused because both GPUs were full.

Manifest: `multi-gpu-pods.yaml`. The pods only ran `sleep`, so no model was served.

## Evidence

**What the scheduler decided (identical for both pods):** both GPU UUIDs, a 0.5
compute portion, and 19968 MiB memory:

```
gpu-compute.portion: "0.5"
gpu-memory.request:  "19968Mi"
gpus.devices: GPU-4fb800e6-...,GPU-59a92a45-...
```

**What fractiond injected:** the memory cap is set separately for each GPU:

```
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=19968M,1=19968M
NVIDIA_GPU_MEMORY_LIMIT=19968
```

**What each pod sees:** `nvidia-smi` shows 2 GPUs at 19968 MiB each. deviceQuery
shows NumDevs = 2, with 54 Multiprocessors and 19968 MB on both devices.

**MPS server (ground truth):** two `kai_*` namespaces, each at 50.00 active
thread percentage, plus `default` at 10.00. That is one namespace per container,
and it covers both GPUs.

**Client override through MPS:** `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100` inside
multi-a still showed 54 SMs on Device 0 and Device 1. The server cap held on
both GPUs.

**Capacity:** after the two pods took 0.5 + 0.5 of each GPU, a third pod stayed
Pending: `Unschedulable ... didn't have enough resources: GPUs`.

## What this does NOT prove

- No model ran. Spanning both GPUs with TP2 and serving inference is the next
  step. Sanjana has done that on her box; we have not yet.
- deviceQuery reports device properties, not where kernels actually ran.
- This only covers clients that go through MPS. The socket-bypass question
  (test 02) is still inconclusive.
- Only symmetric shares were tested. Unequal per-GPU shares have no YAML API.
