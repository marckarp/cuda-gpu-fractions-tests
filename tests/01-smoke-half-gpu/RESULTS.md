# 01 — half-GPU smoke test (symmetric, enforcement)

**PASS — 2026-10-22.** Two pods at `gpu-fraction: "0.5"` bin-packed onto one
A100-40GB; the MPS server pinned each container's namespace at 50% SMs, and a
hostile client override to 100% still saw only 54 of 108 multiprocessors.

Test procedure: Step 5 of the merged Dynamo doc `build-gpu-fractioning-forks.md`,
run unchanged on our box (see `../../setup/01-box-setup.md` for the stack).

## Evidence

**Check 1 — binder decisions.** Both pods got identical annotations and the SAME
physical GPU (bin-packing works):

```
nvidia.com/container.main.gpu-compute.portion: "0.5"
nvidia.com/container.main.gpu-memory.request:  "19968Mi"
nvidia.com/container.main.gpus.devices: ...gpu=GPU-59a92a45-cffb-0f89-d478-3d9ec2968bda
```

19968 = (40960 − 1024 MPS-server reserve) × 0.5, as documented.

**Check 2 — what fractiond injected into the container:**

```
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=19968M
NVIDIA_GPU_MEMORY_LIMIT=19968
CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
```

**Check 3 — MPS server ground truth** (`namespace list --server=shared`): two
`kai_*` namespaces at 50.00 active thread percentage plus `default` at 10.00.

**Check 4 — pod memory view with hostile override**
(`env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 nvidia-smi`): reports 19968 MiB,
the capped slice, not the physical 40960.

**Compute enforcement — deviceQuery with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100`
set in the pod spec:**

```
(054) Multiprocessors, (064) CUDA Cores/MP:    3456 CUDA Cores
Result = PASS
```

54/108 SMs despite asking for 100%. fractiond replaces the client variable at
container create, and the server caps the namespace regardless of what the
client requests.

## What this does NOT prove

- No kernel-execution SM-ID evidence yet (deviceQuery reads device properties;
  actual `%smid` probing is a separate test).
- The MPS socket-bypass attack (point `CUDA_MPS_PIPE_DIRECTORY` at a missing
  path and try a direct CUDA context) was NOT attempted here. That is test 02.
- Functionality, not performance: no throughput or isolation-under-load claims.

## Manifests

- `smoke-pods.yaml` — the two half-GPU sleep pods
- `devicequery.yaml` — the enforcement probe with the 100% override
