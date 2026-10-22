# 02 — MPS socket bypass (symmetric setup)

**Bypass reproduced — 2026-10-22.** A non-root pod with a `gpu-fraction: "0.5"`
cap, running with `CUDA_MPS_PIPE_DIRECTORY` pointed at an empty directory, saw
all 108 SMs instead of the capped 54. This reproduces scheckerNV/mps-multi-gpu
exp 04/05 on the clean annotation-only (symmetric) path, not just the manual
asymmetric static-partition setup.

This answers the open question from the Slack thread: the bypass is NOT specific
to the manual asymmetric configuration.

## Root cause observed

Both GPUs are in **`Default` compute mode, not `Exclusive_Process`:**

```
$ nvidia-smi --query-gpu=index,compute_mode --format=csv
index, compute_mode
0, Default
1, Default
```

In `Default` mode the driver lets any process create its own direct GPU context
without going through MPS. A process that cannot reach the MPS server (empty pipe
directory) therefore falls back to a direct context and sees the whole card.
`Exclusive_Process` is the mode that would refuse the direct open and force every
process through MPS; this fork's `mpsd` is not setting it on these nodes.

## Evidence

Pod: `gpu-fraction: "0.5"`, `sm-sharing`, non-root (uid 1000), caps dropped,
image `nvcr.io/nvidia/k8s/cuda-sample:devicequery`.

```
normal env PIPE_DIR=/tmp/nvidia-mps ATP=50     # fractiond's correct 50% config
=== attempt bypass: empty pipe dir + 100% override ===
  (108) Multiprocessors, (064) CUDA Cores/MP:    6912 CUDA Cores
Result = PASS
```

108 Multiprocessors = full A100. Compare test 01, where the same deviceQuery
WITHOUT breaking the pipe directory (override through MPS) correctly reported 54.

## What holds and what does not

- **Memory cap still holds** under bypass: it is a driver per-process limit
  (`NVIDIA_GPU_MEMORY_LIMIT`), independent of MPS. Not re-measured here; test 01
  check 4 showed the cap survives a client override. Worth a dedicated check.
- **Compute cap holds against override THROUGH MPS** (test 01: 54/108 despite
  ATP=100), but NOT against going AROUND MPS (this test: 108/108).

## What this does NOT prove

- deviceQuery reads device properties. 108 Multiprocessors shows the process has
  a full-GPU context; it is strong but not the same as kernel-execution SM-ID
  evidence (Sanjana's `%smid` probe) showing work actually landing on all 108.
- Not tested: whether setting `Exclusive_Process` on the nodes closes this. That
  is the natural next step and matches the PR #147 enforcement direction.
- One run, one node. No claim about every driver/fork combination.

## Relevance

gpu-fractioning PR #147 (server-enforced compute via per-container MPS
namespaces) is draft. Kyle's status: compute enforcement is the last feature not
yet shipped. This result is consistent with that: the released/forked stack as
installed enforces the SM share for cooperating clients but does not yet close
the around-MPS path, because the GPUs are left in Default compute mode.
