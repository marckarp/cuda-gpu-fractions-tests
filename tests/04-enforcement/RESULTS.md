# 04: memory and compute enforcement across 2 GPUs

**PASS, 2026-10-22.** At 0.25, 0.5 and 0.75 of each of 2 GPUs, memory allocation
failed at the cap on both GPUs, and compute throughput tracked the share. Raising
the memory limit or asking MPS for 100% from inside the process changed neither.

Probe: `probe.py`, run in a `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` pod as
non-root (uid 1000), one fraction at a time, nothing else on the GPUs.

## Results (per GPU, GPU 0 / GPU 1)

| Share x 2 | Cap (MiB) | Allocated before fail | Same with limit raised to 40000M | SMs | Matmul TFLOP/s | Same with ATP=100 |
|---|---|---|---|---|---|---|
| 0.25 | 9984 | 9728 / 9728 | 9728 / 9728 | 26 | 72.4 / 72.0 | 72.4 / 72.5 |
| 0.5 | 19968 | 19712 / 19712 | 19712 / 19712 | 54 | 145.6 / 141.6 | 146.4 / 150.5 |
| 0.75 | 29952 | 29440 / 29440 | 29440 / 29440 | 80 | 195.3 / 191.2 | 191.8 / 201.6 |

- **Memory:** allocations in 256 MiB chunks fail just below the cap (the gap is
  the CUDA context). Setting `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=40000M,1=40000M`
  in the process did not raise it.
- **Compute:** throughput scales with the share: about 72, 145 and 195 TFLOP/s.
  Per SM that is 2.8, 2.7 and 2.4 TFLOP/s, so the 0.75 point is a little below
  linear. Setting `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100` in the process did not
  raise throughput (differences are run-to-run noise, under 5%).
- **What CUDA reports:** `torch.cuda.mem_get_info` total equals the cap (9984,
  19968, 29952 MiB), not 40960. This differs from the Dynamo doc note that CUDA
  reports the physical total.

## What this does NOT prove

- No full-GPU baseline was measured on this box, so we cannot say what fraction
  of full-GPU throughput each share gets, only that throughput rises with the share.
- We did not isolate which layer stops the memory allocation (driver limit from the
  container toolkit hook vs the MPS pinned limit). The cap held either way.
- These are clients that go through MPS. The skip-MPS question (test 02) is
  still inconclusive.
- One pod at a time. No noisy-neighbor test (two pods competing).
