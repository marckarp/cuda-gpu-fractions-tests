# 05: uneven shares on one GPU, and compute set separately from memory

**PASS, 2026-10-22.** Pods with different shares packed onto one A100, and
compute could be set independently of memory (the prefill/decode case).

## Results (one A100-40GB, 108 SMs)

| Pod | Annotations | Compute limit | SMs | Memory cap |
|---|---|---|---|---|
| A | `gpu-fraction: "0.3"` | 30% | 32 | 11980 MiB |
| B | `gpu-fraction: "0.7"` | 70% | 74 | 27955 MiB |
| C | memory only: `gpu-memory.request: 12000Mi` | 30% (filled in by the scheduler) | 32 | 12000 MiB |
| prefill | `gpu-compute.request: "0.7"`, memory 10000Mi | 70% | 74 | 10000 MiB |
| decode | `gpu-compute.request: "0.3"`, memory 28000Mi | 30% | 32 | 28000 MiB |

A and B shared one GPU; the MPS server showed namespaces at 30.00 and 70.00.
prefill and decode shared one GPU.

## Findings

- With `gpu-fraction`, compute and memory use the same share.
- With memory only, the scheduler sets compute = memory / usable memory
  (12000 / 39936 = 0.3).
- With `gpu-compute.request` and `gpu-memory.request` both set, they are
  independent. A compute-heavy pod (prefill) and a memory-heavy pod (decode)
  can share one GPU.
- Compute is a percentage. The driver rounds it to SMs (70% of 108 = 74,
  30% = 32), so an exact SM count cannot be requested.
- The limit caps how many SMs a pod can use at once. It does not reserve
  specific SMs, and idle SMs are not lent to a busier pod.

## What this does NOT prove

- No model ran. Whether prefill actually runs faster at 70% than at 30% is
  untested.
- deviceQuery only shows the configured limit. Test 04 showed the compute
  limit controls real throughput for 0.25/0.5/0.75; 0.3/0.7 was not
  re-measured.
