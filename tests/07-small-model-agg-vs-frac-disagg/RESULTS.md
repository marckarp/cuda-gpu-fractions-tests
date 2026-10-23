# 07: can fraction disagg on one GPU beat aggregated serving for a small model?

**NOT PROVEN, 2026-10-23. Blocked by a KV transfer problem that is not about
fractions.** Under load, every disaggregated layout on this box (fractions on one
GPU, or two whole NVLink GPUs) delivered about 10x less throughput than
aggregated serving on one GPU, because NIXL/UCX KV transfers that take 2 ms when
the system is idle take 6 to 85 seconds when decode is busy. The fraction layout
performed the same as the two-GPU layout while using half the hardware, and its
inter-token latency was better than aggregated, so the idea is still open; the
transfer path has to be fixed first.

## Question

AISimulate was asked for a small model and workload where it prefers aggregated
serving. Qwen3-4B, A100, 4,000 input tokens, 128 output tokens, concurrency 32:

| AISimulate (whole GPUs) | agg, 1 GPU | disagg 1P+1D, 2 GPUs |
|---|---|---|
| output tokens/s per GPU | **529** | 409 |
| ITL avg / p99 | 56 ms / 324 ms | **12.5 ms / 13.5 ms** |
| TTFT avg | 0.6 s | 3.2 s |

So the simulator's call is "agg wins per GPU; disagg only wins on ITL isolation".
A fraction layout puts prefill and decode on one GPU. If it kept the flat ITL and
matched or beat 529 tokens/s on that single GPU, it would beat the recommendation.

## Setup

- Box: Brev `test-cuda-gpu-fractions`, 2x A100-SXM4-40GB (NVLink NV12 between
  them), driver 615.71.09, KAI + gpu-fractioning forks (PR #147).
- Image `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0`, model `Qwen/Qwen3-4B`
  from `/opt/hf-cache` (hostPath). Operator-less pods, file discovery, TCP
  request plane, ZMQ events. All engines `--enforce-eager`, `--max-model-len 8192`.
- Load: AIPerf 0.12 from a `python:3.11-slim` pod on host network. Every run:
  `--concurrency 32 --request-count 192 --warmup-request-count 16`,
  `--prompt-input-tokens-mean 4000 --prompt-input-tokens-stddev 0`,
  `--output-tokens-mean 128 --output-tokens-stddev 0`,
  `--extra-inputs ignore_eos:true --extra-inputs min_tokens:128 --random-seed 7`.
  Logs under `/opt/aiperf-results/` on the box.

| Run | Manifest | Layout |
|---|---|---|
| A | [`agg-4b-1gpu.yaml`](agg-4b-1gpu.yaml) | aggregated, one whole GPU, `--gpu-memory-utilization 0.88` |
| B | [`frac-disagg-4b.yaml`](frac-disagg-4b.yaml) | prefill 60% SMs / 14000 MiB + decode 40% SMs / 25000 MiB, same GPU (`GPU-4fb800e6`) |
| C | [`disagg-4b-2gpu.yaml`](disagg-4b-2gpu.yaml) | prefill and decode on two whole GPUs |
| C2 | C + `UCX_RCACHE_MAX_UNRELEASED=1024` | the setting NIXL warns about |
| C3 | [`disagg-4b-2gpu-hostns.yaml`](disagg-4b-2gpu-hostns.yaml) | C2 + `hostNetwork/hostIPC/hostPID` on both workers |
| C5 | [`disagg-4b-2gpu-tuned.yaml`](disagg-4b-2gpu-tuned.yaml) | C3 + the recipe UCX tuning (`UCX_RNDV_SCHEME=get_zcopy`, `UCX_RNDV_THRESH=0`, `UCX_MEMTYPE_CACHE=n`, `UCX_TLS=cuda_ipc,cuda_copy,sm,tcp`) |
| C5-c4 | same pods as C5 | concurrency 4, 48 requests |

## Results (measured)

| Run | Output tok/s | tok/s per GPU | Req/s | ITL avg / p99 (ms) | TTFT avg / p99 (s) | KV transfer, decode side |
|---|---|---|---|---|---|---|
| **A agg, 1 GPU** | **374** | **374** | 2.92 | 77 / 83 | 0.98 / 6.4 | n/a |
| B fraction disagg, 1 GPU | 37.6 | 37.6 | 0.29 | **43 / 45** | 96 / 120 | 565 MB in 6 to 20 s, 1,782 to 8,484 descriptors |
| C disagg, 2 GPUs | 49.5 | 24.8 | 0.39 | 44 / 47 | 76 / 153 | 565 MB in 85 s avg, 6.6 MB/s |
| C2 + rcache | n/m | n/m | n/m | 44 / 49 | 75 / 152 | 565 MB in 72 s avg, 7.8 MB/s |
| C3 + host namespaces | 53.5 | 26.7 | 0.42 | 43 / 45 | 69 / 115 | 565 MB in 31 s avg, 18 MB/s |
| C5 + recipe UCX tuning | 47.0 | 23.5 | 0.37 | 43 / 45 | 80 / 134 | 565 MB in 33 s avg, 17 MB/s |
| C5-c4, concurrency 4 | 40.1 | 20.1 | 0.31 | 38 / 41 | 7.8 / 11.8 | 220 MB in 3.1 s avg, 71 MB/s |

For reference, the single idle request in test 06 moved 483 MB in 2.3 ms
(~214 GB/s, 28 descriptors).

## What the numbers say

- **Aggregated wins by 10x at this load, against every disagg layout.** That is
  not the comparison the simulator made (it predicted disagg at 409 tok/s per
  GPU); it is a failure of the KV transfer path on this box.
- **Fractions are not the cause.** B (one GPU, MPS caps) and C (two whole GPUs,
  no MPS) collapse the same way. B even matched C's absolute throughput on half
  the GPUs, so per GPU the fraction layout is about 1.5x better than the two-GPU
  layout. The ITL isolation disagg promises is real in both: 43 ms versus 77 ms
  aggregated, p99 45 versus 83.
- **The transfer path collapses with load, not with size.** Idle: 483 MB in
  2 ms. Concurrency 4: 220 MB in 3 s. Concurrency 32: 565 MB in 20 to 85 s.
  Descriptor counts grow with fragmentation (28 idle, thousands under load) and
  transfer time grows with descriptors and with decode step time, while
  bandwidth stays in the single-digit to tens of MB/s. This looks like
  per-descriptor progress that only advances a few descriptors per decode step,
  not a bandwidth limit.
- **None of the usual knobs changed it:** `UCX_RCACHE_MAX_UNRELEASED`, host
  namespaces (so UCX sees one host), and the recipe's rendezvous tuning all
  left transfer times in the same range. Prefill meanwhile sits at
  `Running: 0 reqs, Waiting: 26`, starved because its finished prefills hold KV
  until decode's pull completes.
- The Dynamo recipes contain 10 workers that set `UCX_TLS=^cuda_ipc` (cuda_ipc
  disabled) and route intra-node traffic over InfiniBand instead. That is
  consistent with same-node cuda_ipc being a known weak path, but it is an
  inference, not a confirmed cause.

## What this does NOT prove

- It does not show fraction disagg is slower than agg in principle; it shows the
  transfer path on this stack makes any single-node disagg slower than agg under
  load. The claim to test is still open.
- It does not identify the root cause in code. Candidates: vLLM's NIXL connector
  progressing transfers once per scheduler step, UCX cuda_ipc outstanding-copy
  limits, or the per-block descriptor layout. `UCX_LOG_LEVEL=info` produced no
  transport lines in the captured logs, so the selected transport is not
  confirmed either way.
- All engines ran `--enforce-eager`; decode ITL (38 to 44 ms for a 4B model) is
  worse than it would be with CUDA graphs, which also slows step-coupled
  progress. Same handicap in every run, so the comparison holds, but absolute
  numbers are pessimistic.
- HBM bandwidth contention between fractions is still unmeasured; the transfer
  problem hid it.

## Next steps

1. Raise the transfer behavior with the Dynamo/NIXL team with this setup:
   Qwen3-4B, 1P1D on one node, A100 NVLink, vLLM runtime 1.5.0, concurrency 32,
   4k prompts: transfers of 565 MB taking 20 to 85 s. Ask what single-node
   disagg is expected to use (cuda_ipc versus IB loopback) and whether the
   connector progresses transfers off the scheduler step.
2. Re-run B and C with whatever fix comes back, without `--enforce-eager`, and
   with `--block-size 128` ([`frac-disagg-4b-block128.yaml`](frac-disagg-4b-block128.yaml))
   to halve descriptors.
3. Only then compare against A, and add a noisy-neighbor measurement for the
   fraction layout (prefill busy, decode ITL).
