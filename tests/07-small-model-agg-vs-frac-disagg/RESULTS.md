# 07: can fraction disagg on one GPU beat aggregated serving for a small model?

**RESULT, 2026-10-24: throughput parity with aggregated serving on the same single
GPU, with 45% lower inter-token latency.** Qwen3-4B, A100, 4k-token prompts,
concurrency 32: aggregated 374 tok/s with ITL p99 83 ms; prefill+decode as two
fractions of the same GPU 377 tok/s with ITL p99 45 ms. TTFT is worse (4.7 s vs
1.0 s avg) with the 60/40 SM split used; that is the tunable trade-off.

Getting there took finding a real bug: every disaggregated layout on this box
first ran about 10x slower than aggregated because the KV transfer silently fell
back from CUDA IPC to TCP. Root cause and fix are in "The real bug" below.

## Question
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

## Isolating test: standalone NIXL, same GPU, no vLLM (2026-10-23)

[`nixl_bench.py`](nixl_bench.py), run inside the decode pod (whole A100, UCX 1.21,
`device(cuda_ipc/cuda)` selected): a target process holds 565 MB, an initiator
READs it over NIXL as 28 large descriptors or 8,640 x 64 KB descriptors, polling
completion tightly or once per 40 ms. "busy" adds a third process running a
matmul loop on the same GPU.

| GPU | descriptors | poll | time | MB/s |
|---|---|---|---|---|
| idle | 28 | tight | 2.2 ms | 267,000 |
| idle | 8,640 x 64 KB | tight | 76 ms | 7,750 |
| idle | 8,640 x 64 KB | every 40 ms | 123 ms | 4,800 |
| busy | 28 | tight | 10 ms | 59,000 |
| busy | 8,640 x 64 KB | tight | 0.55 s | 1,070 |
| busy | 8,640 x 64 KB | every 40 ms | 0.69 s | 860 |

(The first run printed `ok: False` for the 8,640 case; that was the script
checking bytes past the covered range because 8,640 does not divide 565 MB. Fixed
in the checked-in script; the transfer itself completed.)

Concurrent transfers (second run, busy GPU, poll every 40 ms, 565 MB each):

| in flight | descriptors each | wall time | aggregate MB/s |
|---|---|---|---|
| 1 | 8,640 x 64 KB | 2.9 s | 200 |
| 4 | 8,640 x 64 KB | 8.3 s | 290 |
| 8 | 28 x 20 MB | 0.04 s | 112,000 |

**Verdict (corrected):** the bottleneck is the per-descriptor cost of UCX cuda_ipc
on a busy GPU, multiplied by vLLM's per-block descriptor layout and by the
number of pulls in flight. Small-descriptor transfers top out at roughly 200 to
300 MB/s aggregate no matter how many run concurrently, while the same bytes in
large descriptors move at over 100 GB/s. With 32 requests in flight at 565 MB
each, 18 GB at 0.25 GB/s is about 70 s per transfer, which is what vLLM showed
(20 to 85 s). A single transfer looked acceptable (0.6 to 3 s); the aggregate is
what collapses. The earlier statement that NIXL/UCX was cleared was wrong: one
transfer is fine, the small-descriptor throughput ceiling is not.

Where it sits: vLLM's connector emits one descriptor per KV block per layer
(64 KB here, thousands per request once blocks fragment); UCX cuda_ipc pays a
fixed cost per descriptor that a busy GPU stretches; nothing coalesces them.
RDMA NICs process thousands of descriptors in hardware, which is why the same
layout performs on InfiniBand/EFA and why recipes disable cuda_ipc in favor of
IB even on one node. For same-GPU or NVLink-only disagg there is no NIC to hide
behind, so the fix has to be fewer, larger descriptors (contiguous block
allocation for a request, larger blocks, layer-contiguous layouts) or a
zero-copy handoff of the blocks themselves.

## Deep dive (2026-10-23, later): what is NOT the cause

A PoC and a series of isolating runs, all on the fraction layout unless noted.

**Contiguous block allocation PoC** ([`patch_block_pool.py`](patch_block_pool.py),
mounted over vLLM 0.28.0's `block_pool.py` via hostPath, manifests `*-contig.yaml`):
when an allocation needs >= 4 blocks it takes a run of consecutive block IDs so
NIXL can merge descriptors. It works: decode's transfer metrics went from
1,782 to 8,484 descriptors down to **54**. Throughput did not move: run B6 gave
41 tok/s, TTFT 87 s, ITL 43 ms, transfers still 17 s (33 MB/s). So descriptor
count was not the limiter inside vLLM.

**Standalone NIXL, every configuration tried, all fast** (`nixl_bench.py` inside
one pod, `nixl_cross.py` across the prefill and decode pods through the shared
hostPath):

| Setup | 28 descriptors | 8,640 x 64 KB | 4 in flight x 8,640 |
|---|---|---|---|
| same pod, GPU idle | 2 ms | 76 ms | |
| same pod, busy matmul in another process | 10 to 40 ms | 0.5 to 2.9 s | 8.3 s (aggregate 290 MB/s) |
| same pod, under MPS 40% cap | 41 ms | 104 ms | |
| same pod, busy compute inside the initiator process | 41 ms | 0.5 s | |
| **across the two pods** (engine topology) | 81 ms | 133 ms | 265 ms |
| across pods, `num_threads=4` like vLLM, 40 ms polls | 81 ms | 108 ms | 205 ms |
| across pods, target running a busy loop | 81 ms | 111 ms | 201 ms |

Worst standalone case is single-digit seconds; vLLM under load is 17 to 85 s.

**Other things excluded:** `UCX_RCACHE_MAX_UNRELEASED`, the recipe UCX tuning
(`UCX_RNDV_SCHEME=get_zcopy`, `UCX_RNDV_THRESH=0`, `UCX_MEMTYPE_CACHE=n`), host
net/IPC/PID namespaces for the pods, KV buffers on host (they are on `cuda`,
`use_host_buffer: False`), NIXL progress-thread config (`num_threads=4` is fast
standalone). UCX's own endpoint summary in the engine shows the peer as
intra-node with a cuda_ipc lane: `intra-node cfg#2 rma_am(tcp/eth0) ...
device(cuda_ipc/cuda) am(tcp/eth0 cuda_ipc/cuda)`.

**Two findings along the way:**
- KAI refuses one pod with two fractional containers ("kai doesn't support
  multiple containers with fractional GPU requests"). Prefill and decode as two
  containers in one pod is the natural shape for intra-GPU disaggregation; today
  it has to be two pods.
- Running the two fraction pods with `hostNetwork/hostIPC/hostPID` made the
  decode pull fail outright (`NIXL_ERR_INVALID_PARAM` from `make_prepped_xfer`).

**Where this leaves the root cause:** inside vLLM's NIXL connector path under
load, specifically something that only appears when the engine is serving many
requests. The transport, the topology, the descriptor layout and the UCX
configuration have each been shown fast in isolation with the same bytes, the
same pods and the same GPU state. The next step is instrumentation inside the
connector (timestamps per transfer handle from post to first `PROC` to `DONE`,
plus UCX statistics in the engine process), which is a vLLM debugging session
rather than a deployment experiment.

## The real bug: identical PIDs in two containers break CUDA IPC

vLLM's EngineCore process gets the **same PID in both containers** (348 in both
pods here, because each container has its own PID namespace and the process tree
is identical). CUDA IPC handles carry the exporting process's PID. When decode
(PID 348) opened a handle exported by prefill (also PID 348), the driver treated it
as the same process and `cuIpcOpenMemHandle` failed (`invalid device context`, or
`out of memory` when the GPU was nearly full). UCX then silently fell back to
software-emulated RMA over TCP at ~175 MB/s. Nothing logged an error at INFO
level; only `UCX_LOG_LEVEL=debug` showed it.

Proof (same deployment, one change): shifting decode's EngineCore to PID 549 by
forking 200 throwaway processes before `exec` in the container command:

| | post->done for one 560 MB transfer | IPC open errors |
|---|---|---|
| decode EngineCore PID 348 (= prefill) | 3.3 s | 1 |
| decode EngineCore PID 549 | **0.015 s** | 0 |

This is not specific to fractions: the two-whole-GPU runs (C, C2, C3, C5) had the
same PID collision and the same collapse. Any two containerized workers with
identical process trees on one node that talk over cuda_ipc can hit it.

Second constraint on fractions: after the PID fix, decode at 0.88 utilization
still failed at transfer post (`NIXL_ERR_INVALID_PARAM`); the IPC mapping of the
peer's memory appears to count against the importer's MPS memory cap, so decode
needs headroom. Prefill 0.80 / decode 0.75 worked.

Manifest with both: [`frac-disagg-4b-pidfix-headroom.yaml`](frac-disagg-4b-pidfix-headroom.yaml)
(no vLLM patches at all). The debugging patches used along the way are checked
in for reference: `patch_block_pool.py` (contiguous allocation, not needed),
`patch_xfer_debug.py` (transfer timing), `patch_ipc_probe.py`, `patch_dump_regions*.py`,
`nixl_bench.py`, `nixl_cross.py`.

## Final comparison (B21 vs A)

| @ concurrency 32 | Output tok/s | tok/s per GPU | ITL avg / p99 | TTFT avg / p99 | KV transfer |
|---|---|---|---|---|---|
| A: aggregated, one whole GPU | 374 | 374 | 77 / 83 ms | 1.0 / 6.4 s | n/a |
| **B21: fraction disagg, one GPU (60/40 SMs), PID fix + headroom** | **377** | **377** | **43 / 45 ms** | 4.7 / 9.2 s | 565 MB in 61 ms avg, 9.3 GB/s |

Same GPU, same model, same load: equal throughput, inter-token latency nearly
halved and flat (p99 45 ms vs 83 ms). TTFT is the cost of giving prefill only 60%
of the SMs at this load; a different split or a second prefill fraction would
move that trade-off. AISimulate's whole-GPU verdict ("agg wins per GPU, disagg
wins ITL") is beaten on the per-GPU axis and matched on the ITL axis.

Caveats: single run per config; all engines `--enforce-eager`; HBM contention is
folded into these numbers rather than measured separately; 4B model only.

## Next steps

1. Report the PID-collision bug (UCX/NIXL/CUDA IPC in containers) with the A/B above; a robust fix belongs in UCX cuda_ipc (compare PID namespace, not bare PID) or NIXL. Also hand over the earlier reproducer:
   Qwen3-4B, 1P1D on one node, A100 NVLink, vLLM runtime 1.5.0, concurrency 32,
   4k prompts: transfers of 565 MB taking 20 to 85 s. Ask what single-node
   disagg is expected to use (cuda_ipc versus IB loopback) and whether the
   connector progresses transfers off the scheduler step.
2. Re-run B and C with whatever fix comes back, without `--enforce-eager`, and
   with `--block-size 128` ([`frac-disagg-4b-block128.yaml`](frac-disagg-4b-block128.yaml))
   to halve descriptors.
3. Only then compare against A, and add a noisy-neighbor measurement for the
   fraction layout (prefill busy, decode ITL).
