# CUDA GPU Fractions: tests and learnings

Hands-on tests of CUDA GPU fractions (KAI-Scheduler + kai-gpu-fractioning + MPS)
with Dynamo, on a 2x A100-40GB test box. Goal: check what is actually enforced,
with evidence, and record every step so it can be reproduced.

## What we have proven

| Test | What | Result |
|---|---|---|
| [01](tests/01-smoke-half-gpu/RESULTS.md) | Two half-GPU pods on one GPU | PASS: 54/108 SMs each, even when a pod asks for 100% |
| [02](tests/02-socket-bypass/RESULTS.md) | Process that skips MPS | INCONCLUSIVE: needs kernel-level evidence |
| [03](tests/03-multi-gpu-symmetric/RESULTS.md) | Half of 2 GPUs per pod | PASS: two pods packed, a third refused |
| [04](tests/04-enforcement/RESULTS.md) | Memory and compute limits at 0.25 / 0.5 / 0.75 of 2 GPUs | PASS: allocation fails at the cap, speed tracks the share |
| [05](tests/05-uneven-shares/RESULTS.md) | Uneven shares, compute set apart from memory | PASS: 0.3 / 0.7 split, prefill 74 SMs + decode 32 SMs on one GPU |
| [06](tests/06-disagg-one-gpu/RESULTS.md) | Dynamo 1.5 disagg, prefill + decode as fractions of one GPU | PASS: requests served; NIXL KV transfer between the two fractions is CUDA IPC, 483 MB in 2.3 ms (~214 GB/s) |
| [07](tests/07-small-model-agg-vs-frac-disagg/RESULTS.md) | Qwen3-4B at concurrency 32: agg on 1 GPU vs fraction disagg on 1 GPU vs disagg on 2 GPUs (AIPerf) | NOT PROVEN: agg 374 tok/s; every disagg layout 38 to 54 tok/s because KV transfers take 6 to 85 s under load. Not a fractions problem (2 whole GPUs collapse the same way); fraction disagg matched 2-GPU disagg on half the hardware with better ITL (43 vs 77 ms) |

All of this is on the PR #147 fork, not the released stack (see
[Stack status](#stack-status-2026-10-22)).

## How to request a GPU fraction

Every fractional pod needs the same base settings. Only the annotations change.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-model
  labels:
    kai.scheduler/queue: default-queue        # a KAI queue
  annotations:
    gpu-fraction: "0.5"                       # <-- pick one of the options below
    nvidia.com/container.main.gpu-compute.mode: sm-sharing
spec:
  schedulerName: kai-scheduler                # KAI places the pod, not the default scheduler
  runtimeClassName: nvidia
  securityContext:
    runAsNonRoot: true                        # root can raise its own limits; keep non-root
    runAsUser: 1000
  containers:
  - name: main                                # annotations name this container
    image: ...
```

Do not also request `nvidia.com/gpu`. The annotations replace it.

### Option 1: same share of compute and memory

```yaml
gpu-fraction: "0.5"
```

Half the SMs and half the memory. The value must be between 0 and 1.

### Option 2: compute and memory set separately

```yaml
nvidia.com/container.main.gpu-compute.request: "0.7"
nvidia.com/container.main.gpu-memory.request: 10000Mi
nvidia.com/container.main.gpu-memory.limit: 10000Mi
```

Use this when a workload needs a different mix, for example prefill (compute
heavy) next to decode (memory heavy, for KV cache) on one GPU. Tested result:

| Pod | Compute | Memory | Got |
|---|---|---|---|
| prefill | 0.7 | 10000 MiB | 74 SMs, 10000 MiB |
| decode | 0.3 | 28000 MiB | 32 SMs, 28000 MiB |

If you set only memory, compute is filled in at the same share
(12000 MiB of 39936 usable = 0.3 compute).

### Option 3: more than one GPU

```yaml
gpu-fraction: "0.75"
gpu-fraction-num-devices: "2"
```

0.75 of each of 2 GPUs = 1.5 GPUs of capacity. See
[Using more than one GPU](#using-more-than-one-gpu).

### What you get

| Share | SMs (of 108) | Memory per GPU (MiB) |
|---|---|---|
| 0.25 | 26 | 9984 |
| 0.3 | 32 | 11980 |
| 0.5 | 54 | 19968 |
| 0.7 | 74 | 27955 |
| 0.75 | 80 | 29952 |

Memory = (40960 - 1024 MPS reserve) x share. SMs are the share of 108, rounded
down by the driver. Inside the pod, `nvidia-smi` and CUDA both report the
capped memory, so size vLLM's `--gpu-memory-utilization` against the slice.

## How GPU fractions work

Two projects do the work: **KAI-Scheduler** (the Kubernetes scheduler, one per
cluster) and **kai-gpu-fractioning** (agents on every GPU node). The NVIDIA
driver and GPU underneath apply the limits.

```
 your pod:   gpu-compute.request: "0.5"     gpu-memory.limit: 20480Mi
                              │
 ┌────────────────────────────▼────────────────────────────────────┐
 │ KAI-SCHEDULER (one per cluster)                                 │
 │ Picks a GPU with room, does the bookkeeping, writes the         │
 │ decision onto the pod. Enforces nothing itself.                 │
 └────────────────────────────┬────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────┐
 │ KAI-GPU-FRACTIONING (agents on every GPU node)                  │
 │                                                                 │
 │  fractiond   at container start: reads KAI's decision, injects  │
 │              the env vars and the pod's private MPS pipe        │
 │                                                                 │
 │  mpsd        runs the MPS server per GPU; gives each container  │
 │              its own MPS namespace with its compute limit       │
 │              stored on the server (this is PR #147)             │
 │                                                                 │
 │  (also: metricsd for metrics, operator to manage these agents)  │
 └────────────────────────────┬────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────┐
 │ NVIDIA DRIVER (r615) + GPU                                      │
 │  memory    allocation past the cap fails                        │
 │  compute   the MPS server limits how many SMs a client's work   │
 │            can use at once                                      │
 └─────────────────────────────────────────────────────────────────┘
```

Key facts:

- **Compute is enforced by the MPS server, not the app.** The limit is stored
  on the server for the container's namespace, so an app that asks for 100%
  still gets its share (tests 01, 03, 04).
- **The limit caps, it does not reserve.** A 30% pod can never use more than
  30% of the SMs at once, even when the GPU is otherwise idle. Idle SMs are
  not lent out.
- **Everything assumes the app goes through MPS.** Whether a process that
  skips MPS is blocked is not yet proven (test 02). Our GPUs run in `Default`
  compute mode, not `EXCLUSIVE_PROCESS`.
- **On the released stack, `sm-sharing` is cooperative.** The same annotation
  only sets an env var the app can change. Server-side enforcement needs the
  PR #147 fork.

## Disaggregated serving on one GPU with fractions

Fractions make a new disaggregation layout possible: prefill and decode as two
engines on the **same** GPU, each with its own SM share and memory cap.

```
 one A100-40GB
 ┌─────────────────────────────┬─────────────────────┐
 │ prefill engine              │ decode engine       │
 │ 60% SMs, 14 GB              │ 40% SMs, 25 GB      │
 │ compute heavy, little KV    │ bandwidth heavy,    │
 │                             │ holds the live KV   │
 └──────────────┬──────────────┴──────────┬──────────┘
                └── KV handoff: copy inside the GPU ───┘
                    (NIXL over CUDA IPC, no network)
```

### Why it is attractive

- **Isolation without a second GPU.** In aggregated serving a long prefill runs
  inside the same step as decode and stretches it. With fractions decode keeps
  its own SM budget, so inter-token latency stays flat. Measured (test 07,
  Qwen3-4B, 4k prompts, concurrency 32): ITL p99 45 ms split vs 83 ms aggregated.
- **The KV handoff is physically free.** Both engines sit on one GPU, so the
  transfer is a device-local copy: 483 MB in 2.3 ms when idle (test 06).
- **No network, no NIC.** Nothing leaves the card.

What it is not: more capacity. It is one GPU's SMs and HBM bandwidth, split. The
goal is the same tokens per GPU as aggregated serving with better tail latency,
not higher throughput. HBM bandwidth is not partitioned by MPS, so a busy prefill
still competes with decode for memory bandwidth; that cost is not measured yet.

### What we measured (tests 06 and 07)

| Layout | Output tok/s | ITL avg / p99 | TTFT avg |
|---|---|---|---|
| Aggregated, one whole GPU | **374** | 77 / 83 ms | 1.0 s |
| Prefill + decode as fractions, one GPU | 38 | **43 / 45 ms** | 96 s |
| Prefill + decode on two whole GPUs (NVLink) | 50 | 44 / 47 ms | 76 s |

Isolation works (ITL), but both disaggregated layouts lost to aggregated by
about 10x at this load, and the fraction layout matched the two-GPU layout on
half the hardware. The reason is not fractions.

### What is slow, in plain terms

When decode pulls a request's KV from prefill, vLLM gives NIXL a list of copy
instructions ("descriptors"), **one per KV block per layer**. A 4k-token prompt
becomes thousands of 64 KB copies, and under load the blocks are fragmented so
none of them merge. Over an RDMA NIC that is fine, the NIC processes thousands of
descriptors in hardware. Over CUDA IPC, the path used inside one GPU or across
NVLink, each descriptor costs about 0.3 ms on a busy GPU, and the aggregate tops
out around 250 MB/s no matter how many transfers run at once. With 32 requests
in flight at 565 MB each, that is 20 to 85 s per handoff, which is what vLLM
showed. The same bytes as 28 large descriptors move in 0.04 s on the same path
(standalone NIXL benchmark, `tests/07-.../nixl_bench.py`).

That was the first theory. A contiguous-allocation patch cut descriptors to 54 and
changed nothing (test 07, deep dive), and standalone NIXL transfers between the
same two pods are fast in every configuration tried, including vLLM's own NIXL
settings and a busy peer. What remains is something inside vLLM's NIXL connector
path that only shows under serving load; it needs instrumentation in the engine,
not more deployment variants. It hits every NVLink-only or same-GPU disaggregated
deployment, not just fractions.

### What has to change to make it fast

1. **Find and fix the slow pull inside vLLM's NIXL connector.** Descriptor
   coalescing alone does not do it (tested); the engine-side cause still has to
   be instrumented. Also note KAI does not yet allow two fractional containers in
   one pod, so prefill and decode must be separate pods today.
2. **Shared KV on the device: no copy at all.** On one GPU the ideal handoff is
   a pointer, not a transfer: one owner of the KV pool, prefill writes a block,
   decode references it by ID. Dynamo's GMS already does this pattern for
   weights. It is an engine-level change, larger than item 1.
3. **Elastic SM split.** MPS caps are fixed; a 60/40 split cannot lend idle SMs.
   Letting the split follow load keeps either side from idling.

Then measure the open question: decode ITL while prefill is saturated on the
same GPU (HBM contention). Until item 1 lands, do not deploy fraction
disaggregation for performance; it loses to aggregated serving at load.

## Using more than one GPU

A fraction is always **per GPU** and must be below 1.0 (`gpu-fraction: "1.5"`
is rejected by KAI's admission webhook). To get more than one GPU's worth, ask
for several GPUs with `gpu-fraction-num-devices`. The same share applies to
each GPU.

```
   1.5 GPUs of capacity = 0.75 x 2 GPUs
   ┌──────── GPU 0 ────────┐   ┌──────── GPU 1 ────────┐
   │ ███████████████░░░░░  │   │ ███████████████░░░░░  │
   │  this pod: 75%  free  │   │  this pod: 75%  free  │
   └───────────────────────┘   └───────────────────────┘
     80 of 108 SMs, 29952 MiB     80 of 108 SMs, 29952 MiB
```

Rules:

- **Same node only.** One pod runs on one machine, so all its GPUs are on that
  node. A model can still span nodes with several pods (Dynamo `multinode`,
  pipeline or tensor parallel), each with its own fraction on its own node.
- **Same share on every GPU.** "1 full GPU + half of another" has no YAML
  option. Use an even split instead (0.75 x 2), which leaves 0.25 of each GPU
  free for other pods.
- **Split the model to use it.** GPU memory does not pool. A TP=1 engine uses
  only one GPU, so the second GPU's share is wasted. Use TP=2 (or PP=2). If you
  only need a bit more KV cache, a bigger share of one GPU (e.g. 0.9 with
  TP=1) is simpler and avoids cross-GPU traffic.

Tested: test 03 (0.5 x 2, two pods packed, third refused) and test 04
(memory and compute limits at 0.25, 0.5, 0.75 x 2).

## Known gaps

- **Not released.** Compute enforcement needs the PR #147 fork, built from source.
- **Driver 615 required.** Most fleets are not on it yet.
- **Skipping MPS:** unproven either way (test 02).
- **Partition override:** on r615 a process can pick a neighbor's MPS partition
  when partitions are set by hand ([Sanjana's exp 08](https://github.com/scheckerNV/mps-multi-gpu/tree/main/experiments/08-two-workers-unequal-compute)).
  Driver r620 adds a per-namespace default that ignores the app's choice.
- **Different shares per GPU in one pod:** no YAML option.
- **No performance comparison yet.** Test 06 served a model, but there is no noisy-neighbor test and no
  full-GPU baseline.

## Stack status (2026-10-22)

| Piece | State |
|---|---|
| Driver mechanism (r615 / CUDA 13.4, MPS namespaces) | GA |
| KAI fraction scheduling (v0.18) | released |
| Memory enforcement | released |
| Server-enforced compute | NOT released, needs the PR #147 fork |
| Dynamo docs + examples | merged |

## Test environment

Brev `test-cuda-gpu-fractions` (GCP a2-highgpu-2g): 2x A100-SXM4-40GB,
Ubuntu 22.04.5, microk8s v1.35.6 / containerd 2.1.6, GPU Operator v26.7.1
(host driver mode), driver 615.71.09 / CUDA 13.4. Setup steps: [`setup/`](setup/).

Fork pins (a matched pair; never mix with upstream releases):

| Repo | PR | Commit |
|---|---|---|
| [KAI-Scheduler](https://github.com/kai-scheduler/KAI-Scheduler/pull/2368) | #2368 | `44d1d0a9` |
| [gpu-fractioning](https://github.com/kai-scheduler/gpu-fractioning/pull/147) | #147 | `af0544e4` |

## Layout

- `setup/`: box preparation, in order (driver, NRI, GPU operator, fork builds, charts)
- `tests/`: one directory per experiment, each with manifests and a RESULTS.md

Rule: every result records what was run, the evidence, and what it does NOT
prove. No conclusions beyond the data.

## References

- [Dynamo GPU sharing docs](https://github.com/ai-dynamo/dynamo/tree/main/docs/fern/pages/use-cases/gpu-sharing) ([PR #15584](https://github.com/ai-dynamo/dynamo/pull/15584))
- [scheckerNV/mps-multi-gpu](https://github.com/scheckerNV/mps-multi-gpu): TP2 multi-GPU fraction experiments
- Compute enforcement tracking: [gpu-fractioning#147](https://github.com/kai-scheduler/gpu-fractioning/pull/147), [#143](https://github.com/kai-scheduler/gpu-fractioning/issues/143), [#157](https://github.com/kai-scheduler/gpu-fractioning/issues/157), [KAI-Scheduler#2191](https://github.com/kai-scheduler/KAI-Scheduler/issues/2191)
- [NVIDIA MPS docs](https://docs.nvidia.com/deploy/mps/index.html)
