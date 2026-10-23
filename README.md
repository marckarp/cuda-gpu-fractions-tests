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
