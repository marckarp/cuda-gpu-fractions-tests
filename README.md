# CUDA GPU Fractions — tests and learnings

Hands-on tests of CUDA GPU fractioning (KAI-Scheduler + kai-gpu-fractioning + MPS v3)
with Dynamo, on a 2× A100-40GB test box. Goal: verify what is actually enforced,
with evidence, and record every step so it can be reproduced.

## How GPU fractions work

You write two numbers on your pod: a compute share and a memory cap. There is
also a shorthand, `gpu-fraction: "0.5"`, meaning half of both; the scheduler
expands it into the explicit annotations below, which is what every later layer
reads. Four layers turn them into reality:

Two projects do the work: **KAI-Scheduler** (the Kubernetes scheduler, cluster-wide)
and **kai-gpu-fractioning** (agents on every GPU node). Below them sits the NVIDIA
driver, which is the only layer that can actually say no.

```
 your pod:   gpu-compute.request: "0.5"     gpu-memory.limit: 20480Mi
                              │
 ┌────────────────────────────▼────────────────────────────────────┐
 │ KAI-SCHEDULER (scheduler, one per cluster)                      │
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
 │              its own MPS namespace pinned to its SM share       │
 │              (server-side compute enforcement = PR #147)        │
 │                                                                 │
 │  (also: metricsd for metrics, operator to manage these agents)  │
 └────────────────────────────┬────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────┐
 │ NVIDIA DRIVER (r615) — the hard gates                           │
 │  memory cap     per-process limit; alloc over cap FAILS         │
 │  EXCLUSIVE_PROCESS   only the MPS server may open the GPU,      │
 │                      so no process can go around MPS            │
 └────────────────────────────┬────────────────────────────────────┘
                              │
 ┌────────────────────────────▼────────────────────────────────────┐
 │ GPU — 108 SMs, 40GB. Has no idea fractions exist.               │
 └─────────────────────────────────────────────────────────────────┘
```

Key facts:

- **Memory is enforced by the driver.** Even a process that escapes MPS entirely
  cannot exceed its memory cap.
- **Compute runs through MPS** because that is the only way multiple processes
  share a GPU in space instead of time-slicing it. In the released stack the SM
  share is cooperative (the client honors it); the PR #147 fork makes the server
  enforce it per container.
- **EXCLUSIVE_PROCESS must be set before the MPS server starts.** It locks the
  direct door so every process is forced through MPS.
- **Symmetric only.** One value per container, applied to every GPU the pod gets
  (`gpu-fraction-num-devices: "2"` spans two GPUs at the same share). There is no
  API for different shares per GPU — see
  [Sanjana's repo](https://github.com/scheckerNV/mps-multi-gpu) for what manual
  asymmetric setup takes and why it is fragile.

## Using more than one GPU

A fraction is always **per GPU** and must be below 1.0 (`gpu-fraction: "1.5"`
is rejected by KAI's admission webhook). To get more than one GPU's worth, ask
for several GPUs with `gpu-fraction-num-devices`, and the same fraction is
applied to each one:

```yaml
annotations:
  gpu-fraction: "0.75"              # share of EACH GPU
  gpu-fraction-num-devices: "2"     # how many GPUs
  nvidia.com/container.main.gpu-compute.mode: sm-sharing
```

```
   1.5 GPUs of capacity = 0.75 x 2 GPUs
   ┌──────── GPU 0 ────────┐   ┌──────── GPU 1 ────────┐
   │ ███████████████░░░░░  │   │ ███████████████░░░░░  │
   │  this pod: 75%  free  │   │  this pod: 75%  free  │
   └───────────────────────┘   └───────────────────────┘
     80 of 108 SMs, 29952 MiB     80 of 108 SMs, 29952 MiB
```

Rules:

- **Same node only.** One pod lives on one machine, so all its GPUs are on that
  node. A model can still span nodes by using several pods (Dynamo `multinode`,
  pipeline or tensor parallel), each with its own fraction on its own node.
- **Same share on every GPU.** "1 full GPU + half of another" has no YAML option.
  Use an even split instead (0.75 x 2), which leaves 0.25 of each GPU free for
  other pods.
- **The model must be split to use it.** GPU memory does not pool. A TP=1
  engine only uses one GPU, so the second GPU's share is wasted. Use TP=2 (or
  PP=2). If you need only a little more KV cache, a bigger share of ONE GPU
  (e.g. 0.9 with TP=1) is simpler and avoids cross-GPU traffic.
- **Size vLLM against the slice.** On this stack CUDA inside the pod reports the
  capped memory (29952 MiB at 0.75; measured in test 04), so
  `--gpu-memory-utilization` applies to the slice, not the 40 GB card. The
  Dynamo doc says CUDA reports the physical total; our measurement differs, so
  check `torch.cuda.mem_get_info()` in your pod before choosing the value.

Tested here: test 03 (0.5 x 2, two pods packed, third refused) and test 04
(memory and compute enforcement at 0.25, 0.5, 0.75 x 2).

## Stack status (2026-10-22)

| Piece | State |
|---|---|
| Driver mechanism (r615 / CUDA 13.4, MPS v3 namespaces) | GA |
| KAI fraction scheduling (v0.18) | released |
| Memory enforcement | released |
| Server-enforced compute | NOT released — needs the PR #147 fork |
| Dynamo docs + examples | merged |

## Test environment

Brev `test-cuda-gpu-fractions` (GCP a2-highgpu-2g): 2× A100-SXM4-40GB,
Ubuntu 22.04.5, microk8s v1.35.6 / containerd 2.1.6, GPU Operator v26.7.1
(host driver mode), driver 615.71.09 / CUDA 13.4.

Fork pins (a matched pair — never mix with upstream releases):

| Repo | PR | Commit |
|---|---|---|
| [KAI-Scheduler](https://github.com/kai-scheduler/KAI-Scheduler/pull/2368) | #2368 | `44d1d0a9` |
| [gpu-fractioning](https://github.com/kai-scheduler/gpu-fractioning/pull/147) | #147 | `af0544e4` |

## Layout

- `setup/` — box preparation, in order: driver, NRI, GPU operator, fork builds, charts
- `tests/` — one directory per experiment, each with manifests and a RESULTS.md
- `learnings/` — notes that span tests (MPS mechanics, enforcement status)

Rule: every result records what was run, the evidence, and what it does NOT
prove. No conclusions beyond the data.

## References

- [Dynamo GPU sharing docs](https://github.com/ai-dynamo/dynamo/tree/main/docs/fern/pages/use-cases/gpu-sharing) ([PR #15584](https://github.com/ai-dynamo/dynamo/pull/15584))
- [scheckerNV/mps-multi-gpu](https://github.com/scheckerNV/mps-multi-gpu) — TP2 multi-GPU fraction experiments
- Compute enforcement tracking: [gpu-fractioning#147](https://github.com/kai-scheduler/gpu-fractioning/pull/147), [#143](https://github.com/kai-scheduler/gpu-fractioning/issues/143), [#157](https://github.com/kai-scheduler/gpu-fractioning/issues/157), [KAI-Scheduler#2191](https://github.com/kai-scheduler/KAI-Scheduler/issues/2191)
- [NVIDIA MPS docs](https://docs.nvidia.com/deploy/mps/index.html)
