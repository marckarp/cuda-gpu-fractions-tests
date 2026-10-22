# CUDA GPU Fractions — tests and learnings

Hands-on tests of CUDA GPU fractioning (KAI-Scheduler + kai-gpu-fractioning, MPS v3)
with Dynamo. Companion to:

- [Dynamo GPU sharing docs](https://github.com/ai-dynamo/dynamo/tree/main/docs/fern/pages/use-cases/gpu-sharing) (PR [#15584](https://github.com/ai-dynamo/dynamo/pull/15584))
- [scheckerNV/mps-multi-gpu](https://github.com/scheckerNV/mps-multi-gpu) — Sanjana's TP2 multi-GPU fraction experiments
- Compute enforcement: [gpu-fractioning PR #147](https://github.com/kai-scheduler/gpu-fractioning/pull/147), issues [#143](https://github.com/kai-scheduler/gpu-fractioning/issues/143) / [#157](https://github.com/kai-scheduler/gpu-fractioning/issues/157), [KAI-Scheduler #2191](https://github.com/kai-scheduler/KAI-Scheduler/issues/2191)

## Status of the stack (2026-10-22)

| Piece | State |
|---|---|
| Driver mechanism (r615 / CUDA 13.4, MPS v3 namespaces) | GA |
| KAI v0.18 fraction scheduling | released |
| Memory enforcement (driver per-process limit via container toolkit hook) | released |
| Server-enforced compute fractions | NOT released — PR #147 (fork builds required) |
| Dynamo docs + examples | merged |

Asymmetric per-GPU shares (e.g. 20/30 GiB or 0.5/0.75 across two GPUs for one pod)
have no supported API: the KAI annotations are one scalar per container, applied to
every GPU the pod gets. See Sanjana's repo for what manual asymmetric setup takes.

## Test environment

Brev `test-cuda-gpu-fractions` (GCP a2-highgpu-2g): 2× A100-SXM4-40GB, Ubuntu 22.04.5,
kernel 6.8.0-1070-gcp, microk8s v1.35.6 / containerd 2.1.6, GPU Operator v26.7.1
(host driver mode), driver 615.71.09 / CUDA 13.4.

Fork pins (matched pair — never mix with upstream releases):

| Repo | PR | Commit |
|---|---|---|
| KAI-Scheduler | [#2368](https://github.com/kai-scheduler/KAI-Scheduler/pull/2368) | `44d1d0a966465c69b1fd0d24b40b9a1ebb73699e` |
| gpu-fractioning | [#147](https://github.com/kai-scheduler/gpu-fractioning/pull/147) | `af0544e4dbd1d8e03ca14fc1e06d6265fabed99b` |

## Layout

- `setup/` — box preparation scripts, in order (driver, NRI, operator, fork builds, charts)
- `tests/` — test manifests and probes, one directory per experiment, each with a RESULTS.md
- `learnings/` — notes that don't fit a single test (MPS/EXCLUSIVE_PROCESS mechanics, symmetric vs asymmetric, enforcement status)

Every test directory records: what was run, exact manifests, pass/fail evidence, and
what it does NOT prove. No conclusions beyond the data.
