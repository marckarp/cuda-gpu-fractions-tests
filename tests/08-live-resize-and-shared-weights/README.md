# Test 08: live resizing of a fraction, and one copy of the weights for prefill + decode

Status: **run 2026-11-10.** Findings are in [`RESULTS.md`](RESULTS.md); every change
made to the box and the deployment, and every failure, is in [`CHANGELOG.md`](CHANGELOG.md).
The box turned out to be one A100 80GB, not the 40GB card named below.

## Questions

| # | Question | Why it matters |
|---|---|---|
| A | Can the SM share of a **running** process be changed (no restart)? | Lets the planner move compute between prefill and decode by load instead of fixing the split at deploy time. |
| B | Can the memory cap of a running process be changed? | Same, for memory. Expected to be limited by vLLM allocating its KV pool once at startup even if the driver allows it. |
| C | Can prefill and decode, as two fraction pods on one GPU, map **one** copy of the model weights (Dynamo GMS)? | Frees one model's worth of memory (~8 GB estimated for Qwen3-4B) for KV cache. |
| D | With shared weights, how long does restarting an engine at a different SM share take? | The fallback form of resizing if A is "no". |
| E | How is the shared mapping charged against each pod's memory cap? | Test 07 saw CUDA IPC mappings charged to the importer; with shared weights both pods would be charged. |

## Method

Same box and stack as tests 06/07: the test box, A100-40GB,
driver 615 / CUDA 13.4, KAI-Scheduler + kai-gpu-fractioning forks, Dynamo
`vllm-runtime:1.5.0`, Qwen3-4B.

**A. Live SM change.** One fraction pod at 30% running a fixed-size matmul loop
that prints iterations/s every second. While it runs, change the share from
outside, three ways, and watch whether the rate of the already-running process
moves:
1. `nvidia-cuda-mps-control namespace set ...` on the pod's MPS namespace (r615).
2. Re-annotate the pod (`gpu-compute.request`) and see whether `fractiond`/`mpsd` react.
3. Classic `set_active_thread_percentage` on the server (expected: new clients only).

Also record whether a **new CUDA context in the same process** picks up the new
share (cheaper than a pod restart if the running context does not).

**B. Live memory change.** Same pod, allocate up to the cap, raise/lower the cap
from outside (toolkit hook value / namespace), retry the allocation.

**C + E. Shared weights.** Prefill and decode pods from
`../../poc-small-model-disagg-on-one-gpu/manifests/20-frac-disagg-1gpu.yaml`
with GMS enabled. Evidence: per-process and total GPU memory with and without
GMS, both engines serving, and how much each pod's cap is charged.

**D. Restart-to-resize.** With C working, delete decode and recreate it at a
different SM share; time from `kubectl apply` to first served token. Compare to
the same restart without GMS.

## Files

- `README.md` this plan
- `CHANGELOG.md` every change and failure, in order
- `RESULTS.md` findings with evidence (written after the runs)
- `manifests/`, `scripts/` added as the tests are built
