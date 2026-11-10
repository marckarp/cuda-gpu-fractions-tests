# Changelog: test 08

Every change to the box, the fraction stack, Dynamo or vLLM, and every failure
hit along the way, in order. All times UTC on 2026-11-10.

## Changes made

**No code was changed.** Dynamo, vLLM, GMS, KAI and the fraction agents ran as
shipped in `vllm-runtime:1.5.0` and the box's existing install. Everything below
is configuration or box state.

Test box:
- Created `/opt/t08` (test files) and `/opt/gms` (GMS socket directory), mode 777.
- Deleted the leftover pods from 8 days ago (`aiperf`, `frontend`, `dl-qwen3-4b`).
  The model cache in `/opt/hf-cache` was kept.
- Emptied `/opt/dyn-discovery` between runs (stale file-discovery entries).
- Restarted the `gpu-fractioning-mpsd` pod once (17:53) to recover a hung MPS
  server; see failure 3.
- Left at the end: no test pods, GPU at 41 MiB, only the `default` MPS namespace.

Deployment changes against the PoC manifest `20-frac-disagg-1gpu.yaml`
(all marked `# t08` in `manifests/21-gms-disagg-1gpu.yaml`):
- Added a `gms` pod: `python3 -m gpu_memory_service --device 0 --tag weights`,
  5% SMs, 10000 MiB, `GMS_SOCKET_DIR=/opt/gms` on a hostPath.
- Prefill and decode: `--load-format gms`, `GMS_SOCKET_DIR=/opt/gms`, the hostPath mounted.
- Decode SM share 0.40 -> 0.35 (the 0.05 goes to the `gms` pod).
- `--gpu-memory-utilization`: prefill 0.80 -> 0.29, decode 0.75 -> 0.45 (failure 4).

## Timeline and failures

1. **17:40 Live change refused.** `namespace set` on the running probe's namespace
   fails for share and for memory: "has clients attached, rejecting ...". Confirmed
   it works on an empty namespace.
2. **17:43 First process-restart run gave no output after the first `set`.** The
   probe's 9 s timeout was too short to tell slow from hung. Replaced the probe
   with `scripts/stage.py`, which prints each stage and exits by itself.
3. **17:44-17:52 MPS server hung for all new clients.** Every new CUDA process,
   in the test pod and in a freshly created one, stopped at context creation while
   the server logged it as ACTIVE. Preceded by: a probe killed by `timeout`,
   several `namespace set` calls, and a namespace memory limit (4G) below the
   process's env limit (10000M). Fixed by deleting the `mpsd` pod (daemonset
   recreated it). Did not reproduce in one later attempt.
   - My own mistakes during this: `pkill -f live_probe` inside `kubectl exec`
     killed its own shell (exit 137); a `grep` on an unreadable log path made a
     later command read the rest of the script from stdin; after the restart a
     stale namespace made my name lookup return two names. Fixed by adding
     `</dev/null` to every remote command and selecting the namespace by the
     pod's container ID.
4. **18:03 Both engines out of memory with GMS at the PoC utilization values.**
   The mapped weights count against each pod's cap but vLLM sized the KV cache
   without them (17.22 GiB in a 24.41 GiB cap). Lowered utilization to 0.29 / 0.45.
5. **18:06 Frontend returned 404 right after decode was recreated.** A race: the
   request was sent before the frontend had re-registered the model. Later runs
   wait for `/v1/models`.
6. **18:07 KV cache much smaller with request < limit** (3.29 GiB against 9.89 GiB
   at the same utilization). In that configuration the mapping is visible in
   CUDA's free memory and vLLM subtracts it. Confirmed with a controlled rerun
   (request = limit, decode restarted alone: 9.89 GiB again).
7. **18:12 Baseline restart is as fast as the GMS restart** (67.9 s against 67 s)
   once the model file is in the page cache. Recorded as a result, not a failure.

## Files

- `manifests/10-live-probe.yaml`, `11-resize-shell.yaml`: probe pods for A / A2.
- `manifests/21-gms-disagg-1gpu.yaml`: the GMS layout.
- `manifests/22-decode-resized.yaml`: decode at 20% SMs, request 17500Mi / limit 25000Mi.
- `manifests/23-decode-alone-req-eq-limit.yaml`: control for failure 6.
- `manifests/24-baseline-decode-resized.yaml`: no-GMS decode at 20% for the restart timing.
- `scripts/`: `live_probe.py`, `stage.py`, `a2-process-restart-resize.sh` (first
  version of the A2 run, superseded by the `stage.py` runs), `c-report.sh`, `longreq.py`.
- `evidence/c-baseline.out`, `c-gms.out` (the out-of-memory run), `c-gms2.out`.
