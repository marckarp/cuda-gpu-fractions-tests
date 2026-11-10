# Changelog: test 09

Every change and failure, in order. All times UTC on 2026-11-10.

## Changes made

**No Dynamo, vLLM, GMS or KAI code was changed.** Everything is deployment
configuration and box state.

Box:
- Created `/opt/t09` (test files) and `/opt/gms-prefill` (second GMS socket directory).
- `aiperf` pod from the PoC helpers, results in `/opt/aiperf-results`.
- Left at the end: no pods in `default`, GPU at 41 MiB, socket and discovery directories emptied.

Deployment, against test 08's `21-gms-disagg-1gpu.yaml`:
- `scripts/mkpod.py`: same prefill / decode pods with the SM share as a parameter.
- `scripts/mkshadow.py`, for the standby runs:
  - GMS without `--tag` (serves `weights` and `kv_cache`), cap 19000 MiB.
  - Engines get `--gms-shadow-mode`, `ENGINE_ID`, `FAILOVER_LOCK_PATH=/opt/gms/failover-<role>.lock`.
  - `--gpu-memory-utilization` back to the PoC's 0.80 / 0.75 (failure 3).
  - Decode cap 25000 to 24000 MiB (failure 4).
  - Prefill only: `GMS_SOCKET_DIR=/opt/gms-prefill`, plus a `gms-prefill-kv` pod
    (`--tag kv_cache --socket-path /opt/gms-prefill/...`, 5% SMs, 4000 MiB) and a
    symlink there to the shared weights socket (failure 7).

## Timeline and failures

1. **18:35 Overbooking check did not run.** I redirected stdin on a `kubectl apply -f -`
   and it received nothing. Rerun from a file at 18:41: KAI leaves the pod Pending.
2. **18:36 Kill-then-start run.** No failure of the test; the outages are the result.
3. **18:45 Shadow decode came up with a 3.2 GiB KV cache.** Test 08's lowered
   utilization is wrong for shadow mode, which subtracts the shared weights itself.
   Restored 0.75 and got 9.7 GiB.
4. **18:48 Standby decode Pending for 4 minutes.** Memory bookings summed to 84 GB
   on the 80 GB card. Cut GMS to 19000 MiB and decode to 24000 MiB.
5. **18:51 Every request 503 although all pods were up.** My mistake: I had emptied
   `/opt/dyn-discovery` while the frontend was running, and its file watch was
   gone (`FileStore delete_expired_files error`). Restarted the frontend. After
   that the directory was only emptied with the frontend down.
6. **19:10 Shadow prefill stuck at "waking".** Waiting for the `kv_cache` writer
   lock held by the active decode. Recorded as a finding.
7. **19:12 Second GMS for prefill's KV pool.** Lets the shadow prefill wake.
8. **19:13 All requests through the shadow prefill fail** (`KeyError: (0, 0)` in the
   NIXL handshake listener). Recorded as a finding; not worked around.

## Files

- `scripts/`: `mkpod.py`, `mkshadow.py`, `loadgen.py`, `timeline.py`,
  `r1-kill-then-start.sh`, `r2-shadow-swap.sh`, `r3-shadow-swap-forcekill.sh`,
  `r4-prefill-shadow-swap.sh`.
- `manifests/`: the generated pods as they were applied.
- `evidence/`: `r1.out` to `r4.out` (timelines), `r*.jsonl` (one line per request),
  `r*.events`, `aiperf-summary.txt`, `decode-a-startup.log`,
  `r3-decode-s3.log`, `r3-prefill-during-swap.log`, `prefill-shadow-keyerror.txt`.
