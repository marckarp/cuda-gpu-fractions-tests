# 09: resizing prefill and decode by replacing pods, under load

**Run 2026-11-10** on the test box: one A100 80GB PCIe, driver
615.71.09, KAI-Scheduler + kai-gpu-fractioning forks, Dynamo `vllm-runtime:1.5.0`
(vLLM 0.28.0, gpu-memory-service 0.9.0), Qwen3-4B, weights shared through GMS
(test 08 layout). No code changes. One run per scenario.

Purpose: a PoC of "change the prefill/decode SM split by killing a pod and bringing
up another", to find what Dynamo would have to add to make that usable.

## Bottom line

| Question | Answer |
|---|---|
| Does the split matter enough to be worth resizing? | **Yes.** Same hardware, same load: 480 tok/s with 0.55 s TTFT at one split, 295 tok/s with 3.8 s TTFT at another. |
| Does kill-then-start work? | **Mechanically yes, but it is an outage**: 96 s for prefill, 80 s for decode. Every request in that window is rejected. |
| Does a pre-started standby (shadow engine) fix it? | **For decode, mostly**: the standby serves 3 s after a hard kill, 7 to 8 s after a normal delete. In-flight requests are still lost, and throughput can stall about 30 s afterwards. |
| For prefill? | **No.** A shadow prefill engine cannot hand KV to decode in Dynamo 1.5.0. |
| Can the new pod start before the old one stops? | **Only with spare SM share.** KAI will not book more than 100%, and a parked standby holds its full SM share and memory cap. |

So resizing this way is possible today only as a short planned outage. The list of
what has to change is at the end.

## 1. The split matters

Closed-loop load, 16 clients, about 3.5k prompt tokens and 128 output tokens per
request, no prefix-cache hits (`scripts/loadgen.py`). Values are steady-state
10 s or 5 s buckets from the timelines in `evidence/r1.out` to `r3.out`.

| Prefill / decode SM share | Output tok/s | TTFT avg | ITL avg |
|---|---|---|---|
| 60% / 35% | 480 | 0.55 s | 29 ms |
| 45% / 35% | 383 | 1.6 to 2.0 s | 27 to 29 ms |
| 45% / 15% | 325 | 0.55 s | 45 ms |
| 35% / 35% | 320 | 3.4 s | 25 ms |
| 35% / 60% | 295 | 3.8 s | 24 ms |

The PoC's AIPerf workload (concurrency 32, 4k prompt, 128 output, 192 requests,
`evidence/aiperf-summary.txt`):

| Prefill / decode | Output tok/s | TTFT avg | ITL avg / p99 |
|---|---|---|---|
| 60% / 35% | 398 | 5.8 s | 30.6 / 33.1 ms |
| 35% / 60% | 251 | 11.9 s | 25.3 / 27.5 ms |

Prefill share drives throughput and TTFT for this prompt-heavy load; decode share
drives ITL. Giving decode more than it needs costs a third of the throughput.

## 2. Kill, then start (plain pods) — `evidence/r1.out`

Resize 60/35 to 35/60 under load. Prefill has to go first, because decode cannot
grow until prefill has released its share.

| Step | Gap with no successful request | What clients saw |
|---|---|---|
| Replace prefill (60% to 35%) | **96 s** (31 s for the old pod to terminate, 65 s for the new one to start) | `503 Model is not ready to serve requests yet` |
| Replace decode (35% to 60%) | **80 s** (8 s to terminate, 72 s to start) | `404`, then `503` |

- With no prefill worker the model is unavailable; decode does not fall back to
  doing its own prefill. With no decode worker the model disappears from the frontend.
- The frontend rejects immediately; nothing is queued for the worker that is on its way.
- The prefill pod ignored SIGTERM for the full 30 s grace period.
- After the swap the new split is in effect (table above).

## 3. Where the 65 s start-up goes — `evidence/decode-a-startup.log`

Decode, weights already in GMS, 76 s in this run (three pods starting together):

| Phase | Seconds |
|---|---|
| Pod scheduled and container started | 2 |
| Python imports in the main process, before the first log line | 15 |
| Model config resolved (two silent gaps around it) | 18 |
| Engine subprocess started (imports everything again) | 17 |
| Model object built, GMS connected | 11 |
| **Weights mapped from GMS** | **1.4** |
| KV cache, NIXL, warm-up, registration | 8 |

About 50 s passes before the GPU is touched. GMS already removed the part it can
remove; the rest only goes away with an engine that is already running.

## 4. Standby decode engine (shadow mode) — `evidence/r2.out`, `r3.out`

All decode engines run `--gms-shadow-mode` and share a lock file. The lock holder
serves; the others park without a real KV cache (about 0.5 GB physical) and wake
when the holder's process exits. GMS serves both `weights` and `kv_cache`.

Works inside fraction pods, across MPS namespaces, with the standby at a different
SM share than the active engine. Wake-up from "lock acquired" to "active": **0.27 s**.

| Swap | How the active pod was removed | Removal to standby active | Then |
|---|---|---|---|
| 35% to 15% | `kubectl delete` | 7 s | no completed request for about 25 s more |
| 15% to 35% | `kubectl delete` | 8 s | normal from the next bucket |
| 35% to 15% | `--grace-period=0 --force` | 3 s | no completed request for about 28 s more |
| 15% to 35% | `--grace-period=0 --force` | 2 s | most clients hung (below) |

What goes wrong around the swap:

- **Requests during the gap are rejected** (404 / 503), same as in section 2.
- **In-flight requests on the old decode are lost.** With a normal delete they
  fail quickly. With a hard kill the frontend never fails them; 26 client
  requests hung until the client's own 120 s timeout, so throughput stayed at a
  fraction until those timed out.
- **Prefill stalls for up to 30 s after a decode dies.** Prefill keeps the KV
  blocks of requests the dead decode never fetched. Its pool went to 85% with
  0 running and 5 waiting, and recovered only when the blocks expired
  (`Releasing expired KV blocks ... retrieved by 0 remote worker(s) before lease
  expired`, `evidence/r3-prefill-during-swap.log`).
- **A new standby takes about 70 s to park**, so swaps cannot be closer together than that.

## 5. What a standby costs

- **SM share.** KAI rejects any pod that takes the GPU past 100% of SMs
  (`not enough GPU compute ... no compute left under sm-sharing`). A parked
  standby holds its share the whole time, so that share is idle until the swap.
- **Memory booking.** A standby books its full cap. Two 25000 MiB decode pods plus
  GMS and prefill came to 84 GB on the 80 GB card and the standby stayed Pending;
  caps had to be cut to 24000 MiB.
- **Physical memory:** about 0.5 GB while parked.

## 6. Standby prefill engine: does not work — `evidence/r4.out`

1. **Blocked on the KV pool.** GMS allows one writer per GPU for `kv_cache`. The
   active decode holds it, so a shadow prefill hangs at "waking" forever.
2. **Worked around** with a second GMS instance serving only prefill's `kv_cache`
   in its own socket directory, the weights socket symlinked to the shared one
   (`scripts/mkshadow.py gms-prefill-kv`). With that the shadow prefill parks,
   takes the lock and reports active.
3. **Then every request through it fails.** Its NIXL handshake thread dies with
   `KeyError: (0, 0)` in `_nixl_handshake_listener`, decode times out fetching KV
   and fails the request (`Failing 1 request(s) due to KV load failure`),
   clients get 500 (`evidence/prefill-shadow-keyerror.txt`).

## 7. Against aggregated, without `--enforce-eager`

Added after the runs above. Aggregated engine on the whole GPU (PoC manifest
`10-agg-1gpu.yaml`) against prefill 60% / decode 35% with shared weights.
`evidence/aiperf-summary-vs-agg.txt`, `agg.out`, `aggg.out`, `g1.out`, `h1.out`.

| Layout | AIPerf, 32 concurrent: tok/s / TTFT avg / ITL avg (p99) | 16 clients: tok/s / TTFT / ITL |
|---|---|---|
| Aggregated, `--enforce-eager` | 398 / 1.05 s / 71.8 (78.9) ms | 375 / 0.40 s / 40 ms |
| Aggregated, CUDA graphs | 407 / 1.03 s / 70.7 (78.0) ms | 455 / 0.42 s / 32 ms |
| Split 60/35, `--enforce-eager` | 398 / 5.76 s / 30.6 (33.1) ms | 480 / 0.55 s / 29 ms |
| Split 60/35, CUDA graphs | 413 / 5.79 s / 28.2 (29.8) ms | 481 / 0.69 s / 28 ms |
| Split 60/35, CUDA graphs, prefill KV 83k tokens instead of 16k | 413 / 5.80 s / 28.1 (29.7) ms | 482 / 0.70 s / 28 ms |

- Throughput is about equal; ITL is 2.5x better for the split at 32 concurrent;
  TTFT is 5.6x worse at 32 concurrent and 0.27 s worse at 16.
- CUDA graphs help aggregated at 16 clients (375 to 455 tok/s) and barely change the split.
- A larger prefill KV cache does not help, so prefill is limited by its SM share.
- Giving prefill a larger KV cache needed a larger decode memory cap as well
  (40000 MiB): with the 25000 MiB cap every request failed with 500, the
  headroom rule from the first PoC (decode maps prefill's KV, and the mapping
  counts against decode's cap).
- CUDA graphs take about 0.6 GiB per engine; utilization was raised to 0.34 /
  0.48 to keep the KV caches (`scripts/graphs.py`).

## What Dynamo would need

| # | Gap | Evidence | Where |
|---|---|---|---|
| 1 | Shadow mode for a prefill (KV producer) engine: NIXL handshake breaks after wake | section 6 | Dynamo vLLM integration / GMS scratch-KV patches |
| 2 | More than one `kv_cache` pool per GPU, one per engine group | section 6 | GMS |
| 3 | Hold or queue requests for a few seconds while a role has no worker, instead of 404 / 503 | sections 2, 4 | frontend / router |
| 4 | Drain the old decode before it exits, or hand its in-flight requests to the successor | section 4 | worker lifecycle. `DYN_GMS_PERSIST_KV` (KV pages outlive the engine) looks like the base for the second option; not tested |
| 5 | Release prefill KV blocks when their decode is gone, instead of waiting for the lease | section 4 | vLLM NIXL connector / Dynamo |
| 6 | Fail streams whose worker vanished | section 4 (hard kill) | frontend |
| 7 | Decode falls back to its own prefill when no prefill worker exists | section 2 | router |
| 8 | An operator or planner action "replace this engine at share X" that does the standby, swap and clean-up | all manual here | operator / planner |
| 9 | A standby that does not hold SM share and a full memory cap, or a surge allowance during a swap | section 5 | KAI / kai-gpu-fractioning |
| 10 | KV sizing and booking with shared weights under fractions | test 08 | GMS, KAI |

A live SM update in MPS (refused today, test 08) would make 1 to 9 unnecessary for
compute resizing. Until then the shortest path to a usable resize is 3 + 5 + 8 for
decode, and 1 + 2 for prefill.

## Caveats

- One run per scenario; timings vary by a few seconds between runs.
- The load generator retries every 0.5 s after an error, so error counts in the
  timelines are inflated. Use the gap lengths, not the counts.
- Closed-loop load: a slower decode lowers the arrival rate at prefill, which is why
  45/15 shows a lower TTFT than 45/35.
- Shadow runs used a 24000 MiB decode cap and the PoC's utilization values; plain
  runs used 25000 MiB and test 08's lowered values. KV cache sizes are close
  (about 9.7 to 9.9 GiB for decode) but not identical.
- 80 GB card, not the PoC's 40 GB one; `--enforce-eager` as in the PoC.
- Not tested: `DYN_GMS_PERSIST_KV`, swaps under AIPerf, more than one model.
