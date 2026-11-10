# 08: live resizing of a fraction, and one copy of the weights for prefill + decode

**Run 2026-11-10** on the test box: one A100 80GB PCIe, driver
615.71.09, KAI-Scheduler + kai-gpu-fractioning forks, Dynamo `vllm-runtime:1.5.0`
(vLLM 0.28.0, gpu-memory-service 0.9.0), Qwen3-4B. One run per configuration.

## Answers

| # | Question | Answer |
|---|---|---|
| A | Change the SM share of a running process? | **No.** The MPS server refuses while a client is attached. |
| A2 | Change it without recreating the pod? | **Yes, by restarting only the process.** With no client attached the share can be set to anything from 10% to 100%; the next process gets it. |
| B | Change the memory cap of a running process? | **No.** Same refusal. |
| B2 | Change it for the next process in the same pod? | **Lower only.** It cannot go above the cap the container started with. |
| C | Prefill and decode share one copy of the weights (GMS)? | **Yes.** 23.0 GB used instead of 31.4 GB, same KV cache sizes, requests served. Needs lower `--gpu-memory-utilization` (see E). |
| D | Restart an engine at a new SM share: how long? | **About 67 s, with or without GMS.** Process startup dominates; the weights are 1.5 s either way once the file is in the page cache. |
| E | How is the shared mapping charged? | **Against each importing pod's cap**, and vLLM does not always see it, which makes it oversize the KV cache and run out of memory. |

## A / B: changing limits on a running process

Pod `live-probe` (30% SMs, 10000 MiB) running `scripts/live_probe.py`: 88.7 TFLOP/s,
32 SMs. From the host:

```
nvidia-cuda-mps-control -p 3 namespace set <ns> --server=shared --active-thread-percentage=70
  -> Failed to execute NAMESPACE SET
  server log: Namespace 1 has clients attached, rejecting active thread percentage update
nvidia-cuda-mps-control -p 3 namespace set <ns> --server=shared --pinned-memory-limit=20G
  -> server log: Namespace 1 has clients attached, rejecting pinned memory limit setting
```

The rate stayed at 88.7 TFLOP/s throughout. The same command on an empty namespace
succeeds, so the refusal is specifically about attached clients.

## A2 / B2: same pod, new process

Pod `resize-shell` (30% SMs, 10000 MiB) runs `sleep infinity`; the probe
(`scripts/stage.py`) is started with `kubectl exec`, exits, the namespace is changed,
and the probe is started again. Pod restarts: 0.

| Namespace share | Process env `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` | SMs | Matmul TFLOP/s |
|---|---|---|---|
| 30 (as deployed) | 30 (as deployed) | 32 | 84.5 |
| 10 | 30 | 10 | 26.5 |
| 70 | 30 | 32 | 84.4 |
| 70 | 70 | 74 | 166.5 |
| 70 | 100 | 74 | 166.0 |
| 100 | 100 | 108 | 218.1 |

The effective share is the lower of the namespace value and the process's own
environment variable. The container's environment still carries the value from pod
creation, so **raising** the share needs the new process to be started with the
variable overridden; **lowering** needs nothing.

Memory, container started with a 10000 MiB cap (CUDA reports total = 10000 MiB in every row):

| Namespace memory limit | Process env limit | Allocation | Result |
|---|---|---|---|
| unlimited (as deployed) | 20000M | 12000 MiB | FAIL |
| 20G | 20000M | 12000 MiB | FAIL |
| 3G | 10000M | 6000 MiB | FAIL |
| 3G | 10000M | 2000 MiB | ok |
| 10000M (control) | 10000M | 6000 MiB | ok |

So the namespace limit can tighten the cap for the next process, but nothing here
raises it above what the container was created with.

**Caveat: KAI does not know.** The namespace was set to 100% while KAI still had the
pod booked at 30%. Nothing prevented it, so anything that resizes this way must
keep the scheduler's booking in step or it can oversubscribe the GPU.

## C: one copy of the weights

Layout of PoC `20-frac-disagg-1gpu.yaml`, plus a `gms` pod (5% SMs, 10000 MiB) and
`--load-format gms` on both engines (`manifests/21-gms-disagg-1gpu.yaml`).

| | Baseline, no GMS | With GMS |
|---|---|---|
| GPU memory used, total | 31,390 MiB | **22,982 MiB** |
| prefill process | 11,940 MiB | 4,060 MiB |
| decode process | 19,408 MiB | 11,142 MiB |
| weights | 7.56 GiB in each engine | 7.56 GiB once, in the GMS pod |
| prefill KV cache | 2.96 GiB, 21,520 tokens | 2.87 GiB, 20,896 tokens |
| decode KV cache | 10.34 GiB, 75,312 tokens | 9.89 GiB, 72,032 tokens |
| `--gpu-memory-utilization` prefill / decode | 0.80 / 0.75 | 0.29 / 0.45 |
| weight load per engine | 32.5 s from cold disk, 1.2 s from page cache | 1.4 s (map) |
| both engines ready after `kubectl apply` | 104 s (cold disk) | 65 s (weights already in GMS) |

The first engine to connect loads from disk into GMS and publishes; the other maps
it (`[GMS] Read mode: imported 7.56 GiB`, `Model loading took 0.01 GiB memory`).
Requests are served; a 3,517-token prompt with 32 output tokens returns in about
0.75 s, so the prefill-to-decode KV handoff is on the fast path. The
"KV Transfer metrics" log line was not emitted in these short runs.

GMS shares memory as a file descriptor over a Unix socket (CUDA VMM
`cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle`), not the legacy
CUDA IPC handles that test 07 found broken when two containers have the same PID.
It worked here between pods in different MPS namespaces with no workaround.

## E: how the mapping is charged

**It counts against the importer's cap.** First GMS run, PoC utilization values
(0.80 / 0.75), evidence in `evidence/c-gms.out`: both engines died allocating the KV
cache. Decode: cap 24.41 GiB, process using 16.68 GiB, CUDA reporting 7.02 GiB free,
and a 490 MiB allocation fails. The missing ~7.5 GiB is the mapped weights.

**Whether vLLM sees it depends on request versus limit** (cause not established):

| decode pod annotations | CUDA free memory before / after mapping | KV cache at utilization 0.45 |
|---|---|---|
| request 25000Mi = limit 25000Mi | 24.23 / 23.32 GiB (mapping not visible) | 9.89 GiB |
| request 17500Mi < limit 25000Mi | 24.30 / 16.79 GiB (mapping visible) | 3.29 GiB |

In the first case vLLM sizes the KV cache as if the weights were free, so the
utilization must be lowered by hand by about `weights / cap`. In the second vLLM
subtracts them itself and the PoC's utilization values would be the right ones.

**Booking.** With request = limit, the three pods book 49,000 MiB (10,000 + 14,000 +
25,000) against 39,000 MiB for the baseline, even though 8.4 GB less is physically
used: the weights are booked three times. Setting the request below the limit is
accepted by KAI and fractiond (decode ran with 17500Mi / 25000Mi), which is the
obvious way to book own memory only. Whether KAI then packs by the request was
not verified.

## D: restart to resize

Decode deleted and recreated at 20% SMs while prefill keeps running:

| | delete + create to ready | weight load |
|---|---|---|
| With GMS | 67.5 s and 66.7 s | 1.4 s (map) |
| No GMS, model file in page cache | 67.9 s | 1.2 s |

The new share was applied (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20`, namespace 20.00).
About 36 s passes between process start and the model loader running (Python
imports and engine start-up), and about 10 s after it (KV allocation, registration).
For this 4B model GMS does not shorten a restart; it would for a cold page cache
(32.5 s here) or a much larger model.

## Unexplained: MPS server stopped accepting clients

After the first A2 sequence (probe killed by `timeout`, then `namespace set` calls
including a memory limit below the process's env limit), every new CUDA process
hung at context creation, including in a freshly created pod with no `set` at all.
The server logged the client as connected and ACTIVE. Restarting the `mpsd` pod
fixed it. One attempt to reproduce (kill a client mid-work, `set` immediately)
did not hang. Not root-caused.

## What this means for the disagg-on-one-GPU design

- **Resizing prefill against decode is a process restart, not a live knob**, and
  costs about 65 s per engine today. The pod, its namespace and its booking can
  stay; only the engine process has to go.
- **The 65 s is start-up, not weights.** The lever is a pre-started engine, not
  faster loading. Dynamo already has one: `--gms-shadow-mode` starts a standby
  engine that shares the weights, skips KV allocation and takes over when the
  active one exits. A standby started at the new SM share would turn a resize into
  a failover. Not tested here.
- **Shared weights are worth it for memory**: 8.4 GB of an 80 GB card for a 4B
  model, more for larger ones, as long as the booking issue above is handled.
- **GMS already has a `kv_cache` tag** (used by shadow mode to hold the KV pool
  outside the engine). That is the starting point for the shared KV arena in the
  zero-copy discussion: the server-owned, fd-shared pool exists; the missing part
  is block ownership handoff inside vLLM.

## For the owners

| # | Finding | Owner |
|---|---|---|
| 1 | `namespace set` is refused while clients are attached, for both share and memory. Is a live update planned? | MPS |
| 2 | Imported VMM mappings count against the importer's limit but are not always visible in free memory (request = limit case). | MPS / container toolkit |
| 3 | GMS under fractions oversizes the KV cache when the mapping is not visible. | Dynamo GMS |
| 4 | Nothing ties a namespace change to KAI's booking. | kai-gpu-fractioning |
| 5 | MPS server hang described above. | MPS |

## Not done

No throughput or latency benchmark with GMS (AIPerf); no test of shadow mode;
no test of whether KAI packs by request when request < limit; raising memory by
other means (toolkit hook) not explored; engines ran with `--enforce-eager` as in
the PoC.
