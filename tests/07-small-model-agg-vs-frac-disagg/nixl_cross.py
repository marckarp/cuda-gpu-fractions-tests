"""Standalone NIXL benchmark ACROSS two pods (target in the prefill pod, initiator in the
decode pod), metadata exchanged through the shared hostPath /opt/dyn-discovery.
Reproduces the engines' topology without vLLM.

  pod A:  python3 nixl_cross.py target            # holds 565 MB, serves reads, exits on 'done' file
  pod B:  python3 nixl_cross.py initiator          # READs it as 28 and 8640 descriptors, 1 and 4 in flight
"""
import os, sys, time, json

TOTAL = int(os.environ.get("NB_TOTAL_MB", "565")) * 1024 * 1024
META = "/opt/dyn-discovery/nixlb"
role = sys.argv[1]

ALLOC = os.environ.get("NB_ALLOC", "torch")
if ALLOC == "expandable":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from nixl._api import nixl_agent, nixl_agent_config

def alloc(fill):
    if ALLOC == "cumem":
        from vllm.device_allocator.cumem import CuMemAllocator
        a = CuMemAllocator.get_instance()
        with a.use_memory_pool(tag="kv_cache"):
            t = torch.empty(TOTAL, dtype=torch.uint8, device="cuda")
    else:
        t = torch.empty(TOTAL, dtype=torch.uint8, device="cuda")
    t.fill_(fill)
    return t
print("alloc mode:", ALLOC, flush=True)

if role == "target":
    os.makedirs(META, exist_ok=True)
    for f in ("target.meta", "target.json", "done"):
        try: os.remove(f"{META}/{f}")
        except FileNotFoundError: pass
    buf = alloc(1)
    a = nixl_agent("xtarget", nixl_agent_config(backends=["UCX"]))
    a.register_memory(a.get_reg_descs([(buf.data_ptr(), TOTAL, 0, "")], "VRAM"))
    open(f"{META}/target.meta", "wb").write(a.get_agent_metadata())
    json.dump({"ptr": buf.data_ptr()}, open(f"{META}/target.json", "w"))
    if os.environ.get("NB_TARGET_BUSY") == "1":
        import threading
        def spin():
            x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
            while not os.path.exists(f"{META}/done"):
                for _ in range(20):
                    x = (x @ x).clamp_(-1, 1)
                torch.cuda.synchronize()
        threading.Thread(target=spin, daemon=True).start()
        print("target busy loop started", flush=True)
    print("target ready", flush=True)
    t0 = time.time()
    while not os.path.exists(f"{META}/done") and time.time() - t0 < 600:
        time.sleep(0.2)
    print("target exiting", flush=True)
    sys.exit(0)

# initiator
t0 = time.time()
TM, TJ = f"{META}/target.meta", f"{META}/target.json"
if os.environ.get("NB_META"):
    TM, TJ = os.environ["NB_META"] + ".meta", os.environ["NB_META"] + ".json"
while not (os.path.exists(TM) and os.path.exists(TJ)):
    time.sleep(0.5)
    if time.time() - t0 > 300:
        raise SystemExit("no target metadata")
time.sleep(1)
LOCAL = int(os.environ.get("NB_LOCAL_MB", "0")) * 1024 * 1024 or TOTAL
TOTAL_REMOTE = TOTAL
TOTAL = LOCAL
buf = alloc(0)
NT = int(os.environ.get("NB_NUM_THREADS", "0")); POLL = float(os.environ.get("NB_POLL_MS", "40")) / 1000
cfg = nixl_agent_config(backends=["UCX"], num_threads=NT, capture_telemetry=True) if NT else nixl_agent_config(backends=["UCX"])
print("initiator config: num_threads=%d poll_ms=%.0f" % (NT, POLL*1000), flush=True)
a = nixl_agent("xinit", cfg)
a.register_memory(a.get_reg_descs([(buf.data_ptr(), TOTAL, 0, "")], "VRAM"))
remote = a.add_remote_agent(open(TM, "rb").read())
_j = json.load(open(TJ))
if "regions" in _j:
    rptr, rlen = _j["regions"][0]
    TOTAL = min(TOTAL, rlen); print("engine region 0:", hex(rptr), rlen, "reading", TOTAL, flush=True)
else:
    rptr = _j["ptr"]
if os.environ.get("NB_MODE") == "prepped":
    # vLLM's pattern: prepped descriptor lists of 64 KB blocks on both sides, then
    # make_prepped_xfer with index lists (arbitrary order under load).
    import random
    BLK = 64 * 1024
    nblk = TOTAL // BLK
    ldl = a.prep_xfer_dlist("NIXL_INIT_AGENT", [(buf.data_ptr() + i * BLK, BLK, 0) for i in range(nblk)], "VRAM")
    rdl = a.prep_xfer_dlist(remote, [(rptr + i * BLK, BLK, 0) for i in range(nblk)], "VRAM")
    for label, order, nconc in [("sorted", list(range(nblk)), 1), ("permuted", None, 1), ("permuted", None, 4)]:
        idx = order if order is not None else random.sample(range(nblk), nblk)
        hs = [a.make_prepped_xfer("READ", ldl, idx, rdl, idx) for _ in range(nconc)]
        t1 = time.perf_counter()
        for h in hs: a.transfer(h)
        pending = list(hs)
        while pending:
            pending = [h for h in pending if a.check_xfer_state(h) != "DONE"]
            time.sleep(POLL)
        dt = time.perf_counter() - t1
        for h in hs: a.release_xfer_handle(h)
        print({"mode": "prepped", "order": label, "blocks": nblk, "nconc": nconc, "sec": round(dt, 3),
               "MBps_total": round(nconc * TOTAL / 1e6 / dt, 1)}, flush=True)
    open(f"{META}/done", "w").write("1")
    sys.exit(0)
for ndesc, nconc in [(28, 1), (8640, 1), (28, 4), (8640, 4)]:
    size = TOTAL // ndesc
    ld = a.get_xfer_descs([(buf.data_ptr() + i * size, size, 0) for i in range(ndesc)], "VRAM")
    rd = a.get_xfer_descs([(rptr + i * size, size, 0) for i in range(ndesc)], "VRAM")
    hs = [a.initialize_xfer("READ", ld, rd, remote) for _ in range(nconc)]
    t1 = time.perf_counter()
    for h in hs: a.transfer(h)
    pending = list(hs)
    while pending:
        pending = [h for h in pending if a.check_xfer_state(h) != "DONE"]
        time.sleep(POLL)
    dt = time.perf_counter() - t1
    for h in hs: a.release_xfer_handle(h)
    covered = size * ndesc
    ok = bool((buf[:covered:max(1, covered // 97)] == 1).all().item())
    print({"nconc": nconc, "ndesc": ndesc, "desc_kb": size // 1024, "sec": round(dt, 3),
           "MBps_total": round(nconc * TOTAL / 1e6 / dt, 1), "ok": ok}, flush=True)
    buf.zero_()
open(f"{META}/done", "w").write("1")
