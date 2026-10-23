"""Standalone NIXL benchmark ACROSS two pods (target in the prefill pod, initiator in the
decode pod), metadata exchanged through the shared hostPath /opt/dyn-discovery.
Reproduces the engines' topology without vLLM.

  pod A:  python3 nixl_cross.py target            # holds 565 MB, serves reads, exits on 'done' file
  pod B:  python3 nixl_cross.py initiator          # READs it as 28 and 8640 descriptors, 1 and 4 in flight
"""
import os, sys, time, json

TOTAL = 565 * 1024 * 1024
META = "/opt/dyn-discovery/nixlb"
role = sys.argv[1]

import torch
from nixl._api import nixl_agent, nixl_agent_config

if role == "target":
    os.makedirs(META, exist_ok=True)
    for f in ("target.meta", "target.json", "done"):
        try: os.remove(f"{META}/{f}")
        except FileNotFoundError: pass
    buf = torch.ones(TOTAL, dtype=torch.uint8, device="cuda")
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
while not (os.path.exists(f"{META}/target.meta") and os.path.exists(f"{META}/target.json")):
    time.sleep(0.5)
    if time.time() - t0 > 300:
        raise SystemExit("no target metadata")
time.sleep(1)
buf = torch.zeros(TOTAL, dtype=torch.uint8, device="cuda")
NT = int(os.environ.get("NB_NUM_THREADS", "0")); POLL = float(os.environ.get("NB_POLL_MS", "40")) / 1000
cfg = nixl_agent_config(backends=["UCX"], num_threads=NT, capture_telemetry=True) if NT else nixl_agent_config(backends=["UCX"])
print("initiator config: num_threads=%d poll_ms=%.0f" % (NT, POLL*1000), flush=True)
a = nixl_agent("xinit", cfg)
a.register_memory(a.get_reg_descs([(buf.data_ptr(), TOTAL, 0, "")], "VRAM"))
remote = a.add_remote_agent(open(f"{META}/target.meta", "rb").read())
rptr = json.load(open(f"{META}/target.json"))["ptr"]
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
