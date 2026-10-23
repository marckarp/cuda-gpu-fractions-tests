"""Standalone NIXL same-GPU transfer benchmark (no vLLM).

Spawns a target process holding a 565 MB GPU buffer, an optional busy process
(matmul loop on the same GPU), and an initiator that READs the buffer over NIXL/UCX
as either 28 large descriptors or 8640 x 64 KB descriptors, polling completion
either tightly or once per 40 ms (per-step style). Prints seconds per transfer.
"""
import os, sys, time, json, multiprocessing as mp

TOTAL = 565 * 1024 * 1024
BLOCK = 64 * 1024
META = "/tmp/nixlb"


def target(ready, stop):
    import torch
    from nixl._api import nixl_agent, nixl_agent_config
    buf = torch.ones(TOTAL, dtype=torch.uint8, device="cuda")
    a = nixl_agent("target", nixl_agent_config(backends=["UCX"]))
    a.register_memory(a.get_reg_descs([(buf.data_ptr(), TOTAL, 0, "")], "VRAM"))
    os.makedirs(META, exist_ok=True)
    open(f"{META}/target.meta", "wb").write(a.get_agent_metadata())
    json.dump({"ptr": buf.data_ptr()}, open(f"{META}/target.json", "w"))
    ready.set()
    while not stop.is_set():
        time.sleep(0.05)


def busy(stop):
    import torch
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    while not stop.is_set():
        for _ in range(20):
            x = (x @ x).clamp_(-1, 1)
        torch.cuda.synchronize()


def initiator(cases, out):
    import torch
    from nixl._api import nixl_agent, nixl_agent_config
    buf = torch.zeros(TOTAL, dtype=torch.uint8, device="cuda")
    a = nixl_agent("initiator", nixl_agent_config(backends=["UCX"]))
    a.register_memory(a.get_reg_descs([(buf.data_ptr(), TOTAL, 0, "")], "VRAM"))
    remote = a.add_remote_agent(open(f"{META}/target.meta", "rb").read())
    rptr = json.load(open(f"{META}/target.json"))["ptr"]
    results = []
    for ndesc, poll_ms in cases:
        size = TOTAL // ndesc
        ld = a.get_xfer_descs([(buf.data_ptr() + i * size, size, 0) for i in range(ndesc)], "VRAM")
        rd = a.get_xfer_descs([(rptr + i * size, size, 0) for i in range(ndesc)], "VRAM")
        h = a.initialize_xfer("READ", ld, rd, remote)
        t0 = time.perf_counter()
        a.transfer(h)
        polls = 0
        while True:
            s = a.check_xfer_state(h)
            polls += 1
            if s == "DONE":
                break
            if s == "ERR":
                raise RuntimeError("xfer ERR")
            if poll_ms:
                time.sleep(poll_ms / 1000)
        dt = time.perf_counter() - t0
        a.release_xfer_handle(h)
        covered = size * ndesc
        ok = bool((buf[:covered:max(1, covered // 97)] == 1).all().item())
        results.append({"ndesc": ndesc, "desc_kb": size // 1024, "poll_ms": poll_ms, "sec": round(dt, 4),
                        "MBps": round(TOTAL / 1e6 / dt, 1), "polls": polls, "ok": ok})
        buf.zero_()
    out.put(results)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    busy_mode = len(sys.argv) > 1 and sys.argv[1] == "busy"
    ready, stop = mp.Event(), mp.Event()
    t = mp.Process(target=target, args=(ready, stop)); t.start()
    ready.wait(120)
    b = None
    if busy_mode:
        b = mp.Process(target=busy, args=(stop,)); b.start(); time.sleep(3)
    out = mp.Queue()
    cases = [(28, 0), (8640, 0), (28, 40), (8640, 40)]
    i = mp.Process(target=initiator, args=(cases, out)); i.start()
    res = out.get(timeout=1800)
    i.join(); stop.set(); t.join()
    if b: b.join()
    print(f"mode={'busy' if busy_mode else 'idle'}")
    for r in res:
        print(r)
