"""Runs forever inside one fraction pod. Every ~2 s prints the matmul rate, the SM
count and memory total CUDA reports, and whether a PROBE_MIB allocation succeeds.
Change the pod's limits from outside while this runs and watch the lines move."""
import os
import time

import torch

dev = "cuda:0"
n = 8192
a = torch.randn(n, n, device=dev, dtype=torch.float16)
b = torch.randn(n, n, device=dev, dtype=torch.float16)
flop = 2 * n**3
probe_mib = int(os.environ.get("PROBE_MIB", "12000"))

while True:
    t0 = time.time()
    it = 0
    while time.time() - t0 < 2:
        c = a @ b
        torch.cuda.synchronize()
        it += 1
    dt = time.time() - t0
    free, total = torch.cuda.mem_get_info()
    try:
        x = torch.empty(probe_mib * 1024 * 1024, dtype=torch.uint8, device=dev)
        del x
        torch.cuda.empty_cache()
        alloc = "ok"
    except Exception as e:  # noqa: BLE001
        alloc = "FAIL"
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(
        time.strftime("%H:%M:%S"),
        f"{it * flop / dt / 1e12:6.1f} TFLOP/s",
        f"sm={sm}",
        f"total={total >> 20}MiB free={free >> 20}MiB",
        f"alloc{probe_mib}MiB={alloc}",
        flush=True,
    )
