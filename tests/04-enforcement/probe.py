"""Memory and compute enforcement probe for a fractional pod.

mem      fill each visible GPU in 256 MiB chunks until allocation fails
compute  fp16 8192x8192 matmul throughput (TFLOP/s) on each visible GPU
all      run both, once with the injected env and once with client overrides
         (memory limit raised to 40000M, active thread percentage 100)
"""
import json
import os
import subprocess
import sys
import time

CHUNK = 256 * 1024 * 1024


def mem():
    import torch
    for d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d)
        chunks, err = [], None
        try:
            while True:
                chunks.append(torch.empty(CHUNK, dtype=torch.uint8, device=d))
        except RuntimeError as e:
            err = str(e).splitlines()[0][:100]
        got = len(chunks) * CHUNK / 2**20
        del chunks
        torch.cuda.empty_cache()
        print(json.dumps({"test": "mem", "gpu": d, "cuda_reported_total_mib": total // 2**20,
                          "allocated_mib_before_fail": int(got), "error": err}), flush=True)


def compute(n=8192, iters=100):
    import torch
    for d in range(torch.cuda.device_count()):
        torch.cuda.set_device(d)
        a = torch.randn(n, n, device=d, dtype=torch.float16)
        b = torch.randn(n, n, device=d, dtype=torch.float16)
        for _ in range(10):
            a @ b
        torch.cuda.synchronize(d)
        t = time.time()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize(d)
        tf = 2 * n**3 * iters / (time.time() - t) / 1e12
        sms = torch.cuda.get_device_properties(d).multi_processor_count
        print(json.dumps({"test": "compute", "gpu": d, "sm_count": sms, "tflops": round(tf, 1)}), flush=True)


def run(mode, label, extra_env=None):
    env = dict(os.environ)
    env.update(extra_env or {})
    print(json.dumps({"run": label, "env_override": extra_env or {}}), flush=True)
    subprocess.run([sys.executable, __file__, mode], env=env, check=False)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "mem":
        mem()
    elif mode == "compute":
        compute()
    else:
        print(json.dumps({"injected": {k: os.environ.get(k) for k in
              ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
               "NVIDIA_GPU_MEMORY_LIMIT"]}}), flush=True)
        n = os.environ.get("NUM_GPUS", "2")
        big = ",".join(f"{i}=40000M" for i in range(int(n)))
        run("mem", "mem-default")
        run("mem", "mem-override", {"CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": big})
        run("compute", "compute-default")
        run("compute", "compute-override", {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100"})
