"""PoC: disable NIXL's background progress thread in vLLM's connector so UCX cuda_ipc
handle opens happen on the engine thread that owns the CUDA context."""
import sys
p = sys.argv[1]; s = open(p).read()
for old in ("nixl_agent_config(backends=self.nixl_backends, capture_telemetry=True)",
            "nixl_agent_config(num_threads=num_threads, capture_telemetry=True)"):
    s = s.replace(old, old[:-1] + ", enable_prog_thread=False)")
open(p, "w").write(s); print("enable_prog_thread=False count:", s.count("enable_prog_thread=False"))
