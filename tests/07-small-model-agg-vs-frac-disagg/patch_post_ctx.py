"""PoC: make the thread that posts the NIXL READ hold a current CUDA context, so UCX
cuda_ipc can open the peer's IPC handle (otherwise: 'invalid device context' -> tcp)."""
import sys
p = sys.argv[1]; s = open(p).read()
MARK = "# [xferdbg-ctx]"
if MARK not in s:
    old = "        # Prepare transfer with Nixl.\n        handle = None\n        try:\n            handle = self.nixl_wrapper.make_prepped_xfer(\n"
    assert s.count(old) == 1, s.count(old)
    new = ("        # Prepare transfer with Nixl.\n        handle = None\n        try:\n"
           "            import torch  " + MARK + "\n"
           "            torch.cuda.set_device(torch.cuda.current_device())\n"
           "            handle = self.nixl_wrapper.make_prepped_xfer(\n")
    s = s.replace(old, new); open(p, "w").write(s); print("patched")
else:
    print("already patched")
