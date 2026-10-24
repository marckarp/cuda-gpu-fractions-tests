"""PoC: give vLLM's NIXL handshake-initiator thread a current CUDA context. UCX cuda_ipc
opens the peer's IPC memory handle on that thread during the handshake; without a
context it fails with 'invalid device context' and UCX falls back to TCP."""
import sys
p = sys.argv[1]; s = open(p).read()
MARK = "# [xferdbg-hsctx]"
if MARK not in s:
    old = ('        self._handshake_initiation_executor = ThreadPoolExecutor(\n'
           '            # NIXL is not guaranteed to be thread-safe, limit 1 worker.\n'
           '            max_workers=1,\n')
    assert s.count(old) == 1, s.count(old)
    new = ('        self._handshake_initiation_executor = ThreadPoolExecutor(\n'
           '            # NIXL is not guaranteed to be thread-safe, limit 1 worker.\n'
           '            max_workers=1,\n'
           '            initializer=(lambda d: (__import__("torch").cuda.set_device(d), __import__("torch").cuda.current_stream()))  ' + MARK + '\n'
           '            if False else (lambda d=__import__("torch").cuda.current_device(): (__import__("torch").cuda.set_device(d), __import__("torch").cuda.current_stream())),\n')
    s = s.replace(old, new); open(p, "w").write(s); print("patched")
else:
    print("already patched")
