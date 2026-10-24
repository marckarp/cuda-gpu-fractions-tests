"""PoC v3: at add_remote_agent entry (handshake path: UCX ep creation + remote rkey unpack),
log the thread and whether a CUDA driver context is current, and bind one if not."""
import sys
p = sys.argv[1]; s = open(p).read()
MARK = "# [xferdbg-ara]"
if MARK not in s:
    old = ('    def add_remote_agent(\n        self,\n        nixl_agent_meta: NixlAgentMetadata,\n'
           '        remote_tp_rank: int = 0,\n        remote_tp_size: int = 1,\n    ) -> str:\n')
    assert s.count(old) == 1, s.count(old)
    new = old + (
        '        import ctypes, threading, torch  ' + MARK + '\n'
        '        _cu = ctypes.CDLL("libcuda.so.1"); _ctx = ctypes.c_void_p()\n'
        '        _cu.cuCtxGetCurrent(ctypes.byref(_ctx))\n'
        '        if not _ctx.value:\n'
        '            torch.cuda.set_device(torch.cuda.current_device()); torch.zeros(1, device="cuda")\n'
        '            _cu.cuCtxGetCurrent(ctypes.byref(_ctx))\n'
        '        logger.info("[XFERDBG] add_remote_agent on thread=%s native=%s ctx=%s",\n'
        '                    threading.current_thread().name, threading.get_native_id(), hex(_ctx.value or 0))\n')
    s = s.replace(old, new); open(p, "w").write(s); print("patched v3")
else:
    print("already patched")
