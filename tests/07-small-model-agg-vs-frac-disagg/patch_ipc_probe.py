"""Inside the engine, right after KV registration: probe whether the KV region is legacy-CUDA-IPC
capable (cuPointerGetAttribute IS_LEGACY_CUDA_IPC_CAPABLE=18, MEMORY_TYPE=2) and whether
cuIpcGetMemHandle succeeds, for the KV region and for a fresh torch allocation in the same process."""
import sys
p = sys.argv[1]; s = open(p).read()
MARK = "# [xferdbg-probe]"
if MARK not in s:
    anchor = "            logger.info('[XFERDBG] dumped agent metadata and %d regions to %s', len(caches_data), _n)\n"
    i = s.rfind(anchor); assert i > 0
    block = (
        "            import ctypes, torch  " + MARK + "\n"
        "            _cu = ctypes.CDLL('libcuda.so.1')\n"
        "            def _probe(tag, ptr):\n"
        "                mt = ctypes.c_uint(); cap = ctypes.c_int(); h = (ctypes.c_byte * 64)()\n"
        "                r1 = _cu.cuPointerGetAttribute(ctypes.byref(mt), 2, ctypes.c_void_p(ptr))\n"
        "                r2 = _cu.cuPointerGetAttribute(ctypes.byref(cap), 18, ctypes.c_void_p(ptr))\n"
        "                r3 = _cu.cuIpcGetMemHandle(h, ctypes.c_void_p(ptr))\n"
        "                logger.info('[XFERDBG] ipc-probe %s ptr=%s memtype(r=%d)=%d legacy_ipc_capable(r=%d)=%d cuIpcGetMemHandle=%d', tag, hex(ptr), r1, mt.value, r2, cap.value, r3)\n"
        "            _probe('kv_region0', int(caches_data[0][0]))\n"
        "            _t = torch.empty(64 << 20, dtype=torch.uint8, device='cuda'); _probe('fresh_torch', _t.data_ptr())\n")
    s = s[:i + len(anchor)] + block + s[i + len(anchor):]
    open(p, "w").write(s); print("patched probe")
else:
    print("already patched")
