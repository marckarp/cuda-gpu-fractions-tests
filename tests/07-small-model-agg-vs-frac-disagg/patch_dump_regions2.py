"""Also dump after the per-layer registration path (register_kv_caches), which is the one actually used."""
import sys
p = sys.argv[1]; s = open(p).read()
MARK = "# [xferdbg-dump2]"
if MARK not in s:
    anchor = "        self._registered_descs.append(descs)\n"
    i = s.rfind(anchor); assert i > 0
    block = (
        "        try:  " + MARK + "\n"
        "            import os, json, socket\n"
        "            _d = '/opt/dyn-discovery/nixlb'; os.makedirs(_d, exist_ok=True)\n"
        "            _n = f'{_d}/engine-{socket.gethostname()}'\n"
        "            open(_n + '.meta', 'wb').write(self.nixl_wrapper.get_agent_metadata())\n"
        "            json.dump({'regions': [list(map(int, r[:2])) for r in caches_data]}, open(_n + '.json', 'w'))\n"
        "            logger.info('[XFERDBG] dumped agent metadata and %d regions to %s', len(caches_data), _n)\n"
        "        except Exception as _e:\n"
        "            logger.warning('[XFERDBG] dump failed: %s', _e)\n")
    s = s[:i + len(anchor)] + block + s[i + len(anchor):]
    open(p, "w").write(s); print("patched dump2 at offset", i)
else:
    print("already patched")
