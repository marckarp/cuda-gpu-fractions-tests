import sys
p = sys.argv[1]; s = open(p).read()
MARK = "# [xferdbg-dump]"
if MARK not in s:
    old = "        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)\n        self._registered_descs.append(descs)\n"
    assert s.count(old) == 1, s.count(old)
    new = old + (
        "        try:  " + MARK + "\n"
        "            import os, json, socket\n"
        "            _d = '/opt/dyn-discovery/nixlb'; os.makedirs(_d, exist_ok=True)\n"
        "            _n = f'{_d}/engine-{socket.gethostname()}'\n"
        "            open(_n + '.meta', 'wb').write(self.nixl_wrapper.get_agent_metadata())\n"
        "            json.dump({'regions': [list(map(int, r[:2])) for r in caches_data]}, open(_n + '.json', 'w'))\n"
        "            logger.info('[XFERDBG] dumped agent metadata and %d regions to %s', len(caches_data), _n)\n"
        "        except Exception as _e:\n"
        "            logger.warning('[XFERDBG] dump failed: %s', _e)\n")
    s = s.replace(old, new); open(p, "w").write(s); print("patched dump")
else:
    print("already patched")
