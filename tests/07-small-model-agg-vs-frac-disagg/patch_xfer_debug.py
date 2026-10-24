"""Instrument vLLM 0.28.0's NIXL pull path with [XFERDBG] log lines (decode side).

  python3 patch_xfer_debug.py /opt/vllm-patch/pull_worker.py /opt/vllm-patch/base_worker.py

pull_worker: at transfer(handle) record post time, request id, descriptor count.
base_worker: in get_finished log the gap since the previous call (polling cadence);
             in _pop_done_transfers log first PROC and DONE per handle with elapsed
             time and the number of polls it took.
"""
import sys

pw, bw = sys.argv[1], sys.argv[2]
MARK = "# [xferdbg]"

s = open(pw).read()
if MARK not in s:
    old = "            # Begin async xfer.\n            self.nixl_wrapper.transfer(handle)\n"
    assert s.count(old) == 1
    new = (
        "            # Begin async xfer.\n"
        "            self.nixl_wrapper.transfer(handle)  " + MARK + "\n"
        "            _dbg = getattr(self, '_xferdbg', None)\n"
        "            if _dbg is None:\n"
        "                _dbg = self._xferdbg = {}\n"
        "            _dbg[handle] = [time.monotonic(), request_id, len(local_block_descs_ids), 0, None]\n"
        "            logger.info('[XFERDBG] post req=%s handle=%s descs=%d inflight=%d', request_id, handle,\n"
        "                        len(local_block_descs_ids), len(_dbg))\n"
    )
    s = s.replace(old, new)
    open(pw, "w").write(s)
    print("patched", pw)
else:
    print("already patched", pw)

s = open(bw).read()
if MARK not in s:
    old1 = "        assert self.transfer_topo is not None\n        done_sending = self._get_new_notifs()\n"
    assert s.count(old1) == 1
    new1 = (
        "        assert self.transfer_topo is not None\n"
        "        _now = time.monotonic()  " + MARK + "\n"
        "        _prev = getattr(self, '_xferdbg_last_poll', None)\n"
        "        self._xferdbg_last_poll = _now\n"
        "        if _prev is not None and self._recving_transfers and (_now - _prev) > 0.2:\n"
        "            logger.info('[XFERDBG] poll gap %.3fs with %d reqs pending', _now - _prev, len(self._recving_transfers))\n"
        "        done_sending = self._get_new_notifs()\n"
    )
    s = s.replace(old1, new1)
    old2 = (
        "                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)\n"
        "                    if xfer_state == \"DONE\":\n"
    )
    assert s.count(old2) == 1
    new2 = (
        "                    xfer_state = self.nixl_wrapper.check_xfer_state(handle)\n"
        "                    _d = getattr(self, '_xferdbg', {}).get(handle)  " + MARK + "\n"
        "                    if _d is not None:\n"
        "                        _d[3] += 1\n"
        "                        if _d[4] is None and xfer_state == 'PROC':\n"
        "                            _d[4] = time.monotonic()\n"
        "                        if xfer_state == 'DONE':\n"
        "                            _t = time.monotonic()\n"
        "                            logger.info('[XFERDBG] done req=%s descs=%d polls=%d post->done=%.3fs post->firstproc=%s inflight=%d',\n"
        "                                        _d[1], _d[2], _d[3], _t - _d[0],\n"
        "                                        ('%.3f' % (_d[4] - _d[0])) if _d[4] else 'n/a', len(self._xferdbg))\n"
        "                            del self._xferdbg[handle]\n"
        "                    if xfer_state == \"DONE\":\n"
    )
    s = s.replace(old2, new2)
    open(bw, "w").write(s)
    print("patched", bw)
else:
    print("already patched", bw)
