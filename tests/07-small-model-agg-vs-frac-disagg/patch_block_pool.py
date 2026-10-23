"""PoC: contiguous block allocation for vLLM 0.28.0 block_pool.py.

When a request needs num_blocks >= MIN_RUN blocks at once (prompt allocation on
prefill, and the matching allocation on decode for a remote-prefill request),
take a run of consecutive block IDs from the free list instead of the first
num_blocks in eviction order. Consecutive IDs on both sides let NIXL merge the
per-block KV descriptors into one per layer. Falls back to the original
behaviour when no run is available. Incremental 1-block decode allocations are
left alone.

Usage: python3 patch_block_pool.py /opt/vllm-patch/block_pool.py
"""
import sys

path = sys.argv[1]
src = open(path).read()
MARK = "# [poc-contig]"
if MARK in src:
    print("already patched")
    sys.exit(0)

old = "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n"
assert src.count(old) == 1, src.count(old)
new = (
    "        ret: list[KVCacheBlock] | None = None  " + MARK + "\n"
    "        if num_blocks >= self._poc_min_run:\n"
    "            ret = self._poc_take_contiguous_run(num_blocks)\n"
    "        if ret is None:\n"
    "            ret = self.free_block_queue.popleft_n(num_blocks)\n"
)
src = src.replace(old, new)

helper = '''
    # [poc-contig] ------------------------------------------------------------
    _poc_min_run = 4
    _poc_stats = {"hit": 0, "miss": 0}

    def _poc_take_contiguous_run(self, num_blocks: int):
        """Return num_blocks free blocks with consecutive IDs, removed from the
        free list, or None if no such run exists. A block is in the free list
        iff its prev/next pointers are set (the queue resets them on removal)."""
        run: list[KVCacheBlock] = []
        blocks = self.blocks
        for i in range(1, len(blocks)):  # block 0 is the null block
            b = blocks[i]
            if b.prev_free_block is not None and b.next_free_block is not None:
                run.append(b)
                if len(run) == num_blocks:
                    for blk in run:
                        self.free_block_queue.remove(blk)
                    self._poc_stats["hit"] += 1
                    return run
            else:
                run = []
        self._poc_stats["miss"] += 1
        return None
    # ------------------------------------------------------------------------

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
'''
anchor = "\n    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:\n"
assert src.count(anchor) == 1
src = src.replace(anchor, helper)
open(path, "w").write(src)
print("patched", path)
