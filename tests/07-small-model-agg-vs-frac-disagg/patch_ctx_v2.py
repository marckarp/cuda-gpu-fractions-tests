"""PoC v2: bind a CUDA context to the connector's handshake thread and to the posting thread.
cudaSetDevice alone does not make the primary context current for a thread; a context-requiring
call (a tiny allocation) does."""
import sys
pw, bw = sys.argv[1], sys.argv[2]
s = open(pw).read()
old = ('            import torch  # [xferdbg-ctx]\n'
       '            torch.cuda.set_device(torch.cuda.current_device())\n')
assert s.count(old) == 1, s.count(old)
s = s.replace(old,
       '            import torch, threading  # [xferdbg-ctx]\n'
       '            _tl = getattr(self, "_xferdbg_tl", None)\n'
       '            if _tl is None:\n'
       '                _tl = self._xferdbg_tl = threading.local()\n'
       '            if not getattr(_tl, "ctx", False):\n'
       '                torch.cuda.set_device(torch.cuda.current_device()); torch.zeros(1, device="cuda"); _tl.ctx = True\n'
       '                logger.info("[XFERDBG] bound CUDA context on posting thread %s", threading.current_thread().name)\n')
open(pw, "w").write(s)
s = open(bw).read()
old = ('            initializer=(lambda d: (__import__("torch").cuda.set_device(d), __import__("torch").cuda.current_stream()))  # [xferdbg-hsctx]\n'
       '            if False else (lambda d=__import__("torch").cuda.current_device(): (__import__("torch").cuda.set_device(d), __import__("torch").cuda.current_stream())),\n')
assert s.count(old) == 1, s.count(old)
s = s.replace(old,
       '            initializer=(lambda d=__import__("torch").cuda.current_device(): (__import__("torch").cuda.set_device(d), __import__("torch").zeros(1, device="cuda"))),  # [xferdbg-hsctx]\n')
open(bw, "w").write(s)
print("patched v2")
