#!/usr/bin/env python3
"""Closed-loop streaming load with a per-request record, for timelines around a resize.

    loadgen.py <out.jsonl> <concurrency> <duration-seconds>

Each worker sends one chat request after another (about 3.5k prompt tokens, 128
output tokens, unique first words so nothing is a prefix-cache hit) and writes
one JSON line per request: start time, TTFT, end time, chunks received, error.
Standard library only, so it runs on the node without installing anything.
"""
import http.client
import json
import random
import sys
import threading
import time

out_path, conc, duration = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
FILLER = "the quick brown fox jumps over the lazy dog. " * 350
lock = threading.Lock()
out = open(out_path, "w")
t_end = time.time() + duration


def one(worker: int, seq: int) -> dict:
    rec = {"w": worker, "seq": seq, "start": time.time(), "ttft": None, "end": None, "chunks": 0, "err": None}
    body = json.dumps({
        "model": "Qwen/Qwen3-4B",
        "messages": [{"role": "user", "content": f"Request {random.random()} {worker}-{seq}. Summarize: {FILLER}"}],
        "max_tokens": 128, "min_tokens": 128, "ignore_eos": True, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    try:
        c = http.client.HTTPConnection("localhost", 8000, timeout=120)
        c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            rec["err"] = f"http {r.status}: {r.read(200).decode(errors='replace')}"
        else:
            done = False
            for line in r:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                try:
                    d = json.loads(payload)
                except ValueError:
                    continue
                if "error" in d:
                    rec["err"] = "stream error: " + str(d["error"])[:200]
                    break
                ch = d.get("choices") or []
                if ch and (ch[0].get("delta") or {}).get("content"):
                    if rec["ttft"] is None:
                        rec["ttft"] = time.time() - rec["start"]
                    rec["chunks"] += 1
            if not done and rec["err"] is None:
                rec["err"] = "stream ended without [DONE]"
        c.close()
    except Exception as e:  # noqa: BLE001
        rec["err"] = f"{type(e).__name__}: {e}"[:200]
    rec["end"] = time.time()
    return rec


def worker(i: int) -> None:
    seq = 0
    while time.time() < t_end:
        rec = one(i, seq)
        seq += 1
        with lock:
            out.write(json.dumps(rec) + "\n")
            out.flush()
        if rec["err"]:
            time.sleep(0.5)  # do not spin on a dead endpoint


threads = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
for t in threads:
    t.start()
for t in threads:
    t.join()
