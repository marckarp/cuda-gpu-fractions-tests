#!/usr/bin/env python3
"""Print a shadow-mode (standby-capable) prefill or decode pod, or the GMS pod for it.

    mkshadow.py gms                                  # GMS serving weights AND kv_cache, 19000 MiB
    mkshadow.py gms-prefill-kv                       # second GMS: kv_cache pool for the prefill group only
    mkshadow.py <prefill|decode> <pod-name> <sm-share> <engine-id>

Engines of one role share a lock file; the holder serves, the others wait
parked without a real KV cache and take over when the holder exits.
"""
import subprocess
import sys

import yaml

if sys.argv[1] == "gms":
    docs = list(yaml.safe_load_all(open("/opt/t08/21-gms-disagg-1gpu.yaml")))
    pod = next(d for d in docs if d["metadata"]["name"] == "gms")
    pod["spec"]["containers"][0]["args"] = ["--device", "0", "--verbose"]  # no --tag: weights + kv_cache
    for k in ("request", "limit"):
        pod["metadata"]["annotations"][f"nvidia.com/container.main.gpu-memory.{k}"] = "19000Mi"
    print(yaml.safe_dump(pod))
    sys.exit(0)

if sys.argv[1] == "gms-prefill-kv":
    # GMS allows one writer per (GPU, tag) socket. The active decode holds the
    # kv_cache writer lock, so a prefill shadow engine can never wake against the
    # same socket. Give the prefill group its own kv_cache server in its own
    # socket directory; the weights socket there is a symlink to the shared one.
    docs = list(yaml.safe_load_all(open("/opt/t08/21-gms-disagg-1gpu.yaml")))
    pod = next(d for d in docs if d["metadata"]["name"] == "gms")
    pod["metadata"]["name"] = "gms-prefill-kv"
    uuid = sys.argv[2]
    pod["spec"]["containers"][0]["args"] = [
        "--device", "0", "--tag", "kv_cache", "--verbose",
        "--socket-path", f"/opt/gms-prefill/gms_{uuid}_kv_cache.sock",
    ]
    for k in ("request", "limit"):
        pod["metadata"]["annotations"][f"nvidia.com/container.main.gpu-memory.{k}"] = "4000Mi"
    pod["spec"]["containers"][0]["volumeMounts"].append({"name": "gmsp", "mountPath": "/opt/gms-prefill"})
    pod["spec"]["volumes"].append({"name": "gmsp", "hostPath": {"path": "/opt/gms-prefill", "type": "Directory"}})
    print(yaml.safe_dump(pod))
    sys.exit(0)

role, name, share, engine_id = sys.argv[1:5]
pod = yaml.safe_load(subprocess.check_output([sys.executable, "/opt/t09/mkpod.py", role, name, share]))
c = pod["spec"]["containers"][0]
c["args"] += ["--gms-shadow-mode"]
# Shadow mode subtracts the shared weights itself when sizing the KV cache, so the
# PoC's original utilization applies (test 08's lowered values would shrink the cache).
i = c["args"].index("--gpu-memory-utilization")
c["args"][i + 1] = "0.80" if role == "prefill" else "0.75"
c["env"] += [
    {"name": "ENGINE_ID", "value": engine_id},
    {"name": "FAILOVER_LOCK_PATH", "value": f"/opt/gms/failover-{role}.lock"},
]
pod["metadata"]["labels"]["t09-engine-id"] = engine_id
if role == "prefill":
    for e in c["env"]:
        if e["name"] == "GMS_SOCKET_DIR":
            e["value"] = "/opt/gms-prefill"
    c["volumeMounts"].append({"name": "gmsp", "mountPath": "/opt/gms-prefill"})
    pod["spec"]["volumes"].append({"name": "gmsp", "hostPath": {"path": "/opt/gms-prefill", "type": "Directory"}})
if role == "decode":
    # A standby books its full cap although it holds almost nothing, so two decode
    # pods at 25000 MiB plus gms and prefill do not fit on the 80 GB card.
    for k in ("request", "limit"):
        pod["metadata"]["annotations"][f"nvidia.com/container.main.gpu-memory.{k}"] = "24000Mi"
print(yaml.safe_dump(pod))
