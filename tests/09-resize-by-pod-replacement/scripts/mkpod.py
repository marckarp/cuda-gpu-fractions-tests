#!/usr/bin/env python3
"""Print a prefill or decode pod for the GMS layout at a given SM share.

    mkpod.py <role: prefill|decode> <pod-name> <sm-share e.g. 0.35> [grace-seconds]

Template: test 08's manifests/21-gms-disagg-1gpu.yaml (memory caps and
--gpu-memory-utilization unchanged, so only the SM share differs between pods).
"""
import sys

import yaml

role, name, share = sys.argv[1], sys.argv[2], sys.argv[3]
grace = int(sys.argv[4]) if len(sys.argv) > 4 else None
docs = list(yaml.safe_load_all(open("/opt/t08/21-gms-disagg-1gpu.yaml")))
pod = next(d for d in docs if d["metadata"]["name"] == role)
pod["metadata"]["name"] = name
pod["metadata"]["labels"]["t09-share"] = share
pod["metadata"]["annotations"]["nvidia.com/container.main.gpu-compute.request"] = share
if grace is not None:
    pod["spec"]["terminationGracePeriodSeconds"] = grace
print(yaml.safe_dump(pod))
