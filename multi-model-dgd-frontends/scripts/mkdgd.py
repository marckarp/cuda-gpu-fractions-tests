#!/usr/bin/env python3
"""Print a DynamoGraphDeployment with any number of frontends and model workers.

    mkdgd.py <dgd-name> <frontend-names, comma separated or "-"> <worker> [<worker> ...]

    worker = <component-name>:<model>:<sm-share>:<memory-MiB>[:<dynamo-namespace>[:<served-name>]]

Each worker is an aggregated vLLM engine in its own GPU fraction. If the fifth
field is given (and is not "-") the worker's DYN_NAMESPACE is set to it; if not,
the operator's default is left alone. A sixth field registers the model under
that name (--served-model-name), so many workers can serve the same weights.
"""
import sys

import yaml

IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0"
name, frontends = sys.argv[1], sys.argv[2]
components = []

if frontends != "-":
    for fe in frontends.split(","):
        components.append({
            "name": fe, "type": "frontend", "replicas": 1,
            "podTemplate": {"spec": {"containers": [{
                "name": "main", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
                "command": ["python3", "-m", "dynamo.frontend"], "args": ["--http-port", "8000"],
            }]}},
        })

for w in sys.argv[3:]:
    parts = w.split(":")
    comp, model, share, mem = parts[:4]
    env = [{"name": "HF_HOME", "value": "/opt/hf-cache"}, {"name": "HF_HUB_OFFLINE", "value": "1"}]
    if len(parts) > 4 and parts[4] != "-":
        env.append({"name": "DYN_NAMESPACE", "value": parts[4]})
    served = ["--served-model-name", parts[5]] if len(parts) > 5 else []
    components.append({
        "name": comp, "type": "worker", "replicas": 1,
        "podTemplate": {
            "metadata": {
                "labels": {"kai.scheduler/queue": "default-queue"},
                "annotations": {
                    "nvidia.com/container.main.gpu-compute.request": share,
                    "nvidia.com/container.main.gpu-memory.request": f"{mem}Mi",
                    "nvidia.com/container.main.gpu-memory.limit": f"{mem}Mi",
                    "nvidia.com/container.main.gpu-compute.mode": "sm-sharing",
                },
            },
            "spec": {
                "schedulerName": "kai-scheduler", "runtimeClassName": "nvidia",
                "securityContext": {"runAsNonRoot": True, "runAsUser": 1000, "runAsGroup": 1000},
                "containers": [{
                    "name": "main", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
                    "command": ["python3", "-m", "dynamo.vllm"],
                    "args": ["--model", model, "--gpu-memory-utilization", "0.80", "--max-model-len", "8192"] + served,
                    "env": env,
                    "volumeMounts": [{"name": "hf", "mountPath": "/opt/hf-cache"},
                                     {"name": "shm", "mountPath": "/dev/shm"}],
                }],
                "volumes": [{"name": "hf", "hostPath": {"path": "/opt/hf-cache", "type": "Directory"}},
                            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"}}],
            },
        },
    })

print(yaml.safe_dump({
    "apiVersion": "nvidia.com/v1beta1", "kind": "DynamoGraphDeployment",
    "metadata": {"name": name, "namespace": "default"},
    "spec": {"backendFramework": "vllm", "components": components},
}, sort_keys=False))
