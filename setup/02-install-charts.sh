#!/usr/bin/env bash
# Install the two fork charts on microk8s (after setup/01 and image import).
# Images were built locally and imported into containerd:
#   docker save <img> | microk8s ctr image import -
set -euo pipefail
T=smshare-44d1d0a-af0544e

cd ~/kai-forks/KAI-Scheduler
microk8s helm3 dependency build ./deployments/kai-scheduler
microk8s helm3 upgrade --install kai-scheduler ./deployments/kai-scheduler \
  -n kai-scheduler --create-namespace \
  --set global.gpuSharingMode=NvFractions \
  --set global.registry="local/kai-fork/kai-scheduler" \
  --set global.tag="$T" \
  --set binder.runtimeClassName=nvidia
microk8s kubectl -n kai-scheduler wait --for=condition=Available deploy --all --timeout=300s

cd ~/kai-forks/kai-gpu-fractioning
microk8s helm3 upgrade --install gpu-fractioning ./operator/charts \
  -n gpu-fractioning --create-namespace \
  --set images.operator.repository="local/kai-fork/kai-gpu-fractioning/operator" \
  --set images.fractiond.repository="local/kai-fork/kai-gpu-fractioning/fractiond" \
  --set images.metricsd.repository="local/kai-fork/kai-gpu-fractioning/metricsd" \
  --set images.mpsd.repository="local/kai-fork/kai-gpu-fractioning/mpsd" \
  --set images.operator.tag="$T" --set images.fractiond.tag="$T" \
  --set images.metricsd.tag="$T" --set images.mpsd.tag="$T" \
  --set fractioningAgent.criSocketPath=/opt/kai-runtime/containerd.sock \
  --set fractioningAgent.nriSocketPath=/var/snap/microk8s/common/run/nri/nri.sock
microk8s kubectl -n gpu-fractioning wait --for=condition=Available deploy --all --timeout=300s
sleep 20
microk8s kubectl -n gpu-fractioning rollout status ds --timeout=300s

# Verify: node condition True, MPS multiuser with default ns at 10%
microk8s kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="gpu-fractioning.nvidia.com/Ready")].status}{"\n"}{end}'
microk8s kubectl -n gpu-fractioning logs ds/gpu-fractioning-mpsd | grep -E "multiuser|active thread percentage" | head -4
