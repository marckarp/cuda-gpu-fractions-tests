#!/usr/bin/env bash
# Test 08 A2/B2. Run on the box. Pod `resize-shell` stays up; between probe runs
# (no CUDA client attached) the namespace limits are changed on the MPS server.
MC="sudo nvidia-cuda-mps-control -p 3"
kubectl delete pod live-probe --ignore-not-found --wait=true </dev/null >/dev/null 2>&1
kubectl apply -f /opt/t08/11-resize-shell.yaml </dev/null
kubectl wait --for=condition=Ready pod/resize-shell --timeout=120s </dev/null
NS=$($MC namespace list --format=csv,noheader </dev/null | grep kai_ | cut -d, -f1 | tr -d ' ')
echo "namespace=$NS"
probe() { # $1 = label, rest = extra env assignments for the process
  local label=$1; shift
  echo "## probe: $label"
  kubectl exec resize-shell -- env "$@" timeout 9 python3 -u /opt/t08/live_probe.py </dev/null 2>&1 | grep TFLOP | tail -1
}
setns() { echo "## set $*"; $MC namespace set "$NS" --server=shared "$@" </dev/null; echo "rc=$?"; }

probe "as deployed (30%, 10000M)"
setns --active-thread-percentage=70
probe "after set 70%"
setns --active-thread-percentage=10
probe "after set 10%"
setns --active-thread-percentage=30
probe "back at 30%"

setns --pinned-memory-limit=20G
probe "namespace memory 20G, process env unchanged (0=10000M)"
probe "namespace memory 20G, process env 0=20000M" CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=20000M
setns --pinned-memory-limit=4G
probe "namespace memory 4G, process env unchanged (0=10000M)" PROBE_MIB=6000
echo "## namespace now"; $MC namespace list </dev/null
echo "## pod restarts: $(kubectl get pod resize-shell -o jsonpath='{.status.containerStatuses[0].restartCount}' </dev/null)"
