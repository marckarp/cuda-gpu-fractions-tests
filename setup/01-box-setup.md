# Box setup log — brev test-cuda-gpu-fractions (2026-10-22)

Fresh GCP a2-highgpu-2g brev instance (2× A100-SXM4-40GB). Shipped with driver
580.178.04 / CUDA 13.0, microk8s v1.35.6, GPU Operator v25.10.1, container
toolkit 1.20.1. Steps below follow the merged Dynamo doc
`build-gpu-fractioning-forks.md`, with microk8s-specific deltas we documented
previously.

## 1. Driver 580 → 615.71.09 (required: MPS v3 / CUDA 13.4)

The 615 driver is in stock Ubuntu jammy-updates — no NVIDIA repo needed on 22.04:

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-driver-615-open
# dkms builds nvidia/615.71.09 for the running kernel
sudo apt-get purge -y 'linux-modules-nvidia-580*' 'nvidia-*-580-server' \
  'libnvidia-*-580-server' 'nvidia-fabricmanager-580' 'libnvidia-nscq-580'
sudo reboot
```

Verify after reboot:

```bash
nvidia-smi | head -4                       # 615.71.09, CUDA UMD 13.4
nvidia-cuda-mps-control -p 3 namespace --help   # lists create/delete/list/set/get
```

Note: a2-highgpu-2g has NVLink but no NVSwitch, so fabric manager is not
required (it is on 8-GPU HGX boxes).

## 2. Enable NRI in microk8s containerd

fractiond is an NRI plugin; microk8s containerd 2.1.6 supports NRI but ships
with it off. The NRI socket must live in its OWN subdirectory (not next to
containerd.sock): the fractioning operator mounts each socket's parent dir,
and two sockets with the same parent produce duplicate volume mounts that
reject the DaemonSet.

```bash
sudo tee -a /var/snap/microk8s/current/args/containerd-template.toml <<'NRI'

[plugins."io.containerd.nri.v1.nri"]
  disable = false
  socket_path = "/var/snap/microk8s/common/run/nri/nri.sock"
  plugin_path = "/var/snap/microk8s/common/run/nri/plugins"
  plugin_config_path = "/var/snap/microk8s/common/run/nri/conf.d"
NRI
sudo mkdir -p /var/snap/microk8s/common/run/nri/{plugins,conf.d}
microk8s stop && microk8s start
ls -l /var/snap/microk8s/common/run/nri/nri.sock   # must exist

# CRI socket symlink in its own dir, for the same mount reason:
sudo mkdir -p /opt/kai-runtime
sudo ln -sf /var/snap/microk8s/common/run/containerd.sock /opt/kai-runtime/containerd.sock
```

## 3. GPU Operator v25.10.1 → v26.7.1

v26.7.1 ships the container toolkit `apply-cuda-memory-limits` CDI hook that
turns the memory cap into a driver-enforced limit. Do NOT use --reuse-values
across this jump (old values lack keys the new chart requires).

```bash
microk8s helm3 repo add nvidia https://helm.ngc.nvidia.com/nvidia
microk8s helm3 upgrade gpu-operator nvidia/gpu-operator --version v26.7.1 \
  -n gpu-operator-resources -f gpu-op-values.yaml
```

values (host driver mode + microk8s containerd wiring): see
`gpu-op-values.yaml` in this directory.

## 4. Clone forks at pinned commits and build

```bash
mkdir ~/kai-forks && cd ~/kai-forks
git clone https://github.com/kai-scheduler/KAI-Scheduler.git
git -C KAI-Scheduler fetch origin pull/2368/head
git -C KAI-Scheduler checkout 44d1d0a966465c69b1fd0d24b40b9a1ebb73699e
git clone https://github.com/kai-scheduler/gpu-fractioning.git kai-gpu-fractioning
git -C kai-gpu-fractioning fetch origin pull/147/head
git -C kai-gpu-fractioning checkout af0544e4dbd1d8e03ca14fc1e06d6265fabed99b

# fractioning: 4 images (operator, mpsd, fractiond, metricsd)
cd kai-gpu-fractioning
make docker-build DOCKER_REPO_BASE="local/kai-fork/kai-gpu-fractioning" \
  VERSION="smshare-44d1d0a-af0544e" PLATFORM=linux/amd64

# KAI: 15 images; default docker builder so they land in the local daemon
cd ../KAI-Scheduler
BUILDX_BUILDER=default make build \
  DOCKER_REPO_BASE="local/kai-fork/kai-scheduler" \
  DOCKER_TAG="smshare-44d1d0a-af0544e" DOCKER_BUILD_PLATFORM=linux/amd64
```

No registry: images are imported into microk8s containerd directly
(`docker save | microk8s ctr image import -`), charts use pullPolicy
IfNotPresent. Avoids the localhost-registry/buildx pitfalls in the doc.

## 5. Install charts — see 02-install-charts.sh (next step)
