# Changelog: multi-model frontends

All times UTC on 2026-11-10. No Dynamo or KAI code was changed.

## Changes to the box

- **Installed the Dynamo platform 1.5.0** with Helm into `dynamo-system` (operator,
  etcd, NATS), the same way as Dynamo's `examples/gpu-sharing/common/install-dynamo-platform.sh`
  but at 1.5.0 to match the runtime image. The disaggregation tests in `tests/` ran without the operator.
  It is still installed.
- Downloaded `Qwen/Qwen3-0.6B` into `/opt/hf-cache`.
- Created `/opt/t10`.
- Left at the end: no DGDs, no pods in `default`, GPU at 41 MiB.

## Timeline

1. First `helm repo add` failed on a DNS timeout; the retry worked.
2. Shapes A, A2, B, C, D deployed one after another, each checked with `check.sh`;
   A2, B, C and D also with `routing.sh`. Nothing failed.
3. The workers did not set `--enforce-eager` (unlike the disaggregation tests) and came up
   with CUDA graphs.

## Notes

- The operator chose Kubernetes discovery (`DYN_DISCOVERY_BACKEND=kubernetes`) by
  default; etcd and NATS were installed but not selected by these DGDs.
- `routing.sh` prints FAIL for a model a frontend is not supposed to see. In
  shape C those lines are the expected isolation, not a routing error.
