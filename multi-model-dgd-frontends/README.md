# Multi-model: several models, one frontend or many, one DGD or many

**Run 2026-11-10.** One A100 80GB, driver 615.71.09, KAI-Scheduler +
kai-gpu-fractioning forks, Dynamo platform 1.5.0 (operator, Kubernetes discovery)
and `vllm-runtime:1.5.0`. Two models, Qwen3-4B and Qwen3-0.6B, each an aggregated
vLLM worker in its own GPU fraction of the same GPU (40% SMs / 14000 MiB and
10% SMs / 6000 MiB). One run per shape.

## Result

| Question | Answer |
|---|---|
| Does it need a frontend per model? | **No.** One frontend serves both models. |
| One frontend and several models in **one DGD**? | **Yes.** |
| Several frontends in one DGD? | **Yes.** Each frontend serves all the models of that DGD. |
| One frontend for models in **different DGDs**? | **Yes**, if the other DGD's worker sets a Dynamo namespace that starts with the frontend's prefix. Without that, each DGD's frontend sees only its own models. |
| Does each request reach the right model? | **Yes**, checked on the engines' own request counters in every shape. |

| Shape | Manifest | Frontends see | Routing check |
|---|---|---|---|
| A. One DGD, one frontend, two models, a Dynamo namespace per model | `a-one-dgd-one-frontend.yaml` | both models | not counted (A2 covers the same path) |
| A2. Same, workers left in the DGD's default namespace | `a2-no-namespaces.yaml` | both models | PASS |
| B. One DGD, two frontends, two models | `b-one-dgd-two-frontends.yaml` | both frontends: both models | PASS through each frontend |
| C. Two DGDs, each a frontend and one model | `c1.yaml`, `c2.yaml` | each frontend: only its own model; the other returns 404 | PASS for the model each frontend owns |
| D. C's first DGD plus a worker-only DGD for the second model, namespace `default-mm-c1-qwen06b` | `c1.yaml`, `d-worker-only-dgd.yaml` | the one frontend: both models | PASS |

## How routing was checked

`scripts/routing.sh` sends 12 requests naming Qwen3-4B and 7 naming Qwen3-0.6B
through a frontend, shuffled and 8 at a time, plus one for a model that does not
exist. It reads `vllm:request_success_total` from each worker pod's metrics port
before and after.

```
--- through mm-b-frontend1
    http status counts:      19 200
    unknown model -> http 404
    mm-b-qwen06b-...: +7 requests (expected +7) PASS
    mm-b-qwen4b-...:  +12 requests (expected +12) PASS
```

The same counts came out for both frontends of shape B, for A2 and for D. In
shape C the frontend that does not own a model answers 404 and the other DGD's
engine counter does not move (the script prints FAIL for those lines because it
expects both models everywhere; for C that is the isolation working).

## How a frontend decides what it serves

The operator gives every frontend `DYN_NAMESPACE=<k8s-namespace>-<dgd-name>` and
the same value as `DYN_NAMESPACE_PREFIX`. A frontend serves every worker whose
Dynamo namespace starts with that prefix.

- Workers in the same DGD get `<k8s-namespace>-<dgd-name>` by default, so they
  match. That is why A2 works with no extra settings.
- A worker in another DGD matches only if its `DYN_NAMESPACE` is set by hand to
  something that starts with the first DGD's prefix (shape D).

**Changed since the MIG test on Dynamo 1.3.0.** There, two different models in
one Dynamo namespace failed with `Cannot register model ... a different model is
already registered there`, so each worker needed its own namespace. On 1.5.0
shape A2 puts both models in the same namespace and both register and serve.
The per-model namespace is still accepted (shape A) but no longer required.

## What this means for many models on shared GPUs

- One DGD per model, each with its own frontend (the layout in Dynamo's public
  `examples/gpu-sharing`), is not required. One frontend can front any number of
  model workers, in one DGD or spread over several.
- One DGD holding all models is the simplest. Separate DGDs with one shared
  frontend (shape D) keep each model's lifecycle independent, at the cost of
  setting one environment variable per worker.
- Several frontends in one DGD give more than one entry point to the same set of
  models; they do not partition the models between them.

## Not tested

More than two models; disaggregated (prefill + decode) workers per model; more
than one replica of a model; load on both models at once; a frontend from one
Kubernetes namespace serving workers in another.

## Files

- `scripts/mkdgd.py`: prints a DGD from a list of frontends and
  `name:model:share:memory[:namespace]` workers.
- `scripts/check.sh`: pods, the namespaces the operator set, what each frontend
  lists, one request per model.
- `scripts/routing.sh`: the counter check above.
- `manifests/`: the DGDs as applied. `evidence/`: output of both scripts per shape.
- [`CHANGELOG.md`](CHANGELOG.md): what was installed and changed on the box.
