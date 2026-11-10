# Multi-model: several models, one frontend or many, one DGD or many

**Run 2026-11-10.** One A100 80GB, driver 615.71.09, KAI-Scheduler +
kai-gpu-fractioning forks, Dynamo platform 1.5.0 (operator, Kubernetes discovery)
and `vllm-runtime:1.5.0`. Two models, Qwen3-4B and Qwen3-0.6B, each an aggregated
vLLM worker in its own GPU fraction of the same GPU (40% SMs / 14000 MiB and
10% SMs / 6000 MiB). One run per shape.

## Result

| Question | Answer |
|---|---|
| Does it need a frontend per model? | **No.** One frontend served both models, and listed all of 4, 8 and 16 models in the larger runs below. |
| One frontend and several models in **one DGD**? | **Yes.** |
| Several frontends in one DGD? | **Yes.** Each frontend serves all the models of that DGD. |
| One frontend for models in **different DGDs**? | **Yes**, if the other DGD's worker sets a Dynamo namespace that starts with the frontend's prefix. Without that, each DGD's frontend sees only its own models. |
| Does each request reach the right model? | **Yes**, checked on the engines' own request counters in every shape. |

| Shape | Manifest | Frontends see | Routing check |
|---|---|---|---|
| A. One DGD, one frontend, two models, a Dynamo namespace per model | [`a-one-dgd-one-frontend.yaml`](manifests/a-one-dgd-one-frontend.yaml) | both models | not counted (A2 covers the same path); [output](evidence/a.out) |
| A2. Same, workers left in the DGD's default namespace | [`a2-no-namespaces.yaml`](manifests/a2-no-namespaces.yaml) | both models | PASS: [check](evidence/a2.out), [routing](evidence/a2-routing.out) |
| B. One DGD, two frontends, two models | [`b-one-dgd-two-frontends.yaml`](manifests/b-one-dgd-two-frontends.yaml) | both frontends: both models | PASS through each frontend: [check](evidence/b.out), [routing](evidence/b-routing.out) |
| C. Two DGDs, each a frontend and one model | [`c1.yaml`](manifests/c1.yaml), [`c2.yaml`](manifests/c2.yaml) | each frontend: only its own model; the other returns 404 | PASS for the model each frontend owns: [check](evidence/c.out), [routing](evidence/c-routing.out) |
| D. C's first DGD plus a worker-only DGD for the second model, namespace `default-mm-c1-qwen06b` | [`c1.yaml`](manifests/c1.yaml), [`d-worker-only-dgd.yaml`](manifests/d-worker-only-dgd.yaml) | the one frontend: both models | PASS: [check](evidence/d.out), [routing](evidence/d-routing.out) |

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
  `examples/gpu-sharing`), is not required for two models. One frontend fronted
  both, in one DGD or spread over two. The routing check was done with two models;
  registration was also checked with 4, 8 and 16 (next section).
- One DGD holding all models is the simplest. Separate DGDs with one shared
  frontend (shape D) keep each model's lifecycle independent, at the cost of
  setting one environment variable per worker.
- Several frontends in one DGD give more than one entry point to the same set of
  models; they do not partition the models between them.

## More than two models: 4, 8 and 16 behind one frontend

An earlier test of this shape on Dynamo 1.3.0 (one frontend, N workers started
together, all serving the same weights under different names) lost models: 1 of
4 at N=4 and 9 of 16 at N=16, twice, while N=1, 2 and 8 were clean. Every pod
was Running; the frontend failed to resolve some models' tokenizer files while
resolving several at once on one shared cache path, did not retry, and
`/v1/models` silently listed fewer models than were deployed.

The same shape rerun here on Dynamo 1.5.0: one DGD, one frontend, N workers
started together, all serving `Qwen/Qwen3-0.6B` as `m01`..`mNN`, each a GPU
fraction (5% SMs, 4500 MiB). `scripts/n-models.sh`.

| N | Pods Running | Frontend lists, checked 3 times over 3 minutes | Registration errors in the frontend log | Manifest, output |
|---|---|---|---|---|
| 4 | 5 of 5 | 4 of 4 | 0 | [`n4-same-weights.yaml`](manifests/n4-same-weights.yaml), [output](evidence/n4.out) |
| 8 | 9 of 9 | 8 of 8 | 0 | [output](evidence/n8.out) |
| 16, first run | 17 of 17 | 16 of 16 | 0 | [`n16-same-weights.yaml`](manifests/n16-same-weights.yaml), [output](evidence/n16a.out), [frontend log](evidence/fe-n16a.log) |
| 16, second run | 17 of 17 | 16 of 16 | 0 | [output](evidence/n16b.out) |

**No model was lost in any run.** In the 1.5.0 frontend log each model's files
sit under a directory of their own, named after the served model
(`.../.cache/dynamo/mdc/by-slug/m10/...`), not under one shared path for the
weights repository. That is consistent with the earlier race being gone, since
the frontend no longer resolves several models against the same files; the
cause of the difference was not traced further than that.

What this does and does not show:

- Four clean runs, two of them at the N that failed twice out of twice before.
  A race can still hide; this is evidence, not proof of absence.
- It differs from the earlier test in more than the Dynamo version: Qwen3-0.6B
  instead of Qwen3-8B, GPU fractions instead of MIG slices, Kubernetes discovery.
- At N=16 only the first and last model were sent a request (both answered). In
  the N=4 and N=8 outputs the request lines say "Model not found": the script
  asked for `m01` while those runs registered `m1`. That was a script bug, since
  fixed; the listing lines are the evidence for those runs.
- Sixteen pods took about 270 s to all be Running.

## Not tested

Requests to every one of 16 models, or load on many models at once;
disaggregated (prefill + decode) workers per model; more than one replica of a
model; a frontend from one
Kubernetes namespace serving workers in another.

## Files

- `scripts/mkdgd.py`: prints a DGD from a list of frontends and
  `name:model:share:memory[:namespace]` workers.
- `scripts/check.sh`: pods, the namespaces the operator set, what each frontend
  lists, one request per model.
- `scripts/routing.sh`: the counter check above.
- `manifests/`: the DGDs as applied. `evidence/`: output of both scripts per shape.
- [`CHANGELOG.md`](CHANGELOG.md): what was installed and changed on the box.
