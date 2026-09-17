# Reproducibility — DVC and MLflow

> **Memory intensive stages:** `split_corpus` and `build_homolog_pairs` run MMseqs2 over the ~2.6M sequence OAS corpus.
> Memory usage scales with available system memory. On memory-constrained local machines, use:
>
> ```bash
> export DVC_BUCKET="gs://<your-bucket-name>"       # set your GCS bucket for DVC
> make jobs-repro STAGES=split_corpus               # execute on cloud instance (recommended)
> EDITJUMPS_ALLOW_LOCAL_HEAVY=1 make repro STAGES=  # local override for entire DAG
> ```
>
> Pipeline configuration bounds MMseqs memory via `params.yaml: mmseqs.split_memory_limit`. Local capacity checks prevent unconstrained local runs.

`editjumps run-pipeline` is a thin pass-through to [`dvc repro`](https://dvc.org):
the DAG lives in `dvc.yaml` (see [`pipeline.md`](pipeline.md) for the stage-by-stage
walk-through), and every tunable knob (thresholds, MMseqs2 identity/coverage, ESM-2
checkpoint, batch size, ...) lives in `params.yaml`. DVC skips any stage whose
`deps`/`params`/`outs` are unchanged, so a re-run after a clean checkout is a no-op
unless you touched something upstream. (Training stages need `uv sync --all-groups`
— see [`installation.md`](installation.md).)

```bash
uv run editjumps run-pipeline                      # = dvc repro
uv run editjumps run-pipeline -f pretrain_esm      # force one stage
uv run editjumps run-pipeline --downstream build_homolog_pairs   # a stage and everything after it
dvc dag                                           # the exact DVC graph
```

**Data lives in DVC, not git** — `data/**` is git-ignored; `dvc.lock` + `.dvc`
metadata are committed. The DVC remote is a **GCS bucket** (`gs://…/dvc`); run the
one-time [`deploy/gcp/`](../deploy/gcp) setup ([`docs/gcp_setup.md`](gcp_setup.md))
and `dvc push`/`pull` hit GCP. Reproduction also still works *from source* —
`dvc repro` regenerates every output from the public downloads (no `dvc pull`
needed) — with the one caveat in `dvc.yaml`'s lineage note: the committed corpus passed a
decontamination stage this repository does not carry, so a from-source rebuild will differ from it
by 370 of 2,587,297 lines.

**Runs log to [MLflow](https://mlflow.org).** Default is local (`sqlite:///mlflow.db`,
`bash editjumps/repo/mlflow_ui.sh` → http://127.0.0.1:5000); set `MLFLOW_TRACKING_URI` to the
shared **GCP server** ([`docs/gcp_setup.md`](gcp_setup.md) — a Cloud Run
service backed by Cloud SQL + GCS) and every run logs there instead, no code change.
`pretrain_esm` logs its config, per-step train/eval loss, and the model dir as an
artifact; `train_edit_flows` and `evotune_esm` are wired identically. Headline metrics are
mirrored to `metrics/pretrain_esm.json` for `dvc metrics`. (`mlflow_ui.sh` also carries a shim for a Python-3.14 /
mlflow 3.14.0 crash — [mlflow#24155](https://github.com/mlflow/mlflow/issues/24155);
drop it once that's fixed.)

**GCP authentication:** Cloud tokens expire after 60 minutes. `editjumps/core/utils.py` refreshes tokens every 40 minutes, and MLflow logging exceptions are non-fatal to ensure training completion even if tracking endpoints become unreachable.

### What `dvc.lock` does and does not cover

The committed `dvc.lock` tracks a fast smoke-test configuration (35M trunk, 200 steps) across 8 of the 14 stages (`download_oas`, `split_corpus`, `fetch_base_checkpoint`, `pretrain_esm`, `build_homolog_pairs`, `seed_homologs`, `train_edit_flows`, `evodiff_msa_baseline`). Full 20,000-step runs and the six §4.1 / evotuning stages are run separately.

Consequently, `dvc status` reflects differences between the smoke lock and `params.yaml`. Authoritative hyperparameters and provenance are recorded in each artifact's model card.

### Deliberately stale stages (what `dvc status` is telling you)

Two stages require GPU execution to refresh and are intentionally left unrefreshed in the default checkout:

| stage | why it is stale | to fix |
|---|---|---|
| `pretrain_esm` | the committed `esm2_oas` predates the `oas.pair_sep` fix — chains were joined with `:`, which ESM-2 tokenizes to `<unk>`; now `.`, a real vocab entry | `make jobs-repro STAGES=pretrain_esm` |
| `train_edit_flows` | the committed editor was trained on oriented pairs, before `build_homolog_pairs` was rebuilt symmetric | `make jobs-train` |

`build_homolog_pairs` is rebuilt symmetric (`homolog_pairs.direction: none`), matching the unoriented EvoFlows specification. The committed smoke checkpoint (`data/pretrain/edit_flows`) predates this update. Reported results (`faithful-appendixa`) were trained on symmetric pairs (verified by md5 in [`edit-flows-editor.md`](model_cards/edit-flows-editor.md)).

Symmetric pairs depend on no property at all, which is why `build_homolog_pairs` declares no
property knobs among its tracked params — its identity is the clustering settings
(`min_seq_id`, `coverage`, `mmseqs_mode`), the per-family cap, `max_lines` and `direction`.

On Apple Silicon the HuggingFace `Trainer` auto-uses MPS, but only where
`torch.backends.mps.is_available()`; some sandboxes report `False`, so run pretraining in a
native terminal there.

