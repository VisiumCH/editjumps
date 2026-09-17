---
# Hugging Face model-card metadata, in the documented key shape. Not published to the Hub: this
# project's weights are private (docs/weights.md).
license: mit
library_name: transformers
base_model: facebook/esm2_t12_35M_UR50D
base_model_relation: finetune
tags:
  - biology
  - protein-language-model
  - fill-mask
  - antibody
pipeline_tag: fill-mask
---

# Model Card for the OAS-continued ESM-2 trunk (`esm2_oas`)

ESM-2 continued on this repository's Observed Antibody Space corpus with the same masked-language
objective it was pretrained with. It is the trunk two of the four Edit Flows editor arms
initialise from.

**Read the tracked artefact, not `params.yaml`.** The `pretrain:` block in `params.yaml` now
specifies a 650M continue-pretrain over the whole corpus; the artefact `dvc.lock` records — and
which every editor arm that used it was initialised from — is **ESM-2 35M for 200 steps at batch
16, i.e. 3,200 sequences**. (`max_seqs: 20000` is a cap on the pool and never bound, so the
shorthand "200 steps over 20,000 sequences" overstates the budget 6.25×.) `params.yaml`'s own
comment describes the raise as done. The lock wins; see
[Training Hyperparameters](#training-hyperparameters).

## Table of Contents

- [Scope of Reproduction](#scope-of-reproduction)
- [Model Details](#model-details)
- [Uses](#uses)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications](#technical-specifications)
- [Citation](#citation)

## Scope of Reproduction

| | |
|---|---|
| claim | `reproduction`, **and it is flagged a deviation** |
| stage | `pretrain_esm` |
| why a deviation | EvoFlows §3.4 states **no** domain-adaptive pretraining — only "the encoder trunk of a pre-trained ESM-2 model". This trunk is therefore an *addition* rather than a reproduction step, and it is arm A of the 2×2 over trunk and head parameterisation, not the paper's setting. Verbatim from `claims.md`. |
| what a result using it supports | this project's question of whether antibody-domain adaptation helps the editor |
| what it cannot support | anything about EvoFlows' own trunk. The paper never states its size; its evotuned baseline is ESM-2-650M, which is what `pretrain.base_checkpoint_loader_name` now names. |

The paper-faithful editor arm exists precisely so this deviation is measurable rather than assumed:
`faithful-appendixa` starts from stock `facebook/esm2_t12_35M_UR50D` and is the **reported** arm, and
`heads-only` is the same head parameterisation over this trunk.

## Model Details

### Model Description

- **Developed by:** `editjumps`
- **Model date:** the tracked build is the one whose metrics are in
  [`metrics/pretrain_esm.json`](../../metrics/pretrain_esm.json), produced by the first
  `make jobs-repro` run
- **Model version:** `dvc.lock` hash `1dae911a396c8203ae5663301a2e8612.dir`, 5 files,
  134,035,643 B
- **Model type:** encoder-only masked language model (`EsmForMaskedLM`)
- **Language(s):** not applicable — amino-acid sequences over ESM-2's 33-token vocabulary. Inputs
  are joined `VH.VL` antibody pairs separated by `.`, truncated at 280 tokens.
- **License:** MIT for this project's code; the initialisation is Meta's ESM-2 under its own terms.
  **This artefact is not publicly obtainable** — it is DVC-tracked in a private bucket
  ([`weights.md`](../weights.md)).
- **Finetuned from model:** `facebook/esm2_t12_35M_UR50D`, fetched by the `fetch_base_checkpoint`
  stage into `data/pretrain/base_checkpoints/esm2_t12_35M_UR50D` (134,030,456 B, 4 files). That
  stage exists so `pretrain_esm` never needs HF Hub access: it downloads via `fair-esm` from
  Meta's own host and repackages to HF format, verified to reproduce the original logits to float32
  rounding precision.

### Model Sources

- **Repository:** `editjumps/pipeline/train/pretrain_esm.py`; stage `pretrain_esm` in `dvc.yaml`
- **Upstream model card:** [`facebook/esm2_t12_35M_UR50D`](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
  — worth knowing that it is 19 lines and defers architecture and training data entirely to the
  ESM-2 paper, so there is no upstream card to inherit hyperparameters or evaluations from
- **Weights:** DVC-tracked at `data/pretrain/esm2_oas`; `uv run dvc pull data/pretrain/esm2_oas`
  with credentials against the `$PROJECT_ID` organisation

## Uses

### Direct Use

Initialising the Edit Flows editor (`train-edit-flows --model-name data/pretrain/esm2_oas`, the
default). Usable directly as a masked LM over joined `VH.VL`
antibody sequences.

### Out-of-Scope Use

* **Not a converged domain-adapted model, and never claimed as one.** `params.yaml`'s own comment
  calls the tracked build "a reproducible pipeline artifact, not a fully-converged model".
* **Not the paper's trunk.** Using it and reporting the result as an EvoFlows reproduction number is
  the error `provenance.py` exists to prevent.
* **Not a property model.** Its eval loss says the continued-pretraining step is learning OAS
  and nothing about any downstream property.

## Bias, Risks, and Limitations

* **Antibody-only, paired-chain only, and length-filtered.** Chains outside 70–200 residues were
  dropped before the corpus was built, and every line is a joined `VH.VL`, so the model has never
  seen a lone V domain — which becomes a real train/test shift downstream, documented on
  [the editor's card](edit-flows-editor.md#factors).
* **Its corpus carries a decontamination step this repository cannot repeat** — 370 of 2,587,297
  lines (0.014%), screened against a separate project's labelled molecules, not against anything
  the paper holds out. That stage is not part of this DAG, so a rebuilt corpus will differ from the
  one this trunk was trained on. **Three counts for it are in circulation and only one describes
  the tracked artefact**; which and why is on
  [the corpus card](datasets/oas-corpus.md#bias-risks-and-limitations), so that it is corrected in
  one place.
* **The budget is the limitation.** At 200 steps × batch 16 the trunk saw 3,200 of 2,586,927
  sequences before `max_seqs: 20000` even bound; expressed against the whole corpus that is 0.12%.
  This was the largest single gap against EvoFlows and is why `params.yaml` was raised.

### Recommendations

If the trunk matters to a result, rerun the stage at `params.yaml`'s current values and adopt the
new lock — do not assume the tracked artefact matches the file. `dvc.lock` is the only place that
records which values actually produced it.

## How to Get Started with the Model

```bash
uv run dvc pull data/pretrain/esm2_oas          # needs credentials; see docs/weights.md
uv run editjumps fetch-base-checkpoint            # the stock initialisation, ~30 s, public
uv run dvc repro pretrain_esm                   # rebuild it at params.yaml's current values
```

A working local end-to-end path that needs no GPU and no bucket access — one OAS data unit instead
of the 611 on record — is [`training.md`](../training.md).

## Training Details

### Training Data

[**The OAS paired corpus**](datasets/oas-corpus.md) — `oas_corpus.train.txt.gz`
(94,867,624 B) for training, `oas_corpus.val.txt.gz` (4,722,905 B) as the eval set. **The
dataset card is the description**: the database, the length filter, the `.` separator, the
decontamination and its three competing numbers, the clustered split and every md5 are there, and
four stages share them, so they are stated once. What is specific to this run:

| | value | source |
|---|---|---|
| corpus | **2,586,927** lines, md5 `b23db3c267d61fc8a30e60aa4f11e28f` | [dataset card](datasets/oas-corpus.md#dataset-structure) |
| val set actually evaluated | ~128,957 samples (633.5 s at 203.563 samples/s) | [`metrics/pretrain_esm.json`](../../metrics/pretrain_esm.json) |
| **sequences the tracked run actually saw** | **3,200** — 200 steps × batch 16. `max_seqs: 20000` is a cap on the *pool*, and it never bound; a shorthand of "200 steps over 20,000 sequences" reports the cap rather than the count and overstates the budget 6.25× | `dvc.lock` `pretrain_esm` |

Objective: masked language modelling at `mlm_probability: 0.15`, via `transformers`'
`DataCollatorForLanguageModeling` — so the standard 80% `<mask>` / 10% random / 10% unchanged
recipe, not a custom one.

### Training Procedure

#### Preprocessing

Tokenize each joined `VH.VL` line with the ESM-2 tokenizer, truncate at `max_length: 280` tokens
(the median joined input is 232 residues, the 99th percentile 243, the maximum 298 — so truncation
binds on the tail), and mask 15% of tokens per batch.

#### Training Hyperparameters

Two columns, and they disagree. The **tracked** column is what `dvc.lock` records for the artefact
that exists and that every editor arm using this trunk was initialised from. The **`params.yaml`**
column is what a fresh `dvc repro pretrain_esm` would run today.

| key | tracked (`dvc.lock`) | `params.yaml` today |
|---|---|---|
| `pretrain.model_name` | `data/pretrain/base_checkpoints/esm2_t12_35M_UR50D` (**35M**) | `.../esm2_t33_650M_UR50D` (**650M**) |
| `pretrain.base_checkpoint_loader_name` | — (not a locked param of this stage) | `esm2_t33_650M_UR50D` |
| `pretrain.max_steps` | **200** | **−1** (run the full `epochs`) |
| `pretrain.max_seqs` | **20000** | **0** (the whole corpus) |
| `pretrain.batch_size` | 16 | 32 |
| `pretrain.epochs` | 1.0 | 1.0 |
| `pretrain.max_length` | 280 | 280 |
| `pretrain.mlm_probability` | 0.15 | 0.15 |
| `pretrain.save_steps` | 50 | 5000 |
| `pretrain.save_total_limit` | 3 | 3 |
| `pretrain.precision` | — the flag did not exist; fp32 | `auto` (bf16 where the GPU supports it) |
| `pretrain.dataloader_workers` | — the flag did not exist | 4 |

**And a third column that is in neither file.** The stage builds a `transformers`
`TrainingArguments` and sets only the keys above, so the optimiser and its schedule are HF
*defaults*, never stated anywhere in this repository. At `transformers` 5.12.1, which is the version
recorded in the artefact's `config.json`, they are:

| | |
|---|---|
| optimizer | `adamw_torch_fused`, betas (0.9, 0.999), eps 1e-8 |
| learning rate | **5e-5** |
| schedule | linear decay, `warmup_steps: 0` |
| weight decay | 0.0 |
| max grad norm | 1.0 |
| seed | **42** |

That is a genuine gap rather than a nitpick: the learning rate of the trunk every editor arm starts
from is not written down in `params.yaml`, in `dvc.yaml` or in `dvc.lock`, and it changes if
`transformers` changes its default.

#### Speeds, Sizes, Times

| | value | source |
|---|---|---|
| whole job (`make jobs-repro`, L4:1) | **1 h 0 m 16 s**, 0 spot recoveries | [`performance.md`](../performance.md), "Training" |
| final train loss | **1.5713479614257813** | [`metrics/pretrain_esm.json`](../../metrics/pretrain_esm.json) |
| final eval loss | **1.1778740882873535** | same |
| epochs covered | **0.16**, bounded by `max_steps` and not by convergence | same |
| eval throughput | 203.563 samples/s, 633.5 s runtime | same |
| artefact size | 134,035,643 B, 5 files | `dvc.lock` |

Read the eval loss as a floor-check on the MLM backbone: it says the continued-pretraining step is
learning OAS, and nothing about any downstream property.

For anyone sizing a real continue-pretrain: a 650M run needs an **A100** (~16–18 GB in bf16, which
outgrows an L4's 24 GB once activations are counted), and fp32 on an A100 runs off the tensor cores
at roughly 8× slower and twice the memory. CPU reference ratios — useful as ratios only — are in
[`throughput_options.md`](../throughput_options.md): batch 16 at length 160 is the throughput
optimum, length 280 costs 1.5–2× length 160, and `esm2_t6_8M` against `esm2_t12_35M` on one config
is a 4.05× model-size effect.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

`data/pretrain/oas_corpus.val.txt.gz`, the clustered 5% split — near-duplicates cannot
straddle it, so eval loss is not measuring memorised CDR keys.

#### Factors

* **Input form.** Joined `VH.VL` only, truncated at 280 tokens. The trunk has never scored a lone V
  domain, which is the form every downstream evaluation uses.
* **Sequence length.** 153–298 residues joined; 70–164 per chain.
* **Training budget.** 0.16 of an epoch. Every number here is a floor-check at a bounded budget.

#### Metrics

MLM cross-entropy on the held-out split. That is all this stage measures — no perplexity, no
downstream probe, no comparison against the stock trunk.

### Results

| | value |
|---|---|
| final train loss | 1.5713 |
| final eval loss | **1.1779** |
| epoch at stop | 0.16 |

#### Summary

**There is no committed comparison of this trunk against the stock trunk *as trunks*** — no MLM
loss for `facebook/esm2_t12_35M_UR50D` on the same validation split — so nothing here says whether
OAS adaptation improved the language model.

**But the downstream question does have a controlled answer, and it is committed.** An earlier
revision of this card said "this card cannot tell you whether OAS adaptation helped", which
under-stated the evidence: `metrics/matched/` holds a full 2×2 over trunk (this one versus stock)
and heads, 3 evaluation seeds per cell, both seed families, at the one config that separates the
four arms — 300 templates × 1 variant, holdout 1000, clock 40, k=3. Spectrum MMD, lower better,
recomputed from those 24 files:

| | our heads (`linear`/`fresh`) | Appendix-A heads (`mlp`/`esm_lm_head`) |
|---|---|---|
| **stock ESM-2 35M** | **D** — 0.649 ± 0.041 Ty1 · 0.728 ± 0.020 HER2-VH | **B** — 0.507 ± 0.034 · 0.556 ± 0.015 |
| **this OAS-adapted trunk** | **A** — 0.508 ± 0.013 · 0.572 ± 0.027 | **C** — 0.489 ± 0.040 · 0.527 ± 0.029 |

**Ranking C < B < A < D on both families**, with the between-arm spread 5.03× (Ty1) and 8.72×
(HER2-VH) the mean within-arm seed standard deviation. The decomposition against arm D:

| | Ty1 | HER2-VH |
|---|---|---|
| Appendix-A heads alone (D → B) | −0.142 | −0.172 |
| **this trunk alone (D → A)** | **−0.141** | **−0.156** |
| both (D → C) | −0.160 | −0.200 |
| **interaction** | **+0.123** | **+0.128** |

(The reproduction ledger prints −0.201 and +0.127 in the HER2-VH column. The difference is
last-digit rounding — the cell means are 0.527442 and 0.727700, giving −0.200258 — and it changes
nothing. Recorded rather than silently reconciled.)

So the answer is: **OAS adaptation helps by about as much as the paper's head parameterisation, and
the two are substitutes rather than complements.** Either intervention alone recovers most of the
available gain — 88.9% and 88.2% on Ty1, 86.0% and 77.7% on HER2-VH — and adding the second buys
0.018 to 0.045 of MMD, the largest of those being Appendix-A heads added to the OAS trunk on
HER2-VH (0.0446, against 0.0281 the other way round). The interaction reproduces to within 0.005 across two
independent families. Arm D — stock trunk with our own linear/fresh heads, the original default —
is the only configuration clearly worse.

**The mechanism is why "substitutes" and not "redundant".** `q_head: esm_lm_head` takes a deep copy
of the trunk's masked-LM head, which is fully trained on UniRef for stock ESM-2 and trained for
**200 steps over 3,200 sequences** here. This continue-pretrain damaged precisely the component
Appendix A reuses, which is what makes the two interventions overlap: *domain-adapt a trunk and you
must train its LM head properly or not reuse it.* The 247,713-parameter head split under
[Model Architecture](#model-architecture-and-objective) is the object in question.

Two caveats on the table. It is one evaluation config and two families — the same four arms at
30 × 10 span only **0.044** between arm means, against a mean within-arm seed sd of **0.063**, so at
that resolution the noise exceeds the signal and reading it as "no difference between the arms" was
a power artefact this project retracted. And the reported editor
arm is **B**, a stock-trunk arm: this trunk is arm A and arm C's initialisation, not the
reproduction's headline. See [the editor's card](edit-flows-editor.md#checkpoint-inventory).

## Environmental Impact

- **Hardware Type:** one NVIDIA L4 (24 GB), GCP managed spot via SkyPilot
- **Hours used:** **1 h 0 m 16 s** for the whole job, which also ran `train_edit_flows` — so this
  stage's own share is not separately recorded
- **Cloud Provider:** Google Cloud Platform
- **Compute Region:** europe-west6 (Zurich), co-located with the storage bucket
- **Carbon Emitted:** **not assessed.** One L4 for under an hour; no grid-intensity figure for the
  region at the time is on record, so no number is offered.

## Technical Specifications

### Model Architecture and Objective

`EsmForMaskedLM` at ESM-2 t12's configuration, unchanged — the stage adapts weights, not
architecture.

| | value |
|---|---|
| layers | 12 |
| hidden size | 480 |
| attention heads | 20 |
| intermediate size | 1920 |
| position embeddings | rotary, `rope_theta` 10000, `max_position_embeddings` 1026 |
| vocabulary | 33 |
| tied word embeddings | yes |
| `token_dropout` | yes |
| parameters | **33,501,394** total. The encoder reports 33,269,521 and the masked-LM head 247,713; those do not sum to the total because the head's decoder weight is *tied* to the input embedding and is counted once |

The parameter split matters downstream: the editor's `q_head: esm_lm_head` takes a deep copy of that
247,713-parameter head *per Q head* (eq 16), which is why the folder is loaded as `EsmForMaskedLM`
and not `EsmModel` — the latter drops the head eq 16 wants. As with every model here, the parameter
count is computed from the architecture and recorded in no artefact; it agrees with the tracked
folder's 134,035,643 B (33,501,394 fp32 slots is 134,005,576 B, the remainder being config and
tokenizer files).

If the 650M configuration in `params.yaml` is built instead, the equivalent figures are 33 layers,
hidden 1280, intermediate 5120, 20 heads, **651,043,254** parameters — 649,400,981 in the encoder
and a 1,684,513-parameter head, which again overlap by the tied 1280x33 decoder weight.

### Compute Infrastructure

#### Hardware

L4:1 for the 35M configuration; **A100:1 (40 GB) required** for the 650M one, in europe-west4 (the
only EU region with A100 in the catalog). Checkpoints are ~2.6 GB at 650M and every save is
uploaded, which is why `save_steps` is 5000 there and not 50 — a 50-step cadence on an ~81k-step run
would ship about 4 TB and spend most of the run doing it.

#### Software

`transformers` `Trainer` (version 5.12.1 recorded in the artefact's `config.json`),
`DataCollatorForLanguageModeling`, `uv`-locked environment. MLflow logging is wired explicitly
rather than through `report_to`. GCS checkpoint mirroring is a callback and is off unless
`checkpoint_uri` is set.

## Citation

The initialisation is Meta's ESM-2, and **this repository records no author list for it** — take
the citation from the upstream card
([`facebook/esm2_t12_35M_UR50D`](https://huggingface.co/facebook/esm2_t12_35M_UR50D)) and the
bioRxiv paper it links, rather than from here. Deliberately not transcribed: a card that invents an
author list is worse than one that points at the source.

For this adapted trunk, cite `CITATION.cff` at the repository root, which lists ESM-2 among the
repository's keywords and the two reproduced papers among its references.

## Model Card Authors

Written from `dvc.lock`, `metrics/pretrain_esm.json` and the stage source by the `editjumps`
maintainers.

## Model Card Contact

Via the project repository.
