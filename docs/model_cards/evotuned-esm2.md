---
# Hugging Face model-card metadata, in the documented key shape. Not published to the Hub: this
# project's weights are private (docs/weights.md).
license: mit
library_name: transformers
base_model: facebook/esm2_t33_650M_UR50D
base_model_relation: finetune
tags:
  - biology
  - protein-language-model
  - fill-mask
  - antibody
  - reproduction
pipeline_tag: fill-mask
---

# Model Card for the evotuned ESM-2 trunks (`evotune-{ty1,her2vh}`, `evotune-dj-{ty1,her2vh}`)

Four ESM-2-650M checkpoints, each MLM-adapted to **one** antibody homolog family. This is
EvoFlows §2.2's Evotuning — "adapts pre-trained PLMs to a specific protein family by further
training on sequences drawn from a homologous sequence set … using the same self-supervised
learning objective" (Alley et al., 2019) — and these four trunks are what the §4.2/§4.3 "Evotuned
PLM" baselines infill with.

One card for four checkpoints, because they differ in exactly two things: **which family**, and
**which population of that family** the adaptation corpus was drawn from. Their
`training_args.bin` files are **byte-identical** (md5 `36d7eabec250d338a516ac9859a1b80a`, all four),
so nothing else about the training run varies.

## Table of Contents

- [Scope of Reproduction](#scope-of-reproduction)
- [Checkpoints](#checkpoints)
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
| claim | `reproduction` — EvoFlows §2.2 (the recipe) and §4.3 (the comparison) |
| stages | `evotune_esm` (the trunk), `evotune_baseline` (§4.3's "Evotuned PLM"), `evotune_baseline_forced` (§4.3's "with forced substitutions") |
| what is the paper's | the recipe — the same MLM objective, on one homolog family — and both baseline readings |
| what is ours | the family, and its train / inference / holdout split. `claims.md` states this explicitly. The paper also never states its evotuning *budget*, so 2,000 steps is ours. |
| trunk size, and why | ESM-2-**650M**. The paper's evotuned baseline is ESM-2-650M, and `evotune.model_name` points at the same local checkpoint the corpus pretrain initialises from, so the baseline and the editor share a starting point and the comparison is about what each did with it. |

Six readings where the paper is silent are recorded in `editjumps/core/evotune/README.md`. The one
that moves a number most is the **budget mode**: §4.2 matches a *mutation* count while the text says
"masked positions are iteratively infilled", which is a *mask* count. Both readings are implemented;
`budget_mode: realised` (the trainer's `top_up=True`) is the default and the only one that satisfies
§4.2. A seventh choice is free and is **not** in that register — the sampling **temperature**, which
the paper never gives and which is 1.0, the model's own distribution, here. It is named because a
reader needs it, not because the register records it.

## Checkpoints

| tag | family | adaptation population | train / val lines | `metrics/evotune/` | GCS prefix |
|---|---|---|---|---|---|
| `evotune-ty1` | `Anti-SARS-CoV-2_VHH_Ty1.fasta` (a VHH) | the family's **train** part of the §4.2 split | **36,417 / 1,916** | `ty1-evotune_esm.json` and both baselines | `models/esm2_evotuned/evotune-ty1/` |
| `evotune-her2vh` | `Anti-HER2_scFv_VH_trastuzumab.fasta` (a VH) | same | **40,489 / 2,130** | `her2vh-evotune_esm.json` and both baselines | `models/esm2_evotuned/evotune-her2vh/` |
| `evotune-dj-ty1` | Ty1 | only members **absent from the editor's training pairs** | **17,362 / 913** | **none** | `models/esm2_evotuned/evotune-dj-ty1/` |
| `evotune-dj-her2vh` | HER2-VH | same | **19,756 / 1,039** | **none** | `models/esm2_evotuned/evotune-dj-her2vh/` |

Line counts are exact, read from each folder's own shipped `evotune_family.{train,val}.txt.gz` —
those files travel with the weights because the baseline reads the corpus to build its entropy
profile. Ty1's 36,417 + 1,916 = 38,333 is exactly the family's 38,553 distinct members less the 20
templates and 200 holdout; HER2-VH's 42,619 is 42,839 less the same 220.

**Why the disjoint pair exists, and why the default pair cannot replace it.** The disjoint §4.2
table draws both its templates and its scoring reference from family members absent from the
editor's training pairs. The default trunks' adaptation corpora hold **194 of Ty1's 200** disjoint
reference sequences and 191 of HER2-VH's, plus 19 of the 20 templates in each — so
`evotune_baseline`'s leak assertion refuses them, correctly. Each disjoint corpus was checked to be
a subset of that split's pool, so the assertion passes on its merits rather than by being weakened.
The `dj` pair is what every cell in [`metrics/disjoint/`](../../metrics/disjoint/README.md) uses.

## Model Details

### Model Description

- **Developed by:** `editjumps`
- **Model date:** the two default trunks are sky jobs **67** (`profound-repro-evotune2`, Ty1) and
  **70** (`profound-repro`, HER2-VH), per `metrics/evotune/README.md`. The two disjoint trunks'
  job numbers are **not recorded**.
- **Model version:** run tags as in [Checkpoints](#checkpoints); step 2,000
- **Model type:** encoder-only masked language model (`EsmForMaskedLM`), family-adapted
- **Language(s):** not applicable — amino-acid sequences over ESM-2's 33-token vocabulary. Inputs
  are **single V domains**, truncated at 160 tokens: a family FASTA holds lone domains, not joined
  `VH.VL` pairs.
- **License:** MIT for this project's code; the initialisation is Meta's ESM-2 under its own terms.
  **Not publicly obtainable** ([`weights.md`](../weights.md)).
- **Finetuned from model:** `data/pretrain/base_checkpoints/esm2_t33_650M_UR50D`, the local
  HF-format conversion of `facebook/esm2_t33_650M_UR50D`

### Model Sources

- **Repository:** trainer `editjumps/pipeline/train/evotune.py`, a thin wrapper over
  `pretrain_esm.py` — evotuning *is* the corpus pretrain with one family in place of the corpus, so
  there is no second trainer. The baseline's own halves are `editjumps/core/evotune/profile.py`
  (where an edit goes, eq 12's entropy profile) and `substitution.py` (what goes there).
- **Paper:** EvoFlows [arXiv:2603.11703](https://arxiv.org/abs/2603.11703) §2.2 and §4.3, citing
  Alley et al. 2019 for the recipe
- **Deviation register:** `editjumps/core/evotune/README.md`
- **Weights:** `gs://$DVC_BUCKET/models/esm2_evotuned/<tag>/` (private) —
  `model.safetensors` 2,604,236,440 B, plus `config.json`, tokenizer files, `training_args.bin`,
  and the two corpus halves

## Uses

### Direct Use

Serving as the MLM half of §4.3's two evotuned baselines: eq 12's per-column entropy profile picks
which positions to mask, and this model infills them. Also usable directly as a family-specialised
masked LM.

### Out-of-Scope Use

* **The default pair must not be scored against a disjoint reference.** Their adaptation corpora
  contain almost all of it. `evotune_baseline` refuses; do not work around it.
* **They are not a generative editor.** Substitutions only, so every generated sequence keeps the
  template's length and coordinate system. `editjumps/core/length_capability.py` is where each
  evaluator declares this, and this one declares it cannot grow.
* **Not converged, deliberately.** 2,000 steps exists to be budget-matched against the editor's
  arms, not to converge — see [Training Hyperparameters](#training-hyperparameters).

## Bias, Risks, and Limitations

* **Single-family by construction, which is the point and also the limitation.** Each trunk has been
  pulled toward one antibody family's statistics; nothing here measures what that did to the rest of
  protein space.
* **The `epochs: 20` figure in `params.yaml` is a trap that has already cost 23 GPU-hours.** It was
  written believing a family is a few hundred sequences. The Ty1 FASTA holds 38,687 usable
  homologs. One epoch at batch 16 is **2,277 steps** — the family's *train* split, which is what the trainer iterates; 38,687/16 = 2,418 counts the whole family and is the wrong denominator — so 20 epochs is **45,540**. (2,277 is the artefact's own figure: `ty1-evotune_esm.json` records `final_epoch` 0.87835 at 2,000 steps.) Sky job 58 ran that for
  23 h 23 m and reached step **825** — a 44-day horizon — and it was also silently on CPU, which is
  a separate bug. `max_steps: 2000` is the explicit bound that replaced it; `epochs` survives only
  as a second cap for a family small enough to exhaust first.
* **The validation split says "did it converge", not "does it generalise".** It is 5% of the
  **train part only**.
* **Nothing measures the disjoint pair as trunks.** They have no `metrics/evotune/` files, so their
  MLM loss is unknown; only the baselines built on them are measured.
* Bias in the demographic sense does not apply — antibody repertoire sequences. Dual-use is stated
  rather than assessed, as on the other cards: these trunks propose substitutions inside a template
  the caller supplies, with no target and no screen, and are not distributed.

### Recommendations

* **Three knobs are shared between stages and must not be set independently.** `n_templates`,
  `holdout_size` and `seed` decide the §4.2 train/inference/holdout partition: `evotune_esm` trains
  on the train part, and both baseline stages score against the holdout part. Set them differently
  in the two places and the baseline is scored against its own training data. `evotune_baseline`
  checks the intersection and refuses rather than reporting a number.
* **Compare a baseline's metrics against its own realised mutation count**, not the target. The
  unforced baseline lands *below* budget because an MLM infilling a conserved framework column
  usually returns the residue already there; the forced one spends its budget in full — its
  `mutation_budget` block records `hit_budget_fraction: 1.0` and `sd_realised: 0.0` on both
  families. Its *Levenshtein* reads 5.00 on Ty1 and 3.9975 on HER2-VH, the shortfall being the
  distance estimator collapsing two adjacent substitutions, not a position left unmutated.
* **Read `alignment` before comparing anything positional across files.** Each run projects onto its
  own `split.templates[0]`: L=121 for Ty1, L=122 for HER2-VH in the default frame. The positional
  metrics live in a different frame per family.

## How to Get Started with the Model

```bash
# adapt a trunk to one family (GPU; A100 for 650M)
make evotune FAMILY=data/interim/seed_families/Anti-SARS-CoV-2_VHH_Ty1.fasta

# both evotuned baselines from a trunk, in the default frame
make evotune-baselines

# the six model-based cells of the disjoint §4.2 table; the four evotune ones need the dj trunks
make disjoint-baselines TRUNKS=data/local_dj
```

`make disjoint-baselines` expects `TRUNKS` to be a directory holding `evotune-dj-ty1/` and
`evotune-dj-her2vh/`; pull them from the GCS prefix above.

## Training Details

### Training Data

One homolog family FASTA per checkpoint, from
[**the seed homolog families**](datasets/seed-homolog-families.md) — that card is the description of
what a family is, how it was searched, why there are two usable ones against the paper's six, and
why the third file holds two sequences. Three consumers share it. What is specific to this trunk:

| | value | source |
|---|---|---|
| Ty1 family | 38,687 members, 38,553 distinct, lengths 107–143, median **121** | [dataset card](datasets/seed-homolog-families.md#dataset-structure) |
| HER2-VH family | 43,088 members, 42,839 distinct, lengths 101–132, median **122** | same |
| adaptation corpus per checkpoint | the family's train part; exact counts in [Checkpoints](#checkpoints) | the shipped `evotune_family.train.txt.gz` |
| held-out for validation | 5% of the **train part only** | `params.yaml` `evotune.val_fraction` |

**One correction, because this card carried it and it was wrong in two ways at once.** An earlier
revision described the search's E-value as "**0.1** — deliberately looser than the paper's 1e-1, as
an experiment: our seed families came back 8–28× smaller than the paper's Table 2". Both halves
fail. **0.1 *is* 1e-1**, so the setting is the paper's own value and not looser than it — the row
was copying a `params.yaml` comment written for a `10.0` value that has since been reverted, and
the comment was not. And "8–28× smaller than Table 2" describes the retracted **394 / 396 / 330**
family counts, which were MMseqs2's default 300-prefilter-hits-per-query cap reporting a tool
default rather than a homolog count; `max_seqs: 100000` lifted it. The tracked families are
**38,687 and 43,088**, which are 11.6× and 3.9× *larger* than Table 2's 3,335 and 10,979 — so the
card contradicted its own family table two rows further down. The real finding, which the
[dataset card](datasets/seed-homolog-families.md#scope-of-reproduction) now owns, is that a 100×
looser E-value moved the counts by **0.00%**, because E-value 0.1 is selective against UniRef30 and
nearly vacuous against an antibody-only database, and that is why `min_identity: 0.7` exists.

The §4.2 partition, shared by all three stages: `n_templates: 20` (generated *from*),
`holdout_size: 200` (generated sets scored *against*), `seed: 0` (which members land where).

Objective: masked language modelling at `mlm_probability: 0.15` — "the same self-supervised
learning objective" as the corpus pretrain, which is §2.2's own phrasing.

### Training Procedure

#### Preprocessing

Tokenize each family member with the ESM-2 tokenizer, truncate at `max_length: 160` — a family
FASTA holds single V domains, not joined pairs, so this is the right length and is not the corpus
pretrain's 280. Mask 15% per batch via `DataCollatorForLanguageModeling`.

#### Training Hyperparameters

Read from `training_args.bin`, which is the authoritative artefact and is **byte-identical across
all four checkpoints**. Where `params.yaml` disagrees, the artefact wins and the table says so.

| key | artefact (all four) | `params.yaml` | note |
|---|---|---|---|
| trunk | `esm2_t33_650M_UR50D`, 33 layers / 1280 wide (from each folder's `config.json`) | `data/pretrain/base_checkpoints/esm2_t33_650M_UR50D` | agrees |
| `max_steps` | **2000** | 2000 | agrees. ~0.88 of a pass over the Ty1 family's train split (`final_epoch` 0.87835), and the same order as the editor arms' 3,000-step sweeps. Explicitly bounded, because `-1` would mean 45,540 steps. |
| `per_device_train_batch_size` | **16** | 16 | agrees |
| `gradient_accumulation_steps` | **1** | — | not a params key |
| `num_train_epochs` | **3.0** | 20.0 | **disagrees, and is inert** — `max_steps` binds first either way. `deploy/gcp/train.sky.yaml`'s `evotune` branch never passes `--epochs`, so the run took the CLI's own default of 3.0 rather than `params.yaml`'s 20.0. |
| `learning_rate` | **5e-05** | — | **not a params key at all.** HF's default. |
| `optim` | **`adamw_torch_fused`** | — | HF's default |
| `lr_scheduler_type` | **linear** | — | HF's default |
| `warmup_steps` / `warmup_ratio` | **0 / None** | — | HF's default: no warm-up |
| `weight_decay` | **0.0** | — | HF's default |
| `adam_beta1` / `adam_beta2` / `adam_epsilon` | **0.9 / 0.999 / 1e-08** | — | HF's defaults |
| `max_grad_norm` | **1.0** | — | HF's default |
| `seed` / `data_seed` | **42 / None** | — | HF's default. Note this is *not* the `evotune.seed: 0` that decides the §4.2 partition — two different seeds, two different jobs. |
| `bf16` / `fp16` | **True / False** | `precision: auto` | `auto` resolved to bf16 on the A100. Training regime: **bf16 mixed precision**. |
| `eval_strategy` / `eval_steps` | **steps / 500** | — | |
| `save_steps` | **500** | 200 | **disagrees** — the run took the evotune CLI default, not `params.yaml`'s. |
| `save_total_limit` | **2** | 2 | agrees |
| `dataloader_num_workers` | **0** | 0 | agrees |
| `logging_steps` | **50** | — | |
| `mlm_probability` | 0.15 (a collator argument, not in `training_args`) | 0.15 | agrees |
| `max_length` | 160 (a tokenizer argument) | 160 | agrees |

**Six of the numbers that decide this model's optimisation are `transformers` defaults that no file
in this repository states** — the learning rate above all. They are recorded here because
`training_args.bin` recorded them; they would change silently if `transformers` changed a default,
and nothing in `dvc.lock` would notice.

**Budget in epochs**, from each corpus's exact size: 2,000 × 16 = 32,000 sequences, so
**0.879 epochs** on Ty1 and **0.790** on HER2-VH — which is what
`metrics/evotune/*-evotune_esm.json` reports as `final_epoch` (0.8783 and 0.7902), so the shipped
corpus counts and the step budget reconstruct the recorded epoch to three decimals. That is the
check that the line counts above are the corpora these runs actually saw. On the disjoint corpora
the same budget is **1.84** and **1.62**
epochs, and that figure is **not recorded anywhere** because those runs banked no metrics.

#### Speeds, Sizes, Times

| | value | source |
|---|---|---|
| training throughput | **5.78 it/s** on an A100 (job 66) | [`performance.md`](../performance.md), from [`findings.md`](../findings.md) |
| eval throughput | **155.9** samples/s on the Ty1 run, **150.0** on HER2-VH | `metrics/evotune/{ty1,her2vh}-evotune_esm.json` |
| eval wall-clock | 12.29 s for the Ty1 run, 14.20 s for HER2-VH | same |
| whole-job wall-clock | **not recorded** for any of the four | — |
| `model.safetensors` | 2,604,236,440 B (fp32; 651,043,254 parameters × 4 = 2,604,173,016, the rest being the safetensors header) | GCS object size |
| shipped corpus halves | 217,725–558,887 B gzipped per train file | GCS object sizes |

Why an A100 at all: **GPU memory**, not host RAM. 650M in bf16 needs ~16–18 GB, which outgrows an
L4's 24 GB once activations are counted. An earlier brief of this said the run "OOMs an L4's 16 GB
host RAM", which is wrong on both counts and is corrected in [`performance.md`](../performance.md).

And a cautionary figure that is **not** throughput: **85 s/step** appears in
[`findings.md`](../findings.md) for `evotune_esm` and is job 58 running entirely on CPU while a
rented A100 sat at 0% utilisation, 0 MiB of 40,960 used, 41 W of 400. When comparing anything
against a historical evotune number, check which side of that fix it falls on.

The same warning applies to the throughput row above, which is why it now reads both families off
their own artefacts. **145.5 samples/s** appears in [`performance.md`](../performance.md) and in
[`findings.md`](../findings.md)'s CUDA-revert comparison, read off a job log. It is **not a measured
throughput for either family**, as [`known_issues.md`](../known_issues.md) says in those words: the
artefacts record **155.941** for Ty1 and **150.042** for HER2-VH.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Two evaluations, and they answer different questions:

1. **The trunk itself** — MLM cross-entropy on 5% of its own adaptation corpus. Says "did it
   converge"; the corpus is family members, so it says nothing about unseen families.
2. **The baselines built on it** — the family's 200-sequence holdout as the scoring reference, 20
   templates × 20 variants, mutation budget matched to the editor's realised mean. The default
   frame's files are `metrics/evotune/{ty1,her2vh}-evotune_baseline{,_forced}.json`; the disjoint
   frame's are `metrics/disjoint/evotune{,-forced}-{ty1,her2vh}.json`.

#### Factors

* **Which family**, and it matters: the two are reported as a mean over two families in §4.2, and
  each family has its own alignment frame.
* **Which population** the trunk was adapted on — the whole point of the `dj` pair.
* **Budget mode.** `realised` tops up until the budget has actually mutated, capped at
  `max_rounds: 8`; `masked` is the literal single-round reading and lands below budget. Both are
  ours.
* **Forced or not.** Blocking the original amino acid at every masked position guarantees every mask
  changes, which is what makes the forced variant hit budget in round one.
* **Reference size.** `agreement_reference_n` is 200 in all four default-frame baseline files. Two
  runs at different reference sizes are not comparable, by a margin wider than any model difference
  measured here.

#### Metrics

The same §4.2 / Appendix B set as every other cell. Each run additionally reports its own realised
budget per sequence — `n_masked`, `n_changed`, `rounds`, `hit_budget` — so a shortfall shows up in
the metrics file rather than hiding in an assumption.

### Results

The trunks, as trunks:

| | final train loss | final eval loss | final epoch |
|---|---|---|---|
| `evotune-ty1` | 0.3430169620513916 | **0.28149229288101196** | 0.8783487044356609 |
| `evotune-her2vh` | 0.33540180110931395 | **0.2873181700706482** | 0.7902015013828526 |
| `evotune-dj-ty1` | **not recorded** | **not recorded** | — |
| `evotune-dj-her2vh` | **not recorded** | **not recorded** | — |

The baselines built on the disjoint trunks, Ty1, in the frame every other cell shares. Full tables:
[`metrics/appendix_table.md`](../../metrics/appendix_table.md).

| method | edits/seq (95% CI) | pooled pairwise Lev. | spectrum MMD | composition KL | covariance agr. | MIP agr. |
|---|---|---|---|---|---|---|
| **evotuned PLM** | 4.47 [4.24, 4.68] | 22.61 | 0.9762 | 0.000540 | 0.9069 | 0.8126 |
| **evotuned PLM, forced** | 5.00 [5.00, 5.00] | 25.22 | 1.0893 | 0.000531 | 0.8702 | 0.7355 |
| the Edit Flows editor, clock 40 | 4.58 [3.73, 5.50] | 24.47 | 0.9839 | 0.000503 | 0.9119 | 0.8267 |
| EvoDiff-MSA | 2.96 [2.67, 3.25] | 22.85 | 0.9722 | 0.000607 | 0.8919 | 0.7793 |

Paired bootstrap, against the unforced evotuned baseline, 2,000 draws:

| comparison | metric | margin | paired 95% | verdict |
|---|---|---|---|---|
| editor − evotuned, Ty1 | covariance | +0.0050 | [−0.0205, +0.0437] | **includes zero** |
| editor − evotuned, Ty1 | MIP | +0.0141 | [−0.0656, +0.0403] | **includes zero** |
| forced − unforced, Ty1 | covariance | −0.0366 | [−0.0654, −0.0229] | excludes zero |
| forced − unforced, Ty1 | MIP | −0.0771 | [−0.1539, −0.0664] | excludes zero |

#### Summary

**This baseline is the editor's closest competitor, and the two are indistinguishable.** All four of
the editor's leads over the unforced evotuned baseline include zero at 100 and at 2,000 paired
draws; the same test separates the forced variant from the unforced one on all four, so the null is
a measurement and not a blunt test. The supported sentence is "indistinguishable", not "our port is
better" — and one earlier claim, that the editor has a better k-mer distribution than evotuning,
died to seed noise: the across-seed MMD standard deviation is ~0.028 against a claimed gap of
0.0298, and on HER2-VH the same gap is 0.004.

**Forcing costs quality.** Blocking the original residue makes the baseline spend its whole budget
and it loses on every metric for that reason, which is exactly what §4.2's matched-budget constraint
exists to expose.

**The floors are no more comparable across studies than the method rows are**, which closes a loop
a reader might otherwise leave open: someone who accepts that the method rows cannot be compared
with the paper might still assume a *floor* can be, since a floor is "just random". Our
random-mutation floor reads covariance 0.7548 and MIP 0.5265 at reference n=200 against the paper's
six-dataset means of 0.897 and 0.660. (Those two are from `metrics/aligned/perseq-ty1.json`, the
leaked frame; the reported disjoint values are 0.8483 and 0.6617, and the point about cross-study
comparison holds in either.) Random pairing scores *above* our own ceiling rather than at it —
see `docs/reproducing.md`, "The floors and the ceiling".

**Do not requote that 13–14 point gap as a measured quantity.** It is a cross-study difference in
exactly the coordinate where cross-study differences are inadmissible: on a good-versus-poor
stand-in, the ratio these agreements express drifts 28 points on covariance and 13 on MIP as the
reference grows from n=20 to n=1600, and the paper states no split size anywhere. The gap is
evidence *that* the comparison fails, and is not itself a comparison. The sharpest form of the same
point: at n=20 the floor and the ceiling **invert** — random mutations 0.411 against real homologs
0.395 — so at a small reference uniform random substitution looks *more* like the family than
actual homologs do. These metrics do not merely lose resolution at small n; they can rank
backwards.

## Environmental Impact

- **Hardware Type:** one NVIDIA A100 (40 GB) per run, GCP managed spot via SkyPilot
- **Hours used:** **not recorded** for any of the four. At 5.78 it/s, 2,000 steps is under 6 minutes
  of stepping; the job wall-clock includes pulling a 2.6 GB checkpoint and is not on record.
- **Cloud Provider:** Google Cloud Platform
- **Compute Region:** europe-west4 — the only EU region with A100 in GCP's catalog
- **Carbon Emitted:** **not assessed.** These are the cheapest training runs in the repository and
  no estimate is offered rather than a fabricated one.

## Technical Specifications

### Model Architecture and Objective

`EsmForMaskedLM` at ESM-2 t33's configuration, unchanged — read from each checkpoint's own
`config.json`, and identical across all four.

| | value |
|---|---|
| layers | 33 |
| hidden size | 1280 |
| attention heads | 20 |
| intermediate size | 5120 |
| position embeddings | rotary, `rope_theta` 10000, `max_position_embeddings` 1026 |
| vocabulary | 33 |
| dropout (`hidden_dropout_prob` / `attention_probs_dropout_prob`) | 0.1 / 0.1 |
| dtype | float32 |
| `transformers` version recorded | 5.12.1 |
| parameters | **651,043,254** total. The encoder reports 649,400,981 and the masked-LM head 1,684,513; those do not sum to the total because the head's decoder weight is *tied* to the input embedding and counted once |

The parameter count is computed from the configuration and recorded in no artefact; it agrees with
the stored `model.safetensors` (651,043,254 fp32 slots is 2,604,173,016 B against the file's
2,604,236,440 B, the difference being the header).

Note the dropout: 0.1 in both places, which is `EsmConfig`'s default and differs from the editor's
trunk config (0.0 in both), because the editor's folder was written by a different code path. It is
recorded here rather than reconciled.

Objective: MLM cross-entropy at 15% masking. The *baseline* on top of it is a two-part construction
that this model is only half of: eq 12's normalised per-column entropy profile picks positions, and
this model iteratively infills them — mask the whole drawn set, then fill one position at a time in
random order, revealing each choice before the next.

### Compute Infrastructure

#### Hardware

A100:1 (40 GB) in europe-west4, managed spot. Never `A100-80GB:1` — the project's quota for it is 0
in every EU region while the 40 GB quota is 64, so an 80 GB request retries forever.

#### Software

`transformers` `Trainer` at 5.12.1, `DataCollatorForLanguageModeling`, `uv`-locked environment,
`--group train`. Launched through `deploy/gcp/train.sky.yaml` with `MODE=evotune`, and
`EVOTUNE_DISJOINT_PAIRS=data/pretrain/oas_homolog_pairs.tsv.gz` for the `dj` pair — that env var is
what passes `--disjoint-from-pairs` and is the whole difference between the two pairs of
checkpoints.

One operational note worth keeping: this trunk was **missing from every store** before the
`evotune` mode existed — never pushed to DVC, in no GCS prefix, and recorded nowhere at all until
the stage opened an MLflow run. Establishing that it was simply gone cost about an hour. And all
three evotune stages write to *fixed* metrics paths, so a second family's run overwrites the first's
numbers; `metrics/evotune/` exists because the files have to be copied out under a family prefix
before the next run lands.

## Citation

The recipe is Evotuning, cited by EvoFlows §2.2 to **Alley et al., 2019** (UniRep). This repository
records the citation only in that form and carries no full author list or venue for it, so take the
entry from the paper rather than from here. The trunk's initialisation is Meta's ESM-2 — see
[the OAS trunk's card](esm2-oas-trunk.md#citation) for the same caution.

For the paper being reproduced and for this reproduction, see
[the editor's card](edit-flows-editor.md#citation) and `CITATION.cff`.

## Model Card Authors

Written from each checkpoint's own `config.json`, `training_args.bin` and shipped corpus halves, plus
`metrics/evotune/`, by the `editjumps` maintainers.

## Model Card Contact

Via the project repository.
