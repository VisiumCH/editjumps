# Model Cards

Model cards for trained checkpoints following the [Hugging Face model-card][hf-cards] format, alongside dataset documentation in [`datasets/`](datasets/) following the [HF dataset-card][hf-datasets] format. These documents record initialisation trunks, training datasets, hyperparameters, compute environment, and data provenance.

Every reported number references its source artifact or experiment run. Where information was unrecorded, it is designated as not recorded (summarised in [Data Lineage and Unrecorded Attributes](#what-no-card-could-fill-in)).

## Model Index

| Model Card | Architecture & Training Scope | Provenance Claim ([`claims.md`](../claims.md)) |
|---|---|---|
| [**Edit Flows Editor**](edit-flows-editor.md) | Headline model: ESM-2 trunk with edit-rate heads trained on OAS homolog pairs (`faithful-appendixa`, staged as `data/pretrain/eval_B_stock_appA`). | `reproduction` (EvoFlows §3.4) |
| [Deterministic Editor](deterministic-editor.md) | Architecture trained on EvoFlows §4.1 synthetic rule dataset to validate rule learning. | `reproduction` (EvoFlows §4.1) |
| [`esm2_oas` Trunk](esm2-oas-trunk.md) | ESM-2 continuously pretrained on OAS corpus with MLM objective. | `reproduction`, **deviation** (§3.4 states no domain adaptation) |
| [Evotuned ESM-2 Trunks](evotuned-esm2.md) | ESM-2-650M checkpoints MLM-adapted to target homolog families (EvoFlows §4.2 baseline). Includes default and disjoint variants. | `reproduction` (EvoFlows §2.2 / §4.3) |

Provenance categories are defined in `editjumps/core/provenance.py` and documented in [`claims.md`](../claims.md).

## Dataset Index

| Dataset Card | Contents | Consuming Pipeline Stages |
|---|---|---|
| [**OAS Paired Corpus**](datasets/oas-corpus.md) | `oas_corpus`: 2,586,927 paired `VH.VL` sequences. | `pretrain_esm`, `build_homolog_pairs`, `build_deterministic_pairs`, `seed_homologs` |
| [**OAS Homolog Pairs**](datasets/oas-homolog-pairs.md) | 1,660,105 unordered sequence pairs for EvoFlows §4.2 coupling. | Editor training and disjoint evaluation sets |
| [**Seed Homolog Families**](datasets/seed-homolog-families.md) | Curated evaluation families and seed sequences. | Evotuned trunks and §4.2 benchmark evaluations |

## Shared Specifications

| Domain | Source of Truth | Rationale |
|---|---|---|
| **Training Data** | [Dataset cards](#dataset-index) | Shared corpus statistics unified across models. |
| **§4.2 Evaluation Protocol** | [`metrics/disjoint/README.md`](../../metrics/disjoint/README.md) | Standardized 20 templates × 20 variants evaluation protocol. |
| **Default Optimizer Settings** | [`esm2_oas` Trunk Card](esm2-oas-trunk.md#training-hyperparameters) | Learning rate 5e-5, AdamW, linear decay, seed 42 shared across MLM training stages. |

## Parameterization Note: Model Heads

The reported experimental checkpoint uses `rate_head=mlp` and `q_head=esm_lm_head` (matching Appendix A and eq 16). In contrast, `params.yaml` configures `linear` and `fresh` for local dry runs. Because head dimensions alter parameter tensor shapes, checkpoints must be loaded with matching `--rate-head` and `--q-head` arguments (see [Edit Flows Editor → Training Hyperparameters](edit-flows-editor.md#training-hyperparameters)).

## Which model-card conventions these follow

The heading strings and levels are the current Hugging Face
[model-card template][hf-template]'s, verbatim, because the strings themselves are load-bearing:
HF's own [regulatory-check tool][hf-regcheck] accepts `## Technical Specifications` and not
"Compute" or "Hardware", so keeping the template's spelling buys machine-readability for free.
The *rationale* for each section is [Mitchell et al. 2019][mitchell]'s, which is also where the
warrant for dropping sections comes from — the paper states its own sections "are not intended to
be complete or exhaustive, and may be tailored depending on the model, context, and stakeholders".

Departures from the template, all in the same direction — a reproduction is judged on setup, not on
release:

* **Added `## Scope of Reproduction`.** The HF template has no home for "which published claim is
  this checkpoint evidence for", which is the whole point of these artefacts. The section is the
  ML Reproducibility Challenge's [§2 *Scope of reproducibility*][mlrc] moved into the card.
* **`#### Training Hyperparameters` is a table whose row labels are the real config keys**, one
  column per arm — the presentation `chandar-lab/AMPLIFY_350M` uses, so a card can be checked
  mechanically against `params.yaml` and the run's own settings JSON.
* **`#### Factors` is reframed, not deleted.** Mitchell's *Groups* sense is scoped in the paper to
  "human-centric machine learning models" and does not apply to protein sequences; its
  *Instrumentation* and *Environment* senses do, and become the database release and clustering
  thresholds, the sequence-length regime, and joined `VH.VL` versus single V domain — which for
  this repository is a real train/test shift and not a formality.

What is deliberately absent, and why, is in each card's own text where it matters. The short list:
`## Model Examination` (marked optional and experimental upstream; nothing here does
interpretability), `## Glossary` (the notation lives in the deviation registers under
`editjumps/core/*/README.md`, which are longer and better), and the annotated template's
`## Societal Impact Assessment` (its prompts are child safety, NCII and violence — genuinely
inapplicable). `## Environmental Impact` is kept with its fields filled in as "not measured"
rather than removed, following `bigscience/bloom`'s precedent that a stated unknown reads as a
decision where an omitted section reads as an oversight.

### The dataset cards, and what was taken from the spec

The three cards in [`datasets/`](datasets/) follow HF's [dataset-card template][hf-datasettemplate]
and [metadata spec][hf-datasets], for the same reason the model cards follow theirs: the heading
strings are what makes a card machine-readable.

**Adopted verbatim.** Every heading and level of the template —
`## Dataset Details` / `### Dataset Description` / `### Dataset Sources`, `## Uses` with
`### Direct Use` and `### Out-of-Scope Use`, `## Dataset Structure`, `## Dataset Creation` with
`### Curation Rationale` and `### Source Data` (and its `#### Data Collection and Processing` /
`#### Who are the source data producers?`), `### Annotations` with
`#### Personal and Sensitive Information`, `## Bias, Risks, and Limitations` with
`### Recommendations`, `## Citation`, `## Dataset Card Authors`, `## Dataset Card Contact`. From the
metadata spec: `license`, `pretty_name`, `size_categories`, `source_datasets` (in its documented
`extended|<name>` shape), `annotations_creators`, `tags`, `configs`, and `task_categories` on the
one artefact where an HF task actually names what it is for (`fill-mask`, the corpus).

**Dropped, each for a stated reason.**

| dropped | why |
|---|---|
| `language`, `multilinguality`, `language_creators` | amino-acid sequences are not a natural language — the same call the model cards make at **Language(s)**, where the field is kept and answered "not applicable" |
| `task_categories` on the pair and family cards | no HF task category names "unordered homolog pairs supervising an edit process" or "the reference set of a distributional comparison"; a wrong category is worse than none |
| `dataset_info` / `features` / split sizes in YAML | the Hub generates those from a parquet conversion of a *published* dataset. Nothing here is published, so a hand-written block would assert a schema no tool had checked. The real schema is in `## Dataset Structure` with the md5 of the file it describes |
| `### Annotations`' `#### Annotation process` and `#### Who are the annotators?` | there is no annotation layer on any of the three. The parent `### Annotations` heading is **kept** and answers "none", on the same `bigscience/bloom` precedent as `## Environmental Impact` above: a stated absence reads as a decision, an omitted section as an oversight |
| `**BibTeX:**` / `**APA:**` under `## Citation` | this repository records surnames and arXiv ids and no full author lists, and `CITATION.cff` says so. A card that invents a BibTeX entry is worse than one that points at the source |
| `Demo`, `Funded by`, `Shared by` | nothing to put in them |

**Two departures, in the same direction as the model cards' three** — a reproduction is judged on
setup, not on release:

* **Added `## Scope of Reproduction`**, the same section the model cards add, in the same
  match / deviation / not-stated table shape. A dataset in a reproduction is a *claim* about what
  the paper's data was, and the HF template has no home for it.
* **Added `## Consumers`.** A meta card only prevents drift if the blast radius of changing it is
  visible from inside it, so each dataset card lists the stages and cards that read it. This is the
  section to check when a rebuild is proposed.

[hf-datasets]: https://huggingface.co/docs/hub/datasets-cards
[hf-datasettemplate]: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md

## What no card could fill in

Collected from the cards, because the list is more useful together than scattered. These are the
facts a future run should capture.

| missing | consequence |
|---|---|
| **No commit hash on any run.** MLflow records no git SHA, and `dvc.lock`'s code hashes disagree with `HEAD`. | No run can be tied automatically to the code that produced it. [`training.md`](../training.md) says to record the commit by hand. |
| **No parameter count in any editor artefact.** | The editor cards compute it from the architecture and check it against the stored tensor bytes; nothing in the repo states it. Writing `n_params` into a settings JSON at save time is the pattern to copy. |
| **No training curve, final loss or `val_loss` for any editor arm** outside MLflow. `metrics/` carries evaluation artefacts for the editor and no training artefacts. | The convergence finding (flat from ~step 6,000; the earlier plateau-by-step-800 reading is retracted in [`findings.md`](../findings.md)) is on a *different* run (job 10), and the reported arm's own convergence is not on record. The five arms' best losses do exist in the job logs — A 135.24, B **130.48**, C 134.76, D 133.04, E 121.70 — so what is missing is a citable file, not the numbers; and the standing finding that validation loss does **not** rank these arms the way generation quality does (D is second on loss **among the four 2×2 arms** — third of five, since the 650M arm's 121.70 leads — and the worst generator of the four) has no committed basis either. |
| **No wall-clock for the reported editor run.** Two 20,000-step A100 runs *are* timed — jobs 76 and 83, 3 h 43 s and 3 h 31 m — but they are the schedule A/B pair, not the reported arm. | Cost claims for the headline model rest on a ~1.05 s/step L4 figure measured on another arm. |
| **No settings JSON for editor runs.** `n_params`, `s_per_step` and the final loss decomposition. | Every editor hyperparameter has to be reconstructed from `deploy/gcp/train.sky.yaml`'s defaults plus the `Makefile` target that launched it. |
| **No committed §4.1 benchmark for the trained deterministic editor**, and its three `dvc.lock` entries are not on `main`. `metrics/deterministic_benchmark.json` is declared `cache: false` in `dvc.yaml` and is absent from the repository. | The only committed §4.1 numbers are the null control from an editor that never saw the rules. **This is a missing artefact, not a missing result:** the run happened (train sky job 48, score job 53) and the reproduction ledger reports it per class against Table 1. A card cannot cite it to a file, which is a different and smaller problem than not knowing it — [deterministic editor → Results](deterministic-editor.md#results) carries the numbers under exactly that label. |
| **`metrics/seed_homologs.json` is declared `cache: false`, recorded in `dvc.lock` with an md5, and absent from the repository.** | The seed-family member counts three cards depend on are sourced to [`findings.md`](../findings.md) rather than to a metrics artefact. |
| **No evaluation of the disjoint-population evotuned trunks as trunks** — no `metrics/evotune/` files for them. | Their MLM loss is unknown; only the baselines built on them are measured. |
| **`params.yaml`'s `pretrain:` block no longer describes the tracked `esm2_oas`.** | The file specifies a 650M full-corpus continue-pretrain; the locked artefact is 35M over 3,200 sequences. The lock wins, and the card says so. |

[hf-cards]: https://huggingface.co/docs/hub/model-cards
[hf-template]: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md
[hf-regcheck]: https://huggingface.co/spaces/society-ethics/model-card-regulatory-check
[mitchell]: https://arxiv.org/abs/1810.03993
[mlrc]: https://reproml.org/
