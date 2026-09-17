---
# Hugging Face model-card metadata, in the documented key shape. Not published to the Hub: this
# project's weights are private (docs/weights.md).
license: mit
library_name: transformers
# The IMMEDIATE parent is `data/pretrain/esm2_oas`, which dvc.yaml hard-codes for this stage (see
# "Finetuned from model" below). That is not a HuggingFace id, so this field names the checkpoint
# esm2_oas itself was adapted from; the two rows are answering different questions on purpose.
base_model: facebook/esm2_t12_35M_UR50D
base_model_relation: finetune
tags:
  - biology
  - protein-language-model
  - sequence-editing
  - flow-matching
  - reproduction
---

# Model Card for the deterministic (§4.1) editor (`edit_flows_deterministic`)

The same Edit Flows architecture as [the main editor](edit-flows-editor.md), trained on EvoFlows
**§4.1's synthetic deterministic-rule dataset** instead of on natural homolog pairs. It exists so
that §4.1's benchmark measures the experiment the paper describes rather than zero-shot transfer to
an unseen rule set.

**Two caveats before any number here is used.** First, the only §4.1 results committed to
`metrics/` were produced by an editor that had **never seen the rules** — a null control, and they
are labelled as such below. Second, this model's three `dvc.lock` entries are **not on `main`**:
they were adopted on an unmerged branch, so a fresh clone's lock carries no hash for the artefact
and `dvc pull data/pretrain/edit_flows_deterministic` will not resolve.

**But the experiment has been run, and it met Table 1 on every class.** Sky job 48 trained this
model and job 53 scored it over the clock sweep; the reproduction ledger reports the result, and
[Results](#results) carries it here, labelled `LEDGER-SOURCED` throughout because
`metrics/deterministic_benchmark.json` was never banked. That distinction is the point: **the gap
in this arm is a missing artefact, not a missing result.** An earlier revision of this card said
§4.1's status was "experiment not yet reported", which was wrong in a way that mattered — it is
reported, it just cannot be cited to a file in this repository.

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
| claim | `reproduction` — EvoFlows §4.1 |
| stages | `build_deterministic_pairs` (§4.1's rules, verbatim, applied to natural sequences), `train_deterministic_editor`, `deterministic_benchmark` |
| why this model has to exist | `claims.md` states it: "§4.1 trains on those pairs; evaluating an editor that never saw the rules measures zero-shot transfer, not the paper's experiment". A flow model can only score above chance on deterministic positional rules if it was trained on them. |
| why §4.1 is worth the trouble | it is **the only evaluation in this repository with a known answer.** Every §4.2 metric is a distributional comparison against a reference; §4.1 has ground truth per position, per edit type. |

Four choices §4.1 does not specify. **Three are in `editjumps/core/edit_flows/README.md`'s "where the
papers are silent" table (its entries 5, 6 and 7); the fourth is not, and that is deliberate on the
register's part** — it files the alignment ceiling under "not implemented" and assigns it to
`editjumps/pipeline/evaluate/deterministic_benchmark.py` rather than to the method folder. It is
listed here because a reader of *this* card needs it, not because the register records it as an
unspecified choice.

* **0- or 1-based indexing** for the rules' `Sub(i + 5, H)` and `Ins(i - 2, S)`. Chosen 0-based. The
  choice shifts every edit by one position and changes no count, so precision and recall per type
  are unaffected — per-position agreement with the paper's figures would not be.
* **What to do when `i + 5` runs past the end, or `i - 2` is negative.** Drop the edit. Clamping
  would pile several edits onto one terminal position and destroy the one-to-one `z0 → z1` property
  the whole section rests on.
* **What happens when one position is both deleted and substituted.** The deletion wins, decided by
  §4.1's own stated order (insertions, then deletions, then substitutions).
* **A ceiling the paper does not acknowledge.** §4.1 claims its construction "yields a unique `z1`
  for each `z0`, enabling precise assessment of mutation type, position, and amino acid identity".
  The first half holds; the second does not follow. `z1` is unique but the *edit script* is not, and
  the rules' script is not always the cheapest — worked example: `z0[4:7] = "GHG"` → `z1[4:7] =
  "SHC"` costs 3 edits by the rules and 2 by a minimum-cost aligner. So a perfect editor cannot
  score 1.0 on the alignment path. `oracle_ceiling` measures that per run, and **per-token
  provenance removes it entirely** — 1.000 on all four classes.

## Model Details

### Model Description

- **Developed by:** `editjumps`
- **Model date:** trained by **sky job 48**, which also pushed the artefact to the DVC remote. The
  lock entries recording it were adopted in a commit that is not an ancestor of `main`.
- **Model version:** `dvc.lock` (job 48's, off-`main`) hash
  `f6c4b5df951ed8f15f56ae24914a69ee.dir`, 6 files, 678,365,048 B
- **Model type:** discrete flow matching over edit operations — the same `EvoFlowsModel` class as
  the main editor, at the `linear` / `fresh` head parameterisation
- **Language(s):** not applicable — amino-acid sequences over ESM-2's 33-token vocabulary. Inputs
  are **single heavy chains**: §4.1 is defined on "a natural protein sequence", and a joined
  `VH.VL` string is two proteins with a separator, across which an alignment is free to shift
  residues.
- **License:** MIT for the code. **Weights are not publicly obtainable**
  ([`weights.md`](../weights.md)).
- **Finetuned from model:** `data/pretrain/esm2_oas` — the stage hard-codes this trunk, unlike
  `train_edit_flows`, whose trunk is a parameter.

### Model Sources

- **Repository:** the rules and the ground-truth scorer are `editjumps/core/edit_flows/deterministic.py`;
  the dataset builder is `editjumps/pipeline/preprocess/pretrain/deterministic_pairs.py`; the scoring
  harness is `editjumps/pipeline/evaluate/deterministic_benchmark.py`. Training reuses
  `editjumps/pipeline/train/train_edit_flows.py` unchanged.
- **Paper:** EvoFlows [arXiv:2603.11703](https://arxiv.org/abs/2603.11703) §4.1 and its Table 1
- **Deviation register:** `editjumps/core/edit_flows/README.md`
- **Weights:** DVC-tracked at `data/pretrain/edit_flows_deterministic`, pushed to
  `gs://$DVC_BUCKET/` by job 48

## Uses

### Direct Use

Being scored by `deterministic_benchmark` over a clock sweep, which is the only measurement in this
repository with per-position ground truth and therefore the only one that can put numbers in the
columns of the paper's Table 1.

### Out-of-Scope Use

* **Not an antibody editor.** It was trained on a synthetic rule set — every pair in its corpus was
  produced by applying fixed positional rules to a natural sequence — so its edits are the rules,
  not homology.
* **Its clocks are not the main editor's.** The rules apply ~17 edits to a ~120-residue VH while the
  homolog calibration puts clock 25/40/60 at only 2.2/4.0/5.1 edits, so the homolog defaults
  under-edit this task by about 4× and would score near zero for that reason alone. This stage
  sweeps `50,100,200,400` instead, which is the paper's own Figure-2 range — the first independent
  reason to think that range is the right scale.
* **Not a released model.** See the lock caveat at the top.

## Bias, Risks, and Limitations

* **No committed evaluation of this model exists.** `metrics/deterministic_benchmark.json` is
  declared as an uncached output of the stage and is not in the repository, so every per-class
  number in [Results](#results) is `LEDGER-SOURCED` — read from the reproduction ledger's account
  of jobs 48 and 53 rather than from a file here. The gap is the artefact, not the knowledge, and
  it is the single largest *bookkeeping* gap in the reproduction's §4.1 arm.
* **The synthetic corpus is derived from the OAS train split**, so it inherits the corpus's
  antibody-only, paired-chain, length-filtered character even though each example is a lone heavy
  chain.
* **The alignment path has a measured ceiling.** On real chains: insertion 0.861, substitution
  0.948, deletion 0.954 (50 sequences). Read a per-class number against that ceiling, or use
  `--provenance` and read it against 1.0.
* Bias in the demographic sense does not apply. Dual-use: this arm generates edits from fixed
  synthetic rules and is the least capable model here as a designer.

### Recommendations

* **Use `--provenance`.** The stage passes it by default now. Scoring from the sampler's own
  per-token provenance index — for each output token, which input token it descends from, or `None`
  if the sampler inserted it — removes the alignment ceiling entirely: measured on 5 real heavy
  chains with a perfect editor in both columns, insertion goes 0.778 → **1.000**, substitution
  0.939 → **1.000**, deletion 0.951 → **1.000**, and substitution identity 0.935 → **1.000**. That
  is what makes the paper's Table 1 insertion precision of 0.820 a comparable number rather than a
  figure sitting 0.04 under our own harness limit.
* **Score the heavy chain alone.** Pointing the benchmark at the joined corpus once cost a
  half-sequence indexing bug: the scorer's alphabet encoder dropped the `.` separator, so alignment
  indices referred to a sequence one shorter than the label list and every position after the
  junction was scored one out. Local validation on pure-alphabet random strings is exactly why it
  passed. `predicted_edit_labels` now raises on any off-alphabet character in `z0`, and
  `test_deterministic_benchmark_ceiling_holds_up_on_real_antibody_sequences` pins it.
* **Adopt job 48's lock before pulling.** `dvc status -c <target>` reports "in sync" for a target
  whose hash the lock does not carry — in-sync compares what the lock knows about, so it is silent
  about an absent entry. Checking that a pull actually resolves is the test that catches it.

## How to Get Started with the Model

The whole §4.1 arm, build through score, is one target:

```bash
make deterministic-benchmark
```

which is `dvc repro build_deterministic_pairs train_deterministic_editor deterministic_benchmark`.
The benchmark step needs `--rate-head linear --q-head fresh`, and the stage passes
`params.yaml`'s values — which for **this** model are the correct ones, unlike for the main editor.
That is worth stating plainly: the two editors were trained with *different* head parameterisations,
so `params.yaml` is right here and stale there.

## Training Details

### Training Data

`data/pretrain/deterministic_pairs.tsv.gz`, md5 `5c2c3ea98b239a0f3b491c85ce034146`, 635,857 B.
Built by applying §4.1's rules verbatim to natural sequences — the rules are the paper's, the
sequences are ours, which is the whole construction `claims.md` classifies as a reproduction.

The natural sequences the rules are applied to come from the corpus described in
[**the OAS corpus dataset card**](datasets/oas-corpus.md) — its filters, decontamination, split and
md5s are there and are not restated here. Only what is specific to this dataset is below.

| | value | source |
|---|---|---|
| source | `data/pretrain/oas_corpus.train.txt.gz`, the corpus **train** split ([card](datasets/oas-corpus.md#dataset-structure)) | `dvc.yaml` `build_deterministic_pairs` |
| sequences drawn | **20,000** | `params.yaml` `deterministic.n_sequences` |
| chain | **heavy** — §4.1 is defined on one natural protein sequence, not a joined `VH.VL` | `params.yaml` `deterministic.chain` |
| edits the rules apply | ~**17** per ~120-residue VH | [`findings.md`](../findings.md), via `params.yaml`'s `deterministic.clocks` comment |
| rules | §4.1's, verbatim: an insertion, a deletion and a substitution keyed off occurrences of specific residues, applied in the paper's own order | `editjumps/core/edit_flows/deterministic.py` |

The `deterministic:` block is deliberately separate from `edit_flows:` in `params.yaml`: it is a
different dataset with a different edit density, and sharing hyperparameters would silently mistune
one of them.

### Training Procedure

#### Preprocessing

Identical to the main editor's — tokenize, align `(x0, x1)` with Needleman–Wunsch at unit cost,
draw `t ~ U(0,1)`, sample the mixture path, and read the per-column supervision off eq 23. The only
difference is what the pairs are.

#### Training Hyperparameters

Read from job 48's lock entry, which is the authoritative record of the artefact that exists. The
comparison column is the main editor's reported arm, and every difference is deliberate.

| key | this model | main editor (`faithful-appendixa`) | why different |
|---|---|---|---|
| `--model-name` (trunk) | **`data/pretrain/esm2_oas`** — hard-coded in the stage | `facebook/esm2_t12_35M_UR50D` | the §4.1 stage takes no trunk parameter |
| `deterministic.max_steps` | **2000** | 20000 (`edit_flows.max_steps`) | The params comment once read "10× `edit_flows.max_steps`", meaning 10× the *old* 200-step value; the raise to 20,000 inverted that, and the comment now says "a tenth of edit_flows.max_steps". The rules are simple and learnable, so the smaller budget is not the constraint it looks like. |
| `deterministic.batch_size` | **8** | 16 | |
| `deterministic.lr` | **1.0e-4** | 1e-4 | same |
| `deterministic.save_steps` | **500** | 1000 | |
| `edit_flows.rate_head` | **`linear`** | `mlp` | `params.yaml`'s value, and the stage does not override it |
| `edit_flows.q_head` | **`fresh`** | `esm_lm_head` | same |
| `edit_flows.path` | `needleman_wunsch` | `needleman_wunsch` | same |
| `edit_flows.schedule` | `linear` | `linear` | same |
| `edit_flows.loss` | `edit_flow` | `edit_flow` | same |
| optimizer | `torch.optim.Adam`, constant lr, grad clip 1.0 | same | one trainer |
| training regime | fp32 non-mixed | fp32 non-mixed | |
| seed | 0 | 0 | |
| validation | 5% by family (the trainer's own default), but **no early stopping**: the stage passes no `--patience`, and the trainer defaults `patience` to 0 | 5%, `patience: 6`, `min_delta: 1.0` | the sky YAML passes the early-stopping envs; `dvc repro` does not |

**Budget in epochs:** 2,000 steps × 8 pairs = 16,000 of 20,000 pairs, about **0.8 of an epoch**.

**Total parameters: 34,057,270** at this head parameterisation (33,269,521 trunk + 787,749 heads),
computed from the architecture — 1.16M fewer than the main editor's arm, because `linear` rate
heads and `fresh` token heads are smaller than Appendix A's. See
[the editor's card](edit-flows-editor.md#model-architecture-and-objective) for the full breakdown
of all four combinations.

#### Speeds, Sizes, Times

| | value | source |
|---|---|---|
| wall-clock | **not recorded** | — |
| step rate | **not recorded** for this arm. The comparable figure is ~1.05 s/step for a 35M editor on an L4, which would put 2,000 steps near 35 minutes. | [`findings.md`](../findings.md), "Editor training dynamics" |
| final loss / any training curve | **not recorded** outside MLflow | — |
| artefact | 678,365,048 B, 6 files — `encoder/` plus `evoflows_model.pt` plus the resume `checkpoint.pt` | job 48's lock entry |
| the pairs file | 635,857 B gzipped | same |

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

`data/pretrain/oas_corpus.val.txt.gz`, the corpus **validation** split — so the evaluated
sequences are disjoint from the ones the synthetic pairs were built from, which were drawn from the
train split. `deterministic.eval_n: 50` sequences, `eval_n_steps: 50` Euler steps each, over the
clock sweep `50,100,200,400`.

#### Factors

* **The clock**, which is the point of the sweep. §4.1's rules apply ~17 edits; the clock sets the
  expected edit count, and the homolog defaults (25/40/60) sit ~4× below the task.
* **Scoring path** — Needleman–Wunsch alignment against per-token provenance. This changes the
  achievable ceiling, not the model.
* **Chain form.** Single heavy chains only; the joined form is a measured bug source here, not an
  option.
* **Whether the editor was trained on the rules.** The dominant factor, and the reason this model
  exists.

#### Metrics

Per-edit-class precision and recall — insertion, deletion, substitution, `no_op` — plus
substitution *identity* (did it write the right residue), exact-match rate against `z1`, mean
Levenshtein to `z1`, and a **no-op baseline** (return `z0` unchanged), which is the floor a useful
editor must beat. `oracle_ceiling` and `oracle_ceiling_provenance` are re-measured in-run rather
than assumed, and `insertions_unattributable` and `inserted_tokens` count the ambiguity that cannot
be removed rather than hiding it.

### Results

**No result for this model is committed to `metrics/`.** Three things were measured, and each is
labelled with where it can be read from: the trained editor's own sweep (`LEDGER-SOURCED` — a real
run, no committed artefact), the null control (committed), and the harness ceiling (committed).

#### The trained editor, over the clock sweep — `LEDGER-SOURCED`

Sky job 48 trained this model on the synthetic pairs; job 53 scored it with `--provenance` on 50
held-out heavy chains at 50 Euler steps, over clocks 50 / 100 / 200 / 400. **Every value in the two
tables below is read from the reproduction ledger's §4.1 section, not from a file in this
repository**, because `metrics/deterministic_benchmark.json` was never banked. Treat them as a
recorded run whose artefact is missing, and re-run `make deterministic-benchmark` before quoting
them anywhere the artefact matters.

| clock | exact match ↑ | reading |
|---|---|---|
| 50 | 0.000 | too small a budget to finish the job |
| 100 | 0.100 | — |
| **200 — best** | **0.620** | mean Levenshtein to `z1` **0.52**, against **17.46** for editing nothing |
| 400 | 0.540 | over-editing degrades it again |

**A clear interior optimum, degrading on both sides** — 0.000 → 0.100 → 0.620 → 0.540. That shape
independently reproduces the precision/recall trade-off the paper's Figure 2 sweeps the clock to
show, on different data and a different implementation. Against the no-op baseline the margin is
**34×**: 0.52 residues from `z1` where returning `z0` unchanged sits 17.46 away.

Per class at clock 200, against the paper's Table 1:

| mutation class | EvoFlows Table 1, P / R | ours, P / R | verdict |
|---|---|---|---|
| `no_op` | 0.982 / 0.983 | **0.995 / 0.997** | at or above |
| insertion | 0.820 / 0.796 | **0.870 / 0.861** | at or above |
| substitution | 0.804 / 0.843 | **0.915 / 0.920** | at or above |
| deletion | 0.910 / 0.850 | **0.956 / 0.951** | at or above |

**Three caveats, all load-bearing, and none of them is "the numbers are provisional".**

* **Different data, with genuinely different class prevalences.** Ours is antibody VH, theirs a
  protein mix: `no_op` / insertion / substitution / deletion at 0.8518 / 0.0173 / 0.0593 / 0.0716
  against their 0.888 / 0.047 / 0.043 / 0.022. Precision and recall are prevalence-sensitive, so
  "at or above" means **at or above on our own class mix**.
* **n = 50 sequences, one run, no replicate seeds.**
* **The clock was swept and the best taken; theirs is unstated.** Their Figure 2 sweeps 50–1000 and
  Table 1 prints one number, so it cannot be told whether theirs is a best or a default. Our 0.620
  is a best-of-four.

**The two scoring paths agree at the operating clock**, which is what makes the table above
independent of which one you believe: job 53 re-scored the same generations from the sampler's own
per-token provenance (oracle ceiling 1.000) rather than by aligning output back to input (ceilings
0.861 / 0.948 / 0.954), and **6 of 32 values moved, all at clock 50, all tiny; clocks 200 and 400
are identical.** Our insertion precision of 0.870 sits *above* the 0.861 alignment ceiling, which
is the concrete demonstration that the ceiling was never a bound on a model — see
[Scope of Reproduction](#scope-of-reproduction).

#### The null control — committed

The *homolog*-trained editor scored on §4.1, job 43, 36 m, 50 val sequences, 50 Euler steps, at the
homolog clocks:

| clock | exact match | mean Levenshtein to `z1` | no-op baseline |
|---|---|---|---|
| 25 | 0.000 | 44.46 | **38.34** |
| 40 | 0.000 | 48.46 | **38.34** |
| 60 | 0.000 | 52.88 | **38.34** |

Per class at clock 40: insertion P = R = 0.000; substitution P = 0.037, R = 0.029; deletion
P = 0.128, R = 0.005; `no_op` R = 0.949.

**The harness ceiling**, so a per-class number can be read against something:

| class | via Needleman–Wunsch, 50 real chains | via Needleman–Wunsch, 5 real chains | via provenance, 5 real chains |
|---|---|---|---|
| insertion | 0.861 | 0.778 | **1.000** |
| substitution | 0.948 | 0.939 | **1.000** |
| deletion | 0.954 | 0.951 | **1.000** |
| `no_op` (precision) | — | 0.995 | **1.000** |
| substitution identity | — | 0.935 | **1.000** |

#### Summary

**Read the null-control table as a null control.** The editor is worse than doing nothing,
monotonically in the clock — returning `z0` unchanged scores 38.34 and every clock scores worse,
which is the signature of edits that are random with respect to the task. **This is not evidence
about the model**, because the experiment was wrong: that editor was trained on OAS homolog pairs
and had never seen these rules, so the run measures zero-shot transfer to an unseen rule set, which
can only return ~0. It says nothing about the model and, worse, nothing about clock normalisation —
the assumption the benchmark had been wired to settle.

The missing piece was a training stage on the synthetic pairs, and **that control is what
established that §4.1 requires training on those pairs — a conclusion the paper never states.**
§4.1 says only that it "validate[s] edit flows on a synthetic task with known expected mappings"
and constructs the set "to evaluate the model in a controlled setting"; it never says the model is
fitted to those pairs. Table 1's magnitudes are the evidence, and they are impossible zero-shot,
which the null control above demonstrates directly. So "§4.1 trains on its own pairs" is **an
inference from Table 1, not a quotation**, and it should be requoted that way.

**That stage now exists, this model is its output, and it meets Table 1 on every class** — the
tables at the top of this section, from jobs 48 and 53. §4.1's status is therefore: harness
verified (the ceiling is measured, the provenance path removes it, the indexing bug is found and
pinned by a test), **experiment run and reported, artefact not banked.** The remaining work is one
`make deterministic-benchmark` and a commit of `metrics/deterministic_benchmark.json`, not an
experiment.

One retraction is worth carrying, because the earlier reading changed a decision: *"the alignment
ceiling is far worse on real antibodies"* was an off-by-one, and the corrected ceilings are the
0.861 / 0.948 / 0.954 above rather than 0.463 / 0.556 / 0.577. So alignment ambiguity is **modest,
not disqualifying**, and the conclusion drawn from it — that per-token provenance was promoted from
"optional" to "necessary" — does not follow. Provenance remains the principled fix for the residual
gap; it is no longer urgent.

## Environmental Impact

- **Hardware Type:** GCP GPU via SkyPilot; **which GPU job 48 used is not recorded** in any
  committed artefact. A 35M editor fits an L4:1, which is this project's default.
- **Hours used:** **not recorded.** 2,000 steps at the measured L4 rate would be roughly 35 minutes.
- **Cloud Provider:** Google Cloud Platform
- **Compute Region:** **not recorded** for this job; europe-west6 is the L4 default
- **Carbon Emitted:** **not assessed.** Under an hour on one GPU; the hours and region an estimate
  would need are not on record.

## Technical Specifications

### Model Architecture and Objective

Identical in structure to [the main editor](edit-flows-editor.md#model-architecture-and-objective),
at a different head parameterisation and over a different trunk:

| component | this model |
|---|---|
| trunk | `data/pretrain/esm2_oas` — ESM-2 t12: 12 layers, hidden 480, 20 heads, intermediate 1920, rotary, vocab 33 |
| time MLP + FiLM | 292,800 + 461,760 |
| 3 × rate head, `rate_head=linear`: `Linear(480→1)` | 1,443 |
| 2 × token head, `q_head=fresh`: randomly initialised `Linear(480→33)` | 31,746 |
| **total** | **34,057,270** |

Objective: eq 6's Bregman rate-matching loss, unchanged. Sampling for the benchmark is Euler
tau-leaping at 50 steps, with the clock sweep applied as `scale = clock / len(x)`.

Two things the harness cannot score, and says so rather than dropping them: insertions and
deletions in the *alignment* scoring path recover a cheaper edit script than the rules used, and an
inserted token has no origin, so an inserted run is only known to sit *between* two surviving input
positions. The convention is "nearest preceding surviving position + 1", the same one
`edit_targets` uses, and on 5 real chains it costs nothing (`insertions_unattributable` 0 of 9
insertions) because §4.1's insertion and deletion rules rarely fire on adjacent positions.

### Compute Infrastructure

#### Hardware

Not recorded for job 48. L4:1 is the default for a 35M arm.

#### Software

`uv`-locked, `--group train`. The stage runs through `dvc repro`, unlike the main editor's
hyperparameter arms — which is why it has a lock entry at all, and why the entry's absence from
`main` is a real problem rather than a cosmetic one.

## Citation

See [the main editor's card](edit-flows-editor.md#citation). §4.1 is a section of the same paper.

## Model Card Authors

Written from job 48's `dvc.lock` entries, `params.yaml`, `dvc.yaml` and
[`findings.md`](../findings.md) by the `editjumps` maintainers.

## Model Card Contact

Via the project repository.
