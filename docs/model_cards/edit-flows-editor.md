---
# Hugging Face model-card metadata, in the documented key shape. These cards are NOT published to
# the Hub: this project's weights are private (docs/weights.md). The block is here so the card is
# machine-readable in the same way a Hub card is.
license: mit
library_name: transformers
base_model: facebook/esm2_t12_35M_UR50D
base_model_relation: finetune
tags:
  - biology
  - protein-language-model
  - sequence-editing
  - flow-matching
  - antibody
  - reproduction
---

# Model Card for the Edit Flows editor (`faithful-appendixa`)

A discrete edit-flow generative model over amino-acid sequences: an ESM-2 encoder trunk with
per-position edit-rate and token heads, trained on aligned antibody homolog pairs to carry one
natural sequence to another by insertions, deletions and substitutions. It is this repository's
reproduction of the editor in **EvoFlows** (arXiv 2603.11703) §3.4, over **Edit Flows**
(arXiv 2506.09018), and it is the headline model of the reproduction.

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
- [Checkpoint inventory](#checkpoint-inventory)
- [More Information](#more-information)
- [Citation](#citation)

## Scope of Reproduction

Which published claims this checkpoint is evidence for, and which it is not. Classified in
`editjumps/core/provenance.py`, rendered in [`claims.md`](../claims.md).

| | |
|---|---|
| claim | `reproduction` — EvoFlows / Edit Flows. Evidence about that paper. |
| stage | `train_edit_flows` (eq 6 Bregman rate-matching loss, eqs 13–16 architecture) |
| what a result here supports | that the published specification, executed, behaves as described |
| what it can never support | "matches their implementation numerically". Neither paper released code or weights, so there is nothing to match against. |

Five knowing deviations from the two papers and eight choices the papers do not specify are
recorded in `editjumps/core/edit_flows/README.md`, with a measurement of what each one costs. The four
that most affect how a number reads:

* **The alignment scoring is unit gap/mismatch cost, not BLOSUM62 with affine gaps.** §3.2 states no
  scoring while citing Henikoff & Henikoff 1992, so the affine/BLOSUM reading is defensible and is
  implemented (`edit_flows.path: needleman_wunsch_blosum62`) — but the alignment's columns *are* the
  training labels, so switching it changes the label distribution. The unit-cost default is what
  every recorded number was trained against. Measured, the swap is **not** the lever it looks like:
  it re-organises indels (gap runs 3.47 → 2.41) without re-weighting the classes (substitution
  share 0.2396 → 0.2438), and the gap penalty it forces us to *invent* moves the labels about **7×
  more** than BLOSUM62 itself does (register deviation 2).
* **The `kappa(t)` schedule is linear**, where Edit Flows' experiments state cubic and its Figure 13
  code ships linear. Measured on a matched A/B pair — jobs 76 and 83, 20,000 steps each on one
  A100, held out on both families (`metrics/schedule/`): **no effect distinguishable from
  run-to-run variation** ([`findings.md`](../findings.md), "The schedule deviation costs nothing
  measurable"). Carry its own caveat with it: **the two arms are not budget-matched** — cubic edits
  7–16% more at the same clock (+0.71 edits on Ty1, +0.37 on HER2-VH) — so what small difference
  exists is confounded by an edit-count change the schedule itself causes, and the signs disagree
  across families. That the schedule moves the realised edit count at fixed clock is a finding in
  its own right: §4.2's matched-mutation requirement is not automatically satisfied between two
  schedules.
* **Clock normalisation is `scale = clock / len(x)`**, ours. §3.3 names the mechanism and its
  purpose ("a length-normalized rate scaling") and never gives the value. The formula is what
  makes the expected edit count a function of `clock` alone, which is what §4.2's matched mutation
  budget needs, and it is confirmed against real weights at two lengths — 11.30 edits predicted
  against 11.76 measured at L=127, 5.24 against 5.567 at L=235.
* **The Gillespie sampler is a deviation of ours, not the paper's procedure** (register deviation
  4). §3.3's eq 9–10 hold the state at `x_tn` and *numerically integrate* the rate over `s`; this
  implementation freezes the rates at the current `(x, t)` and draws the waiting time from that
  constant total, which is exact only if the generator is read as piecewise-constant between
  events. Closing it exactly costs one numerical integration per event. The cost is bounded by
  measurement rather than by argument: Euler and Gillespie differ by **+0.08 ± 0.21 edits** at the
  50 steps every reported run uses, and by 0.6–1.6% on pairwise Levenshtein. **Do not describe this
  as "EvoFlows' own sampler".** Nor is Gillespie "the exact one" and Euler "the approximate one" —
  §3.3 presents its method as "formally very close to the Euler integration proposed by Havasi et
  al. (2025)", differing by replacing repeated Bernoulli draws with a single uniform one so as to
  avoid "potentially large sub-leading terms". Both are approximations of the same generator.

## Model Details

### Model Description

- **Developed by:** `editjumps`; reproduction of a paper by Deutschmann, Ferragu, Ziegler,
  Aziznejad and Bixby (Cradle)
- **Model date:** the reported arm's loadable folder was written **2026-08-16** (GCS object
  timestamp on `models/edit_flows/faithful-appendixa/`)
- **Model version:** run tag `faithful-appendixa`, step 20,000. Staged locally for evaluation as
  `data/pretrain/eval_B_stock_appA`, and once under the alias `data/pretrain/eval_faithful-appendixa`
  — the same weights under two folder names (`metrics/aligned/README.md`).
- **Model type:** discrete flow matching over edit operations (CTMC over insert / delete /
  substitute), encoder-only trunk with five heads
- **Language(s):** not applicable — amino-acid sequences over a 33-token ESM-2 vocabulary. Inputs
  are joined `VH.VL` antibody chains, separated by `.`, which is a real token in that vocabulary.
- **License:** MIT for the code (`LICENSE`). **The weights are not released** and cannot be
  obtained from outside the `$PROJECT_ID` GCP organisation — see [`weights.md`](../weights.md).
- **Finetuned from model:** stock `facebook/esm2_t12_35M_UR50D`. This arm deliberately does *not*
  start from the repo's OAS-adapted trunk, because §3.4 describes no domain-adaptive pretraining —
  only "the encoder trunk of a pre-trained ESM-2 model".

### Model Sources

- **Repository:** this one. The method is `editjumps/core/edit_flows/`; the training stage and the
  torch model class are `editjumps/pipeline/train/evoflows.py`.
- **Paper:** EvoFlows, [arXiv:2603.11703](https://arxiv.org/abs/2603.11703), building on Edit Flows,
  [arXiv:2506.09018](https://arxiv.org/abs/2506.09018). Neither has released code or weights.
- **Deviation register:** `editjumps/core/edit_flows/README.md`
- **Weights:** `gs://$DVC_BUCKET/models/edit_flows/faithful-appendixa/`
  (private) — `encoder/` plus `evoflows_model.pt` (the loadable layout), and `checkpoint.pt`
  (resume state, which no evaluation can open)

## Uses

### Direct Use

Reproducing and auditing EvoFlows §4.2's distributional comparison, and generating homolog-like
variants of a template antibody sequence at a controlled edit budget. `editjumps edit` is the
front door; `editjumps generation-eval` is the scoring harness.

### Out-of-Scope Use

Three things it does not do, restated from [`limitations.md`](../limitations.md) because each has
been assumed:

* **It does not score its own variants, and cannot.** Every distributional quality metric needs a
  holdout of natural sequences from the input's own family, which one pasted sequence does not come
  with. Edit distance is not confidence. The model's edit-rate field was tested as a naturalness
  ranker: it separates natural from *corrupted* sequences almost perfectly and carries no signal
  among good ones.
* **It does not know any downstream property.** Nothing is told what a target property is at edit
  time; the model proposes evolutionarily plausible neighbours and which of them is better is still
  an assay's answer. EvoFlows generation is unconditional and this repository ships no property.
* **It is not a property filter, a design tool for wet-lab commitment, or a safety-screened
  model.** No generated sequence has been expressed or assayed.

Fine-tuning these weights further is not an intended use and is not supported by any stage here.

## Bias, Risks, and Limitations

The demographic-fairness reading of "bias" does not apply — the training data are antibody
repertoire sequences, not records about people. Three limitations and one risk that do apply:

* **Corpus bias is real and measured.** `homolog_pairs.max_pairs_per_family: 20` caps how many pairs
  each homolog family contributes, and it changes the edit-distance distribution it trains on: mean
  60.76 edits at cap 20 against an estimated 65.42 uncapped, with the substitution share 86.26%
  against 84.23% — indels under-represented by 12.9% relative, which is the part most likely to
  matter, since the loss is defined over alignment operations. The cap is engineering — full
  enumeration is 255,548,117 pairs — but the editor sees a different pair population than §4.2
  enumerates. Numbers, mechanism and the ~3% rebuild caveat:
  [dataset card](datasets/oas-homolog-pairs.md#the-cap-does-bias-the-distribution--measured).
* **The corpus is antibody-only and paired-chain only.** OAS paired, so the model has never seen a
  non-antibody protein, and never seen a single V domain as an input (see
  [Factors](#testing-data-factors--metrics)). Three of the paper's six seed proteins are therefore
  outside anything this model could be evidence about
  ([seed-families card](datasets/seed-homolog-families.md#the-papers-six-seeds-and-why-two-survive)).
* **The edit budget is approximate.** `--edits 5` is calibrated per checkpoint and per input and
  lands within roughly a factor of two. The per-variant edit distance printed with each result is
  the truth.
* **Dual-use, stated rather than assessed.** This is a generative model for antibody variable
  domains. It proposes evolutionarily plausible neighbours of an input the caller supplies; it has
  no target, no affinity model and no screen. No biosecurity evaluation has been performed, and the
  weights are not distributed.

### Recommendations

* **Read a table row against its own realised edit count**, never against the target budget. Only
  the forced evotuned baseline actually spends the §4.2 budget.
* **Do not compare an agreement score across runs at different holdout sizes.** `covariance_agreement`
  and `mip_agreement` move more with the reference size than they do between methods
  (`metrics/agree/README.md`).
* **What was easy and what was hard**, in the ML Reproducibility Challenge's sense: the loss and
  the path were implementable from the equations, and Euler τ-leaping is Edit Flows' own sampler.
  **The Gillespie sampler was not** — eq 9–10 specify an integration this implementation replaces
  with a frozen rate, which is a deviation of ours rather than an ambiguity inherited from the
  paper. What was not stated at all, and had to be invented, is the alignment scoring, the `kappa`
  schedule, the clock formula, and — the one that cost the most — the evaluation frame. **Six of
  the ten quantities in Figure 3, the paper's only results figure, cannot be recomputed from the
  paper as written**: three are plotted and never defined (entropy delta, JS divergence, profile
  log-likelihood); Covariance and MIP are undefined one level down, since B.3's eq 17–22 give the
  two *matrices* and no equation or prose maps either to the single number the panel plots; and KL
  has a defined formula (eq 23–24) with an undefined representation. Only three are exactly
  recomputable, and one of those three only modulo the unstated kernel `k`. The tenth panel, avg
  pairwise Levenshtein, is *not* counted undefined: it names its quantity and leaves only the
  pooling open, settled here by the triangle inequality
  ([`findings.md`](../findings.md), "This makes it 6 of 10 undefined, not 3").

## How to Get Started with the Model

The two head flags are **not optional**, and are the single most load-bearing line in this card:

```bash
uv run --group train editjumps edit \
  --sequence QVQLVESGGGLVQPGGSLRLSCAAS \
  --model data/pretrain/eval_B_stock_appA \
  --rate-head mlp --q-head esm_lm_head
```

Recovering the loadable folder from the GCS resume checkpoint, if that is all you have:

```bash
make restore-editor CKPT=gs://$DVC_BUCKET/checkpoints/experiments/faithful-appendixa/checkpoint.pt
```

`restore-editor` defaults `--model-name` to `data/pretrain/esm2_oas` and the two heads to
`params.yaml`'s values, so **this arm needs all three overridden** — see
[`weights.md`](../weights.md), "Restoring an editor from a checkpoint".

Training your own instead, which is the only route for a reader outside the organisation, is
[`training.md`](../training.md); the full-scale launcher for this exact arm is
`make jobs-train-faithful`.

## Training Details

### Training Data

[**OAS homolog pairs**](datasets/oas-homolog-pairs.md) — `data/pretrain/oas_homolog_pairs.tsv.gz`,
md5 `4cb9f191144bd20e3615e40d17ea128e`. Two columns per row, `x0<TAB>x1`, both fed to the model as
sequences.

**The dataset card is the description, and this card does not restate it.** Go there for the
corpus, the family and pair construction as a match/deviation/not-stated register, the pair
edit-distance table, and the measured cap bias. Eight models share that file; it is described once
so that a correction to it cannot land in one card and miss seven. What is specific to *this* arm:

| | value | source |
|---|---|---|
| pairs available | **1,660,105**, symmetric (`direction: none`) | [dataset card](datasets/oas-homolog-pairs.md#dataset-structure) |
| sequences fed to the model | 2 × 1,660,105 = **3,320,210** — this counts *sequences*, not examples: §4.2's "all unordered pairs" means one training example per pair | [`findings.md`](../findings.md), "What the training corpus actually is" |
| input form | joined `VH.VL`, length min / median / max **153 / 232 / 298** | [dataset card](datasets/oas-homolog-pairs.md#dataset-structure) |
| pairs actually consumed | 320,000 of 1,660,105 — see **Budget in epochs** below | — |
| mean edit distance between a pair's two members | **70.89** joined (median 71) on the 200,000-row head of the file, against **60.76** over the whole file. Both are real; they are not interchangeable, and the [dataset card reconciles them](datasets/oas-homolog-pairs.md#edit-distance-between-a-pairs-two-members) | that dataset card, which is the only record of the 70.89 |

Held-out set: 5% of pairs (`edit_flows.val_frac`), split **by family** rather than by pair — a
random pair-level split would put up to 20 pairs sharing one family's members on both sides. The
split is recovered from the pairs themselves (connected components over pairs-as-edges cannot merge
two families) and enforced sequence-disjoint. It is **not** homology-disjoint: a component can split
one family, so `val_loss` is an early-stopping and divergence signal, not a generalisation estimate.

### Training Procedure

#### Preprocessing

Per pair, per step: tokenize both sequences with the ESM-2 tokenizer, globally align them with
Needleman–Wunsch at unit gap and mismatch cost to get an equal-length `(z0, z1)` over the alphabet
plus one gap sentinel, draw `t ~ U(0,1)`, and sample the mixture path `z_t` (eq 5) at `kappa = t`.
The per-column supervision — which edit each column of `z_t` still needs — is eq 23's indicator.
Every example is re-aligned and re-sampled each step; nothing is cached.

The single sentinel is deviation 1: Figure 13's two-sentinel delete branch fires on an `(EPS, EPS)`
column where eq 23 says no edit is needed. Measured on 2,000 real pairs, that reading adds a mean
1.372 spurious deletions against 32.56 real edits and touches 49.2% of examples. Their reading is
selectable as `edit_flows.loss: edit_flow_figure13`.

#### Training Hyperparameters

Row labels are the real config keys. The reported arm is column **B**; the other three columns are
the 2×2 over trunk and head parameterisation. Values come from `params.yaml`, from
`deploy/gcp/train.sky.yaml`'s env defaults (which is the path a hyperparameter run takes — it
deliberately writes nothing to `dvc.lock`), and from the `Makefile` target that launched each arm.

| key | A `edit_flows_baseline` | **B `faithful-appendixa` (reported)** | C `heads-only` | D `stock-linear` | locked `dvc.lock` entry |
|---|---|---|---|---|---|
| `--model-name` (trunk) | `data/pretrain/esm2_oas` | **`facebook/esm2_t12_35M_UR50D`** | `data/pretrain/esm2_oas` | `facebook/esm2_t12_35M_UR50D` | `data/pretrain/esm2_oas` |
| `edit_flows.rate_head` | `linear` | **`mlp`** | `mlp` | `linear` | `linear` (params.yaml's value; the flag postdates the lock, like `chain` below) |
| `edit_flows.q_head` | `fresh` | **`esm_lm_head`** | `esm_lm_head` | `fresh` | `fresh` (same) |
| `edit_flows.chain` | `joined` | **`joined`** | ← | ← | the flag postdates these runs; all four trained on joined `VH.VL`, which is the input-form shift this card documents |
| `edit_flows.max_steps` | 20000 | **20000** | 20000 | 20000 | **200** |
| `edit_flows.batch_size` | 16 | **16** | 16 | 16 | **2** |
| `edit_flows.lr` | 1e-4 | **1e-4** | 1e-4 | 1e-4 | 1e-4 |
| `edit_flows.path` | `needleman_wunsch` | **`needleman_wunsch`** | ← | ← | `needleman_wunsch` |
| `edit_flows.schedule` | `linear` | **`linear`** | ← | ← | `linear` |
| `edit_flows.loss` | `edit_flow` | **`edit_flow`** | ← | ← | `edit_flow` |
| `edit_flows.val_frac` | 0.05 | **0.05** | ← | ← | not a locked param |
| `edit_flows.eval_steps` | 500 | **500** | ← | ← | " |
| `edit_flows.val_pairs` | 256 | **256** | ← | ← | " |
| `edit_flows.patience` | 6 | **6** | ← | ← | " |
| `edit_flows.min_delta` | 1.0 | **1.0** | ← | ← | " |
| `edit_flows.save_steps` | 1000 | **1000** | ← | ← | 100 |
| optimizer | `torch.optim.Adam`, constant lr, no warm-up, no decay, no weight decay | **same** | ← | ← | same |
| gradient clipping | 1.0 (max norm) — the loss weight `kappa_dot/(1-kappa)` spikes as `t → 1` | **1.0** | ← | ← | same |
| training regime | **fp32 non-mixed**. The trainer has no autocast; the 20 GB VRAM figure quoted for a 650M editor is this loop in fp32. | ← | ← | ← | ← |
| seed | 0 (`train_edit_flows`'s default; the launcher never passes one) | **0** | ← | ← | 0 |

**"Appendix A's parameterisation" is a reading of the paper, not a quotation of it.** Appendix A
specifies the token heads **twice, and incompatibly** — as "shallow MLPs" in one passage and as
"ESM2 language prediction heads" in another. The `mlp` / `esm_lm_head` pair above reads the first
phrase as the *rate* heads' specification and the second as the *token* heads', which is one
defensible reading; on re-reading the source, both passages bear on the token heads, so **which of
those two rows is quoting its own specification is not settled by the text.** Arm B follows the
second reading and arm A follows neither, so "paper-faithful" for arm B means "faithful to the more
specific of the paper's two mutually inconsistent statements" and not "faithful, full stop". Nothing
downstream changes — arm B is still the reported arm and still the one that follows Appendix A on
all three of trunk, rate heads and Q heads — but the phrase should not be read as though the paper
determined it.

**The `params.yaml` / reported-arm mismatch, stated once and precisely.** `params.yaml` holds
`rate_head: linear` and `q_head: fresh`. The reported arm was trained with `mlp` and `esm_lm_head`.
The heads change parameter *shapes*, so `EvoFlowsModel.load_trained` inspects the state dict and
raises rather than loading partially — correct behaviour, but the error names `params.yaml`, and
`params.yaml` cannot be right for four arms at once. Anything that opens this checkpoint must pass
both flags. `params.yaml` and `stages.py` name this mismatch at the knob
itself.

**The `dvc.lock` column is a smoke-scale run, not this model.** Hyperparameter runs go through
`deploy/gcp/train.sky.yaml` and not `dvc repro`, because a DVC stage declares one output and one
lock entry and two concurrent arms would fight over both. So the locked `train_edit_flows` entry
records 200 steps at batch 2 — 400 pairs, 0.024% of the 1,660,105, enough to prove the pipeline runs —
while every reported number comes from a 20,000-step run that `dvc.lock` does not describe.

**Budget in epochs:** 20,000 steps × 16 pairs = 320,000 of 1,660,105 pairs, about **19% of one
epoch**. No arm here completed an epoch, and none is claimed to be converged.

#### Speeds, Sizes, Times

| | value | source |
|---|---|---|
| wall-clock for this arm | **not recorded** | — |
| 35M editor on an L4:1 | ~**1.05 s/step**, so ~5.8 h for 20,000 steps | [`findings.md`](../findings.md), "Editor training dynamics" (job 10) |
| 35M editor on an A100:1 | ~**0.54 s/step**, derived from job 76 reaching step 19,990 in 3 h 0 m 43 s | [`findings.md`](../findings.md)'s A100 job table, and [`performance.md`](../performance.md), which now publishes the same derivation. Both come from the managed-job queue rather than from a committed artefact. |
| convergence | the run goes flat around **step 6,000**, improving to ~175 there. An earlier reading said step 800, from steps 1500–1630 and 2700–2870 sharing a windowed mean to one decimal — but 2500–2630 sits below both, so that was ±40 single-batch scatter. ~4.1 h of the 5.8 still buy little on training loss; the onset was off by ~7x. Measured on job 10, on *training* loss with no held-out set — a flat train loss with a still-falling val loss would look identical. **~175 and ~6,000 are read from the job log**, not from a committed file; the retracted step-800 reading is refuted by the windowed table in `findings.md`, which is committed. | same |
| this arm's own loss curve | **not recorded** outside MLflow; `metrics/` holds no training artefact for any editor arm | — |
| best validation loss | **130.48** at step 18,500 (early stop) — the best of the four 2×2 arms, against A 135.24, C 134.76, D 133.04 and the 650M arm's 121.70. **Read from the job logs, not from a file in this repository.** And do not read it as a quality ranking: arm D is second-best on validation loss and the worst generator of the four, so the training objective and generation quality order these arms differently | job logs, via the reproduction ledger |
| `evoflows_model.pt` (loadable weights) | 140,946,461 B | GCS object size |
| `encoder/model.safetensors` | 133,099,700 B | " |
| `checkpoint.pt` (weights + Adam state + RNG) | 422,852,185 B | " |
| checkpoint cadence | every 1,000 steps to GCS, bounding preemption loss to ~17 min | `deploy/gcp/train.sky.yaml` |

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

[**Two seed homolog families**](datasets/seed-homolog-families.md) — Ty1 (a VHH, 38,687 members,
median 121) and trastuzumab VH (43,088, median 122) — described on their own card, including why
there are two and not the paper's six, and why there is no light-chain arm.

Every reported cell is drawn in **one frame**, and
[`metrics/disjoint/README.md`](../../metrics/disjoint/README.md) is that frame's only description:
20 templates × 20 variants, holdout 200, agreement ceiling from 300 real homologs, seed 0, clock 40,
and `--disjoint-from-pairs` so **both** the templates and the reference come from family members
absent from the training pairs. Every cell records `reference_in_training_pairs`, and it is 0 in
every one. This card names its own cells — `metrics/disjoint/editor-{ty1,her2vh}.json` — and links
out for the rest, because `test_the_disjoint_table_cells_all_share_one_frame` asserts the frame
against the cells and a paraphrase here could only drift from it. The full metric set is
[`metrics/appendix_table.md`](../../metrics/appendix_table.md).

That frame exists because the earlier one leaked: 113 of 200 reference sequences on Ty1 and 11 of
20 templates had been in the editor's training pairs, while the baselines' own leak assertions had
refused the same overlap. **Every §4.2 number in this card postdates that fix.** A pre-fix figure
for this arm is not a comparable earlier reading of the same quantity — the reference set itself
changed, along with the alignment width (L=121 → L=119) — so the difference between the two is not
a measurement of what the leak cost, and nothing here should be differenced against a run in the
old frame.

#### Factors

Mitchell et al.'s *Groups* factor is scoped in that paper to human-centric models and does not
apply here. Its *Instrumentation* and *Environment* senses do, and one of them is a genuine
train/test shift rather than a formality:

* **Input form — the shift.** The editor trains on joined `VH.VL` (min length 153) and is evaluated
  on lone V domains (max length 143). **The two supports do not overlap at all**: 0 of 3,320,210
  training cells are ≤143, and 0 of 81,777 family members are ≥153. The *sequences* are half
  in-distribution — 52.0% of distinct Ty1 members appear verbatim as the VH half of a training line
  — so the shift is in the input form, not the content. Measured consequence: giving the editor its
  training input form does **not** improve the reported numbers
  ([`findings.md`](../findings.md), "Measured: giving the editor its training input form does not
  improve the reported numbers").
* **Alignment frame.** Positional metrics live in the frame of the split's first template — L=119
  for Ty1, L=118 for HER2-VH in the disjoint table. Comparable within a family only.
* **Reference size.** The two agreement metrics move more with the holdout size than between
  methods, and at n=20 the random-mutation floor and the real-homolog ceiling *invert*.
* **Clock setting.** `clock: 40` puts every arm on record at 3.47–6.17 edits **on the Ty1 seed family**
  (the twelve `metrics/dn/` runs, which are all Ty1); joined `VH.VL` input pushes the same clock higher. A "3–11 edit band"
  was previously described here as *stated by* §4.2; **it is not** — §4.2 gives no mutation count,
  and the band is not recoverable from Figures 3 or 5 either (see `params.yaml`, ⑤). Unclocked, this
  checkpoint spends **11.76** edits per sequence (`metrics/sweep/arm-B-c00.json`, the same template
  set as its 3.96 at clock 40). The ~54 sometimes quoted for this belongs to a *different* checkpoint
  on corpus `VH.VL` input; `docs/findings.md` exists partly to forbid that transfer. Whatever target you centre on, **one clock cannot centre
  two arms:** re-centring wants clock ≈ 71 for arm B and ≈ 52 for arm D, so matched clocks are not
  matched budgets. That conclusion is independent of where the target comes from, which is why it
  survives the band being withdrawn.
  That is a finding about §4.2's "matching the expected number of mutations per sequence across
  methods" as much as about this editor, and `edit_flows.clock` is left at 40 rather than retuned.

#### Metrics

The paper's own §4.2 / Appendix B set, split into what the paper defines and what it plots without
defining. Each cell keys its metrics under two names, and **the artefact's own split is not quite
the right one**, so read this rather than the JSON key. `defined_by_the_paper` in the files holds
six headline metrics: Levenshtein to template (the realised edit budget), pooled pairwise
Levenshtein (diversity), spectrum-kernel MMD, BLOSUM-smoothed composition KL, covariance agreement
and MIP agreement. `our_interpretation` holds five: entropy delta (signed), positional KL,
positional JS, plain JS, and profile log-likelihood — note that **positional KL is on the
`our_interpretation` side**, which is where the paper leaves it.

**Two of those six do not belong under `defined_by_the_paper`, and this is the correction rather
than the artefact.** B.3's eq 17–22 define the covariance and MIP *matrices* and stop there; no
equation and no prose in the paper maps either matrix to the single scalar its Figure-3 panel plots.
`matrix_agreement`'s reduction is a correlation, **inferred from the axis ranges** (0.742–0.995 and
0.227–0.992 fit a correlation and fit nothing else tried) rather than read off a definition. So the
right classification is "matrix defined, scalar reduction ours", and the finding is stronger than
"the reference size is unstated": *the plotted quantity is unstated*. The pooled-versus-within
pairwise reduction is ambiguous in the same way. Both are reasons a number here could differ from
the paper's for reasons unrelated to any model.

Appendix B.2's ESM-2 pseudo-log-likelihood is **not measured** in any disjoint cell — no run passed
`--pll-model` — and that is the metric least worth wanting. Where it has been measured it is the
one quantity that *flatters* this editor, mechanically: it beats the real-homolog bar because it is
barely changed, and ESM-2 rates a lightly-edited sequence as more probable. **Optimising PLL
optimises for doing less.** Spectrum MMD is the other half of that trap — no sign and no natural
zero, so it reports "too conservative" and "too wild" identically.

Intervals resample **templates**, not sequences: the centre of every distance is a mean over
templates, and the 20 variants of one template are correlated by construction, so a sequence-level
bootstrap is about 2.5× too narrow. Comparisons against another method use a **paired** bootstrap
that resamples one set of template indices and scores both methods on it.

### Results

Ty1, the disjoint frame, clock 40. Full tables including HER2-VH:
[`metrics/appendix_table.md`](../../metrics/appendix_table.md).

| method | edits/seq (95% CI) | pooled pairwise Lev. | spectrum MMD | composition KL | covariance agr. | MIP agr. |
|---|---|---|---|---|---|---|
| **this editor** | **4.58** [3.73, 5.50] | 24.47 | 0.9839 | 0.000503 | 0.9119 | 0.8267 |
| evotuned PLM (§4.3) | 4.47 [4.24, 4.68] | 22.61 | 0.9762 | 0.000540 | 0.9069 | 0.8126 |
| evotuned PLM, forced | 5.00 [5.00, 5.00] | 25.22 | 1.0893 | 0.000531 | 0.8702 | 0.7355 |
| EvoDiff-MSA | 2.96 [2.67, 3.25] | 22.85 | 0.9722 | 0.000607 | 0.8919 | 0.7793 |
| random mutations (floor) | 4.81 | 30.01 | 1.4054 | 0.000768 | 0.8483 | 0.6617 |
| random homolog pairing (ceiling) | 23.38 | 23.53 | 0.5966 | 0.000137 | 0.9751 | 0.9422 |

#### Summary

**The editor beats the random-mutations floor and ties the evotuned PLM.** The other model-free
row, random homolog pairing, is the **ceiling** — real homologs, exempted by §4.2 from the matched
budget — and the editor loses to it on every distributional column (MMD 0.9839 against 0.5966, KL
0.000503 against 0.000137, covariance 0.9119 against 0.9751, MIP 0.8267 against 0.9422), as any
generator should. All four of the editor's leads over the
unforced evotuned baseline (two metrics × two families) **include zero** under the paired bootstrap,
at 100 and at 2,000 draws. The same test on the same 20 templates separates the *forced* evotuned
baseline from the unforced one on all four,
in 100% of draws on three and 99.95% on the fourth (HER2-VH covariance) — so the null is a
measurement rather than a blunt test.

**One claim of the paper's this table does not support, and it is the one the paper is about.**
Every cell here also carries `diversity_novelty`, and it says the edits move the wrong way. Novelty
is the mean distance from each generated sequence to its nearest family neighbour, templates
excluded; novelty Δ subtracts the novelty the template already had, so **its sign says which way
the edit travelled** — negative is toward the family, positive is away.

| set, Ty1 disjoint cell | diversity | novelty Δ |
|---|---|---|
| **this editor** | 24.47 | **+1.855** |
| random mutations, at a matched count | 30.01 | +4.455 |

HER2-VH gives +1.947 for this editor against +3.605 for random mutation. **Every model row is
positive**: less far out than mutating at random, but the wrong way, where a real family member
sits at a *negative* Δ. Diversity is near natural (24.47 against 23.53 for a drawn-in family
member), so the shortfall is not diversity — it is direction. Two caveats on the table: the
`random_homolog_pairing` row cannot appear in it, because its "generated" sequences are drawn from
the same reference the novelty is measured against — 0.0 on Ty1 and 0.2875 on HER2-VH, where a few
of the drawn sequences fall outside the 400 nearest, so effectively zero by construction; and both definitions are ours, from the generative-protein literature, since EvoFlows
defines neither. What this means is worth stating plainly: an editor whose ~4.6-residue edits move
novelty *away* from the family has not been shown to make the kind of small in-distribution move
this class of model is wanted for. "Ties the evotuned PLM on MMD" and "produces useful antibody variants" are much further
apart than that summary line implies.

Four things that constrain how far these numbers travel:

* **The agreement ceiling is well below 1.0 and differs per family.** 300 real homologs against the
  same reference reach covariance 0.9534 / MIP 0.9317 on Ty1 and 0.9747 / 0.9283 on HER2-VH.
* **Only four columns may sit beside the published Figure 3, and one of them carries a condition.**
  `edits/seq`, pooled pairwise Levenshtein, spectrum MMD and positional KL are `cross_study`;
  everything else depends on this reference set and must not be compared with the paper. **Spectrum
  MMD travels only with its template count attached.** On one fixed 35M checkpoint at a constant
  300 generated sequences, MMD spans **1.588 → 0.544** from 10 × 30 to 300 × 1 — a 2.9× swing,
  larger than any model difference measured in this project, and enough on its own to move an arm
  across the paper's **best-method** band (0.29–1.10, the Random-pairing row of `panel_8`). Their
  panel as a whole spans 0.29–3.68 across all six methods, so the swing crosses the interesting part
  of it rather than the whole thing. The paper never states its template count, so "our MMD
  is inside their range" is not a result; a rev-10 headline of this project's was retracted for
  exactly that. The mechanism is reference coverage, not sample size: more templates seeds the
  generated set from more distinct family members, so its k-mer spectrum covers more of the natural
  one.
* **Covariance and MIP are within-study only, and normalising them does not fix it.** Expressing
  each as a fraction of its own real-homolog ceiling was proposed so the unstated reference size
  would cancel; tested, it does not. On a real-homolog "good generator" against a mutated "poor"
  one, the ratio drifts from 96.8% at reference n=20 to 68.6% at n=1600 on covariance and
  57.5% → 44.2% on MIP — a 28-point drift, wider than any model difference here. A small reference
  flatters a poor generator. These two columns may be differenced across the rows of *this* table,
  beside `agreement_reference_n: 200`, and against no source whose reference size is unknown. The
  paper's Figure 3 is exactly such a source: **it states no split size anywhere**, checked against
  all 22 pages — §4.2 gives only "Each homolog set is split into train, inference, and holdout".
* **Run-to-run variation is larger than several of the differences discussed.** `sched-linear`
  (job 76) and this arm are replicates of one configuration and differ by **0.0343** in Ty1 MIP
  agreement — larger than the editor's margin over the evotuned baseline. Across-seed MMD standard
  deviation is ~0.028 at the sweep's sample size. Read that control as an order of magnitude and
  not as an interval on the table above: it was measured in the **pre-disjoint** frame (L=121 /
  L=122, a reference that shares members with the training pairs, `metrics/schedule/`), so it sets
  the scale a difference must beat without being a bound computed in this frame. The pair that
  *is* computed in this frame is the paired bootstrap, and on Ty1 MIP it is sharper than "includes
  zero": the editor leads by +0.0141 at the centre while the resampled difference **centres on
  −0.0120 and puts the editor ahead in only 35% of 2,000 paired draws.**

## Environmental Impact

- **Hardware Type:** NVIDIA L4 (24 GB), one GPU, GCP managed spot via SkyPilot. Established
  indirectly: [`findings.md`](../findings.md)'s replicate comparison says `sched-linear` (job 76,
  A100) and this arm ran on "different hardware (A100 vs L4)". No artefact of this run names its
  machine.
- **Hours used:** **not recorded for this run.** The comparable figure is ~5.8 h for 20,000 steps
  at ~1.05 s/step on an L4.
- **Cloud Provider:** Google Cloud Platform
- **Compute Region:** europe-west6 (Zurich) by default for L4 work, co-located with the storage
  bucket; europe-west4 for A100 work, which is the only EU region with A100 in the catalog
- **Carbon Emitted:** **not assessed.** No estimate is offered rather than a made-up one; the
  [ML CO2 impact calculator](https://mlco2.github.io/impact#compute) would need the hours this run
  did not record. For scale, the whole arm is single-GPU and single-digit hours.

One efficiency fact is worth more than an emissions estimate here: **job 21 trained at 0.6% MFU**
over 6.4 h of wall clock — a couple of minutes of arithmetic. The loop is launch-bound, so a bigger GPU is
the wrong answer; `make jobs-calibrate` measures this before a machine is rented.

## Technical Specifications

### Model Architecture and Objective

Encoder trunk plus five heads, all fine-tuned jointly. `t` enters through a FiLM modulation of the
trunk's output, which is why the trunk itself needs no conditioning argument.

| component | shape at this arm | parameters |
|---|---|---|
| ESM-2 trunk (`EsmModel`, 12 layers, hidden 480, 20 heads, intermediate 1920, rotary positions, vocab 33) | — | 33,269,521 |
| time embedding MLP: `Linear(128→480)` → SiLU → `Linear(480→480)` | 128 Fourier features in | 292,800 |
| FiLM projection `Linear(480→960)` → (gamma, beta) | — | 461,760 |
| 3 × rate head (`insert`, `delete`, `substitute`), `rate_head=mlp`: `Linear(480→480)` → SiLU → `Linear(480→1)` | Appendix A's "shallow MLPs" | 694,083 |
| 2 × token head (`insert_q`, `substitute_q`), `q_head=esm_lm_head`: a deep copy of the trunk's pretrained masked-LM head per Q head (eq 16) | dense 480→480, LayerNorm, decoder 480→33, bias | 495,426 |
| **total** | | **35,213,590** |

**The parameter count is computed, not recorded.** No editor artefact states it. The figure above is
the architecture instantiated at this arm's config and counted; it agrees with the stored
`evoflows_model.pt` (140,946,461 B ÷ 4 = 35,236,615 fp32 slots, the difference being pickle
overhead) and with `encoder/model.safetensors` (133,099,700 B ÷ 4 = 33,274,925, the difference being
the safetensors header). For the other head combinations at the same trunk: `linear`/`fresh`
34,057,270; `linear`/`esm_lm_head` 34,520,950; `mlp`/`fresh` 34,749,910.

Objective: eq 6's Bregman rate-matching loss — minimise every output rate, plus a cross-entropy over
the correct edit at each still-wrong position, weighted by `kappa_dot / (1 - kappa)`. Sampling is
Euler tau-leaping by default (`edit_flows.sampler: euler`, 50 steps in every reported eval), with
a grid-free frozen-rate Gillespie next-event simulation selectable — ours rather than §3.3's
integrated procedure, per the caveat above; the two agree to 0.6–1.6% at the
clocks used.

### Compute Infrastructure

#### Hardware

L4:1 (24 GB) for this 35M arm; A100:1 (40 GB) for anything that outgrows 24 GB — the 650M arms need
about 20 GB in this fp32 loop. Never `A100-80GB:1`: the project's quota for it is 0 in every EU
region. Managed spot, which is 3–4× cheaper than on-demand and safe only because the trainer
checkpoints and resumes.

#### Software

`uv`-locked environment, `--group train` for torch. `transformers` 5.12.1 wrote the encoder config
in this checkpoint. Launch is SkyPilot (**not** installed by `uv sync`) through
`deploy/gcp/train.sky.yaml`; tracking is MLflow on Cloud Run, falling back to a local `mlflow.db`
when unreachable. Details in [`installation.md`](../installation.md) and
[`gcp_setup.md`](../gcp_setup.md).

## Checkpoint inventory

Every editor run whose loadable folder is in
`gs://$DVC_BUCKET/models/edit_flows/`. Trunk width and depth are read from each
folder's own `encoder/config.json`; the heads column is read from the launching `Makefile` target or
from the eval artefacts that name the arm. **None of these is a separately released model** — each
varies one thing against the reported arm, which is why they are a table here and not more cards.

| run tag | trunk (layers / hidden) | what it varies | evaluated in |
|---|---|---|---|
| **`faithful-appendixa`** | 12 / 480, stock | **the reported arm** | `metrics/disjoint/editor-*`, `metrics/aligned/`, `metrics/agree/`, `metrics/clock/` |
| `stock-linear` | 12 / 480, stock | arm D: stock trunk, `linear`/`fresh` heads | `metrics/dn/D-*`, `metrics/matched/D-*`, `metrics/sweep/arm-D-c40.json` |
| `heads-only` | 12 / 480, `esm2_oas` | arm C: OAS trunk, Appendix-A heads | `metrics/dn/C-*`, `metrics/matched/C-*`, `metrics/sweep/arm-C-c40.json` |
| (arm A, staged as `edit_flows_baseline`) | 12 / 480, `esm2_oas` | arm A: OAS trunk, `linear`/`fresh` — no loadable folder survives in that prefix | `metrics/dn/A-*`, `metrics/matched/A-*`, `metrics/generation_eval_A.json` |
| `sched-linear` | 12 / 480, stock | the linear half of the schedule A/B (job 76, A100) — a replicate of the reported arm | `metrics/schedule/` |
| `sched-cubic` | 12 / 480, stock | `kappa = t^3` (job 83, A100) | `metrics/schedule/` |
| `abl-eq23` | 12 / 480, `esm2_oas` | the deletion-supervision ablation, eq 23's indicator | — |
| `abl-fig13` | 12 / 480, `esm2_oas` | the same, Figure 13's two-sentinel delete branch | — |
| `size-8m` | 6 / 320, stock | trunk scale | — |
| `size-35m` | 12 / 480, stock | trunk scale | — |
| `size-150m` | 30 / 640, stock | trunk scale | — |
| `faithful-650m` | 33 / 1280, stock | trunk scale at Appendix-A heads | `metrics/lr/faithful-650m-*` |
| `lr650m-1e-5` | 33 / 1280, stock | the 650M learning-rate grid, 3,000 steps each | `metrics/lr/lr1e-5-*` |
| `lr650m-3e-5` | 33 / 1280, stock | " | `metrics/lr/lr650m-3e-5-*` |
| `lr650m-1e-4` | 33 / 1280, stock | " | `metrics/lr/lr650m-1e-4-*` |
| `lr650m-3e-4` | 33 / 1280, stock | " | `metrics/lr/lr650m-3e-4-*` |

The learning-rate grid has committed evaluation artefacts under `metrics/lr/`; the size sweep's
rows above say `—` because it has none. Neither has **a write-up in this repository**: no section of [`findings.md`](../findings.md) reports a
trunk-scale or learning-rate result. The reproduction ledger does write them up, and its reading is
worth having — but **it is behind the artefacts, in the direction that matters.** It reports the
650M arm at MMD 0.534 and the 3e-4 point at 0.509, both "n=1", and rests a caveat on the single
run; `metrics/lr/` now holds three seeds for each. Recomputed from the committed files, at the same
300 × 1 / holdout 1000 / clock 40 / k=3 config:

| arm | spectrum MMD ↓ | seeds |
|---|---|---|
| 650M, Appendix-A heads, lr 1e-4 (`faithful-650m`) | **0.519 ± 0.013** | 3 |
| 35M, Appendix-A heads (arm B, `metrics/matched/`) | **0.507 ± 0.034** | 3 |
| 650M at 3,000 steps, lr 1e-5 | 0.681 ± 0.009 | 2 |
| lr 3e-5 | 0.5505 | 1 |
| lr 1e-4 | 0.5885 | 1 |
| lr 3e-4 | **0.486 ± 0.021** | 3 |

**The 650M arm's own step count is not settled and this card does not assert one.** The ledger
labels it 20,000 steps in two tables and then argues from wall clock that it can only have been
3,000 — 8 h 40 m on an A100 at ~10.4 s/step, where 20,000 steps would have taken ~58 h — and no
committed artefact records it either way. `metrics/lr/faithful-650m-*.json` are evaluation cells and
carry no training metadata. Anyone quoting a 650M cost figure has to settle this first.

The ledger's conclusion survives and gets stronger, not weaker: **19× the parameters buys no
measurable gain** — 0.519 ± 0.013 against 0.507 ± 0.034 is now three seeds against three seeds
rather than one against three — and no learning rate in the searched range rescues the larger
trunk, though the best point sits on the grid boundary so no optimum is claimed. Read the JSONs,
not this card, for anything finer.

## More Information

**No contact with the original authors**, and no code or weights to compare against — EvoFlows and
Edit Flows both released neither. That is why `success` for this stage is defined as "the published
specification, executed, behaves as described" and never as numerical agreement.

A run finished, spent GPU hours and banked nothing loadable: job 21 wrote only `checkpoint.pt`,
the one format no evaluation can open. `restore_editor.py` and the sky YAML's end-of-job copy-out
exist because of it ([`findings.md`](../findings.md), "Regressions the test suite pins").

## Citation

The paper being reproduced:

```bibtex
@article{evoflows2026,
  title  = {EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering},
  author = {Deutschmann and Ferragu and Ziegler and Aziznejad and Bixby},
  year   = {2026},
  note   = {arXiv:2603.11703. ICLR 2026 "Foundation Models for Science" workshop}
}
@article{editflows2025,
  title  = {Edit Flows: Flow Matching with Edit Operations},
  author = {Havasi and Karrer and Gat and Chen},
  year   = {2025},
  note   = {arXiv:2506.09018. NeurIPS 2025}
}
```

Surnames only, because that is the whole of what this repository records — `CITATION.cff` says so
explicitly, and both author lists there trace to
[`evoflow_reproduction.md`](../evoflow_reproduction.md). Check the arXiv listings before publishing
either entry. For this reproduction, cite `CITATION.cff` at the repository root.

## Model Card Authors

Written from committed artefacts by the `editjumps` maintainers. Only the *developer* role of HF's
annotated template is staffed here; there is no sociotechnic review behind the
[Bias, Risks, and Limitations](#bias-risks-and-limitations) section, which is why its bias half is
narrow and says so.

## Model Card Contact

Via the project repository.
