---
# Hugging Face dataset-card metadata, in the documented key shape. NOT published to the Hub: the
# artefact is DVC-tracked in a private bucket (docs/weights.md).
license: mit
pretty_name: OAS homolog pairs (symmetric, cap 20 per family)
size_categories:
  - 1M<n<10M
annotations_creators:
  - machine-generated
source_datasets:
  - extended|oas_corpus
tags:
  - biology
  - protein-sequences
  - antibody
  - sequence-editing
  - flow-matching
---

# Dataset Card for the OAS homolog pairs (`oas_homolog_pairs.tsv.gz`)

**1,660,105 unordered pairs** of antibody sequences drawn from the same MMseqs2 homolog family, two
tab-separated joined `VH.VL` strings per row. It is EvoFlows §4.2's `(x0, x1)` coupling, built over
[the OAS corpus](oas-corpus.md) instead of over the paper's six seed families, and it is the
training data for **every trained editor in this repository**.

This card is where the pair-construction facts live. Eight models cite it; none of them restates
it. Two things it carries that no per-model card should own:

* **the family and pair construction as a match / deviation / not-stated table** — the reproduction
  ledger's §03 and §04, which is what makes an editor number a reproduction claim rather than a
  measurement of an arbitrary corpus;
* **the mean edit distance between a pair's two members**, which is not a detail: it is what sets
  the edit count of any method without a clock, and it is the single fact that decides whether
  the §4.2 results table reads correctly.

## Table of Contents

- [Scope of Reproduction](#scope-of-reproduction)
- [Consumers](#consumers)
- [Dataset Details](#dataset-details)
- [Uses](#uses)
- [Dataset Structure](#dataset-structure)
- [Dataset Creation](#dataset-creation)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [Citation](#citation)

## Scope of Reproduction

### Family construction (ledger §03)

The section the ledger most wants checked, "because it defines what the model thinks evolution
looks like".

| decision | the paper | this build | status |
|---|---|---|---|
| search method | iterative profile search vs UniRef30, with expansion and realignment (§4.2) | MMseqs2 `easy-cluster` — flat clustering, one pass, no profiles | **deviation** |
| sensitivity threshold | E-value ≤ 10⁻¹ (§4.2) | ≥50% sequence identity (`homolog_pairs.min_seq_id: 0.5`) | **deviation** — an E-value threshold and a percent-identity floor are not comparable quantities, and neither converts into the other |
| coverage filter | query coverage ≥ 0.8 (§4.2) | `coverage: 0.8` | **match** — the one filter matched exactly, and the least likely to matter alone |
| extra depth | ColabFold environmental database, in addition (§4.2) | none | **deviation** |
| family distribution | uniform over the family, π(x,x′) = p(x)p(x′) (eq 7–8) | uniform within family, drawn without orientation | **match** |

Profile search reaches remote homologs where clustering reaches close ones, so **these families are
shallower than the paper's and the model sees smaller edits.**

### Pair construction (ledger §04)

| decision | the paper | this build | status |
|---|---|---|---|
| pairs per family | all unordered pairs enumerated (§4.2) | `max_pairs_per_family: 20` → 1,660,105 pairs | **deviation** — and it biases the edit distribution, measured below |
| pair orientation | unordered; direction carries no meaning (eq 8) | `direction: none`, the only value this build accepts | **match** |
| alignment | Needleman–Wunsch global alignment → `(z0, z1)` in the ε-extended alphabet (§3.2) | Needleman–Wunsch, ε-extended, **unit** gap and mismatch cost | **match** on the algorithm; the *scoring* is unstated and is a separate deviation, on [the editor's card](../edit-flows-editor.md#scope-of-reproduction) |
| held-out split | train / inference / holdout per homolog set (§4.2), ratios never given | 5% validation, split **by family** | *not stated* |

**"Unordered" is a citation, not an inference.** §4.2 trains on "all unordered sequence pairs" of a
homolog family, which also settles a question it was never asked: **one training example per pair,
not two.** (The 2 × 1,660,105 = 3,320,210 figure some cards quote counts *sequences* fed to the
model, not examples.)

**The cap is engineering, not oversight.** The paper's largest family (24,304 homologs) alone yields
~295M unordered pairs; ours would yield 15,059,116 from one family. Whether equalising lineages or
weighting them by clonal expansion is biologically right is an open question for a wet-lab reader.

## Consumers

Every model trained on sequence pairs in this repository. A rebuild invalidates all of them.

| model | card | how it reads the file |
|---|---|---|
| Edit Flows editor, all four 2×2 arms plus the 650M, schedule and LR arms | [edit-flows-editor](../edit-flows-editor.md) | joined `VH.VL`, both columns |
| the four evotuned trunks | [evotuned-esm2](../evotuned-esm2.md) | **not as training data** — as the *exclusion* set that defines the disjoint population |
| every §4.2 evaluation cell | [`metrics/disjoint/README.md`](../../../metrics/disjoint/README.md) | same, via `--disjoint-from-pairs` |

The last two rows are why this file is load-bearing for evaluation and not only for training: the
disjoint frame is *defined* as the family members absent from it.

## Dataset Details

### Dataset Description

- **Curated by:** `editjumps`, built by the `build_homolog_pairs` stage
- **Language(s):** not applicable — amino-acid sequences over ESM-2's 33-token vocabulary
- **License:** MIT for the code; the sequences are OAS's, and this artefact is private
  ([`weights.md`](../../weights.md))
- **Derived from:** [the OAS paired corpus](oas-corpus.md)

### Dataset Sources

- **Repository:** `editjumps/pipeline/preprocess/pretrain/homolog_pairs.py`, clustering in
  `editjumps/core/cluster_split.py`
- **Stage:** `build_homolog_pairs`
- **Paper:** EvoFlows [arXiv:2603.11703](https://arxiv.org/abs/2603.11703) §4.2

## Uses

### Direct Use

Supervising a discrete edit-flow model to carry one natural antibody to a relative of it. Per pair, per step: align, draw `t`, sample the mixture path, read the per-column
supervision off the alignment.

### Out-of-Scope Use

* **Not a benchmark.** There is no ground-truth edit script per row and no held-out answer; the
  only evaluation in this repository with a known answer is
  [§4.1's synthetic set](../deterministic-editor.md).
* **Not a source of small edits.** See the distance table below: **0.459%** of pairs are within 5
  edits joined — 917 of 200,000 — and 0.816% (1,631) on the heavy halves, which is what
  `edit_flows.chain: heavy` trains on. A method whose edit count is set by its training data will
  not land on §4.2's ~5-edit budget, and filtering the file cannot fix it: neither count is a
  training set.
* **The oriented build is a different dataset, and is not reproducible here.** `direction:
  improving` kept 1,403,255 of the pairs; it orients each pair toward the better member of a target
  property, which needs a property registry that is not part of this repository. No number in any
  card comes from it, and `homolog_pairs.py` refuses the setting with that reason.

## Dataset Structure

| | value | source |
|---|---|---|
| file | `data/pretrain/oas_homolog_pairs.tsv.gz`, 64,987,998 B, md5 `4cb9f191144bd20e3615e40d17ea128e` | `dvc.lock` `build_homolog_pairs` |
| row format | `x0<TAB>x1`, both joined `VH.VL` | — |
| families | 352,605, of which **178,940** have ≥2 members | [`findings.md`](../../findings.md), "Homolog pairs" |
| **pairs** | **1,660,105** | same |
| oriented build, for reference | 1,403,255 kept | same |
| joined length: min / median / max | 153 / 232 / 298 | [`findings.md`](../../findings.md) |

### Edit distance between a pair's two members

**Read this table before reading any distance column in any results table.** Two measurements of
the same quantity are in circulation and they disagree; both are load-bearing somewhere, so both
are here with their provenance.

| measurement | mean | median | over |
|---|---|---|---|
| joined `VH.VL` | **70.89** | 71 | a 3,000-pair sample at seed 0 of the 200,000-pair head of the file. **This card is the only record of it** — it is not in `findings.md` |
| heavy half only (`--chain heavy`) | **48.77** | 55 | the same sample; 39.7% of a 122.9-residue mean length |
| joined `VH.VL`, cap 20 | **60.76** | 63 | the **full** 1,709,573-pair rebuild of the cap-bias study ([`findings.md`](../../findings.md), "The `max_pairs_per_family: 20` cap does bias the edit distribution"), which names its own script |

The 10-edit gap between 70.89 and 60.76 is **not** explained by the rebuild's own ~3% reproduction
error: it is a sample-versus-population difference. 70.89 comes from the first 200,000 rows, which
are not a random draw from a file written family by family; 60.76 is the whole distribution. Where
a card needs "what the models were fit on", 70.89 and 48.77 are the right figures: every run
loaded exactly those first 200,000 rows. Where a card needs "how the cap moved the distribution",
60.76 against 65.42 is the right pair. **Neither number substitutes for the other,
and this is the only place in the repository that says so.**

Exact counts, over all 200,000 rather than sampled: pairs within 5 edits are **1,631 of 200,000
(0.816%)** on the heavy halves and 917 (0.459%) joined. `E[l1 − l0] = −0.23` and
`E[|l1 − l0|] = 3.52`, so of ~49 edits in a heavy-chain pair **at most ~3.5 can be indels** —
substitutions outnumber length changes 14×.

### The cap does bias the distribution — measured

Full enumeration is **255,548,117** pairs, so cap 20 keeps 0.67% of them. The uncapped column is a
Horvitz–Thompson estimate over pairs drawn without replacement per family (effective sample size
93,014, standard error 0.08 edits), not a build.

| | cap 20 — shipped | cap 200 | uncapped (est.) |
|---|---|---|---|
| pairs | 1,709,573 | 7,564,699 | 255,548,117 |
| mean edit distance | **60.76** | 66.04 | **65.42** |
| p10 | 21 | 28 | 30 |
| substitutions | 86.26% | 85.67% | **84.23%** |
| Wasserstein-1 to uncapped | **4.661** | 3.00 | 0 |

Capped pairs are **7.1% closer together** than the pairs §4.2 would have used, and the whole
distribution moves, not only its mean: Wasserstein-1 equals the gap in means to three decimals,
which happens only if the two CDFs never cross, so capped pairs are uniformly stochastically
closer. **Indels are under-represented by 12.9% relative** — the part most likely to matter, since
the edit-flow loss is defined over alignment operations and §4.1 scores insertion precision and
recall directly.

The mechanism is the opposite of the hypothesis it was built to test: divergence *rises* with
family size (44.4 edits at k=2 to 72.0 at k=101–500), and the cap moves 91% of the pair mass off
the 3,106 largest families onto small ones. Subsampling would have hidden it — the same script over
200,000 corpus lines reads the gap as 1.2%, not 7.1%.

Caveat carried by every number in this sub-section: the rebuild reproduces the tracked build only
to ~3% — 184,102 multi-member families against 178,940 (2.9%) and 1,709,573 pairs against 1,660,105
(3.0%) — because MMseqs2's cascaded clustering is not bit-reproducible across binary versions. (The
*total* family count is the one quantity that does not reproduce to 3%: 396,842 against 352,605, or
12.55%. Singleton families are where the difference lands, and they emit no pairs.)

## Dataset Creation

### Curation Rationale

§4.2 needs unordered pairs of homologs. The paper's families come from profile search against a
universe of sequences this project does not have, so families are built by clustering the OAS
corpus instead. `max_pairs_per_family` exists because full enumeration is 255M pairs; nothing was
changed once the bias was measured, because `max_pairs_per_family` is a tracked DVC param and which
bias is the right one is a question for a wet-lab reader, not a bug.

### Source Data

#### Data Collection and Processing

`build_homolog_pairs` over [the clean corpus](oas-corpus.md): MMseqs2 `easy-cluster` at
`min_seq_id: 0.5`, `coverage: 0.8` over the CDR-H3+L3 keys, group into families, emit up to 20
unordered pairs per family. Locked params: `homolog_pairs.min_seq_id: 0.5`, `coverage: 0.8`,
`mmseqs_mode: easy-cluster`, `max_pairs_per_family: 20`, `max_lines: 0`, `direction: none`.

The trainer logs the pair count and construct it actually loaded rather than trusting config. It
does **not** verify symmetry: that check measured whether `x1` was the better member for a target
property, and with no property to orient toward it is skipped rather than faked
(`train_edit_flows.py`). The historical measurement, from the build that had one, was 44.9% of
pairs with `x1` better against ~100% under an oriented construction.

#### Who are the source data producers?

See [the corpus card](oas-corpus.md#who-are-the-source-data-producers).

### Annotations

**None**, and that is the point of `direction: none`: a pair carries no label and no direction. The
oriented build added one, from a closed-form descriptor rather than from a measurement, and is not
part of this repository.

#### Personal and Sensitive Information

As [the corpus card](oas-corpus.md#personal-and-sensitive-information).

## Bias, Risks, and Limitations

* **The cap bias is real, one-signed and 7% on the mean** (above). Against 20,000 steps × batch 16
  ≈ 320,000 pairs consumed, cap 20 already supplies 5× more pairs than a run reads, so the cap
  costs distribution fidelity rather than data.
* **The 5% validation split is by family but not homology-disjoint.** A connected component over
  pairs-as-edges cannot merge two families, but it can split one, so `val_loss` is an early-stopping
  and divergence signal and **not** a generalisation estimate.
* **The pairs overlap the evaluation families, and that leaked once.** 52.0% of distinct Ty1 members
  and 50.9% of HER2-VH members appear verbatim as the VH half of a training line. Before
  `--disjoint-from-pairs`, 113 of 200 Ty1 reference sequences and 11 of 20 templates had been in
  the editor's training pairs while the baselines' own assertions refused the same overlap. Read
  [`metrics/disjoint/README.md`](../../../metrics/disjoint/README.md) before comparing any cell to
  any other.
* **The edit distance is a property of the corpus and it sets the edit count of any method without a
  clock.** Some arms realise far more edits per sequence than the editor's 4.58 for
  this reason and not because they over-edit.

### Recommendations

* **Never read a distance column without its row's own realised edit count.**
* **Quote `60.76` or `70.89` with the measurement that produced it**, always. They are not
  interchangeable.

## Citation

EvoFlows, [arXiv:2603.11703](https://arxiv.org/abs/2603.11703) §4.2 for the coupling; Edit Flows,
[arXiv:2506.09018](https://arxiv.org/abs/2506.09018) §3.2 for the ε-extended alignment. For this
build, cite `CITATION.cff` at the repository root.

## Dataset Card Authors

Written from `dvc.lock`, `params.yaml` and [`findings.md`](../../findings.md) by the `editjumps`
maintainers.

## Dataset Card Contact

Via the project repository.
