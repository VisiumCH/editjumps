---
# Hugging Face dataset-card metadata, in the documented key shape. NOT published to the Hub: the
# artefact is DVC-tracked in a private bucket (docs/weights.md).
license: mit
pretty_name: EvoFlows seed homolog families, searched against OAS
size_categories:
  - 10K<n<100K
annotations_creators:
  - machine-generated
source_datasets:
  - extended|oas_corpus
tags:
  - biology
  - protein-sequences
  - antibody
  - homology-search
---

# Dataset Card for the seed homolog families (`data/interim/seed_families/`)

Three FASTA files, one per seed sequence from EvoFlows §4.2, holding the OAS corpus members that a
homology search from that seed returned. **Two are usable — Ty1 (38,687 members) and
trastuzumab VH (43,088). The third holds two sequences and nothing evaluates on it.**

These families are the *evaluation* substrate of the whole reproduction. Every §4.2 cell — the
editor's, both evotuned baselines', EvoDiff-MSA's, and the two
model-free baselines — draws its templates and its scoring reference from one of them, and the
four evotuned trunks are MLM-adapted to one each. This card is the single description of what they
are and what they are not.

**The measurement frame that turns these families into a table is not described here.** It has one
owner, [`metrics/disjoint/README.md`](../../../metrics/disjoint/README.md), and restating it would
be the duplication this card exists to remove.

## Table of Contents

- [Scope of Reproduction](#scope-of-reproduction)
- [The paper's six seeds, and why two survive](#the-papers-six-seeds-and-why-two-survive)
- [Consumers](#consumers)
- [Dataset Details](#dataset-details)
- [Uses](#uses)
- [Dataset Structure](#dataset-structure)
- [Dataset Creation](#dataset-creation)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [Citation](#citation)

## Scope of Reproduction

| decision | the paper | this build | status |
|---|---|---|---|
| seed sequences | six named proteins (Table 2, Appendix C) | three, all antibody, from the PDB: 6ZXN (Ty1, tags trimmed) and 1N8Z (trastuzumab, constant domains trimmed) | **deviation** — forced; see below |
| search sensitivity | E-value ≤ 10⁻¹ (§4.2) | `evalue: 0.1`, i.e. **the paper's value** | **match** |
| query coverage | ≥ 0.8 (§4.2) | `coverage: 0.8` | **match** |
| search method | iterative profile search | MMseqs2 single-pass at `sensitivity: 7.5` | **deviation** |
| identity floor to the seed | none — E-value and coverage only | **`min_identity: 0.7`** | **deviation**, and a necessary one |
| prefilter hits per query | not applicable | `max_seqs: 100000` | *not stated* |
| split into train / inference / holdout | stated, ratios and counts **never given anywhere in the paper** | 20 templates, 200 holdout, seed 0 | *not stated* |

**Why `min_identity: 0.7` exists, correctly.** The paper's two filters are selective against
UniRef30, which spans all of protein space, and **nearly vacuous against an antibody-only
database**, because every V domain shares the framework scaffold: at the paper's own E-value our
corpus returns 100,000+ members per seed against its 3,335. A 100× *looser* E-value moved the
counts by **0.00%**, which is what established that the E-value is not the binding filter here.
The 0.7 floor makes "family" mean a set of relatives rather than the whole database. `0.0` is the
paper's filter and is selectable.

**Two counts have been retracted and must not be requoted.** The family sizes **394 / 396 / 330**
were MMseqs2's default 300-prefilter-hits-per-query cap reporting a tool default rather than a
homolog count; the *uncapped, unfiltered* counts are 100,655 / 101,784 / 71,540. Neither is the
number below: **38,687 / 43,088 / 2** is what `max_seqs: 100000` and `min_identity: 0.7` return
together, and it is the tracked build. Any sentence of the form "our families came back 8–28×
smaller than Table 2" is describing the retracted 394/396 era and is **false of this artefact** —
Ty1's 38,687 is 11.6× *larger* than Table 2's 3,335.

**Their thresholds cannot be ported by copying the numbers.** That is a reproducibility finding
about the paper, not a gap in this build, and it partly vindicates the 50%-identity clustering used
for [the training pairs](oas-homolog-pairs.md).

## The paper's six seeds, and why two survive

Table 2's six seed proteins, verbatim, against what this repository can reach. **Four are
unreachable, and one of the four is unreachable for the paper too.**

| seed protein | homologs (Table 2) | length (Table 2) | reachable here | why not |
|---|---|---|---|---|
| Anti-SARS-CoV-2 VHH (Ty1) | 3,335 | 109.2 ± 4.3 | **yes** | — |
| Anti-HER2 scFv | 10,979 | 246.4 ± 4.8 | **VH only** | the scFv is VH + linker + VL; our search runs at `chain: heavy`. The VL family returns 2 members, so **there is no light-chain evaluation arm and there never was one** |
| Anti-EphA2 VHH | 24,304 | 118.2 ± 3.7 | **no** | **the sequence does not appear to exist in public data, and the paper mis-cites it** — see below |
| Serine-pyruvate aminotransferase | 6,565 | 363.0 ± 16.7 | no | not an antibody; an antibody-only corpus has no members for it |
| Chicken FGF2 | 1,435 | 141.6 ± 11.4 | no | same |
| Haloalkane dehalogenase DhaA | 4,697 | 282.2 ± 11.5 | no | same |

**The anti-EphA2 VHH is unrecoverable.** Appendix C cites Roovers et al. 2011, "Efficient
inhibition of EGFR signaling and of tumour growth by antagonistic anti-EGFR nanobodies" (Cancer
Immunol Immunother 60(8):1047–1058, doi:10.1007/s00262-011-1027-2) — an anti-**EGFR** paper with no
EphA2 VHH in it. The sequence is in none of the PDB (all 104 EphA2-crossreferenced entries are
conventional Fabs), SAbDab, PubMed, Europe PMC, PLAbDab-nano, or any US/EP/WO/JP patent sequence
deposit. The single hit anywhere, CN120230214, postdates the paper, is unconnected to its authors,
runs 132 aa against Table 2's 118.2 ± 3.7, and carries a synthetic non-natural 13-aa CDR1; it is
not adopted, because an unrelated sequence would manufacture agreement.

**So: two usable seed families, permanently, against the paper's six.** Three of the four losses
are a consequence of choosing an antibody-only corpus; the fourth is a limit of the paper. An
earlier reading of this repository's own artefacts attributed all four losses to the corpus; that
reading was withdrawn.

## Consumers

| what | card / owner | which family |
|---|---|---|
| the editor's §4.2 cells | [Edit Flows editor](../edit-flows-editor.md) | templates and scoring reference, drawn disjointly from one family |
| EvoDiff-MSA's §4.2 cells | [`metrics/disjoint/README.md`](../../../metrics/disjoint/README.md) | same partition; the family also supplies its MSA rows |
| `evotune-ty1`, `evotune-her2vh` | [evotuned ESM-2](../evotuned-esm2.md) | the family's **train** part |
| `evotune-dj-ty1`, `evotune-dj-her2vh` | same | the train part **restricted to members absent from the training pairs** |
| the two model-free baselines (random homolog pairing, random mutations) | [`metrics/disjoint/README.md`](../../../metrics/disjoint/README.md) | drawn inside each editor run |

## Dataset Details

### Dataset Description

- **Curated by:** `editjumps`, by the `seed_homologs` stage
- **Language(s):** not applicable — amino-acid sequences; **single V domains**, not joined
  `VH.VL`, which is the input-form shift documented on
  [the editor's card](../edit-flows-editor.md#factors)
- **License:** MIT for the code; the sequences are OAS's. **Not publicly obtainable**
  ([`weights.md`](../../weights.md))
- **Derived from:** [the OAS paired corpus](oas-corpus.md), searched from three seeds

### Dataset Sources

- **Repository:** `editjumps/pipeline/preprocess/pretrain/seed_homologs.py`; the splitter is
  `editjumps/core/family_split.py`
- **Stage:** `seed_homologs`; output `data/interim/seed_families`, md5
  `edb693c0378ef255c491ba1fe58a7cd2.dir`, 3 files, 12,854,218 B (`dvc.lock`)
- **Seeds:** `editjumps/pipeline/preprocess/pretrain/seeds/evoflows_seeds.fasta`, md5 `646352bf049f393cdd09cec58bf2f393`
- **Paper:** EvoFlows [arXiv:2603.11703](https://arxiv.org/abs/2603.11703) §4.2, Table 2,
  Appendix C

## Uses

### Direct Use

Supplying the templates a generator edits, the natural holdout every distributional metric is
scored against, and the real-homolog population behind the agreement ceiling; and as the adaptation
corpus for an evotuned trunk.

### Out-of-Scope Use

* **Not comparable to Table 2.** These are families over a different universe of sequences, built
  with a filter the paper does not use. Reading a member count against Table 2's compares two
  different definitions of "family".
* **Not a light-chain arm.** The VL file holds 2 members; `split_family` needs
  `n_templates + 2` distinct members and raises.
* **The default (non-`dj`) evotuned trunks must not be scored against a disjoint reference.** Their
  adaptation corpora hold 194 of Ty1's 200 disjoint reference sequences and 191 of HER2-VH's, plus
  19 of the 20 templates in each. `evotune_baseline` refuses, correctly.

## Dataset Structure

| family | members | distinct | min | median | max | absent from the training pairs |
|---|---|---|---|---|---|---|
| `Anti-SARS-CoV-2_VHH_Ty1.fasta` (a VHH) | **38,687** | 38,553 | 107 | **121** | 143 | **18,495** |
| `Anti-HER2_scFv_VH_trastuzumab.fasta` (a VH) | **43,088** | 42,839 | 101 | **122** | 132 | **21,015** |
| `Anti-HER2_scFv_VL_trastuzumab.fasta` | **2** | 2 | 107 | 107 | 107 | — |

Source for the length columns: [`findings.md`](../../findings.md), "What the evaluation families
are (measured)", which carries no absent-from-pairs figure — the last column comes from
`editjumps/pipeline/evaluate/generation_eval.py:171` and is over the **distinct** population, not
the member count beside it. None of the three families contains the `.` pair separator — these are lone domains.

**52.0% of distinct Ty1 members and 50.9% of HER2-VH members appear verbatim as the VH half of a
training line.** The sequences are half in-distribution; the *input form* is not, and the two
length supports do not overlap at all (0 of 3,320,210 training cells are ≤ 143 residues, 0 of
81,777 family members are ≥ 153). This is why the disjoint frame exists, and why
`sequences_in_pairs` indexes both the joined line and each chain: over the joined line alone the
leak reads 0%, and a 0% leak reads as reassurance rather than as a unit mismatch.

## Dataset Creation

### Curation Rationale

Table 2 is the one place the paper prints numbers this repository's data pipeline can be checked
against, so the search runs the paper's own two filters. It is also the place where the check
*failed informatively*: the same thresholds mean something different against a repertoire corpus,
and that is a finding about porting the paper rather than about this build.

### Source Data

#### Data Collection and Processing

`seed_homologs`: MMseqs2 search from each seed against
[`oas_corpus.txt.gz`](oas-corpus.md) at `--chain heavy`, `--evalue 0.1`, `--coverage 0.8`,
`--sensitivity 7.5`, `--max-seqs 100000`, `--min-identity 0.7`. Locked params:
`seed_homologs.chain: heavy`, `evalue: 0.1`, `coverage: 0.8`, `sensitivity: 7.5`,
`max_seqs: 100000`, `min_identity: 0.7`.

#### Who are the source data producers?

The corpus's, plus two PDB depositions for the seeds themselves (6ZXN, 1N8Z). See
[the corpus card](oas-corpus.md#who-are-the-source-data-producers).

### Annotations

**None.** A family is a set of sequences; membership is the only structure, and it comes from a
search rather than from a curator.

#### Personal and Sensitive Information

As [the corpus card](oas-corpus.md#personal-and-sensitive-information).

## Bias, Risks, and Limitations

* **Every per-family number in this reproduction rests on two families**, and one of them
  (HER2-VH) is the heavy half of a seed the paper evaluates as an scFv at twice the length.
* **`min_identity: 0.7` is ours and it defines what "family" means.** A looser floor gives a
  100,000-member family that teaches a model nothing about a lineage; a tighter one gives relatives
  so close the edits are trivial. Whether 0.7 is the right answer for antibodies — conserved
  frameworks, variable CDRs — is a question for a wet-lab reader.
* **HER2-VH is the noisier family to evaluate on**, on the metrics that depend on the covariance
  matrix. Over `metrics/matched/` (four arms × three seeds, one fixed config), the three-seed
  standard deviation of `covariance_frobenius_generated` is 0.014–0.046 on Ty1 against 0.189–0.216
  on HER2-VH. It does *not* hold everywhere — on spectrum MMD the two families are comparable
  (0.013–0.041 against 0.015–0.029) — so this is a property of the family and the metric together,
  not of any model. (An earlier revision quoted 0.020–0.045 against 0.113–0.179 without naming a
  metric; those exact ranges are not reproducible from any committed artefact.)
* **`metrics/seed_homologs.json` is declared `cache: false`, recorded in `dvc.lock`, and absent
  from the repository.** The counts above are read from [`findings.md`](../../findings.md), not
  from a committed metrics file.

### Recommendations

* **Cite this card for a family fact; cite
  [`metrics/disjoint/README.md`](../../../metrics/disjoint/README.md) for a frame fact.** The two
  are different axes and each has one owner.
* **Never requote 394 / 396 / 330.** They are a tool default, and they are still in circulation.

## Citation

EvoFlows, [arXiv:2603.11703](https://arxiv.org/abs/2603.11703), Table 2 and Appendix C. Note that
Appendix C's citation for the anti-EphA2 seed does not support it. For this build, cite
`CITATION.cff` at the repository root.

## Dataset Card Authors

Written from `dvc.lock`, `params.yaml` and [`findings.md`](../../findings.md) by the `editjumps`
maintainers.

## Dataset Card Contact

Via the project repository.
