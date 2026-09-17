---
# Hugging Face dataset-card metadata, in the documented key shape. NOT published to the Hub: the
# artefact is DVC-tracked in a private bucket (docs/weights.md). The block is here so the card is
# machine-readable in the same way a Hub card is.
license: mit
pretty_name: OAS paired antibody corpus (clustered split)
size_categories:
  - 1M<n<10M
annotations_creators:
  - no-annotation
task_categories:
  - fill-mask
tags:
  - biology
  - protein-sequences
  - antibody
  - immune-repertoire
configs:
  - config_name: train
    data_files: data/pretrain/oas_corpus.train.txt.gz
  - config_name: validation
    data_files: data/pretrain/oas_corpus.val.txt.gz
---

# Dataset Card for the OAS paired corpus (`oas_corpus`)

Observed Antibody Space's `paired` collection, crawled, length-filtered, joined into one
`VH.VL` string per antibody, and split 95 / 5 by CDR cluster. **2,586,927 lines.** It is the root of
every sequence artefact in the repository: the editor's training pairs, §4.1's synthetic pairs, the two evaluation families and the
OAS-adapted trunk are all built from it.

This card exists so that those four consumers cite one description instead of restating it four
times — the failure this repository has already been bitten by, where six committed artefacts
carried a caveat the code had since fixed. **If a number about the corpus is wrong here, it is
wrong once.** Every card that needs a corpus fact links to this one and states only what is
specific to its own use.

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

Which of EvoFlows' stated data decisions this corpus matches, knowingly differs from, or is free to
choose because the paper does not say. The rows are the reproduction ledger's §02, checked against
`dvc.lock` and `params.yaml`; `editjumps/core/provenance.py` is the authority on the claim.

| decision | the paper (arXiv 2603.11703) | this corpus | status |
|---|---|---|---|
| sequence database | UniRef30 + ColabFold's environmental database; OAS named only as a queryable source (§3.1, §4.2) | OAS `paired` collection only | **deviation** |
| scope of the training set | 6 seed proteins, each with a homolog set, split train / inference / holdout (§4.2, Table 2) | corpus-wide, 2,586,927 lines | **deviation** — six seed families against a corpus is a different experiment, and no setting closes it |
| chain handling | VHH and scFv — single chain, nothing to join (§4.2, Table 2) | VH and VL joined with `.` | **deviation** |
| length filter | not stated; observed lengths 109–363 residues (Table 2) | 70–200 **per chain**, so joined strings run roughly twice that | *not stated* |

**Antibody-only by design.** The paper's seeds span enzymes, a growth factor and antibodies, plus
an environmental database this project does not use. Three of its six seeds have no counterpart
here at all — see
[the seed-families card](seed-homolog-families.md#the-papers-six-seeds-and-why-two-survive).

**The separator is a vocabulary fact, not a style choice.** `.` is a real token in ESM-2's
33-token vocabulary; `:` tokenises to `<unk>`, which would collapse every VH/VL boundary onto one
token. This repository hit that once.

## Consumers

A change to this corpus invalidates all of these. That is the reason this card is a single file.

| stage | artefact it produces | card |
|---|---|---|
| `pretrain_esm` | `data/pretrain/esm2_oas` | [`esm2_oas` trunk](../esm2-oas-trunk.md) |
| `build_homolog_pairs` | `data/pretrain/oas_homolog_pairs.tsv.gz` | [OAS homolog pairs](oas-homolog-pairs.md) |
| `build_deterministic_pairs` | `data/pretrain/deterministic_pairs.tsv.gz` | [deterministic (§4.1) editor](../deterministic-editor.md#training-data) |
| `seed_homologs` | `data/interim/seed_families/` | [seed homolog families](seed-homolog-families.md) |

## Dataset Details

### Dataset Description

- **Curated by:** `editjumps`, from the Observed Antibody Space release the `download_oas`
  stage crawled
- **Language(s):** not applicable — amino-acid sequences, one antibody per line
- **License:** MIT for this repository's code. OAS's own terms govern the sequences; this project
  redistributes none of them, and the artefact is private
  ([`weights.md`](../../weights.md))
- **Not publicly obtainable.** DVC-tracked in `gs://$DVC_BUCKET/`; needs
  credentials against the `$PROJECT_ID` organisation

### Dataset Sources

- **Repository:** `editjumps/pipeline/preprocess/pretrain/download_oas.py`, `split_corpus.py`;
  clustering in `editjumps/core/cluster_split.py`
- **Stages:** `download_oas` → `split_corpus` in `dvc.yaml`
- **Upstream:** Observed Antibody Space (Olsen, Boyles & Deane 2022) — take the citation from OAS
  itself; this repository records no author list for it

## Uses

### Direct Use

Continued masked-language pretraining of an ESM-2 trunk; as the search database for
`seed_homologs`; and as the population that `build_homolog_pairs` clusters into families.

### Out-of-Scope Use

* **Not a general protein corpus.** Every line is an antibody variable-domain pair. Nothing here
  has ever seen an enzyme, and three of the paper's six seed proteins therefore cannot be searched
  against it.
* **Not a source of lone V domains.** Every line is joined, minimum length 153 residues. The
  downstream train/test shift this creates is measured on
  [the editor's card](../edit-flows-editor.md#factors): **0 of 3,320,210** training cells are ≤ 143
  residues and **0 of 81,777** evaluation family members are ≥ 153 — the two supports do not
  overlap at all.
* **The committed artefact carries a decontamination step this repository cannot repeat.** The
  files under DVC were built when the DAG still ran a `decontaminate_corpus` stage, which screened
  OAS against a separate property-prediction project's labelled molecules. Those labels are not
  here, so neither is the stage: `split_corpus` now reads `oas_corpus.txt.gz` directly. The gap is
  370 lines of 2,587,297 (0.014%), and it is why a fresh `dvc repro` will not reproduce the md5s
  below. Nothing the paper holds out was ever in that exclusion set.

## Dataset Structure

One gzipped text file per split, one sequence per line, no header.

| | value | source |
|---|---|---|
| raw data units crawled | 611 | `dvc.lock` `download_oas` |
| `oas_corpus.txt.gz` | 100,352,972 B, md5 `1c46d10867fb77a84b62bd0e232fe7af` | same |
| lines crawled | 2,587,297 | the removed `decontaminate_corpus` stage's lock entry; see the note above |
| dropped by the removed decontamination stage | **370** (0.014%) | same |
| **corpus every model here was trained on** | **2,586,927** lines, md5 `b23db3c267d61fc8a30e60aa4f11e28f` | the historical `decontaminate_corpus` lock entry |
| unique CDR-H3+L3 keys | 2,461,250 | [`findings.md`](../../findings.md) |
| `oas_corpus.train.txt.gz` | 94,867,624 B, md5 `dd74f58f159f395c2aaf3a0ee70005b4` | `dvc.lock` `split_corpus` (recorded under the old `.clean.` name) |
| `oas_corpus.val.txt.gz` | 4,722,905 B, md5 `016a89c03eb00184f80b82354c245619` | same |
| joined length: min / median / p99 / max | 153 / **232** / 243 / 298 residues | [`findings.md`](../../findings.md), "What the training corpus actually is" |
| per-chain length: min / median / max | 70 / **114** / 164 | same |

## Dataset Creation

### Curation Rationale

The reproduction needs a homolog population, and the paper's is not reachable: its families are
defined by iterative profile search against UniRef30 plus an environmental database. An
antibody-only corpus is a deliberate substitution — it makes the families buildable and it makes
a downstream question askable — and it is filed as a deviation rather than as
a reproduction step.

### Source Data

#### Data Collection and Processing

1. **`download_oas`** — crawl the `paired` collection, keep chains of 70–200 residues, emit
   `VH.VL` per antibody plus a CDR-H3/L3 key file. Locked params: `oas.collection: paired`,
   `min_len: 70`, `max_len: 200`, `pair_chains: true`, `pair_sep: "."`, `emit_cdr_keys: true`.
2. **`split_corpus`** — 95 / 5 by line count, but **clustered**: MMseqs2 `easy-cluster` at
   `min_seq_id: 0.9`, `coverage: 0.8` over the CDR keys, so near-duplicates cannot straddle the
   split. Locked params: `split.val_frac: 0.05`, `min_seq_id: 0.9`, `coverage: 0.8`,
   `mmseqs_mode: easy-cluster`.

#### Who are the source data producers?

Human and mouse B-cell repertoire sequencing studies deposited in OAS by their original authors.
This repository does not record which studies the 611 units came from.

### Annotations

**None.** There is no annotation layer: the corpus is unlabelled sequence, and this repository has
no label table of any kind — see [`docs/claims.md`](../../claims.md), "Why there is no target
property".

#### Personal and Sensitive Information

OAS entries derive from human donors and carry no donor identifiers into this artefact. Nothing
here is a record about a person; the demographic reading of "bias" does not apply, which is why the
section below is narrow and says so.

## Bias, Risks, and Limitations

* **Antibody-only and paired-chain-only**, so nothing built on it can be evidence about the
  paper's non-antibody seeds.
* **Three decontamination counts are in circulation, and only one describes this artefact.** A
  threshold sweep over 2.59M sequences reported ≥0.90 dropping 8,151 lines (0.315%) — the same
  8,151 [`known_issues.md`](../../known_issues.md) attributes to the 8G MMseqs memory cap, because
  the cap and the threshold acted on one run and neither figure is the other's independent
  confirmation. A third count, from a larger exclusion set, has no surviving source. Only
  **370 (0.014%)** is the tracked build's number, and it is the only one that describes the corpus
  every model here was trained on. None of them is reproducible from this repository, whose DAG
  carries no decontamination stage at all.
* **The validation split answers "did it converge", not "does it generalise".** Clustering at 0.9
  identity over CDR keys stops near-duplicates straddling the split; it does not make the two sides
  homology-disjoint.
* **Truncation binds on the tail.** `pretrain.max_length: 280` against a maximum joined length of
  298.

### Recommendations

* **Quote the tracked md5, not the params values.** `params.yaml` is what a fresh `dvc repro` would
  run; `dvc.lock` is what exists. They have already diverged for the trunk built on this corpus
  ([`esm2_oas` trunk](../esm2-oas-trunk.md#training-hyperparameters)).
* **`metrics/seed_homologs.json` is declared `cache: false` in `dvc.yaml`, recorded in `dvc.lock`,
  and absent from the repository.** The family statistics several cards quote therefore rest on
  [`findings.md`](../../findings.md) rather than on a committed metrics file.

## Citation

For OAS, take the citation from OAS. For this corpus and the reproduction built on it, cite
`CITATION.cff` at the repository root.

## Dataset Card Authors

Written from `dvc.lock`, `params.yaml`, `dvc.yaml` and [`findings.md`](../../findings.md) by the
`editjumps` maintainers.

## Dataset Card Contact

Via the project repository.
