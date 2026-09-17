# Preprocessing

Data preparation for the reproduction: unlabelled antibody sequence corpora, and the homolog pairs
built from them. This repository reproduces unconditional EvoFlows generation, so there is no
labelled half — nothing here builds or reads a property label table.

## `pretrain/` — unlabelled data (antibody sequences, no property labels)

Downloads OAS antibody repertoires and builds an unlabelled sequence corpus
(`data/pretrain/oas_corpus.txt.gz`) to **continue-pretrain** ESM-2 (MLM) before
fine-tuning. OAS carries **no** property labels — it is purely a sequence corpus.

```bash
# per-chain (single-chain ESM-2 MLM):
uv run python -m editjumps.pipeline.preprocess.pretrain.download_oas --collection paired
# joined VH.VL per antibody (whole-Fv / paired models):
uv run python -m editjumps.pipeline.preprocess.pretrain.download_oas --collection paired --pair-chains
```

The corpus then feeds `editjumps/pipeline/train/pretrain_esm.py`. `homolog_pairs` also lives here: it
clusters the corpus into homolog families and samples pairs from them. Pairs are **unoriented**
(`params.yaml`'s `homolog_pairs.direction: none`), which is the EvoFlows-faithful setting and what
every recorded run used — the coupling in eq 8 is symmetric, so there is no better member to point
`x1` at.
