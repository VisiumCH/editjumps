# Editor runs in the baselines' alignment

Every file here projects its per-position metrics onto `split.templates[0]`, the same coordinate
system `evotune_baseline` and `evodiff_msa_baseline` use, so a score here may sit beside a baseline's.
Runs under `metrics/agree/` predate that fix and are in the old `chosen[0][0]` frame; the two sets are
not comparable on covariance, MIP or positional JS. Each file records its own `alignment` field --
read it before putting two numbers in one table.

| file | job | family | holdout (reference n) | L | KL reading |
| --- | --- | --- | --- | --- | --- |
| `aligned-ty1.json` | 68 | Anti-SARS-CoV-2_VHH_Ty1 | 200 | 121 | pooled only |
| `aligned-her2vh.json` | 69 | Anti-HER2_scFv_VH_trastuzumab | 200 | 122 | pooled only |
| `h800-ty1.json` | 79 | Anti-SARS-CoV-2_VHH_Ty1 | 800 | 121 | pooled only |
| `h800-her2vh.json` | 78 | Anti-HER2_scFv_VH_trastuzumab | 800 | 122 | pooled only |
| `perseq-ty1.json` | local CPU re-run | Anti-SARS-CoV-2_VHH_Ty1 | 200 | 121 | **both** |
| `perseq-her2vh.json` | local CPU re-run | Anti-HER2_scFv_VH_trastuzumab | 200 | 122 | **both** |

**The KL column is the one to read before putting a KL number in a table.** `kl_generated_vs_natural`
sits in `defined_by_the_paper` and applies eq 23-24 to the POOLED amino-acid composition, which two
sets of close homologs share almost exactly — it reads ~12x below the paper's KL panel and is
comparable across our arms only. Eq 23-24 define the formula and never say what it is computed over,
so the pooled key is a reading, not the paper's number. `kl_divergence_positional` in
`our_interpretation` is the same formula per aligned column and is on their scale: Ty1 **0.0973**,
HER2-VH **0.0848**, inside the paper's 0.0059–0.1237. **But these two are in this directory's leaked
frame** (`train_overlap_113` / `train_overlap_103`), so quote the reported disjoint cells instead —
0.0713 and 0.0524, also inside that range (`metrics/disjoint/editor-*.json`). Only the `perseq-*`
pair here carries the key at all; files without it were not back-filled, because a number keeps the
provenance it was measured with.

`perseq-ty1.json` and `perseq-her2vh.json` are the SAME runs as `aligned-ty1.json` and
`aligned-her2vh.json` — same arm B weights under two folder aliases (`eval_faithful-appendixa`,
`eval_B_stock_appA`), same clock 40, seed 0, 20×20, same L, and every shared value agrees to ~1e-15
— re-emitted with the per-template and per-sequence lists, the baselines' per-position columns, and
now the positional KL. `editjumps consolidate`'s **legacy** table therefore sources the editor and
its two nested model-free baselines from the `perseq-*` pair; the reported table
(`consolidate.DISJOINT`) sources them from `metrics/disjoint/`, because these are `train_overlap_*`
rows. Each carries a sibling `.fasta` with all 1200
sequences (model + both baselines), so the positional KL can be recomputed without a GPU.

All six are arm B (`faithful-appendixa`, clock 40, `facebook/esm2_t12_35M_UR50D`, mlp rate head,
`esm_lm_head` Q head) at 20 templates x 20 variants. The `h800-*` pair differs from the `aligned-*`
pair in **one** input, `--holdout-size`, so the difference between them is the reference size and
nothing else: the generated sets are bit-identical (`levenshtein_to_template`,
`pairwise_levenshtein_pooled`, `entropy_delta` and `profile_log_likelihood` all match to every
recorded digit).

**What the reference size does to the numbers is in `docs/findings.md`**, "Normalising to the ceiling
does NOT cancel the reference size". Short version: between n=200 and n=800 the two-family mean
score/ceiling **ratio** falls 3.5 pp on covariance and rises 1.1 pp on MIP (the individual cells move
further and not always together), so neither the score nor the ratio is comparable across reference
sizes. `agreement_reference_n` and `ceiling_n` are in every file for exactly this reason.

`spectrum_mmd` is comparable **only** between runs whose `mmd_diagnostic.n_reference` matches, which
is 200 for the `aligned-*` pair and 800 for the `h800-*` pair. They are not comparable to each other.
