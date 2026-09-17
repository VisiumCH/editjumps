# Evotuned-PLM baseline runs, one file per family

§4.2's two evotuned baselines are reported as a mean over two families, but `params.yaml`'s
`evotune.family` names ONE — and all three stages (`evotune_esm`, `evotune_baseline`,
`evotune_baseline_forced`) write to fixed paths (`metrics/evotune_esm.json`,
`metrics/evotune_baseline.json`, `metrics/evotune_baseline_forced.json`). So a second family's run
overwrites the first's numbers, in the repo and in
`gs://$DVC_BUCKET/repro-artifacts/` alike. Hence this directory: the same three
files per family, under a family prefix, copied out of the bucket before the next run lands.

| prefix | family | sky job | GPU |
| --- | --- | --- | --- |
| `ty1-` | `Anti-SARS-CoV-2_VHH_Ty1.fasta` | 67 (`profound-repro-evotune2`) | A100:1 |
| `her2vh-` | `Anti-HER2_scFv_VH_trastuzumab.fasta` | 70 (`profound-repro`) | A100:1 |

Each file's own `family` field is the authority on which family it is; the prefix is a filename, the
field is data. Only the prefixed copies are kept here: unprefixed duplicates of the Ty1 set were
committed alongside them for a while, byte-identical and referenced by nothing, which is exactly the
overwrite hazard this directory exists to avoid.

Both runs used the same §4.2 partition (`n_templates: 20`, `holdout_size: 200`, `seed: 0`,
`n_variants: 20`, `mutations: 4`, `budget_mode: realised`) and the same 2,000-step evotuning budget,
which is what makes the two families' numbers averageable and each family's two baselines comparable
to each other.

Read `alignment` before comparing anything positional across files. Each run projects onto ITS OWN
`split.templates[0]`: L=121 for Ty1, L=122 for HER2-VH. `covariance_agreement`, `mip_agreement` and
`js_divergence_positional` therefore live in a different frame per family — comparable between the
two baselines of one family, only approximately so against an editor run, which projects onto
whichever template its own permutation puts first (L=115 and L=127 in `metrics/agree/agree2-*.json`).

`agreement_reference_n` is 200 in all four baseline files (two families x forced/unforced). Two runs at different reference sizes are
not comparable, by a margin wider than any model difference measured here — see `metrics/agree/README.md`.
