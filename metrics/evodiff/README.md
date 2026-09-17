# EvoDiff-MSA baseline runs

`her2vh-evodiff_msa_baseline.json` is sky job 82: §4.2's EvoDiff-MSA baseline on the
Anti-HER2 VH family, 20 templates x 20 variants, holdout 200, alignment L=122.

The Ty1 run is NOT here. It lives at `metrics/evodiff_msa_baseline.json`, the path
`dvc.yaml` declares as the stage's output and the one the tracked lock entry hashes.
Only the second family needs a prefix, because only one of the two can occupy the
stage's declared output path.

Both carry `covariance_agreement`, `mip_agreement`, their in-run ceilings,
`agreement_reference_n`, `pairwise_levenshtein_pooled` and `js_divergence_positional`.
An earlier Ty1 artefact predates those keys; if a file lacks `agreement_reference_n`
it is that older one and its per-position columns cannot be compared with anything.
