# Agreement Evaluation Runs

`agree2-ty1.json` and `agree2-her2vh.json` contain agreement evaluations for the headline editor (arm B, `faithful-appendixa`, clock 40) using 20 templates × 20 variants and holdout size 200 per family.

Every run records `agreement_reference_n` (the reference set size used for covariance and MIP evaluation). Comparison across runs requires matching reference set sizes, as agreement metrics vary systematically with reference size (see [`docs/findings.md`](../../docs/findings.md)).

## The holdout-800 pair, in this same old alignment

`h800-oldalign-ty1.json` and `h800-oldalign-her2vh.json` are sky jobs 74 and 75: the same arm B editor
as `agree2-*`, same 20x20, `--holdout-size 800` instead of 200, and the same pre-fix `chosen[0][0]`
alignment (L=115 Ty1, L=127 HER2-VH). They are kept for one purpose: they replicate, in a second
coordinate system and with a second pair of runs, the direction each agreement moves when the
reference grows -- covariance down (-0.0227 Ty1, -0.0115 HER2-VH), MIP up (+0.0335, +0.0516). The
`metrics/aligned/` pair shows the same signs at L=121/122, so the effect is not an alignment artefact.

**The `h800-oldalign-*` pair carries ceilings** — `covariance_agreement_ceiling` 0.9869 (Ty1) /
0.9798 (HER2-VH), `mip_agreement_ceiling` 0.9546 / 0.9545, both at `ceiling_n: 300`. The `agree2-*`
pair carries none of the three keys at all: absent, not null, its only agreement fields being
`covariance_agreement`, `mip_agreement` and `agreement_reference_n`. What argues for using
`metrics/aligned/` for a score/ceiling cell is the FRAME, not a missing ceiling: these ran on the pre-fix `chosen[0][0]` alignment, so their
ceilings are not comparable with anything scored at L=121/122.
