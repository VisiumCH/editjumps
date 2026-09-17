# Clock sweep — one alignment frame per seed

`arm-B` (`eval_B_stock_appA`, `rate_head=mlp`, `q_head=esm_lm_head`) over clock
{25, 40, 60, 80, 100, 120} x seed {0, 1, 2} x family {Ty1 VHH, HER2 scFv VH}.
Regenerate any missing cell with `editjumps generation-eval --clock C --seed S
--family-fasta F --n-templates 10 --n-variants 10 --n-steps 50 --holdout-size 200`.

## Read these PAIRED WITHIN SEED, never averaged across seeds

The alignment length comes from the seeded split — `split.templates[0]` — so it differs by seed and
by family (Ty1: **L=121, 118, 124**; HER2-VH: **L=122, 120, 128**, for seeds 0, 1, 2) and is
*constant for a given seed across every clock*. That makes the within-seed curve a single-frame
comparison and the between-seed spread a nuisance.

Averaging across seeds puts a **0.18–0.86** spread on `spectrum_mmd`, which is larger than the
effect and buries it. Per family, because the two are not alike: Ty1 is 0.18–0.30 and narrows as the
clock rises, HER2-VH is 0.50–0.86 and widens. A single "~0.3" describes Ty1 only. Paired within seed, an interior minimum at **clock 80** appears in **5 of the 6
family-seed curves**, monotone on both flanks. The exception is **HER2-VH seed 2**, which falls
monotonically across the whole sweep to its minimum at clock 120 (1.350 → 0.944) and shows no
interior minimum at all — so 80 is where the optimum usually sits, not where it always sits. Same
numbers, opposite conclusion, depending only on whether the nuisance parameter was paired away or
averaged through.

## Why this directory exists rather than more files in `metrics/sweep/`

`metrics/sweep/` holds points from before the reference-template fix, at **L=127**. Three points
added to it afterwards were at L=121, and a curve assembled across the boundary attributed part of a
denominator change to the clock. The tell was the random-pairing floor reading `22.40` at clocks 25,
40 and 60 — identical to the digit, because those three shared a cached holdout the new ones did not.
See `docs/findings.md`, "Three structural checks that need no ground truth".

The three post-fix files were removed from `metrics/sweep/` rather than left there. That does **not**
make `metrics/sweep/` single-frame — its `arm-B-c40-n*` files sweep the **template and variant
counts** (10x10, 20x10, 30x10, 25x20, 50x10, 50x20; the holdout reference is 200 in all six). The
template count decides which template the run aligns onto, so they sit at L=127, 116, 120, 116, 124
and 124 respectively. What the removal fixes is narrower and is the thing that actually bit: the
four **clock** points (`c00`, `c25`, `c40`, `c60`) are now all L=127, so assembling a clock curve
from that directory no longer crosses a frame boundary. The `n*` files are not a clean sample-size
sweep and must not be read as one — `docs/findings.md` records the retraction of exactly that
reading.
