# Measured findings

Every number here was measured by a pipeline stage, and every row names the artifact that
regenerates it. **This file is where a measured value is established**; `performance.md`,
`known_issues.md`, `reproducing.md` and the `metrics/*/README.md` files quote from it or from an
artefact directly, and where any of them disagrees with the artifact, the artifact wins.

Docstrings used to carry them inline, and they rotted twice within a day: a
correlation was quoted as +0.29 when the params override that produced it had been removed
(true value +0.31), and a bridge improvement was quoted as 16% for a variant that was never
shipped (12% for the one that was). A number in a docstring has nothing checking it. A number
in `metrics/*.json` is rewritten by `dvc repro`, and a stale one here is at least
cross-checkable against that file in one step.

**So: if you change a threshold, a source, or a descriptor, re-run the stage and update the
row.** If a value here disagrees with its artifact, the artifact wins.

## Homolog pairs

Current build: 2,586,927 corpus lines → 352,605 families (178,940 with ≥2 members) →
**1,660,105 pairs**, unoriented. `homolog_pairs.direction: none` is the EvoFlows-faithful setting
and the only one this repository accepts, so that is the whole pipeline.

The charge-oriented variant is recorded for reference only and cannot be rebuilt here: it kept
1,403,255 of those pairs — 15.5% dropped — the median |Δ net charge| between members being 2.37,
so a `min_score_delta` of 0.5 removed the directionless near-ties while 0.05 filtered nothing. (An
earlier revision said "~8%", against a pair total that has since been corrected to 1,660,105.)
Orienting a pair needs a property registry, which is not part of this repository — see
`docs/claims.md`, "Why there is no target property".

## The `max_pairs_per_family: 20` cap does bias the edit distribution

§4.2 enumerates *all* unordered pairs of a homolog family; `sample_pairs` caps each family at
`homolog_pairs.max_pairs_per_family: 20`. The cap is engineering, not oversight — our own largest
family would contribute 15,059,116 pairs by itself — but it changes which pairs the editor sees,
and until now nobody had checked whether it changes the edit-distance *distribution*. It does.

**How this was measured, without a full `dvc repro`.** `editjumps/measurements/homolog_cap_bias.py` (default
flags, ~6 min on a laptop) rebuilds the families from `oas_corpus.txt.gz` with the pipeline's
own clustering — MMseqs2 `easy-cluster --min-seq-id 0.5 -c 0.8 --cov-mode 1 --kmer-per-seq 100
--split-memory-limit 8G` over the 2,461,250 unique CDR-H3+L3 keys of its 2,586,927 lines, through
the stage's own `cluster_keys` and `group_families` — giving 396,842 families, **184,102** with ≥2
members and **1,709,573** pairs at cap 20, against the tracked build's 352,605 / 178,940 / 1,660,105. So the
multi-member family count reproduces to 2.9% and the pair count to 3.0%; MMseqs2's cascaded
clustering is not bit-reproducible across binary versions, so an exact match was not available and
every number below carries that ~3%. Full enumeration is **255,548,117** pairs, i.e. the cap
discards **99.33%** of them, which is why the uncapped arm is an *estimate* rather than a build:
each family's pairs were drawn uniformly without replacement (up to 1000), and weighting a family's
draw by `C(k,2) / n_drawn` is unbiased for the full enumeration. Effective sample size 93,014,
standard error on the mean **0.08** edits — the differences below are tens of standard errors.
Distances and the operation mix are rapidfuzz `Levenshtein.distance` / `.editops` over the joined
`VH.VL` strings, so this is CPU-only; no torch was involved.

Subsampling the corpus was tried first and **would have hidden the effect**: the same script over
`--max-lines 200000` puts cap 20 at 63.61 against an uncapped 64.38, a 1.2% gap instead of 7.1%.
Shrinking the corpus shrinks families, and it is the largest families that the cap acts on, so a
subsample understates the very thing being measured. Recorded because it is the obvious shortcut.

| | cap 20 (shipped) | cap 50 | cap 200 | cap 1000 | uncapped (est.) |
|---|---|---|---|---|---|
| pairs | 1,709,573 | 3,167,560 | 7,564,699 | 18,518,458 | 255,548,117 |
| mean edit distance | **60.76** | 63.06 | 66.04 | 68.54 | **65.42** |
| sd | 28.25 | 27.81 | 27.13 | 26.48 | **24.35** |
| p10 / median / p90 | 21 / 63 / 97 | 24 / 65 / 99 | 28 / 67 / 101 | 31 / 69 / 103 | **30 / 65 / 97** |
| substitutions | 86.26% | 86.02% | 85.67% | 85.32% | **84.23%** |
| insertions + deletions | 13.74% | 13.98% | 14.33% | 14.68% | **15.77%** |
| Wasserstein-1 to uncapped | **4.66** | — | 3.00 | 3.52 | 0 |

At cap 20 the pairs the editor trains on are **7.1% closer together** than the pairs §4.2 would
have used (60.76 against 65.42 edits), and the whole distribution moves, not only its mean: the
Wasserstein-1 distance to the uncapped distribution is 4.661, which equals the difference of the
means to three decimals — that happens only when the two CDFs never cross, so capped pairs are
uniformly stochastically closer. The lower tail moves most, p10 21 → 30 (+43%), and the capped
distribution is also *wider* (sd 28.25 against 24.35), because equalising families is what lets
between-family heterogeneity survive into the pair set.

The operation mix moves the same way: **the cap under-represents insertions and deletions by 12.9%
relative** (13.74% of operations against 15.77%). That is the part of this most likely to matter,
since the edit-flow loss is defined over alignment operations rather than over a distance, and the
§4.1 benchmark measures insertion precision and recall at exactly **0.000** (below). It is a
direction to look in, not an explanation: a 13% relative shortfall in indel supervision is far too
small to produce a zero on its own.

**Why, and the guess that was backwards.** The suspicion was that large clonal families would be
*tight*, so equalising them would inflate distances. The opposite is true — pair divergence rises
with family size almost monotonically, and so does the indel share:

| family size k | families | mean pair distance | median | sd | indel share | share of cap-20 pairs | share of uncapped pairs |
|---|---|---|---|---|---|---|---|
| 2 | 58,500 | 44.43 | 43 | 28.13 | 12.00% | 3.42% | 0.02% |
| 3 | 27,617 | 48.68 | 50 | 28.30 | 12.34% | 4.85% | 0.03% |
| 4 | 16,817 | 51.58 | 54 | 28.24 | 12.65% | 5.90% | 0.04% |
| 5–6 | 19,843 | 55.13 | 57 | 28.23 | 13.04% | 14.09% | 0.09% |
| 7–10 | 19,732 | 59.29 | 62 | 27.99 | 13.40% | 23.08% | 0.24% |
| 11–20 | 18,970 | 64.46 | 66 | 27.05 | 13.98% | 22.19% | 0.76% |
| 21–50 | 14,326 | 68.58 | 69 | 26.23 | 14.53% | 16.76% | 2.87% |
| 51–100 | 5,191 | 70.90 | 71 | 25.64 | 14.93% | 6.07% | 5.07% |
| 101–500 | 2,881 | 71.96 | 71 | 25.80 | 15.35% | 3.37% | **22.02%** |
| 501+ | **225** | 64.41 | 64 | 24.97 | 16.15% | 0.26% | **68.85%** |

The last two columns are the mechanism. The 3,106 families with more than 100 members hold
**90.9%** of the uncapped pair mass and only **3.6%** of the capped mass; the cap moves that mass
onto families of 2–20 members, whose pairs average 44–64 edits against the 64–72 of the families
that dominate the full enumeration. The mean only moves 7% because the per-bin means span 44–72,
not orders of magnitude.

The relation is **not** monotone at the very top: the 225 largest families sit at 64.41, below the
101–500 bin's 71.96, and they carry 68.85% of all uncapped pairs. That is why the uncapped mean
(65.42) lands *below* cap 200 (66.04) and cap 1000 (68.54) rather than above them, and it is why
raising the cap does not converge on the uncapped answer from below: the cap-20 mean is 60.76, the
sequence rises to 68.54 at cap 1000, and the true uncapped value is back down at 65.42.

**What to do about it, as a recommendation, not a change.** `max_pairs_per_family` is a tracked DVC
param, so nothing here was touched. Two readings:

* **cap ≈ 100** is the cheapest fix for the mean: cap 200 already matches the uncapped mean to
  +0.9% for 4.4× the pairs, and something near 100 would sit on it. No cap reproduces the uncapped
  *shape* (the best Wasserstein-1 on offer is ~3.0, against 4.66 at cap 20), because the uncapped
  distribution is dominated by 225 families whose own distribution is tighter than the corpus-wide
  one.
* **Or leave it at 20 and say so.** The bias is real, one-signed, and 7% on the mean; against
  `max_steps: 20000 × batch_size: 16 = ~320k` pairs consumed, cap 20 already supplies 5× more pairs
  than a run reads, so the cap costs distribution fidelity rather than data. Which of the two
  matters is an experiment (train one editor per cap, compare §4.1 indel recall), not an argument.

## Editor training dynamics — where the budget actually goes

Measured on the first real-budget editor run (job 10, L4, `max_steps: 20000`, `batch_size: 16`,
`lr: 1e-4`). Single-batch training loss, so read windowed means rather than individual steps.

| window (steps) | mean loss |
|---|---|
| 0 → 40 | 779 → ~324 |
| 800–1100 | 225.1 |
| 1500–1630 | 204.9 |
| 2500–2630 | 199.5 |
| 2700–2870 | 204.9 |

**"The run plateaus by ~step 800" — retracted.** The reading was that steps 1500–1630 and
2700–2870 have the *same* mean to one decimal, so 1,200 steps bought nothing. Look at the window
*between* them: 2500–2630 is **199.5**. The two equal values straddle a lower one, which is what
single-batch scatter of ±40 looks like — not a plateau. Against 800–1100's 225.1 the loss still
fell ~11% after step 800, and the same run went on improving to **~175 by step 6000**, only going
flat there. That last figure is the one part of this you cannot check here: job 10's curve is not a
committed artefact (`metrics/` carries no training artefact for any editor arm), so ~175 and ~6,000
come from the job log. The retraction itself does not depend on them — the table above is enough to
show the step-800 reading was scatter.

What survives: most of the descent happens in the first ~40 steps, throughput is ~1.05 s/step, and
the full 20,000 steps is **~5.8 h**. The shape of the conclusion survives too — flat from ~6000
still leaves ~4.1 h of the 5.8 buying little on training loss — but its onset was off by a factor
of ~7, and the specific "roughly 5 of those hours buy under 10%" was computed from the retracted
onset and is not restated. `patience: 6` at `eval_steps: 500` (a run gets ~3,000 steps to stop
failing to improve) is sized from the slow real decline; `params.yaml`'s `early_stopping:` block
carries the same note.

Caveat on all of the above: it is *training* loss on a run with no held-out set. Held-out
`val_loss` arrived after this run, so going flat is suggestive of convergence, not proof of it —
a flat train loss with a still-falling val loss would look identical here. That applies to the
step-6,000 reading exactly as it applied to the retracted step-800 one.

**Divergence threshold.** `lr: 0.02` on the 35M trunk goes non-finite by step 2 (gradient clipping
does not save it). Worth knowing before raising the learning rate far: the loop now aborts on the
first non-finite loss rather than spending the budget producing NaN.

**And a diverged run used to ship the NaN weights while logging the opposite.** On that same `lr:
0.02` run, early stopping fired and the log said "best val_loss 31.978 at step 0, and those are the
saved weights" — but the restore was guarded by `best_val < final_val`, and `31.978 < nan` is False,
so it never ran: **209 of the 216 saved tensors were non-finite**. Non-finite finals are now handled
explicitly rather than left to a comparison, guarded by
`test_nan_comparisons_cannot_silently_ship_diverged_weights`. The general form: a NaN comparison is
False in *both* directions, so it cannot be the only thing between a diverged run and a saved model.

**What the validation split does and does not guarantee.** The held-out set is *sequence*-disjoint
from train, recovered from the pairs themselves (connected components over pairs-as-edges cannot
merge two families). It is **not** homology-disjoint: components can split one family, so two
near-identical members can sit on opposite sides. Treat `val_loss` as an early-stopping and
divergence signal, not as a generalization estimate for unseen antibody families.

## Clock normalization: the formula is confirmed, and the two calibrations never disagreed

`clock_scale` multiplies every edit rate by `clock / len(x)`. **That formula is ours** — §3.3 names
the mechanism repeatedly and never defines it — and it was the reproduction's largest unverified
assumption. It is now checked against real weights, twice, and the check is available because
"unclocked" is not a separate mode: `scale = 1` is exactly `clock = len(x)`, so an unclocked run is
a *data point on the clock curve*, at clock equal to the sequence length.

Recorded clock sweep, one model (`eval_B_stock_appA`), one template set, L=127
(`metrics/sweep/arm-B-c{00,25,40,60}.json`):

| clock | mean Levenshtein to template | edits per clock unit |
|---|---|---|
| off (= 127) | 11.76 | 0.0926 |
| 25 | 2.23 | 0.0892 |
| 40 | 3.96 | 0.0990 |
| 60 | 5.07 | 0.0845 |

Least squares through the origin on the three *clocked* rows gives **0.08899** edits per clock
unit. Extrapolated to clock = L = 127 that predicts **11.30** edits; the unclocked run measured
**11.76**, i.e. **−3.9%**. That is an extrapolation to twice the largest clock the line was fitted
on, predicting an edit count 2.2× the largest one it saw (11.30 against clock 60's 5.07), and it
landed.

Second confirmation at a different length, so this is not a coincidence of L=127. Model B on the
pairs-file `VH.VL` templates, L=235: unclocked **30.80** edits (`generation_eval_B_full.json`)
implies 0.1311 per position, hence 5.24 at clock 40; `generation_eval_B_clock40.json` measures
**5.567**, **+6.2%**. Two lengths, two checkpointed runs, both within 7% — `clock / len(x)` is the
right reading of §3.3.

**And the "factor-of-two disagreement" between our two calibrations of `clock: 40` is not a
disagreement.** Both use the same arithmetic, `edits ≈ clock × λ̄` with `λ̄` the mean per-position
rate (the length cancels, which is the mechanism's whole point). They differ only in *which* `λ̄`,
and `λ̄` is a property of the checkpoint **and** the input distribution, not of the clock:

| run | model | inputs | λ̄ |
|---|---|---|---|
| `arm-C-c40` | `eval_C_oas_appA` | ty1 VHH family, L=127 | 0.0933 |
| `arm-B-c40` | `eval_B_stock_appA` | ty1 VHH family, L=127 | 0.0990 |
| `arm-A-c40` | `edit_flows_baseline` | ty1 VHH family, L=127 | 0.1028 |
| `arm-D-c40` | `eval_D_stock_linear` | ty1 VHH family, L=127 | 0.1350 |
| `generation_eval_B_clock40` | `eval_B_stock_appA` | pairs-file `VH.VL`, L=235 | 0.1392 |
| guidance sweep, unclocked | **`edit_flows`** | corpus val `VH.VL`, L=229.4 | 0.2331 |

A 2.5× spread, and the same model moves 1.4× between two input sets. So:

* **The ~4.0 figure is right, and it is not an estimate.** It is the measurement: 3.96 edits at
  clock 40 in the sweep above, 3.47–6.17 across the twelve `metrics/dn/` arms (all Ty1), 5.57 on
  `VH.VL`.
* **The ~8.6 figure is not wrong arithmetic; it is another model's λ̄.** The sweep it came from ran
  `--model-folder data/pretrain/edit_flows`, which is none of the four evaluated arms, on
  `oas_corpus.val.txt.gz` rather than a seed family. Transferring its λ̄ to arm B is the
  entire discrepancy. (The sweep was a property-guidance diagnostic and does not ship here — see
  `docs/claims.md` — but its λ̄ is a measurement of a real checkpoint and is
  kept because it is the widest point of the spread above.)
* **It is also 8% low on its own terms.** It assumed a ~250-residue `VH.VL`; the sweep edits the
  first 50 lines of that val file, which average **229.36** residues (231.36 tokens, and the
  sampler divides by the token length). The unclocked 53.94 edits therefore imply λ̄ = 0.2331 and
  **9.33** edits at clock 40 for *that* checkpoint — a prediction still unmeasured, since
  re-running the sweep clocked needs a GPU.

So `clock: 40` was not chosen on a contradiction, and neither derivation needs retracting. Two
corrections fall out of it:

* `params.yaml` once attributed "clock 25/40/60 → 2.2/4.0/5.1 edits" to the `deterministic:`
  sweep's own calibration. Those are the *homolog* clock sweep's numbers (2.23 / 3.96 / 5.07 in
  `metrics/sweep/arm-B-c{25,40,60}.json`); no deterministic-benchmark artefact records any of them,
  and `deterministic.clocks` is `50,100,200,400`. Right numbers, wrong provenance — corrected in
  `params.yaml`, which now says "our calibration".
* At clock 40 the twelve `metrics/dn/` arms — all of them Ty1 — spend **3.47–6.17** edits on the
  seed family, and `metrics/generation_eval_B_clock40.json` spends **5.57** on `VH.VL`. Across all
  115 committed clock-40 artefacts the range is wider, **3.47–6.90**. Either way it is the bottom
  third of a "3–11 edit band" quoted at the time as §4.2's. **§4.2 states no mutation count**; the
  band's provenance is unknown and it is retracted below, so read this paragraph as a record of how
  clock 40 was chosen rather than as an anchor. Centring that band at ~7 edits would want clock
  ≈ **71** for arm B and ≈ **52** for arm D, and no single clock centres both. That is a finding
  about the paper's "matching the expected number of mutations per sequence across methods" (§4.2)
  as much as about us: matched clocks are not matched budgets. Recommended, not changed —
  `edit_flows.clock` is a tracked DVC param.

## §4.1 deterministic benchmark — the harness works, the experiment does not

First run (job 43, 36m, 50 val sequences, 50 Euler steps). **The artefact is not in the
repository**: `metrics/deterministic_benchmark.json` is declared `cache: false` in `dvc.yaml` and
was never banked, so this table is the record — a missing artefact, not a missing result, as
`docs/model_cards/README.md` puts it.

| clock | exact match | mean Levenshtein to `z1` | no-op baseline |
|---|---|---|---|
| 25 | 0.000 | 44.46 | **38.34** |
| 40 | 0.000 | 48.46 | **38.34** |
| 60 | 0.000 | 52.88 | **38.34** |

Per class at clock 40: insertion P=R=**0.000**; substitution P=0.037 R=0.029; deletion P=0.128
R=0.005; `no_op` R=0.949.

**The editor is worse than doing nothing, monotonically in the clock.** Returning `z0` unchanged
scores 38.34; every clock setting scores worse, and more editing is further away. That is the
signature of edits that are random with respect to the task.

**This is not evidence about the model, because the experiment is wrong.** §4.1 constructs a
synthetic *dataset* of `(z0, z1)` pairs — the same shape as a training pair. A flow model can only
score above chance on deterministic positional rules if it was **trained on them**. Our editor was
trained on OAS homolog pairs and has never seen these rules, so this run measures zero-shot
transfer to an unseen rule set, which can only return ~0. It says nothing about the model and,
worse, nothing about clock normalisation — the assumption the benchmark was wired to settle.
Missing piece at the time: a training stage on synthetic pairs, then evaluate. These numbers are
therefore a **null control**, not a result.

**That stage now exists** (`train_deterministic_editor` in `dvc.yaml`), and the experiment has been
run — sky job 48 trained it, job 53 scored it over the clock sweep, and it met Table 1 on every
class. What is missing is the *artefact*: `metrics/deterministic_benchmark.json` is declared
`cache: false` and was never banked, so the result cannot be cited to a file here. See
`docs/model_cards/deterministic-editor.md`, which carries those numbers labelled `LEDGER-SOURCED`.

**RETRACTED: "the alignment ceiling is far worse on real antibodies."** It is not. That reading
was my own off-by-one, and the corrected numbers are:

| class | as run in job 43 | corrected (single chain) |
|---|---|---|
| insertion | 0.463 | **0.861** |
| substitution | 0.556 | **0.948** |
| deletion | 0.577 | **0.954** |

The benchmark was pointed at the joined `VH.VL` corpus, whose `.` separator is not an amino acid.
The scorer's alphabet encoder dropped it, so alignment indices referred to a sequence one shorter
than the label list, and every position after the junction — roughly half of each sequence — was
scored one out. The local validation used pure-alphabet random strings, which is exactly why it
passed.

So alignment ambiguity is **modest**, not disqualifying, and the earlier conclusion drawn from it —
that per-token provenance was promoted from "optional" to "necessary" — does not follow. It remains
the principled fix for the residual 0.86–0.95 gap; it is no longer urgent.

`predicted_edit_labels` now raises on any off-alphabet character in `z0` rather than silently
mis-indexing, and the runner scores the heavy chain alone. That is also the more faithful reading:
§4.1 is defined on "a natural protein sequence", and a joined string is two proteins with a
separator, across which an alignment is free to shift residues.

Verified in passing: the `encoder.pooler` fix works ("dropped 2 unused encoder-pooler tensor(s)"),
so `load_trained` can open current checkpoints again.

### Why a perfect editor cannot score 1.0 on the alignment path — the worked example

§4.1 claims its construction "yields a unique z1 for each z0, enabling precise assessment of mutation
type, position, and amino acid identity, unlike natural MSAs where alignments may admit multiple
explanations". The first half holds; the second does not follow. `z1` is unique, but the **edit script
that produces it is not**, and the rules' own script is not always the cheapest. Measured on
`z0[4:7] = "GHG"` → `z1[4:7] = "SHC"`:

| script | edits |
|---|---|
| the rules': delete G@4, insert S before H@5, delete G@6 | **3** |
| substitute G@4 → S, delete G@6 | **2** — same `z1` |

Needleman-Wunsch minimises cost, so it recovers the 2-edit script and charges the model for the
difference. The ceiling is therefore a property of §4.1's construction meeting a minimum-cost aligner,
not of the model, and `oracle_ceiling` measures it per run. The provenance path (below) has no such
ceiling because it reads the sampler's own decisions rather than a cheapest explanation.

### Per-token provenance removes the ceiling entirely (1.000 on all four classes)

`sample_edits` / `sample_edits_gillespie` now optionally carry a per-token **provenance index** —
for each output token, the index into the tokenized input it descends from, or `None` if the sampler
inserted it. `deterministic_benchmark --provenance` scores from that instead of from an alignment.

Measured on 5 **real** heavy chains (the ones vendored in this repo; `oracle_provenance` replays
§4.1's ground-truth edits through the sampler's own `apply_event` bookkeeping, so this is a perfect
editor in both columns — same sequences, same scorer, only the edit-recovery path differs):

| class | via Needleman-Wunsch | via provenance |
|---|---|---|
| insertion | 0.778 | **1.000** |
| substitution | 0.939 | **1.000** |
| deletion | 0.951 | **1.000** |
| no_op (precision) | 0.995 | **1.000** |
| substitution identity | 0.935 | **1.000** |

Precision and recall are equal within each cell (a perfect editor's misses are symmetric). The
alignment column is lower than the 50-sequence figures above (0.861 / 0.948 / 0.954) only because
5 chains is a small sample; the direction and the size of the gap are the same.

So the ceiling is not merely raised, it is **gone**: per-class numbers on the provenance path are
read against 1.0. That is what makes the paper's Table 1 insertion precision of 0.820 a comparable
number rather than a figure sitting 0.04 under our own harness limit. The run report carries
`oracle_ceiling_provenance` so this is re-measured in-run rather than assumed.

One ambiguity is irreducible and is counted, not hidden. An inserted token has no origin, so an
inserted run is only known to sit *between* two surviving input positions; if the positions between
them were deleted, which one §4.1 would call the insertion's position is not recoverable. We
attribute to "nearest preceding surviving position + 1" (the convention `edit_targets` already
uses). On the 5 real chains that costs nothing (`insertions_unattributable` 0 of 9 insertions),
because §4.1's insertion and deletion rules rarely fire on adjacent positions. `inserted_tokens`
reports how much output material has no input position at all — edits landing on it are outside
§4.1's ground truth and are reported as unscoreable rather than dropped.

## Appendix-B evaluation — four construction choices the numbers themselves exposed

Regenerate: `make generation-eval`, `make figure-extract`. All four were found because a number came
out impossible, not because the code looked wrong.

**1. The natural reference.** Two constructions were tried and both are wrong in different
directions: one partner per template gives only `n_templates` sequences, and MMD's biased estimator
against 8–25 samples is dominated by that small side; the whole held-out tail gives enough samples
but spans **many families**, while the paper compares variants of one seed against a holdout from
*that seed's* family. Switching from the first to the second moved MMD **3.28 → 4.66** — it did not
converge on a stable number, which is the signal that the construction and not the model was in
question. `--family-fasta` is the paper's construction (§4.2, one family split into inference and a
disjoint holdout); without it the report carries a `caveats` entry saying MMD and KL are comparable
across our arms but **not** against the paper's figures.

**2. The random-homolog-pairing ceiling needs as many DISTINCT natural sequences as the model
generated.** Two versions shipped before that was true:

* one partner per template → n=25 against the model's 500. MMD's biased V-statistic inflates at
  small n, which put the ceiling at **1.512** against the model's **0.995** and made it look as
  though our editor beat real homologs. It did not: corrected, the ceiling is **0.549**. All three
  numbers are `metrics/sweep/arm-B-c40-n500.json` and `metrics/decomp/ceiling-fixed.json`, the same
  25x20 run before and after. (This paragraph read "0.964" until an audit caught it — that is
  `metrics/dn/A-ty1-s0.json`, a different 30x10 run, and pairing a buggy ceiling with an unrelated
  run's model value understated the size of the bug.)
* `n_variants` copies of those same 25 → the right count, but 25 distinct sequences, so the set's
  internal diversity (hence its kernel diagonal) is nothing like the model's.

**3. The baseline sets are template-MAJOR.** Entry `i` of `baseline_random` is a mutated copy of
template `i // n_variants`. The obvious-looking `[c[0] for c in chosen] * n_variants` is
variant-major (`i % n_templates`) and agrees with the generated order only at `i=0`, so every
baseline edit distance was measured against the **wrong template**. The tell: `random_mutations`,
constructed to make the same ~4 edits as the model, reported **24.7** — simply the distance between
two unrelated family members (`metrics/sweep/arm-B-c40-n500.json`, and
`metrics/decomp/ceiling-fixed.json`, which is the same 25x20 run with only the ceiling repaired).
Fixed, that run reports **3.87** — `metrics/decomp/lev-fixed.json`, the only committed file carrying
the corrected value. This said "~28" until an audit checked it against the artefacts; no committed
file carries that. MMD and KL compare set against set with no pairing, so they were
unaffected, and the model's own Levenshtein comes from `per_template` rather than from there.

**4. Figure 3's y-axes are not all linear.** The Levenshtein-to-template panel is log-scaled (ticks
2, 5, 10, 100). Choosing the scale by comparing linear and log residuals picked *linear* there and
produced a mean edit distance of **−4.78**, which is impossible — that is how the bug announced
itself. The scale is now inferred from a property of the ticks instead: a positive axis whose
labelled range spans an order of magnitude or more is log10. Panel clustering has the same shape of
hazard: in this figure the panel-to-panel x gap is **22.0 pt** and the method columns *within* a
panel are **7.9 pt** apart, so the threshold has to sit between them. At 25 it merged all five
panels of a row and still "worked" — 180 points, a plausible-looking calibration, silently wrong
axes. It is 15.

## ⑤ Gillespie sampler, and the diversity hypothesis it does NOT explain

`edit_flows.sampler: gillespie` is a grid-free next-event simulation of §3.3's eqs 9–11 with the
rate frozen between events — ours, not the paper's, which integrates it (see "Our Gillespie is a
deviation with a stated alternative" below):
inverse-CDF waiting time, one edit per event, model re-evaluated after each. It is now selectable
alongside Euler τ-leaping, both driven from the same rate field through the same clock
normalisation.

**Budget, checked against arithmetic.** With insertions only, clock normalisation cancels the length
exactly (total rate = `len(x) · clock/len(x)` = `clock` at every state), so the expected edit count
over `t ∈ [0,1]` is just `clock`. Gillespie hits it: mean **19.70** edits at clock 20 and **50.11**
at clock 50 over 300 seeds on a length-61 input. Euler agrees at the settings the pipeline runs —
**40.22** vs **40.02** at `n_steps=50`, clock 40 — and diverges only once `h·λ` approaches 1: at
`n_steps=1`, clock 200,
`h·λ = 3.3`, so every position inserts exactly once and Euler saturates at exactly **61.00** edits
(the sequence length) where Gillespie still spends **199.49**. Euler's error is therefore regime-dependent, not one-signed: it
slightly *over*-fires while `h·λ` is small (`h·λ > 1 − e^(−h·λ)`) and *under*-fires once clipping
bites.

**The diversity hypothesis does not hold.** The hypothesis under test was that Euler's within-step
conditional independence, versus Gillespie's re-conditioning after every edit, could explain our
generated sets being ~4× less internally diverse than the paper's (average pairwise Levenshtein
7.16 against their 30.4–262.8). **That framing is itself retracted below** — our 7.16 is the
*within-template* mean and theirs is a set-level quantity, so the ~4× was never a like-for-like
comparison; on the commensurable pooled metric the gap is 1.1–1.3×. The probe below is unaffected,
because what it tests is whether the sampler choice moves diversity at all. Probe: a synthetic **self-limiting** rate field — a position's
substitution rate collapses once it has been fixed, which is the strongest re-conditioning signal a
rate field can carry — on 4 templates × 27 variants of length 100, both samplers at the same clock.

| n_steps | clock | h·λ | sampler | mean edits | mean pairwise Levenshtein |
|---|---|---|---|---|---|
| 50 | 25 | 0.005 | euler | 8.42 | 15.31 |
| 50 | 25 | 0.005 | gillespie | 8.32 | **15.06** |
| 50 | 40 | 0.008 | euler | 12.88 | 22.23 |
| 50 | 40 | 0.008 | gillespie | 12.88 | **22.10** |
| 5 | 25 | 0.050 | euler | 8.82 | 15.92 |
| 5 | 25 | 0.050 | gillespie | 8.32 | **15.06** |
| 5 | 500 | 0.990 | euler | 59.97 | 72.27 |
| 5 | 500 | 0.990 | gillespie | 43.10 | **55.75** |

At the settings we actually run the two are indistinguishable: **1.6% apart** on diversity at clock
25 and **0.6%** at clock 40, nowhere near the 4× gap to be explained. The mechanism only bites at
`h·λ ≈ 1` (bottom two rows), and at `n_steps=50` with clock 25–60 on a ~130-residue chain `h·λ` is
~5×10⁻³ — Euler already re-evaluates the model 50 times to fire ~10 edits, so there is almost
nothing left for Gillespie to re-condition.

**What the numbers point at instead.** In the probe, mean pairwise Levenshtein tracks the edit
budget at a near-constant ratio: 15.31/8.42 = 1.82, 22.23/12.88 = 1.73. The same ratio holds in a
real recorded run (`metrics/dn/B-ty1-s0.json`, clock 40): Levenshtein-to-template 3.78, pairwise
6.92, ratio **1.83**. Two variants each *k* edits from one template differ by ~2*k* minus their
overlap, so diversity is a function of the budget and essentially nothing else. Our 6.9–7.2 pairwise
comes from spending **3.8** edits per sequence. Reading the paper's 30.4–262.8 as implying ~17–146
edits assumes their panel is the same within-template quantity, which the section below shows it is
not; the inference is left here as written and withdrawn there. What survives, and is what this
probe is for: diversity is a function of the **budget**, not of the sampler, so switching samplers
cannot move it — raising the realised edit count can.

**Caveats, stated.** This is a synthetic rate field, not our trained model: it is evidence about the
sampler mechanism under a stand-in, and no GPU was available to run either sampler through real
weights. The structural part of the argument does not depend on the stand-in (the two samplers
provably coincide as `h·λ → 0`, and `h·λ` is measurably ~5×10⁻³ in our configuration), but the
diversity/budget ratio of ~1.8 is measured, not derived. Excluded from the table: `n_steps=2`,
clock 500, where Gillespie's `max_events` guard (`10 × n_steps` = 20) truncated trajectories and the
row measures the guard rather than the sampler.

## ② Alignment scoring — unit costs vs affine gaps + BLOSUM62, and what it does to the labels

Regenerate: `make alignment-scoring` (`editjumps alignment-scoring`), artifact
`metrics/alignment_scoring.json`. 5,000 homolog pairs reservoir-sampled from the 1,660,105 in
`data/pretrain/oas_homolog_pairs.tsv.gz`, seed 0, CPU-only pure Python — no torch, no GPU.

The aligner is not preprocessing: its columns **are** the training labels. `(real, real)` equal is a
no-op, unequal a substitution, `(ε, real)` an insertion, `(real, ε)` a deletion, and EvoFlows' Table
1 is precision/recall/F1 *per mutation type*. Two things about ours were flagged in the PR #60
review: the scoring is unit gap/mismatch cost, which is **ours** (§3.2 cites Henikoff & Henikoff
1992 — BLOSUM — at the claim that elementary edits are "actual biochemical events", so affine gaps +
BLOSUM62 is the standard reading of that citation), and the traceback's fixed `diag > delete >
insert` tie-break makes the alignment direction-dependent, while their coupling is symmetric by
construction (eq 8) and our pairs are built unoriented. Both are now measurable and selectable
(`edit_flows.path: needleman_wunsch_blosum62`); **neither is the default**, for the reasons below.

**Label distribution, pooled over all aligned columns** (the quantity Table 1's prevalences are):

| ② scoring | no-op | substitution | insertion | deletion | edits/pair | gap runs/pair | mean run length |
|---|---|---|---|---|---|---|---|
| unit, as drawn (**default**) | 0.7361 | 0.2396 | 0.0118 | 0.0125 | 62.48 | 3.47 | 1.66 |
| unit, canonical orientation | 0.7361 | 0.2396 | 0.0118 | 0.0125 | 62.48 | 3.47 | 1.66 |
| unit, pairs drawn reversed | 0.7361 | 0.2396 | 0.0118 | 0.0125 | 62.48 | 3.47 | 1.66 |
| BLOSUM62, gap 11/1 (BLASTP default) | 0.7326 | **0.2438** | 0.0115 | 0.0121 | 63.27 | **2.41** | **2.32** |
| BLOSUM62, gap 5/1 | 0.7341 | 0.2367 | 0.0143 | 0.0149 | 63.10 | 3.32 | 2.09 |
| BLOSUM62, gap 1/1 (≈ linear) | 0.7321 | 0.2063 | **0.0305** | **0.0311** | 64.63 | **9.43** | 1.58 |

Read three things off it.

**The mechanism works, and it is about gap *structure*, not class balance.** Going unit → BLOSUM62
at 11/1 moves the four prevalences by at most **+0.0041** (substitution) and −0.0034 (no-op) —
under half a percentage point. What actually moves is contiguity: gap runs per pair **3.47 → 2.41**
(−31%) at a mean run length of **1.66 → 2.32** (+40%), on the *same* indel mass (insertion +
deletion prevalence changes by −0.0007). That is exactly what a gap-*opening* charge is for: with a
linear penalty, one 3-residue indel and three scattered 1-residue indels cost the same and the
tie-break picks between them. So the deviation is real and it lands where the theory says, but it
re-*organises* the indel labels rather than re-*weighting* the classes.

**The gap penalty we would have to invent moves the labels ~7× more than BLOSUM62 does.** EvoFlows
states no penalty at all, so 11/1 is as much ours as unit cost is. Sweeping only the opening charge
(11 → 5 → 1) swings insertion prevalence 0.0115 → 0.0143 → 0.0305 and gap runs/pair 2.41 → 3.32 →
9.43. Adopting "the faithful scoring" therefore means adopting an unstated hyperparameter whose
plausible range changes the label distribution by more than the change we were making to be faithful.

**Both readings are far from the paper's own prevalences, for a reason the scoring cannot fix.**
Table 1 reports 0.888 no-op / 0.047 insertion / 0.043 substitution / 0.022 deletion, i.e. 0.069
indel. Ours is 0.732–0.736 no-op and 0.206–0.244 substitution across all six scorings — those two
are what the argument rests on, and they are stable. The indel share is **not**: it runs 0.0236 to
**0.0616**, and at a 1/1 gap charge it is close to their 0.069 rather than far from it. (This
paragraph said "0.024 indel under *every* scoring". Four rows are near it — the three unit scorings
at 0.0243 and BLOSUM62 11/1 at 0.0236 — while 5/1 is 0.0292 and 1/1 is 0.0616, 2.5× the quoted
figure. A first correction of this sentence then wrote the substitution span as 0.206–0.240, which
excludes 11/1's 0.2438 from its own table three lines above.) Our pairs are ~15 points more divergent and ~5× more substitution-heavy than
theirs, which is a **pair-construction** difference (CDR-H3+L3 cluster families at `homolog_pairs.min_seq_id: 0.5`, versus their per-seed
homolog search) an order of magnitude larger than any scoring choice here. If a Table 1 comparison
is the goal, that is the lever, not the substitution matrix.

**Asymmetry — measured, then removed by construction:**

| ② scoring | population | direction-dependent pairs | rate | mean \|Δcolumns\| when it bites |
|---|---|---|---|---|
| unit, as drawn (**default**) | 5,000 homolog pairs | 43 | **0.0086** | 0.91 |
| unit, as drawn | 3,000 random short pairs | 41 | 0.0137 | 0.76 |
| unit, canonical orientation | both | **0** | **0.0000** | — |
| BLOSUM62 11/1, as drawn | 5,000 homolog pairs | 9 | 0.0018 | 1.89 |
| BLOSUM62 11/1, as drawn | 3,000 random short pairs | 96 | 0.0320 | 0.17 |
| BLOSUM62 11/1, canonical orientation | both | **0** | **0.0000** | — |

"Direction-dependent" means `NW(a, b)` is not the mirror of `NW(b, a)` — insertions and deletions
swap when a pair is reversed, so the mirrored alignment is the same labelling seen from the other
side, and anything else is a different set of labels for the same pair. The review's figure (22 of
3,000 random pairs) reproduces **in kind but not in value**, because the rate is a property of the
generator, not of the aligner. Same measurement, same code, varying only the random-pair generator
(`--random-alphabet/--random-min-len/--random-max-len`):

| alphabet | length band | rate |
|---|---|---|
| 4 | 7 | 0.0227 |
| 4 | 10–20 | 0.0413 |
| 4 | 20–40 | 0.0877 |
| 8 | 7 | 0.0063 |
| 8 | 5–9 | 0.0090 |
| 20 | 7 | 0.0000 |
| 20 | 10–20 | 0.0030 |
| 20 | 20–40 | 0.0113 |
| 20 | 100–250 | 0.0523 |

Two orders of magnitude of range (0.0000–0.0877), with 22/3000 = 0.0073 sitting inside it — so the
number to quote is the one on real pairs: **43 of 5,000 = 0.86%** of homolog pairs get a different
label set depending on which member is drawn as the source, and when that happens the two directions
disagree by ~1 column, i.e. by one edit. Canonical orientation (align the lexicographically smaller
id sequence first, mirror the result back) makes both draw orders run the identical DP call, so the
residual rate is **exactly 0**, not merely small — 0 of 5,000 and 0 of 3,000.

And yet: pooling those 43 pairs back into the corpus, symmetrising changes the pooled prevalences by
**< 1e-5** (no-op −8.9e-7, substitution +7.6e-6) and `edits/pair` not at all at 3 decimals. Drawing
every pair the other way round and relabelling insertion↔deletion gives the same table again. The
asymmetry is a real correctness defect in the *per-pair* labelling; it is not a distributional one at
corpus scale.

**Decision: the default does not change.** `edit_flows.path` stays `needleman_wunsch` (unit costs,
legacy tie-break), and `needleman_wunsch_blosum62` (affine 11/1 + BLOSUM62, canonically oriented) is
selectable beside it. The evidence for leaving it alone: the class prevalences move by <0.5 pp, which
is not enough to explain any gap to Table 1, while the gap-open value we would have to invent moves
them by ~2 pp and is not in the paper; symmetrising moves the corpus-level distribution by <1e-5;
and every recorded editor number — job 9's run, the §4.1 benchmark, the §4.2 arms — was trained
against the unit-cost labels, so flipping the default silently invalidates all of them to buy a
sub-percent shift. The affine path is the right thing to switch on at the next full retrain, where
new numbers are being produced anyway and the contiguity change (2.41 vs 3.47 gap runs per pair) can
be evaluated against §4.1's per-class F1 rather than assumed.

Caveats, stated. The tokenization here is a **stand-in**: `ord(residue)` ids with BOS/EOS sentinels,
because the lean CPU env has no transformers, so these are not the ids a training run sees (the
aligner only compares ids for equality and indexes the substitution matrix, and BLOSUM62 scoring is
built from whatever vocab it is handed, so the op mix is unaffected — but it is not literally the
trainer's tokenizer). The BLOSUM62 non-residue rule (BOS/EOS and the `VH.VL` separator score +11
against themselves, −4 against anything else) is ours too; BLOSUM62 has nothing to say about them.
This stage is deliberately not a `dvc.yaml` stage: it reads a pipeline output and answers a question
about the pipeline's *configuration*, so wiring it into the DAG would rerun it on every pair rebuild
while the answer only changes when the aligner does.

## Alignment scoring — correction

> **CORRECTION to the paragraph above.** Comparing this homolog-pair op mix against the paper's
> Table 1 prevalences (0.888 / 0.047 / 0.043 / 0.022) is **not a valid comparison**: Table 1 is
> captioned "Mutation classification on the **deterministic benchmark**", so its prevalences are
> §4.1's *synthetic rules applied to natural sequences* — a different dataset from homolog pairs.
> Homolog pairs and rule-generated pairs have no reason to share a class balance.
>
> The commensurable comparison is our own §4.1 pair builder against Table 1:
>
> | | no-op | insertion | substitution | deletion |
> |---|---|---|---|---|
> | ours (§4.1 rules, antibody VH) | 0.8518 | 0.0173 | 0.0593 | 0.0716 |
> | paper Table 1 (§4.1, their proteins) | 0.888 | 0.047 | 0.043 | 0.022 |
>
> Much closer on no-op, and the residual difference has a mechanical explanation rather than a
> methodological one: §4.1's rules are **composition-triggered** (`A`→Sub, `C`→Ins, `G` between an
> `L` and a `K`→Del). Measured over 3,000 antibody VH sequences: G 10.28%, A 6.30%, C **1.76%**. So
> a G-rich, C-poor framework yields many deletions and few insertions — our del/ins ratio is 4.1x
> where theirs is 0.47x, which is what the rules predict from composition alone.
>
> So the "pair construction is the lever for Table 1 comparability" conclusion does not follow from
> this table. The recommendation to leave the default scoring alone stands on the other two
> arguments, which are sound: BLOSUM62 re-organises indels without re-weighting classes, and the
> gap penalty it forces us to invent moves labels ~7x more than BLOSUM62 itself does.

## EvoDiff-MSA baseline — what installing and running their model actually cost

§4.2's fifth baseline is somebody else's released model, so the measurements worth recording are
about the model and the environment rather than about our code. Regenerate: `bash
editjumps/core/evodiff_msa/install_evodiff.sh` then `make evodiff-baseline` (stage `evodiff_msa_baseline`,
`metrics/evodiff_msa_baseline.json`).

**Nothing below is on the project's own seed family.** No `data/interim/seed_families` existed on
the machine these were taken on, so the runs used a 40-member synthetic VHH family (a 125-residue
nanobody scaffold, 3–12 random substitutions per member). They are measurements of the *plumbing and
the model's behaviour*, not of the baseline's standing against the editor — that needed a run on
the real family. **That has since happened**: `metrics/disjoint/evodiff-msa-{ty1,her2vh}.json` are
20x20 runs on the committed seed families and are the baseline's actual standing. The table below is
kept as the plumbing measurement it always was.

| | measured |
|---|---|
| `evodiff` version that resolves | **1.1.2** (with `sequence-models` 1.8.0) |
| what it does to `.venv` | numpy **2.4.6 → 1.26.4**, scipy **1.18.0 → 1.17.1** — hence the isolated env |
| MSA checkpoint download | ~380 MB, Zenodo record 8045076 (`msa-oaar-maxsub.tar`), HTTP 200 |
| self-test | loads on CPU, and the argmax at one masked position of a 66-column MSA recovers the true residue |
| `inpaint` throughput | 8 forward passes in **~8 s wall including checkpoint load**, 8×125 MSA, CPU |
| `unconditional` throughput | one 125-residue query row in **~30 s**, same MSA, CPU |

### Their own entry point is 37× off a matched budget

The load-bearing number. `evodiff.generate_msa.generate_query_oadm_msa_simple` — the package's only
MSA entry point — decodes the whole query row from a fully masked row. On the synthetic family, at a
matched budget of 2, it produced **74 mutations on a 125-residue query**: it does not edit x₀, it
replaces it. So §4.2's "matching the expected number of mutations per sequence across methods"
cannot be satisfied by calling their function, which is exactly why the budget-matched mode masks a
subset of x₀'s positions and decodes only those, and why that construction is labelled ours rather
than theirs (`editjumps/core/evodiff_msa/proposer.py`, `docs/evoflow_reproduction.md` §2.6).

Read `--mode unconditional`'s distances with that in mind: they are a different question wearing the
same column heading, and the metrics file flags them `matched: false`.

## The diversity/budget ratio of ~1.83 is not a law, and the Figure-3 comparison it supports is not admissible

The section above ends on a ratio: `pairwise_levenshtein / levenshtein_to_template` = 6.9163 /
3.7767 = **1.8313** in `metrics/dn/B-ty1-s0.json`, read as "diversity is a function of the budget
and essentially nothing else". The natural extension is the one that motivated this check: real
homologs sit **23.6167** edits from a template (`baselines.random_homolog_pairing`), and 23.6167 ×
1.83 = **43.22**, which falls inside the paper's recovered Figure-3 pairwise band of 30.43–262.77 —
so our diversity shortfall is a budget gap rather than a model defect. The arithmetic is right to
three digits. Three things about it are not.

**1. The ratio decays with the budget, and 1.83 is measured at one sixth of the budget it is being
extrapolated to.** Across every recorded run with more than one variant per template — over a
hundred distinct `(model, clock, lev, pairwise)` combinations in `metrics/` — the ratio runs
1.4142 to 1.9203, and it is not noise: it is a clean function of the budget *as a fraction of
length*, which is what the collision argument ("2·k minus their overlap") actually predicts. One free parameter:

    ratio ≈ 2 − 4.051 · (levenshtein_to_template / L)

with residual sd **0.036** (RMS 0.046) and worst residual **0.121** over a 10× span of `lev/L`
(0.0176 → 0.1713). The population, spelled out because the sd moves with it and an earlier revision
quoted 0.0479 from a set that cannot be recovered: every `metrics/**/*.json` with
`n_variants_per_template > 1`, an `L=` in `alignment`, and both
`defined_by_the_paper.levenshtein_to_template` and `.pairwise_levenshtein`, deduplicated on
`(model, clock, lev, pairwise)` and restricted to the editor's own arms — 92 combinations.
Adding the twelve rows carrying a `method` (EvoDiff-MSA, and the evotuned PLM forced and unforced)
gives 104, and takes the sd to 0.066 and the worst residual to −0.40, in
`metrics/disjoint/evotune-ty1.json`: the fit is a statement about the editor, not about anything
that makes edits. The two spans are stable across both readings — `lev/L` 0.0176–0.1713 and the ratio 1.4142–1.9203,
both extremes being editor rows either way — while the sd nearly doubles and the worst residual
moves from +0.12 to −0.40. That is the point. (This sentence said "only the `lev/L` span"; the
ratio span, which this paragraph opens with, is equally stable.) **The two model-free baselines are in
neither population**, and cannot be: 0 of the 300 `baselines.*` blocks under `metrics/` carries
`pairwise_levenshtein` at all, as `metrics/aligned/perseq-ty1.json`'s own `reading_notes.diversity`
records.
The `2` is not fitted, it is the triangle-inequality ceiling: two variants of one template are at
most `d(u,x) + d(x,v)` apart, so the ratio can never reach 2 and can only fall as edits start
colliding. `4.051` says collisions arrive ~4× faster than uniformly-placed edits would manage, i.e.
the editor works within an effective target of `L/4.05` ≈ **30 positions** on a 121-mer — CDR-sized,
which is its own small corroboration.

Applied to the 23.6167-edit budget on the L=120 alignment the 23.6167 came from, the ratio is
**1.203**, giving pairwise **28.41**, not 43.22. The claim is **1.52× high**. And the correction
does not depend on the fit: the nearest *measured* point is `generation_eval_D.json` at `lev/L` =
0.1713 with ratio 1.4271, the ratio decreases in the budget, so 23.6167 × 1.4271 = **33.70** is a
measured upper bound that 43.22 already exceeds.

**2. Real homologs do not have that ratio; they have ~1.0.** The ratio is a property of *our
sampler* — many near-independent trajectories out of one shared start — not of sequence divergence.
Family members share ancestry, so their mutual distances sit far below twice their distance to any
one of them. Already on record, in the same files: the `random_homolog_pairing` set's pooled
diversity against its own distance to the template is **0.962–1.040** across the twelve
`metrics/dn/` arms and the two `metrics/decomp/` probes. (115 files carry a `diversity_novelty`
block in all; over every one where the ratio is computable it is 0.941–1.117, so the tight range is
a property of that fixed configuration, not of the metric.) Measured directly on the two seed families, mirroring the evaluator's
construction (30 templates, 10 natural pool members each, `editjumps/core/family_split.py`):

| family | members | natural lev to template | natural within-template pairwise | ratio | pooled pairwise |
|---|---|---|---|---|---|
| `Anti-SARS-CoV-2_VHH_Ty1` | 38,687 | 25.12 | 23.54 | **0.937** | 23.53 |
| `Anti-HER2_scFv_VH_trastuzumab` | 43,088 | 23.40 | 23.38 | **0.999** | 23.18 |

So at 23.6 edits from a template, genuine family members are **23.5** apart. Both the claim's 43.22
and the corrected 28.41 describe a set *more* internally diverse than the natural one at the same
distance — 1.8× and 1.2× respectively. A budget-gap story whose endpoint overshoots nature is not
describing a deficit.

**3. Our `pairwise_levenshtein` and the paper's Figure-3 panel are not the same quantity, so the
band test cannot be run.** Ours is within-template (`generation_eval.per_template`, averaged over
templates), hence ceilinged at twice `levenshtein_to_template`. The paper's recovered panels are
`Avg Levenshtein to x0` ∈ [1.57, 247.71] and `Avg pairwise Levenshtein` ∈ [30.43, 262.77]
(`metrics/evoflows_figure3.json`, 36 points each, after the 2026-08-28 log-axis recalibration
recorded below).

Pair each point with its own method and dataset, and **30 of the 36 ratios exceed 2** — impossible
under a within-template reading. The six that do not are all *Random pairing*, whose minimum ratio
is 0.99: that arm swaps in a real homolog rather than editing, so a ratio near 1 is what a genuine
family member gives and it says nothing about the panel's definition. Across the five generative
methods the smallest ratio is **2.196**. The band's top, 262.77, also exceeds any member length in
either seed family (107–143 and 101–132). Whatever their panel measures, it is set-level, not
within-template.

**An earlier version of this argument used the cross-method worst case** — "their largest to-x0
against their smallest pairwise, 30.43 / 10.82 = 2.81". That 10.82 predates the recalibration; the
current maximum is 247.71, and the cross-method bound now gives 0.12, which supports nothing. The
rank-wise comparison above is the one that holds, and it is the right one anyway: a bound built from
two different datasets was never the quantity the triangle inequality constrains.

The commensurable number is one we already record: `diversity_novelty.diversity`, the pooled
pairwise distance over the generated set. Across the 12 `metrics/dn/` arms it is **23.07–27.74**
(mean 25.04), against real homologs at 23.58–24.33 and the paper's floor of 30.43 — a gap of
**1.10–1.32×**, against the **2.74–4.78×** the within-template metric reports over the same twelve arms and the same floor. On
the metric that can be compared, our generated sets are already as internally diverse as real
homolog sets, and short of the paper's floor by ~20%.

**Where this leaves the conclusion.** The ratio is a real and tight regularity *inside the clocked
operating regime* — 1.7937–1.8345 across all 12 `metrics/dn/` runs, four models × three seeds — and
the mechanism behind it (diversity tracks the edit budget) survives intact: our unclocked runs on
corpus `VH.VL` spend 30.6–40.1 edits and reach 46.1–57.2 pairwise (the one unclocked *seed-family*
run, `metrics/sweep/arm-B-c00.json`, spends 11.76 — the input form moves this as much as the clock), so raising the budget does buy diversity, close
to linearly. That is still evidence against a sampler defect. What does not survive is the specific
extrapolation, the 43.22, and the Figure-3 band comparison built on it — including §4.4's "4× below
their minimum" in `docs/evoflow_reproduction.md`, which is computed from the within-template metric
and should be restated against pooled diversity.

**One trap worth recording.** `metrics/matched/` cannot corroborate any of this: all 24 files ran
`n_variants_per_template: 1`, and `mean_pairwise_levenshtein` returns `0.0` for a set of one. Those
zeros are structural, not measurements, and averaging `pairwise_levenshtein` across `metrics/`
silently pulls in 24 of them.

## Figure 3 has no legend — the methods are its x-axis tick labels

The colour-to-method mapping was the last thing blocking a per-row comparison against the paper,
and it was never in a legend: the six methods are the **x-axis tick labels**, rotated 45°, repeated
under all ten panels. `figure_extract.py` was swallowing them as panel-title noise, which is why
panel 8's title came out `(forced) pairing (ours) (forced) 4 0 MMD`.

Order, left to right, now read from the PDF by all ten panels independently:

| x-rank | method | marker colour |
| --- | --- | --- |
| 1 | Random pairing | `(0.969, 0.745, 0.584)` |
| 2 | EvoFlow (ours) | `(0.627, 0.255, 0.0)` |
| 3 | EvoDiff-MSA | `(0.361, 0.514, 0.914)` |
| 4 | Evotune | `(0.471, 0.745, 0.992)` |
| 5 | Evotune (forced) | `(0.18, 0.435, 0.663)` |
| 6 | Random mutations | `(0.702, 0.855, 0.992)` |

Two independent confirmations. Geometrically, each label's first line starts 2.4 pt left of its
marker column and both share a 9.87 pt pitch, uniform to 0.01 pt over five gaps. Semantically, rank
1 is best on every quality panel and has the largest neighbourhood — what pairing two real homologs
looks like — and rank 6 is worst on every one, which the caption states outright ("Random mutations
perform worst across all metrics").

**Their Figure 3 is not budget-matched across all six rows.** Levenshtein-to-x0 is 14.52–69.89 for
EvoFlow, 15.17–80.99 for Evotune-forced and 13.83–69.89 for random mutations — matched, and EvoFlow
and random mutations share a maximum to five digits — but 4.40–30.44 for EvoDiff-MSA and 1.57–41.98
for Evotune, which undershoot. (Those six ranges are the post-recalibration `panel_0`; an earlier
revision quoted 5.18–7.79 and the rest from the log-axis reading corrected on 2026-08-28, recorded
below.) The caption concedes both ("typically
explores a smaller neighborhood", "introduces very few mutations, in some cases as low as 1.5"). So
**EvoFlows is 4th of 6 on MMD** (0.870–1.556, behind random pairing 0.289–1.098, Evotune 0.298–1.122
and EvoDiff-MSA 0.418–1.127) and only wins among the three rows that are actually matched. The claim
is Pareto, not a ranking, and a single-column table would misrepresent it.

**Per-dataset paper values are not obtainable.** Within one method's column the six datasets sit at
the same x — a strip plot with no jitter — so point identity across panels is lost. Per-method
summaries over the six datasets are exact; anything finer is not. Cell format is therefore
`median [min, max]` over datasets.

Their six datasets (Table 2, p.19) are Ty1 VHH (3335 homologs, L=109.2), anti-HER2 scFv (10979,
246.4), anti-EphA2 VHH (24304, 118.2), chicken FGF2 (1435, 141.6), serine-pyruvate aminotransferase
(6565, 363.0) and haloalkane dehalogenase DhaA (4697, 282.2). **We have 2 of 6**, both antibodies,
so our mean is over the two least divergent datasets and covers no enzyme or growth factor — which
is where their longest sequences and hardest MMD values live. They also use HER2 as one 246-residue
scFv; ours are two separate 122–127 residue chains.

## Three Figure-3 metrics were unmeasurable as recorded, and the fixed readings land in the paper's ranges

Three of the ten panels could not be compared against, for reasons that are properties of our
estimators rather than of the models:

* **MIP.** `mip.mean()` is **zero by construction** — eq 21's average product correction subtracts a
  term built to cancel the matrix mean, so our runs stored `1e-18` whatever the model did. Their
  panel spans 0.22–0.99, so it is an agreement score between the generated and natural MIp matrices.
* **Covariance.** We stored two Frobenius norms. A norm ratio can read 1.0 with the two matrices
  completely uncorrelated, so it cannot separate a model that reproduces the family's couplings from
  one that invents its own at the same magnitude. Also an agreement score.
* **JS / KL.** Computed over the **global** amino-acid composition, pooled across all positions and
  sequences. Two sets of close homologs share that almost exactly, so the number was tiny and nearly
  constant: 4.8e-5 to 3.8e-3 across all 172 committed artefacts, against the paper's 0.006–0.124. We were not 10–40× better; we were
  measuring something much easier.

`matrix_agreement` (Pearson over the upper triangle, pairs with `|i-j| < 5` excluded so backbone
contacts do not inflate every method — `np.triu_indices(L, k=min_separation)`, which KEEPS the
well-separated pairs; this sentence had the inequality the wrong way round) and `js_divergence_positional` (per aligned column, then averaged) replace
them. The old keys are kept: 166 committed artefacts carry `covariance_frobenius_generated` and
`mip_mean_generated`, and 172 carry the pooled `js_divergence` / `kl_generated_vs_natural`.
(Counts as of this writing. Four committed artefacts —
`metrics/disjoint/editor-{ty1,her2vh}.json` and `metrics/aligned/perseq-{ty1,her2vh}.json` — carry a
baked `reading_notes.kl` saying 153 rather than 151: it was written before three duplicate
`metrics/evotune/` files were removed, and those artefacts are records, not live views. The
generator now emits 151, so a re-run would not reproduce the 153.)

Measured on Ty1, real homologs against a 200-sequence holdout versus uniform random substitution:

| set | covariance | MIP | JS positional | JS composition (old) | pairwise pooled |
| --- | --- | --- | --- | --- | --- |
| real homologs (pool) | **0.9688** | **0.9212** | **0.0057** | 0.000024 | 23.75 |
| random mutations, 3 subs | 0.7063 | 0.4273 | 0.0303 | 0.001014 | 30.00 |
| random mutations, 30 subs | 0.7014 | 0.1414 | 0.0817 | 0.005644 | 63.42 |
| **paper's Figure 3** | 0.742–0.995 | 0.227–0.992 | 0.0015–0.0349 | — | 30.4–262.8 |

All three fixed metrics land inside the paper's range and order the two extremes the way the paper
does — real homologs at their random-pairing end, uniform substitution at their random-mutation end.
That is the strongest check available without a model, and it is what the old readings failed: the
composition JS on real homologs sits **62× below** the JS panel's minimum (0.000024 against 0.0015).

**`pairwise_levenshtein` is also replaced for table use.** The old key is the mean over templates of
the within-template variant-set mean, which is structurally **0** whenever `n_variants` is 1 — every
run under `metrics/matched/` is 300x1, so all 24 files carry a zero there — and the triangle
inequality caps it at twice the mean distance to the template. That cap is what made the comparison
inadmissible: paired within method and dataset, 30 of the paper's 36 points imply a ratio above
that ceiling of 2.0 — 2.196 at worst across the five generative methods — so their panel must be
pooled across templates. `pairwise_levenshtein_pooled` is the
comparable one.

## Figure 3's Covariance and MIP panels plot a quantity the paper never defines

Checked exhaustively against arXiv 2603.11703v2, all 22 pages, because a claim about another
paper's contents has to be verifiable and not remembered.

**No split size is stated anywhere.** §4.2 gives only "Each homolog set is split into train,
inference, and holdout" — no proportions, no counts. Every occurrence of the split vocabulary across
the paper was checked for a nearby number; the only hits are the method count ("6 methods"), page
numbers, and residue letters in a figure axis. Table 2 gives homolog totals per dataset (1435 to
24304, a 17x spread) but never how they divide.

**B.3 defines the covariance and MIP MATRICES and stops there.** Eq 17 gives `f_i`, `f_ij` and
`C_ij`; eq 18 contracts to `||C_ij||_F`, an `(L, L)` matrix; eq 19-22 build `MIp_ij`, also `(L, L)`.
No equation maps either matrix to the single number the panels plot, and no prose does either. So
the finding is stronger than "the reference size is unstated" — **the plotted quantity is
unstated**. Our `matrix_agreement` is an inference from the axis range (0.742-0.995 and 0.227-0.992
both fit a correlation and fit nothing else we tried), not a reading of a definition.

**No ceiling or reference baseline is reported** for either panel, and B.4's small-sample discussion
covers only KL, where it adds BLOSUM62 smoothing (eq 24). Nothing is applied to B.2 or B.3, and
smoothing would not help: the problem there is variance in the estimate, not zero counts.

### This makes it 6 of 10 undefined, not 3 — in three different ways

`generation_metrics_undefined.py` was built around three named-but-undefined metrics — entropy delta,
JS divergence, profile log-likelihood. Three more belong in that count, each for a different reason.
Two because their matrices are defined and their scalar reductions are not. And **KL**, which has
equations — eq 23 the divergence, eq 24 the BLOSUM62 smoothing — and still could not be recomputed,
because neither says what the divergence is computed *over*. Our pooled amino-acid composition read
0.0005 against their panel's 0.0059–0.1237, ~12x below it and in the same direction the pooled JS was
off by; the per-aligned-column reading of the same formula reads 0.0713, inside their range. A
defined formula is not a recipe without its input representation. Panel by panel, what can be
recomputed from the paper as written:

| panel | status |
| --- | --- |
| Avg Levenshtein to x0 | defined |
| Avg pairwise Levenshtein | reduction ambiguous (within-template vs pooled) |
| Covariance | **matrix defined, scalar reduction undefined** |
| ESM2 PLL | defined in prose (B.2, single random mask order) |
| Entropy delta | undefined |
| JS divergence | undefined |
| KL divergence | **formula defined (eq 23-24), representation undefined** |
| MIP | **matrix defined, scalar reduction undefined** |
| MMD | defined (eq 25-27), except `k` |
| Profile log-likelihood | undefined |

Three of ten are exactly recomputable, and one of those three only modulo the unstated `k`. Six are
undefined: three with no definition anywhere, two with a defined matrix and an undefined reduction,
one with a defined formula and an undefined representation. The tenth, avg pairwise Levenshtein, is
defined but its reduction is ambiguous — settled here by the triangle inequality rather than by the
paper. **The number to state is 6, not 7.** A panel counts as undefined when the paper's text does
not determine the quantity at all; avg pairwise Levenshtein names its quantity and leaves only the
pooling open, so it is kept as its own row rather than folded in to make the headline larger.

Both KL readings are kept, as `pairwise_levenshtein` and `pairwise_levenshtein_pooled` are:
`kl_divergence_positional` is the one to place beside Figure 3, `kl_generated_vs_natural` is the
pooled composition. 172 committed run artefacts report a KL; 21 of them carry the positional
reading as well, and the remaining 151 carry the pooled reading only. Nothing was
rewritten in place — a number keeps the provenance it was measured with.

### Their panels are still readable; ours are the ones with a problem

Their numbers are not artefacts of the unstated reference size, and the evidence says so clearly.
On Ty1, real homologs on both sides — where the honest answer is ~1.0 — agreement rises with the
reference and saturates near 0.985:

| ref n | covariance | MIP |
| --- | --- | --- |
| 20 | 0.7901 | 0.6399 |
| 50 | 0.8648 | 0.8074 |
| 100 | 0.9214 | 0.8827 |
| 200 | 0.9615 | 0.9386 |
| 400 | 0.9820 | 0.9713 |
| 800 | 0.9850 | 0.9818 |
| 1600 | 0.9855 | 0.9840 |

**All six** of their methods peak above that saturation — random pairing 0.9945, EvoDiff-MSA
0.9937, Evotune 0.9933, EvoFlows 0.9911, Evotune-forced 0.9900, random mutations 0.9881. (This read
"five of their six ... and even random mutations reaches 0.9881", which sets 0.9881 up as the
exception it is not: it clears the 0.9855 saturation like the rest.) A small reference cannot
produce that across six methods, so their reference sets are large.
What the missing number blocks is comparison *between runs at different reference sizes*, not
interpretation of theirs.

The consequence falls on us. Our default holdout is 200, which caps a **perfect** editor near 0.96,
so a reader comparing our 0.9x against their 0.99 reads a design difference as a model difference.
Recording `agreement_reference_n` does not fix that; the attainable ceiling does, and it needs no
model, because real-homolog-versus-real-homolog is computable from the family alone.

The reference is **200**, not 300. `generation_eval` sliced `natural_all[:300]` when this was
written (it is `max(300, holdout_size)` now, so an explicit larger holdout is honoured — see
"Normalising to the ceiling does NOT cancel the reference size"), but `natural_all` is
the family holdout, and `holdout_size` defaults to 200 — so the slice is inert and 169 of the 173
committed occurrences of `agreement_reference_n` are 200. The four that are not are the deliberate
`--holdout-size 800` runs (`metrics/agree/h800-oldalign-*`, `metrics/aligned/h800-*`), which are the
subject of "Normalising to the ceiling does NOT cancel the reference size" below and postdate this
paragraph. A first set of ceilings quoted at "n=300" was wrong in a second way as
well: it drew the reference from holdout+pool rather than from the holdout the runs actually score
against. Both errors are corrected below, together — the corrected figures are the table at the end
of this section, not here, because a second error had to be fixed first.

And the reference must be projected onto the **same template** the run scored against, which the
first two attempts also got wrong. `generation_eval` sets `reference_template = chosen[0][0]`, where
`chosen` is a *seeded sample* of the template list — not its head. So `split.templates[0]` is a
different sequence: L=121 against the run's L=115 on Ty1, L=122 against L=127 on HER2-VH, and the
artefacts say so themselves (`agree2-ty1.json` records "L=115"). Ceilings built on
`split.templates[0]` sat in a different coordinate system from the scores they were meant to
normalise — numerator and denominator in different alignments.

Corrected: the run's own template, its own 200-sequence reference, and a ceiling set of 400 (matching
the generated set's n), over 8 independent draws:

| family | covariance ceiling | MIP ceiling | positional-JS floor | natural pairwise |
| --- | --- | --- | --- | --- |
| Ty1 (L=115) | 0.9693 ± 0.0090 | 0.9125 ± 0.0083 | 0.00482 ± 0.00014 | 23.67 |
| HER2-VH (L=127) | 0.9563 ± 0.0079 | 0.9296 ± 0.0082 | 0.00644 ± 0.00046 | 23.57 |

These are read from the per-draw working further down this file ("Per-family, per-draw working"),
which is the only place the eight draws are written out. Its **MIP** means and sds reproduce from
those draws exactly; the covariance ceilings, the positional-JS floors and the natural-pairwise
values in that block have no per-draw working anywhere in this file, so they are recorded rather
than checkable. This table previously gave 0.9717 / 0.9149 and 0.9542 / 0.9269 for the same stated
construction — the same family averages (0.9628, 0.9210) split differently between the two families,
which is why the disagreement survived. The per-draw values win.

**Why MIP moved and covariance did not.** Across the three construction choices — reference size,
ceiling-set size, and alignment — covariance's ceiling is insensitive to all three and does not even
move in the same direction across families, while MIP's is sensitive to all three (Ty1's MIP ceiling
is 0.9348 at L=121 against 0.9193 at L=115 under the alignment comparison below — a third value
again, from a construction whose working is not written out; treat the *direction* as the finding
and not the third digit, which is the point of the next paragraph). That is why the covariance cell survived three separate
construction errors unchanged and the MIP cell survived none of them.

**And the third significant figure was never supportable.** The ceiling's own sampling error is
±0.5–0.6 percentage points, several times the difference the earlier drafts reported between
readings. The intervals now travel with the numbers.

Third instance of the same move — the §4.1 alignment ceiling, a chance floor in work that is no
longer part of this repository, and now this — so it is a house rule rather than three fixes:
**report the number, the sample size that produced it, and the best attainable value at that
sample size.**

The single cleanest demonstration, from the ceiling error above: our covariance agreement of 0.8217
is **83.6%** of a ceiling of 0.9827 and **85.1%** of a ceiling of 0.9661. Same editor, same run, same
arithmetic — the only thing that varies is which reference set the ceiling was computed from, and no
reader of either figure could tell which one they had been given.

A note on the direction, because the obvious generalisation is wrong. It is tempting to say an
uncontrolled number flatters its author; across the four controls this section corrects, it did the
opposite every time. The too-high ceilings pushed our percentages down (83.6% against 85.1% on
covariance, 74.2% against 76.5% on MIP); the too-low JS floors pushed our multiple up (4.31x against
2.75x); the 20-sequence-reference bug made a working editor read as broken (0.790/0.493 against
0.822/0.713); and the single-peak reading of their reference size under-claimed evidence that
supported them. The shared property is not optimism, it is **uninterpretability** — a number without
its control cannot be checked or compared by anyone, in either direction. That is the claim the
evidence carries, and it is why the house rule above is stated without reference to direction.

### Normalising to the ceiling cancels the reference size — and their random-pairing row IS their ceiling

> **Retracted.** The first half of this heading is false, measured on our own editor some 500 lines
> below: ["Normalising to the ceiling does NOT cancel the reference
> size"](#normalising-to-the-ceiling-does-not-cancel-the-reference-size-measured-on-our-own-editor).
> The second half — that §4.2's random-pairing row is their ceiling — stands. This page keeps
> withdrawn claims in place rather than deleting them, but the heading read as a live conclusion
> for **nearly five hundred lines** with nothing pointing forward.

The unstated split stops blocking the comparison once both sides are expressed as a fraction of
their own ceiling, because the reference size affects the score and the ceiling together. And their
ceiling is already in the figure: §4.2 describes random inference homolog pairing as "an approximate
upper bound on family-level similarity, since both sequences are drawn from the natural homolog
distribution" — which is exactly the real-versus-real construction. Comparing means is exact even
though our extraction sorts values within each method, since the mean is order-invariant.

| | ours (Ty1 + HER2-VH, n=200) | paper (EvoFlow, 6 datasets) |
| --- | --- | --- |
| covariance, % of ceiling | 0.8217 / 0.9628 = **85.3 ± 0.6%** | 0.9718 / 0.9878 = **98.4%** |
| MIP, % of ceiling | 0.7131 / 0.9210 = **77.4 ± 0.6%** | 0.8855 / 0.9268 = **95.5%** |
| positional JS, x floor | 0.0151 / 0.00564 = **2.68x** | 0.00448 / 0.00249 = **1.80x** |
| pairwise, x natural | 25.24 / 23.62 = **1.07x** | 117.44 / 117.13 = **1.00x** |

Earlier revisions of this table read 85.1%, 76.5% and 2.75x, computed against ceilings in the wrong
alignment (above). The 76.5 in particular was not reproducible: 77.43 ± 0.58 is the figure **for the
runs this table uses** — the L=115/127 `metrics/agree/agree2-*` pair. A later section
("Three values for MIP are now in circulation") records that the *fixed* runs, at L=121/122, give
**84.3% and 78.9%** instead, and those are the numbers to quote for the current artefacts. Both
computations are internally correct; they are of different runs. Note that
the two covariance columns are **not** comparable anyway, for reasons the next section gives; the
row is kept here because the ratio is the right *within-study* summary, not because the comparison
is legitimate.

So the gap is real and it is specific: **our editor reproduces amino-acid composition and overall
diversity about as well as theirs, and reproduces coevolutionary structure considerably worse.** It
reaches 85% of the attainable covariance agreement against their 98%, and 77% of MIP against their
96%. Diversity is the one axis where we are not behind — 1.07x natural against their 1.00x, i.e. we
edit slightly past the natural spread rather than short of it.

Two caveats that keep this honest. Their JS is the composition reading for all we know (B.4 defines
only KL, and the JS panel is undefined), so that row may not compare like with like. And the
covariance and MIP scalar reductions are our inference, so if theirs is not a correlation, none of
these four rows compares the same quantity — see above.

## A rented A100 sat at 0% for 23 hours, and the fix was already written down in this repo

Sky job 58 ran `evotune_esm` for 23 h 23 m on an on-demand `a2-highgpu-1g` and reached step 825 of
45,540 — a 44-day horizon. Two independent bugs, either fatal alone.

**It was on the CPU the whole time.** Measured on the box, not inferred: A100-SXM4-40GB at **0%**
utilisation, **0 MiB** of 40,960 used, **41 W** of a 400 W limit, load average 1.26 on 12 vCPUs, one
core pegged, 28 GB resident. The venv the trainer actually ran from held `torch 2.12.1+cu130` with
`cuda avail False`, while setup had printed `CUDA available: True | torch 2.12.1+cu129`.

Mechanism: setup installs the driver-matched `+cu129` wheel and verifies it. Then `dvc repro`
launches each stage's own `cmd:` from `dvc.yaml`, and `evotune_esm`'s reads
`uv run --group train python -m editjumps.pipeline.train.evotune` — **no `--no-sync`**. That re-resolves
the environment and reverts torch to the locked default `+cu130`, which the VM's driver cannot run.
The revert happens *after* setup's fail-fast check, so setup reports CUDA fine and the run is on CPU
anyway.

`train.sky.yaml` documents this exactly, including "Measured: an L4 job ran 100 steps in 27 minutes
on CPU". The guard was never carried into `repro.sky.yaml`. Third occurrence of the class, after that
L4 run and EvoDiff's 11.6 CPU-hours.

The fix is `export UV_NO_SYNC=1` in the run block, not a flag. The commands that matter are not in
the sky yaml at all — they are the `cmd:` strings in `dvc.yaml`, and adding `--no-sync` there would
change every stage's command string and so its dvc hash, invalidating the lock for stages whose
behaviour did not change. The env var reaches them without touching a hash. Plus a run-environment
CUDA re-check, because a check that runs in `setup:` cannot see a revert that happens after it.

**And the step budget asked for 100x the work intended.** `evotune.max_steps: -1` with
`epochs: 20.0`, justified by a comment reading "a family is a few hundred sequences, so one epoch is
a few dozen steps". The Ty1 FASTA holds **38,687** usable homologs. One epoch at batch 16 is **2,277 steps** — the family's *train* split, which is what the trainer iterates; 38,687/16 = 2,418 counts the whole family and is the wrong denominator — so 20 epochs is **45,540**. (2,277 is the artefact's own figure: `ty1-evotune_esm.json` records `final_epoch` 0.87835 at 2,000 steps.) Now `max_steps: 2000` explicitly — about 0.88 of a pass, the same
order as the editor arms' 3,000, and appropriate for a baseline whose purpose is to be
budget-matched rather than converged.

Also worth recording: eval was 1,934 samples at `eval_samples_per_second: 0.638`, so **each
evaluation took 50 minutes** and ran every 800 steps. On CPU that is a symptom, not a cause; it is
noted because it is the number that first looked wrong.

### Three tests, and the third one caught the first two

`test_gpu_jobs_pin_the_environment_before_any_uv_run` accepts either mechanism — `--no-sync` per
call, or `UV_NO_SYNC` exported before `dvc repro` — because the rule is that the environment must
not be re-resolved, not which mechanism prevents it. It also requires a run-environment CUDA check.

Two earlier versions of that test failed on the file they exist to protect, both by matching their
own explanatory comments: first on the words "uv run", then on "dvc repro". A guard whose prose
describes the thing it forbids has to scan code lines, not the file. Worth remembering, because the
same shape will recur in any test that documents itself.

`test_no_training_stage_leaves_its_step_budget_unbounded` flags `max_steps: -1` combined with
`epochs > 1`. Not `epochs >= 1`: the corpus `pretrain` stage deliberately runs one pass, and "one
pass" is a defined budget that needs no corpus measurement. More than one pass with no step cap
multiplies a size nobody measured, which is precisely where 45,540 came from.

## A 413 from MLflow destroyed the record of a run that had already succeeded

Sky job 66 completed 2,000 evotune steps on the A100, reached `eval_loss` **0.2812**, saved its
model — and then died. The failure was `HTTPError: 413 Client Error: Request Entity Too Large` from
`mlflow.log_artifacts`, uploading the 650M model directory to the Cloud Run tracking server, whose
request body cap is 32 MB against a ~2.6 GB safetensors file.

Nothing durable survived. `metrics/evotune_esm.json` was never written and nothing reached
`repro-artifacts/`, so the only record of that run is the job log. Two independent defects, and the
repo already contained the argument against both.

**`log_artifacts` was not guarded, while `log_artifact` was.** `editjumps/core/utils.py` wraps MLflow calls
so that "tracking must never abort real work", and the plural — the one that uploads a whole model
directory, and therefore the one that fails against a remote server — was missing from the list.
`mlflow.log_text` was missing too; the regression test found that one, not us.

**The metrics JSON was written after the upload.** So the most failure-prone call in the function ran
before the durable record. The same file's own comment on `end_run` says a tracking failure there
"would raise *after* the work and the metrics JSON were finished, which is the worst place to fail" —
this was that failure, one call earlier, where the JSON had not been written yet.

Fixed three ways: `log_artifacts`, `log_text`, `log_dict` and `log_figure` joined the guarded list;
the JSON write moved ahead of the upload; and the upload is now skipped when the largest file exceeds
`MLFLOW_ARTIFACT_FILE_LIMIT` (24 MB, under Cloud Run's 32 MB cap), since the binding constraint is
the biggest file rather than the total — MLflow uploads one file per request. The model was never at
risk: `repro.sky.yaml` copies it to GCS and `EDITJUMPS_CHECKPOINT_URI` covers periodic checkpoints.

### The guard test derives its allow-list, because a restated one drifted immediately

`test_mlflow_tracking_failures_cannot_abort_a_finished_run` scans every `mlflow.*` call site in the
pipeline and requires each to be a guarded name. Its allow-list is parsed out of `utils.py` rather
than written out again: the first version hardcoded it, and guarding `log_text` then failed the test
on the very change that fixed the bug it was written for. Same shape as the two `uv run` guards that
matched their own explanatory comments — a test that restates what it checks will disagree with it.

### What the fix bought, measured

Job 58 against job 66 on identical work, after the CUDA revert was fixed:

| | job 58 | job 66 | ratio |
| --- | --- | --- | --- |
| training | 85 s/step | 5.78 it/s | 490x |
| eval throughput | 0.638 samples/s | 145.5 samples/s | 228x |
| eval wall-clock | 3,005 s | 13.2 s | 228x |

## §4.2's last two methods land, and our port beats both — the first comparison it wins

> **Superseded.** This section's comparison was drawn in the LEAKY frame — its rows come from
> `metrics/agree/agree2-*` and `metrics/aligned/perseq-*`, which `consolidate` labels
> `train_overlap_113` / `train_overlap_103`, and rule 1 of [`reproducing.md`](reproducing.md) says a
> row from that frame does not go beside a `disjoint` one. ["The two metrics our port leads on do
> not survive an interval — the disjoint frame is a
> draw"](#the-two-metrics-our-port-leads-on-do-not-survive-an-interval--the-disjoint-frame-is-a-draw)
> supersedes it on frame grounds, independently of any interval. The paragraph below claiming this
> comparison "is admissible in the way the cross-study covariance columns were not" is false by the
> repo's own later rule. Kept in place, as this page keeps withdrawn claims — but it read as a live
> conclusion, and as the *headline* of a win, for **more than a thousand lines** with nothing
> pointing forward. (No line number here on purpose: an earlier version of this banner gave one,
> and the edit that added it moved the headings it counted between.)

Sky job 67 closed Evotuning and Evotuning-with-forced-substitutions, so all six of §4.2's methods now
exist. The evotuned trunk reached `eval_loss` **0.28149** in 2,000 steps (155.9 eval samples/s, against
job 58's 0.638 before the CUDA revert was fixed).

Ty1 only, because `evotune.family` is Ty1 — one family, one reference size, one eval config, so this
comparison is admissible in the way the cross-study covariance columns were not — **with one
qualification the next section forces.** The editor row's three per-position cells come from
`metrics/agree/agree2-ty1.json` (L=115) while both baselines are at L=121, so covariance, MIP and
positional JS below are compared across coordinate systems. Read in a single frame
(`metrics/aligned/aligned-ty1.json`, L=121) the leads become covariance +0.0269 / +0.0756, MIP
+0.0619 / +0.1599, and positional JS **−0.000045 / −0.002433** — the first of those a tie rather
than a win. Edits, pooled pairwise and MMD do not depend on the alignment and stand as printed.

| method | edits | pairwise pooled | covariance | MIP | JS positional | MMD |
| --- | --- | --- | --- | --- | --- | --- |
| our EvoFlows port | 4.71 | 26.22 | **0.8191** | **0.6871** | **0.01448** | **1.5553** |
| Evotuning | 3.84 | 25.16 | 0.7868 | 0.6592 | 0.01713 | 1.5851 |
| Evotuning (forced) | 4.00 | 27.05 | 0.7380 | 0.5612 | 0.01952 | 1.7442 |
| random mutations | 4.83 | — | — | — | — | 2.0150 |
| random pairing (ceiling/floor) | 24.89 | 23.48 | 0.9625 | 0.9359 | 0.00562 | 0.7166 |

**Two files, and the row says so.** `metrics/agree/agree2-ty1.json` (L=115) carries only four keys
in its `baselines.random_homolog_pairing` block — edits 24.89, MMD 0.7166, KL and entropy delta —
and no `agreement_reference_n`. The other four printed cells (23.48 pooled pairwise, 0.9625, 0.9359, 0.00562)
are the same baseline in `metrics/aligned/perseq-ty1.json` at **L=121**, with
`agreement_reference_n` 200. That puts this row's per-position cells in the same L=121 frame as the two
Evotuning rows, and so under the same qualification stated above — not in the editor row's L=115.
An earlier revision printed 23.67 / 0.9725 / 0.9348 / 0.00500 there instead: those are not this
baseline at all but the **drafted Ty1 ceilings** this file retracts below ("Three values for MIP are
now in circulation"), built against a 400-member ceiling set. The row is the baseline; the ceilings
are a different construction and live in their own table.

**Our port leads on every quality metric against both variants**: covariance +0.0323 and +0.0810,
MIP +0.0279 and +0.1258, positional JS −0.00266 and −0.00504, MMD −0.0298 and −0.1889. Only MMD is
alignment-independent; in a single frame the positional-JS lead over the unforced baseline is
−0.000045, a tie. See the qualification above.

**And the confound runs against us, which is what makes it worth reporting.** The budgets are not
matched — our port spends 4.71 edits against Evotuning's 3.84, 23% more — and more editing moves a
set further from the template, which normally costs distributional quality. So our port wins while
carrying the handicap.

Contrast the paper's own ordering, where Evotuning beats EvoFlows on MMD (0.687 against 1.128, the
two methods' `panel_8` means) while editing *less*. The 4.59-against-6.19 edit figures quoted here
are from the write-up's own reading and do not correspond to any statistic of the current
`panel_0` — whose method means are 13.46 and 32.14 after the 2026-08-28 recalibration — so treat
the ordering as the claim and the two edit numbers as superseded. Both comparisons are budget-confounded; ours is confounded
against its winner and theirs in favour of it. Their result is not wrong for it — their caption
concedes the undershoot — but ours is the cleaner comparison of the two, and
it is the one place so far where being unable to match budgets exactly works in our favour rather
than against.

What this does **not** say: nothing here compares our port to the paper's EvoFlows. Those numbers are
at different reference sizes on different datasets, and for covariance and MIP that comparison is
inadmissible outright (see the section on agreement scores across studies).

## The editor aligned to a different template than every baseline, and nobody read the field that said so

`generation_eval` projected its per-position metrics onto `chosen[0][0]`; `evotune_baseline` and
`evodiff_msa_baseline` both project onto `split.templates[0]`. On Ty1 that is **L=115 for the editor
against L=121 for both baselines**. Covariance, MIP and positional JS are all per-position, so every
comparison between the editor and a baseline was between different coordinate systems.

Each run recorded its own alignment — `agree2-ty1.json` says "L=115", `evotune_baseline.json` says
"L=121" — and no metric value looked wrong. The numbers were individually correct and jointly
meaningless.

**`chosen[0][0]` was unstable in a second way.** `chosen` is `rng.sample(held_out, n_templates)`, so
the alignment moved with the seed *and* with the template count. Two editor runs at different
template counts sat in different frames, so our own per-position numbers were not comparable with
each other either — which means no amount of internal consistency checking would have surfaced it.

Fixed to `split.templates[0]` whenever there is a family: deterministic given the split, and matching
all three baselines. The guard asserts every evaluator names it.

### What the bug cost, measured

The re-run at L=121 / L=122, with the ceiling now computed **inside the run** in the same alignment as
the score:

| metric | withdrawn | corrected | shift |
| --- | --- | --- | --- |
| covariance, % of ceiling | 85.1 | **84.3** | −0.8 |
| MIP, % of ceiling | 76.5 | **78.9** | +2.4 |

MIP's +2.4 is about four times its own sampling error, and it ran *against* us — the bug had been
understating our coevolution reproduction. Covariance's −0.8 is inside the interval and is not a
move. That asymmetry is the same one the construction sweep found: covariance's ceiling is
insensitive to reference size, ceiling-set size and alignment alike; MIP's is sensitive to all three.
One defect therefore produced an undetectable error on covariance and a material one on MIP.

**The fix validates itself.** The alignment-free columns came back identical to every digit — edits
4.80, pairwise pooled 25.24, MMD 1.3620, KL 0.001529. Only projected metrics could move, and only
projected metrics moved.

**Three values for MIP are now in circulation and only one describes a coherent run.** 76.5 was the
original, normalising an L=115/127 score against an L=121/122 ceiling. 77.4 was a recomputation that
was correct *for the old runs*, normalising L=115/127 against L=115/127 at ceiling-set 400. **78.9 is
the fixed run.** Only the last has a numerator and denominator built the same way.

## Three structural checks that need no ground truth, and the pattern they all catch

Every serious measurement error this project has caught was found by a *structural* check rather than
by a plausibility check. Not "is this number in a sensible range" — all of them were — but "does this
number stand in the relation to that one that construction requires". Recorded as checks to run,
because each has now paid for itself.

**1. A quantity that must AGREE and does not.** Random pairing pairs x0 with a random homolog of its
own family, so its average distance to x0 *is* an average pairwise homolog distance. Figure 3's panel
1 said 117.13; panel 0 said 8.47. That 13.8x gap was a log-axis calibration bug which had left every
value in a plausible range and correctly ordered all six methods. Only the cross-panel identity could
see it. The regression test is the identity, not the corrected number.

**2. A quantity that must VARY and does not.** A clock sweep at 25, 40 and 60 reported the
random-pairing floor as **22.40 at every point**, to the digit. Denominators identical across points
that were supposed to differ means those points share a cached artefact — and they did: three were
cached from before the reference-template fix (L=127) while three new ones were post-fix (L=121), so
the curve mixed two alignment frames and the jump across the boundary was partly the denominator
moving. This tell was visible in the table at the moment it was assembled and read as *reassuring*.

**3. A nuisance parameter that moves with the seed must be PAIRED away, not averaged through.** The
alignment length is drawn from the seeded split: L=121, 118, 124 on Ty1 and 122, 120, 128 on
HER2-VH for seeds 0, 1, 2, and constant for a given seed across every clock. Averaging across seeds
put a **0.18–0.86** spread on spectrum MMD and buried a real effect (~0.3 is the Ty1 half alone;
HER2-VH runs 0.50–0.86 and widens with the clock); pairing within seed showed an interior
minimum at clock 80 in **5 of the 6 family-seed curves**, monotone on both flanks — the exception,
HER2-VH seed 2, falls monotonically to its minimum at clock 120. Check that the nuisance parameter is fixed within a stratum before pairing,
then pair.

### The pattern: the metric I was watching improved

Both of the two worst near-misses share one shape, and naming it is more useful than the incidents.

* Assembling a clock curve from cached artefacts, the floors came back identical across three points.
  That looked like *consistency* — exactly the thing one hopes for — and it was evidence that three
  of the six points were not independent measurements at all.
* Cutting the paper to a page limit, a slice whose two indices resolved to the same position deleted
  the Discussion's two main findings. The page count fell by exactly the amount being optimised for,
  so the operation read as success. Found two hours later by a paragraph-size audit, not by the page
  count.

In both cases the number under observation moved the right way for the wrong reason. The defence is
not more care; it is to watch a quantity that the *failure mode* would move, rather than the one the
*goal* moves. A page-count edit is checked by section sizes. A sweep is checked by whether its
denominators vary.

## Own-study normalisation works for some metrics and not others, and which is predictable

Three metrics have now been normalised by an own-study floor or ceiling to cancel the reference-set
size so that two studies could be compared. Two failed and one held. The first draft of this section
recorded it as "three tried, three failed" and that was wrong — the paper session tested the third
and it passed, which turns a warning into a rule.

The paper session had put a cross-study comparison in Results and Conclusions: their EvoFlows sits
at 2.62x its own MMD floor on the two families we share, ours at 2.09x, so our port is *closer* to
its data distribution than theirs is to its own. (Their 2.62x is per-dataset and comes from the
write-up's Figure 5 extraction, which is not committed here: `metrics/evoflows_figure3.json` loses
point identity across panels, so it cannot be rederived from this repository.) The defence for comparing across studies at all was
that dividing by each study's own floor cancels the reference-set size. Tested, from artefacts that
already existed at two reference sizes:

| family | ref n | editor | floor | ratio |
| --- | --- | --- | --- | --- |
| Ty1 | 200 | 1.5553 | 0.7166 | 2.1703 |
| Ty1 | 800 | 1.5054 | 0.4618 | **3.2600** |
| HER2 VH | 200 | 1.1688 | 0.5831 | 2.0046 |
| HER2 VH | 800 | 1.0022 | 0.3979 | **2.5189** |

The paired mean moves from 2.0874 to 2.8895, **+38.4%**, over a 4x change in a parameter their paper
does not state. That is not a widening interval, it is a reversal:

| | ours | theirs | reading |
| --- | --- | --- | --- |
| at ref n 200 | 2.09 | 2.62 | we are closer |
| at ref n 800 | 2.89 | 2.62 | we are further |

**The mechanism is the denominator, and it makes the direction predictable.** Ty1's floor falls 36%
(0.7166 to 0.4618) while its editor moves 3% (1.5553 to 1.5054). Random homolog pairing is natural
against natural, so its MMD shrinks as the reference sample grows and the two samples converge on
one distribution. The editor's distance is dominated by a real distributional difference that more
reference data does not shrink. So a study with a larger reference set reports a larger ratio purely
as an artefact, and a study that does not state its reference size cannot be placed on the scale by
anyone.

The claim was withdrawn from Results and Conclusions rather than softened.

**What survives.** The nuisance parameter is constant *within* a study, so within-study comparisons
are untouched. Their six datasets not being one population — 2.86x on their three antibody sets
against 1.17x on their three enzyme and growth-factor ones, either side of the **1.73x** their
six-dataset mean gives — stands as a finding about reading their Figure 3 as a single number. Only
the 1.73x is rederivable here (a ratio of `panel_8` means); the antibody/non-antibody split needs
the per-dataset identity that Figure 3's strip plots lose, so those two come from the write-up's
Figure 5 extraction. Every within-study comparison of ours stands for the same reason. Only "ours versus theirs"
dies.

### The edit ratio passes the same test, and the difference is predictable

Same four artefacts, same arm, same alignment, `levenshtein_to_template` instead of MMD:

| family | ref n | editor | floor | ratio |
| --- | --- | --- | --- | --- |
| Ty1 | 200 | 4.715 | 24.885 | 0.1895 |
| Ty1 | 800 | 4.715 | 25.505 | 0.1849 |
| HER2 VH | 200 | 4.888 | 23.345 | 0.2094 |
| HER2 VH | 800 | 4.888 | 23.435 | 0.2086 |

Paired mean moves **−1.4%** where MMD moved **+38.4%** on the identical files. So the cross-study
edit comparison survives, and now on measurement rather than on assumption.

**State it at one scope, and say which.** Ours is 0.1995, the mean of the two families' ratios.
Theirs must be read at the same scope to be compared with it: on the two families we share it is
0.4278, the mean of their anti-SARS-CoV-2 VHH 0.480 and anti-HER2 scFv 0.376, from the paper
session's Figure 5 extraction. Their **0.27** is a different quantity — the six-dataset mean from
Figure 3's panel 0, 32.14 / 117.90 — and pairing our two-family number against their six-dataset one
mixes scopes in exactly the way this section is about. The first draft of this paragraph did that.
Both of their numbers are correct; only one is comparable with ours.

Note also that both of theirs are ratios of means, and a paired comparison calls for the mean of the
ratios. 0.4278 is the paired reading; the ratio of means at that scope is 0.40. Ours is 0.1995 under
either construction, because it is already a mean of per-family ratios.

**The difference is not luck, and it can be reasoned about before measuring.** Spectrum MMD is a
sample-size-*biased* estimator of a distributional distance: its floor is natural against natural, so
the floor shrinks as the reference grows while a genuine difference does not, and the ratio inherits
the bias. Mean pairwise distance within a family is a property of the family, not of the sample drawn
from it, so there is no bias to inherit.

The artefacts show this directly, and more starkly than the ratios do. The editor's
`levenshtein_to_template` is *identical* at both reference sizes — 4.715 and 4.888 to three decimals
— because distance to the template never touches the reference set. Only the floor moves at all, by
2.5% and 0.4%, which is sampling noise rather than bias. Under MMD *both* terms depend on the
reference set, and the floor moves 36%.

**The rule, worth stating in the paper.** Own-study normalisation is not a substitute for a stated
reference size, and whether it rescues a given metric has to be *measured per metric* rather than
assumed once. The prior that predicts the answer is whether the metric is a sample-size-biased
estimator: if it is, normalising by an own-study baseline inherits the bias instead of cancelling it.
When a generative-protein-model paper reports a distributional metric without its reference-set size,
that number cannot be compared to anyone else's — and normalising does not rescue it unless the
metric has been shown to be size-invariant.

## Agreement scores are not comparable across studies, and normalising does not rescue them

The results table wanted a covariance and an MIP column comparing our editor to the paper's
EvoFlows. Both are agreement scores, both depend on the reference set's size, and the defence was
to report each as a fraction of its own real-homolog ceiling, so that the size would cancel. It does
not cancel. Tested rather than assumed, and the assumption is false.

A real-homolog set stands in for a good generator and a mutated set for a poor one, so no model is
needed; the question is whether the *ratio* between them is stable in reference size. Ty1, 400
sequences per side:

| ref n | cov good | cov bad | ratio | MIP good | MIP bad | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.8102 | 0.7846 | **96.8%** | 0.6353 | 0.3653 | **57.5%** |
| 50 | 0.8770 | 0.7689 | 87.7% | 0.8049 | 0.3908 | 48.5% |
| 100 | 0.9228 | 0.7558 | 81.9% | 0.8757 | 0.4159 | 47.5% |
| 200 | 0.9571 | 0.7289 | 76.1% | 0.9309 | 0.4335 | 46.6% |
| 400 | 0.9738 | 0.6917 | 71.0% | 0.9616 | 0.4314 | 44.9% |
| 800 | 0.9761 | 0.6853 | 70.2% | 0.9702 | 0.4340 | 44.7% |
| 1600 | 0.9737 | 0.6683 | **68.6%** | 0.9702 | 0.4293 | **44.2%** |

A 28-point drift on covariance — wider than any model difference measured in this project — and 13
points on MIP.

It is the estimator, not one test set. Holding the reference sizes fixed and varying how degraded
the poor generator is gives the same shift every time:

| degradation | ratio at n=200 | at n=800 | shift |
| --- | --- | --- | --- |
| 2 substitutions | 77.9% | 71.7% | −6.2 |
| 5 substitutions | 76.3% | 70.5% | −5.8 |
| 10 substitutions | 74.4% | 68.1% | −6.3 |
| 30 substitutions | 76.5% | 70.2% | −6.3 |

**Mechanism, and it is why the sign has to be this way.** At small n the sampling noise in the
reference dominates both comparisons, so a good and a bad generator resemble each other; only a
large reference separates them. A small reference therefore **flatters a poor generator**, and the
effect is on the ratio, not just on the level.

**Which means the bias ran in our favour.** Our runs score at n=200; the paper's own values reach
covariance 0.9878, which the earlier saturation curve puts well above n=400. So the 85.1% and 76.5%
this table then carried were inflated relative to their 98.4% and 95.5%, and the gap we were about to publish was a **lower
bound** on the real one. The −6 is not applied as a correction: it was measured on synthetic
degradations rather than on the editor, so the direction is established and the magnitude is not. A
six-point adjustment would look like precision and be a guess wearing a number.

**What survives as cross-study.** Edits, spectrum MMD and pooled pairwise. Those rest on the
structural argument — MMD is not an agreement score, so it has none of the reference-size failure
mode above — plus one control, and **the control is weaker than this file first reported**. The
random-pairing MMD reads **0.6527** for the paper against **0.5348** for us in the frame the
reported table uses (`metrics/disjoint/editor-*.json`, `baselines.random_homolog_pairing`: 0.5966 Ty1
and 0.4730 HER2-VH). That is 18% apart. The **0.6498** first quoted here, 0.4% from the paper, is the
mean of `metrics/aligned/perseq-*.json`, whose frame `consolidate` labels `train_overlap_113` /
`train_overlap_103` — the leaked reference the disjoint rebuild exists to replace. Corrected here and
in `docs/reproducing.md`; the licence for normalising is the structural argument, and an 18% gap on a
model-free control is the size of difference a cross-study MMD comparison cannot resolve.

Covariance and MIP are within-study only, reported beside `agreement_reference_n`, and never against
a source whose reference size is unknown. The paper's Figure 3 is exactly such a source: it states
no split sizes anywhere (see the section on its undefined quantities). The warning lives in
`matrix_agreement`'s docstring, because that is where someone reaches before using it.

**Second time in this project that an assumption proposed for marking turned out false when tested.**
The first was the reference size itself, where recording `agreement_reference_n` looked sufficient
and the ceiling turned out to be needed. The rule that follows is not about care: an assumption
carrying a headline number is a task and should be sized as one, because a marked assumption
preserves the claim while a tested one can remove it.

## Our Gillespie is a deviation with a stated alternative, not an ambiguity we inherited

`next_event_time`'s docstring claimed that on the time-inhomogeneity of the rate field "the paper
says only 'next-event' and does not resolve it". It resolves it precisely, and we had not read it.

EvoFlows §3.3 eq 9 gives the first-event CDF as

    P(tau <= T | x_tn) = 1 - exp( int_{t_n}^{t_n+T} u_s(x_tn | x_tn) ds )

and eq 10 draws `U ~ Uniform(0,1)` and **numerically integrates** until

    log(U) - int_{t_n}^{t_n+Delta_n} u_s(x_tn | x_tn) ds = 0

The state is held at `x_tn` while `s` varies, so the rate's time-dependence is integrated. We freeze
the rate at the current `(x, t)` and draw from that constant total, which is exact only if the
generator is read as piecewise-constant between events.

So this is a fourth deviation, and it had been filed under the wrong heading — as something the
source left open rather than as a choice of ours against a stated alternative. That is the opposite
of the error this project keeps making: usually we over-read the paper's precision, here we
under-read it, and the effect was to make one of our approximations look forced.

**What it costs is bounded by the quantity that already bounds Euler's error** — the rate's variation
over one waiting time. Euler and Gillespie agree to 0.6-1.6% at the clocks we run, and both
approximations are first order in the same small parameter, so that agreement bounds the frozen-rate
error over the same interval. Closing it exactly means one numerical integration per event.

Worth recording what the paper says about the relationship, because it reframes why the method exists:
"this method is formally very close to the Euler integration proposed by Havasi et al. (2025) but
replaces repeated Bernoulli variable sampling with a single uniform sampling. This approach has the
practical benefit that it does not require adding potentially large sub-leading terms." So §3.3 is
not presented as an exactness improvement over Euler at all — it is presented as avoiding a
correction term. Our earlier framing of Gillespie as "the exact one" and Euler as "the approximate
one" is not the paper's framing.

## Normalising to the ceiling does NOT cancel the reference size, measured on our own editor

The claim above -- "the unstated split stops blocking the comparison once both sides are expressed as
a fraction of their own ceiling" -- is now measured rather than assumed, by re-running the same editor
at `--holdout-size 800` against the 200 every previous run used. Nothing else changed: same
checkpoint, same 20 templates x 20 variants, same seed, same `split.templates[0]` alignment, and the
ceiling set held at its own 300 so that only the reference moved. Sky jobs 79 (Ty1) and 78 (HER2-VH);
`metrics/aligned/h800-*.json` sits beside the 200 in `metrics/aligned/aligned-*.json`.

| | Ty1 @200 | Ty1 @800 | HER2-VH @200 | HER2-VH @800 |
| --- | --- | --- | --- | --- |
| covariance / ceiling | 0.8137 / 0.9694 = 83.9% | 0.7785 / 0.9846 = **79.1%** | 0.8120 / 0.9600 = 84.6% | 0.8107 / 0.9825 = **82.5%** |
| MIP / ceiling | 0.7211 / 0.9280 = 77.7% | 0.7428 / 0.9608 = **77.3%** | 0.7407 / 0.9258 = 80.0% | 0.7918 / 0.9581 = **82.6%** |
| positional JS | 0.01709 | 0.01453 | 0.01548 | 0.01388 |

Two-family means: **covariance 84.3% -> 80.8%** (-3.5 pp), **MIP 78.9% -> 80.0%** (+1.1 pp).

**The ratio is not size-invariant, so these cells carry a reference size or they carry nothing.** A
four-fold reference moves the covariance cell by 3.5 pp -- roughly six times the ceiling's own
+/-0.5-0.6 pp sampling error -- and it moves the two metrics in *opposite* directions. The paper's own
figures imply a reference well above 400 (their random-pairing covariance of 0.9878 needs several
hundred), so on the covariance row our 84.3% at n=200 flatters us relative to what their regime would
have reported, by more than the whole alignment fix moved it.

**The raw scores move in opposite directions, and that is the finding.** Covariance *falls* with a
larger reference (-0.0352 Ty1, -0.0013 HER2-VH) while MIP *rises* (+0.0217, +0.0511); both ceilings
rise (covariance +0.0152 / +0.0225, MIP +0.0327 / +0.0322). Covariance's ceiling rises faster than its
score, MIP's slower. The pattern replicates in a second coordinate system: the same two families at
holdout 200 against 800 in the *old* `chosen[0][0]` alignment (`metrics/agree/agree2-*.json` against
`metrics/agree/h800-oldalign-*.json`, jobs 65/64 against 74/75) give covariance -0.0227 / -0.0115 and
MIP +0.0335 / +0.0516 -- same signs, four independent runs, so it is not an alignment artefact.

### The synthetic stand-in predicts covariance and does not predict MIP

The reference-size sweep that motivated this run used real Ty1 homologs as a stand-in for a good
generator and a mutated set for a poor one, and predicted that the score/ceiling ratio falls as the
reference grows: covariance 76.1% -> 70.2% and MIP 46.6% -> 44.7% between n=200 and n=800.

- **Covariance: the direction holds.** -4.9 pp (Ty1) and -2.1 pp (HER2-VH) against the sweep's
  -5.9 pp. Same sign in both families, same order of magnitude on Ty1.
- **MIP: the direction does not hold.** Ty1's -0.4 pp is smaller than the ceiling's own sampling error
  and is not a move at all; HER2-VH goes **+2.6 pp**, against the sweep's -1.9 pp.

So the mutated-set stand-in models our editor's covariance behaviour and does **not** model its MIP
behaviour: the sweep cannot be used to reason about the MIP cell, only a re-run can. Third finding in
a row with this shape -- covariance is insensitive to every construction choice (alignment,
ceiling-set size, and now reference size in one family of the two), MIP is sensitive to all of them. A
metric that moves is not a worse metric, but it is one that must never be reported without the
construction that produced it.

### What did and did not move, as a control

The alignment-free columns behave exactly as they must. `levenshtein_to_template` (4.715 / 4.8875),
`pairwise_levenshtein` (8.3321 / 8.8242) and `pairwise_levenshtein_pooled` (26.2195 / 24.2625) are
identical to every recorded digit at both holdouts -- they are properties of the generated set alone,
and the generated set is bit-identical between the two runs. So are `entropy_delta` and
`profile_log_likelihood`, which deliberately keep the 20-sequence partner reference. `spectrum_mmd`
(1.5553 -> 1.5054, 1.1688 -> 1.0022) and `kl_generated_vs_natural` (0.002445 -> 0.002109, 0.000614 ->
0.000415) both moved, which is correct: both are scored against the holdout.

**The two MMD readings are not comparable, and the runs say so.** `mmd_diagnostic.n_reference` is 200
in one and 800 in the other, and the biased V-statistic is not comparable across reference sizes.
Read the pair as evidence that MMD responds to the holdout, not as an improvement. The size of that
response is **0.050** on Ty1 and **0.167** on HER2-VH, against a six-method spread at fixed
reference of 0.809 and 0.904 — so, unlike an earlier framing of this sentence, the reference axis
does NOT move the number further than the model does. The `random_homolog_pairing` baseline's edit distance also moves (24.885 -> 25.505 on
Ty1) because that baseline draws its partners from the holdout by construction; `random_mutations`,
which draws from nothing, is identical to every digit.

### The knob that had to be built before any of this could be measured

`generation_eval` had taken `--holdout-size` since the agreements were fixed, but no managed job
could reach it: `train.sky.yaml`'s geneval block never forwarded one. And the knob alone would not
have worked -- `aligned_natural_full` sliced the holdout at a hard-coded `[:300]`, so a run asked for
800 would have measured at 300 and *recorded* `agreement_reference_n: 300`, which is the honest
failure mode but not the requested measurement. `evotune_baseline.score_generations` had the same cap
on the same quantity, which would have put a baseline at 300 against an editor at 800 -- numerator and
denominator built differently for the fifth time in this session. Both caps now yield to an explicit
holdout; both are unchanged at any holdout <= 300, i.e. for every run recorded before this one.

## The MMD claim died to seed noise, and the schedule question cannot be answered from loss

Two things the results table rested on, neither of which had been measured until asked for.

### "Better k-mer distribution than evotuning" is not supported

`metrics/matched/` holds our editor's four arms on both families at three seeds each — 24 runs — so the
across-seed spread of MMD was already available:

| arm / family | MMD mean | MMD sd | MMD range |
| --- | --- | --- | --- |
| A / HER2 | 0.5720 | 0.0275 | 0.0546 |
| A / Ty1 | 0.5078 | 0.0131 | 0.0246 |
| B / HER2 | 0.5555 | 0.0147 | 0.0259 |
| B / Ty1 | 0.5067 | 0.0335 | 0.0656 |
| C / HER2 | 0.5274 | 0.0293 | 0.0584 |
| C / Ty1 | 0.4890 | 0.0399 | 0.0798 |
| D / HER2 | 0.7277 | 0.0204 | 0.0402 |
| D / Ty1 | 0.6489 | 0.0407 | 0.0807 |

**Median across-seed sd 0.0284, against a claimed gap of 1.5851 − 1.5553 = 0.0298.** One standard
deviation, with ranges reaching 0.08. And on HER2-VH the same gap is **0.004** — an order of
magnitude below the noise. So the supported sentence is "indistinguishable on k-mer distribution",
not "better".

**The KL side survives.** Seed sd runs 1.8e-5 to 2.9e-4, median 6.4e-5, against a gap of 3.7e-4 —
roughly six times the spread. The composition *loss* is real where the k-mer *win* is not, and the
asymmetry is worth noting: the claim that survived is the one against us.

Caveat on those seeds: they are the 300x1 config with MMD near 0.5, while the claim is at 20x20 with
MMD near 1.55. MMD's variance depends on both sample size and value, so this bounds the noise without
measuring it, and 0.0284 is not the error bar for the table's cells.

### Two runs of the same schedule differ by more than some method gaps

`sched-linear` (job 76) and the reported `faithful-appendixa` are both linear-schedule, same trunk,
same heads, same 20,000-step budget — replicates of one configuration:

| | Ty1 cov | Ty1 MIP | HER2 cov | HER2 MIP | Ty1 MMD |
| --- | --- | --- | --- | --- | --- |
| sched-linear | 0.8038 | 0.6868 | 0.8051 | 0.7235 | 1.5453 |
| faithful-appendixa | 0.8137 | 0.7211 | 0.8120 | 0.7407 | 1.5553 |
| difference | −0.0099 | **−0.0343** | −0.0069 | −0.0172 | −0.0100 |

They are not bit-identical runs — different hardware (A100 vs L4) and possibly a different stopping
step — so this conflates training nondeterminism with hardware. But it is the right order of
magnitude for "how much does one configuration move between runs", and **0.0343 on MIP is larger
than several differences we have discussed between methods.** The cubic arm trains on the same A100
as `sched-linear`, so the schedule pair itself is free of the hardware term; this pair sets the scale
a schedule effect must beat.

### The training losses of the two schedules are not comparable

The loss weights each supervised edit by `kappa_dot / (1 - kappa)`, and that differs between
schedules at every t:

| t | linear | cubic | ratio |
| --- | --- | --- | --- |
| 0.25 | 1.333 | 0.190 | 0.14 |
| 0.50 | 2.000 | 0.857 | 0.43 |
| 0.75 | 4.000 | 2.919 | 0.73 |
| 0.90 | 10.000 | 8.967 | 0.90 |

Seven-fold at t=0.25. So a cubic run showing a lower training loss than a linear one says nothing
about quality — it is a different objective, not a better model. The schedule question is answerable
only by a held-out generation eval of both checkpoints, which is why both arms get one.

### Our random-mutations floor sits below the paper's, by an amount that depends on the frame

In the **leaked** frame (`metrics/aligned/perseq-ty1.json`, reference n=200): covariance **0.7548**
and MIP **0.5265**, against the paper's six-dataset Random means of **0.8970** and **0.6598** — 14.2
and 13.3 points below, same direction in both columns. That is where the "13-14 points" comes from.

In the **reported disjoint** frame the gap is much smaller and one column changes sign: covariance
**0.8483** (4.9 points below) and MIP **0.6617**, which is 0.2 points *above* the paper's. So the
conclusion below does turn on which frame is used, and the disjoint numbers only support the
weaker form of it — that a floor is not a fixed quantity either, not that ours sits well below
theirs.

That closes a loop. A reader who accepts that our method rows cannot be compared with the paper's
might still assume the floors can be, since a floor is "just random". They cannot.

Random pairing does **not** confirm that reading in the reported frame: there it scores *above* the
ceiling on all four measures (Ty1 covariance 0.9751 against a 0.9534 ceiling, MIP 0.9422 against
0.9317), because 400 drawn variants and a 300-homolog ceiling set put the two estimators at
different effective sample sizes. Read it as §4.2's unbudgeted arm — real homologs, a reference
point rather than a bound. `docs/reproducing.md` carries the full retraction; the 0.9625 / 0.9694
and 0.9359 / 0.9280 figures that used to appear here are Ty1-only and from the leaked frame.

And the sharpest version of the n-dependence: at n=20 the floor and ceiling **invert** — random
mutations 0.411 against real homologs 0.395. At that reference size uniform random substitution looks
*more* like the family than actual homologs do. These metrics do not merely lose resolution at small
n; they can rank backwards.

## The schedule deviation costs nothing measurable — the last one, closed

Edit Flows states a cubic scheduler `kappa_t = t^3` in its experiments (p9) and ships `kappa = t` in
its Figure 13 training code. We use linear, so which half of the paper we deviate from depends on
which half is read. Both arms trained 20,000 steps on the same A100, identical but for the schedule,
then evaluated held out on both families.

| arm | family | edits | pairwise | covariance | MIP | positional JS | MMD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear | Ty1 | 4.59 | 26.04 | 0.8038 | 0.6868 | 0.01702 | 1.5453 |
| cubic | Ty1 | 5.30 | 26.52 | 0.7988 | 0.6798 | 0.01703 | 1.5823 |
| linear | HER2-VH | 5.17 | 24.35 | 0.8051 | 0.7235 | 0.01504 | 1.1734 |
| cubic | HER2-VH | 5.53 | 24.92 | 0.8113 | 0.6876 | 0.01543 | 1.1947 |

Read against the control that two runs of the **same** configuration differ by 0.0343 in Ty1 MIP:

* Ty1: covariance −0.0051, MIP −0.0070. Both well inside the control.
* HER2-VH: covariance **+0.0062** — opposite sign to Ty1 — and MIP −0.0358, barely over the control
  on one family only.
* MMD +0.0370 and +0.0212, against an across-seed sd of ~0.028. About one sd, and less.

**And the arms are not budget-matched: cubic edits 7–16% more at the same clock** (+0.71 on Ty1,
+0.37 on HER2-VH). More editing costs distributional quality, so cubic's slightly worse MMD is
plausibly its higher edit count rather than the schedule. That the schedule changes the realised edit
count at fixed clock is itself worth recording — it means §4.2's matched-mutation requirement is not
automatically satisfied between two schedules, and a comparison that matched clocks would still be
comparing different budgets.

So: **no schedule effect distinguishable from run-to-run variation**, signs inconsistent across
families, and what difference exists confounded by an edit-count change the schedule causes. The
deviation is retired as costing nothing measurable, which is a different claim from "the paper was
wrong to state cubic" and is the only one the data supports.

Recorded for whoever revisits it: the two arms' **training losses were never comparable**, because
the loss weights each supervised edit by `kappa_dot / (1 - kappa)` and that differs at every t — by a
factor of seven at t=0.25. Only the held-out evaluation could answer this, which is why both arms got
one rather than a loss-curve comparison.

## Wall-clock and step rates, from the jobs that produced the schedule comparison

Recorded because they were quoted in a brief before they existed anywhere in the repo, and the agent
receiving that brief correctly refused three of them and caught a fourth as wrong. The numbers below
are read from the managed-job queue and the jobs' own logs.

| job | what | hardware | job duration |
| --- | --- | --- | --- |
| 76 | edit-flow training, 35M, 20,000 steps, linear schedule | A100:1 on-demand | **3 h 43 s** |
| 83 | the same, cubic schedule | A100:1 on-demand | **3 h 31 m** |
| 84 | `generation_eval`, 20x20, holdout 200, HER2-VH | L4:1 spot | **12 m 19 s** |
| 85 | the same, Ty1 | L4:1 spot | **12 m 31 s** |

Job 76 reached step 19,990 of 20,000, so **~0.54 s/step for the 35M editor on an A100**, against the
~1.05 s/step measured for the same arm on an L4. So an A100 is roughly **2x** faster for this
workload — which is worth stating because `jobs-calibrate`'s comment warns that a bigger GPU cannot
fix a launch-bound loop, and that warning was measured on a different arm. It does not generalise to
this one.

It also did not help the schedule pair, because **europe-west4 is the only EU region with A100 in the
catalog**, so two arms could not run concurrently there. Sequential A100 (~7 h for the pair) is a wash
against parallel L4s (~5.8 h each), at roughly eight times the cost.

### Correction: the 650M limit is VRAM, not host RAM

An earlier brief of mine said the 650M evotune "OOMs an L4's 16 GB host RAM". That is wrong on both
counts. The Makefile records the actual constraint: stages needing **more than 24 GB** of GPU memory
require an A100, and "650M in bf16 needs ~16-18 GB". The L4 has 24 GB of VRAM, and the binding
constraint is the editor at ~20 GB in fp32 rather than host memory. No host-RAM OOM is recorded
anywhere in this repo.

The distinction matters for anyone sizing a machine: host RAM is cheap to add and VRAM is not.

## Regressions the test suite pins

Infrastructure failures, not measurements — recorded here because they used to live as
multi-paragraph docstrings, which is the wrong place to read them: a test docstring is read while
the test is failing, when the reader wants the invariant that broke, not the history. Each row names
the test that guards it; the docstring states the invariant and points here. **Some rows say
`no longer guarded here`**: the incident is real history, but the test that pinned it went with the
code it covered when this repository was split out of a larger one.

| what broke | how it showed up | guarded by |
|---|---|---|
| The hand-written `dvc pull` list for GPU jobs omitted a stage's cached dep. | Two managed jobs died on a fresh VM *after* a full provision + setup. The list drifted three times before it was derived from `dvc.yaml` instead — which is what `editjumps/repo/stage_data_deps.py` now does. | no longer guarded here |
| Checkpointing was epoch-only, so a run that died mid-epoch saved nothing. | Lost a ~9 h pretrain run. | `test_pretrain_esm_default_checkpointing_is_step_based` |
| `pretrain_esm`'s `metrics_path` was hardcoded. | Any ad-hoc run from the repo root overwrote the tracked metrics file. Only `dvc.yaml` should pass `--metrics-path`. | `test_pretrain_esm_metrics_path_defaults_to_none` |
| A finished training job left only `checkpoint.pt` in GCS — resume state, not `encoder/` + `evoflows_model.pt`. | Job 21 trained 6.5 h across a preemption and nothing could open what it produced. The round-trip was later verified against that checkpoint: 214 tensors, zero differing. | `test_training_job_ships_the_model_folder_not_only_the_checkpoint` |
| MLflow's tracking token was read once rather than per request. | A 401 an hour into a training run, once the minted identity token expired. | `test_mlflow_rereads_token_from_env_on_every_request` |
| A diverged run shipped its diverged weights. | With `lr=0.02` the loss went to inf and early stopping fired, but the restore guard was `best_val < final_val` and `31.978 < nan` is False — so 209 of 216 saved tensors were non-finite while the log said "best val_loss 31.978 at step 0, and those are the saved weights". | `test_nan_comparisons_cannot_silently_ship_diverged_weights` |
| MMseqs sizes itself to available RAM, and the full-corpus run had no error path. | Swapped a 24 GB machine to death twice, with no error and no crash report. | `test_local_capacity_guard_blocks_only_big_jobs_on_tight_machines` |
| Knobs reached `params.yaml` without reaching `dvc.yaml`, repeatedly: `rate_head`/`q_head`, then the `evotune` block, then `evodiff`. | An untracked param cannot invalidate a stage, so the pipeline keeps serving results computed from the old value. | `test_head_parameterisation_knobs_are_tracked_by_dvc`, `test_evotune_params_are_all_tracked_by_dvc`, `test_evodiff_params_are_all_tracked_by_dvc` |
| The Makefile drifted to ten direct `python -m editjumps.<...>` calls. | `editjumps/main.py` exists so the build has one discoverable entry point and the CLI cannot drift from `dvc repro`. | `test_makefile_recipes_go_through_the_editjumps_cli` |
| Configurable gap costs on the legacy Needleman-Wunsch re-derived traceback moves by float `==`. | IndexError on 144 of 6,000 random non-integer calls: `6*0.7 = 4.199999999999999` against `5*0.7 + 0.7 = 4.2`, so no branch matched and the bare `else` walked off the matrix. | `test_needleman_wunsch_takes_no_cost_parameters`, `test_affine_traceback_survives_non_integer_gap_penalties` |
| The MLflow logging loop read one key and nothing else. | A run recorded two point estimates while dropping the bootstrap CI that makes them a tie, and every per-panel n. A reader would have drawn the opposite conclusion from the same run. | `test_flatten_metrics_logs_every_number_including_n_and_the_ci` |
| Three evaluation paths called the ⑤ sampler with no `clock=`. | Their edit counts sat on a different scale from the two paths that passed one, and from §4.2's matched "expected number of mutations per sequence". | `test_every_evaluation_path_passes_the_clock_to_the_sampler` |
| The first version of the batched-forward test compared the two paths in `train()` mode. | It "failed" at 3% — ESM dropout draws different masks per path, which is a property of dropout, not of batching. | `test_batched_forward_gives_the_same_gradients_as_accumulating_one_at_a_time` |
| §4.1's alignment ceiling was first measured on random strings rather than real chains. | Hid a half-sequence indexing bug: the ceiling is a property of real residue composition. | `test_deterministic_benchmark_ceiling_holds_up_on_real_antibody_sequences` |
| A resumed run's learning-rate schedule could have restarted at step 0 of the warm-up. | A preempted spot run resumes many times, so a per-process schedule would spend the run re-warming — and a resume past the warm-up would jump the rate back UP. The rate is a pure function of the absolute step for that reason. | no longer guarded here |

## The Figure-3 comparison's denominator was never recorded, and one of its four cells is wrong

Independent recomputation of the four-row table in
"Normalising to the ceiling cancels the reference size", from the raw artefacts and not from that
section.

**The paper's column reproduces exactly.** Each cell is the ratio of the six-dataset mean of Figure
3's `EvoFlow (ours)` row to the mean of its `Random pairing` row, read out of
`metrics/evoflows_figure3.json`: covariance **0.98376**, MIP **0.95541**, positional JS **1.79626**,
pairwise **1.00268**. Rounded: 98.4, 95.5, 1.80, 1.00. All four confirmed.

One caveat that only shows up on recomputation: **the pairwise row is the one estimator-sensitive
cell.** Ratio-of-means gives 1.003, mean-of-ratios 1.036. The other three move by under 0.03 between
the two (98.38/98.36, 95.54/95.57, 1.796/1.769). So the single row where we are *not* behind rests on
the least robust of their four values — under mean-of-ratios their 1.00 becomes 1.04 and our
"slightly past the natural spread" reading nearly disappears.

**Our raw scores reproduce exactly too**, as family means over `metrics/agree/agree2-*.json`:
covariance 0.82166, MIP 0.71313, positional JS 0.015123, pooled pairwise 25.241.

**Our ceilings do not reproduce, because the artefacts behind those draft tables do not record
them.** `agree2-*.json` carries `agreement_reference_n` but **none of the three ceiling keys at
all** — absent, not null, as `metrics/agree/README.md` says: `generation_eval.py`'s
`covariance_agreement_ceiling`, `mip_agreement_ceiling` and `ceiling_n` postdate those two runs. (Every run
*since* carries them — `metrics/disjoint/`, `metrics/aligned/`, `metrics/schedule/` and the
`h800-oldalign-*` pair in this same directory all do.) Both draft tables therefore normalised to a
denominator computed by hand afterwards and kept nowhere.

Recomputed from the family FASTAs, at the one defensible reference size — **400 real held-out
homologs, the number of sequences the generated set actually has** — over 8 independent draws per
family. The split and seed reproduction was validated by rebuilding each run's own §4.3
random-pairing baseline and matching its recorded diversity to every digit (23.401 for HER2-VH,
23.479 for Ty1), so these ceilings sit on exactly the alignment and holdout the runs used.

| | recomputed | draft claimed | verdict |
| --- | --- | --- | --- |
| covariance, % of ceiling | **85.3 ± 0.6** | 85.1 | inside the range (84.7–86.1) |
| MIP, % of ceiling | **77.4 ± 0.6** | 76.5 | **outside all 8 draws** (76.8–78.3) |
| positional JS, × floor | **2.68 ± 0.10** | 2.75 | inside the range (2.53–2.82) |
| pairwise, × natural | **1.070 ± 0.011** | 1.07 | matches |

**Two things follow, and the second is the one that matters.** 76.5 is not reproducible: 77 is the
figure, which is what the paper's own prose already said while its table said 76.5. And the third
significant figure in that column was never supportable in either version — the ceiling's own
sampling error is ±0.6 percentage points, four times the difference being reported to the reader.

**Why the drafted ceilings were wrong — RETRACTED AND REPLACED.** A first version of this section
said the drafted ceilings had been taken at several different reference sizes. That was wrong, and a
colleague's reproduction of their own construction is what showed it. The six drafted values come
from ONE consistent construction: `split_family(members, 20, 200, 0)`, reference = `split.reference`
(n=200), ceiling set = `pool[:400]`. Re-run, it reproduces all three drafted cells to the last digit:

| | drafted ceiling | drafted cell | reproduces? |
| --- | --- | --- | --- |
| covariance | 0.9661 (HER2-VH 0.9597, Ty1 0.9725) | 85.05% → 85.1 | exactly |
| MIP | 0.9318 (HER2-VH 0.9287, Ty1 0.9348) | 76.53% → 76.5 | exactly |
| positional JS | 0.00552 (0.0060, 0.0050) | 2.739× → 2.75 | exactly |

**The actual defect is the alignment, not the sample size.** That construction takes
`split.templates[0]` as its coordinate system. The runs whose scores it normalises took
`reference_template = chosen[0][0]` — the first entry of the *seeded sample* of the same list, which
is not the same sequence. The lengths differ, and each artefact records which one it used in its own
`alignment` field:

| | drafted ceiling's template | the runs' template (recorded) |
| --- | --- | --- |
| HER2-VH | `split.templates[0]`, L=122 | `chosen[0][0]`, **L=127** |
| Ty1 | `split.templates[0]`, L=121 | `chosen[0][0]`, **L=115** |

So numerator and denominator were in different coordinate systems. Re-running the identical
construction in the runs' own alignment, changing nothing else:

| | drafted (L=122/121) | runs' alignment (L=127/115) |
| --- | --- | --- |
| covariance | 0.9661 → **85.05%** | 0.9654 → **85.11%** |
| MIP | 0.9318 → **76.53%** | 0.9230 → **77.26%** |
| positional JS | 0.00552 → **2.739×** | 0.00561 → **2.695×** |

Covariance moves 0.06 percentage points and survives; MIP moves 0.73 and does not. Almost all of it
is Ty1's MIP ceiling, 0.9348 at L=121 against 0.9193 at L=115, where HER2-VH's barely moves
(0.9287 → 0.9267).

**Per-family, per-draw working** for the recomputed numbers above — reference n=200, ceiling-set
n=400, eight independent draws of 400 from the pool, in the runs' alignment:

| | HER2-VH (L=127) | Ty1 (L=115) |
| --- | --- | --- |
| MIP ceiling per draw | 0.9293 0.9251 0.9246 0.9317 0.9334 0.9414 0.9364 0.9146 | 0.9158 0.9051 0.9054 0.9237 0.9239 0.9150 0.9034 0.9076 |
| mean ± sd | 0.9296 ± 0.0082 | 0.9125 ± 0.0083 |
| cov ceiling mean ± sd | 0.9563 ± 0.0079 | 0.9693 ± 0.0090 |

Family-averaged: covariance ceiling 0.9628 ± 0.0065 → **85.34 ± 0.58%**; MIP ceiling 0.9210 ± 0.0068
→ **77.43 ± 0.58%**; JS floor 0.00564 ± 0.00020 → **2.684 ± 0.095×**.

**MIP's ceiling carries a sensitivity covariance's does not**, and this is the better reason to
distrust the third significant figure. Across a 4× change in the size of the *ceiling set* alone,
reference held at 200, in the runs' alignment:

| ceiling-set n | HER2-VH cov / MIP | Ty1 cov / MIP |
| --- | --- | --- |
| 200 | 0.9577 / 0.9068 | 0.9717 / 0.9005 |
| 400 | 0.9547 / 0.9267 | 0.9760 / 0.9193 |
| 800 | 0.9499 / 0.9354 | 0.9776 / 0.9313 |

Covariance moves under 1% and not even in the same direction across the two families (−0.8% and
+0.6%); MIP moves +3.2% and +3.4%, consistently. So MIP's ceiling is sensitive to the ceiling set's
size, to the reference's size, and to the alignment, while covariance's is comparatively sensitive to
none of the three — which is why covariance's drafted cell happened to survive three separate
construction choices and MIP's did not.

**The fix is structural rather than arithmetical.** `score_set` now scores the §4.3 random-pairing
baseline on both agreements and the positional JS, inside the evaluator, projected onto the same
`reference_template` and against the same full reference the model's own score uses. A ceiling in
another alignment, at another sample size, or from another run is then not expressible. A test
asserts the projection specifically, because that is the part that was wrong and would not have been
caught by asserting the metric names alone.

## Euler's step-size error vanishes by n_steps=50, and its SIGN depends on the clock

`metrics/sampler_step_size.json`, from `editjumps/pipeline/evaluate/sampler_step_size.py`. CPU only and
model-free on purpose: the discretisation error is a property of the sampler, so running it through
trained weights would confound the two. Constant per-position rates (insert 1.0, delete 0.5,
substitute 1.0) on a length-120 start, 2000 trajectories per cell, Gillespie exact by construction.

**Clock 40 — the configuration every run in this project uses.** Exact: 71.29 ± 0.15 edits.

| n_steps | Euler − exact, edits | resolved? |
| --- | --- | --- |
| 2 | **+4.44 ± 0.20** (+6.2%) | yes |
| 5 | +1.48 ± 0.21 (+2.1%) | yes |
| 10 | +0.76 ± 0.21 (+1.1%) | yes |
| 25 | +0.10 ± 0.21 | no |
| **50** (pipeline default) | **+0.08 ± 0.21** | **no** |
| 100 | +0.05 ± 0.21 | no |

So at the setting the runs use, Euler is indistinguishable from the exact sampler at 2000
trajectories — the bound is 0.3% of the edit count. Reporting Euler runs costs nothing here, and
that is measured rather than assumed.

**The sign is not fixed, and a one-signed claim would be wrong half the time.** Same rate field,
clock off: exact 157.19 ± 0.35 edits, and Euler at n_steps=2 is **−4.03 ± 0.42** with mean length
−10.80. The mechanism is that Euler freezes the rate field for a whole step, so the error follows
whether the total rate is rising or falling along the trajectory: with the clock, the growing
sequence *lowers* `clock/len(x)` and Euler's stale larger value over-fires; without it, the total
rate is proportional to length and *rises*, so the stale smaller value under-fires. This is separate
from the `h·λ` clipping recorded above under "⑤ Gillespie sampler" — nothing here is near clipping.

## Spectrum MMD is not comparable across evaluation shapes — match n AND the template count

`generation_metrics.spectrum_mmd_estimators` reports the paper's **biased V-statistic** as its
headline, which is what the paper says it uses. The V-statistic keeps the diagonal `k(x, x)` terms —
the largest entries in an unnormalised k-mer inner product — so `MMD²` carries an `O(1/n)` positive
bias, so it is not comparable across sample sizes even in principle. Measured on one arm and one
family, over every cell of the sweep (`n_reference` fixed at 200 throughout):

| file | templates × variants | n_generated | MMD |
|---|---|---|---|
| `arm-B-c40-n100u` | 10 × 10 | 100 | 1.633 |
| `arm-B-c40-n200` | 20 × 10 | 200 | 1.107 |
| `arm-B-c40-n300` | 30 × 10 | 300 | 1.038 |
| `arm-B-c40-n500` | 25 × 20 | 500 | 0.995 |
| `arm-B-c40-n500u` | 50 × 10 | 500 | 1.117 |
| `arm-B-c40-n1000` | 50 × 20 | 1000 | 1.118 |

**An earlier reading of this table was wrong and is retracted.** Three of these cells — 100, 200 and
500 — were quoted as "a 0.64 swing from sample count alone". They vary the TEMPLATE count too
(10 → 20 → 25), and the template axis is what moves the number: the two cells at `n_generated = 500`
differ by 0.12 on templates alone, and at 50 templates doubling `n_generated` from 500 to 1000 moves
MMD by **0.0006**. The bias term is real theory; it is not what those three cells measured.

Two consequences, both load-bearing:

* **MMD is comparable between runs only at equal n on both sides.** Our own 35M-vs-650M comparison
  was invalid until they were matched (100 vs 500 generated): the apparent 1.7× win for the larger
  trunk shrank to 0.03 once matched.
* **The paper never states its sample sizes**, so a like-for-like comparison to its Figure-3 MMD
  values is not currently possible. Not because our pipeline is wrong — because the quantity depends
  on an unreported number. This is the first unstated detail we have found that blocks a comparison
  outright rather than merely requiring a guess, and it is a question for the authors.

The unbiased U-statistic drops the diagonal, estimating the same population quantity without the
`1/n` term, and is the right diagnostic for "is this difference real". It is returned as `MMD²`
rather than its root so a negative value survives; negative is not an error, it means `MMD²` is
within noise of zero.

This is also why `editjumps/core/family_split.py` exists as one shared function rather than a few lines in
each evaluator: at these magnitudes an editor and a baseline are only comparable if both were scored
against the same holdout at the same n.

## Alignment costs: why `GAP_COST` is a module constant and not a parameter

`needleman_wunsch`'s traceback re-derives each move by comparing accumulated costs with `==`. That
is exact for unit integers and unsound for anything else. The costs used to be keyword arguments; no
caller ever passed anything but the defaults, and they carried the only reachable crash in the
module.

Measured over 6,000 random calls with non-integer costs: **144 raised `IndexError`** (no branch
matches, so the bare `else` walks off the matrix) and **2 returned alignments that did not strip back
to their inputs**. Minimal case: `gap_cost=0.7` gives `6*0.7 = 4.199999999999999` against
`5*0.7 + 0.7 = 4.2`.

So configurable costs need a stored **traceback-pointer matrix**, which is what `_align_affine`
(Gotoh, affine gaps) does — a stored pointer cannot disagree with the move that produced it. That
removes the class of bug rather than documenting it, and it is why BLOSUM62 with a fractional gap
penalty goes through the pointer aligner and not the unit-cost one.

## The model-free baseline rows now carry the model's columns, and the floor sits below the paper's

The two model-free baselines reported four numbers each — `levenshtein_to_template`,
`kl_generated_vs_natural`, `spectrum_mmd`, `entropy_delta` — while the method rows carried three
per-position metrics on top (`covariance_agreement`, `mip_agreement`, `js_divergence_positional`)
plus `pairwise_levenshtein_pooled`. Those baseline cells were **absent rather than pending**: no job
would have filled them, only a code change. `score_set` now reports all of them, in the model's own
frame — same `reference_template`, same `aligned_natural_full` reference, the already-computed
`strength_full` / `mip_full` as the shared denominator.

Measured on the real Ty1 family with only the checkpoint-dependent calls stubbed out (the baselines
and every metric run as shipped; the model's own row is a stub and is *not* reported here), at the
pipeline's 20 x 20 config, holdout 200, L=121, `agreement_reference_n` 200:

| | covariance | MIP | pos. JS | pooled edits | MMD | KL |
|---|---|---|---|---|---|---|
| `random_mutations` (floor) | 0.7548 | 0.5265 | 0.0273 | 32.26 | 2.015 | 3.68e-3 |
| `random_homolog_pairing` (ceiling) | 0.9625 | 0.9359 | 0.0056 | 23.48 | 0.717 | 9.31e-5 |
| in-run real-homolog ceiling (n=300) | 0.9694 | 0.9280 | — | — | — | — |

Two things fall out, and the second one is the interesting one.

**The random-pairing baseline IS the ceiling construction, confirmed.** 0.9625 against the in-run
ceiling's 0.9694 on covariance, and 0.9359 against 0.9280 on MIP — the pairing row is *slightly
above* on MIP, which is what a matched sample size predicts: it scores 400 sequences where the
ceiling key scores 300, and MIP's ceiling rises with the scored set's size. So the floor row is
labelled correctly and the two constructions agree to within their own sample-size sensitivity.

**Our floor is much lower relative to the ceiling than the paper's.** As a fraction of the pairing
row, ours is 78.4% on covariance and 56.3% on MIP. Figure 3's random-mutations row is 0.897 against
its random-pairing 0.988 (91%) and 0.660 against 0.927 (71%). Absolutely, 0.755 against their 0.897
and 0.527 against their 0.660 — a 13-14 point gap on both, in the same direction, **in the leaked
frame**; in the reported disjoint frame it is 4.9 points on covariance and +0.2 the other way on
MIP (see "Our random-mutations floor"). This is not a new
defect; it is the effect
"Agreement scores are not comparable across studies" measures, seen from the other end. A small
reference flatters a poor generator, and the paper states no split sizes anywhere, so their
random-mutations row being *closer* to their ceiling is consistent with their reference being smaller
than our 200 — and nothing in Figure 3 lets us check that. **Put our floor and their floor in the
same table only with both reference sizes stated, and never subtract them.**

**The ordering is a large-sample property.** At 20 scored sequences against a 20-sequence reference
the floor/ceiling ordering does not merely narrow, it *inverts* — 0.411 for random mutations against
0.395 for real homologs on the synthetic family the unit test uses. Same mechanism as the table in
`matrix_agreement`'s docstring (good/bad ratio 96.8% at n=20, 71.0% at n=400). Hence the test asserts
the columns and the frame, not the ordering: asserting the ordering at n=20 would pin noise.

`alignment_length` is now recorded **per baseline row**, measured off the projected sequences rather
than restated from the intended template's length. The alignment bug above ("The editor aligned to a
different template than every baseline") moved no metric value at all, so a recorded width, compared,
is the only thing that could ever have caught it — and a width copied from the *intended* template
would agree with the model's by construction and report nothing.

## The editor trains on joined `VH.VL` and is evaluated on single V domains — the two length supports do not overlap

Every §4.2 number in this repo is measured across a train/test **input-form** shift. The editor's
training inputs are two-domain `VH.VL` lines; both evaluation families are lone V domains. This is
not a leak and not a crash — the code handles both forms — but it is a distribution shift that some
of the ten Figure-3 metrics are structurally sensitive to and others are structurally blind to, and
until now it was recorded nowhere.

### What the training corpus actually is (measured)

`data/pretrain/oas_homolog_pairs.tsv.gz`, md5 `4cb9f191144bd20e3615e40d17ea128e`, the current
`build_homolog_pairs` output. Two columns, `x0<TAB>x1`, both fed to
the model as sequences, so 2 × 1,660,105 = **3,320,210 training inputs**:

| | value |
|---|---|
| rows | 1,660,105 (all exactly 2 columns) |
| cells containing exactly one `PAIR_SEP` (`.`) | **3,320,210 / 3,320,210 = 100.00%** |
| cells containing zero or ≥2 separators | **0** |
| input length: min / p1 / median / p99 / max | 153 / 222 / **232** / 243 / 298 |
| per-chain length after splitting (200k-row sample, 800k chains) | min 70 / median **114** / max 164 |

**There is not one single-domain line in the training corpus.** The separator is a real ESM-2
vocabulary token (`editjumps/core/sequences.py`, pinned by
`test_params_yaml_pair_sep_is_tokenizer_valid`), so the model saw `.` at a consistent position in
every one of 3.32M inputs.

### What the evaluation families are (measured)

`data/interim/seed_families/`, the `seed_homologs` output at `chain: heavy`:

| family | n | min | p5 | median | p95 | max | contain `PAIR_SEP` |
|---|---|---|---|---|---|---|---|
| `Anti-SARS-CoV-2_VHH_Ty1` | 38,687 (38,553 distinct) | 107 | 116 | **121** | 127 | 143 | **0** |
| `Anti-HER2_scFv_VH_trastuzumab` | 43,088 (42,839 distinct) | 101 | 116 | **122** | 127 | 132 | **0** |
| `Anti-HER2_scFv_VL_trastuzumab` | **2** | 107 | — | 107 | — | 107 | 0 |

**The two supports are disjoint.** The shortest training input is **153** residues; the longest
evaluation input is **143**. Counted both ways: 0 of 3,320,210 training cells are ≤ 143, and 0 of
81,777 family members are ≥ 153. Not a shifted distribution with a shared tail — no overlap at all.

**But the sequences themselves are half in-distribution.** Indexing the pairs file the way
`family_split.sequences_in_pairs` does (joined line *and* each chain), **52.0%** of distinct Ty1
members (20,058 / 38,553) and **50.9%** of distinct HER2-VH members (21,824 / 42,839) appear
verbatim as the VH half of a joined training line. So the model has seen roughly half of each
evaluation family — but only ever as the first 121 residues of a 232-residue input, never as a
121-residue input. The shift is in the **input form**, not in the sequence content. (This is also
why `sequences_in_pairs` indexes both forms: over the joined line alone the leak reads 0%, and a
0% leak reads as reassurance rather than as a unit mismatch.)

The third file is a degenerate stub: the HER2 VL family has **2 members**, so `split_family` cannot
partition it (it needs `n_templates + 2` distinct members and raises otherwise). Nothing evaluates
on it. There is no light-chain evaluation arm, and there never was one.

### Is the VHH label correct, and is it used correctly? Yes, everywhere

Checked `params.yaml`, `dvc.yaml`, `dvc.lock`, `Makefile`, `README.md`,
`docs/`, `deploy/`, every `metrics/**` artefact, and every `editjumps/**` default.

* **Ty1 is correctly labelled a VHH.** Its record in
  `editjumps/pipeline/preprocess/pretrain/seeds/evoflows_seeds.fasta` (one of three there, the others
  being the trastuzumab VH and VL) is a 118-residue V domain with exactly **two cysteines**, at
  sequence positions 22 and 96 — IMGT 23 and 104, the
  canonical intradomain pair. One domain, one disulfide. A paired Fv carries four. The family
  agrees: 35,268 of 38,687 Ty1 members have exactly two cysteines, and the same for 39,330 of
  43,088 HER2-VH members (the 3–6 Cys tails are real extra-cysteine repertoire members, not
  concatenated chains — every one of them is ≤ 143 residues).
* **`params.yaml`'s claim "a seed family FASTA holds single V domains" holds.** 0 of 81,777 members
  carry a separator; max length 143 against a paired form's 153 floor. The claim is relied on in
  two places (`evotune.max_length: 160`, `seed_homologs.chain`) and holds in both.
* **`chain: heavy` is the right ANARCI setting for a camelid VHH.** `cdr.CHAIN_TYPES` maps
  `heavy -> {"H"}`, and ANARCI numbers VHH as an H domain. Measured, not assumed: a numbering pass
  over the families recorded **3 failures in 5,000** scanned Ty1 members (0.06%) and **0 of 200** on
  the scoring reference. The IMGT path does not need a light chain and does not look for one. (That
  pass ran under a stage which is no longer part of this repository, so the numbers are a record
  rather than something `dvc repro` reproduces.)

### What frame did EvoFlows itself use?

**Single-chain, and three of their six datasets are not antibodies at all.** The repo holds no PDF —
what it holds is `docs/evoflow_reproduction.md` (fact-checked against the primary PDFs, 2026-07-02)
and this file's exhaustive re-check against arXiv 2603.11703v2, all 22 pages. From those:

* Their §4.2 datasets (Table 2, p.19) are Ty1 VHH (L=109.2), anti-HER2 **scFv** (246.4), anti-EphA2
  VHH (118.2), chicken FGF2 (141.6), serine-pyruvate aminotransferase (363.0) and haloalkane
  dehalogenase DhaA (282.2). An scFv is one chain — VH, linker, VL — and the three non-antibodies
  are ordinary proteins. So **their evaluation unit is one natural single-chain sequence**, never a
  separator-joined two-domain construct.
* Their training corpus is described as families from **UniRef and OAS, including antibody fragments
  (VHH and ScFv)** — both single chains — and §3.1's family construction (homolog search from a
  seed) is defined on single sequences. **The repo does not contain enough of the paper to prove
  they never joined an OAS VH to its VL.** It records no statement either way. Stating "they trained
  on single chains" as a fact would be a guess; what can be said is that every unit the paper is
  recorded as naming, on both the training and the evaluation side, is a single chain.

**So the joined form is ours, not theirs.** It comes from `oas.pair_chains: true`, adopted for two
reasons that both needed both chains: the leakage-aware cluster split, and — in the property-guided
line that has since left this repository — a whole-Fv descriptor that oriented the pairs. Only the
first still applies here, and neither is a §4.2 requirement. It follows that EvoFlows had **no** train/test
input-form shift and we have one — a reproduction deviation to state, alongside the two already on
record (the Gillespie sampler, the `clock/len(x)` formula).

Two further frame differences are already recorded above and are not restated here: their anti-HER2
dataset is one 246-residue scFv where ours is a 122-residue VH, and we have 2 of their 6 datasets,
both antibodies.

### Per-metric sensitivity

Two recorded runs bracket this almost exactly: `metrics/generation_eval_B_family.json` and
`metrics/generation_eval_B_clock40.json` are the **same checkpoint** (`eval_B_stock_appA`), the
**same clock** (40.0), the **same budget** (12 templates × 10 variants = 120 generated), one on the
Ty1 family at L=127 and one on pairs-file `VH.VL` at L=235. They are *not* a controlled experiment:
the natural reference also changes (a 200-sequence family holdout vs a 48-sequence tail of the pairs
file), so any metric that reads the reference is confounded and is marked as such. Metrics computed
from the generated set and the alignment frame alone are not.

Recall the mechanism, already measured in "Clock normalization" above: `clock_scale` is `clock /
len(x)`, so the clock **already cancels the length arithmetic**. What it does not cancel is λ̄, the
per-position edit rate, which is a property of the checkpoint *and the input distribution*: the same
checkpoint sits at **0.0990** on a single V domain and **0.1392** on joined `VH.VL`. That 1.4× is
the shift, in its purest form.

| # | Figure-3 panel (our key) | verdict | reason | recorded L=127 → L=235 |
|---|---|---|---|---|
| 1 | Avg Levenshtein to x0 (`levenshtein_to_template`) | **sensitive** | the budget itself. The clock cancels L; the residual is λ̄ moving with the input distribution | 3.667 → 5.567 (**1.52×**) |
| 2 | Avg pairwise Levenshtein | **sensitive** | inherits (1) — diversity tracks the budget, ratio 1.79–1.83 across all 12 `metrics/dn/` arms | 6.835 → 10.619 (**1.55×**). These two are `pairwise_levenshtein`, the WITHIN-template key: neither of the two runs bracketing this section carries `pairwise_levenshtein_pooled`, which postdates them |
| 3 | Covariance (`covariance_agreement`) | **unknown** | the *replacement* key is a Pearson correlation, so the scale term cancels — but it estimates an (L,L) matrix from a fixed 120 sequences, and at L=235 there are 3.4× as many entries per sequence. Direction unknown without a matched-reference run. (The retired `covariance_frobenius` key is strongly sensitive: 6.69 → 45.19, **6.76×** — steeper than the L² factor of 3.42 the frame change alone would give) | — |
| 4 | ESM-2 PLL | **insensitive as recorded, sensitive through the budget** | stored as a per-position mean, so L cancels — but PLL rewards conservatism, and (1) says the single-domain frame spends 1.4× fewer edits, which flatters it. Magnitude unmeasurable from the artefacts: the L=235 clock-40 run stored no PLL | not comparable |
| 5 | Entropy delta | **insensitive** | per-position mean over the aligned frame | 0.08102 → 0.08030 (**0.99×**) — confounded reference, and it still barely moves |
| 6 | JS divergence | **insensitive** | composition JS pools over positions; positional JS is a per-column mean. Both normalise L away | 3.637e-4 → 3.521e-4 (**0.97×**), confounded |
| 7 | KL divergence | **insensitive** | 20-letter composition; two sets of V domains share it whether or not a VL is appended | 1.533e-3 → 1.406e-3 (**0.92×**), confounded |
| 8 | MIP (`mip_agreement`) | **unknown** | same argument as (3) — a scale-free correlation over an (L,L) matrix at fixed n. (The retired `mip_mean` key is insensitive *and* meaningless: 1e-18 at both lengths, zero by construction) | — |
| 9 | MMD (`spectrum_mmd`) | **sensitive by construction** | eq 26–27 uses *unnormalised* contiguous k-mer counts, so the kernel grows ≈L — **not L²**, corrected below — and the biased V-statistic with it. Already known to move with the evaluation shape too ("Spectrum MMD is not comparable across evaluation shapes") | 1.313 → 4.662 (**3.55×**), confounded but the direction is structural |
| 10 | Profile log-likelihood | **sensitive, structurally** | a *sum* over aligned columns, hence extensive in L. Never comparable across input forms without dividing by L | −82.69 → −204.62 (**2.47×**) |

**The pattern is one this project has hit before.** In work that has since moved out of this
repository, a chance floor moved from 300× to 145× because a denominator counted IMGT positions the
experiment never corrupted — a quantity that had to be restricted to the region actually touched,
and was not. Same shape here: metrics that are
*means over positions* (4–7) are blind to whether 121 or 232 positions are in the frame, and metrics
that are *sums or counts over the frame* (1, 2, 9, 10) carry the frame's size in the number. Sorting
the ten panels by that one question predicts every measured ratio in the table above.

**Direction of the bias.** The single-domain frame makes the editor **more conservative**: 1.4×
fewer edits per position at the same clock. That inflates PLL (which rewards conservatism),
deflates diversity, and — most concretely — pushes the realised budget to **3.47–6.17** edits
against the "3–11 edit band" then taken for §4.2's — a band the paper does not state, retracted
below — where §4.2's whole comparison rests on "matching the expected number of mutations per
sequence across methods". The already-recorded recommendation to
raise `edit_flows.clock` to ≈71 for arm B is, read through this section, a correction *for the
input-form shift* and not only a re-centring.

### The caveat sentence for the paper

> Every §4.2 number reported here is measured under an input-form shift the original does not have:
> our editor is trained exclusively on joined two-domain `VH.VL` lines (100% of 3,320,210 training
> inputs, 153–298 residues) and evaluated exclusively on single V domains (81,777 family members,
> 101–143 residues — the two length supports do not overlap at a single sequence), a shift that on
> one checkpoint at a fixed clock already changes the per-position edit rate by 1.4×, and to which
> four of the ten Figure-3 metrics are structurally sensitive (1, 2, 9, 10), three structurally
> blind (5, 6, 7), two of unknown sensitivity (3, 8), and one — ESM-2 PLL (4) — insensitive as
> recorded but sensitive through the edit budget.

The cheapest thing that would close it is not a retrain: it is one scoring run of the same
checkpoint on `VH.VL` templates against a family-derived reference, so the reference is held fixed
and the input form is the only thing that moves. That run has now been taken, and it moves the
conclusion — see the next section.

## Measured: giving the editor its training input form does not improve the reported numbers

The section above priced the shift from two runs that were never a controlled pair, and predicted
per-metric sensitivities from the shape of each formula. Five matched runs now measure it, and the
headline is short: **the reported §4.2 numbers are not meaningfully a distribution-shift artefact.**
On a frame where the input form is the only thing that moves, every metric is within a few percent
of where the single-domain evaluation puts it, and what does move mostly moves the *wrong* way — the
joined arm is slightly worse, not better.

`editjumps/measurements/measure_input_form_shift.py` builds the arms and scores them; artefacts in
`metrics/inputform/`.

### What had to be held fixed, and why each one would otherwise price itself

Appending a domain changes four things at once, and any of them will masquerade as the shift.

1. **The molecules.** Both arms are written from ONE member list, filtered for absence from the
   training pairs *before* the join. Filtering afterwards passes trivially for the joined arm — a
   constructed string is in no pairs file — and the two arms silently stop being the same
   antibodies. Since `split_family` shuffles a de-duplicated, file-ordered list under a fixed seed,
   template *i* of one arm is template *i* of the other plus a light chain, and the holdouts match.
2. **The frame.** The joined arm is rescored restricted to the VH coordinates. Read off the full
   226-position alignment, the comparison prices ~107 appended columns that are *identical in every
   sequence on both sides*: a denominator counting positions the experiment never touched. That is
   the same error as the 300×→145× chance floor described above. It is not small here —
   spectrum MMD reads 2.038 on the full alignment against 1.130 restricted, so the uncorrected
   comparison would have reported a 2.0× degradation that is entirely the appended block.
3. **The edit budget.** `clock_scale` is `clock / len(x)`, so a fixed clock does *not* fix the
   number of edits: λ̄ moves with the input distribution. At clock 40 the joined arm's VH block gets
   **0.63×** the single arm's edits, so a metric that moves might only be reporting a smaller
   budget. A third arm at clock 76 — solved through `CALIBRATION_EXPONENT` — restores it to 1.19×.
4. **The reference.** Held fixed by construction: same holdout, same 300-member agreement ceiling.
   Both ceilings come out identical to five decimals across arms (0.97468 / 0.92829), which is the
   check that the frame really is shared rather than merely intended to be.

The appended chain is the **real trastuzumab light chain**, read from the pinned public
download. The HER2-VH family is a repertoire around trastuzumab's heavy chain, so `VH.VL` is that
antibody rather than an invented chimera. The same chain is appended to Ty1, where it is a shape
probe and not a molecule — a camelid VHH has no light chain — and reusing the identical block is
what makes any family-to-family difference attributable to the heavy domain.

### The reconstruction is verified, and the single-domain arms *are* the reported cells

`write_generated_fasta` keeps the generated sets and not the holdout, so the split is reconstructed.
Run the rescorer on a single-domain arm, where truncation is a no-op, and all fourteen metrics must
come back identical to that run's own artefact. They do, to 1e-9, on both families. Better: both
single-domain arms reproduce the committed `metrics/disjoint/editor-{ty1,her2vh}.json` cells
**bit-for-bit** — every key, to the last digit. So this is a comparison against the reported numbers
themselves, not against a re-derivation of them.

### The result

Percentages are the joined arm against the single-domain arm, both at L=118, same 20 antibodies.

| metric | HER2 single | joined, clock 40 | Δ | joined, budget-matched | Δ | Ty1 Δ (clock 40) |
|---|---|---|---|---|---|---|
| `levenshtein_to_template` | 4.083 | 2.553 | −37.5% | 4.870 | +19.3% | −36.5% |
| `pairwise_levenshtein_pooled` | 24.732 | 24.540 | −0.8% | 25.985 | +5.1% | −1.1% |
| `covariance_agreement` | 0.9345 | 0.9169 | −1.9% | 0.9334 | **−0.1%** | −3.2% |
| `mip_agreement` | 0.8659 | 0.8222 | −5.0% | 0.8578 | **−0.9%** | −7.0% |
| `kl_generated_vs_natural` (pooled) | 0.00063 | 0.00066 | +4.7% | 0.00072 | +15.7% | +25.6% |
| `spectrum_mmd` | 1.0210 | 1.1298 | +10.7% | 1.0786 | +5.6% | +9.9% |
| `entropy_delta` | 0.0833 | 0.0751 | −9.9% | 0.1188 | +42.5% | −27.3% |
| `js_divergence_positional` | 0.01026 | 0.01182 | +15.3% | 0.01058 | **+3.1%** | +10.9% |
| `kl_divergence_positional` | 0.05242 | 0.05910 | +12.7% | 0.05889 | **+12.3%** | +8.2% |
| `profile_log_likelihood` | −59.03 | −58.70 | +0.6% | −63.29 | −7.2% | +1.9% |

**Matching the budget is what settles it.** At clock 40 six metrics look 5–15% worse on joined
input, which is the reading a less careful version of this experiment would have published. Raise
the joined arm's VH budget to parity and the two agreements collapse to −0.1% and −0.9%, and
positional JS from +15.3% to +3.1%. Those differences were the smaller edit budget, not the input
form. `entropy_delta` and the PLL move the other way at the higher budget for the same reason —
they track the budget, not the form, in both directions.

**One metric survives budget matching: positional KL, +12.7% at clock 40 and +12.3% matched.** It is
the only quantity here that is genuinely worse on the model's own training input form, and it is
worse by the same amount whatever the budget. Note which reading that is — the *positional* KL that
`fix/kl-representation-propagation` established as the comparable one. The pooled composition KL,
which the sensitivity table above marked "insensitive", is the reading now demoted; the prediction
was right about the number it was made on and does not transfer.

**Two predictions overturned, several confirmed.** Covariance and MIP were marked *unknown*: they
are insensitive once the budget is matched. Composition KL was marked *insensitive* on the pooled
reading, and the positional reading is the one metric that is not. Confirmed as written: the
Levenshtein budget is sensitive (whole-string 6.435 against 4.083, **1.58×**, against the 1.52×
predicted from λ̄), and MMD and profile log-likelihood carry the frame size (2.0× and 1.46× on the
full alignment, both gone under restriction) — the "means over positions versus sums over the frame"
rule sorts the panels correctly, and its predictions hold wherever the frame is what changed.

### The editor spends its edits on the appended domain

The finding nothing predicted. On the joined arm the whole 226-position string receives 6.435 edits
and the VH block receives **2.552** — 39.7% of the edits on 52.2% of the positions, so the heavy
domain is under-edited by 1.3× relative to its share. The same ratio holds on Ty1 (0.63× and 0.64×
of the single-domain budget). And **all 400 generations kept exactly one separator**: the model has
learned the two-domain structure well enough never to break it, insert a second one, or delete it.

Two readings, and this experiment cannot separate them: the model may prefer to edit light chains,
or it may prefer to edit *this* light chain, which is the same single sequence appended to every
joined input and may
simply sit at unusual likelihood in context. Sampling a different natural VL per member separates
them and is the obvious follow-up. Either way it sharpens the caveat above rather than softening it:
a joined input does not spend its extra budget where the single-domain evaluation would.

### What this does to the caveat sentence

The measured version, replacing the projected one:

> Our editor is trained exclusively on joined two-domain `VH.VL` lines and evaluated exclusively on
> single V domains, whose length supports do not overlap at a single sequence. Scoring the same
> checkpoint on joined inputs, with the molecules, the alignment frame, the reference and the edit
> budget all held fixed, moves the two coupling agreements by 0.1% and 0.9% and the positional JS by
> 3.1%, leaving positional KL the one metric materially affected (+12%) — so the reproduction's
> §4.2 numbers are not substantially an artefact of the input-form shift.

What this does **not** license: the shift is still real, still uncontrolled in the reported runs,
and still worth stating. It is now measured rather than feared, and it is smaller than the reference
size effect that was already withdrawn a cross-study claim over.

## The two metrics our port leads on do not survive an interval — the disjoint frame is a draw

`metrics/appendix_table.md` reported `covariance_agreement` and `mip_agreement` as bare centres, and
those are the only two of the ten §4.2 metrics on which our editor leads the evotuned-PLM baseline on
**both** seed families. Everything else in the disjoint frame goes the other way: spectrum MMD,
positional KL, positional JS, `entropy_delta` and the profile log-likelihood all favour Evotuning on
both families; pooled diversity is closer to the random-pairing target for Evotuning on HER2-VH and
about level on Ty1; the pooled composition KL splits (ours on Ty1, theirs on HER2-VH); and
`edits/seq` is a matched budget rather than a score, with overlapping intervals. So the paper's §4.2
claim was down to two columns with nothing on them but a centre.

### Why there was no interval, and why the reason was incomplete

`consolidate.RESAMPLE_FROM` gave an interval only to `edits_per_sequence`, because that is the only
metric whose artefact stores the per-unit values its centre averages over. Sound, and incomplete: the
generated set is 20 template blocks of 20 variants, template-major, and the sequences are committed
beside every cell. An interval on a set-level metric can therefore be built by **resampling template
blocks and recomputing the metric** — at the same unit every other centre in the table averages over.
That is `editjumps agreement-interval`, a new module command in the shape of `rescore_positional`: it
re-reads the committed FASTA, never re-generates, reproduces the run's own centre from it to 1e-9
before writing anything, and stamps every interval as recovered rather than emitted.

### Reproduction: the margins reproduce, the intervals do not, and the reason is the pairing

The four margins reproduce exactly (+0.0050 / +0.0141 on Ty1, +0.0052 / +0.0311 on HER2-VH). The
paired intervals came out **about 3x narrower** than the hand measurement this work was built on, and
the cause is a real defect in the hand measurement rather than a seed difference:

**`generation_eval` does not walk its templates in split order.** It walks
`random.Random(seed).sample(rows, n_templates)`, while `evotune_baseline` and `evodiff_msa_baseline`
walk `split.templates` in order. At seed 0 that permutation is
`(12, 13, 1, 8, 15, 6, 19, 4, 7, 5, 9, 3, 2, 11, 17, 0, 14, 16, 18, 10)` — so block 0 of
`editor-ty1.fasta` is template 12 and block 0 of `evotune-ty1.fasta` is template 0. Differencing two
bootstraps on block index therefore pairs *different templates* and cancels none of the shared
variance, which is the entire purpose of pairing. Reproducing that mistake on purpose gives Ty1
covariance `[-0.0993, +0.1155]` against the hand-measured `[-0.0964, +0.0794]`, and HER2-VH MIP
`[-0.1515, +0.1713]` against `[-0.1266, +0.1702]` — i.e. the hand measurement is the mispaired
version, to within the jitter of 100 draws. It is kept in each report under
`if_paired_on_raw_block_index` so the size of the mistake stays visible.

The fix does not assume any evaluator's shuffle: `template_of_each_block` recovers the mapping from
the sequences themselves by nearest edit distance (a block's variants are ~4 edits from their own
template and ~20 from any other) and refuses anything that is not a bijection. Its answer is exactly
`random.Random(0).sample(range(20), 20)`, on both families, which is the independent check on it.

### The result, at 100 and at 2,000 paired draws

Paired percentile bootstrap over template indices, seed 0, both counts drawn from one generator so
the smaller is a prefix of the larger. Each draw resamples one set of template indices and scores
both methods on it.

| | margin | paired 95%, 100 draws | paired 95%, 2,000 draws | share of draws ahead |
|---|---|---|---|---|
| Ty1 covariance | +0.0050 | [−0.0224, +0.0478] | [−0.0205, +0.0437] | 64% |
| Ty1 MIP | +0.0141 | [−0.0744, +0.0512] | [−0.0656, +0.0403] | 35% |
| HER2-VH covariance | +0.0052 | [−0.0168, +0.0811] | [−0.0172, +0.0579] | 52% |
| HER2-VH MIP | +0.0311 | [−0.0189, +0.0606] | [−0.0251, +0.0581] | 75% |

**All four include zero, at both draw counts.** The draw count was raised because 100 is coarse — a
percentile interval from 100 draws puts its endpoints on the 2nd and 98th order statistics, so its
width is itself noisy — and it changed nothing: every one of the twelve comparisons in the two
reports gives the same verdict at 100 and at 2,000. 2,000 paired draws cost ~18s per comparison
(project each set once, then it is numpy over cached per-template joint-frequency tensors), so there
was no reason to stop at 100.

**Ty1's MIP lead does not survive in sign.** Its bootstrap mean is **−0.0120** against a measured
margin of +0.0141, and our port is ahead in only 35% of draws. That is a stronger statement than an
interval containing zero: on Ty1's MIP the measured direction is the minority outcome across
redraws of the twenty templates.

**The test is not merely blunt.** The same 20 templates, the same estimator and the same draw counts
*do* separate other pairs: the forced evotuned baseline is distinguishably worse than the unforced
one on all four family × metric cells (e.g. Ty1 MIP −0.0771, `[−0.1539, −0.0664]`), and EvoDiff-MSA
is distinguishably worse on Ty1's MIP. So a null here is a measurement, not an absence of power.

### A second finding: a marginal interval on these two metrics is a SPREAD, not a coverage interval

Sixteen of the 24 per-method percentile intervals written into these cells **do not contain their own
centres** — Ty1's editor covariance is 0.9119 with `[0.7391, 0.8934]`. This is not a bug and it is
worth stating plainly, because a reader who takes it for a confidence interval will read it as a
catastrophe. A with-replacement draw of 20 templates leaves **12.8 distinct** ones on average, and
`matrix_agreement`'s own docstring already records that these estimators move with the effective
sample size behind the matrix by more than they move between methods. So every resampled set is
effectively smaller than the one measured and scores lower. Each cell records that as its
`bootstrap_shift`: 0.077–0.132 on the generated rows, 0.004–0.017 on the near-ceiling
random-pairing rows.

That shift is why the paired form is the only readable one: it is within 0.003 of identical for the
two methods (Ty1 covariance: −0.0814 for the editor, −0.0838 for Evotuning) and cancels in the
difference. Each interval carries the warning in its own `reads_as` field, the CSV distinguishes the
two kinds in a new `ci_kind` column (`mean_over_templates` against
`template_resample_spread`), and the appendix table marks the recovered kind with `‡` and renders the
paired difference tables underneath each family, from the artefact rather than from prose.

Read as a spread the number is still useful, and it is the reason the verdict was predictable: the
`‡` intervals are 0.15–0.23 wide, which is **7 to 38x** our port's margin over the evotuned PLM.

### The sentence for the paper

> On a scoring reference provably disjoint from the editor's training pairs, our reproduction leads
> the evotuned-PLM baseline on two of ten §4.2 metrics — covariance and MIP agreement — by 0.005 to
> 0.031, and neither lead survives a paired percentile bootstrap over the twenty templates: all four
> 95% intervals on the difference include zero at 100 and at 2,000 draws, and on Ty1's MIP the
> resampled difference is negative in 65% of draws. We therefore report no metric on which the
> reproduction is distinguishably better than a masked-LM finetune of the same trunk on the same
> family.

What this does **not** say. It is not a claim that Evotuning is distinguishably *better*: on these
two metrics the paired interval spans zero in both directions, and the eight metrics our port trails
on were not given intervals here, so "we trail" on those remains a point estimate. It also does not
retract the earlier section "§4.2's last two methods land, and our port beats both" — that comparison
was drawn in the leaky frame, at a reference overlapping the editor's training pairs, and this one
supersedes it on frame grounds independently of the interval. And it says nothing about the paper's
own EvoFlows numbers: covariance and MIP are `within_study` at a stated `agreement_reference_n`, and
comparing them across studies is inadmissible outright.

## The spectrum kernel is unnormalised because the paper's eq 26-27 is, and it grows ≈L, not ≈L²

Checked against the published PDF (arXiv 2603.11703, §B.5) rather than against the shape of our own
numbers, because the previous justification was a range-match — "our values sit inside the paper's
Figure 3 MMD axis (1 to 4)" — which is a check that passes for the wrong reason: landing in [1,4] is
consistent with many kernels. The equations as printed:

    Phi_k(x)_a = |{i : x[i : i + k] = a}|                            (26)
    k(x, y) = <Phi_k(x), Phi_k(y)> = sum_a Phi_k(x)_a * Phi_k(y)_a   (27)

Raw occurrence counts, plain inner product. No `sqrt(k(x,x) k(y,y))`, no unit-norm features. Our
`spectrum_features` / `spectrum_kernel` match it exactly, so **the unnormalised form is faithful and
must not be "fixed"**.

**The definitional citation is Leslie et al. (2002), not ProteoGAN.** Kucera et al. 2022 appears
exactly once in the paper, as the fourth item in a list of the kernel's attractive properties ("Its
simplicity ensures robustness when evaluating artificial sequences"). That is a robustness
endorsement, not the estimator's source. Whatever ProteoGAN normalises internally is not what
EvoFlows specifies. A reading that went to ProteoGAN first concluded we had a normalisation bug; we
do not.

**`k` is still unstated.** Confirmed by reading the whole PDF: the only occurrences of "-mer" are the
four in §B.5, all with a symbolic `k`. Note the PDF sets it as math-italic U+1D458, so an ASCII
`grep "k-mer"` returns nothing and looks like absence of evidence. Our `k=3` remains our own choice.

**The kernel grows ≈L, not ≈L².** With ~120 near-distinct 3-mers drawn from 8,000 slots, collisions
are rare, so `k(x,x) = sum_a c_a^2 ≈ L` and `MMD ∝ sqrt(L)`. Measured: recomputing every arm with a
normalised kernel gives a ratio of 11.04-11.35× against the unnormalised value, and 11.2² ≈ 125 ≈ the
mean sequence length of 122. The per-metric sensitivity table above said ≈L² and is corrected.

**Normalisation would change no comparative claim.** All 7 arms × 2 families, recomputed both ways
against one common 200-sequence natural reference at n_generated=400: the ranking is *identical* on
both families, and the ratio is constant to ±1.4%, i.e. a near-pure global rescale. The reason is
that the clock cancels length, so arm mean lengths sit within ~2% of each other (121.9-124.7, a 2.3% spread). The
length-conflation hazard is real in principle and negligible here.

Related length checks, measured rather than assumed. Over the eight committed `metrics/disjoint/`
sets the per-set mean length is **121.8-122.8** (a 0.8% spread) and the within-set sd **3.45-4.18 aa**
(2.8-3.4% CV); the wider 121.9-124.7 and 3.3-5.8 quoted above come from a seven-arm sweep whose
sequence sets are not all committed here, so treat those two as recorded rather than checkable. The
editor's *net* length drift, +0.04 to +0.13 aa per sequence against ~4.6 edits, is in the same
category: no committed metrics file carries a length-delta key, and the generated sets alone cannot
supply the template mean to difference against, so it is recorded and not recomputable here either.
What it says is that insertions and deletions very nearly cancel — net drift ~2.7% of the edit
count, not 100%. Consequence for the clock sweep: even assuming every extra edit from clock 80 to
120 were an insertion, length could explain at most 16-37% of the observed MMD rise; at the measured
drift it explains ~0.4% of a ~10% rise. **The interior optimum is not a length artefact.**

Two traps worth keeping, both hit while checking the above:
* The **editor's two** `metrics/disjoint/*.fasta` files hold **three** 400-sequence sets (`model`,
  `random_mutations`, `random_homolog_pairing`); the other six hold one. Length statistics taken
  without splitting on the header prefix mix the editor with its own controls.
* A naive FASTA parser that treats the leading `;` comment block as sequence data concatenates it
  into one ~200-character pseudo-sequence, which reads as a 2× length spread that does not exist.

## The "3-11 edit band" is not stated by §4.2, and the budget has no external anchor

Several places cited a "3–11 edit band" as **stated by §4.2** — `params.yaml`,
`docs/model_cards/edit-flows-editor.md`, a guidance-sweep module that no longer ships, and this
file (twice above, at the `VH.VL` note and in the input-form section; both now carry a forward
pointer to this section rather than asserting the band). The module site was missed
on the first pass and found by re-grepping `editjumps/` as well as `docs/` — the inventory of a wrong
claim is itself a claim worth checking. Checked against the PDF: **§4.2 gives no mutation count.** Its only relevant
sentence is "matching the expected number of mutations per sequence across methods (except random
pairing)", with no number attached. The single mutation count anywhere in the paper's prose is "in
some cases as low as 1.5 mutations per sequence", describing evotuning.

Nor is 3–11 recoverable from the extracted figures: Figure 3's `panel_0` spans 1.57–247.7 across six
datasets, and Figure 5 gives 2.54 / 4.43 on the VHH and 11.75 / 13.80 on the HER2 scFv. Provenance
unknown. One plausible mechanism: it may predate the 2026-08-28 recalibration of Figure 3's
distance-to-x0 panel, the one where only four of its seven y-ticks carry a printed label and
moved random pairing from 8.47 to 117.90. **Do not cite it as the paper's until someone derives it.**

This matters because the band was the stated rationale for `clock: 40`. What actually sets the
budget, and is checkable in the artefacts:

* The three budget-matched baselines are run with `--mutations-from <editor metrics>` (`standard.py`), which
  reads the **editor's own realised mean edit distance** and rounds it. `DEFAULT_STEPS` runs the
  editor first for exactly this reason.
* So the targets are per-family and downstream of us: Ty1 editor 4.580 → baseline target **5**;
  HER2 editor 4.082 → baseline target **4**. `params.yaml`'s `evotune.mutations: 4` is only the
  fallback for the unmatched path — Ty1's 5 could not have come from it.
* Therefore **the clock sets the budget for every arm, and nothing external pins it.** Not §4.2, not
  the band, not params.yaml. Two consequences: "clock 40 is non-optimal for MMD" is a statement
  about an unmotivated operating point rather than a defect; and raising the clock would re-match
  every baseline automatically, so a clock-80 comparison would still be budget-matched — the
  objection to it is that the clock would have been selected on the reported metric, not that it
  would be unfair.

This is the gap a clock-calibration step would close: anchoring the budget to each family's own
natural divergence replaces a number we chose with one the data supplies. **Nothing implements it**
— there is no such stage, target or CLI verb — so this is a stated next step, not a described tool.

Also traced while checking the above: the **93.9** cited for the original's scFv edit floor is real
and lives in the accompanying write-up rather than in this repository, which is why searching this
repo's history for it found nothing at the time —
`results/evoflows_figure5.json`, Anti-HER2 scFv, Random pairing = **93.86151**, quoted in
`scripts/make_table2.py`'s module docstring. Note the "our 23.3" it is set against is now 23.38
(Ty1) / 23.25 (HER2) on the disjoint frame, so the cross-study ratio needs its denominator restated.

## `.` is not ESM's chain separator — there is no such thing — but ours is trained, and one trunk's is not

Checked against fair-esm's own alphabet rather than against habit, after the question "did we
rightfully use the dot as the chain sep?".

**What ESM defines.** `.` and `-` both sit in `proteinseq_toks`, shared verbatim by ESM-1, ESM-1b,
MSA Transformer and ESM-2 (`esm/data.py`, `from_architecture`). They are inherited from the
*alignment* lineage — in A3M/Stockholm, `-` is a deletion gap and `.` an insertion-column gap, which
is what MSA Transformer needs them for. `-` is documented as a gap token; **`.` is documented nowhere
as a separator.** Where ESM does define multi-chain input it is ESMFold, and the convention there is
a **colon**. So `PAIR_SEP = "."` is ours, not ESM's, and belongs in the silent-choices register —
entry 8 of `editjumps/core/edit_flows/README.md`, alongside deviation 5's clock formula.

**In stock ESM-2 the embedding is untrained.** Measured on `esm2_t12_35M_UR50D`:

| token | ‖embedding‖ | z vs the 20 residues |
|---|---|---|
| 20 standard residues | 2.265 (sd 0.250) | — |
| `.` | 1.025 | **−5.0** |
| `-` | 1.006 | −5.0 |
| `<null_1>` | 1.029 | −5.0 |

`.` is statistically indistinguishable from `<null_1>`, which `esm/data.py:110-111` creates purely to
pad the vocabulary to a multiple of 8. ESM-2 trained on UniRef50, which contains no alignment
characters, so the vector never left initialisation.

**That makes it a free slot, not a wrong choice — and we train it.** Embedding movement from stock
during edit-flows training of `eval_B_stock_appA`:

| token | shift | relative to mean residue shift |
|---|---|---|
| 20 residues (mean) | 0.2118 | 1.00× |
| **`.`** | **0.2845** (cos 0.969) | **1.34×** |
| `-`, `<null_1>`, `X`, `B` | 0.0000 | 0.00× |

The separator moved *further than the average residue*, while every other non-standard token moved
exactly zero. It is a genuinely learned token in the reported checkpoint.

**The trap, and it is live.** The OAS-adapted trunk `esm2_oas` has `.` at **1.021** against stock's
1.025 — unmoved. Its pretraining corpus therefore carried no **`.`** separators. Not because it was
built without `--pair-chains` — it was not: `split_corpus`'s `dvc.lock` deps include
`oas_corpus.clean.cdr_keys.tsv.gz`, and `download_oas.py:234` refuses `--emit-cdr-keys` without
`--pair-chains`. It is because that corpus predates the `oas.pair_sep` fix and joined its chains
with **`:`**, which ESM-2 tokenizes to `<unk>` (`docs/reproducibility.md`'s staleness table). **Any arm that starts from `esm2_oas` and is not subsequently trained on joined
lines is using an untrained separator.** The reported arm is safe: it starts from *stock* ESM-2 and
trains on joined pairs, which is what silent-choice 8 records. Check this before reusing
`esm2_oas` for anything joined — including the `--chain heavy` retrain, where the question is the
mirror image (heavy-only training never sees a separator either).

This is the same shape as the two output bugs found the same day: the vocabulary offers more tokens
than the alphabet admits, and nothing checked which of them carried meaning.
