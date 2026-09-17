# Evotuning Baselines (EvoFlows §2.2 / §4.2)

Baseline generation using family-adapted masked language models (MLMs) following EvoFlows ([arXiv 2603.11703][evoflows]) §4.2. Models are adapted on homolog sequences (Alley et al., 2019) via `editjumps/pipeline/train/evotune.py`.

The generation pipeline decouples position selection from token generation:

- `profile.py`: Computes per-column Shannon entropy $H(l) = -\sum_a p_l(a) \log(p_l(a) + \epsilon)$ to construct position sampling distributions mapped to template coordinates (eq. 12).
- `substitution.py`: In-fills selected masked positions using the adapted MLM with temperature scaling and budget matching constraints.

Decoupling position weighting from the neural proposer enables headless unit testing and allows `editjumps/core/evodiff_msa/` to reuse the same sampling logic. Evaluation is driven by `editjumps/pipeline/evaluate/evotune_baseline.py`.

**Substitutions only, so no realignment.** The baseline masks positions and infills them, never
inserting or deleting, so every generated sequence keeps the template's length and coordinate system.
Nothing here re-aligns its own output, and "mapped to ungapped coordinates" is a one-time operation
on the profile rather than a per-step one. That is a property of the method, not a simplification —
`editjumps/core/length_capability.py` is where each evaluator declares what it can do to a sequence's
length, and this one declares it cannot grow.

Sites are tagged in the code:

```bash
grep -rn "DEVIATION\|UNSPECIFIED" editjumps/core/evotune/*.py
```

## Where the paper is silent

Six readings. None of them is presented as the paper's, and each fires only where the text runs out.

| # | site | question | choice | reason |
|---|---|---|---|---|
| 1 | `substitution.py`, `substitute_by_profile` | §4.2 matches "the expected number of mutations per sequence", so the budget is a *mutation* count — but the text says "masked positions are iteratively infilled", which is a *mask* count. For the forced variant the two coincide, since blocking the original amino acid guarantees every masked position changes. For the unforced variant they do not: an MLM infilling a conserved framework position usually returns the residue already there. | Both readings, `top_up=True` the default. `False` masks exactly `budget` positions in one round and lands below budget; `True` keeps drawing until `budget` positions have actually *changed*, capped at `max_rounds`. | `True` is the one that satisfies §4.2's constraint, and it makes the forced variant a no-op relative to the other since forcing hits budget in round one. Either way the realised count is measured and reported per sequence (`n_masked`, `n_changed`, `rounds`, `hit_budget`), so a shortfall shows up in the metrics file instead of hiding in an assumption. The realised gap is not academic: on Ty1 our port spent 4.71 edits against Evotuning's 3.84 (`docs/findings.md`, "§4.2's last two methods land"). |
| 2 | `substitution.py`, `substitute_by_profile` | What "iteratively infilled" means. | Mask the whole drawn set, then fill one position at a time in random order, revealing each choice before the next. | Standard MLM iterative decoding, and the only reading under which "iteratively" adds anything over a single batched forward pass. The *random* order is a second reading on top: it exists so the last position filled is not systematically the C-terminal one. |
| 3 | `profile.py`, `sample_positions` | What to do with a zero-entropy column, which eq 12 gives weight 0. | Drawn, but only once every positive-weight position is spent. | A zero-entropy column means "nothing varies here", not "never touch this", and refusing them outright would make the budget unreachable on a highly conserved family — which is exactly the regime an antibody framework is in. |
| 4 | `profile.py`, `normalise_weights` | What distribution to use when the whole profile has zero entropy: a perfectly conserved family, or a family of one. | Uniform. | Eq 12's normalisation is undefined there; the alternative is an all-zero vector no position can be drawn from. It only fires where the paper's distribution does not exist. |
| 5 | `profile.py`, `column_entropy` | What `p_l(a)`'s denominator counts. The paper says "excluding gaps" and nothing about anything else. | Amino acids in the column, not sequences: gaps *and* every character that is not one of the 20 standard residues (`X`, `.`, `*`) are excluded. | "Excluding gaps" is the paper's word, so the denominator is amino acids. Extending that to ambiguity codes and the chain separator is ours, and it is the only consistent extension to a corpus that contains masks. An all-gap column has no distribution at all and is reported as `0.0`. |
| 6 | `substitution.py`, `matched_budget` | How to round the editor's mean edit distance into an integer budget. | `max(1, round(x))`. | The budget is a property of the *comparison*, not of the baseline — it is read off whatever the editor did — so it is named once here and both baselines call it, rather than each rounding its own way. `generation_eval` already applies the same rule for the paper's random-mutation baseline. The floor of 1 is there because a comparison at zero mutations compares nothing. |

## Not deviations

- **The template's own row is in the family alignment** the profile is built from. That is the
  paper's construction: the profile describes the family the seed belongs to.
- **`ungapped_weights` is an identity map in the default configuration**, because the runner projects
  the alignment onto template coordinates (`generation_eval.project_to_template`) and the template row
  then has no gaps. It is still written and tested as the general case: a profile from an external
  aligner — the more literal reading of "aligned column" — does put gaps in the template row, and
  indexing such a profile by sequence position would shift every substitution after the first gap.
- **The profile is built from the family's train part only**, never the scoring holdout. Enforced in
  `editjumps/pipeline/evaluate/evotune_baseline.py`, for the same reason as the EvoDiff-MSA baseline's
  alignment: a profile from the holdout is a profile of the answer.

## Tests

One file per code file, under `tests/`. Tests that need the MLM half or the stage's
budget/holdout plumbing are `editjumps/pipeline/evaluate/evotune_baseline.py`'s and live in
`editjumps/test/pipeline/test_evotune_baseline.py`.

[evoflows]: https://arxiv.org/abs/2603.11703
