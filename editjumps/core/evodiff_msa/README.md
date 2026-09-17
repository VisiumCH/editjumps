# EvoDiff-MSA Baseline (EvoFlows §4.2)

Integration of EvoDiff-MSA as evaluated in EvoFlows ([arXiv 2603.11703][evoflows]) §4.2. EvoDiff-MSA conditions on a multiple sequence alignment (MSA) rather than a profile HMM. Comparisons match the expected number of mutations per sequence.

To ensure benchmark integrity, the official upstream package, model weights, and tokenizer are used directly.

- `alignment.py`: Construction of A3M alignment files conditioned on the template query sequence.
- `sampling.py`: Position selection and residue sampling routines.
- `proposer.py`: Subprocess interface managing the isolated environment runner.

Due to conflicting NumPy constraints, EvoDiff-MSA runs in an isolated virtual environment (`.evodiff_env`) built by `editjumps/core/evodiff_msa/install_evodiff.sh` and interfaced via standard I/O pipes. Evaluation pipeline execution is handled by `editjumps/pipeline/evaluate/evodiff_msa_baseline.py`.

Sites are tagged in the code:

```bash
grep -rn "DEVIATION\|UNSPECIFIED" editjumps/core/evodiff_msa/*.py
```

## Deviations

Four things are ours rather than theirs, and every one of them changes the numbers.

| # | site | theirs | this code | reason |
|---|---|---|---|---|
| 1 | `proposer.py`, the whole budget-matched path | `generate_query_oadm_msa_simple`, the package's only MSA entry point, decodes the *entire* query row from a fully masked one. | `budget` positions of `x0` are masked and only those are decoded — the natural conditional use of an absorbing-state model. | Their function does not edit `x0`, it replaces it: measured on a 40-member synthetic VHH family at a matched budget of 2, it produced **74 mutations on a 125-residue query**, 37x the budget. So §4.2's constraint cannot be satisfied by calling it, and evodiff ships no MSA-inpainting entry point. Their function stays runnable verbatim as `--mode unconditional`, and the metrics file flags those rows `matched: false` so they are not read as the same question. `docs/findings.md`, "Their own entry point is 37x off a matched budget". |
| 2 | `sampling.py`, `PROPOSABLE` and `softmax_choice` | Their loop softmaxes over the 26 non-gap alphabet entries and adds `penalty_value` to the **last** column of that already-sliced tensor. | The 20 standard amino acids only, drawn from a temperature-scaled softmax in *our* seeded RNG. | That last column is `U` (selenocysteine), not the gap the penalty is named for, so the line does not do what its name says. Restricting to the 20 subsumes their intended gap exclusion and makes that line inoperative. Sampling on this side of the pipe is what makes a run reproducible from this project's single `--seed` rather than from torch's global generator in another interpreter. Their rule is unchanged in `unconditional` mode. |
| 3 | `sampling.py`, `uniform_weights` | §4.2 matches the mutation *count* and never says where a budget-matched run puts its mutations. EvoDiff-MSA has no eq-12 entropy profile to derive them from. | Positions drawn **uniformly**. | It is the choice the paper's own random-mutation baseline makes ("samples mutation positions … uniformly"), which leaves this baseline and that one differing in exactly one thing: what fills the masked position. |
| 4 | the alignment's membership (built in `editjumps/pipeline/evaluate/evodiff_msa_baseline.py`, written by `alignment.py`) | "each homolog set is split into train, inference, and holdout", and nothing about which part conditions the generator. | The family's *train* part only, never the scoring holdout. | Conditioning the generator on the holdout it is scored against would let it read the answer. |

## Where the paper is silent

| # | site | question | choice | reason |
|---|---|---|---|---|
| 1 | `proposer.py`, `MODELS` | Which released checkpoint. | The two order-agnostic (`oa-dm`) ones, `msa-oa-dm-maxsub` (default) and `msa-oa-dm-randsub`. | The D3PM MSA checkpoints decode over a fixed timestep schedule for the *whole* row, which has no partial-row form, so they cannot be budget-matched by masking a subset of positions — deviation 1 is unavailable to them. Each is mapped to the MSA subsampling it was trained with. |
| 2 | `proposer.py`, `max_seq_len` | What column cap to pass. | The alignment's own width. | `subsample_msa` slices a wider alignment at a *random offset*, which on an antibody V domain would cut the query in half. The argument exists only to bound the model's input. |
| 3 | `proposer.py`, `n_sequences` | How many rows to condition on. | 64 by default, and below 2 is refused. | An "MSA" of one row is the query alone, and the conditioning is the whole point of this baseline. |

## Not deviations

- **`expect_query`.** The a3m crosses a process boundary as a *path*, so between the write and the
  read the file is only as stable as the filesystem: a second run writing the same
  `template_%04d.a3m` replaces it, and the proposer then edits a different sequence's alignment while
  every column index still looks valid. Checking row 0 is a guard against our own concurrency, not a
  departure from anything.
- **stderr to a file rather than a pipe.** The runner prints a checkpoint-download progress bar on
  first use (~380 MB, Zenodo record 8045076) and an unread pipe of that size deadlocks the child.

## Tests

`tests/test_alignment.py` and `tests/test_sampling.py` cover the two stdlib halves.
`tests/test_proposer.py` covers `proposer.py`. Its tests drive a fake runner speaking `PROTOCOL`,
and that fake is shared with the stage-level tests of
`editjumps/pipeline/evaluate/evodiff_msa_baseline.py`, so it lives once in `editjumps/test/fakes.py`
rather than being duplicated into both.

[evoflows]: https://arxiv.org/abs/2603.11703
