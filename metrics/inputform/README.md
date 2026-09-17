# Input-form shift: joined `VH.VL` against a single V domain

Five scoring runs of one checkpoint (`eval_B_stock_appA`, mlp / `esm_lm_head`, 20 templates x 20
variants, holdout 200, ceiling 300, seed 0, `--disjoint-from-pairs`) that differ in the input form
and in nothing else. See `docs/findings.md`, "Measured: giving the editor its training input form
does not improve the reported numbers", and `editjumps/measurements/measure_input_form_shift.py`.

| file | what it is |
|---|---|
| `run-single.json` / `.fasta` | HER2-VH, single domain, clock 40 — reproduces `metrics/disjoint/editor-her2vh.json` bit-for-bit |
| `run-joined.json` / `.fasta` | the same 20 antibodies with the real trastuzumab VL appended, clock 40 |
| `run-joined_c76.json` / `.fasta` | the joined arm at clock 76, which restores the VH-block edit budget |
| `run-ty1_single.json` / `.fasta` | Ty1, single domain, clock 40 — reproduces `metrics/disjoint/editor-ty1.json` bit-for-bit |
| `run-ty1_joined.json` / `.fasta` | Ty1 with the same chain appended; a shape probe, since a VHH has no light chain |
| `vh-*.json` | the same runs rescored restricted to the VH coordinates, so both arms sit in the family's own frame — `frame.alignment_length` 119 on Ty1, 118 on HER2-VH. These carry no free-text `alignment` field; the structured `frame` block is the authority for them |
| `full-her2vh-joined-c40.json` | the joined arm on its whole 226-position alignment, kept to show what the restriction is worth |

**Paths in these files are redacted.** Their `frame.family_fasta` and `frame.fasta` fields recorded
absolute paths under a scratch directory on the machine the runs were taken on; the leading
directory is replaced with `<scratch>/`. The filenames are kept because they name the arm, but
nothing under `<scratch>/` is recoverable and these are not paths to follow.

**Read the `vh-*` files, not the `run-*` files, to compare arms.** A joined run's own artefact is
scored over ~107 appended columns that are identical in every generated and every reference
sequence: they cannot carry a difference between the sets but they are in the denominator of every
metric that averages over positions, and in the k-mer spectrum of every sequence. Spectrum MMD reads
2.038 that way against 1.130 restricted — a 1.8x inflation of the restricted number, and the reason
the joined arm reads as a 2.0x "degradation" against the single-domain arm's 1.021 when the whole
gap is the appended block.

The `full-` file is kept precisely so that comparison stays checkable rather than asserted.
