# Edit Flows / EvoFlows Generative Editor

Implementation of discrete continuous-time Markov chain (CTMC) generative sequence editing following *Edit Flows* ([arXiv 2506.09018][editflows]) and EvoFlows ([arXiv 2603.11703][evoflows]).

A discrete flow between sequences of different lengths relies on positional correspondences established via global alignment. Sequences $(x_0, x_1)$ align to equal-length sequences $(z_0, z_1)$ over the amino-acid alphabet and gap sentinels. Each alignment column corresponds to an elementary mutation (substitution, insertion, deletion, or identity). The neural network predicts transition rate $\lambda$ and categorical token distribution $Q$ for each operation across positions.

- `alignment.py`, `path.py`: Global alignment and interpolation paths $(x_0, x_1) \to (z_0, z_1) \to z_t$
- `targets.py`: Per-column edit supervision (eq. 23)
- `loss.py`: Cross-entropy edit rate loss (eq. 6)
- `inference.py`: Euler $\tau$-leaping and Gillespie simulation (eqs 9–11)
- `stages.py`: Pipeline stage registries
- `metrics.py`: Rate and edit sequence metrics
- `mask.py`: Position masking and editing constraints
- `deterministic.py`: Ground-truth synthetic rule benchmark (EvoFlows §4.1)

See [`docs/evoflow_reproduction.md`](../../../docs/evoflow_reproduction.md) for architectural details and experimental findings. Specific deviations from cited papers are documented below and annotated in code comments via `DEVIATION` and `UNSPECIFIED`.

## Deviations

Five sites where the papers' text is not followed.

| # | site | paper | this code | reason |
|---|---|---|---|---|
| 1 | `path.py`, `EPS`; `targets.py`, `edit_targets` | Fig. 13 carries two sentinels, `epsilon_0_id` and `epsilon_1_id`, and its delete branch is `token_t != epsilon_0_id and token_1 == epsilon_1_id`. | ONE sentinel, so that branch cannot fire on an `(EPS, EPS)` column. | Their branch is True on a column where `z_t` has already taken the gap, i.e. it supervises a deletion where eq. 23's indicator `1[z_1^i != z_t^i]` says no edit is needed, and §3.2 says "The special token eps is a blank token that is not added to the vocabulary". Measured on 2,000 examples from the real OAS homolog pairs, their reading adds a mean **1.372 spurious deletions against 32.56 real edits**, touches **49.2%** of examples, and makes **4.04%** of its own delete supervision spurious. Their reading stays reachable as `edit_flows.loss: edit_flow_figure13` so the difference is ablatable rather than argued about. |
| 2 | `alignment.py`, `needleman_wunsch` | §3.2 states no scoring, and cites Henikoff & Henikoff 1992 (BLOSUM) exactly where it claims elementary edits "correspond to actual biochemical events" — so affine gaps + BLOSUM62 is the standard reading of what it cites. | Unit gap/mismatch cost, linear gaps. | The alignment's columns **are** the training labels and Table 1 is F1 per mutation *type*, so this is the deviation most likely to move a reported number. Both readings are implemented and selectable (`edit_flows.path: needleman_wunsch_blosum62`); the unit-cost one stays the default because every recorded number was trained against it, and because the measurement says the swap is not the lever it looked like: BLOSUM62 at BLASTP's 11/1 re-organises indels (gap runs 3.47 -> 2.41, mean run 1.66 -> 2.32) without re-weighting the classes (substitution 0.2396 -> 0.2438), while the gap penalty it forces us to *invent* moves the labels ~7x more than BLOSUM62 itself does (`docs/findings.md`, "② Alignment scoring" and its "Alignment scoring — correction"). |
| 3 | `alignment.py`, `needleman_wunsch` | EvoFlows' coupling is symmetric by construction (eq 8, `pi(x, x') = p(x) p(x')`), and our homolog pairs are built unoriented. | A fixed `diag > delete > insert` tie-break, so `needleman_wunsch(a, b)` is not the mirror of `needleman_wunsch(b, a)`. | `needleman_wunsch_affine(symmetric=True)` fixes it by canonical orientation rather than by hunting for a mirror-consistent tie-break inside the DP, so the residual asymmetry rate is zero rather than small — but it is not the default, for deviation 2's reason. Measured: the op mix is identical as drawn, under canonical orientation and with the pairs drawn reversed (0.7361 / 0.2396 / 0.0118 / 0.0125 in all three), so the asymmetry does not show up in the pooled label distribution even though it is real per pair. |
| 4 | `inference.py`, `next_event_time` | eq 9 gives the first-event CDF as `1 - exp(int_{t_n}^{t_n+T} u_s(x_tn\|x_tn) ds)` and eq 10 draws `U` and **numerically integrates** until `log(U) - int u_s ds = 0`: the state is held at `x_tn` while `s` varies, so the rate's time-dependence is integrated. | The rates are frozen at the current `(x, t)` and the waiting time drawn from that constant total, `dt = -ln(1-u)/R`. | Exact only if the generator is read as piecewise-constant between events, first-order accurate otherwise. The cost is bounded by the same quantity as Euler's error — the rate's variation over one waiting time — and Euler and Gillespie agree to **0.6-1.6%** at the clocks we run, which bounds it. Closing it exactly needs one numerical integration per event. `docs/findings.md`, "Our Gillespie is a deviation with a stated alternative, not an ambiguity we inherited". |
| 5 | `metrics.py` and `inference.py`, `clock_scale` | §3.3 names "clock normalization" and its purpose repeatedly, and never defines it. | `scale = clock / length`, applied to every rate. | Expected edits per Euler step are `h · Σλ · scale` and `Σλ` grows with length, so dividing by the current length is what makes the expected edit count a function of `clock` alone — which is what §4.2's "matching the expected number of mutations per sequence across methods" needs. Confirmed against real weights in `docs/findings.md`, "Clock normalization: the formula is confirmed, and the two calibrations never disagreed". It is duplicated in the two modules on purpose: the eval CLIs must report the multiplier without importing torch, and the samplers must apply it without importing the eval stack. |

## Where the papers are silent

Eight choices neither paper specifies.

| # | site | question | choice | reason |
|---|---|---|---|---|
| 1 | `stages.py`, `SCHEDULES` | Which `kappa(t)` to use. eq 5 requires only that it increase from 0 to 1; Edit Flows itself used cubic. | `linear` (`kappa = t`), with `cubic` selectable. | The loss weight is `dkappa/(1-kappa)`, so linear weights early `t` roughly evenly while cubic puts almost no weight before `t ≈ 0.5` and then rises steeply — they put the training signal in different places. Selectable so the reproduction can measure it rather than assume linear is theirs; measured, it "costs nothing measurable" (`docs/findings.md`, "The schedule deviation costs nothing measurable — the last one, closed"), but note the same document's warning that two runs of *one* schedule differ by more than some method gaps. |
| 2 | `alignment.py`, `BLOSUM62_GAP_OPEN` / `BLOSUM62_GAP_EXTEND` | What gap penalty pairs with BLOSUM62. The paper states none at all. | BLASTP's default 11/1, in BLASTP's convention (open once per gap *run*, extend once per gap position) rather than biopython's. | It has to be invented to use the scoring deviation 2's citation implies, and it is the part that moves the labels: 11/1 gives 2.41 gap runs per pair against 9.43 at ≈linear 1/1, and insertion/deletion prevalence 0.0115/0.0121 against 0.0305/0.0311. So the penalty matters ~7x more than the substitution matrix, which is the third reason the default was left alone. |
| 3 | `alignment.py`, `NON_RESIDUE_MATCH_SCORE` / `NON_RESIDUE_MISMATCH_SCORE` | How BOS/EOS, the `VH.VL` separator and `<pad>` score against each other. BLOSUM62 says nothing about them. | Identical non-residues take BLOSUM62's largest self-score (`W<->W` = 11), any mixed pair its smallest off-diagonal (-4). | They must not drift: the separator marks the chain boundary, and the tokenizer's BOS is what keeps every insertion's attach index non-negative. So the alignment anchors on them as hard as on a conserved tryptophan. |
| 4 | `inference.py`, `enabled_events` / `euler_trace` | Whether position 0 is editable. | Never deleted or substituted; insertion *after* it is allowed. | Position 0 is the tokenizer's BOS/CLS, and eq 13 defines `ins(x, i, a)` only for `i in {1..n(x)}`, so an insertion attaching before it is a state the parameterization cannot represent (`targets.py` refuses it outright). The leading token is still a valid attach point. |
| 5 | `deterministic.py` | Whether §4.1's `Sub(i + 5, H)` and `Ins(i - 2, S)` are 0- or 1-based. | 0-based. | The choice shifts every edit by one position but changes no count, so precision and recall per type are unaffected; per-position agreement with their figures would not be. |
| 6 | `deterministic.py` | What to do when `i + 5` runs past the end or `i - 2` is negative. | Drop the edit. | Clamping would pile several edits onto one terminal position and destroy the one-to-one `z0 -> z1` property the whole section rests on. |
| 7 | `deterministic.py` | What happens when one position is both deleted and substituted. | The deletion wins. | §4.1's own stated order — insertions, then deletions, then substitutions — decides it: the deletion happens first, so there is nothing left for the substitution to act on. Two A's five apart both targeting one position is idempotent (both write `H`) and needs no rule. |
| 8 | `sequences.py`, `PAIR_SEP` | How to join `VH` and `VL` into one line. Neither paper says, and **ESM-2 has no chain-separator convention at all** — where ESM defines multi-chain input it is ESMFold, which uses a colon. | `.`, ESM-2 token id 29 (30 is `-`, the gap character). | `.` and `-` are in ESM's alphabet only because it is shared with MSA Transformer, where they are A3M alignment characters (insertion-column and deletion gap). ESM-2 trained on UniRef50, which has neither, so **in stock ESM-2 the `.` embedding is untrained**: norm 1.025 against a 2.265 residue mean, z = -5.0, statistically identical to `<null_1>` (1.029) — the token `esm/data.py:110` creates purely to pad the vocabulary to a multiple of 8. That makes it a free slot rather than a wrong choice, and we do train it: in `eval_B_stock_appA` the `.` embedding moved **1.34x further from stock than the average residue** (0.2845 against 0.2118, cos 0.969), while `-`, `<null_1>`, `X` and `B` moved exactly 0.0000. **The trap:** the OAS-adapted trunk `esm2_oas` did *not* learn it (1.021 against stock's 1.025 — unmoved), because that corpus predates the `oas.pair_sep` fix and joined its chains with `:`, which ESM-2 tokenizes to `<unk>`. It was built WITH `--pair-chains`; see `docs/findings.md`, "The trap, and it is live". Any arm starting from `esm2_oas` and not subsequently trained on joined lines uses an untrained separator. The reported arm is safe because it starts from stock and trains on joined pairs. |

## Not deviations

Three things that look like departures.

- **`substitute_q` is indexed by token** in `loss.py`. Fig. 13's `substitute_q[x_t_index]` carries no
  token index, which multiplies the rate by a whole distribution; eq 15 specifies the single token's
  probability, which is what the code computes. A typo fix, not a choice.
- **`GAP_COST` and friends are module constants, not parameters.** `needleman_wunsch`'s traceback
  re-derives each move by comparing accumulated costs with `==`, which is exact for unit integers and
  unsound for anything else — a fractional cost makes no branch match and the bare `else` walks off
  the matrix. `_align_affine` keeps pointer matrices instead, which removes the class of bug rather
  than documenting it, so configurable costs live there. `docs/findings.md`, "Alignment costs".
- **`levenshtein` lives in `alignment.py`.** It is the same dynamic program as `needleman_wunsch`
  without a traceback, so it is one aligner among three rather than a stray utility.

## Not implemented

- **Guidance of any kind.** EvoFlows generation is unconditional, so nothing here steers the rates
  or `Q`: no CFG (Edit Flows' class-conditional scaling needs a class label this repository has no
  source for), no closed-form property proxy, no gradient guidance on a supervised head. The
  property-steered variants live in the upstream project; `docs/claims.md`'s "Why there is no target
  property" says where a property name is refused and why — two settings that raise, and one guard
  the trainer skips and says so, which is an omission rather than a refusal. Directional pair orientation (`build-homolog-pairs --direction improving`) is a
  data-construction change rather than guidance, but it needs the same registry and is refused here
  for the same reason — `none`, unoriented pairs, is EvoFlows' own setting.
- **Any schedule but `linear` and `cubic`**; `SCHEDULES` is the extension point.
- **Insertions or deletions in the §4.1 harness's own scoring**, which is
  `editjumps/pipeline/evaluate/deterministic_benchmark.py`'s problem rather than this folder's: a
  minimum-cost aligner recovers a cheaper edit script than the rules used, so a perfect editor
  cannot score 1.0 per class. Per-token provenance (`inference.py`'s `Provenance`) removes that
  ceiling entirely — 1.000 on all four classes (`docs/findings.md`, "§4.1 deterministic benchmark"
  and "Per-token provenance removes the ceiling entirely").

## Tests

One file per code file, under `tests/`. `deterministic.py` has no test file here: every test of it
goes through the §4.1 scoring harness in `editjumps/pipeline/evaluate/deterministic_benchmark.py`, so
those tests are that stage's and live in `editjumps/test/pipeline/test_deterministic_benchmark.py`.

[editflows]: https://arxiv.org/abs/2506.09018
[evoflows]: https://arxiv.org/abs/2603.11703
