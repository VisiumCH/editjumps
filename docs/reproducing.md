# Reproducing Reported Numbers

Procedures for regenerating reported evaluation results, along with comparability constraints governing cross-run and cross-study comparisons.

## Contents

- [Read `consolidate.py` and the directory READMEs first](#read-consolidatepy-and-the-directory-readmes-first)
- [The comparability rules](#the-comparability-rules)
- [The one-command table](#the-one-command-table)
- [Where the per-method artefacts live](#where-the-per-method-artefacts-live)
- [Regenerating the §4.2 table](#regenerating-the-42-table)
- [Regenerating each method](#regenerating-each-method)
- [Regenerating the §4.1 benchmark](#regenerating-the-41-benchmark)
- [Collecting results from a cloud job](#collecting-results-from-a-cloud-job)
- [Before you publish a comparison](#before-you-publish-a-comparison)

## Read `consolidate.py` and the directory READMEs first

Authoritative benchmark definitions and comparability constraints are defined in:

- **[`editjumps/pipeline/evaluate/consolidate.py`](../editjumps/pipeline/evaluate/consolidate.py)** — Consolidator script. The module docstring documents comparability columns, and the `DISJOINT` constant specifies primary evaluation artifacts.
- **[`metrics/disjoint/README.md`](../metrics/disjoint/README.md)** — Defines the §4.2 evaluation frame and validation criteria. Subdirectories under `metrics/` contain local documentation for each experiment.

## The comparability rules

Comparative evaluations require matching sample frames, coordinate systems, and reference sets.

### 1. `frame` is the primary partition; `disjoint` defines the benchmark table

`consolidate.py` computes `frame` from each run's `reference_in_training_pairs` metadata:

| `frame` | Definition |
|---|---|
| `disjoint` | Scoring reference shares zero sequences with training pairs. Primary reported benchmark. |
| `train_overlap_N` | Reference shares N sequences with training pairs (diagnostic / ablation runs). |
| `unrecorded` | Run predates automated partition tracking. |
| `published` | Values reported in the original paper (Figure 3). |

Primary benchmark rows are stored under **[`metrics/disjoint/`](../metrics/disjoint/README.md)**: eight JSON files containing twelve evaluation configurations (six comparison methods across two antibody families). Control baselines (`random mutations` and `random homolog pairing`) are recorded within editor result files to guarantee evaluation against identical references and alignments. All eight files share a consistent evaluation frame (20 templates × 20 variants, holdout 200, ceiling 300, seed 0).

Commands in [Regenerating the §4.2 table](#regenerating-the-42-table) generate these standardized cells.

### 2. `scope = within_study` must not be compared directly against published values

Consolidated outputs (`metrics/all_results.{json,csv}`) record evaluation `scope`:

| `scope` | Metrics | Valid for direct cross-study comparison? |
|---|---|---|
| `cross_study` | `spectrum_mmd` | Yes |
| `cross_study` (with domain caveats) | `edits_per_sequence`, `pairwise_pooled`, `kl_positional` | Reported alongside provenance; domain lengths differ |
| `within_study` | `covariance_agreement`, `mip_agreement`, ceilings, `js_positional`, `entropy_delta`, `kl_composition_pooled`, `profile_log_likelihood` | No |

**Structural vs. numerical checks:** `test_our_metrics_land_in_the_papers_plotted_ranges` verifies that within-study metrics fall within published order-of-magnitude ranges as a representation check, not as an assertion of equivalence.

**Divergence formulation:** `kl_composition_pooled` computes global amino-acid composition divergence across the full domain. Because close homologs have near-identical global compositions, values are lower (~0.0005) than positional distributions. `kl_positional` (`kl_divergence_positional`) evaluates per-column divergence on aligned positions, spanning 0.0480 to 0.0860 in the disjoint frame.

**Agreement metric reference dependency:** Agreement metrics (`covariance_agreement`, `mip_agreement`) scale with the reference set size $N$. Normalizing by ceiling does not eliminate this dependency: on synthetic sweeps, covariance agreement ratio drifts by ~28 percentage points as reference size increases from $N=20$ to $N=1600$. Because reference sizes differ across publications, cross-study comparisons of raw agreement values are invalid.

**Model-free controls:** Comparing model-free random pairing baselines indicates structural differences across studies:

| Metric | Published paper | This study | Ratio | Within published range? |
|---|---|---|---|---|
| `spectrum_mmd` | 0.6527 | 0.5348 | 1.22× | Yes ([0.289, 1.098]) |
| `edits_per_sequence` | 117.90 | 23.32 | 5.06× | No (antibody domain vs. pooled 100–360 aa proteins) |
| `pairwise_pooled` | 117.13 | 23.28 | 5.03× | No (domain length difference) |
| `kl_positional` | 0.0097 | 0.0272 | 2.79× | No |

`spectrum_mmd` is stable across sequence lengths and supports cross-study comparison. Distance-based metrics reflect sequence length differences between antibody variable domains (~121 aa) and pooled multi-family benchmarks (100–360 aa).
### 3. Spectrum MMD requires matching reference, generated, and template counts

Spectrum MMD is computed using the biased V-statistic, which retains diagonal $k(x, x)$ terms and carries an $O(1/n)$ positive sample-size bias. MMD is sensitive to the number of seed templates:

| Run Configuration | Templates × Variants | $N_{\text{generated}}$ | MMD |
|---|---|---|---|
| `arm-B-c40-n100u` | 10 × 10 | 100 | 1.633 |
| `arm-B-c40-n200` | 20 × 10 | 200 | 1.107 |
| `arm-B-c40-n300` | 30 × 10 | 300 | 1.038 |
| `arm-B-c40-n500` | 25 × 20 | 500 | 0.995 |
| `arm-B-c40-n500u` | 50 × 10 | 500 | 1.117 |
| `arm-B-c40-n1000` | 50 × 20 | 1000 | 1.118 |

Holding template count fixed at 50, doubling $N_{\text{generated}}$ from 500 to 1000 shifts MMD by 0.0006. In contrast, varying template count (25 vs. 50 templates at $N=500$) changes MMD by 0.122.

Comparisons of spectrum MMD must control for $N_{\text{reference}}$, $N_{\text{generated}}$, and $N_{\text{templates}}$. Diagnostic reporting also provides the unbiased U-statistic ($MMD^2$), which eliminates diagonal terms and may take small negative values within estimation noise.

### 4. Positional metrics require documented reference sample size

Agreement metrics (`covariance_agreement`, `mip_agreement`) and positional divergences (`js_divergence_positional`) require a documented `agreement_reference_n` to ensure score calibration. Legacy evaluation runs lacking this field cannot be compared directly against calibrated benchmark runs.

### 5. Positional metrics require matching coordinate alignment frames

Positional statistics are evaluated within projected alignment coordinate systems:

- `metrics/disjoint/`: Aligned to family template frames ($L=119$ for Ty1, $L=118$ for HER2-VH).
- `metrics/aligned/`: Projected onto baseline template reference ($L=121$ for Ty1, $L=122$ for HER2-VH).
- `metrics/agree/`: Evaluated in older template coordinate spaces; used only to verify reference scaling behavior.

Each evaluation JSON records its target width in the `alignment` field, parsed as `alignment_length` in consolidated outputs.

### 6. Uncertainty intervals reflect differing bootstrap methodologies

Evaluation tables report two classes of interval (`ci_kind`):

1. `mean_over_templates`: Empirical bootstrap interval over per-template averages.
2. `template_resample_spread`: Distribution spread observed when resampling templates for set-level metrics.

Because template resampling spreads for non-additive set statistics do not center symmetrically around point estimates, paired bootstrap hypothesis tests (`metrics/paired_agreement_bootstrap_*.json`) are used to evaluate statistical significance between models.

### 7. Resolution thresholds and loss comparability

- **Noise floor:** Paired runs under identical configurations (`sched-linear` and `faithful-appendixa`) exhibit run-to-run variation of up to ~0.034 in MIP agreement. Differences below this threshold are within operational noise.
- **Loss comparability across schedules:** Training loss weights edits by $\dot{\kappa}_t / (1 - \kappa_t)$. Linear and cubic schedules weight timepoints differently across trajectories (e.g. 1.33 vs. 0.19 at $t=0.25$), making cross-schedule loss magnitudes incomparable.

## The one-command table

```bash
make consolidate
```

Aggregates §4.2 evaluation artifacts into `metrics/all_results.json` and `metrics/all_results.csv`:

| Column | Description |
|---|---|
| `method`, `family`, `metric`, `value` | Primary metric result |
| `frame` | Primary partition (`disjoint`, `train_overlap_N`, `unrecorded`, `published`) |
| `scope` | Cross-study comparability scope (`cross_study`, `within_study`) |
| `n_templates` | Seed template count |
| `n_generated`, `n_reference` | Sample sizes for generated variants and reference sets |
| `reference_n` | Reference holdout size used for agreement metrics |
| `alignment_length` | Coordinate system sequence length |
| `ci_low`, `ci_high`, `ci_kind` | Uncertainty interval and estimation method |
| `recomputed` | Whether metric was recorded during run or recomputed from variant sequences |
| `source_file` | Source artifact path |

Additional options:
- `uv run editjumps consolidate --out-md metrics/appendix_table.md` outputs the formatted appendix markdown table.
- `--no-include-legacy` excludes runs with training set overlap (`train_overlap_*`).

## Where the per-method artefacts live

Benchmark source metrics are organized by experiment subfolder under `metrics/`:

| Directory | Contents | Comparability Notes |
|---|---|---|
| `metrics/disjoint/` | Authoritative §4.2 benchmark runs | Matched 20×20 frame, holdout 200, ceiling 300, seed 0 |
| `metrics/aligned/` | Editor runs aligned to baseline coordinates | Arm B (20×20); isolates reference holdout size (200 vs 800) |
| `metrics/evotune/` | Evotuned ESM baselines | Fixed output paths split by family prefix |
| `metrics/evodiff/` | EvoDiff-MSA baselines | Standardized and budget-matched masking runs |
| `metrics/schedule/` | Linear vs. cubic schedule comparisons | Matched 20k step budget on single hardware target |
| `metrics/agree/` | Reference-scaling diagnostic evaluations | Legacy coordinate frame |

## Regenerating the §4.2 table

Three targets, in this order, and the order is not cosmetic — the baselines match their mutation
budget to the editor's realised edit count and refuse to run without its cells:

```bash
make disjoint-editor         # the editor's 2 cells + the 2 model-free baselines nested in them
make disjoint-baselines      # EvoDiff-MSA and the two evotuned PLMs (needs make install-evodiff)
make consolidate             # the table over all of it
```

Each target carries the frame in the Makefile rather than leaving it to whoever types the command,
because every cell must share `n_templates`, `n_variants`, `holdout_size`, `ceiling_n`, `seed` and
the pairs file. `disjoint-editor`'s comment names the artefact field each of its flags was read off,
and a test asserts they still agree. The per-method commands in the next section are the older
single-cell forms: useful for one arm in isolation, and **not** a way to add a row to this table.

Costs, measured on this project's MPS box: ~15 min per family for the editor, ~3 h for the four
Branching Flows cells (45–80 min each, and that target is re-runnable — every template's variants
are cached, so a killed run resumes to the same sequences). `disjoint-editor` has no such cache.

### Execution provenance

New evaluation runs generate a `provenance` metadata block capturing git commit hash, repository dirty status, and CLI arguments via `editjumps.core.utils.write_metrics`.

## Regenerating each method

Independent evaluations require a trained model checkpoint and the `train` dependency group (`uv sync --all-groups`). Checkpoints can be downloaded from configured storage or trained locally from scratch.

### Jump-Process Editor (20 templates × 20 variants)

```bash
make generation-eval MODEL=<folder> N_TEMPLATES=20 N_VARIANTS=20 HOLDOUT_SIZE=200 \
                     RATE_HEAD=mlp Q_HEAD=esm_lm_head FAMILY=<family.fasta> OUT=metrics/aligned/aligned-ty1.json
```

Cloud GPU execution:

```bash
export DVC_BUCKET="gs://<your-bucket-name>"
make jobs-geneval RUN_TAG=<tag> MODEL_NAME=facebook/esm2_t12_35M_UR50D \
                  RATE_HEAD=mlp Q_HEAD=esm_lm_head METRICS_TAG=<tag> \
                  N_TEMPLATES=20 N_VARIANTS=20 HOLDOUT_SIZE=200
```

`RATE_HEAD` and `Q_HEAD` must match the model configuration. Set `CLOCK` to override `edit_flows.clock` in `params.yaml`.

### Evotuned-PLM Baselines

```bash
make evotune FAMILY=<family.fasta>                   # MLM-adaptation of ESM-2 to target family
make evotune-baselines FROM=metrics/generation_eval.json
```

`MUTATIONS` can be specified manually or read automatically via `FROM=<path>`. Evaluates both unforced and forced mutation baselines.

### EvoDiff-MSA Baseline

```bash
make install-evodiff        # isolated environment setup
make evodiff-baseline FROM=metrics/generation_eval.json
```

The default mode performs budget-matched masking against the seed template. `MODE=unconditional` runs unconditional query MSA generation.

### Control Baselines and Ceilings

`generation-eval` automatically computes two model-free reference baselines:
- `random mutations`: Uniform random substitution floor matching the target edit budget.
- `random homolog pairing`: Sampling real natural homologs from the family holdout set.

### Regenerating the §4.1 benchmark

```bash
make deterministic-benchmark
```

Runs pipeline stages `build_deterministic_pairs`, `train_deterministic_editor`, and `deterministic_benchmark` to evaluate token recovery precision and recall on synthetic sequences across clock rates.

## Collecting results from a cloud job

To fetch results and pipeline state from remote execution:

```bash
make fetch-repro                    # or RUN_TAG=<tag>
```

Staged pipeline locks and metrics are downloaded to review prior to staging into git.

## Before you publish a comparison

Verify comparability checklist:

1. **Partition frame:** Ensure both runs share the `disjoint` partition.
2. **Evaluation scope:** Verify `scope` compatibility before comparing to external literature.
3. **Reference sample size:** Match `agreement_reference_n` and `holdout_size`.
4. **Sample and template counts:** Match `n_generated`, `n_reference`, and `n_templates` for spectrum MMD.
5. **Coordinate frame:** Verify `alignment_length` matches for positional agreement metrics.
6. **Uncertainty interval:** Check `ci_kind` (`mean_over_templates` vs. `template_resample_spread`).
7. **Resolution floor:** Observe the ~0.034 run-to-run noise threshold on agreement metrics.
8. **Provenance:** Ensure all stages are registered in `editjumps/core/provenance.py` (`make provenance`).

## Where to go next
| Document | Description |
|---|---|
| [`gcp_setup.md`](gcp_setup.md) | Cloud infrastructure and instance configuration |
| [`known_issues.md`](known_issues.md) | Diagnostic symptoms and operational caveats |
