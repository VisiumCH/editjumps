# Reproducing EvoFlows

Technical documentation and implementation details for the discrete edit-flow matching framework, cross-referenced against published literature.

## 1. Specification and Scope

**EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering** (Deutschmann et al., Cradle, ICLR 2026 workshop; arXiv:2603.11703).
Builds on Meta/FAIR's **Edit Flows** (Havasi et al., NeurIPS 2025; arXiv:2506.09018).

The framework implements a discrete edit-flow model modeling homolog-to-homolog mutational trajectories (substitutions, insertions, deletions) over seed templates, predicting edit type and position conditioning on an ESM-2 encoder trunk. Evaluated on protein families from UniRef and OAS, including antibody variable domains (VHH and ScFv).

## 2. Core Formulation and Architecture

### 2.1 Discrete Flow Matching over Edits

A continuous-time Markov chain $X_t$ over sequences with local flow $u_t$:
$P(X_{t+h}=x' \mid X_t=x) = \delta[x'-x] + h \cdot u_t(x' \mid x) + o(h)$, transporting $x_0 \sim p_0$ to $x_1 \sim p_1$ along joint $\pi(x_0, x_1)$. Sequences are mapped to an extended alphabet $\mathcal{Z} = \mathcal{A} \cup \{\varepsilon\}$ ($\varepsilon$ = blank token), lifting pairs to equal-length $(z_0, z_1)$.

Conditional probability path:
$$p_t(z \mid z_0, z_1) = \prod_i [(1-\kappa_t)\delta_{z_0^i}(z^i) + \kappa_t \delta_{z_1^i}(z^i)]$$
with interpolation schedule $\kappa_t: [0, 1] \to [0, 1]$ (cubic $\kappa_t = t^3$ or linear $\kappa_t = t$).

Loss formulation (Bregman divergence):
$$\mathcal{L}(\theta) = \mathbb{E}_{\pi(z_0, z_1), t \sim U(0,1), p_t(z \mid z_0, z_1)} \left[ \sum_{x' \neq x} u_t^\theta(x' \mid x) - \frac{\dot{\kappa}_t}{1-\kappa_t} \sum_{i : z_t^i \neq z_1^i} \log u_t^\theta(\text{edit}_i \to z_1^i \mid x) \right]$$
where $x = x(z_t)$ strips blank tokens $\varepsilon$.

### 2.2 Homolog Pair Construction

Training trajectories are defined over homolog pairs:
- For seed $x^\dagger$, homolog families $R(x^\dagger)$ are derived via clustering or profile search. Target distribution $\pi(x, x') = p_{x^\dagger}(x) p_{x^\dagger}(x')$.
- Unordered pairs $(x_0, x_1)$ are globally aligned via Needleman–Wunsch into $(z_0, z_1) \in \mathcal{A}_\varepsilon^L \times \mathcal{A}_\varepsilon^L$, mapping differences to insertions, deletions, and substitutions.

### 2.3 Model Architecture

- **Trunk:** Pre-trained ESM-2 encoder trunk producing representations $h_t = \text{ESM2}(x_t) \in \mathbb{R}^{L \times D}$. As specified in Appendix A, encoder weights are finetuned jointly during training.
- **Time Conditioning:** Sinusoidal time embeddings processed via MLP, integrated via FiLM modulation: $\tilde{h}_t[i] = h_t[i] \odot (1 + \gamma_t) + \beta_t$.
- **Rate and Prediction Heads:** Total edit rates $u_t^\theta(x' \mid x) = \lambda_\bullet(\tilde{h}_t[i]) Q_\bullet(\tilde{h}_t[i], i, a)$ for substitutions and insertions, and $\lambda_{\text{del}}(\tilde{h}_t[i])$ for deletions. Heads can be configured as linear projections or MLPs with ESM language model heads (`rate_head=mlp`, `q_head=esm_lm_head`).

### 2.4 Sampling Algorithm

Sampling proceeds from template $x_0$ at $t=0$ via Gillespie jump simulation:
1. Compute total escape rate $R_{\text{tot}} = \sum_{x' \neq x_t} u_t(x' \mid x_t)$.
2. Sample next jump time $\Delta t$ by solving $\log U = -\int_t^{t+\Delta t} R_{\text{tot}}(s) ds$ for $U \sim \text{Uniform}(0, 1)$.
3. Update $t \leftarrow t + \Delta t$. If $t \ge 1$, terminate.
4. Select discrete edit transition proportionally to $u_t(\cdot \mid x_t)$ and update sequence $x_t$.
5. Apply clock normalization to rescale total edit rates and calibrate realized mutation budgets.

### 2.5 Evaluation Baselines

Evaluation (§4.2) benchmarks generated sets $\{x_1\}$ against family holdouts under matched mutation budgets:

| Method | Type | Description |
|---|---|---|
| Random mutations | Model-free | Uniform random substitutions matching realized mutation count |
| Random homolog pairing | Model-free | Natural homologs sampled from the family holdout |
| Evotuned PLM | Learned | Masked language model adapted to target family |
| Evotuned PLM (forced) | Learned | Family-adapted MLM with identity substitutions masked out |
| EvoDiff-MSA | Diffusion | Order-agnostic MSA diffusion baseline (Alamdari et al., 2024) |

#### Evotuned PLM Baselines

1. Target family adaptation via masked language modeling.
2. Mutation sites sampled from per-column entropy profiles $H(l) = -\sum_{a \in \mathcal{A}} p_l(a) \log(p_l(a) + \varepsilon)$ mapped to template coordinates.
3. Masked positions iteratively infilled by the adapted model with temperature scaling.

#### EvoDiff-MSA Baseline

- Uses upstream checkpoints (`msa-oa-dm-maxsub`, `msa-oa-dm-randsub`) executed in an isolated environment (`.evodiff_env`) via a line-oriented JSON protocol.
- Evaluated under budget-matched masking on the template query sequence (`matched: true`). Unconditional query generation is provided via `--mode unconditional`.

## 3. Implementation Parameters and Notes

1. **Alignment scoring:** Global Needleman–Wunsch alignment defaults to unit mismatch/gap scoring, with BLOSUM62 affine scoring supported via `edit_flows.path: needleman_wunsch_blosum62`.
2. **Trunk initialization:** Default configurations initialize from `esm2_t12_35M_UR50D` or OAS-pretrained checkpoints.
3. **Clustering:** Homolog families are partitioned to prevent sequence overlap between training, inference, and holdout splits.

## 4. Findings and Experimental Calibration

### 4.1 Measurement Controls and Estimator Calibration

Empirical controls identified critical sensitivities in baseline and metric formulations:

1. **Sample size bias in spectrum MMD:** The biased V-statistic carries an $O(1/n)$ positive sample-size bias. As template counts vary (from 10 to 25 templates at fixed reference size), values shift from 1.633 to 0.995. At 50 templates, increasing $N_{\text{generated}}$ from 500 to 1000 shifts MMD by only 0.0006, confirming that template count drives estimator variance. Evaluation reports both biased and unbiased U-statistics alongside sample counts.
2. **Ceiling calibration:** Evaluating the natural homolog ceiling requires matching generated sample sizes. Scoring $N=25$ natural sequences against $N=500$ model outputs artificially inflates the ceiling; calibrating sample sizes yields an expected ceiling of 0.549 against 0.995 for model outputs.
3. **Template-major baseline indexing:** Baseline generation is structured template-major (index $i$ maps to template $i // n_{\text{variants}}$). Paired edit-distance comparisons must preserve this indexing order.

### 4.2 Template Allocation Sensitivity in Spectrum MMD

Holding the generated sequence count fixed ($N=300$) and holdout reference size fixed ($N=1000$) on an ESM-2 35M trunk, varying template allocation produces substantial shifts in spectrum MMD:

| Templates × Variants | MMD |
|---|---|
| 10 × 30 | 1.588 |
| 30 × 10 | 1.042 |
| 100 × 3 | 0.628 |
| 300 × 1 | 0.544 |

Cross-study MMD comparisons are valid only when template and variant allocations are matched. Normalization schemes such as $(\text{floor} - \text{model}) / (\text{floor} - \text{ceiling})$ exhibit higher variance across holdout sizes and kernel parameters than raw MMD.

### 4.3 Architectural Ablations and Scaling

Evaluating combinations of trunk pre-training and prediction heads at 300 templates × 1 variant:

| Configuration | Linear Heads + Fresh Weights | Appendix-A MLP + LM Heads |
|---|---|---|
| Standard ESM-2 35M | 0.649 / 0.728 | 0.507 / 0.556 |
| OAS-Adapted ESM-2 35M | 0.508 / 0.572 | **0.489 / 0.527** |

*(Values: Spectrum MMD on Ty1 / HER2-VH).*

Appendix-A heads and domain-adapted trunks act as functional substitutes, each providing ~80–90% of the combined performance gain. Increasing trunk capacity from 35M to 650M yields 0.486 ± 0.021 versus 0.489 ± 0.040, showing that larger capacity accelerates convergence steps without substantially altering converged distributional fit.

### 4.4 Diversity, Novelty, and Metric Pooling

Evaluating generative diversity and novelty relative to seed templates:

| Method | Diversity | Novelty $\Delta$ |
|---|---|---|
| Natural homologs | 24.00 | −3.15 |
| Jump-Process Editor (Arms A–D) | 23.1–27.7 | +1.41 … +2.80 |
| Random mutations | 30.19 | +4.02 |

Natural homologs are closer to the overall family distribution than individual seed templates ($\text{Novelty } \Delta < 0$). Model-generated edits exhibit modest divergence from the reference manifold. Pairwise distance evaluations must account for pooled versus within-template aggregation: within-template distances are bounded by twice the distance-to-template, whereas pooled metrics reflect global sequence spread.

### 4.5 Replication Observations on Original Specification

1. **Deterministic benchmark (§4.1):** Synthetic edit recovery evaluates precision and recall across discrete edit categories, providing ground-truth verification of clock normalization.
2. **Anti-EphA2 seed citation:** Appendix C cites Roovers et al. (2011), which describes anti-EGFR nanobodies and contains no anti-EphA2 VHH sequence. Benchmarks focus on verified Ty1 and HER2-VH families.
3. **Metric definitions:** Several distributional metrics (entropy delta, JS divergence, profile log-likelihood) require explicit implementation assumptions regarding matrix reductions and coordinate projection.
4. **Dirichlet normalization:** Equation 24 divides by $N + \alpha$ to maintain standard Dirichlet distribution normalization.

### 4.6 Sequence Deduplication in Family Splits

Homolog searches against public databases can yield duplicate sequences across accessions. Partitioning families into training, inference, and holdout splits requires string deduplication prior to partitioning to prevent cross-split data leakage.

## Sources

- EvoFlows (Deutschmann et al., 2026, arXiv:2603.11703)
- Edit Flows (Havasi et al., 2025, arXiv:2506.09018)
- EvoDiff (Alamdari et al., 2024)
