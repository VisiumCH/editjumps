#!/usr/bin/env bash
# Sensitivity of the Appendix-B metrics to the EVALUATION knobs the paper never states.
#
# "Our MMD lands inside the paper's best-method band (0.29-1.10)" is unfalsifiable as stated, because
# MMD moves 2.9x on a knob the paper does not report (one fixed 35M model, 300 generated sequences):
#
#   templates x variants   MMD
#   10 x 30                1.588
#   30 x 10                1.042
#   100 x 3                0.628
#   300 x 1                0.544
#
# So no model-to-model comparison means anything until the evaluation config is pinned. This measures
# what each knob is worth, so the pinning is informed. Findings (one 35M arm, Ty1, seed 0):
#
#   * REFERENCE SIZE converges and is safe to match to the paper: MMD 1.0382 / 1.0419 / 1.0235 at
#     holdout 200 / 1000 / 2000 -- flat, inside seed noise. Their reference is ~1,100 (Table 2
#     families of 3,335-24,304, split three ways in §4.2), so holdout 1000 matches and is past
#     convergence; the old 200 cap was not distorting anything.
#   * TEMPLATE COUNT is the dominant knob (above). §4.2 generates variants per INFERENCE sequence
#     over a ~1,100-sequence split, so it implies MANY templates with FEW variants -- not our 10x10.
#   * SPECTRUM-KERNEL k barely moves the model (MMD 1.025 at k=2 to 1.100 at k=5) but moves the
#     baselines a lot: it matters for any ratio, little for a raw comparison.
#   * Normalising to (floor - model) / (floor - ceiling) does NOT rescue comparability -- it is WORSE
#     than raw MMD (11.6 pt swing across holdout size, 20.0 pt across k, vs 0.09 and 0.075 raw). The
#     ceiling is real homologs at true distance ~0, so its biased V-statistic is almost all 1/n
#     diagonal bias (0.651 -> 0.352 as the reference grows) and dividing by it imports that bias.
#     Report floor and ceiling as context, never as a normaliser.
set -uo pipefail
cd "$(dirname "$0")/../.."

FAM_DIR=data/interim/seed_families
TY1="$FAM_DIR/Anti-SARS-CoV-2_VHH_Ty1.fasta"
HER2="$FAM_DIR/Anti-HER2_scFv_VH_trastuzumab.fasta"
OUT=metrics/sens
ARM=${ARM:-data/pretrain/eval_B_stock_appA}
mkdir -p "$OUT"

run () {  # run <label> <templates> <variants> <holdout> <k> <family>
  local label=$1 templates=$2 variants=$3 holdout=$4 k=$5 family=$6
  if [ -f "$OUT/$label.json" ]; then echo "  = $label (cached)"; return; fi
  uv run --group train editjumps generation-eval \
    --model-folder "$ARM" --rate-head mlp --q-head esm_lm_head \
    --clock 40 --n-steps 50 --seed "${SEED:-0}" \
    --n-templates "$templates" --n-variants "$variants" \
    --holdout-size "$holdout" --k "$k" --family-fasta "$family" \
    --metrics-path "$OUT/$label.json" >"$OUT/$label.log" 2>&1 \
    && echo "  + $label" || echo "  ! $label FAILED (see $OUT/$label.log)"
}

# Sweep 1 -- reference size. Fixed generated set, so every difference is estimator, not sampling.
echo "Sweep 1 - reference/holdout size (the Table 2 match):"
for h in 200 500 1000 2000; do run "s1_ty1_h$h" 30 10 "$h" 3 "$TY1"; done

# Sweep 2 -- template count at (near) constant total generated. The dominant knob.
echo "Sweep 2 - template count at ~300 generated:"
run s2_ty1_t10_v30  10 30 1000 3 "$TY1"
run s2_ty1_t30_v10  30 10 1000 3 "$TY1"
run s2_ty1_t100_v3 100  3 1000 3 "$TY1"
run s2_ty1_t300_v1 300  1 1000 3 "$TY1"

# Sweep 3 -- spectrum-kernel k. Also a fixed generated set; only the kernel changes.
echo "Sweep 3 - spectrum-kernel k:"
for k in 2 3 4 5; do run "s3_ty1_k$k" 30 10 1000 "$k" "$TY1"; done

# One second family, to check the trends are not a property of Ty1 alone.
echo "Cross-family check (HER2 VH):"
run x_her2_h200  30 10  200 3 "$HER2"
run x_her2_h1000 30 10 1000 3 "$HER2"

echo "done -> $OUT"
