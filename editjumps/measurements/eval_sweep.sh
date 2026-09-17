#!/usr/bin/env bash
# Generation-only sweep over {arm} x {clock} x {seed family}. No training, so it costs minutes of
# CPU/MPS rather than GPU hours.
#
# Two assumptions the ledger's headline rests on:
#   1. Is the paper-comparable result robust across SEED FAMILIES? Every such number so far is Ty1.
#   2. Does the ARM RANKING survive the correct evaluation? Every arm comparison in the ledger used
#      the broken (mixed-family) reference and no clock normalization.
#
# Deliberately not a full cross product (5 arms x 4 clocks x 2 seeds = 40 runs, most cells answering
# nothing): one row per question.
set -uo pipefail
cd "$(dirname "$0")/../.."

FAM_DIR=data/interim/seed_families
TY1="$FAM_DIR/Anti-SARS-CoV-2_VHH_Ty1.fasta"
HER2="$FAM_DIR/Anti-HER2_scFv_VH_trastuzumab.fasta"
OUT=metrics/sweep
mkdir -p "$OUT"

complete () {  # complete <json> -- true only if the file parses AND carries the scored metric
  [ -f "$1" ] || return 1
  python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
p=d.get('defined_by_the_paper',{})
sys.exit(0 if isinstance(p.get('levenshtein_to_template'),(int,float)) else 1)
" "$1" 2>/dev/null
}

run () {  # run <label> <model-folder> <rate-head> <q-head> <clock> <family>
  local label=$1 folder=$2 rh=$3 qh=$4 clock=$5 family=$6
  # A file EXISTING is not evidence the run finished: a crash after the open leaves a truncated
  # JSON that this would then treat as a cached success forever. Require it to parse and to carry
  # the key every consumer reads.
  if complete "$OUT/$label.json"; then echo "  = $label (cached)"; return; fi
  uv run --group train editjumps generation-eval \
    --model-folder "$folder" --rate-head "$rh" --q-head "$qh" \
    --clock "$clock" --family-fasta "$family" \
    --n-templates 10 --n-variants 10 --n-steps 50 --holdout-size 200 \
    --pll-model facebook/esm2_t12_35M_UR50D --pll-positions 10 \
    --metrics-path "$OUT/$label.json" >/dev/null 2>&1
  # Assert on a POSITIVE artefact, not on the absence of an error. Exit 0 with no usable output is
  # a real outcome -- and "no error in the log" is also what a run that died before writing returns.
  if complete "$OUT/$label.json"; then echo "  + $label"; else echo "  ! $label FAILED"; fi
}

echo "Q1 — same arm and clock, two different seed families:"
run seed-ty1-B-c40   data/pretrain/eval_B_stock_appA   mlp    esm_lm_head 40 "$TY1"
run seed-her2-B-c40  data/pretrain/eval_B_stock_appA   mlp    esm_lm_head 40 "$HER2"

echo "Q2 — all four 35M arms, correct reference, matched edit budget:"
run arm-A-c40 data/pretrain/edit_flows_baseline   linear fresh       40 "$TY1"
run arm-B-c40 data/pretrain/eval_B_stock_appA     mlp    esm_lm_head 40 "$TY1"
run arm-C-c40 data/pretrain/eval_C_oas_appA       mlp    esm_lm_head 40 "$TY1"
run arm-D-c40 data/pretrain/eval_D_stock_linear   linear fresh       40 "$TY1"

echo "Q3 — clock sensitivity on one arm, to see how much the ranking could move with it:"
run arm-B-c25 data/pretrain/eval_B_stock_appA     mlp    esm_lm_head 25 "$TY1"
run arm-B-c60 data/pretrain/eval_B_stock_appA     mlp    esm_lm_head 60 "$TY1"
run arm-B-c00 data/pretrain/eval_B_stock_appA     mlp    esm_lm_head 0  "$TY1"

echo "done -> $OUT"
