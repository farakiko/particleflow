#!/bin/bash
# Comparison plots for the clean-flag PU retrains. Assumes preds_test already exist
# (run eval_infer_clean.sh first). CLEAN_EXP=1 makes the eval scripts read the -clean
# experiment dirs. Output goes to a separate phase2_physics_clean folder so it never
# overwrites the buggy-flag plots.
#
#   NFILES=1000 bash scripts/phase2/cluster/eval_compare_clean.sh
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
export CLEAN_EXP=1
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow
git pull --ff-only origin phase2 2>&1 | tail -2

OUT=${OUT:-/shared/mlpf-phase2/eval_plots/phase2_physics_clean}
NFILES=${NFILES:-1000}
mkdir -p "$OUT"

# head models: indices in the shared MODELS list (0 baseline, 1 pu, 2 pu-rew/T0, 3 pu-soft/T2,
# 4 pu-ctx/T3, 5 pu-feats/T1). Per-head: model-vs-ideal (physics) and hard-vs-soft removal.
for IDX in 1 2 3 4 5; do
  python scripts/phase2/model_vs_ideal_pu.py --model "$IDX" --nfiles "$NFILES" --out "$OUT" || echo "model_vs_ideal idx $IDX rc=$?"
  python scripts/phase2/hard_vs_soft_pu.py   --model "$IDX" --nfiles "$NFILES" --out "$OUT" || echo "hard_vs_soft idx $IDX rc=$?"
done

# multi-model comparisons (skip models whose preds are absent, e.g. the no-mitigation baseline)
python scripts/phase2/pu_head_perf.py      --nfiles "$NFILES" --out "$OUT" || echo "pu_head_perf rc=$?"
python scripts/phase2/eval_phase2_plots.py --nfiles "$NFILES" --out "$OUT" --modes model,ideal || echo "eval_phase2_plots rc=$?"
python scripts/phase2/pu_threshold_scan.py --nfiles "$NFILES" --out "$OUT" || echo "pu_threshold_scan rc=$?"
echo "=== eval_compare_clean done -> $OUT $(date) ==="
