#!/bin/bash
# Post-training evaluation for the PU-mitigation tests (T2 soft target, T3 global context, ...).
# For each model: run the pipeline `test` to produce preds_test (if missing), then the PU eval
# scripts. Multi-model comparison scripts skip any model whose preds are not present yet.
#
#   SPECS="<expsubdir>:<model-name>:<idx> ..."  bash scripts/phase2/cluster/eval_pu_tests.sh
# <idx> is the model's index in the eval MODELS lists (0 baseline, 1 pu, 2 pu-rew/T0,
# 3 pu-soft/T2, 4 pu-ctx/T3). Runs on the pytorch training image (mlpf env + 1 GPU for inference).
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow
git pull --ff-only origin phase2 2>&1 | tail -2

EXPROOT=/shared/mlpf-phase2/experiments
DATADIR=/shared/mlpf-phase2/tfds
OUT=/shared/mlpf-phase2/eval_plots/phase2_physics
NTEST=${NTEST:-2000}
NFILES=${NFILES:-1000}
SPECS=${SPECS:-"ttbar200-heptv2-pu-soft:pyg-cms-phase2-200pu-heptv2-pu-soft:3 ttbar200-heptv2-pu-ctx:pyg-cms-phase2-200pu-heptv2-pu-ctx:4"}

for spec in $SPECS; do
  IFS=: read -r EXP MODEL IDX <<< "$spec"
  E="$EXPROOT/$EXP"
  echo "===== eval $MODEL ($EXP, idx $IDX) $(date) ====="
  if [ "${FORCE:-0}" != "1" ] && [ -d "$E/preds_test" ] && [ -n "$(ls -A "$E/preds_test" 2>/dev/null)" ]; then
    echo "preds_test present -> skip inference (FORCE=1 to redo)"
  else
    CKPT=$(ls -v "$E"/checkpoints/checkpoint-*.pth 2>/dev/null | tail -1)
    [ -z "$CKPT" ] && { echo "no checkpoint for $EXP -> skip"; continue; }
    echo "inference with $CKPT"
    python mlpf/pipeline.py --spec-file particleflow_spec.yaml \
      --model-name "$MODEL" --production-name cms_phase2_ngt \
      --data-dir "$DATADIR" --experiment-dir "$E" \
      test --gpus 1 --ntest "$NTEST" --load "$CKPT" || echo "inference rc=$?"
  fi
  # per-model physics comparisons (model vs ideal; hard mask vs soft weight)
  python scripts/phase2/model_vs_ideal_pu.py --model "$IDX" --nfiles "$NFILES" --out "$OUT" || true
  python scripts/phase2/hard_vs_soft_pu.py   --model "$IDX" --nfiles "$NFILES" --out "$OUT" || true
done

# multi-model comparisons across all models whose preds exist (missing ones are skipped)
python scripts/phase2/pu_head_perf.py       --nfiles "$NFILES" --out "$OUT" || true
python scripts/phase2/eval_phase2_plots.py  --nfiles "$NFILES" --out "$OUT" --modes model,ideal || true
echo "=== eval_pu_tests done -> $OUT $(date) ==="
