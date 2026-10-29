#!/bin/bash
# Inference (preds_test) for the 5 clean-flag PU retrains, reading the clean tfds.
# Produces preds_test under each -clean experiment dir; the comparison plot scripts
# (pu_head_perf / eval_phase2_plots / model_vs_ideal_pu / hard_vs_soft_pu) are run
# separately once their MODELS lists point at the -clean dirs.
#
#   NTEST=2000 bash scripts/phase2/cluster/eval_infer_clean.sh
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow
git pull --ff-only origin phase2 2>&1 | tail -2

EXPROOT=/shared/mlpf-phase2/experiments
DATADIR=/shared/mlpf-phase2/tfds_clean
NTEST=${NTEST:-2000}

# <exp-subdir>:<spec model-name>  (clean retrains)
SPECS=${SPECS:-"\
ttbar200-heptv2-pu-clean:pyg-cms-phase2-200pu-heptv2-pu \
ttbar200-heptv2-pu-rew-clean:pyg-cms-phase2-200pu-heptv2-pu-rew \
ttbar200-heptv2-pu-feats-clean:pyg-cms-phase2-200pu-heptv2-pu-feats \
ttbar200-heptv2-pu-soft-clean:pyg-cms-phase2-200pu-heptv2-pu-soft \
ttbar200-heptv2-pu-ctx-clean:pyg-cms-phase2-200pu-heptv2-pu-ctx"}

for spec in $SPECS; do
  IFS=: read -r EXP MODEL <<< "$spec"
  E="$EXPROOT/$EXP"
  echo "===== inference $MODEL ($EXP) $(date) ====="
  if [ "${FORCE:-0}" != "1" ] && [ -d "$E/preds_test" ] && [ -n "$(ls -A "$E/preds_test" 2>/dev/null)" ]; then
    echo "preds_test present -> skip (FORCE=1 to redo)"
    continue
  fi
  CKPT=$(ls -v "$E"/checkpoints/checkpoint-*.pth 2>/dev/null | tail -1)
  [ -z "$CKPT" ] && { echo "no checkpoint for $EXP -> skip"; continue; }
  echo "load $CKPT"
  python mlpf/pipeline.py --spec-file particleflow_spec.yaml \
    --model-name "$MODEL" --production-name cms_phase2_ngt \
    --data-dir "$DATADIR" --experiment-dir "$E" \
    test --gpus 1 --ntest "$NTEST" --load "$CKPT" || echo "inference rc=$?"
done
echo "=== eval_infer_clean done $(date) ==="
