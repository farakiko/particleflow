#!/bin/bash
# Permutation feature-importance for the PU heads (which input features the head relies on).
# Runs on the pytorch training image (mlpf env + 1 GPU for inference). Uses the latest checkpoint
# of each -clean experiment, so it is meaningful once the clean retrains have progressed/finished.
#
#   NEVENTS=500 NREPEAT=3 bash scripts/phase2/cluster/pu_feat_importance.sh
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow
git pull --ff-only origin phase2 2>&1 | tail -2

NEVENTS=${NEVENTS:-500}
NREPEAT=${NREPEAT:-3}
NCONFIGS=${NCONFIGS:-2}
OUT=${OUT:-/shared/mlpf-phase2/eval_plots/phase2_physics}

python scripts/phase2/pu_feature_importance.py \
  --experiments-root /shared/mlpf-phase2/experiments \
  --data-dir /shared/mlpf-phase2/tfds_clean \
  --out "$OUT" \
  --nevents "$NEVENTS" --nrepeat "$NREPEAT" --nconfigs "$NCONFIGS" --groups
echo "=== pu_feat_importance done -> $OUT/pu_feat_importance $(date) ==="
