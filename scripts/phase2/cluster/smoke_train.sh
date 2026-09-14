#!/bin/bash
# 300-step 1-GPU smoke training of pyg-cms-phase2-v1 on the ttbar tfds.
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
mkdir -p "$HOME"
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow
echo "=== smoke train start $(date) ==="
python mlpf/pipeline.py --spec-file particleflow_spec.yaml \
  --model-name pyg-cms-phase2-v1 --production-name cms_phase2_ngt \
  --data-dir /shared/mlpf-phase2/tfds \
  --experiments-dir /shared/mlpf-phase2/experiments \
  train --gpus 1 \
  --num_steps 300 --val_freq 100 --checkpoint_freq 100 --nvalid 1000 --ntest 500
echo "=== smoke train exit=$? $(date) ==="
