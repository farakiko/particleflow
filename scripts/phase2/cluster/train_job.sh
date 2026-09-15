#!/bin/bash
# Full-length training of pyg-cms-phase2-v1 — entrypoint of kube/train-job.yml.
# Extra args are passed through to the pipeline (e.g. --num_steps 100000 --nvalid 100000).
# An eos_sync loop mirrors the run-time eval artifacts to EOS throughout.
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
mkdir -p "$HOME" /shared/mlpf-phase2/logs
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow
nohup bash scripts/phase2/cluster/eos_sync.sh --loop 300 > /shared/mlpf-phase2/logs/eos_sync_job.log 2>&1 &
SYNC_PID=$!
echo "=== big train start $(date) | args: $* ==="
set +e
python mlpf/pipeline.py --spec-file particleflow_spec.yaml \
  --model-name pyg-cms-phase2-v1 --production-name cms_phase2_ngt \
  --data-dir /shared/mlpf-phase2/tfds \
  --experiments-dir /shared/mlpf-phase2/experiments \
  train --gpus 1 "$@"
rc=$?
set -e
kill "$SYNC_PID" 2>/dev/null || true
bash scripts/phase2/cluster/eos_sync.sh   # final mirror
echo "=== big train exit=$rc $(date) ==="
exit $rc
