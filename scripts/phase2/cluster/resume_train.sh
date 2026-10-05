#!/bin/bash
# Resumable long training. Always resumes from the LATEST checkpoint in a STABLE experiment
# dir, so any death (crash, eviction, 7-day pod cutoff) is recovered by simply restarting the
# pod/Job -- the entrypoint picks up where it left off. Validation every VALFREQ steps writes
# valid loss + jet metrics to <EXP>/history/step_<N>.json for easy monitoring.
#
# Env overrides (all optional):
#   MODEL      model-name in particleflow_spec.yaml   (default pyg-cms-phase2-v3-s3-ls)
#   EXP        stable experiment dir                  (default .../experiments/s3ls-prod)
#   NSTEPS     total optimizer steps                  (default 2530000 = 5 epochs @ batch 32)
#   GPUS       number of GPUs (DDP if >1)             (default 1)
#   BMULT      gpu_batch_multiplier (x16 = batch)     (default 2  -> batch 32)
#   LR         peak learning rate                     (default 0.0001)
#   VALFREQ    steps between validations              (default 10000)
#   CKPTFREQ   steps between checkpoints              (default 10000)
#   NVALID     events per validation                  (default 50000)
#   LOGNAME    log basename under logs/               (default s3ls_prod)
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow

MODEL=${MODEL:-pyg-cms-phase2-v3-s3-ls}
EXP=${EXP:-/shared/mlpf-phase2/experiments/s3ls-prod}
NSTEPS=${NSTEPS:-2530000}
GPUS=${GPUS:-1}
BMULT=${BMULT:-2}
LR=${LR:-0.0001}
VALFREQ=${VALFREQ:-10000}
CKPTFREQ=${CKPTFREQ:-10000}
NVALID=${NVALID:-50000}
LOGNAME=${LOGNAME:-s3ls_prod}
LOG=/shared/mlpf-phase2/logs/${LOGNAME}.log
mkdir -p "$EXP" /shared/mlpf-phase2/logs

LATEST=$(ls -v "$EXP"/checkpoints/checkpoint-*.pth 2>/dev/null | tail -1)
LOADARG=""; [ -n "$LATEST" ] && LOADARG="--load $LATEST"

nohup bash scripts/phase2/cluster/eos_sync.sh --loop 600 > /shared/mlpf-phase2/logs/eos_sync_${LOGNAME}.log 2>&1 &
SYNC=$!

echo "=== ${LOGNAME} start/resume $(date) | model=$MODEL gpus=$GPUS bmult=$BMULT lr=$LR nsteps=$NSTEPS load=${LATEST:-FRESH} ===" | tee -a "$LOG"
set +e
python mlpf/pipeline.py --spec-file particleflow_spec.yaml \
  --model-name "$MODEL" --production-name cms_phase2_v3_ngt \
  --data-dir /shared/mlpf-phase2/tfds_v3 --experiment-dir "$EXP" \
  train --gpus "$GPUS" --gpu_batch_multiplier "$BMULT" \
  --num_steps "$NSTEPS" --val_freq "$VALFREQ" --checkpoint_freq "$CKPTFREQ" --nvalid "$NVALID" \
  --hyperparameters.lr "$LR" $LOADARG \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}   # python's exit code, NOT tee's
kill "$SYNC" 2>/dev/null || true
bash scripts/phase2/cluster/eos_sync.sh
echo "=== ${LOGNAME} exit=$rc $(date) ===" | tee -a "$LOG"
exit $rc
