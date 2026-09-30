#!/bin/bash
# Per-model evaluation + 4-curve validation plots, into a model-tagged EOS directory.
#   bash eval_and_plot.sh <MODEL> <EXPERIMENT_DIR> <TAG> [NTEST] [MAXFILES]
# Runs the pipeline `test` (produces preds_test on the disjoint test split) if not already
# present, then plot_mlpf_validation_v3.py (gen/target/TICL/MLPF) to
#   /eos/user/f/fmokhtar/mlpf/phase2/plots/<TAG>/
# TAG should name the model/spec, e.g. v3-s3-35.4M-30k or v3-s3ls-35.4M-layerscale.
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow

MODEL=${1:?usage: eval_and_plot.sh MODEL EXP TAG [NTEST] [MAXFILES]}
EXP=${2:?need experiment dir}
TAG=${3:?need tag}
NTEST=${4:-5000}
MAXFILES=${5:-1250}
OUT=/eos/user/f/fmokhtar/mlpf/phase2/plots/${TAG}
DATA_DIR=/shared/mlpf-phase2/tfds_v3

echo "=== eval_and_plot MODEL=$MODEL TAG=$TAG $(date) ==="
echo "EXP=$EXP"

# 1. test-split evaluation -> preds_test (skip if already there, unless FORCE_EVAL=1)
if [ "${FORCE_EVAL:-0}" != "1" ] && [ -d "$EXP/preds_test" ] && [ -n "$(ls -A "$EXP/preds_test" 2>/dev/null)" ]; then
  echo "preds_test already present -> skipping eval (set FORCE_EVAL=1 to redo)"
else
  CKPT=$(ls "$EXP"/checkpoints/checkpoint-*.pth 2>/dev/null | sort -V | tail -1)
  echo "evaluating checkpoint: $CKPT"
  python mlpf/pipeline.py --spec-file particleflow_spec.yaml \
    --model-name "$MODEL" --production-name cms_phase2_v3_ngt \
    --data-dir "$DATA_DIR" --experiment-dir "$EXP" \
    test --gpus 1 --make-plots --ntest "$NTEST" --load "$CKPT"
  echo "test eval rc=$?"
fi

# 2. tagged 4-curve validation plots (CPU)
mkdir -p "$OUT"
python scripts/phase2/plot_mlpf_validation_v3.py \
  --preds-glob "$EXP/preds_test/*/*.parquet" \
  --outdir "$OUT" --max-files "$MAXFILES"
echo "=== done TAG=$TAG -> $OUT ($(ls "$OUT"/*.pdf 2>/dev/null | wc -l) pdfs) $(date) ==="
