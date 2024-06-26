#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 80G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-04-30
cd ~/particleflow

singularity exec -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
    --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --conv-type attention --num-epochs 100 --gpu-batch-multiplier 40 --num-workers 4 --prefetch-factor 50 --checkpoint-freq 1 --comet \
    --load experiments/pyg-cms_20240430_094836_751206/checkpoints/checkpoint-25-17.631161.pth
