#!/bin/bash
# Launch one scaling-scan training as a k8s Job (docs/phase2.md 15).
#   bash scripts/phase2/cluster/launch_scan_job.sh s2            # 11.6M, MIG slice
#   bash scripts/phase2/cluster/launch_scan_job.sh s3            # 35.4M, MIG slice (~1.5 d)
#   bash scripts/phase2/cluster/launch_scan_job.sh s4 full       # 109M -- needs a full H100
# Runs pyg-cms-phase2-v3-<point> for NSTEPS steps on the v3 tfds (fixed-steps scan).
# "full" as the 2nd arg requests nvidia.com/gpu: 1 instead of the MIG slice.
# NSTEPS env overrides the step budget (default 30000); e.g. NSTEPS=100000 for a long run.
# When NSTEPS!=30000 the Job name + log get a -<N>k suffix so long runs do not collide.
set -euo pipefail
POINT=${1:?usage: launch_scan_job.sh s2|s3|s4|s3-ls [full]   (NSTEPS=100000 for long run)}
NSTEPS=${NSTEPS:-30000}
SUFFIX=""; [ "$NSTEPS" != "30000" ] && SUFFIX="-$((NSTEPS/1000))k"
GPU_LINE='nvidia.com/mig-1g.12gb: 1'
[ "${2:-}" = "full" ] && GPU_LINE='nvidia.com/gpu: 1'
MODEL="pyg-cms-phase2-v3-${POINT}"
NAME="pf-scan-${POINT}${SUFFIX}"

kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${NAME}
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        home: "eos"
        mount-eos: "true"
    spec:
      restartPolicy: Never
      volumes:
      - name: shared
        persistentVolumeClaim:
          claimName: shared
      containers:
      - name: train
        image: registry.cern.ch/ngt/pytorch:2.3.1
        command: ["/bin/bash", "-c"]
        args:
        - |
          MODEL=${MODEL} PRODUCTION=cms_phase2_v3_ngt DATA_DIR=/shared/mlpf-phase2/tfds_v3 \\
          bash /shared/particleflow/scripts/phase2/cluster/train_job.sh \\
            --num_steps ${NSTEPS} --val_freq 5000 --checkpoint_freq 10000 --nvalid 100000 \\
            2>&1 | tee -a /shared/mlpf-phase2/logs/scan_${POINT}${SUFFIX}.log
        env:
        - {name: PYTHONUNBUFFERED, value: "1"}
        # expandable_segments avoids the MIG NVML allocator assert (fragmentation OOM);
        # required for s3 (35.4M) to train at batch 64 on a 1g.12gb slice. See docs/phase2.md 15.
        - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
        volumeMounts:
        - name: shared
          mountPath: /shared
        resources:
          requests: {cpu: "6", memory: "16Gi"}
          limits:
            cpu: "16"
            memory: "48Gi"
            ${GPU_LINE}
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H100-NVL
EOF
echo "launched ${NAME} (${MODEL}, ${NSTEPS} steps); log: /shared/mlpf-phase2/logs/scan_${POINT}${SUFFIX}.log"
