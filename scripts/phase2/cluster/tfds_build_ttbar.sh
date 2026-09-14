#!/bin/bash
# Build cms_pf_phase2_ttbar_nopu tfds from the eos v2 pkls.
#
# Sizing (measured 2026-09-14 on 900 real events): final ArrayRecord ~19 KB/ev -> full
# ttbar ~87 GB. The build's shuffle stage transiently holds ~3x a config's final size as
# uncompressed temp buckets (~35 GB peak per config incl. finals).
# ⚠ /scratch is an auto-injected emptyDir capped at 60G — exceeding it EVICTS the pod
# (learned the hard way). So: build DIRECTLY on /shared, in waves of 3 configs
# (~105 GB transient peak vs ~200 GB free), no staging, no rsync.
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
export PHASE2_PKL_SUBDIR=.
DATA_DIR=/shared/mlpf-phase2/tfds
LOGS=/shared/mlpf-phase2/logs
mkdir -p "$HOME" "$DATA_DIR" "$LOGS"
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow
echo "=== tfds build ttbar start $(date) ==="
for wave in "1 2 3" "4 5 6" "7 8 9" "10"; do
  free_gb=$(df -BG --output=avail /shared | tail -1 | tr -dc 0-9)
  echo "--- wave [$wave] start $(date); /shared free ${free_gb}G"
  if [ "$free_gb" -lt 120 ]; then
    echo "ABORT: <120G free on /shared before wave [$wave]"; exit 1
  fi
  for i in $wave; do
    (
      tfds build mlpf/heptfds/cms_pf_phase2/ttbar_nopu --config $i \
        --manual_dir /eos/user/f/fmokhtar/mlpf/phase2/pkl_links_v2 \
        --data_dir "$DATA_DIR" --overwrite \
        > "$LOGS/tfds_ttbar_$i.log" 2>&1 \
        && echo "config $i OK $(date)" || echo "config $i FAILED $(date)"
    ) &
  done
  wait
done
echo "=== tfds build ttbar done $(date) ==="
du -sh "$DATA_DIR/cms_pf_phase2_ttbar_nopu" 2>/dev/null
df -h /shared | tail -1
