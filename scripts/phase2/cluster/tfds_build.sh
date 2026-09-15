#!/bin/bash
# Build one cms_pf_phase2 sample's tfds from the eos v2 pkls, in waves of parallel
# configs, directly on /shared (see kube/README-train.md §4 for the sizing model:
# final ~19-30 KB/ev; shuffle temp transiently ~3x a config's final size; /scratch
# is a 60G-capped emptyDir -> never stage there).
#
#   tfds_build.sh SAMPLE WAVE_WIDTH [GUARD_GB] [CONFIG...]
#     SAMPLE      ttbar_nopu | qcd_nopu | zll_nopu   (builder module name)
#     WAVE_WIDTH  configs built in parallel per wave
#     GUARD_GB    abort if /shared free space (GB) is below this before a wave (default 120)
#     CONFIG...   configs to build (default 1..10) — pass the missing ones to resume
#
# Examples:  tfds_build.sh zll_nopu 10          # small sample, one wave
#            tfds_build.sh qcd_nopu 3 150      # big sample, guarded waves of 3
#            tfds_build.sh qcd_nopu 2 120 7 9  # resume just configs 7 and 9
set -uo pipefail
SAMPLE=${1:?usage: tfds_build.sh SAMPLE WAVE_WIDTH [GUARD_GB] [CONFIG...]}
WIDTH=${2:?need WAVE_WIDTH}
GUARD=${3:-120}
shift $(( $# < 3 ? 2 : 3 ))
CONFIGS=("$@"); [ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(1 2 3 4 5 6 7 8 9 10)

export HOME=/shared/mlpf-phase2/home
export PHASE2_PKL_SUBDIR=.
DATA_DIR=/shared/mlpf-phase2/tfds
LOGS=/shared/mlpf-phase2/logs
mkdir -p "$HOME" "$DATA_DIR" "$LOGS"
source /shared/envs/mlpf/bin/activate
cd /shared/particleflow

echo "=== tfds build $SAMPLE start $(date) | width=$WIDTH guard=${GUARD}G configs=${CONFIGS[*]} ==="
i=0
while [ $i -lt ${#CONFIGS[@]} ]; do
  wave=("${CONFIGS[@]:$i:$WIDTH}")
  free_gb=$(df -BG --output=avail /shared | tail -1 | tr -dc 0-9)
  echo "--- wave [${wave[*]}] start $(date); /shared free ${free_gb}G"
  if [ "$free_gb" -lt "$GUARD" ]; then
    echo "ABORT: ${free_gb}G free < ${GUARD}G guard before wave [${wave[*]}]"; exit 1
  fi
  for c in "${wave[@]}"; do
    (
      tfds build "mlpf/heptfds/cms_pf_phase2/${SAMPLE}" --config "$c" \
        --manual_dir /eos/user/f/fmokhtar/mlpf/phase2/pkl_links_v2 \
        --data_dir "$DATA_DIR" --overwrite \
        > "$LOGS/tfds_${SAMPLE}_$c.log" 2>&1 \
        && echo "config $c OK $(date)" || echo "config $c FAILED $(date)"
    ) &
  done
  wait
  i=$(( i + WIDTH ))
done
echo "=== tfds build $SAMPLE done $(date) ==="
du -sh "$DATA_DIR"/cms_pf_phase2_* 2>/dev/null
df -h /shared | tail -1
