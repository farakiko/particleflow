#!/bin/bash
# Mirror the light run-time evaluation artifacts of /shared experiments to EOS so they
# are browsable via CERNBox: loss-curve plots, per-step validation plots, tensorboard
# event files, history jsons, logs, configs. Heavy artifacts (checkpoints/, preds_step_*/)
# stay on /shared only.
#   eos_sync.sh              one-shot sync
#   eos_sync.sh --loop [S]   sync every S seconds (default 300) until killed
set -uo pipefail
export HOME=/shared/mlpf-phase2/home
SRC=/shared/mlpf-phase2/experiments/
DST=/eos/user/f/fmokhtar/mlpf/phase2/experiments/

sync_once() {
  mkdir -p "$DST" 2>/dev/null || { echo "$(date) EOS not writable (quota?); skipping"; return 1; }
  # -rt (not -a): eos-fuse dislikes perm/owner replication
  rsync -rt --exclude "checkpoints/" --exclude "preds_step_*/" "$SRC" "$DST" \
    && echo "$(date) synced to $DST" || echo "$(date) rsync FAILED"
}

if [ "${1:-}" = "--loop" ]; then
  INT="${2:-300}"
  echo "eos_sync loop every ${INT}s: $SRC -> $DST"
  while true; do sync_once; sleep "$INT"; done
else
  sync_once
fi
