#!/bin/bash
# Batch-postprocess NanoAOD root files -> pkl_run3style (parallel). Overwrites existing pkls.
#
# Portable: override any of these via environment, e.g. on the cluster:
#   PY=python NANO=/path/to/nano SAMPLES="ttbar_0pu qcd_0pu zll_0pu" NPROC=16 ./reprocess_run3style.sh
#
#   PY      python interpreter with uproot/awkward/numpy/fastjet (default: python)
#   REPO    repo root (default: auto-detected from this script's location)
#   NANO    dir containing <sample>/*.root (default: $REPO/data/cms/phase2/offline/Aug31/nano)
#   SAMPLES space-separated sample subdirs (default: ttbar_0pu qcd_0pu zll_0pu)
#   NPROC   parallel workers (default: 6)
set -u
PY=${PY:-python}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
SCRIPT=$REPO/scripts/phase2/postprocessing_run3style.py
NANO=${NANO:-$REPO/data/cms/phase2/offline/Aug31/nano}
SAMPLES=${SAMPLES:-"ttbar_0pu qcd_0pu zll_0pu"}
NPROC=${NPROC:-6}

do_one() {
  local root="$1"
  local dir="$(dirname "$root")/pkl_run3style"
  local base="$(basename "$root" .root)"
  mkdir -p "$dir"
  "$PY" "$SCRIPT" --input "$root" --output "$dir/run3style_$base.pkl" >/dev/null 2>>"$dir/../reprocess.err"
  echo "done $base"
}
export -f do_one; export PY SCRIPT

for s in $SAMPLES; do
  ls "$NANO/$s"/*.root
done | xargs -P "$NPROC" -I {} bash -c 'do_one "$@"' _ {} | \
  awk 'END{print "ALL DONE"} {c++; if(c%25==0) print c" files"}'
