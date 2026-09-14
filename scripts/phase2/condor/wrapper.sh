#!/bin/bash
# HTCondor job wrapper: postprocess FILES_PER_JOB files (this job's slice of FILELIST) and
# xrdcp the pkls to EOS. Mirrors moanwar's wrapper_postprocessing_ticl.sh, adapted to our
# self-contained postprocessing_run3style.py (transferred with the job).
#
# args:  $1 = JOB_INDEX (condor $(Process))   $2 = FILELIST (one /eos root path per line)
# env (set via the .sub `environment`):
#   FILES_PER_JOB  files per condor job            (default 10)
#   MLPF_VENV      venv with fastjet on /afs        (default /afs/cern.ch/work/f/fmokhtar/private/mlpf_env)
#   LCG_SETUP      cvmfs LCG view (uproot/awkward/numpy/vector)
#   EOS_REDIR      xrootd redirector                (default root://eosuser.cern.ch)
#   EOS_PATH       output base dir on eos           (default /eos/user/f/fmokhtar/mlpf/phase2/pkl_links)
#   CALO           calo collection for postprocessing (default links; clue3d for the other variant)
set -u
JOB_INDEX=$1
FILELIST=$2
FILES_PER_JOB=${FILES_PER_JOB:-10}
MLPF_VENV=${MLPF_VENV:-/afs/cern.ch/work/f/fmokhtar/private/mlpf_env}
LCG_SETUP=${LCG_SETUP:-/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh}
EOS_REDIR=${EOS_REDIR:-root://eosuser.cern.ch}
EOS_PATH=${EOS_PATH:-/eos/user/f/fmokhtar/mlpf/phase2/pkl_links}
CALO=${CALO:-links}
PP_MODE=${PP_MODE:-run3style}          # run3style = ours (v2, +GSF) | ticl = colleague's verbatim script (v1) | mohamed = our port (v1/v3)
GEN_FILTER=${GEN_FILTER:-on}           # mohamed only: on = v1 (gen-match filter)  |  off = v3 (endcap cut, keep cuts+frag)
case "${PP_MODE}" in
  run3style) PP_SCRIPT=postprocessing_run3style.py;       OUT_PREFIX=run3style; PP_ARGS=(--calo "${CALO}") ;;
  ticl)      PP_SCRIPT=postprocessing_ticl_ttbar_nopu.py; OUT_PREFIX=ticl;      PP_ARGS=() ;;   # colleague's script; gen-filter baked in
  mohamed)   PP_SCRIPT=postprocessing_mohamed.py;         OUT_PREFIX=mohamed;   PP_ARGS=(--gen-filter "${GEN_FILTER}") ;;
  *) echo "unknown PP_MODE=${PP_MODE} (want run3style|ticl|mohamed)"; exit 1 ;;
esac

echo "=== job ${JOB_INDEX} on $(hostname) mode=${PP_MODE} args=${PP_ARGS[*]} $(date) ==="
# LCG/venv setup scripts are not `set -u`-clean (e.g. reference an unset COMPILER) -> relax around sourcing
set +u
source "${LCG_SETUP}"
source "${MLPF_VENV}/bin/activate"
set -u
IMPORTS="uproot,awkward,numpy,fastjet,vector"
[ "${PP_MODE}" = "ticl" ] && IMPORTS="${IMPORTS},networkx,tqdm"   # colleague's script needs networkx/tqdm
python3 -c "import ${IMPORTS}" || { echo "ENV MISSING PACKAGES (see setup_venv.sh)"; exit 1; }

START=$(( JOB_INDEX * FILES_PER_JOB + 1 ))
END=$(( START + FILES_PER_JOB - 1 ))
echo "lines ${START}-${END} of ${FILELIST}"

ok=0; fail=0; skip=0
while IFS= read -r ROOT; do
  [ -z "$ROOT" ] && continue
  base=$(basename "$ROOT" .root)
  sample=$(basename "$(dirname "$ROOT")")          # qcd_0pu / ttbar_0pu / zll_0pu
  out="${OUT_PREFIX}_${base}.pkl"
  dst="${EOS_PATH}/${sample}/${out}"
  # resumable: skip if already on eos
  if xrdfs "${EOS_REDIR}" stat "${dst}" >/dev/null 2>&1; then
    echo "skip (exists) ${sample}/${out}"; skip=$((skip+1)); continue
  fi
  echo "--- ${ROOT}"
  if python3 "${PP_SCRIPT}" --input "${ROOT}" --output "${out}" "${PP_ARGS[@]}"; then
    if [ -f "${out}" ]; then
      if xrdcp -f "${out}" "${EOS_REDIR}/${dst}"; then ok=$((ok+1)); else echo "XRDCP FAIL ${out}"; fail=$((fail+1)); fi
    else
      echo "empty/corrupt input (no output) -> skip ${ROOT}"; skip=$((skip+1))
    fi
  else
    echo "POSTPROC FAIL ${ROOT}"; fail=$((fail+1))
  fi
  rm -f "${out}"
done < <(sed -n "${START},${END}p" "${FILELIST}")

echo "=== job ${JOB_INDEX} done: ok=${ok} skip=${skip} fail=${fail} $(date) ==="
[ "${fail}" -eq 0 ]
