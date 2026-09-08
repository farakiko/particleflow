#!/bin/bash
# Run on lxplus from scripts/phase2/condor/. For each sample: build the filelist, create the eos
# output subdir, and condor_submit one cluster (ceil(nfiles / FILES_PER_JOB) jobs).
# Re-run any time to submit only what's missing (wrapper.sh skips files already on eos).
set -e
NANO=${NANO:-/eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/nano}
SAMPLES=${SAMPLES:-"qcd_0pu ttbar_0pu zll_0pu"}
FILES_PER_JOB=${FILES_PER_JOB:-10}
EOS_REDIR=${EOS_REDIR:-root://eosuser.cern.ch}
EOS_PATH=${EOS_PATH:-/eos/user/f/fmokhtar/mlpf/phase2/pkl_run3style}

mkdir -p filelists logs
for s in ${SAMPLES}; do
  find "${NANO}/${s}" -name '*.root' | sort > "filelists/${s}.txt"
  n=$(wc -l < "filelists/${s}.txt")
  njobs=$(( (n + FILES_PER_JOB - 1) / FILES_PER_JOB ))
  echo "${s}: ${n} files -> ${njobs} jobs (FILES_PER_JOB=${FILES_PER_JOB})"
  xrdfs "${EOS_REDIR}" mkdir -p "${EOS_PATH}/${s}" || true
  condor_submit sample="${s}" filelist="filelists/${s}.txt" njobs="${njobs}" postprocess.sub
done
echo "submitted. monitor: condor_q ; tail logs/*.out ; count: xrdfs ${EOS_REDIR} ls ${EOS_PATH}/<sample> | wc -l"
