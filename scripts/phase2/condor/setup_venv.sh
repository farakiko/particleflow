#!/bin/bash
# One-time (run on lxplus): build the venv the condor jobs source. LCG_106 already provides
# uproot/awkward/numpy/vector; only `fastjet` is missing, so we layer a --system-site-packages
# venv on top and pip-install fastjet. Put it on /afs so condor nodes can read it.
set -e
LCG_SETUP=${LCG_SETUP:-/cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh}
MLPF_VENV=${MLPF_VENV:-/afs/cern.ch/work/f/fmokhtar/private/mlpf_env}

source "${LCG_SETUP}"
python -m venv --system-site-packages "${MLPF_VENV}"
source "${MLPF_VENV}/bin/activate"
pip install --no-cache-dir fastjet
python -c "import uproot,awkward,numpy,vector,fastjet; print('venv OK:', '${MLPF_VENV}')"
echo "done. point wrapper.sh / postprocess.sub at MLPF_VENV=${MLPF_VENV}"
