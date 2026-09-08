# Postprocessing the full Phase-2 dataset via lxplus HTCondor

Mirrors moanwar's condor flow (`.staging/cms/`), adapted to our **self-contained**
`postprocessing_run3style.py` (it imports only numpy/awkward/uproot/fastjet — so it ships with the
job, no repo clone needed). Reads `/eos` NanoAOD directly, writes pkls back to your `/eos` via `xrdcp`.

63,626 files total (qcd 22668, ttbar 18488, zll 22470). At `FILES_PER_JOB=10` → ~6,363 jobs.

## Files
| file | role |
|---|---|
| `setup_venv.sh` | one-time: build the `/afs` venv (LCG_106 + `pip install fastjet`) |
| `submit.sh` | make per-sample filelists, create eos dirs, `condor_submit` one cluster/sample |
| `postprocess.sub` | condor submit description (parameterized by sample/filelist/njobs) |
| `wrapper.sh` | per-job: setup env, process this job's slice, `xrdcp` to eos (resumable) |

## Run (on lxplus)
```bash
# 0. get the code
git clone -b phase2 https://github.com/farakiko/particleflow ~/particleflow   # or: cd ~/particleflow && git pull
cd ~/particleflow/scripts/phase2/condor

# 1. one-time env (LCG_106 has uproot/awkward/numpy/vector; only fastjet is pip-installed)
MLPF_VENV=/afs/cern.ch/work/f/fmokhtar/private/mlpf_env bash setup_venv.sh

# 2. edit the knobs (EOS_PATH / MLPF_VENV) in postprocess.sub (environment=...) — must match your paths
# 3. submit (generates filelists, makes eos subdirs, submits qcd/ttbar/zll)
bash submit.sh
```

## Monitor / resume
```bash
condor_q                                        # queue status
tail -f logs/qcd_0pu_0.out                      # a job's log  (look for "ok=.. skip=.. fail=..")
xrdfs root://eosuser.cern.ch ls /eos/user/f/fmokhtar/mlpf/phase2/pkl_run3style/qcd_0pu | wc -l
```
Re-run `bash submit.sh` any time — `wrapper.sh` checks `xrdfs stat` and **skips files already on eos**,
so a resubmission only fills the gaps (failed/held/evicted jobs).

## Tuning / notes
- `FILES_PER_JOB` (default 10): larger = fewer, longer jobs. `+JobFlavour` ("workday" = 8h) must cover
  `FILES_PER_JOB × time-per-file`.
- Output pkls carry the 29-feature `Xelem`; consumed directly by `mlpf/heptfds/cms_pf_phase2`.
- If a condor node lacks `/eos` fuse for **reading**, change the filelist paths to xrootd URLs
  (`root://eoscms.cern.ch//eos/...`) — `uproot.open` and our `--input` accept them unchanged.
- This is the alternative to the k8s Job in `kube/` (batch-pod start was flaky there); condor is the
  proven path your colleague uses.
