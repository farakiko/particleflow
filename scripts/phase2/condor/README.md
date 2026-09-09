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

# 2. (recommended) smoke-test the new script on 2 real files before the full submit:
cp ../postprocessing_run3style.py .
head -2 <(find /eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/nano/zll_0pu -name '*.root') > filelists/test.txt
mkdir -p filelists logs; xrdfs root://eosuser.cern.ch mkdir -p /eos/user/f/fmokhtar/mlpf/phase2/pkl_links/zll_0pu
FILES_PER_JOB=2 CALO=links bash wrapper.sh 0 filelists/test.txt   # want: ok=2 skip=0 fail=0
rm -f postprocessing_run3style.py

# 3. submit (generates filelists, makes eos subdirs, submits qcd/ttbar/zll with CALO=links)
bash submit.sh
```
Knobs are set in `submit.sh` (CALO, EOS_PATH, MLPF_VENV, FILES_PER_JOB) and passed through to the jobs,
so mkdir and write-path stay consistent. Default: **CALO=links**, output `…/pkl_links/<sample>/`.

## Monitor / resume
```bash
condor_q                                        # queue status
tail -f logs/qcd_0pu_0.out                      # a job's log  (look for "ok=.. skip=.. fail=..")
xrdfs root://eosuser.cern.ch ls /eos/user/f/fmokhtar/mlpf/phase2/pkl_links/qcd_0pu | wc -l
```
Re-run `bash submit.sh` any time — `wrapper.sh` checks `xrdfs stat` and **skips files already on eos**,
so a resubmission only fills the gaps (failed/held/evicted jobs).

## Tuning / notes
- `FILES_PER_JOB` (default 10): larger = fewer, longer jobs. `+JobFlavour` ("workday" = 8h) must cover
  `FILES_PER_JOB × time-per-file`.
- `CALO=links` is the settled collection (see `docs/phase2_target_comparison.md`); `CALO=clue3d` produces
  the pre-linking variant (point `EOS_PATH` at a different dir, e.g. `…/pkl_clue3d`).
- Output pkls carry the **37-feature** `Xelem` (depth/timing/PCA + muon-ID/track-errors/track-density);
  consumed directly by `mlpf/heptfds/cms_pf_phase2`.
- If a condor node lacks `/eos` fuse for **reading**, change the filelist paths to xrootd URLs
  (`root://eoscms.cern.ch//eos/...`) — `uproot.open` and our `--input` accept them unchanged.
- This is the alternative to the k8s Job in `kube/` (batch-pod start was flaky there); condor is the
  proven path your colleague uses.
