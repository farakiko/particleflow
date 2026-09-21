# Postprocessing the full Phase-2 dataset via lxplus HTCondor

Mirrors moanwar's condor flow (`.staging/cms/`). Two postprocessors ship with the job (no repo clone
on the node), selected via `PP_MODE`:
- **v2** = our `postprocessing_run3style.py` (numpy/awkward/uproot/fastjet; now includes GSF-track
  elements for electrons) — `PP_MODE=run3style`
- **v1** = the colleague's **verbatim** `postprocessing_ticl_ttbar_nopu.py` (also needs networkx/tqdm) —
  `PP_MODE=ticl`

Both read `/eos` NanoAOD directly and write pkls back to your `/eos` via `xrdcp`.

63,626 files total (qcd 22668, ttbar 18488, zll 22470). At `FILES_PER_JOB=10` → ~6,363 jobs.

## Files
| file | role |
|---|---|
| `setup_venv.sh` | one-time: build the `/afs` venv (LCG_106 + `pip install fastjet`) |
| `submit.sh` | make per-sample filelists, create eos dirs, `condor_submit` one cluster/sample |
| `postprocess.sub` | condor submit description (parameterized by sample/filelist/njobs) |
| `wrapper.sh` | per-job: setup env, process this job's slice, `xrdcp` to eos (resumable) |

## Run (on lxplus)

**v3 = the AGREED production (2026-09-22)**: moanwar's script with the gen↔sim ΔR matching
replaced by his own symmetric acceptance (`postprocessing_ticl_acceptance.py`; one-function
diff, docs/phase2.md 13). Sizing: ~6.6 MB/pkl → full 63,626-file production ≈ **420 GB** on eos.
Older recipes (v2 run3style, v1 verbatim, mohamed port) remain selectable via `PP_MODE`.

```bash
# 0. get the code
git clone -b phase2 https://github.com/farakiko/particleflow ~/particleflow   # or: cd ~/particleflow && git pull
cd ~/particleflow/scripts/phase2/condor

# 1. one-time env (already built): LCG_106 + fastjet + networkx/tqdm
MLPF_VENV=/afs/cern.ch/work/f/fmokhtar/private/mlpf_env bash setup_venv.sh

# 2. smoke-test the v3 mode on 2 real files before the full submit:
cp ../postprocessing_ticl_acceptance.py ../postprocessing_ticl_ttbar_nopu.py .
mkdir -p filelists logs
find /eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/nano/zll_0pu -name '*.root' | head -2 > filelists/test.txt
xrdfs root://eosuser.cern.ch mkdir -p /eos/user/f/fmokhtar/mlpf/phase2/pkl_test/zll_0pu
PP_MODE=acceptance FILES_PER_JOB=2 EOS_PATH=/eos/user/f/fmokhtar/mlpf/phase2/pkl_test bash wrapper.sh 0 filelists/test.txt  # want ok=2
rm -f postprocessing_ticl_acceptance.py postprocessing_ticl_ttbar_nopu.py

# 3. submit v3 (THE production)                   ->  pkl_v3/<sample>/acc_<stem>.pkl
PP_MODE=acceptance EOS_PATH=/eos/user/f/fmokhtar/mlpf/phase2/pkl_v3 bash submit.sh

# (legacy recipes)
# PP_MODE=run3style EOS_PATH=.../pkl_links_v2 bash submit.sh    # v2, ours
# PP_MODE=ticl      EOS_PATH=.../pkl_links_v1 bash submit.sh    # v1, his verbatim
```

> **Regenerating v2:** the GSF fix changed the v2 schema (37 raw `Xelem` fields, added `gsf_type`), so
> any earlier `pkl_links_v2` on eos is stale. `wrapper.sh` **skips files already on eos**, so clear the
> old dir first or the jobs will no-op:
> `xrdfs root://eosuser.cern.ch rm -r /eos/user/f/fmokhtar/mlpf/phase2/pkl_links_v2`  (submit.sh recreates it).

`submit.sh` passes `PP_MODE`/`EOS_PATH`/`CALO`/`FILES_PER_JOB` through to the jobs, so mkdir and
write-path stay consistent.

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
- v2 pkls carry a **37-field** raw `Xelem` (depth/timing/PCA + muon-ID/track-errors/track-density +
  `gsf_type`) → **38** model features after the `cms_pf_phase2` adapter (`phi`→`sin/cos`). v1 pkls use
  the colleague's own **36-field** schema (see `docs/phase2_feature_mapping.md`) and will need their own
  tfds feature-list before training.
- If a condor node lacks `/eos` fuse for **reading**, change the filelist paths to xrootd URLs
  (`root://eoscms.cern.ch//eos/...`) — `uproot.open` and our `--input` accept them unchanged.
- This is the alternative to the k8s Job in `kube/` (batch-pod start was flaky there); condor is the
  proven path your colleague uses.
