# Training MLPF Phase-2 on the NGT k8s cluster

Trains on the **v2 (Run3-style) pkls** produced by lxplus condor onto
`/eos/user/f/fmokhtar/mlpf/phase2/pkl_links_v2/{ttbar_0pu,qcd_0pu,zll_0pu}/` (37-field raw
`Xelem` → 38 model features). Design decisions (see also `docs/phase2.md` §11):

| choice | decision |
|---|---|
| data format | **tfds/ArrayRecord** via the existing `mlpf/heptfds/cms_pf_phase2` builders (canonical `mlpf` pipeline; random access, no load-all-into-RAM) |
| tfds location | **`/shared/mlpf-phase2/tfds`** (500Gi RWX PVC; eos-fuse is too slow for random reads). ttbar first — full 3-sample set may exceed the PVC |
| python env | **persistent uv venv `/shared/envs/mlpf`** built once from the repo `pyproject.toml` (`uv sync` ⇒ repo-pinned torch/cu128 + tfds + array-record); pods only need the base image + this venv |
| entry point | `python mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-cms-phase2-v1 --production-name cms_phase2_ngt` |
| GPU | 1× `nvidia.com/mig-1g.12gb` MIG slice on H100-NVL (known-good from `kube/pod.yml`) |

Cluster facts: no kubectl access to `nodes` (namespace-scoped RBAC); no ResourceQuota set;
`shared` PVC is RWX so build + train pods can mount it together. **Never force-delete**
anything; plain `kubectl delete` only.

## 0. Pod

```bash
kubectl apply -f kube/train-pod.yml
kubectl get pod pf-train-smoke -w          # wait Running; if Pending >2-3 min:
kubectl describe pod pf-train-smoke        #   events list per-node reasons (reveals GPU availability)
kubectl exec -it pf-train-smoke -- bash
```

## 1. Image + GPU sanity (inside the pod)

```bash
nvidia-smi                                  # MIG 1g.12gb visible; note the DRIVER version (need >=525 for cu128 wheels)
python3 -V; python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
df -h /shared /eos; ls /shared
```

## 2. First-time inventory (sizes drive the storage plan)

```bash
du -sh /shared/* 2>/dev/null | sort -h                     # what the 500Gi already holds
for s in ttbar_0pu qcd_0pu zll_0pu; do
  d=/eos/user/f/fmokhtar/mlpf/phase2/pkl_links_v2/$s
  echo "$s: $(ls $d 2>/dev/null | wc -l) pkls, $(du -sh $d 2>/dev/null | cut -f1)"
done
```

Schema check on one production pkl (must print **37** fields, incl. `gsf_type`,
`min_dR_track`…`n_trk_dR05`; the *local laptop* pkls are the stale 28-field schema):

```bash
python3 - <<'EOF'
import pickle, glob
f = sorted(glob.glob("/eos/user/f/fmokhtar/mlpf/phase2/pkl_links_v2/ttbar_0pu/*.pkl"))[0]
ev = pickle.load(open(f, "rb"))[0]
print(f, len(ev["Xelem"].dtype.names), ev["Xelem"].dtype.names)
EOF
```

## 3. One-time env build (`/shared/envs/mlpf`)

Scripted: `nohup bash scripts/phase2/cluster/env_build.sh > /shared/mlpf-phase2/logs/env_build.log 2>&1 &`
(from `/shared/particleflow` after a `git pull`; Farouk pushes, the agent never does).
Built + verified 2026-09-14: torch **2.11.0+cu128** sees the MIG slice (driver 590.48),
`tensorflow_datasets/array_record/fastjet/comet_ml` import, MLPF instantiates from the spec
at **2.72M params / input_dim 38 / 6 classes**.

Hard-won details baked into the script:
- **`HOME=/shared/mlpf-phase2/home`** — the EOS home is **over quota** (even 0-byte writes
  fail), so every dotfile/cache write must stay off $HOME-on-eos. All cluster scripts set this.
- `UV_CACHE_DIR` on `/scratch` (node-local NVMe, throwaway), standalone python + venv on `/shared`.
- `uv pip` ignores `UV_PROJECT_ENVIRONMENT` → the editable install must pass
  `--python /shared/envs/mlpf/bin/python`.

Fallback if a node's driver were too old for cu128 wheels (not the case here): venv with
`--system-site-packages` over the image torch 2.3.1 — version-skewed, smoke-test only.

## 4. Build a sample's tfds (guarded waves of parallel configs)

`scripts/phase2/cluster/tfds_build.sh SAMPLE WAVE_WIDTH [GUARD_GB] [CONFIG...]` — e.g.

```bash
nohup bash scripts/phase2/cluster/tfds_build.sh zll_nopu 10       > /shared/mlpf-phase2/logs/tfds_build_zll.log 2>&1 &
nohup bash scripts/phase2/cluster/tfds_build.sh qcd_nopu 3 150    > /shared/mlpf-phase2/logs/tfds_build_qcd.log 2>&1 &
```

Per-config logs: `logs/tfds_<sample>_<i>.log`. Each builder splits the file list into
10 configs, 90/10 train/test *by file*. `PHASE2_PKL_SUBDIR=.` adapts to the eos layout.

Sizing model (measured): final ArrayRecord ≈ **19 KB/ev** (ttbar; scales with
elements/event) → ttbar came out at **70 GB**. The shuffle stage transiently holds ~3×
a config's final size as uncompressed temp buckets, so the script builds **directly on
/shared** in width-limited waves with a free-space guard before each wave (⚠ never
stage on /scratch — it's a 60G-capped emptyDir; exceeding it EVICTS the pod).
To resume after an abort/failure, rerun with the missing config numbers as arguments.

## 5. Smoke training (1 GPU, ~minutes)

Scripted: `nohup bash scripts/phase2/cluster/smoke_train.sh > /shared/mlpf-phase2/logs/smoke_train.log 2>&1 &`, which runs:

```bash
cd /shared/particleflow && source /shared/envs/mlpf/bin/activate
python mlpf/pipeline.py --spec-file particleflow_spec.yaml \
  --model-name pyg-cms-phase2-v1 --production-name cms_phase2_ngt \
  --data-dir /shared/mlpf-phase2/tfds \
  --experiments-dir /shared/mlpf-phase2/experiments \
  train --gpus 1 \
  --num_steps 300 --val_freq 100 --checkpoint_freq 100 --nvalid 1000 --ntest 500
```

Pass criteria: loss decreases over 300 steps; checkpoint written under
`/shared/mlpf-phase2/experiments/…`; GPU actually used (`nvidia-smi` in a second exec).
Real runs afterwards: drop the caps (defaults: `num_steps=100000`, full nvalid) and move
to a Job manifest instead of the interactive pod.

## 6. EOS mirror of results (browse via CERNBox)

`scripts/phase2/cluster/eos_sync.sh` mirrors the light evaluation artifacts of every
experiment (plots_step_*, history, tensorboard `runs/`, logs, configs — ~2 MB/run) to
**`/eos/user/f/fmokhtar/mlpf/phase2/experiments/`**; `checkpoints/` and `preds_step_*/`
stay on /shared. A background loop runs on the training pod:
`nohup bash scripts/phase2/cluster/eos_sync.sh --loop 300 > /shared/mlpf-phase2/logs/eos_sync.log 2>&1 &`
(re-arm after pod restarts; it skips gracefully if EOS quota is full again).

## 7. Monitor / cleanup

```bash
kubectl exec pf-train-smoke -- nvidia-smi
kubectl exec pf-train-smoke -- tail -20 /shared/mlpf-phase2/tfds_ttbar_1.log
kubectl port-forward pod/pf-train-smoke 6006:6006   # then: tensorboard --logdir … --host 0.0.0.0 inside the pod
kubectl delete pod pf-train-smoke                   # plain delete ONLY — never --force/--grace-period=0
```

## Known issues / open items

- **EOS quota**: was exhausted 2026-09-14 (broke v1 condor writes); freed 2026-09-15 by
  removing `pkl_links_v1` (v1 to be re-produced when storage is found). Scripts keep
  `HOME=/shared/mlpf-phase2/home` regardless; `eos_sync.sh` skips gracefully if it refills.
- **v2 pkl counts** (2026-09-14, all written 12:45–13:27 same day, schema-uniform 37-field):
  ttbar 15,399/18,488 inputs, qcd 25,136, zll 18,206 — ttbar mop-up via condor resubmit
  when quota allows; a later tfds rebuild of the tail = version bump (1.0.1).
- `pf-postprocess-test` is stuck Terminating (its `vscode-mkdir-eos-target` container
  can't be killed — kubelet `DeadlineExceeded`). Do **not** force-delete; if it lingers,
  ask the NGT admins. Root cause is the `vscode-in-shared` label on batch pods — avoided here.
- Available GPU profiles beyond `mig-1g.12gb` on H100-NVL are unknown (no node RBAC);
  discover via the Pending-pod events trick or the NGT docs/admins before scaling up.
- Full 3-sample tfds (~est. 0.5 TB) likely exceeds the free space on the 500Gi PVC —
  measure the ttbar ratio first, then decide (cap qcd files / second PVC / prune stale dirs).
- v1 (colleague-target) pkls use their own 36-field schema and need their own adapter +
  builders before they can be trained on (`docs/phase2_feature_mapping.md`).
