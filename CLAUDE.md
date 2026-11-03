# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## What this repo is

MLPF: machine-learned particle-flow reconstruction (upstream `jpata/particleflow`). A transformer
takes reconstructed tracks and calorimeter clusters ("elements") and predicts, in one shot, a
particle per element (null / PID / momentum). Supported detectors: CMS Run 3 (deployed in CMSSW
via ONNX), CLIC and CLD (Key4hep). The `phase2` branch extends CMS MLPF to **Phase-2 / HL-LHC**:
tracks + **TICL tracksters** from the HGCAL endcap, first at 0 PU and now at PU 200.

Component map: `docs/map.md`. Phase-2 living notes: `docs/phase2.md` (running log + decisions),
`docs/phase2_pu200_notes.md` (PU 200 target and architecture).

## Layout (the parts you will touch most)

- `particleflow_spec.yaml`: single source of truth for sites, `productions:` and `models:`.
  Resolution order is `mlpf/conf.py` defaults, then the spec, then CLI overrides (`--model.num_convs 6`).
- `mlpf/conf.py`: Pydantic schemas, `X_FEATURES` / `Y_FEATURES`, `ELEM_TYPES_NONZERO`, `CLASS_LABELS`
  per dataset (`cms_phase2`, `cms_phase2_v3`, ...).
- `mlpf/pipeline.py`: training / test CLI. `mlpf/model/`: `mlpf.py` (model), `heptv2.py` (sub-quadratic
  attention for PU 200), `losses.py`, `PFDataset.py` (loader; `X[..., 0] == 0` is the **padding mask**),
  `training.py`, `utils.py` (`unpack_target` / `unpack_predictions`).
- `mlpf/heptfds/cms_pf_phase2/`: Phase-2 tfds builders + `cms_phase2_utils.py` (pkl to tensor adapter).
- `scripts/phase2/`: Phase-2 postprocessing (NanoAOD to pkl), studies, validation plots.
  `scripts/phase2/cluster/` (NGT Kubernetes training jobs), `scripts/phase2/condor/` (lxplus postprocessing, see its README).
- `kube/README-train.md`: cluster training runbook.
- `data/cms/phase2/offline/nano/<sample>/` raw Phase-2 NanoAOD, `.../offline/pkl/<sample>/` postprocessed pkls
  (both gitignored). Samples: `ttbar_0pu`, `qcd_0pu`, `zll_0pu`, `ttbar_200pu`.

## Environment

- Local (macOS): `pixi run python3 <script>` from the repo root. torch is **not** in the root pixi env;
  uproot / awkward / numpy / matplotlib / mplhep / scipy / fastjet are.
- Upstream / cluster convention: `uv run python3 ...`; the NGT cluster uses the venv `/shared/envs/mlpf`.
- Read ROOT files with **uproot + awkward**, not PyROOT.
- Lint: pre-commit runs black (`--line-length=150`) and flake8 (max 180). Tests: `pytest tests/`.

## Phase-2 conventions and gotchas

- **Inputs are always tracks + TICL tracksters** (`ticlTrackstersCLUE3DHigh`), never hits. Truth info
  (associator scores, `isPU`, sim links) is for labelling and evaluation only, never a model input.
- **Scope is the HGCAL endcap** (1.5 < |eta| < 3.0). Truth (`SimTICLCandidates`) is incomplete in the barrel,
  so restrict truth-based studies to the endcap unless the question is explicitly about the barrel.
- **"v2" is ambiguous; always say which one.** `pkl_links_v2` = the old Run3-style target
  (`postprocessing_run3style.py`). `nano_v2` = `postprocessing_ticl_nopu_nano_v2.py`, the current target (also used for PU 200).
  v3 = `postprocessing_ticl_acceptance.py`, which the 0 PU scaling-scan models were trained on (`cms_pf_ticl_nopu:2.0.0`).
- **Element `typ` codes differ between postprocessing families.** run3style: track 1, GSF 2, trackster 4.
  nano_v2 / v3: track 1, GSF 2, trackster 3, EM 4. Check `TYP_MAP` in `cms_phase2_utils.py` matches the
  pkls you feed it: an unmapped type becomes `typ_idx 0`, i.e. silently masked as padding.
- NanoAOD quirks (verified): `ticlTrackstersCLUE3DHigh_regressed_energy` is all zeros (use `raw_energy`);
  for electrons `SimTICLCandidates_energy` is about 0.38x truth (use `raw_energy`). `SimTICLCandidates_isPU`
  is {-1, 0, 1}, but -1 is a CMSSW lookup failure, not a physics state: about 15% of candidates even at 0 PU,
  where all are signal. The resolution is also wrong in about 28% of the 0/1 values (wrong parent CaloParticle).
  Use the 1:1 index-aligned `ticlSimTrackstersfromCPs_isPU` instead (never -1). `ticlSimTracksters_caloParticleIdx`
  shares the bug for SimCluster-seeded sim tracksters (`TICLCandidateExtraTableProducer.cc`,
  `SimTracksterTableProducer.cc`).
  Trackster `EV1-3` are the cleaned, energy-weighted PCA eigenvalues of the layer-cluster positions (cm^2) and are
  0 for tracksters with at most 2 layer clusters (PCA not run). `eVector0_y` and `eVector0_z` are **swapped** in all
  trackster tables (`DPGAnalysis/HGCalNanoAOD/python/hgcalTracksters_cfi.py`); swap them back before use.
- tfds dataset names are the builder class name in snake_case: `CmsPfPhase2Ttbar200pu` -> `cms_pf_phase2_ttbar200pu`.
- Documentation discipline: in `docs/phase2*.md`, mark claims **[VERIFIED]** (checked in data/code) or **[OPEN]**.
  Keep a measured number separate from its interpretation, and never present a hypothesis as a finding.

## Plots

- **PDF only**, under the gitignored `plots/` tree in a per-study subdirectory (e.g. `plots/phase2/<study>/`).
- Paper-style figures use mplhep CMS style: `mplhep.style.use("CMS")` and
  `mplhep.cms.label("Work in progress", data=False, ...)`.

## Workflow rules

- Work on the `phase2` branch. **Do not commit or push unless explicitly asked**; leave changes in the
  working tree. Never push to the `joosep` (upstream) remote.
- Laptop and cluster checkouts sync **only through git** (push here, `git pull` there), never by copying files.
- Notebooks (`notebooks/*.ipynb`): hand back code snippets to paste. Don't write to the `.ipynb`
  (the VS Code buffer overwrites external edits).
- NGT Kubernetes: never force-delete pods/jobs. Don't put the `vscode-in-shared` label on batch pods
  (its init container wedges them in Terminating). Phase-2 workspace is `/shared/mlpf-phase2/`; `/shared/mlpf`
  belongs to an unrelated project. Set `HOME=/shared/mlpf-phase2/home` in cluster scripts.
- On MIG GPU slices set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (masked OOMs otherwise).
