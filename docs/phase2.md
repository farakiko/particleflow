# MLPF Phase 2 — working notes

Adapting CMS MLPF from Run 3 to the **Phase-2 / HL-LHC endcap**, using
**tracks + TICL tracksters** (HGCAL) as inputs instead of Run-3 PF clusters.
Claims are **[VERIFIED]** (checked in data/code) unless marked **[OPEN]**.

---

## 1. Data & layout

`data/cms/phase2/offline/Aug31/nano/`
- `ttbar_0pu/` — raw Phase-2 **NanoAOD** (ttbar, 0 PU): **100 valid files** (~30k events).
  TICL-focused NanoAOD, 484 branches; calo side is **endcap-only** (no barrel calo).
  - `pkl_run3style/` — our Run3-style target (§4A)
  - `pkl_mohamed/` — colleague's target logic ported to NanoAOD (§4B)
- `pkl_output/` — colleague's original target graphs (7,486 pkls; QCD/ttbar/Z→ll, from NanoAOD).

Scripts live in `scripts/phase2/` (see §6). Plots in `plots/phase2/` (gitignored).
Git workflow: work on `phase2` branch; Farouk commits/pushes, not the agent.

---

## 2. Inputs (VERIFIED)

- **Calo input = `ticlTrackstersCLUE3DHigh`** (~56/ev, |η| 1.36–3.12, endcap). CLUE3D
  tracksters are the Phase-2 analog of a Run-3 PF cluster: the 3D pattern-recognition
  output, *before* linking. We deliberately do **not** use the linked/supercluster
  collections (`ticlTracksterLinks`, `…SuperclusteringDNN`, `TICLCandidates`) — that
  linking is exactly what MLPF should learn. All Run-3 cluster features are present or
  derivable; extras: PCA shape (`EV1-3`, `eVector0`).
- **⚠ Trackster energy [VERIFIED]:** `ticlTrackstersCLUE3DHigh_regressed_energy` is
  **all-zeros** in this NanoAOD (min=median=max=0). Use **`raw_energy`** (median 2.7 GeV,
  = summed calo deposit) as the trackster element energy. Using regressed_energy made the
  neutral input `e` feature = 0 and the neutral regression target = `log(truth/0)` — i.e.
  un-learnable. Fixed 2026-09-07 in `postprocessing_run3style.py`; see §4A.
- **Track input = `GeneralTrack`** (~154/ev) + **`GSFTrack`** (electrons, ~3.5/ev).
  Full Run-3 track feature set. Highlight: `GeneralTrack_hgcal_*` = each track already
  **extrapolated to the HGCAL surface** (handle for track↔trackster association).

Event size ≈ 56 tracksters + 154 tracks ≈ 210 elements — same order as Run-3 MLPF.

---

## 3. Truth: `SimTICLCandidates` (VERIFIED)

One per CaloParticle (~184/ev), carrying 4-momentum, PID, `isPU`, and track links.

- **Energy fields:** `energy` = truth MC energy; `raw_energy` = calo deposit.
  Median `raw_energy/energy`: µ 0.13, charged had ~0.5, neutral had 0.39, γ 0.72, e ~1.0.
- **PID taxonomy:** photon (22), charged had (211), neutral had (130), e (11), µ (13).
- **Track link:** `trackIdx` is a **direct index into `GeneralTrack`**; valid for
  charged (≈73%), sentinel `-2^31` for neutrals.
- **⚠ Electron energy defect [VERIFIED]:** for electrons, `energy` and `pt` are
  **~0.38× truth**, but **`raw_energy` ≈ 0.97× truth**. → the target must use
  `raw_energy` for electrons (see §4A). Other classes: `energy` is correct.
- **[OPEN] `isPU` ∈ {0, −1}** (not {0,1}) even in 0 PU; −1 likely "PU-undetermined".

---

## 4. Two target designs

### 4A. Run3-style target (ours) — current best

`scripts/phase2/postprocessing_run3style.py`. Mirrors the Run-3 paper: **one target
per physical particle, truth energy, unique element assignment, NO fragmentation.**

- Truth = `SimTICLCandidates`; each assigned to **one** primary element:
  - charged → `GeneralTrack[trackIdx]` (η matches candidate exactly, VERIFIED);
  - neutral → the trackster with **max shared energy** from
    `SimCP2ticlTrackstersCLUE3DHighByHits[i]` (candidate idx i == SimCP idx i, VERIFIED).
- One target per element; multiple particles on one element → **merged**. Non-primary
  elements → `pid=0` (null).
- **Electron energy uses `raw_energy`** (fixes the §3 defect); all others `energy`.
- Elements: `GeneralTrack` (typ 1) + `ticlTrackstersCLUE3DHigh` (typ 4). Trackster
  element energy = **`raw_energy`** (regressed_energy is empty — see §2 fix, 2026-09-07).

**Learnability check (VERIFIED, 2026-09-07):** the regression target is
`log(target_E / element_E)`, so it must be O(1) per class. Measured per assigned element:
- track-anchored: chad **0.98**, muon **0.99**, electron **0.76** — leptons/charged
  sit on their track (e⁻ 98–100% typ-1, µ 100% typ-1), as expected.
- trackster-anchored (after the raw_energy fix): photon **1.33** (EM response), neutral
  had **3.78** (hadron calo response; tail %>5 ≈ 36% = real hadron-response spread).
- *Before* the fix, neutral ratios were ~5–13 **million** (element energy = 0) →
  un-learnable, which explains the poor neutral eval in the first 10-epoch run.

**Why no score cuts are needed:** the raw neutral→trackster association is *not* 1-1
(mean 2.7 tracksters/particle, 44% >1). We enforce 1-1 by **argmax shared-energy**
(one best trackster per particle) + **merge** on the element side (78% clean 1-1, 22%
merged), rather than connect-to-all-then-cut. ~53% of tracksters end up null.

**Results (ttbar 0 PU, endcap, VERIFIED):**
- **Jet response** (target/gen jet pT, |η|>1.5, gen pT>20, ΔR<0.2, 42.8k jets):
  **median 0.970, IQR/median 0.095** — scale at ~1, tight.
- **ΣpT per class** (target/gen): charged had 1.05, photon 0.90, neutral had 0.90,
  electron **0.92** (after the raw_energy fix; was 0.43), muon 0.99 — all within ~10%.
- **Efficiency** (gen→target, same-class): charged 0.96, e 0.97, µ 0.99, photon 0.72,
  **neutral had 0.40**. **Fake rate**: charged had ~0.23 (over-produced), others ~0.05–0.18.
- Coverage gap: 182k target vs 283k gen endcap jets — misses ~36% of gen jets, the
  **soft unreconstructed neutrals** (an energy-cheap but count-heavy tail; neutral-had
  ΣpT is still 0.90 because the missed ones are soft).
- Plots: `plots/phase2/run3style_{targetjet_response,targetjet_vs_gen_pt,efficiency_vs_pt,fakerate_vs_pt,sumpt_per_class}.pdf`.

### 4B. Colleague's (Mohamed) target — SimTICLCandidate-native

`scripts/phase2/postprocessing_mohamed.py` (port of
`.staging/cms/postprocessing_ticl_ttbar_nopu.py` to NanoAOD). Uses gen-matching filter
+ associator **score cuts** (r>0.6/s>0.9) + **fragments** neutrals into one target per
trackster. Faithful port (reproduces his ~18 particles/ev and balanced PID ratio).

- **Jet response 0.71** vs our 0.97 (same files, same gen). Lower because heavy
  filtering keeps only ~18 particles/ev (vs ~106) → sparser jets capture less energy.
- Comparison plots: `plots/phase2/compare_targetjet_{response,vs_gen_pt}.pdf`.

---

## 5. Design choices that matter (Run-3 vs colleague, distilled)

Full 28-item catalog is in git history; the substantive ones and their resolution:

| # | choice | Run-3 | colleague | resolution |
|---|---|---|---|---|
| Target energy | truth | `energy` most, `raw_energy` for e⁻ | **e⁻ MUST use `raw_energy`** (VERIFIED defect); others `energy`. His e⁻ special-case is correct. |
| Neutral fragmentation | one target/particle | one target per trackster | ~3% of particles affected at 0 PU; we chose **one/particle** (argmax). Re-check at high PU. |
| Association | CMSSW shared-E | associator score cuts | We avoid cuts via argmax+merge. Score cuts define the "is-particle" label; fine, revisit thresholds at high PU. |
| Gen-matching filter | none | keep only gen-matched simcands | Strong filter (→ sparse target, low response). We don't filter → denser, better response. |
| pt definition | true pt | `E·sinθ` massless | SimTICLCandidates are massless (mass=0) so equivalent; we use simcan pt directly. |
| Overlap/merge | merge CPs | ≤1 cp/element | ~2% at 0 PU; we merge. Re-check at high PU. |

**Score-cut mechanism (VERIFIED):** associator score is 0=perfect…1=no match. The cut
acts on the **target only** (every trackster stays an input); it sets each element's
"is this a particle?" label. Not needed in our argmax approach.

---

## 6. Scripts (`scripts/phase2/`)

| script | purpose |
|---|---|
| `postprocessing_run3style.py` | our Run3-style target builder (NanoAOD → pkl) |
| `postprocessing_mohamed.py` | colleague's target logic ported to NanoAOD |
| `plot_run3style_jets.py` | target vs gen jet pT + response |
| `eff_fake_run3style.py` | efficiency/fake rate vs pT, per class (same-class vs any) |
| `sumpt_run3style.py` | per-event ΣpT per class, target vs gen |
| `train_local_mps.py`, `eval_local_mps.py` | local MPS trainer + 3-curve eval (target/TICL/MLPF vs gen): response, pT spectra, eff/fake (same- & any-pid), ΣpT, confusion matrix |
| `investigate_anchoring.py` | §8A lepton track anchoring + §8B neutral energy closure |
| `investigate_gsf_scores.py` | §8A GSF-rescue check + §8C associator-score discrimination |
| `investigate_e_acceptance.py` | §8A trackless-electron pT/η breakdown |
| `investigate_ts_features.py` | §8D trackster shape-feature separation (γ/nhad, closure) |
| `sample_stats.py`, `jet_response*.py`, `jet_diagnostics.py` | analysis of `pkl_output/` |

Env: needs `uproot awkward numpy matplotlib fastjet vector`.

---

## 7. Open questions / TODO

- **High pileup (PU~200):** everything above is **0 PU**. Overlap, fragmentation, and
  pileup rejection (the score cuts) will matter much more — re-measure when PU samples exist.
- **Neutral-hadron coverage:** particle efficiency only 0.40 (same-class) — many soft
  neutral hadrons get no trackster. Energy impact small (ΣpT 0.90) but a real gap.
- **Charged-hadron over-production (VERIFIED):** target charged-hadron ΣpT is 1.045×
  gen (only class >1; γ/nhad/e/µ are 0.90–0.99, i.e. subsets as expected). Cause: 23%
  of target charged hadrons have **no gen status-1 partner** but are **soft** (median
  0.35 GeV, 98% <2 GeV) → only +6.5% of ΣpT. These are reconstructable **secondary-like**
  charged particles (tracks but not gen primaries — the documented "target ⊋ truth"
  effect): NOT in high-pT jets (2% within ΔR<0.4 of a pT>10 gen jet); ~half sit ΔR
  0.2–0.4 from a primary charged hadron (never on top); forward-edge (2.7<|η|<3.0)
  over-density. Exact origin (delta rays / nuclear interactions / forward tracking) open.
- **`isPU ∈ {0,−1}`** semantics; barrel/η-transition (1.5–1.7) handling; richer `Xelem`
  feature set (PCA shape, timing, track errors) before training — **now investigated, see §9D**.

---

## 8. First local training on the fixed data (VERIFIED, 2026-09-08)

Reprocessed all **307 pkls** (100 ttbar + 100 qcd + 107 zll) with the raw_energy fix (§2);
learnability re-confirmed (trackster input E zero-fraction 0%; per-class targets O(1)).

**Train** (`train_local_mps.py`, MPS): 0.69M-param attention model, input_dim=12, 6 classes,
3 convs; 77,814 events (train 70k / val 7.8k); 10 epochs; **best val 4.016 @ ep9** (still
descending, no overfit). Loss `mlpf_loss` = FocalLoss(γ=2) PID + 10·CE binary presence +
MSE regression (pt/energy scaled by √pt).

**Eval** (`eval_local_mps.py`, 7,781 val events, 3 curves vs gen — target / TICL baseline / MLPF):
- **Jet response median: MLPF 0.962 vs target ceiling 0.976 vs TICL 0.677.** MLPF learns the
  jet energy scale — near the target, far above the CMSSW TICL baseline.
- Per class (eff same-class / ΣpT-to-gen): chad **0.96 / 1.08** (=target), photon 0.69 / **1.36**
  (energy over-scaled), nhad **0.08 / 0.30**, electron 0.16 / 0.44, muon 0.37 / 0.65.

**Diagnosis — the bottleneck is the classification head, not regression or data:**
- **Any-pid** efficiency (spatial match, class ignored): nhad **0.61**, e **0.99**, µ **1.00**,
  chad 0.96 — i.e. the particles ARE found spatially, then mislabeled.
- **Confusion** (per-element target→pred): **nhad→γ 47%** (only 13% correct), **e→chad 71%**
  (8% correct), **µ→chad 67%** (27% correct); chad 87%, γ 81% held. Rare/hard classes collapse
  into their dominant neighbours.
- **Imbalance mitigation in this run = focal γ=2 only.** `losses.py` builds `FocalLoss(gamma=2.0)`
  with **alpha=None** (no per-class weights); binary loss is unweighted CE; `train_local_mps`
  does no oversampling/resampling. At this imbalance (e 2.2k, µ 0.8k in val) + 10 epochs, focal-
  without-alpha isn't enough. Levers: **focal `alpha` = inverse-freq class weights**, more epochs.

---

## 9. Pre-scaling physics-soundness checks (VERIFIED, ttbar 0 PU, ~7.5k events)

### 9A. Lepton/charged track anchoring & GSF
- Assignment (postproc logic): **muon 100%** GeneralTrack; **electron 73.8% track, 3.2%
  trackster-fallback, 23% dropped** (no track & no trackster); charged had 84% / 13% dropped.
- **GSF tracks exist and are correctly linked** (`GSFTrack`, `SimTICLCandidates_gsftrackIdx`;
  muons 0% GSF confirms the read) but **do NOT rescue trackless electrons**: of the ~26% with no
  GeneralTrack, only **0.7%** have a GSF. GSF is ~a subset of already-tracked e (20% both, 0.2%
  GSF-only). GSF's real value = brem-corrected **momentum quality** for tracked e, not acceptance.
- **Trackless electrons are negligible**: median pT **60 MeV**, 96.7% <1 GeV, **2.7% of electron
  energy**, forward (trackless rate 21%→51% over |η| 1.5→3.2); hard-central (pT>10,|η|<2.5) = 0.3%
  of trackless. → tolerable soft-EM acceptance loss (conversions/brem/deltas), **not a blocker**.

### 9B. Neutral energy closure (anchored trackster rawE / truth E)
- **photon median 0.86** (= calo response 0.90 × trackster capture 0.90; single trackster).
- **neutral had median 0.32** (= calo response **0.68** × capture **0.48**; median **2** tracksters
  share the shower); 39% below 0.25. → the input trackster under-represents nhad energy, forcing a
  large, high-variance regression correction. NOTE: target ΣpT is still ~0.95 (target carries TRUTH
  energy) — the closure gap is an **input/regression** problem, not a target-energy one.

### 9C. Associator scores & the two neutral options
- s_score (`SimCP2…CLUE3DHighLinks_score`) & r_score (`Recoticl…2SimCPLinks_score`): 0=perfect…
  1=no-match, **MC-truth-derived → NOT usable as model inputs** (unavailable in data = leakage);
  they are target-building/eval tools only.
- **Option 1 (one nhad per energetic trackster): rejected.** Fragment tracksters beyond the argmax
  anchor carry median **1%** of attributable energy, s_score median **1.0** (junk); only 2% of nhad
  have a real promotable second trackster. The missing energy is not in promotable tracksters.
- **Option 2 (score as input): rejected as framed** (truth-only). Also at 0 PU r_score does not
  discriminate roles (anchor/fragment/unused all median r≈0, ~95% pure) — its power is **latent PU
  rejection**; re-test at PU 200.
- **Score as a truth-level LOSS WEIGHT — deferred option for PU samples.** The legitimate use of the
  scores is not as an input but to **weight the target**: down-weight the regression/classification
  loss on genuinely un-capturable neutrals (high s_score) instead of *deleting* them. Colleague's hard
  `r>0.6`/`s>0.9` cut removes that energy → sparser target, jet response **0.71** vs our **0.97**; a soft
  weight keeps the energy (honest jet scale) while not penalizing the model for energy it fundamentally
  cannot see — the middle ground between our dense-but-noisy and his clean-but-sparse targets. Scores are
  MC-truth so this only works in simulation, and it is exactly where r_score also becomes the pileup-
  rejection signal. **Decision: keep the target AS-IS (no cut, no weight) at 0 PU; adopt the soft-weight
  for PU~200 samples when they exist.**

### 9D. Data-legal trackster shape features (all present in NanoAOD)
- Present: `EV1/EV2/EV3`, `eVector0_{x,y,z}`, `barycenter_{x,y,z}`, `time`/`timeError`,
  `raw_em_energy`, `n…vertices`. Current `Xelem` uses only pt/eta/phi/E/em/nhits → **missing depth,
  PCA shape, timing** (no build needed; just read).
- Separation (AUC; ttbar):
  - **Classification γ vs nhad**: **|barycenter_z| depth AUC 0.28** (hadrons deeper) — strongest
    single handle; nhits 0.61, time 0.35, em_frac 0.62; EV ratios ~none. Current input has **zero**
    longitudinal/shape info → real untapped signal against the 47% nhad→γ confusion.
  - **Regression (nhad under- vs well-captured)**: dominated by rawE (0.18) & nhits (0.22) which are
    **already inputs**; new features add only modest signal (time 0.65, EV2/EV1 0.61, depth 0.61).
- **Decision:** adding `barycenter_z` (depth), `time`, `EV1–3`, em-fraction to `Xelem` is worth it,
  **mainly as a γ/nhad classification fix**; expect only modest neutral-energy gain (physics-limited).
  [OPEN] whether to do the schema change + re-postprocess + retrain now.

### 9E. Feature parity vs colleague's target (`.staging/cms/postprocessing_ticl_ttbar_nopu.py`)
His `elem_branches` include depth + timing + track-quality/muon-ID + track-density features we don't
input. He does **NOT** use PCA `EV*`/`eVector0` (agrees with 9D: EV ratios AUC≈0.5, useless). Each of
our confusions has a matching data-available feature. Our input is 12 features
(typ/pt/eta/sin,cos φ/e/charge/px,py,pz/em_energy/nhits). His extras + availability in **our** NanoAOD:

| feature | elem | attacks | our NanoAOD |
|---|---|---|---|
| `barycenter_z` (depth) | trackster | nhad→γ (47%) | ✓ direct — the AUC-0.72 winner |
| `time`, `timeError` | trackster | nhad→γ | ✓ `ticlTrackstersCLUE3DHigh_time/_timeError` |
| `muon_type`, `muon_dt_hits`, `muon_csc_hits` | track | **µ→chad (67%)** | ✓ `GeneralTrack_muon_*` (+`isMuon`,`isTrackerMuon`) |
| `ptErr,etaErr,phiErr,lambdaErr,qoverpErr` | track | e→chad, fakes | ✓ `GeneralTrack_*Err` — direct |
| `vx,vy,vz` | track | secondaries/fakes | ✓ `GeneralTrack_vx/vy/vz` — direct |
| `min_dR_track,near_track_pt,sum_pt_dR10,n_trk_dR01–05` | trackster | charged/neutral split | ⚙ derive from `GeneralTrack_hgcal_eta/phi` + trackster η/φ (cheap) |
| `shower_depth` (E-weighted vertex \|z\|) | trackster | nhad→γ | ✗ per-vertex z/E not in nano → use `barycenter_z` |
| `track_gsf_type` | track | e→chad | ✗ no direct branch → derive via GeneralTrack↔GSFTrack (harder) |
| `n_clusters` | trackster | — | already = our `nhits` |

Direct-branch adds are ~free (read more columns in postprocessing). Physically motivated: depth+time →
nhad/γ, muon hits → µ/chad, track errors(+gsf) → e/chad.

---

## 10. Extended `Xelem` feature set — IMPLEMENTED (2026-09-08)

Acted on §9D/§9E. `Xelem` grew from **12 → 29** input features; input_dim auto-updates via
`X_FEATURES['cms_phase2']` in `mlpf/conf.py`. Track-only features are zero on trackster elements and
vice-versa (standard MLPF split encoding). Changed in three files:
`postprocessing_run3style.py` (read branches + fill), `mlpf/conf.py` (`X_FEATURES`),
`mlpf/heptfds/cms_pf_phase2/cms_phase2_utils.py` (adapter `col` map). All 307 pkls re-postprocessed.

Added features (name → NanoAOD branch):

| feature | element | source | targets |
|---|---|---|---|
| `bary_z` = **\|barycenter_z\|** | trackster | `ticlTrackstersCLUE3DHigh_barycenter_z` | **nhad→γ** (depth, AUC 0.72) |
| `time`, `timeerror` | trackster | `..._time`, `..._timeError` | nhad→γ |
| `ev1, ev2, ev3` | trackster | `..._EV1/2/3` | shape (weak, cheap) |
| `muon_type, muon_dt_hits, muon_csc_hits` | track | `GeneralTrack_muon_*` | **µ→chad** |
| `pterror, etaerror, phierror, lambdaerror, qoverperror` | track | `GeneralTrack_*Err` | **e→chad**, fakes |
| `vx, vy, vz` | track | `GeneralTrack_vx/vy/vz` | secondaries/fakes |

**Data hygiene (VERIFIED on 1 file):** `bary_z` stored as `|z|` (depth; endcap already in `eta` sign);
trackster `time == -99` is the "no-timing" sentinel → zeroed (with its error) so it isn't a fake input;
`muon_dt_hits == -1` on non-muon tracks is **kept** (it *is* the "not a muon" separator). No NaN/inf; X
shape (N, 29); track/trackster features zero-split correctly. **Dropped as not-worth-it:** `shower_depth`
(per-vertex z/E not in nano — `bary_z` covers it), `track_gsf_type` (no direct branch), track-density
features (need derivation; revisit if e/chad still confuses).

**Model sizing (VERIFIED, instantiated MLPF):** current 0.69M is small because embed dim = heads×head_dim
= 8×16 = 128, 3 convs. Decision: **keep 3 convs, widen to embed 256** (`--num-heads 16 --head-dim 16`) →
**2.72M** (cf. Run-3 paper ~5M; 6 convs @256 = 3.9M if we want deeper later). Param scan:
128→0.69M, 256→2.72M (3 convs)/3.91M (6 convs), 512→10.8M.

**Next:** retrain on the 29-feature pkls with embed 256 (+ focal-α class weighting still to add) and
re-check the §8 confusion matrix / classification-ceiling plot to confirm the gaps close.
