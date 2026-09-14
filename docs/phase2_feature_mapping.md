# Phase-2 input-feature mapping: colleague's target (v1) vs ours (v2)

Both postprocessors read the **same** CMS Phase-2 NanoAOD (moanwar `nano`) and emit a per-element
`Xelem` record array (→ tfds → model input). Their feature *sets* overlap almost entirely; most of
the apparent differences are **naming only**. This doc maps them field-by-field so it is unambiguous
what is common, what differs only by name, and what is genuinely different.

- **his** = `.staging/cms/postprocessing_ticl_ttbar_nopu.py` (also copied verbatim to
  `scripts/phase2/postprocessing_ticl_ttbar_nopu.py`) — used for **v1**.
- **v2 (ours)** = `scripts/phase2/postprocessing_run3style.py` — the `X_FEATURES["cms_phase2"]` in
  `mlpf/conf.py`.

All rows below are VERIFIED against the two source files and one real pkl of each.

---

## A. Common — identical name **and** meaning

| field | meaning | note |
|---|---|---|
| `typ` | element-type code | **same name, different code values** — see §D |
| `pt`, `eta`, `phi` | element kinematics | pkl stores raw `phi`; the tfds adapter splits it into `sin_phi`,`cos_phi` in both pipelines |
| `energy` | element energy (track `p`, trackster raw energy) | |
| `charge` | element charge | |
| `px`, `py`, `pz` | Cartesian momentum | |
| `em_energy` | trackster EM (raw) energy; 0 on tracks | |
| `bary_z` | trackster barycenter z | his: signed `ts_z`; ours: `|z|` (endcap sign already in `eta`) |
| `min_dR_track` | ΔR to nearest track at HGCAL surface | track-density block (identical formula) |
| `near_track_pt` | pT of that nearest track | |
| `sum_pt_dR10` | Σ track pT within ΔR<0.1 | |
| `n_trk_dR01…05` | # tracks within ΔR<0.01…0.05 | 5 fields, identical |

## B. Same meaning — **different name only** (his → ours)

| his field | v2 (our) field | meaning |
|---|---|---|
| `ts_time` | `time` | trackster time |
| `ts_time_err` | `timeerror` | trackster time error |
| `track_muon_type` | `muon_type` | muon-ID: reco::Muon type bitmask |
| `track_muon_dt_hits` | `muon_dt_hits` | muon DT hits |
| `track_muon_csc_hits` | `muon_csc_hits` | muon CSC hits |
| `track_gsf_type` | `gsf_type` | 1 on GSF-track elements, else 0 (added to v2 — see §E) |
| `track_pt_err` | `pterror` | track pT error |
| `track_eta_err` | `etaerror` | track η error |
| `track_phi_err` | `phierror` | track φ error |
| `track_lambda_err` | `lambdaerror` | track λ error |
| `track_qoverp_err` | `qoverperror` | track q/p error |
| `track_vx`,`track_vy`,`track_vz` | `vx`,`vy`,`vz` | track vertex |

> Naming convention: v2 drops the `track_` prefix and uses the Run-3 CMS names (`muon_type`,
> `pterror`, `vx`…), so the Phase-2 schema lines up with `X_FEATURES["cms"]`.

## C. Genuinely different (not just a name)

| field | in his? | in v2? | what's going on |
|---|---|---|---|
| `nhits` | yes (track hits; **0 on tracksters**) | yes (track hits on tracks, **LC-count on tracksters**) | v2 overloads one field; his keeps two (see next row) |
| `n_clusters` | **yes** (LC count on tracksters) | no (folded into `nhits`) | same physics as our trackster `nhits` |
| `shower_depth` | **yes** (energy-weighted Σ z·E / Σ E over layer clusters) | no (≈ our `bary_z`) | the barycenter *is* the energy-weighted centroid, so his `shower_depth` ≈ his own `bary_z` ≈ our `bary_z` (near-duplicate) |
| `ev1`,`ev2`,`ev3` | no | **yes** (trackster PCA eigenvalues) | shower-shape elongation/sphericity; γ-vs-neutral-hadron discriminator we added |

Net: after v2 adds `gsf_type` (§E), the only real feature-content differences are
**v2 has PCA eigenvalues `ev1/ev2/ev3`; his has `shower_depth`+`n_clusters`** (both ≈ things v2
already encodes via `bary_z` / trackster `nhits`).

## D. `typ` code values differ (same field, different encoding)

| element | his `typ` | v2 `typ` |
|---|---|---|
| general track | 1 | 1 |
| GSF track | 2 | 2 |
| trackster (`ticlTracksterLinks`) | 3 | **4** |
| EG supercluster | 4 | — (v2 has no supercluster elements) |

The tfds adapters remap `typ`→`typ_idx` (a small contiguous embedding index), so the raw code
value is internal to each pipeline; only the *set of element types* matters for the comparison.

## E. Element-set differences (beyond per-field features)

| element type | his | v2 |
|---|---|---|
| general tracks | ✅ | ✅ |
| **GSF tracks** | ✅ | ✅ (added — see below) |
| `ticlTracksterLinks` tracksters | ✅ | ✅ |
| EG superclusters (`…SuperclusteringDNN`) | ✅ (optional, off by default) | ❌ |

GSF tracks were **added to v2** so the electron-momentum path matches his (electrons prefer their
GSF track; `raw_energy` truth). This removes GSF as a confound in the v1-vs-v2 training comparison —
after it, the pipelines differ only in {target definition, PCA vs shower_depth/n_clusters,
superclusters}.

---

## Feature counts

| | raw `Xelem` fields | model features (after adapter) |
|---|---|---|
| his | 36 | 37 (`phi`→`sin_phi`+`cos_phi`) |
| v2 (before GSF) | 36 | 37 |
| **v2 (after adding `gsf_type`)** | **37** | **38** |
