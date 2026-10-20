# Phase-2 PU mitigation — literature synthesis & experiment roadmap

Scientist-level read of the ML/non-ML pileup-mitigation literature in
`references/ML4PU_papers`, distilled for our CMS MLPF Phase-2 PU-head program
(HGCAL endcap, PU 200, inputs = tracks + TICL CLUE3D tracksters).

Papers read: **PUPPI** (non-ML baseline), **PUPPIML / GNN-PUPPI**, **PUMA**,
**PUMiNet / Attention-PU**, **SSL-GNN PU**.

Discipline: claims are tagged **[VERIFIED]** (checked in our data/code or stated
directly in the paper) or **[OPEN]** (hypothesis / needs a check). AUC numbers
from the papers are **not** apples-to-apples with ours — most are Delphes/gen-level
with perfect vertexing and no timing/shower-shape; take their *ideas*, not their numbers.

---

## 0. One-paragraph verdict (answer to "are we missing relevant input?")

**Yes, and it is specific and unanimous.** Every method's neutral-PU power comes
from a **charged, vertex-anchored reference**: a neutral is signal if it sits in a
neighborhood of charged particles from the hard (leading) vertex. We carry raw
per-track `vz` but **not an identified leading vertex or a track→PV association**,
so we cannot yet form that anchor. [VERIFIED in our feature list] PUPPIML quantifies
the stakes: **AUC 0.96 with the charged anchor → 0.70 without it.** [paper] The second,
complementary gap: **HGCAL timing and CLUE3D shower-shape** are exactly what the ML
papers never had — and in our data the shower-shape PCA eigenvalues (`ev1/ev2/ev3`)
are **all zero** and trackster `time` is a **−99 sentinel for a large fraction** of
elements. [VERIFIED] Those are our would-be differentiators, currently unused.

---

## 1. UPSTREAM — information to request be built

Ranked by value; "backed by" = which papers argue for it.

| # | Item | Backed by | Priority |
|---|---|---|---|
| U1 | **Reconstructed primary-vertex collection + per-track LV/PV association** (LV z-position, track→vertex flag, N_vertices). The gating dependency for all neutral-PU power. | all 5 | **P0** |
| U2 | **Continuous hard-energy-fraction truth per element** from sim-link decomposition, `y = E_LV/(E_LV+E_PU)`. Enables a soft regression target; resolves the `isPU=−1` "unresolved" category. *(See §5: our tfds `ispu` is already continuous in [0,1] — needs checking whether it already is this.)* | PUMA, PUMiNet | **P0** |
| U3 | **Populate the shower-shape PCA eigenvalues** (`ev1/ev2/ev3`, currently all zero) — the primary *neutral* discriminant none of the ML papers even had. | PUPPI, PUMA, PUMiNet, PUPPIML | **P0** |
| U4 | **Complete trackster timing + per-event LV time t0** (full coverage + time resolution; kill the −99 sentinels). Enables Δt-to-LV for neutrals. | all flag timing | **P1** |
| U5 | Official pileup-density **ρ** / **N_vertices** if already produced upstream. | PUPPI, PUMA, PUPPIML | P2 |

## 2. BUILD OURSELVES — from NanoAOD / what we already carry

| # | Feature | How | Backed by | Priority |
|---|---|---|---|---|
| B1 | **PUPPI α variable per element**, β=1,2,3, two versions: α^F (all neighbors) and α^C (charged-LV only). `α^β_i = log Σ_{Rmin<ΔR<0.4} (p_T,j / ΔR_ij^β)`. Weights **neighbors'** pT, not own → intrinsically pT-decorrelated. The single highest-value engineered feature. | pt/η/φ we have; α^C needs the LV anchor (use B2 proxy) | PUPPI, PUPPIML | **P0** |
| B2 | **dz-to-LV proxy**: estimate z_LV as the pT²-weighted mode of track `vz`; feed `|vz − z_LV|` per track. Stand-in for U1 until it lands. | from vx/vy/vz | PUPPI, PUMA, PUPPIML | **P0** |
| B3 | **Split existing local-density features by LV-track vs all-track** (nearest-LV-track pT, Σ LV-track pT in cone, N LV-tracks). Current ones use *all* tracks; signal is proximity to *LV* tracks. | reuse, gate by B2 | PUPPIML | P1 |
| B4 | **Radial-shape descriptors per element**: ring fractions f_ring1–4 (annuli to ΔR<0.4), ⟨ΔR²⟩ (pT²-weighted), p_T^D, local pooled ΣpT/energy/multiplicity, cluster ΔR spread. | kinematics only | PUPPI (Table 4), PUMA, PUMiNet | P1 |
| B5 | **Neutral-side handles from existing columns NOW**: trackster time significance t/σ_t and Δt-to-bunch(~0), barycenter-z significance vs z_LV. | time/bary_z we have (partial) | PUMiNet (our edge) | P1 |
| B6 | **Event globals** ρ, ρ_C, ρ_N and an N_vtx proxy (count `vz` peaks), concatenated to every element. | from vz + kinematics | PUPPIML, PUMA | P2 |
| B7 | **signed-χ² relative discriminant** `(α−ᾱ_PU)|α−ᾱ_PU|/RMS²` using in-sample PU α median/RMS. | derived from α (B1) | PUPPI | P2 |

## 3. STRATEGY — model / training / loss

**P0 — the two big structural changes (strongest convergence):**

- **S1. Reframe the PU head as a soft hard-energy-fraction regression** (MSE),
  instead of / alongside binary. Two independent papers (PUMA, PUMiNet) show this is
  *the* key: a PU200 element is physically an LV+PU **mixture**, not a clean label,
  so a soft target structurally removes the "low-pT ⇒ PU" shortcut and composes with
  MLPF's momentum head (scale the 4-vector by ŷ). Likely replaces our pt-reweighting hack
  with something physically grounded.
- **S2. Charged-anchor + SSL masking for neutral labels** (SSL-GNN; echoed by PUPPIML):
  supervise the head on *charged* elements (vertex-derived label), randomly mask each
  element's own label so it "looks neutral," keep neighbor charged labels as inputs,
  apply to real neutrals at inference. Zero neutral MC truth; trainable on real data.
  *(User deferred SSL for now — keep on the shelf.)*

**P1:**

- **S3. Fix the shortcut with features, not just decorrelation.** Every paper avoids the
  pT shortcut by *feeding α/density/neighbor context* so the model does not *need* pT
  (PUMA even up-weights high-pT). A/B-test our current pt-reweighting vs "rich features
  + continuous target."
- **S4. Keep charged (even obvious-PU) elements in the attention** — they are the
  informative neighbors for neutrals. Don't prune the PU-head input.
- **S5. Event-wide context on top of local HEPTv2**: a few global "register" tokens
  (full-attention, O(few)) + trackster→track cross-attention, so neutral tracksters inherit
  nearby tracks' vertex context (PUMiNet: event-wide attention beats per-object with
  identical features).
- **S6. Condition on event PU scale** (N_vtx/ρ) and **η-bin the normalization** for the
  endcap — a relative, PU-adaptive threshold instead of a fixed pT floor (PUPPI's
  `A + B·N_vtx`, transfer factors across |η|=2.5/3.0).

**P2:**

- **S7. Output a calibrated soft weight and down-weight momentum** rather than hard-mask →
  smoother jet/MET resolution (PUPPI, PUPPIML, PUMA).
- **S8. Evaluate like the field does**: PU-stability of jet/MET/mass resolution vs N_vtx,
  against a PUPPI-like baseline *and* a PU0 gold standard ("fraction of the gap to PU0
  closed"), plus cross-process generalization (train ttbar → test qcd/zll, which we have).
- **S9. Train across pileup levels** for robustness; note SSL extrapolates *up* in PU, not *down*.

**Our differentiator to protect:** timing + shower-shape. If U3/U4 populate those columns,
we can beat PUMA/PUMiNet/PUPPIML/SSL on *neutral* PU, which they could not touch.

---

## 4. Per-paper one-liners

- **PUPPI**: α local-shape variable (charged-LV neighbors), median/RMS PU subtraction,
  N_vtx-dependent cut, η-dependent normalization, soft weight rescales momentum. The
  non-ML target to beat; everything downstream re-derives its α.
- **PUPPIML / GNN-PUPPI**: GNN on (η,φ)-local graph; charged nodes in-graph anchor neutrals;
  α^F/α^C as inputs; AUC 0.96→0.70 without the charged anchor; locality beats global ordering.
- **PUMA**: per-element hard-energy-fraction **regression** (soft target); cluster/locality
  features; up-weights high-pT; evaluates on jet/MET resolution vs N_PV.
- **PUMiNet / Attention-PU**: event-wide attention + pooling-and-concat global context;
  impact-parameter inputs; soft target; attention beats per-object DNN with identical features.
- **SSL-GNN**: self-supervised random-masking to label neutrals from charged-vertex truth;
  trainable on data, no neutral MC truth; extrapolates to higher PU.

---

## 5. Experiment roadmap — 5 controlled tests (one change at a time)

Baseline for all comparisons = **current pt-weighted binary PU head**
(`ttbar200-heptv2-pu-rew`). Each test branches off the *same* base config and changes
**only one thing**, measured on the *same* eval harness (eff/fake with pT>1 GeV, jet-pT
response, PU-head ROC/AUC by type; add S8 downstream PU-stability metrics when ready).

| Test | Change | Needs reprocessing? | Maps to |
|------|--------|---------------------|---------|
| T1 | Add engineered input features (α, dz-to-LV proxy, split densities, shape descriptors) to the parquets, retrain the *same* pt-weighted binary head | **YES** — new postprocessing pass writes new columns | B1–B7 |
| T2 | Reframe PU head as soft hard-energy-fraction regression (MSE) | **[OPEN] likely NO** — see note below | S1 / U2 |
| T3 | Add event-wide context (global/register tokens + trackster→track cross-attention) | NO — model-only | S5 |
| T4 | Condition on event PU scale (N_vtx/ρ) + η-bin normalization | NO* — derivable from existing X at load time | S6 / B6 |
| T5 | Output calibrated soft weight + down-weight momentum (vs hard mask) | NO — loss/inference-only | S7 |

**Parquet reuse — [VERIFIED] for T3, T5; [OPEN] nuance for T2, T4:**

- **T3, T5 reuse the current parquets unchanged** — pure model/loss/inference changes. ✅
- **T2 [OPEN]:** our tfds `ytarget` `ispu` is already **continuous in [0,1]** (we binarize at
  0.5 for training). *If* that continuous value is a genuine hard/PU energy (or assignment)
  fraction, T2 needs **no reprocessing** — we just regress it instead of thresholding. This
  needs a quick check of what the continuous value actually represents in the postprocessing
  before we commit. If it is only a soft assignment weight and not an energy fraction, T2 is
  gated on U2 (reprocessing / upstream).
- **T4 [OPEN]:** N_vtx/ρ are **not stored features**, but can be **derived from existing `vz`
  per event in the dataloader** (count vz peaks / pT²-weighted density) → no reprocessing. A
  *clean reco* N_vtx would need U1 upstream. So T4 is doable on current parquets with a
  derivation step; the quality is proxy-limited until U1.

So the user's framing holds: **only T1 strictly requires reprocessing**; T3/T5 are free;
T2 is probably free (pending the `ispu`-meaning check); T4 is free with an on-the-fly derivation.

**Ordering rationale:** run T1 first (it's the only one blocked on a dataset pass, so kick it
off early), and because "do the extra features help/hurt/are-neutral" is the cheapest way to
learn whether the whole feature-engineering thesis (§2) is worth the upstream asks (§1). T2–T5
can proceed in parallel on the existing parquets once we lock the base config and harness.
