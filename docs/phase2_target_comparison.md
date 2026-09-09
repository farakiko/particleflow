# Target comparison: `run3style` (ours) vs colleague's (`mohamed`) postprocessing

Rigorous, tested comparison. Every claim is **[CODE file:line]** (logic) or **[RUN]** (measured on the
*same* 100 ttbar 0PU events). Generated 2026-09-09.

- **ours:** `scripts/phase2/postprocessing_run3style.py`
- **his original:** `.staging/cms/postprocessing_ticl_ttbar_nopu.py` (reads a multi-tree TiclDumper ntuple)
- **his logic on NanoAOD:** `scripts/phase2/postprocessing_mohamed.py` (faithful port of his *target* logic)

**Caveat (port vs original):** the port faithfully reproduces his **target definition + association**
(verified: score cuts, fragmentation, gen-filter, representative-element), but **not** (a) his feature
set (port=10, original=~37) nor (b) his genjet (port uses all-gen like us; original uses matched-gen-only).
Feature comparison uses his **original**; the empirical jet-response comparison is apples-to-apples.

## 1. The five fundamental differences

| axis | his | ours |
|---|---|---|
| **Calo input** | `ticlTracksterLinks` — **post-linking** (CMSSW merged showers) [mohamed:28] | `ticlTrackstersCLUE3DHigh` — **pre-linking** raw tracksters [run3:32] |
| tracksters/ev [RUN] | 42 | 57 |
| **Target granularity** | **fragments**: one target per trackster (neutral/γ) [mohamed:168-190] | **one target per particle** (argmax trackster) [run3:112-123] |
| **Association** | associator `simToReco sharedE` + **score cuts** r>0.6 or s>0.9 → drop [mohamed:135] | charged→`trackIdx`; neutral→**argmax sharedEnergy**, **no cuts** [run3:113-123] |
| **Gen filter** | keep only simcands matched to a stable gen (ΔR, in HGCAL) [mohamed:64-90] | none |
| **Energy** | truth `energy` (e⁻ `raw_energy`), **split across tracksters** [mohamed:128,188] | same, e⁻ `raw_energy`, **all on one element** [run3:102-105] |

Shared: anti-kt R=0.4, pT≥3; neutral pT = E·sinθ (massless); endcap via eta-sign.

## 2. Input features
His **original** `elem_branches` [orig:53-66] is richest (~37): adds **track-density** features we lack
(`min_dR_track`, `near_track_pt`, `sum_pt_dR10`, `n_trk_dR01..05`) and `shower_depth` (E-weighted layer |z|),
plus the muon-ID/track-errors/`bary_z`/timing we now have. Ours = 29 (§10). Port = 10 (target-only).

## 3. Empirical, same 100 ttbar events [RUN]

| per event | gen (endcap) | mohamed | run3style |
|---|---|---|---|
| chad count | 35.8 | 10.9 | 80.0 |
| nhad count | 9.1 | 2.3 | 5.8 |
| photon count | 17.5 | 8.8 | 16.0 |
| target particles | — | 22.4 | 102.9 |
| chad ΣpT [GeV] | 89 | 58 (0.65×) | 141 (1.58×) |
| nhad ΣpT | 27 | 12 (0.44×) | 29 (1.07×) |
| photon ΣpT | 33 | 18 (0.55×) | 31 (0.94×) |
| **jet response median** | 1.0 | **0.750** | **0.980** |
| **jet resolution IQR/med** | — | **0.424** | **0.068** |

## 4. Interpretation (physics)
- His gen-filter + score cuts **discard real energy** (every class 0.44–0.65× gen ΣpT) → jets at **0.75**,
  broad (IQR/med 0.42). A ~25% jet-scale loss baked into the target.
- Ours **preserves energy** → jets at **0.98**, 6× tighter (0.068). Decisive for a learnable, correct jet scale.
- Ours **over-produces charged hadrons** (80 vs 36, ΣpT 1.58×): every track → a particle, incl. detector
  secondaries (nuclear interactions, conversions, decays) with no gen-primary. His filter removes these
  (purer); ours keeps them (complete but contaminated). Jet response stays 0.98 (extras are soft/spread).
- nhad: ours closes ΣpT (1.07×) vs his 0.44×.

## 5. The calo-input fork (the deepest difference)
TICL chain: layer clusters → **CLUE3DHigh tracksters** (raw 3D; one particle → several) → **TracksterLinks**
(CMSSW links the fragments per particle) → TICLCandidates (final PF).
- **His (post-link):** MLPF starts from CMSSW-merged per-particle blobs — easier, but **capped by CMSSW's
  linking**.
- **Ours (pre-link):** MLPF must **learn the linking** (gather a nhad's scattered tracksters, split e/brem) —
  the core MLPF thesis; higher ceiling, harder. This is *why* our nhad task is hard (§9B: single-trackster
  capture ~0.48; the network must re-gather what the linker would have merged).

## 6. Tested: does adding his gen-match chad cleanup help? [RUN] — NO
Added *only* his charged gen-match filter to `run3style` (scratch experiment), same 100 events:

| | chad/ev | chad ΣpT | jet resp | IQR/med |
|---|---|---|---|---|
| run3style, no filter | 80.0 | 141 | 0.980 | 0.068 |
| + gen-match chad filter | **13.4** | 65 | **0.915** | 0.205 |

- Reproduces his numbers (13.4 ≈ his 10.9) → faithful. But it's **blunt**: drops chad to **below gen** (13.4
  vs 35.8). It matches the track's **HGCAL-extrapolated** position to the gen **vertex** at ΔR<0.1; charged
  tracks **bend** (3.8 T), so low-pT real ones fail (count→0.17× but ΣpT→0.46× = keeps high-pT, drops low-pT).
- Costs jet scale (**0.98→0.915**) and 3× resolution. The removed secondary energy is real.
- **Conclusion:** the secondary-chad over-count needs a *surgical* fix (charge/pT-aware match, or remove only
  fake/duplicate tracks), **not** his blunt ΔR-to-vertex filter. Not adopting it as-is.

## 7. Verdict — which is better, for what
- **Jet energy scale & resolution (headline physics): ours** — 0.98/tight vs 0.75/broad; his structurally
  can't reach 1 (throws energy away).
- **Target purity (charged fakes): his** — his filter removes secondaries (at a jet-scale cost, see §6).
- **Conceptual (calo input): ours does the real PF job** (learns linking); his offloads it to CMSSW.
- **Features: his original is richest** — has track↔trackster proximity features we lack (cheap to add, §9E).

**Best-of-both target:** keep ours (complete energy, pre-linking, correct jet scale) + add his **track-density
features** + a **surgical** (not blunt) secondary-chad cleanup. His blunt gen-filter is *not* worth adopting
(tested, §6).

---

## 8. Settled design on LINKS — v2 (ours) vs v1 (colleague)

We adopt **`ticlTracksterLinks`** as the collection (stays close to the colleague; links slightly
edges neutral-hadron energy in the 10-epoch trained comparison, §clue3d-vs-links). On that common
collection, the two *target definitions* differ as:

| aspect | v1 (colleague) | v2 (ours) |
|---|---|---|
| granularity | fragment: one target per trackster | one target per particle (argmax) |
| truth energy | split across tracksters | full on the primary element |
| association score cuts | yes (`r>0.6` or `s>0.9` drop) | none |
| gen-match filter | yes (unmatched tracks → null/PU) | none (every simcand is a target) |
| charged anchoring / e⁻ energy | trackIdx / raw_energy | same |
| philosophy | purity-first (sparse) | completeness-first (keep all energy) |

**Measured, target-vs-gen, 0 PU** (`scripts/phase2/final_target_comparison.py`, full 3 samples):

| metric | v1 | v2 |
|---|---|---|
| jet-pT response — ttbar | 0.72 | **0.97** |
| jet-pT response — qcd | 0.69 | **0.98** |
| jet-pT response — zll | 0.96 | **0.99** |
| charged-had efficiency | low | **high (~0.95)** |
| neutral-had ΣpT / gen | suppressed | **close to gen** |
| fake rate | **lower (purer)** | higher (secondaries) |
| particle pT resolution | comparable | comparable |

**Two knob-isolation studies on links** (`target_study_{granularity,cuts}`):
- **Granularity/energy is moot on links** — 86% of neutrals link to ≤1 link-trackster (merging already
  collapsed them), so fragment ≡ one-per-particle on jets & ΣpT. → keep one-per-particle.
- **Score cuts hurt** — jet response 0.972→0.949, nhad ΣpT 0.84→0.64. → keep no cuts.

**Conclusion:** v2 = **links collection (v1's) + complete, uncut, one-per-particle target (ours)** — near-
unity jet scale and high efficiency, at a higher (mostly-secondary) fake rate. v1's cuts/filter trade
~25–30% of hadronic jet energy for purity. Plots: `plots/phase2/final_target_<sample>/` (v2=red, v1=blue).
