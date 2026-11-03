# PU200 target processing — findings & next steps (2026-10-13)

> **TODO (fix later) — do NOT drop input elements on a truth flag.** The committed v2 postprocessing
> drops `isPU == -1` input tracks/tracksters (truth-based), agreed with Mohamed *only for this study*.
> This cannot be reproduced on real data (no truth `isPU` at reco) → train/inference mismatch, and it
> works against the PU-head's purpose (the model never learns to reject the hardest pileup). Revert to
> keeping all inputs (and let the PU classifier handle them) once the A/B study is done. (2026-10-14)


Ran Mohamed's `scripts/phase2/postprocessing_ticl_nopu_nano_v2.py` on the PU200 ttbar nano
(`data/cms/phase2/offline/nano/ttbar_200pu`, 54 files). It runs as-is (schema-compatible,
~1 s/event, no crash). Target validated against gen; plots in `plots_pu200_target/`.
Scale per event (PU200 vs 0PU): inputs ~6,100 vs ~140 elements; SimTICLCandidates ~10,300
(82% flagged pileup); target ~957 particles; `HGCalGenPart` is signal-only (186/ev).

Target definition is settled: **reconstruct everything + store the per-particle `isPU` flag**
(Run-3 approach — train on the full target, subtract pileup at validation via `isPU<0.5`, let
PUPPI handle mitigation downstream). Three action items follow.

## 1. `isPU == -1` is RESOLVED IN-NANO — no production request needed (VERIFIED 2026-10-14)

At PU200, **~16.9% of `SimTICLCandidates` carry `isPU = -1`** (softest deposits, median pT
0.35 GeV). This is a *candidate-level* defect only: the **trackster-level** flags
(`ticlSimTracksters_isPU`, `ticlSimTrackstersfromCPs_isPU`) have **zero −1** — fully resolved
into 0/1.

`SimTICLCandidates` are **1:1 index-aligned with `ticlSimTrackstersfromCPs`** (identical
per-event counts; resolved flags agree **100.00%** on signal, **99.96%** on PU; verified over
4 files / 40 events / 407k candidates). So the fix is a one-line swap in postprocessing:

> **Use `ticlSimTrackstersfromCPs_isPU[i]` as the per-candidate PU label instead of
> `SimTICLCandidates_isPU[i]`.** Resolves all −1, needs no new production, no CaloParticle bx.

(There is no `CaloParticle` collection in the nano and `bunchCrossing` is a single event-level
scalar, so the walk-to-parent-CP-bx route is unavailable — but it's not needed.)

**What the resolved −1 actually are: ~99% pileup, not soft signal.** Resolved via the aligned
trackster flag, the −1 pool is **99.0% PU / 1.0% signal by count (98.1% / 1.9% by energy)**.

This **corrects the earlier reading of the jet plots.** Validating pileup-free target vs gen
jets (endcap, matched ΔR<0.2, gen pT>10):

| pileup-free definition | jets/ev (gen=5.8) | jet pT / gen pT |
|---|---|---|
| resolve −1 → drop PU (≈ exclude −1) | 5.3 (right count/spectrum) | 0.790 (21% low) |
| keep −1 as signal (old `isPU<0.5`)  | 26.3 (≈20 fake soft jets)  | 0.957 |

Keeping the −1 did NOT recover signal energy — the 0.957 was ~98% *pileup* energy contaminating
the jet cones, and that same PU is what spawned the ~20 fake soft jets/ev (the fake-jet
explosion *was the tell* that −1 = pileup). The honest pileup-free target resolves −1, drops the
99% that are PU, keeps the 1% signal → right jet count, JES ~0.79.

**Consequence:** the ~21%-low JES is a *separate* problem (soft hard-scatter signal lost below
the SimCand pT≥1 floor + genuine response), NOT fixable by reclassifying the −1. That floor is
the lever to revisit for JES — see Validation status below.

## 2. (Our side) Restrict inputs to the HGCAL endcap at processing time

The script currently keeps all ~6,100 input elements/event, ~43% of which are barrel
(|η|<1.5) and irrelevant to HGCAL MLPF. Add a |η| ∈ [1.5, 3.0] cut on the input elements
(and the upper 3.0 bound on the target, which currently only has the lower 1.5) in
`make_graph` / the element builder. First lever on the O(n²) attention cost; cuts the element
count substantially before any architecture change.

## 3. (Our side) Architecture for the input scale — HEPTv2

Even endcap-only, the element count (~thousands) is far above 0PU (~140). Full self-attention
(what the 0PU scan used — the `MATH` SDPA backend) is O(n²) and won't scale. Flash attention
(`AttentionType.FLASH`, H100-supported) is a memory/speed win but still O(n²), and our padded-batch
`key_padding_mask` doesn't fit the flash SDPA backend without packing/varlen plumbing.

Move to **HEPTv2** (sub-quadratic locality-sensitive attention) — already wired:
`conv_type="heptv2"`, `HEPTv2Layer` (`mlpf/model/heptv2.py`), `HEPTv2Config` in `conf.py`.
Action: build a PU200 `heptv2` model config and test it on the endcap-restricted input.

**VERIFIED (2026-10-13, smoke on MIG 1g.12gb / H100-NVL, bf16 fwd+bwd, d=256 h=16 w=1024,
heptv2 block_size=256 n_hashes=3 num_regions=140):** heptv2 is linear and fits with batch
headroom; dense attention (`MATH` SDPA) is O(n²) and OOMs right at our scale.

| N elements | heptv2 ms / mem | dense-attn ms / mem |
|---|---|---|
| 1024 | 6.5 / 0.14 GB | 5.5 / 0.35 GB |
| 2048 | 5.6 / 0.19 GB | 18.5 / 1.16 GB |
| 3584 | 8.1 / 0.27 GB | 55.8 / 3.39 GB |
| 5120 | 11.7 / 0.35 GB | 111 / 6.82 GB |
| 7680 | 16.9 / 0.49 GB | **OOM** |

PU200 endcap ~5,700 elements/ev → heptv2 ~0.4 GB/ev (can batch ~20+ on a 12 GB slice);
dense attention needs ~6.8 GB for a single event and cannot fit one with batching. Note:
heptv2's LSH blocking requires the padded sequence length to be a **multiple of `block_size`**
(pad the batch up and mask the tail) — otherwise the bucket reshape fails. Next: build the
PU200 `heptv2` model config and train on endcap-restricted input.

## Validation status

The pileup-free target (`isPU<0.5`) reconstructs gen hard-scatter jets at **~0.96 JES, ~0.26
resolution** (matched, endcap) — processing is sound modulo the −1 refinement above. The pT≥1
SimCand floor is what makes the exclude-−1 variant miss soft signal; revisit if tighter low-pT
JES is needed.
