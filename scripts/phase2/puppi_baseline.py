#!/usr/bin/env python
"""
PUPPI baseline for Phase-2 200PU pileup removal, to compare against the MLPF PU head.

Faithful (within our inputs) reimplementation of CMS PUPPI as described in
"Pileup mitigation at CMS in 13 TeV data", CMS-JME-18-001 / arXiv:2003.00503, Sec. 4:

  * Leading vertex (LV) = highest sum-pT^2 vertex (proxy: densest sum-pT^2 track-vz bin).
  * CHARGED particles (here: tracks, typ_idx==1) get a weight from vertex association:
        w = 1 if |dz to LV| < 0.3 cm  (LV),  else 0  (PU).                 [paper: 0.3 cm]
  * NEUTRAL particles (here: tracksters, typ_idx!=1) get a weight from the shape variable alpha:
        alpha_i = log sum_{j!=i, dR_ij<R0} (pT_j / dR_ij)^2     (Eq. 1, R0=0.4)
        neighbours j = charged-from-LV for |eta_i|<2.5 ; all particles for |eta_i|>2.5.
    The alpha of charged-PU particles gives the expected PU distribution (median aPU, RMS).
    A signed chi^2 (Eq. 2) is formed and turned into a weight (chi^2 CDF, 1 dof):
        schi2_i = (alpha_i - aPU)|alpha_i - aPU| / RMS_PU^2
        w_i = F_{chi2,1}(schi2_i) = erf(sqrt(schi2_i/2))   for schi2_i>0, else 0.
    For |eta|>2.5 aPU/RMS are transferred from |eta|<2.5 via CMS Table-1 transfer factors.
  * Rejections: w<0.01 -> 0 ; neutral with w*pT < A + B*N_vtx -> 0  (CMS Table-1 A,B).

This is a PUPPI-*style* baseline on tracks + TICL tracksters in the HGCAL endcap -- NOT the
officially tuned CMS HGCAL PUPPI (R0, dz, A/B, transfer factors were tuned for 2016, mu~23).
Label it "PUPPI-style" in the paper. Truth (isPU) is used only for evaluation, never as input.

Outputs under <out>/puppi_baseline/:
  puppi_weight_dist.pdf   PUPPI weight, truth signal vs pileup
  alpha_dist.pdf          alpha, charged-LV / charged-PU / neutral
  signed_chi2_dist.pdf    signed chi^2 for neutral signal vs pileup
  summary.json            per-particle AUC (all + per type), sum-pT response, config

Usage (mlpf venv):
  python scripts/phase2/puppi_baseline.py --data-dir /shared/mlpf-phase2/tfds_clean \
      --out /shared/mlpf-phase2/eval_plots/phase2_physics_clean --nevents 400
"""
import os
import json
import math
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.special import erf as _erf  # vectorised
except Exception:
    _erf = np.vectorize(math.erf)
try:
    import mplhep as hep

    hep.style.use("CMS")
    _HEP = True
except Exception:
    _HEP = False
import tensorflow_datasets as tfds

# ----------------------------------------------------------------------------- columns / constants
C_TYP, C_PT, C_ETA, C_SPHI, C_CPHI, C_VZ = 0, 1, 2, 3, 4, 29  # cms_phase2 X columns
YC_CLS, YC_PT, YC_ETA, YC_ISPU = 0, 2, 3, 7                   # ytarget columns (Y_FEATURES)
SAMPLE = "cms_pf_phase2_ttbar200pu"
RLABEL = r"$\mathrm{t}\bar{\mathrm{t}}$, 200 PU (14 TeV)"
ENDCAP = (1.48, 3.0)         # particles evaluated
NEIGH = (1.3, 3.2)           # elements allowed as neighbours (a bit wider)
R0 = 0.4                     # alpha cone (paper)
DZ_LV = 0.3                  # cm, charged LV association (paper: 0.3 cm ~ 15 sigma)
SIG_C, PU_C = "#1f77b4", "#d62728"
TYPES = [(1, "chrg_had"), (2, "neut_had"), (3, "photon"), (4, "electron"), (5, "muon")]


def table1(abseta):
    """CMS PUPPI Table 1: per-|eta| A,B [GeV] and transfer factors for aPU median / RMS."""
    A = np.where(abseta < 2.5, 0.2, 2.0)
    B = np.where(abseta < 2.5, 0.015, 0.13)
    tf_med = np.where(abseta < 2.5, 1.0, np.where(abseta < 3.0, 0.9, 0.75))
    tf_rms = np.where(abseta < 2.5, 1.0, np.where(abseta < 3.0, 1.2, 0.95))
    return A, B, tf_med, tf_rms


def f_chi2_1(x):
    """CDF of chi^2 with 1 dof: F(x) = erf(sqrt(x/2)) for x>=0, else 0."""
    out = np.zeros_like(x, dtype=np.float64)
    pos = x > 0
    out[pos] = _erf(np.sqrt(x[pos] / 2.0))
    return out


def z_lv_proxy(typ, pt, vz):
    """Leading-vertex z = pT^2-weighted mean of tracks in the densest (max sum-pT^2) 1 mm vz bin."""
    m = (typ == 1) & np.isfinite(vz) & (pt >= 1.0)
    if m.sum() == 0:
        return 0.0
    v = vz[m]
    w = pt[m] ** 2
    edges = np.arange(v.min() - 0.1, v.max() + 0.2, 0.1)
    if len(edges) < 2:
        return float(np.average(v, weights=w))
    idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
    sums = np.bincount(idx, weights=w, minlength=len(edges) - 1)
    sel = idx == int(np.argmax(sums))
    return float(np.average(v[sel], weights=w[sel])) if sel.any() else float(np.average(v, weights=w))


def alpha(i_idx, j_idx, eta, phi, pt):
    """alpha_i = log sum_{j in j_idx, j!=i, dR<R0} (pT_j/dR_ij)^2 ; 0 if no neighbour (paper)."""
    if len(i_idx) == 0:
        return np.zeros(0)
    ei, pi = eta[i_idx], phi[i_idx]
    ej, pj, qj = eta[j_idx], phi[j_idx], pt[j_idx]
    deta = ei[:, None] - ej[None, :]
    dphi = np.arctan2(np.sin(pi[:, None] - pj[None, :]), np.cos(pi[:, None] - pj[None, :]))
    dR = np.sqrt(deta ** 2 + dphi ** 2)
    same = i_idx[:, None] == j_idx[None, :]
    within = (dR < R0) & (~same)
    contrib = np.where(within, (qj[None, :] / np.clip(dR, 1e-3, None)) ** 2, 0.0).sum(1)
    return np.where(contrib > 0, np.log(np.where(contrib > 0, contrib, 1.0)), 0.0)


def puppi_weights(X, n_vtx, apply_ab):
    """Return per-element PUPPI weight w in [0,1] and diagnostic dict, for one event's X."""
    typ = X[:, C_TYP]
    pt = X[:, C_PT]
    eta = X[:, C_ETA]
    phi = np.arctan2(X[:, C_SPHI], X[:, C_CPHI])
    vz = X[:, C_VZ]
    abseta = np.abs(eta)
    N = len(X)
    allid = np.arange(N)

    real = typ != 0
    neigh = real & (abseta >= NEIGH[0]) & (abseta <= NEIGH[1])
    zlv = z_lv_proxy(typ, pt, vz)
    dz = np.abs(vz - zlv)

    is_chg = real & (typ == 1)
    chg_lv = is_chg & (dz < DZ_LV)
    chg_pu = is_chg & (dz >= DZ_LV)
    is_neu = real & (typ != 1)

    w = np.zeros(N)
    w[chg_lv] = 1.0  # charged from LV
    # charged PU stay 0

    lv_idx = allid[chg_lv & neigh]      # charged-LV neighbours (|eta|<2.5 rule)
    all_idx = allid[neigh]              # all neighbours (|eta|>2.5 rule)

    # reference PU alpha distribution: charged-PU in |eta|<2.5, alpha wrt charged-LV neighbours
    ref_idx = allid[chg_pu & (abseta < 2.5) & neigh]
    a_ref = alpha(ref_idx, lv_idx, eta, phi, pt)
    if len(a_ref) >= 5:
        a_pu_med = float(np.median(a_ref))
        a_pu_rms = float(np.sqrt(np.mean((a_ref - a_pu_med) ** 2)))
    else:
        a_pu_med, a_pu_rms = 0.0, 1.0
    a_pu_rms = max(a_pu_rms, 1e-3)

    # neutral alpha: |eta|<2.5 wrt charged-LV; |eta|>2.5 wrt all
    neu_in = allid[is_neu & (abseta < 2.5) & (abseta >= ENDCAP[0])]
    neu_out = allid[is_neu & (abseta >= 2.5) & (abseta <= ENDCAP[1])]
    a_neu = np.zeros(N)
    a_neu[neu_in] = alpha(neu_in, lv_idx, eta, phi, pt)
    a_neu[neu_out] = alpha(neu_out, all_idx, eta, phi, pt)

    neu = np.concatenate([neu_in, neu_out]).astype(int)
    _, _, tf_med, tf_rms = table1(abseta)
    med_i = a_pu_med * tf_med[neu]
    rms_i = a_pu_rms * tf_rms[neu]
    d = a_neu[neu] - med_i
    schi2 = d * np.abs(d) / rms_i ** 2            # Eq. 2 (signed)
    w[neu] = f_chi2_1(schi2)

    w[w < 0.01] = 0.0                              # reject >99% PU-like
    if apply_ab:
        A, B, _, _ = table1(abseta)
        cut = A + B * n_vtx
        kill = is_neu & (w * pt < cut)
        w[kill] = 0.0

    return w, dict(zlv=zlv, dz=dz, a_pu_med=a_pu_med, a_pu_rms=a_pu_rms,
                   a_neu=a_neu, schi2_full=_scatter(schi2, neu, N), chg_lv=chg_lv, chg_pu=chg_pu, is_neu=is_neu)


def _scatter(vals, idx, N):
    out = np.full(N, np.nan)
    out[idx] = vals
    return out


def auc(score, label):
    label = label.astype(float)
    P = label.sum()
    Nn = len(label) - P
    if P == 0 or Nn == 0:
        return np.nan
    _, inv, c = np.unique(score, return_inverse=True, return_counts=True)
    cs = np.cumsum(c)
    avg = cs - (c - 1) / 2.0
    r = avg[inv]
    return float((r[label == 1].sum() - P * (P + 1) / 2.0) / (P * Nn))


def est_nvtx(X):
    """Rough N_vertices proxy = number of 2 mm track-vz bins with >=2 tracks (pt>0.5)."""
    typ, pt, vz = X[:, C_TYP], X[:, C_PT], X[:, C_VZ]
    m = (typ == 1) & (pt > 0.5) & np.isfinite(vz)
    if m.sum() == 0:
        return 1
    b = np.round(vz[m] / 0.2).astype(int)
    u, c = np.unique(b, return_counts=True)
    return int(max(1, (c >= 2).sum()))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  ->", path)


def cms_label(ax, note=None):
    if _HEP:
        hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=15)
    if note:
        ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center", fontsize=11, style="italic", color="0.3")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/shared/mlpf-phase2/tfds_clean")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics_clean")
    ap.add_argument("--nevents", type=int, default=400)
    ap.add_argument("--nconfigs", type=int, default=2)
    ap.add_argument("--apply-ab", action="store_true",
                    help="apply the CMS A+B*Nvtx neutral-pT cut (mis-tuned at 200PU; off by default)")
    args = ap.parse_args()

    outdir = os.path.join(args.out, "puppi_baseline")
    os.makedirs(outdir, exist_ok=True)

    # accumulate per-element: PU score (1-w), truth ispu, type; and distributions
    S, L, T = [], [], []
    wa_sig, wa_pu = [], []
    a_neu_sig, a_neu_pu = [], []
    sc_sig, sc_pu = [], []
    sumpt_soft, sumpt_hard, sumpt_gen, sumpt_ideal = [], [], [], []
    nev = 0
    for ci in range(1, args.nconfigs + 1):
        ds = tfds.builder(f"{SAMPLE}/{ci}", data_dir=args.data_dir).as_data_source(split="test")
        for k in range(len(ds)):
            rec = ds[k]
            X = np.asarray(rec["X"], np.float64)
            Y = np.asarray(rec["ytarget"], np.float64)
            if X.shape[0] == 0:
                continue
            nvtx = est_nvtx(X)
            w, dg = puppi_weights(X, nvtx, args.apply_ab)

            typ = X[:, C_TYP]
            abseta = np.abs(X[:, C_ETA])
            real = typ != 0
            cls = Y[:, YC_CLS].astype(int)
            ispu = (Y[:, YC_ISPU] >= 0.5).astype(int)
            tpt = Y[:, YC_PT]
            teta = np.abs(Y[:, YC_ETA])

            # per-particle AUC selection: real target particle in endcap (matches MLPF head AUC)
            sel = real & (cls != 0) & (teta >= ENDCAP[0]) & (teta <= ENDCAP[1])
            S += list(1.0 - w[sel]); L += list(ispu[sel]); T += list(cls[sel])

            # distributions
            ec = real & (abseta >= ENDCAP[0]) & (abseta <= ENDCAP[1]) & (cls != 0)
            wa_sig += list(w[ec & (ispu == 0)]); wa_pu += list(w[ec & (ispu == 1)])
            neu = dg["is_neu"] & ec
            a_neu_sig += list(dg["a_neu"][neu & (ispu == 0)]); a_neu_pu += list(dg["a_neu"][neu & (ispu == 1)])
            sc_sig += list(dg["schi2_full"][neu & (ispu == 0)]); sc_pu += list(dg["schi2_full"][neu & (ispu == 1)])

            # sum-pT response, referenced to the TRUE SIGNAL sum-pT (hard scatter only), so it is
            # comparable to the ideal (truth-removal) number -- NOT the all-target sum (PU-dominated).
            e2 = real & (teta >= ENDCAP[0]) & (teta <= ENDCAP[1]) & (cls != 0)
            g = tpt[e2 & (ispu == 0)].sum()   # true signal sum-pT
            if g > 0:
                sumpt_soft.append((w[e2] * X[e2, C_PT]).sum() / g)
                sumpt_hard.append(X[e2 & (w > 0.5), C_PT].sum() / g)
                sumpt_ideal.append(X[e2 & (ispu == 0), C_PT].sum() / g)  # truth removal, reco pt
                sumpt_gen.append(1.0)
            nev += 1
            if nev >= args.nevents:
                break
        if nev >= args.nevents:
            break

    S, L, T = np.array(S), np.array(L), np.array(T)
    print(f"events={nev}  endcap target particles={len(S)}  pileup_frac={L.mean():.3f}")
    a_all = auc(S, L)
    print(f"PUPPI per-particle AUC (all types) = {a_all:.4f}   [MLPF T3 head = 0.925]")
    per_type = {}
    names = dict(TYPES)
    for t in sorted(set(T.tolist())):
        m = T == t
        if m.sum() > 50 and 0 < L[m].mean() < 1:
            at = auc(S[m], L[m])
            per_type[names.get(t, str(t))] = round(at, 4)
            print(f"   {names.get(t, t):<9s} n={m.sum():>7d} pu_frac={L[m].mean():.3f}  AUC={at:.4f}")

    def med(a):
        return float(np.median(a)) if len(a) else float("nan")

    resp = dict(sumpt_soft=med(sumpt_soft), sumpt_hard=med(sumpt_hard),
                sumpt_ideal_truth=med(sumpt_ideal), n_events=nev)
    print(f"sum-pT/gen  soft(w)={resp['sumpt_soft']:.3f}  hard(w>0.5)={resp['sumpt_hard']:.3f}  "
          f"ideal(truth)={resp['sumpt_ideal_truth']:.3f}")

    # ---- plots
    fig, ax = plt.subplots(figsize=(8, 5.5))
    b = np.linspace(0, 1, 41)
    ax.hist(wa_sig, bins=b, histtype="step", lw=2, density=True, color=SIG_C, label="signal (hard scatter)")
    ax.hist(wa_pu, bins=b, histtype="step", lw=2, density=True, color=PU_C, label="pileup")
    ax.set_xlabel("PUPPI weight w"); ax.set_ylabel("a.u."); ax.set_yscale("log"); ax.legend(fontsize=11)
    cms_label(ax, "endcap, all target particles"); _save(fig, os.path.join(outdir, "puppi_weight_dist.pdf"))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    b = np.linspace(-5, 25, 60)
    ax.hist(a_neu_sig, bins=b, histtype="step", lw=2, density=True, color=SIG_C, label="neutral signal")
    ax.hist(a_neu_pu, bins=b, histtype="step", lw=2, density=True, color=PU_C, label="neutral pileup")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("a.u."); ax.legend(fontsize=11)
    cms_label(ax, "tracksters (neutral)"); _save(fig, os.path.join(outdir, "alpha_dist.pdf"))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    b = np.linspace(-10, 10, 60)
    ax.hist(np.clip(sc_sig, -10, 10), bins=b, histtype="step", lw=2, density=True, color=SIG_C, label="neutral signal")
    ax.hist(np.clip(sc_pu, -10, 10), bins=b, histtype="step", lw=2, density=True, color=PU_C, label="neutral pileup")
    ax.set_xlabel(r"signed $\chi^2$"); ax.set_ylabel("a.u."); ax.legend(fontsize=11)
    cms_label(ax, "tracksters (neutral)"); _save(fig, os.path.join(outdir, "signed_chi2_dist.pdf"))

    summary = dict(auc_all=round(a_all, 4), auc_per_type=per_type, pileup_frac=round(float(L.mean()), 4),
                   n_particles=int(len(S)), response=resp, apply_ab=bool(args.apply_ab),
                   mlpf_t3_head_auc=0.925, config=dict(R0=R0, dz_lv=DZ_LV, note="PUPPI-style, tracks+tracksters, endcap"))
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  ->", os.path.join(outdir, "summary.json"))


if __name__ == "__main__":
    main()
