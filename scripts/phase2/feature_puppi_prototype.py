#!/usr/bin/env python
"""
Phase-2 200PU -- prototype & validate the engineered PUPPI-style features (T1) on the EXISTING
tfds, BEFORE committing to a postprocessing re-run. The tfds X already carries per-element
eta / phi / pt / vz, so we can compute the candidate features here and check whether they
separate pileup from hard-scatter *at fixed pT* (the real test), per particle type.

Features prototyped (per endcap input element):
  z_LV      event leading-vertex z proxy = pT^2-weighted median of track (typ==1) vz
  dz_to_lv  |vz - z_LV|                       (charged/track discriminant; tracksters have no vz)
  alpha_F   log( sum_{dR<0.4, j!=i} pT_j / dR_ij )   over ALL neighbors
  alpha_C   same, but only over LV charged tracks (typ==1 & |vz-z_LV|<DZCUT)   <- PUPPI's alpha^C

For each feature we overlay pileup vs hard-scatter (1D), draw feature-vs-pT (2D), and print the
per-feature AUC (pileup vs signal) overall and per type. If alpha_C / dz_to_lv discriminate at
fixed pT where the current inputs do not, the reprocess to add them is justified.

Output: <out>/feature_puppi_prototype/{<feat>_1d.pdf, <feat>_2d.pdf}
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import tensorflow_datasets as tfds

hep.style.use("CMS")
RLABEL = r"$\mathrm{t}\bar{\mathrm{t}}$, 200 PU (14 TeV)"
ENDCAP = (1.48, 3.0)
NEIGH_ETA = (1.3, 3.2)        # elements considered as neighbours (a bit wider than the endcap)
C_TYP, C_PT, C_ETA, C_SPHI, C_CPHI, C_VZ = 0, 1, 2, 3, 4, 29
YC_CLS, YC_ETA, YC_ISPU = 0, 3, 7
DRMAX = 0.4
DR_FLOOR = 1e-3
DZCUT = 0.10                   # cm, |vz - z_LV| for a track to count as leading-vertex
SIG_C, PU_C = "#1f77b4", "#d62728"
TYPES = [(1, "chrg_had"), (2, "neut_had"), (3, "photon"), (4, "electron"), (5, "muon")]


def z_lv_proxy(typ, pt, vz):
    """Leading-vertex z proxy = the vz bin with the largest sum of track pT^2 (the standard
    'highest-sum-pT^2 vertex' definition), refined to the pT^2-weighted mean of tracks in it.
    A global pT^2-weighted median instead sits at the luminous-region centre, not the LV."""
    m = (typ == 1) & np.isfinite(vz) & (pt >= 1.0)
    if m.sum() == 0:
        return 0.0
    v = vz[m]; w = (pt[m] ** 2)
    edges = np.arange(v.min() - 0.1, v.max() + 0.2, 0.1)   # 1 mm vz bins
    if len(edges) < 2:
        return float(np.average(v, weights=w))
    idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
    sums = np.bincount(idx, weights=w, minlength=len(edges) - 1)
    b = int(np.argmax(sums))
    sel = idx == b
    return float(np.average(v[sel], weights=w[sel])) if sel.any() else float(np.average(v, weights=w))


def event_features(X):
    """Return dict of per-element arrays (alpha_F, alpha_C, dz_to_lv) over endcap elements,
    plus the index mask into X of those elements."""
    typ = X[:, C_TYP]; pt = X[:, C_PT]; eta = X[:, C_ETA]; vz = X[:, C_VZ]
    phi = np.arctan2(X[:, C_SPHI], X[:, C_CPHI])
    zlv = z_lv_proxy(typ, pt, vz)

    sel = (typ != 0) & (np.abs(eta) >= NEIGH_ETA[0]) & (np.abs(eta) <= NEIGH_ETA[1])
    idx = np.where(sel)[0]
    e, p, pts, tp, vzs = eta[idx], phi[idx], pt[idx], typ[idx], vz[idx]
    M = len(idx)
    alpha_f = np.full(M, -5.0); alpha_c = np.full(M, -5.0)
    dz = np.abs(vzs - zlv)
    is_lv_trk = (tp == 1) & (dz < DZCUT)
    if M > 1:
        deta = e[:, None] - e[None, :]
        dphi = np.arctan2(np.sin(p[:, None] - p[None, :]), np.cos(p[:, None] - p[None, :]))
        dR = np.sqrt(deta ** 2 + dphi ** 2)
        np.fill_diagonal(dR, 99.0)
        within = dR < DRMAX
        contrib = pts[None, :] / np.clip(dR, DR_FLOOR, None)   # pT_j / dR_ij
        sf = np.where(within, contrib, 0.0).sum(axis=1)
        sc = np.where(within & is_lv_trk[None, :], contrib, 0.0).sum(axis=1)
        alpha_f = np.log(sf + 1e-3)
        alpha_c = np.log(sc + 1e-3)
    return idx, dict(alpha_F=alpha_f, alpha_C=alpha_c, dz_to_lv=dz, _typ=tp)


def auc(score, label):
    order = np.argsort(-score, kind="mergesort")
    s = label[order].astype(float); P = s.sum(); N = len(s) - P
    if P == 0 or N == 0:
        return np.nan
    tpr = np.cumsum(s) / P; fpr = np.cumsum(1 - s) / N
    tr = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(tr(np.concatenate([[0], tpr]), np.concatenate([[0], fpr])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfds-base", default="/shared/mlpf-phase2/tfds/cms_pf_phase2_ttbar200pu")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--configs", default="1,2")
    ap.add_argument("--max-events", type=int, default=300)
    args = ap.parse_args()
    outdir = f"{args.out}/feature_puppi_prototype"; os.makedirs(outdir, exist_ok=True)

    feats = {k: [] for k in ("alpha_F", "alpha_C", "dz_to_lv")}
    lab, pts, typs, rawrows = [], [], [], []
    nev = 0
    for v in args.configs.split(","):
        bdir = f"{args.tfds_base}/{v.strip()}/1.0.0"
        if not os.path.isdir(bdir):
            continue
        b = tfds.builder_from_directory(bdir)
        for split in b.info.splits:
            ds = b.as_data_source(split=split)
            for i in range(len(ds)):
                if nev >= args.max_events:
                    break
                e = ds[i]; X = np.asarray(e["X"], np.float64); yt = np.asarray(e["ytarget"], np.float64)
                idx, f = event_features(X)
                # keep elements that are a real target particle in the endcap
                cls = yt[idx, YC_CLS]; teta = np.abs(yt[idx, YC_ETA]); tispu = yt[idx, YC_ISPU]; tpt = yt[idx, C_PT]
                keep = (cls != 0) & (teta >= ENDCAP[0]) & (teta <= ENDCAP[1])
                for k in feats:
                    feats[k].append(f[k][keep])
                lab.append((tispu[keep] >= 0.5).astype(int))
                pts.append(yt[idx, 2][keep]); typs.append(cls[keep].astype(int))
                rawrows.append(X[idx][keep])
                nev += 1
            if nev >= args.max_events:
                break
    for k in feats:
        feats[k] = np.concatenate(feats[k])
    lab = np.concatenate(lab); pts = np.concatenate(pts); typs = np.concatenate(typs)
    rawX = np.concatenate(rawrows); raws = {c: rawX[:, c] for c in (C_VZ, 13, 12, 26)}
    pu = lab == 1
    print(f"events {nev}, elements {len(lab)}  (signal {int((~pu).sum())}, pileup {int(pu.sum())})")

    def auc_line(tag, v):
        fin = np.isfinite(v)
        return (f"  {tag:11s} overall {auc(v[fin], lab[fin]):.3f} | " +
                " ".join(f"{nm}:{auc(v[fin & (typs == cid)], lab[fin & (typs == cid)]):.3f}" for cid, nm in TYPES))

    print("\nENGINEERED feature AUC (pileup vs signal):")
    for k in feats:
        print(auc_line(k, feats[k]))

    print("\nEXISTING raw-input AUC (for reference; what the model already sees):")
    for col, tag in [(C_VZ, "vz"), (13, "time"), (12, "bary_z"), (26, "qoverperr")]:
        rv = raws[col].copy()
        if tag == "time":                       # drop the -99 sentinel
            rv = np.where(rv > -90, rv, np.nan)
        print(auc_line(tag, rv))

    # 1D overlays + 2D vs pT
    for k in feats:
        v = feats[k]; fin = np.isfinite(v)
        s = v[fin & ~pu]; pp = v[fin & pu]
        lo, hi = np.percentile(v[fin], [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.hist(pp, bins=bins, histtype="step", lw=2.6, color=PU_C, density=True, label=f"pileup ({len(pp)})")
        ax.hist(s, bins=bins, histtype="step", lw=2.6, color=SIG_C, density=True, label=f"hard scatter ({len(s)})")
        ax.set_xlabel(k); ax.set_ylabel("elements (normalised)")
        ax.legend(fontsize=13, title="HGCAL endcap target particles", title_fontsize=10)
        hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=16)
        fig.tight_layout(); fig.savefig(f"{outdir}/{k}_1d.pdf", bbox_inches="tight"); plt.close(fig)
        # 2D vs pT (column-normalised)
        PTB = np.logspace(0, 2, 15); yb = np.linspace(lo, hi, 22)
        def cn(H):
            sm = H.sum(axis=1, keepdims=True); sm[sm == 0] = 1; return H / sm
        Hs = cn(np.histogram2d(pts[fin & ~pu], s, [PTB, yb])[0])
        Hp = cn(np.histogram2d(pts[fin & pu], pp, [PTB, yb])[0])
        vmax = max(Hs.max(), Hp.max())
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
        for ax, H, t in [(a1, Hs, "hard scatter"), (a2, Hp, "pileup")]:
            pc = ax.pcolormesh(PTB, yb, H.T, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xscale("log"); ax.set_xlabel("truth particle pT [GeV]"); ax.set_xlim(PTB[0], PTB[-1])
            ax.text(0.04, 0.95, t, transform=ax.transAxes, color="w", va="top", fontsize=14, fontweight="bold")
        a1.set_ylabel(k); fig.colorbar(pc, ax=[a1, a2], label=r"P(feat | $p_T$)", pad=0.02)
        hep.cms.label(ax=a1, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=13)
        fig.savefig(f"{outdir}/{k}_2d.pdf", bbox_inches="tight"); plt.close(fig)
        print(f"  -> {k}_1d.pdf, {k}_2d.pdf")
    print("DONE ->", outdir)


if __name__ == "__main__":
    main()
