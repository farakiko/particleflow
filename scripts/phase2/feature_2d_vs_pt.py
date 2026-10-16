#!/usr/bin/env python
"""
Phase-2 200PU -- input features vs pT, pileup vs hard scatter (2D).

For each feature, two side-by-side 2D maps (hard scatter | pileup) of feature vs truth pT.
Each pT column is normalised to unit area, so the colour is P(feature | pT, class): if the two
panels differ at a given pT, the feature discriminates *at fixed pT* (real, pT-independent power);
if they only differ because the classes live at different pT, the panels look alike column-by-column.

Output: <out>/feature_2d_vs_pt/<feature>.pdf
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
C_CLS, C_PT, C_ETA, C_ISPU = 0, 1, 3, 7
# (X col, filename, y-axis label, restriction, logy)
FEATURES = [
    (29, "vz", "track vertex z [cm]", "track", False),
    (13, "time", "trackster time", "trackster", False),
    (14, "timeerror", "trackster time error", "trackster", False),
    (12, "bary_z", "trackster barycenter z [cm]", "trackster", False),
    (30, "min_dR_track", r"min $\Delta R$ to track", "trackster", False),
    (31, "near_track_pt", "nearest-track pT [GeV]", "trackster", True),
    (32, "sum_pt_dR10", r"$\sum p_T$ tracks $\Delta R<1.0$ [GeV]", "trackster", True),
    (37, "n_trk_dR05", r"N tracks in $\Delta R<0.5$", "trackster", False),
    (22, "pterror", "track pT error", "track", True),
    (26, "qoverperror", "track q/p error", "track", True),
    (25, "lambdaerror", "track lambda error", "track", True),
    (2, "eta", r"$|\eta|$", "all", False),
    (10, "em_energy", "EM energy [GeV]", "trackster", True),
    (11, "nhits", "N hits", "track", False),
]
PTB = np.logspace(0, 2, 15)


def load(tfds_base, configs):
    Xs, typ, ispu = [], [], []
    for v in configs.split(","):
        bdir = f"{tfds_base}/{v.strip()}/1.0.0"
        if not os.path.isdir(bdir):
            continue
        b = tfds.builder_from_directory(bdir)
        for split in b.info.splits:
            ds = b.as_data_source(split=split)
            for i in range(len(ds)):
                e = ds[i]; X = np.asarray(e["X"]); yt = np.asarray(e["ytarget"])
                m = (yt[:, C_CLS] != 0) & (np.abs(yt[:, C_ETA]) >= ENDCAP[0]) & (np.abs(yt[:, C_ETA]) <= ENDCAP[1])
                Xs.append(X[m]); typ.append(X[m, 0]); ispu.append(yt[m, C_ISPU])
    return np.concatenate(Xs), np.concatenate(typ), np.concatenate(ispu) >= 0.5


def colnorm(H):
    s = H.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return H / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfds-base", default="/shared/mlpf-phase2/tfds/cms_pf_phase2_ttbar200pu")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--configs", default="1,2,3")
    args = ap.parse_args()
    outdir = f"{args.out}/feature_2d_vs_pt"
    os.makedirs(outdir, exist_ok=True)
    X, typ, pu = load(args.tfds_base, args.configs)
    print(f"loaded {len(X)} elements ({(~pu).sum()} signal, {pu.sum()} pileup)")

    for col, fname, ylabel, restr, logy in FEATURES:
        tm = {"track": typ == 1, "trackster": typ >= 2, "all": np.ones(len(X), bool)}[restr]
        pt = X[tm, C_PT]; val = X[tm, col]; lab = pu[tm]
        if fname == "eta":
            val = np.abs(val)
        if fname == "time":
            good = val > -90; pt, val, lab = pt[good], val[good], lab[good]
        if logy:
            good = val > 0; pt, val, lab = pt[good], val[good], lab[good]
        if (~lab).sum() < 50 or lab.sum() < 50:
            print(f"  skip {fname}"); continue
        lo, hi = np.percentile(val, [0.5, 99.5])
        yb = np.logspace(np.log10(max(lo, 1e-4)), np.log10(hi), 22) if logy else np.linspace(lo, hi, 22)
        Hs = colnorm(np.histogram2d(pt[~lab], val[~lab], [PTB, yb])[0])
        Hp = colnorm(np.histogram2d(pt[lab], val[lab], [PTB, yb])[0])
        vmax = max(Hs.max(), Hp.max())
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
        for ax, H, ttl in [(a1, Hs, "hard scatter"), (a2, Hp, "pileup")]:
            pc = ax.pcolormesh(PTB, yb, H.T, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("truth particle pT [GeV]"); ax.set_xlim(PTB[0], PTB[-1])
            ax.text(0.04, 0.95, ttl, transform=ax.transAxes, fontsize=15, color="w",
                    va="top", fontweight="bold")
        a1.set_ylabel(ylabel)
        fig.colorbar(pc, ax=[a1, a2], label="P(feature | pT)  [column-normalised]", pad=0.02)
        hep.cms.label(ax=a1, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=13)
        fig.savefig(f"{outdir}/{fname}.pdf", bbox_inches="tight"); plt.close(fig)
        print(f"  -> {fname}.pdf")
    print("DONE ->", outdir)


if __name__ == "__main__":
    main()
