#!/usr/bin/env python
"""
Phase-2 200PU -- input-feature separation between pileup and hard-scatter.

For each relevant input feature, overlay its distribution for target particles that are pileup
(target ispu >= 0.5) vs hard scatter (ispu < 0.5), using the element type the feature lives on
(track vs trackster). Shows which inputs give the model its pileup-mitigation power.
Reads the tfds (X aligned with ytarget), endcap, real target particles only.

Output: <out>/feature_pu_vs_signal/<feature>.pdf
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
C_CLS, C_ETA, C_ISPU = 0, 3, 7           # ytarget columns
SIG_C, PU_C = "#1f77b4", "#d62728"
# (X column, filename/axis label, element restriction: 'track'|'trackster'|'all', logx)
FEATURES = [
    (29, "vz", "track vertex z [cm]", "track", False),
    (27, "vx", "track vertex x [cm]", "track", False),
    (28, "vy", "track vertex y [cm]", "track", False),
    (13, "time", "trackster time", "trackster", False),
    (14, "timeerror", "trackster time error", "trackster", False),
    (12, "bary_z", "trackster barycenter z [cm]", "trackster", False),
    (30, "min_dR_track", r"min $\Delta R$ to track", "trackster", False),
    (31, "near_track_pt", "nearest-track pT [GeV]", "trackster", False),
    (32, "sum_pt_dR10", r"$\sum p_T$ tracks in $\Delta R<1.0$ [GeV]", "trackster", False),
    (37, "n_trk_dR05", r"N tracks in $\Delta R<0.5$", "trackster", False),
    (22, "pterror", "track pT error", "track", True),
    (26, "qoverperror", "track q/p error", "track", True),
    (25, "lambdaerror", "track lambda error", "track", True),
    (1, "pt", "pT [GeV]", "all", True),
    (2, "eta", r"$|\eta|$", "all", False),
    (10, "em_energy", "EM energy [GeV]", "trackster", False),
    (11, "nhits", "N hits", "track", False),
]


def cms_label(ax):
    hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfds-base", default="/shared/mlpf-phase2/tfds/cms_pf_phase2_ttbar200pu")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--configs", default="1,2,3")
    args = ap.parse_args()
    outdir = f"{args.out}/feature_pu_vs_signal"
    os.makedirs(outdir, exist_ok=True)

    Xs, typ, ispu = [], [], []
    for v in args.configs.split(","):
        bdir = f"{args.tfds_base}/{v.strip()}/1.0.0"
        if not os.path.isdir(bdir):
            continue
        b = tfds.builder_from_directory(bdir)
        for split in b.info.splits:
            ds = b.as_data_source(split=split)
            for i in range(len(ds)):
                e = ds[i]; X = np.asarray(e["X"]); yt = np.asarray(e["ytarget"])
                m = (yt[:, C_CLS] != 0) & (np.abs(yt[:, C_ETA]) >= ENDCAP[0]) & (np.abs(yt[:, C_ETA]) <= ENDCAP[1])
                Xs.append(X[m]); typ.append(X[m, 0]); ispu.append(yt[m, C_ISPU])
    X = np.concatenate(Xs); typ = np.concatenate(typ); pu = np.concatenate(ispu) >= 0.5
    print(f"loaded {len(X)} endcap target elements  ({(~pu).sum()} signal, {pu.sum()} pileup)")

    for col, fname, xlabel, restr, logx in FEATURES:
        tmask = {"track": typ == 1, "trackster": typ >= 2, "all": np.ones(len(X), bool)}[restr]
        val = X[tmask, col]; lab = pu[tmask]
        if fname == "eta":
            val = np.abs(val)
        if fname == "time":                       # drop the -99 "no-time" sentinel
            good = val > -90; val, lab = val[good], lab[good]
        s = val[~lab]; p = val[lab]
        if len(s) < 50 or len(p) < 50:
            print(f"  skip {fname}: too few ({len(s)} sig, {len(p)} pu)"); continue
        # robust binning
        allv = np.concatenate([s, p]); lo, hi = np.percentile(allv, [0.5, 99.5])
        if logx:
            lo = max(lo, np.percentile(allv[allv > 0], 1) if (allv > 0).any() else 1e-3)
            bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(max(hi, lo * 10)), 50)
        else:
            bins = np.linspace(lo, hi, 50)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.hist(p, bins=bins, histtype="step", lw=2.6, color=PU_C, density=True, label=f"pileup ({len(p)})")
        ax.hist(s, bins=bins, histtype="step", lw=2.6, color=SIG_C, density=True, label=f"hard scatter ({len(s)})")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("elements (normalised)")
        tag = {"track": "tracks", "trackster": "tracksters", "all": "all elements"}[restr]
        ax.legend(fontsize=13, title=f"{tag}, HGCAL endcap", title_fontsize=11)
        cms_label(ax)
        fig.tight_layout(); fig.savefig(f"{outdir}/{fname}.pdf", bbox_inches="tight"); plt.close(fig)
        print(f"  -> {fname}.pdf")
    print("DONE ->", outdir)


if __name__ == "__main__":
    main()
