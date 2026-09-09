#!/usr/bin/env python
"""Per-class |eta| distribution of the target particles: gen (Pythia) vs v1 (colleague) vs v2 (ours).
Plotted over the FULL |eta| range (no endcap pre-cut) so the target acceptance is visible:
gen spans all |eta|, v1 is |eta|>1.5 (his HGCAL-acceptance filter), v2 is sharply 1.5<|eta|<3 (our cut).
One figure per class, per sample -> final_target_<sample>/eta_<class>.pdf.
Reads <sample>/pkl_links (v2) and <sample>/pkl_mohamed/moh_* (v1); gen from the v2 pkls.
"""
import os, glob, pickle, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NANO = "/Users/fmokhtar/projects/particleflow/data/cms/phase2/offline/Aug31/nano"
NHAD = {130, 310, 2112, 3122, 3322, 3212, 421, 511}; NEUTR = {12, 14, 16}
CLASSES = ["photon", "nhad", "chad", "electron", "muon"]
CLAB = {"photon": "photons", "nhad": "neutral hadrons", "chad": "charged hadrons", "electron": "electrons", "muon": "muons"}
SL = {"ttbar_0pu": r"$t\bar{t}$ 0 PU", "qcd_0pu": "QCD 0 PU", "zll_0pu": r"$Z\to\ell\ell$ 0 PU"}


def pdgcls(p):
    a = int(abs(p))
    return "photon" if a == 22 else "electron" if a == 11 else "muon" if a == 13 else ("nhad" if a in NHAD else "chad")


def per_eta(files, source):
    out = {c: [] for c in CLASSES}
    for f in files:
        for ev in pickle.load(open(f, "rb")):
            if source == "gen":
                yp = np.asarray(ev["pythia"]).reshape(-1, 5)
                m = ~np.isin(np.abs(yp[:, 0]), list(NEUTR))          # no endcap cut -> show full acceptance
                pids, eta = yp[m, 0], yp[m, 2]                       # signed eta
            else:
                yt = ev["ytarget"]; v = yt["pid"] != 0
                pids, eta = yt["pid"][v], yt["eta"][v]               # signed eta
            cls = np.array([pdgcls(p) for p in pids]) if len(pids) else np.array([])
            for c in CLASSES:
                out[c].extend(eta[cls == c].tolist())
    return out


COL = {"gen": ("black", "gen (Pythia)"), "v1": ("tab:blue", "v1 (colleague)"), "v2": ("tab:red", "v2 (ours)")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=["ttbar_0pu", "qcd_0pu", "zll_0pu"])
    ap.add_argument("--outbase", default="/Users/fmokhtar/projects/particleflow/plots/phase2")
    a = ap.parse_args()
    b = np.linspace(-4.0, 4.0, 100)
    for s in a.samples:
        fl = sorted(glob.glob(f"{NANO}/{s}/pkl_links/*.pkl")); fm = sorted(glob.glob(f"{NANO}/{s}/pkl_mohamed/moh_*.pkl"))
        M = {"gen": per_eta(fl, "gen"), "v2": per_eta(fl, "target"), "v1": per_eta(fm, "target")}
        outdir = f"{a.outbase}/final_target_{s}"; os.makedirs(outdir, exist_ok=True)
        for c in CLASSES:
            fig, ax = plt.subplots(figsize=(7.4, 5.2))
            for k in ("gen", "v1", "v2"):
                ax.hist(np.array(M[k][c]), bins=b, histtype="step", lw=1.9, color=COL[k][0], label=COL[k][1])
            for lo, hi in [(-3.0, -1.5), (1.5, 3.0)]:
                ax.axvspan(lo, hi, color="0.9", zorder=0)
            for x in (-3.0, -1.5, 1.5, 3.0):
                ax.axvline(x, color="0.6", ls="--", lw=1)
            ax.set_xlabel(r"$\eta$", fontsize=14); ax.set_ylabel("particles / bin", fontsize=13); ax.set_yscale("log")
            ax.set_title(f"{CLAB[c]} target $\\eta$ — {SL.get(s, s)}  (shaded = HGCAL endcap)", fontsize=12)
            ax.legend(fontsize=11, frameon=False)
            plt.tight_layout(); plt.savefig(f"{outdir}/eta_{c}.pdf"); plt.close()
        print(f"{s}: wrote eta_<class>.pdf x{len(CLASSES)} -> {outdir}")


if __name__ == "__main__":
    main()
