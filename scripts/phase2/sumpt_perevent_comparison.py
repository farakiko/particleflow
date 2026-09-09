#!/usr/bin/env python
"""Per-event ΣpT spectra per particle class: gen (Pythia) vs v1 (colleague) vs v2 (ours), CMS-style.
For each event, sum the pT of all particles of a class (endcap 1.5<|eta|<3), then histogram those
per-event values (Events/bin, log-log). One figure per class, per sample -> final_target_<sample>/.

Reads:  <sample>/pkl_links (v2, ours)  and  <sample>/pkl_mohamed/moh_* (v1, colleague).
Gen (Pythia) is taken from the v2 pkls (identical events in both).
"""
import os, glob, pickle, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NANO = "/Users/fmokhtar/projects/particleflow/data/cms/phase2/offline/Aug31/nano"
NHAD = {130, 310, 2112, 3122, 3322, 3212, 421, 511}; NEUTR = {12, 14, 16}
CLASSES = ["photon", "nhad", "chad", "electron", "muon"]
XLAB = {"photon": "Photon", "nhad": "Neutral Hadron", "chad": "Charged Hadron", "electron": "Electron", "muon": "Muon"}
SL = {"ttbar_0pu": r"$t\bar{t}$ 0 PU", "qcd_0pu": "QCD 0 PU", "zll_0pu": r"$Z\to\ell\ell$ 0 PU"}
ELO, EHI = 1.5, 3.0


def pdgcls(p):
    a = int(abs(p))
    return "photon" if a == 22 else "electron" if a == 11 else "muon" if a == 13 else ("nhad" if a in NHAD else "chad")


def perevent_sumpt(files, source):
    out = {c: [] for c in CLASSES}
    for f in files:
        for ev in pickle.load(open(f, "rb")):
            if source == "gen":
                yp = np.asarray(ev["pythia"]).reshape(-1, 5)
                m = (np.abs(yp[:, 2]) > ELO) & (np.abs(yp[:, 2]) < EHI) & ~np.isin(np.abs(yp[:, 0]), list(NEUTR))
                pids, pts = yp[m, 0], yp[m, 1]
            else:
                yt = ev["ytarget"]; v = yt["pid"] != 0
                pid, pt, eta = yt["pid"][v], yt["pt"][v], yt["eta"][v]
                mm = (np.abs(eta) > ELO) & (np.abs(eta) < EHI); pids, pts = pid[mm], pt[mm]
            cls = np.array([pdgcls(p) for p in pids]) if len(pids) else np.array([])
            for c in CLASSES:
                out[c].append(pts[cls == c].sum() if len(pts) else 0.0)
    return out


def cms_labels(ax, right):
    ax.text(0.0, 1.015, "CMS", transform=ax.transAxes, fontweight="bold", fontsize=15)
    ax.text(0.125, 1.015, "Simulation", transform=ax.transAxes, style="italic", fontsize=13)
    ax.text(1.0, 1.015, right, transform=ax.transAxes, ha="right", fontsize=13)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=["ttbar_0pu", "qcd_0pu", "zll_0pu"])
    ap.add_argument("--outbase", default="/Users/fmokhtar/projects/particleflow/plots/phase2")
    a = ap.parse_args()
    b = np.logspace(0, np.log10(500), 60)
    for s in a.samples:
        fl = sorted(glob.glob(f"{NANO}/{s}/pkl_links/*.pkl"))
        fm = sorted(glob.glob(f"{NANO}/{s}/pkl_mohamed/moh_*.pkl"))
        gen = perevent_sumpt(fl, "gen"); v2 = perevent_sumpt(fl, "target"); v1 = perevent_sumpt(fm, "target")
        outdir = f"{a.outbase}/final_target_{s}"; os.makedirs(outdir, exist_ok=True)
        for c in CLASSES:
            fig, ax = plt.subplots(figsize=(7, 5.3))
            for arr, lab, col in [(gen[c], "Pythia (gen)", "black"), (v1[c], "v1 (colleague)", "tab:blue"), (v2[c], "v2 (ours)", "tab:red")]:
                ax.hist(np.array(arr), bins=b, histtype="step", lw=1.9, color=col, label=lab)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(rf"$\Sigma p_T$ per event {XLAB[c]} (GeV)", fontsize=14)
            ax.set_ylabel("Events / bin", fontsize=14)
            cms_labels(ax, SL.get(s, s)); ax.legend(fontsize=11, loc="upper right", frameon=False)
            plt.tight_layout(); plt.savefig(f"{outdir}/sumpt_perevent_{c}.pdf"); plt.close()
        print(f"{s}: wrote sumpt_perevent_<class>.pdf x{len(CLASSES)} -> {outdir}")


if __name__ == "__main__":
    main()
