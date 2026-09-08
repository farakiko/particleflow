#!/usr/bin/env python
"""
Efficiency (vs gen) and fake rate (vs target) as a function of pT, per particle class,
for the Run3-style target. Matching dR<0.2. Two curves each:
  - "same class": require the matched partner to be the same particle class
  - "any":        match to any class
Region: HGCAL 1.5<|eta|<3.0.  gen = pythia (status1, no-nu);  target = ytarget.
"""
import glob, pickle, argparse, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

DR = 0.2
ETA_LO, ETA_HI = 1.5, 3.0
CLASSES = ["photon", "nhad", "chad", "electron", "muon"]
CLABEL  = {"photon": "photons", "nhad": "neutral hadrons", "chad": "charged hadrons",
           "electron": "electrons", "muon": "muons"}
NHAD = {130, 310, 2112, 3122, 3322, 3212, 421, 511}
PT_EDGES = np.array([1, 2, 3, 5, 8, 13, 20, 35, 60, 100, 200], float)
PT_CEN = np.sqrt(PT_EDGES[:-1]*PT_EDGES[1:])
NB = len(PT_CEN)

def cls(apid):
    a = int(abs(apid))
    if a == 22: return "photon"
    if a == 11: return "electron"
    if a == 13: return "muon"
    if a in NHAD: return "nhad"
    return "chad"

def ptbin(pt):
    return int(np.clip(np.digitize(pt, PT_EDGES) - 1, 0, NB-1))

def collect(files):
    # eff[class] = [tot, matched_any, matched_same] per ptbin ; fake[class] similarly (target side)
    eff = {c: np.zeros((NB, 3)) for c in CLASSES}
    fake = {c: np.zeros((NB, 3)) for c in CLASSES}
    for f in files:
        for ev in pickle.load(open(f, "rb")):
            p = ev["pythia"]; yt = ev["ytarget"]
            # gen: [apid,pt,eta,phi]
            if len(p):
                gm = (np.abs(p[:, 2]) > ETA_LO) & (np.abs(p[:, 2]) < ETA_HI)
                g_pid = p[gm, 0]; g_pt = p[gm, 1]; g_eta = p[gm, 2]; g_phi = p[gm, 3]
                g_cls = np.array([cls(x) for x in g_pid])
            else:
                g_pt = np.array([])
            # target
            vt = yt["pid"] != 0
            t_pid = yt["pid"][vt]; t_pt = yt["pt"][vt]; t_eta = yt["eta"][vt]
            t_phi = np.arctan2(yt["sin_phi"][vt], yt["cos_phi"][vt])
            tm = (np.abs(t_eta) > ETA_LO) & (np.abs(t_eta) < ETA_HI)
            t_pid = t_pid[tm]; t_pt = t_pt[tm]; t_eta = t_eta[tm]; t_phi = t_phi[tm]
            t_cls = np.array([cls(x) for x in t_pid])

            # ---- efficiency: each gen -> nearest target ----
            for i in range(len(g_pt)):
                c = g_cls[i]; bb = ptbin(g_pt[i]); eff[c][bb, 0] += 1
                if len(t_pt) == 0: continue
                dphi = np.arctan2(np.sin(t_phi-g_phi[i]), np.cos(t_phi-g_phi[i]))
                dR = np.hypot(t_eta-g_eta[i], dphi)
                if np.min(dR) < DR: eff[c][bb, 1] += 1
                same = t_cls == c
                if same.any() and np.min(dR[same]) < DR: eff[c][bb, 2] += 1

            # ---- fake rate: each target -> nearest gen ----
            for i in range(len(t_pt)):
                c = t_cls[i]; bb = ptbin(t_pt[i]); fake[c][bb, 0] += 1
                if len(g_pt) == 0:
                    fake[c][bb, 1] += 1; fake[c][bb, 2] += 1; continue
                dphi = np.arctan2(np.sin(g_phi-t_phi[i]), np.cos(g_phi-t_phi[i]))
                dR = np.hypot(g_eta-t_eta[i], dphi)
                if np.min(dR) >= DR: fake[c][bb, 1] += 1          # no gen match -> fake (any)
                same = g_cls == c
                if not (same.any() and np.min(dR[same]) < DR): fake[c][bb, 2] += 1  # no same-class gen -> fake
    return eff, fake

def merge(dst, src):
    for c in CLASSES: dst[c] += src[c]

def ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num/den, np.nan)
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample", default="ttbar_0pu")
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    SL = {"ttbar_0pu": r"$t\bar{t}$ 0 PU", "qcd_0pu": "QCD multijet 0 PU",
          "zll_0pu": r"$Z\to\ell\ell$ 0 PU"}.get(a.sample, a.sample)
    files = sorted(glob.glob(os.path.join(a.dir, "*.pkl")))
    EFF = {c: np.zeros((NB, 3)) for c in CLASSES}; FAKE = {c: np.zeros((NB, 3)) for c in CLASSES}
    nch = min(a.workers*3, len(files)) or 1
    chunks = [files[i::nch] for i in range(nch)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(collect, c) for c in chunks]):
            e, fk = fu.result(); merge(EFF, e); merge(FAKE, fk)

    plt.rcParams.update({"font.size": 12})
    def grid(metric, title, fn, num_any, num_same):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); axs = axs.ravel()
        for k, c in enumerate(CLASSES):
            ax = axs[k]; d = metric[c]; tot = d[:, 0]
            ax.plot(PT_CEN, ratio(num_any(d), tot), "o-", lw=2, label="any class")
            ax.plot(PT_CEN, ratio(num_same(d), tot), "s--", lw=2, label="same class")
            ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
            ax.set_title(CLABEL[c]); ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]")
            ax.grid(alpha=0.3); ax.legend(fontsize=10)
        axs[0].set_ylabel(title); axs[3].set_ylabel(title)
        axs[5].axis("off")
        fig.suptitle(title + f"  ({SL}, $1.5<|\\eta|<3.0$, $\\Delta R<0.2$)", fontsize=14)
        plt.tight_layout(); plt.savefig(os.path.join(a.outdir, fn)); plt.close(fig)

    grid(EFF, "Efficiency", "run3style_efficiency_vs_pt.pdf",
         lambda d: d[:, 1], lambda d: d[:, 2])
    grid(FAKE, "Fake rate", "run3style_fakerate_vs_pt.pdf",
         lambda d: d[:, 1], lambda d: d[:, 2])
    for c in CLASSES:
        tote = EFF[c][:, 0].sum(); totf = FAKE[c][:, 0].sum()
        print(f"{c:9s}: gen={int(tote):7d} eff_any={EFF[c][:,1].sum()/max(tote,1):.2f} "
              f"eff_same={EFF[c][:,2].sum()/max(tote,1):.2f} | "
              f"target={int(totf):7d} fake_any={FAKE[c][:,1].sum()/max(totf,1):.2f} "
              f"fake_same={FAKE[c][:,2].sum()/max(totf,1):.2f}")

if __name__ == "__main__":
    main()
