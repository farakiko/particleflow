#!/usr/bin/env python
"""Per-event sum-pT per particle class: target vs gen (Run3-style target).
Region 1.5<|eta|<3.0.  One panel per class."""
import glob, pickle, argparse, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

ETA_LO, ETA_HI = 1.5, 3.0
CLASSES = ["chad", "photon", "nhad", "electron", "muon"]
CLABEL = {"photon": "photons", "nhad": "neutral hadrons", "chad": "charged hadrons",
          "electron": "electrons", "muon": "muons"}
NHAD = {130, 310, 2112, 3122, 3322, 3212, 421, 511}

def cls(apid):
    a = int(abs(apid))
    if a == 22: return "photon"
    if a == 11: return "electron"
    if a == 13: return "muon"
    if a in NHAD: return "nhad"
    return "chad"

def collect(files):
    out = {c: {"gen": [], "tgt": []} for c in CLASSES}
    for f in files:
        for ev in pickle.load(open(f, "rb")):
            p = ev["pythia"]; yt = ev["ytarget"]
            gsum = {c: 0.0 for c in CLASSES}; tsum = {c: 0.0 for c in CLASSES}
            if len(p):
                m = (np.abs(p[:, 2]) > ETA_LO) & (np.abs(p[:, 2]) < ETA_HI)
                for pid, pt in zip(p[m, 0], p[m, 1]): gsum[cls(pid)] += pt
            vt = yt["pid"] != 0
            te = yt["eta"][vt]; tp = yt["pt"][vt]; tpid = yt["pid"][vt]
            mm = (np.abs(te) > ETA_LO) & (np.abs(te) < ETA_HI)
            for pid, pt in zip(tpid[mm], tp[mm]): tsum[cls(pid)] += pt
            for c in CLASSES:
                out[c]["gen"].append(gsum[c]); out[c]["tgt"].append(tsum[c])
    return {c: {"gen": np.array(v["gen"]), "tgt": np.array(v["tgt"])} for c, v in out.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample", default="ttbar_0pu")
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    SL = {"ttbar_0pu": r"$t\bar{t}$ 0 PU", "qcd_0pu": "QCD multijet 0 PU", "zll_0pu": r"$Z\to\ell\ell$ 0 PU"}.get(a.sample, a.sample)
    files = sorted(glob.glob(os.path.join(a.dir, "*.pkl")))
    agg = {c: {"gen": [], "tgt": []} for c in CLASSES}
    nch = min(a.workers*3, len(files)) or 1
    chunks = [files[i::nch] for i in range(nch)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(collect, c) for c in chunks]):
            r = fu.result()
            for c in CLASSES:
                agg[c]["gen"].append(r[c]["gen"]); agg[c]["tgt"].append(r[c]["tgt"])
    D = {c: {"gen": np.concatenate(agg[c]["gen"]), "tgt": np.concatenate(agg[c]["tgt"])} for c in CLASSES}

    plt.rcParams.update({"font.size": 12})
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); axs = axs.ravel()
    for k, c in enumerate(CLASSES):
        ax = axs[k]; gen = D[c]["gen"]; tgt = D[c]["tgt"]
        hi = np.percentile(np.concatenate([gen, tgt]), 99.5); hi = max(hi, 1.0)
        b = np.linspace(0, hi, 60)
        ax.hist(gen, bins=b, histtype="step", lw=2, color="black",
                label=f"gen (⟨ΣpT⟩={gen.mean():.1f})")
        ax.hist(tgt, bins=b, histtype="step", lw=2, color="tab:orange",
                label=f"target (⟨ΣpT⟩={tgt.mean():.1f})")
        ax.set_yscale("log"); ax.set_title(CLABEL[c])
        ax.set_xlabel(r"event $\Sigma p_\mathrm{T}$ [GeV]"); ax.set_ylabel("Events")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.text(0.97, 0.80, f"target/gen = {tgt.sum()/max(gen.sum(),1e-9):.2f}",
                transform=ax.transAxes, ha="right", fontsize=10)
    axs[5].axis("off")
    fig.suptitle(r"Per-event $\Sigma p_\mathrm{T}$ per class: target vs gen  ("+SL+r", $1.5<|\eta|<3.0$)", fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "run3style_sumpt_per_class.pdf")); plt.close(fig)
    for c in CLASSES:
        print(f"{c:9s}: target/gen total ΣpT = {D[c]['tgt'].sum()/max(D[c]['gen'].sum(),1e-9):.3f}  "
              f"(gen ⟨ΣpT⟩={D[c]['gen'].mean():.1f}, target ⟨ΣpT⟩={D[c]['tgt'].mean():.1f})")

if __name__ == "__main__":
    main()
