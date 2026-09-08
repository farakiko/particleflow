#!/usr/bin/env python
"""Target (Run3-style) vs gen jet validation. Reads pkl_run3style/*.pkl.
Fig1: target vs gen jet pT spectra (|eta|>1.5).  Fig2: target/gen jet response."""
import glob, pickle, argparse, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

DR, GENPT, ETAMIN = 0.2, 20.0, 1.5

def collect(files):
    tj, gj, resp = [], [], []
    for f in files:
        for ev in pickle.load(open(f, "rb")):
            t, g = ev["targetjet"], ev["genjet"]
            if len(t): tj.append(t[:, :2])
            if len(g): gj.append(g[:, :2])
            if len(t) and len(g):
                for gg in g:
                    if gg[0] < GENPT or abs(gg[1]) < ETAMIN: continue
                    dphi = np.arctan2(np.sin(t[:, 2]-gg[2]), np.cos(t[:, 2]-gg[2]))
                    dR = np.hypot(t[:, 1]-gg[1], dphi); b = int(np.argmin(dR))
                    if dR[b] < DR: resp.append(t[b, 0]/gg[0])
    tj = np.concatenate(tj) if tj else np.zeros((0, 2))
    gj = np.concatenate(gj) if gj else np.zeros((0, 2))
    return tj, gj, np.array(resp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample", default="ttbar_0pu")
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    SL = {"ttbar_0pu": r"$t\bar{t}$, 0 PU", "qcd_0pu": "QCD multijet, 0 PU",
          "zll_0pu": r"$Z\to\ell\ell$, 0 PU"}.get(a.sample, a.sample)
    files = sorted(glob.glob(os.path.join(a.dir, "*.pkl")))
    print(f"{len(files)} pkls")
    nch = min(a.workers*3, len(files)) or 1
    chunks = [files[i::nch] for i in range(nch)]
    TJ, GJ, R = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(collect, c) for c in chunks]):
            t, g, r = fu.result()
            if len(t): TJ.append(t)
            if len(g): GJ.append(g)
            if len(r): R.append(r)
    tj, gj, resp = np.concatenate(TJ), np.concatenate(GJ), np.concatenate(R)
    plt.rcParams.update({"font.size": 14})

    # Fig 1: pT spectra
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    b = np.logspace(np.log10(3), np.log10(800), 50)
    for arr, lab, c in [(gj, "gen (status 1, no ν)", "black"), (tj, "MLPF target", "tab:orange")]:
        m = np.abs(arr[:, 1]) > ETAMIN
        ax.hist(arr[m, 0], bins=b, histtype="step", lw=2, color=c, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_\mathrm{T}$ [GeV]"); ax.set_ylabel("Jets")
    ax.set_title(r"Jet $p_\mathrm{T}$: target vs gen  ($|\eta|>1.5$)")
    ax.text(0.97, 0.93, SL + " (Run3-style target)", transform=ax.transAxes,
            ha="right", va="top", fontsize=13)
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, "run3style_targetjet_vs_gen_pt.pdf")); plt.close(fig)

    # Fig 2: response
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    bb = np.logspace(-1, 1, 150)
    med = np.median(resp); iqr = (np.percentile(resp, 75)-np.percentile(resp, 25))/med
    ax.hist(resp, bins=bb, histtype="step", lw=2, color="tab:orange",
            label=f"MLPF target (med {med:.2f}, IQR/med {iqr:.2f})")
    ax.axvline(1, color="k", ls="--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_\mathrm{T}$ / genjet $p_\mathrm{T}$"); ax.set_ylabel("Jets")
    ax.set_title(r"Target jet response vs gen (gen $p_\mathrm{T}>20$, $|\eta|>1.5$)")
    ax.text(0.97, 0.93, SL, transform=ax.transAxes, ha="right", va="top", fontsize=13)
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, "run3style_targetjet_response.pdf")); plt.close(fig)

    print(f"target jets(|eta|>1.5)={int((np.abs(tj[:,1])>ETAMIN).sum())}  "
          f"gen jets(|eta|>1.5)={int((np.abs(gj[:,1])>ETAMIN).sum())}")
    print(f"matched response: N={len(resp)}  median={med:.3f}  IQR/med={iqr:.3f}")

if __name__ == "__main__":
    main()
