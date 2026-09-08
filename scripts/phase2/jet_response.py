#!/usr/bin/env python
"""Target-jet pT response w.r.t truth (gen) jets: match target<->gen jets by dR,
take pt_target/pt_gen. Produces response histogram + response/resolution vs pT."""
import os, glob, pickle, argparse, re
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

DR_MATCH = 0.2
def sample_of(f): return re.sub(r"_[0-9]+_[0-9]+\.pkl$", "", os.path.basename(f))

def collect(files):
    out = defaultdict(list)   # sample -> list of (gen_pt, gen_eta, response)
    for f in files:
        try: data = pickle.load(open(f, "rb"))
        except Exception: continue
        s = sample_of(f)
        for ev in data:
            gj = np.atleast_2d(ev["genjet"]);    tj = np.atleast_2d(ev["targetjet"])
            if gj.size == 0 or tj.size == 0: continue
            gj = gj.reshape(-1, 4); tj = tj.reshape(-1, 4)
            if gj.shape[1] != 4 or tj.shape[1] != 4: continue
            for g in gj:
                if g[0] < 10: continue
                dphi = np.arctan2(np.sin(tj[:, 2] - g[2]), np.cos(tj[:, 2] - g[2]))
                dR = np.hypot(tj[:, 1] - g[1], dphi)
                j = int(np.argmin(dR))
                if dR[j] < DR_MATCH:
                    out[s].append((g[0], g[1], tj[j, 0] / g[0]))
    return {k: np.array(v) for k, v in out.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--per-sample", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    by = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.dir, "*.pkl"))): by[sample_of(f)].append(f)
    files = [f for fs in by.values() for f in fs[: a.per_sample]]
    print(f"{len(files)} files, {a.workers} workers")

    nch = min(a.workers * 4, len(files)) or 1
    chunks = [files[i::nch] for i in range(nch)]
    res = defaultdict(list)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(collect, c) for c in chunks]):
            for k, v in fu.result().items():
                if len(v): res[k].append(v)
    res = {k: np.concatenate(v) for k, v in res.items()}

    order = [s for s in ["ticl_qcd_nopu","ticl_ttbar_nopu","ticl_zll_nopu"] if s in res]
    titles = {"ticl_qcd_nopu":"QCD 0PU","ticl_ttbar_nopu":"ttbar 0PU","ticl_zll_nopu":"Z→ll 0PU"}
    colors = {"ticl_qcd_nopu":"tab:blue","ticl_ttbar_nopu":"tab:red","ticl_zll_nopu":"tab:green"}
    plt.rcParams.update({"font.size":14,"axes.titlesize":16,"axes.labelsize":15,
                         "xtick.labelsize":12.5,"ytick.labelsize":12.5,
                         "legend.fontsize":12.5,"legend.title_fontsize":13})

    # ---- Fig 1: response histogram (gen pt>30, |eta|>1.5) ----
    fig, ax = plt.subplots(figsize=(7.2,5.4))
    bins = np.logspace(np.log10(0.05), np.log10(3.0), 61)
    for s in order:
        d = res[s]; m = (d[:,0] > 20) & (np.abs(d[:,1]) > 1.5)
        if m.sum() < 20: continue
        r = d[m,2]
        ax.hist(r, bins=bins, histtype="step", lw=2, density=False, color=colors[s],
                label=f"{titles[s]} (med {np.median(r):.2f}, IQR/med {(np.percentile(r,75)-np.percentile(r,25))/np.median(r):.2f})")
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Target jet pT / truth jet pT"); ax.set_ylabel("Jets")
    ax.set_title("Target jet response  (gen pT > 20 GeV, |η| > 1.5)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(a.outdir,"target_jet_response_hist.pdf")); plt.close(fig)

    # ---- Fig 2: median response & resolution vs gen pt ----
    edges = np.array([10,20,30,50,75,100,150,200,300]); cen = np.sqrt(edges[:-1]*edges[1:])
    fig, (a1,a2) = plt.subplots(1,2, figsize=(13,5.2))
    for s in order:
        d = res[s]; sel = np.abs(d[:,1]) > 1.5
        pt, r = d[sel,0], d[sel,2]
        med=[]; reso=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mm = (pt>=lo)&(pt<hi)
            if mm.sum()<20: med.append(np.nan); reso.append(np.nan); continue
            q1,q2,q3=np.percentile(r[mm],[25,50,75]); med.append(q2); reso.append((q3-q1)/q2)
        a1.plot(cen, med, marker="o", lw=1.8, color=colors[s], label=titles[s])
        a2.plot(cen, reso, marker="o", lw=1.8, color=colors[s], label=titles[s])
    a1.axhline(1.0, color="gray", ls="--", lw=1)
    for ax in (a1,a2):
        ax.set_xscale("log"); ax.set_xlabel("Truth jet pT [GeV]")
        ax.set_xticks([10,20,50,100,200]); ax.set_xticklabels([10,20,50,100,200]); ax.legend()
    a1.set_ylabel("Median target jet response"); a1.set_title("Target jet response vs truth pT (|η|>1.5)")
    a2.set_ylabel("Response resolution (IQR/median)"); a2.set_title("Target jet resolution vs truth pT (|η|>1.5)")
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir,"target_jet_response_vs_pt.pdf")); plt.close(fig)

    # summary
    for s in order:
        d=res[s]; m=(d[:,0]>20)&(np.abs(d[:,1])>1.5)
        print(f"{s}: matched(gen>20,|eta|>1.5)={m.sum()}, median_resp={np.median(d[m,2]):.3f}" if m.sum() else f"{s}: too few")
    print("saved: target_jet_response_hist.pdf, target_jet_response_vs_pt.pdf")

if __name__ == "__main__":
    main()
