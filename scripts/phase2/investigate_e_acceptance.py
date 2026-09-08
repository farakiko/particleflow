#!/usr/bin/env python
"""(a) Size the trackless-electron acceptance loss: are the ~26% of electrons without
a GeneralTrack confined to the soft / forward tail (tolerable) or do they include
hard, central electrons (a real problem)?  Split truth electrons by GeneralTrack
availability and compare pT / |eta|, and how much *energy* the trackless set carries."""
import glob, argparse
import numpy as np, awkward as ak, uproot

SENTINEL = -2147483648
NANO = "/Users/fmokhtar/projects/particleflow/data/cms/phase2/offline/Aug31/nano"
BR = ["SimTICLCandidates_pdgID", "SimTICLCandidates_trackIdx", "SimTICLCandidates_pt",
      "SimTICLCandidates_eta", "SimTICLCandidates_energy", "GeneralTrack_pt"]


def dist(tag, pt, eta, en):
    qs = np.percentile(pt, [25, 50, 75, 90]) if len(pt) else [np.nan]*4
    print(f"  {tag:10s} N={len(pt):>6}  pT[GeV] q25/50/75/90 = {qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.2f}"
          f"   |eta| med={np.median(np.abs(eta)) if len(eta) else np.nan:.2f}   ΣE={en.sum():.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="ttbar_0pu")
    ap.add_argument("--max-files", type=int, default=30)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{NANO}/{a.sample}/*.root"))[:a.max_files] if a.max_files > 0 \
        else sorted(glob.glob(f"{NANO}/{a.sample}/*.root"))

    T = {"pt": [], "eta": [], "en": []}   # tracked electrons
    U = {"pt": [], "eta": [], "en": []}   # trackless (untracked) electrons
    for fn in files:
        E = uproot.open(fn)["Events"].arrays(BR)
        for iev in range(len(E)):
            g = lambda b: ak.to_numpy(E[b][iev])
            pid = g("SimTICLCandidates_pdgID"); trk = g("SimTICLCandidates_trackIdx")
            pt = g("SimTICLCandidates_pt"); eta = g("SimTICLCandidates_eta"); en = g("SimTICLCandidates_energy")
            n_gt = len(g("GeneralTrack_pt"))
            for i in range(len(pid)):
                if abs(int(pid[i])) != 11:
                    continue
                d = T if (int(trk[i]) != SENTINEL and 0 <= int(trk[i]) < n_gt) else U
                d["pt"].append(pt[i]); d["eta"].append(eta[i]); d["en"].append(en[i])

    for k in T: T[k] = np.array(T[k]); U[k] = np.array(U[k])
    ntot = len(T["pt"]) + len(U["pt"]); etot = T["en"].sum() + U["en"].sum()
    print(f"sample={a.sample}  files={len(files)}\n")
    print(f"electrons: {ntot} total; trackless = {len(U['pt'])} ({100*len(U['pt'])/max(ntot,1):.1f}% by count, "
          f"{100*U['en'].sum()/max(etot,1e-9):.1f}% by energy)\n")
    dist("tracked", T["pt"], T["eta"], T["en"])
    dist("trackless", U["pt"], U["eta"], U["en"])

    print("\ntrackless electrons — cumulative fraction below a pT cut:")
    for c in [1, 2, 5, 10, 20]:
        print(f"    pT < {c:>2} GeV : {100*np.mean(U['pt'] < c):>5.1f}%")
    print("\ntrackless electrons — by |eta| band:")
    for lo, hi in [(1.5, 2.0), (2.0, 2.5), (2.5, 2.7), (2.7, 3.2)]:
        m = (np.abs(U["eta"]) >= lo) & (np.abs(U["eta"]) < hi)
        mt = (np.abs(T["eta"]) >= lo) & (np.abs(T["eta"]) < hi)
        rate = 100*m.sum()/max(m.sum()+mt.sum(), 1)   # trackless rate within the band
        print(f"    {lo:.1f}-{hi:.1f}: {m.sum():>5} trackless  ({rate:.0f}% of electrons in band are trackless)")
    print("\n  hard, central trackless electrons (pT>10 & |eta|<2.5):",
          int(np.sum((U["pt"] > 10) & (np.abs(U["eta"]) < 2.5))),
          f"({100*np.mean((U['pt']>10)&(np.abs(U['eta'])<2.5)):.1f}% of trackless)")


if __name__ == "__main__":
    main()
