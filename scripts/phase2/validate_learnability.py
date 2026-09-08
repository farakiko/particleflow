#!/usr/bin/env python
"""Post-fix validation: confirm the trackster INPUT energy is now populated and the
per-class regression target log(target_E / element_E) is O(1) (learnable).

The fix (raw_energy instead of the all-zero regressed_energy) changes only the INPUT
element energy, not the target particles -- so target-level jet/eff/sumpt plots are
unchanged. This script validates the part that DID change."""
import os, sys, glob, pickle, argparse
import numpy as np

NANO = "/Users/fmokhtar/projects/particleflow/data/cms/phase2/offline/Aug31/nano"
NAME = {211: "chad", 130: "nhad", 22: "photon", 11: "electron", 13: "muon"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=["ttbar_0pu", "qcd_0pu", "zll_0pu"])
    ap.add_argument("--max-files", type=int, default=20)
    a = ap.parse_args()

    agg = {k: {"n": 0, "t1": 0, "t4": 0, "r": []} for k in NAME}
    ts_e = []          # trackster input energies (should be > 0 now)
    ts_zero = 0; ts_tot = 0
    nfiles = 0
    for s in a.samples:
        fs = sorted(glob.glob(f"{NANO}/{s}/pkl_run3style/run3style_*.pkl"))
        if a.max_files > 0:
            fs = fs[:a.max_files]
        for f in fs:
            nfiles += 1
            for ev in pickle.load(open(f, "rb")):
                Xe, yt = ev["Xelem"], ev["ytarget"]
                typ, en = Xe["typ"], Xe["energy"]
                m = typ == 4
                ts_e.append(en[m]); ts_zero += int(np.sum(en[m] <= 0)); ts_tot += int(np.sum(m))
                pid, te = yt["pid"], yt["energy"]
                for i in range(len(pid)):
                    p = abs(int(pid[i]))
                    if p == 321:
                        p = 211
                    if p not in NAME or te[i] <= 0:
                        continue
                    A = agg[p]; A["n"] += 1
                    A["t1"] += int(typ[i] == 1); A["t4"] += int(typ[i] == 4)
                    if en[i] > 0:
                        A["r"].append(te[i] / en[i])

    ts_e = np.concatenate(ts_e) if ts_e else np.array([0.0])
    print(f"files read: {nfiles}")
    print(f"\ntrackster INPUT energy: median={np.median(ts_e):.3f} GeV, "
          f"zero-fraction={100*ts_zero/max(ts_tot,1):.2f}%  (was 100% before fix)")
    print(f"\n{'class':10s}{'N':>9} {'%typ1':>6} {'%typ4':>6} | "
          f"{'tE/eE med':>10} {'log(med)':>9} {'%>2':>5} {'%>5':>5}")
    for k, nm in NAME.items():
        A = agg[k]; n = max(A["n"], 1)
        r = np.array(A["r"]) if A["r"] else np.array([np.nan])
        md = np.median(r)
        print(f"{nm:10s}{A['n']:>9} {100*A['t1']/n:>5.0f}% {100*A['t4']/n:>5.0f}% | "
              f"{md:>10.2f} {np.log(md):>9.2f} {100*np.mean(r>2):>4.0f}% {100*np.mean(r>5):>4.0f}%")


if __name__ == "__main__":
    main()
