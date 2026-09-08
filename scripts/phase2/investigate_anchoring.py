#!/usr/bin/env python
"""Physics soundness checks on the Run3-style anchoring, straight from NanoAOD.

Q1 (leptons): do all e/mu have a track, or do they fall back to a trackster / get
    dropped? Replicates postprocessing_run3style.py's exact anchoring logic:
      charged & trackIdx in range        -> TRACK
      charged & trackIdx == SENTINEL      -> tries TRACKSTER (neutral branch)
      charged & trackIdx out of range     -> DROPPED (no fallback in postproc)

Q2 (neutrals): for photon / neutral-hadron, how well does the anchored trackster's
    energy close on the truth energy? Decompose the overall closure
      trackster_rawE / truthE  =  (calo deposit / truth) x (lead trackster / calo deposit)
    i.e. calo response x trackster-capture (fragmentation), reported separately.
"""
import glob, argparse, os
import numpy as np, awkward as ak, uproot

SENTINEL = -2147483648
CHARGED_PIDS = {11, 13, 211, 321}
TS  = "ticlTrackstersCLUE3DHigh"
S2R = "SimCP2ticlTrackstersCLUE3DHighByHits"
NANO = "/Users/fmokhtar/projects/particleflow/data/cms/phase2/offline/Aug31/nano"

BRANCHES = [
    "SimTICLCandidates_pdgID", "SimTICLCandidates_energy", "SimTICLCandidates_raw_energy",
    "SimTICLCandidates_trackIdx", "SimTICLCandidates_isPU",
    "GeneralTrack_pt", f"{TS}_raw_energy",
    f"{S2R}_n{S2R}Links", f"{S2R}Links_index", f"{S2R}Links_sharedEnergy",
]


def frac_table(r, edges):
    """fraction of ratios r falling in each [edge_i, edge_{i+1}) bin."""
    r = np.asarray(r)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(100.0 * np.mean((r >= lo) & (r < hi)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="ttbar_0pu")
    ap.add_argument("--max-files", type=int, default=30)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{NANO}/{a.sample}/*.root"))
    if a.max_files > 0:
        files = files[:a.max_files]

    # Q1 accumulators: per class -> [track, trackster_via_sentinel, dropped_sentinel_no_ts, dropped_badidx]
    LEP = {11: "electron", 13: "muon", 211: "charged had"}
    q1 = {k: np.zeros(4, int) for k in LEP}
    # Q2 accumulators
    NEU = {22: "photon", 130: "neutral had"}
    clos= {k: [] for k in NEU}      # trackster_rawE / truthE   (overall closure)
    calo = {k: [] for k in NEU}     # raw_energy / truthE       (calo response)
    capt = {k: [] for k in NEU}     # lead trackster_rawE / raw_energy (fragmentation capture)
    nlink = {k: [] for k in NEU}    # # tracksters sharing energy with the particle
    sumsh = {k: [] for k in NEU}    # sum of shared energy over all tracksters / truthE

    nev = 0
    for fn in files:
        E = uproot.open(fn)["Events"].arrays(BRANCHES)
        for iev in range(len(E)):
            nev += 1
            g = lambda b: ak.to_numpy(E[b][iev])
            pid = g("SimTICLCandidates_pdgID"); en = g("SimTICLCandidates_energy")
            raw = g("SimTICLCandidates_raw_energy"); trk = g("SimTICLCandidates_trackIdx")
            n_trk = len(g("GeneralTrack_pt")); tsraw = g(f"{TS}_raw_energy"); n_ts = len(tsraw)
            cnt = g(f"{S2R}_n{S2R}Links"); off = np.concatenate([[0], np.cumsum(cnt)]).astype(int)
            aidx = g(f"{S2R}Links_index"); ashe = g(f"{S2R}Links_sharedEnergy")

            for i in range(len(pid)):
                ap_ = abs(int(pid[i]))
                ti = int(trk[i])
                # ---- Q1: charged (leptons + charged had) ----
                if ap_ in LEP:
                    if ti != SENTINEL and 0 <= ti < n_trk:
                        q1[ap_][0] += 1                          # TRACK
                    elif ti == SENTINEL:
                        ii = aidx[off[i]:off[i+1]]; ss = ashe[off[i]:off[i+1]]
                        has_ts = len(ss) and ss.max() > 0 and (0 <= int(ii[int(np.argmax(ss))]) < n_ts)
                        q1[ap_][1 if has_ts else 2] += 1         # TRACKSTER-via-sentinel / DROPPED
                    else:
                        q1[ap_][3] += 1                          # DROPPED bad idx
                # ---- Q2: neutrals (photon, neutral had) ----
                if ap_ in NEU and en[i] > 0:
                    ii = aidx[off[i]:off[i+1]]; ss = ashe[off[i]:off[i+1]]
                    if len(ss) and ss.max() > 0:
                        best = int(ii[int(np.argmax(ss))])
                        if 0 <= best < n_ts:
                            closs = tsraw[best] / en[i]
                            clos[ap_].append(closs)
                            calo[ap_].append(raw[i] / en[i] if raw[i] >= 0 else np.nan)
                            capt[ap_].append(tsraw[best] / raw[i] if raw[i] > 0 else np.nan)
                            nlink[ap_].append(int(np.sum(ss > 0)))
                            sumsh[ap_].append(ss[ss > 0].sum() / en[i])

    print(f"sample={a.sample}  files={len(files)}  events={nev}\n")

    # ---- Q1 ----
    print("Q1  lepton/charged anchoring (how each truth particle is assigned):")
    print(f"{'class':13s}{'N':>8}{'track':>9}{'trackster':>11}{'drop(noTS)':>12}{'drop(badix)':>13}")
    for k, nm in LEP.items():
        c = q1[k]; N = max(c.sum(), 1)
        print(f"{nm:13s}{c.sum():>8}" + "".join(f"{100*x/N:>10.1f}%" for x in c))
    print()

    # ---- Q2 ----
    EDG = [0, 0.25, 0.5, 0.8, 1.2, 1e9]
    HDR = "  <25%  25-50 50-80 80-120  >120"
    print("Q2  neutral energy closure (anchored trackster rawE / truth E):")
    print(f"{'class':13s}{'N':>7}{'median':>8}{'  |  frac in bins:':>0}{HDR}")
    for k, nm in NEU.items():
        r = np.array(clos[k])
        if not len(r):
            continue
        fr = frac_table(r, EDG)
        print(f"{nm:13s}{len(r):>7}{np.median(r):>8.2f}   " + " ".join(f"{x:>5.0f}" for x in fr))
    print("\n    decomposition of the median closure  =  calo response x trackster capture:")
    print(f"{'class':13s}{'closure':>9}{'caloResp':>10}{'tsCapture':>11}{'  medians':>0}"
          f"{'   <n_ts share>':>16}{'  sumShared/E':>14}")
    for k, nm in NEU.items():
        if not len(clos[k]):
            continue
        print(f"{nm:13s}{np.median(clos[k]):>9.2f}{np.nanmedian(calo[k]):>10.2f}"
              f"{np.nanmedian(capt[k]):>11.2f}{np.median(nlink[k]):>16.1f}{np.median(sumsh[k]):>14.2f}")


if __name__ == "__main__":
    main()
