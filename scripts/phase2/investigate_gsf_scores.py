#!/usr/bin/env python
"""Two follow-ups, straight from NanoAOD:

A) GSF rescue for electrons. SimTICLCandidates has BOTH a GeneralTrack link
   (trackIdx, what postprocessing uses) and a GSFTrack link (gsftrackIdx, unused).
   Do the electrons that lack a GeneralTrack have a GSF track? -> would adding
   GSFTrack as an input element recover the ~23% dropped e-.

B) Associator scores for the neutral-hadron fragmentation problem. Two scores exist
   (0 = perfect match, 1 = no match):
     s_score = SimCP2ticlTrackstersCLUE3DHighByHitsLinks_score   (sim -> trackster)
     r_score = RecoticlTrackstersCLUE3DHigh2SimCPByHitsLinks_score (trackster -> sim)
   B1: for each neutral hadron, how much energy sits in the *fragment* tracksters
       (not the argmax anchor), and are those fragments real (good s_score)?
       -> feasibility of option (1): one neutral-had target per energetic trackster.
   B2: does r_score separate tracksters we anchor as target particles from the
       ones left null (fakes/PU/noise)? -> value of feeding score as an input (option 2).
"""
import glob, argparse
import numpy as np, awkward as ak, uproot

SENTINEL = -2147483648
TS  = "ticlTrackstersCLUE3DHigh"
S2R = "SimCP2ticlTrackstersCLUE3DHighByHits"
R2S = "RecoticlTrackstersCLUE3DHigh2SimCPByHits"
NANO = "/Users/fmokhtar/projects/particleflow/data/cms/phase2/offline/Aug31/nano"

BRANCHES = [
    "SimTICLCandidates_pdgID", "SimTICLCandidates_energy", "SimTICLCandidates_pt",
    "SimTICLCandidates_trackIdx", "SimTICLCandidates_gsftrackIdx",
    "GeneralTrack_pt", "GSFTrack_pt", f"{TS}_raw_energy",
    f"{S2R}_n{S2R}Links", f"{S2R}Links_index", f"{S2R}Links_score", f"{S2R}Links_sharedEnergy",
    f"{R2S}_n{R2S}Links", f"{R2S}Links_index", f"{R2S}Links_score", f"{R2S}Links_sharedEnergy",
]


def pct(x):
    return f"{100*x:>6.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="ttbar_0pu")
    ap.add_argument("--max-files", type=int, default=30)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{NANO}/{a.sample}/*.root"))[:a.max_files] if a.max_files > 0 \
        else sorted(glob.glob(f"{NANO}/{a.sample}/*.root"))

    LEP = {11: "electron", 13: "muon"}
    qA = {k: np.zeros(4, int) for k in LEP}     # [both, general-only, gsf-only, neither]
    # B1 fragment analysis (neutral hadrons)
    frag_n = []        # # fragment tracksters (shared>0, not anchor) per nhad
    frag_efrac = []    # fragment shared energy / total shared energy, per nhad
    big_frag = []      # # fragments per nhad with rawE>1GeV AND s_score<0.6 (own-target candidates)
    anchor_sscore = []; frag_sscore = []
    # B2 r_score by trackster role
    rrole = {"anchor": [], "fragment": [], "unused": []}

    nev = 0
    for fn in files:
        E = uproot.open(fn)["Events"].arrays(BRANCHES)
        for iev in range(len(E)):
            nev += 1
            g = lambda b: ak.to_numpy(E[b][iev])
            pid = g("SimTICLCandidates_pdgID"); en = g("SimTICLCandidates_energy")
            trk = g("SimTICLCandidates_trackIdx"); gsf = g("SimTICLCandidates_gsftrackIdx")
            n_gt = len(g("GeneralTrack_pt")); n_gsf = len(g("GSFTrack_pt"))
            tsraw = g(f"{TS}_raw_energy"); n_ts = len(tsraw)

            # ---- A: electron/muon track availability ----
            for i in range(len(pid)):
                ap_ = abs(int(pid[i]))
                if ap_ in LEP:
                    hg = (int(trk[i]) != SENTINEL) and (0 <= int(trk[i]) < n_gt)
                    hs = (int(gsf[i]) != SENTINEL) and (0 <= int(gsf[i]) < n_gsf)
                    qA[ap_][0 if (hg and hs) else 1 if hg else 2 if hs else 3] += 1

            # ---- sim->trackster links (s_score) ----
            cnt = g(f"{S2R}_n{S2R}Links"); off = np.concatenate([[0], np.cumsum(cnt)]).astype(int)
            s_idx = g(f"{S2R}Links_index"); s_scr = g(f"{S2R}Links_score"); s_she = g(f"{S2R}Links_sharedEnergy")
            # ---- trackster->sim links (r_score): best (min) score per trackster ----
            rcnt = g(f"{R2S}_n{R2S}Links"); roff = np.concatenate([[0], np.cumsum(rcnt)]).astype(int)
            r_scr = g(f"{R2S}Links_score")
            best_r = np.full(n_ts, np.nan)
            for t in range(n_ts):
                seg = r_scr[roff[t]:roff[t+1]]
                if len(seg): best_r[t] = seg.min()

            role = np.array(["unused"] * n_ts, dtype=object)
            for i in range(len(pid)):
                ap_ = abs(int(pid[i]))
                is_neutral = ap_ in (22, 130)
                ii = s_idx[off[i]:off[i+1]]; ss = s_she[off[i]:off[i+1]]; sc = s_scr[off[i]:off[i+1]]
                pos = ss > 0
                if not pos.any():
                    continue
                iiv, ssv, scv = ii[pos], ss[pos], sc[pos]
                order = np.argsort(ssv)[::-1]          # by shared energy, descending
                anchor_t = int(iiv[order[0]])
                if is_neutral and 0 <= anchor_t < n_ts:
                    role[anchor_t] = "anchor"
                    for j in order[1:]:                # fragments
                        tt = int(iiv[j])
                        if 0 <= tt < n_ts and role[tt] == "unused":
                            role[tt] = "fragment"
                # B1 stats for neutral hadrons only
                if ap_ == 130 and en[i] > 0 and 0 <= anchor_t < n_ts:
                    tot = ssv.sum()
                    anchor_sscore.append(float(scv[order[0]]))
                    frags = order[1:]                  # indices (into iiv/ssv/scv) of non-anchor links
                    frag_n.append(len(frags))
                    frag_efrac.append(float(ssv[frags].sum()/tot) if tot > 0 else 0.0)
                    nbig = 0
                    for j in frags:
                        tt = int(iiv[j])
                        frag_sscore.append(float(scv[j]))
                        if 0 <= tt < n_ts and tsraw[tt] > 1.0 and scv[j] < 0.6:
                            nbig += 1
                    big_frag.append(nbig)
            for t in range(n_ts):
                if not np.isnan(best_r[t]):
                    rrole[role[t]].append(best_r[t])

    print(f"sample={a.sample}  files={len(files)}  events={nev}\n")

    # ---- A ----
    print("A  electron/muon track availability (GeneralTrack vs GSFTrack):")
    print(f"{'class':10s}{'N':>8}{'both':>9}{'general-only':>14}{'GSF-only':>10}{'neither':>9}")
    for k, nm in LEP.items():
        c = qA[k]; N = max(c.sum(), 1)
        print(f"{nm:10s}{c.sum():>8}{pct(c[0]/N):>9}{pct(c[1]/N):>14}{pct(c[2]/N):>10}{pct(c[3]/N):>9}")
    for k, nm in LEP.items():
        c = qA[k]
        no_gt = c[2] + c[3]                       # electrons without a GeneralTrack
        rescue = c[2] / max(no_gt, 1)
        print(f"    {nm}: of the {no_gt} without a GeneralTrack, {pct(rescue)} have a GSF track")
    print()

    # ---- B1 ----
    print("B1  neutral-hadron fragment tracksters (beyond the argmax anchor):")
    print(f"    N nhad analyzed        : {len(frag_n)}")
    print(f"    median # fragments/nhad: {np.median(frag_n):.1f}   (mean {np.mean(frag_n):.2f})")
    print(f"    frag shared-E / total  : median {np.median(frag_efrac):.2f}  mean {np.mean(frag_efrac):.2f}")
    print(f"    anchor link s_score    : median {np.median(anchor_sscore):.2f}")
    print(f"    fragment link s_score  : median {np.median(frag_sscore) if frag_sscore else float('nan'):.2f}")
    print(f"    fragments with rawE>1GeV & s_score<0.6 (own-target candidates):")
    bf = np.array(big_frag)
    print(f"        mean {bf.mean():.2f}/nhad ; nhad with >=1 such fragment: {pct(np.mean(bf>=1))}")
    print()

    # ---- B2 ----
    print("B2  trackster r_score (reco->sim, 0=pure real ... 1=fake) by role in our target:")
    print(f"{'role':10s}{'N':>9}{'median r':>10}{'frac r<0.4':>12}{'frac r>0.8':>12}")
    for rk in ("anchor", "fragment", "unused"):
        v = np.array(rrole[rk])
        if not len(v):
            continue
        print(f"{rk:10s}{len(v):>9}{np.median(v):>10.2f}{pct(np.mean(v<0.4)):>12}{pct(np.mean(v>0.8)):>12}")


if __name__ == "__main__":
    main()
