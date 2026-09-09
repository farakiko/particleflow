#!/usr/bin/env python
"""Definitive target comparison (presentation-ready): OUR target (run3style, --calo links:
one-per-particle, no cuts, full truth energy) vs the COLLEAGUE's target (mohamed: fragment +
associator score cuts + gen-match filter, also on links). Same collection -> isolates the
target definition. Pure target-vs-gen (no training).

Per sample, overlays ours (green) vs colleague (red) vs gen, writing to
plots/phase2/final_target_<sample>/:
  jet_response, particle_resolution_{same,any}pid, efficiency_{same,any}pid,
  fakerate_{same,any}pid, sumpt_per_class.

Reads the stored pkls:  <sample>/pkl_links (ours)  and  <sample>/pkl_mohamed (colleague).
"""
import os, glob, pickle, argparse
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NANO = "/Users/fmokhtar/projects/particleflow/data/cms/phase2/offline/Aug31/nano"
NHAD = {130, 310, 2112, 3122, 3322, 3212, 421, 511}
NEUTR = {12, 14, 16}
CLASSES = ["photon", "nhad", "chad", "electron", "muon"]
CLAB = {"photon": "photons", "nhad": "neutral had", "chad": "charged had", "electron": "electrons", "muon": "muons"}
ELO, EHI, DR, GPT = 1.5, 3.0, 0.2, 20.0
PTED = np.array([1, 2, 3, 5, 8, 13, 20, 35, 60, 100, 200.]); PTC = np.sqrt(PTED[:-1]*PTED[1:]); NB = len(PTC)
SL = {"ttbar_0pu": r"$t\bar{t}$, 0 PU", "qcd_0pu": "QCD, 0 PU", "zll_0pu": r"$Z\to\ell\ell$, 0 PU"}


def pdgcls(p):
    a = int(abs(p))
    return "photon" if a == 22 else "electron" if a == 11 else "muon" if a == 13 else ("nhad" if a in NHAD else "chad")


def jet_response(tj, gj, out):
    if not len(tj) or not len(gj): return
    for g in gj:
        if g[0] < GPT or abs(g[1]) < ELO or abs(g[1]) > EHI: continue
        d = np.arctan2(np.sin(tj[:, 2]-g[2]), np.cos(tj[:, 2]-g[2])); b = int(np.argmin(np.hypot(tj[:, 1]-g[1], d)))
        if np.hypot(tj[b, 1]-g[1], d[b]) < DR: out.append(tj[b, 0]/g[0])


def compute(files):
    m = dict(resp=[], sump={c: 0.0 for c in CLASSES}, gsump={c: 0.0 for c in CLASSES},
             eff={c: np.zeros((NB, 2)) for c in CLASSES}, effA={c: np.zeros((NB, 2)) for c in CLASSES},
             fake={c: np.zeros((NB, 2)) for c in CLASSES}, fakeA={c: np.zeros((NB, 2)) for c in CLASSES},
             rsame={c: [] for c in CLASSES}, rany={c: [] for c in CLASSES})
    for f in files:
        for ev in pickle.load(open(f, "rb")):
            tj = np.asarray(ev["targetjet"]).reshape(-1, 4); gj = np.asarray(ev["genjet"]).reshape(-1, 4)
            jet_response(tj, gj, m["resp"])
            yp = np.asarray(ev["pythia"]).reshape(-1, 5)
            gmk = (np.abs(yp[:, 2]) > ELO) & (np.abs(yp[:, 2]) < EHI) & ~np.isin(np.abs(yp[:, 0]), list(NEUTR))
            gpt, geta, gphi = yp[gmk, 1], yp[gmk, 2], yp[gmk, 3]; gcls = np.array([pdgcls(p) for p in yp[gmk, 0]])
            yt = ev["ytarget"]; v = yt["pid"] != 0
            tpt = yt["pt"][v]; teta = yt["eta"][v]; tphi = np.arctan2(yt["sin_phi"][v], yt["cos_phi"][v])
            tcls = np.array([pdgcls(p) for p in yt["pid"][v]])
            tm = (np.abs(teta) > ELO) & (np.abs(teta) < EHI); tpt, teta, tphi, tcls = tpt[tm], teta[tm], tphi[tm], tcls[tm]
            for c in CLASSES:
                m["sump"][c] += tpt[tcls == c].sum(); m["gsump"][c] += gpt[gcls == c].sum()
            for gi in range(len(gpt)):
                cc = gcls[gi]; bb = int(np.clip(np.digitize(gpt[gi], PTED)-1, 0, NB-1)); m["eff"][cc][bb, 0] += 1; m["effA"][cc][bb, 0] += 1
                if len(tpt):
                    d = np.arctan2(np.sin(tphi-gphi[gi]), np.cos(tphi-gphi[gi])); dr = np.hypot(teta-geta[gi], d)
                    if dr.min() < DR: m["effA"][cc][bb, 1] += 1
                    sm = tcls == cc
                    if sm.any() and np.hypot(teta[sm]-geta[gi], np.arctan2(np.sin(tphi[sm]-gphi[gi]), np.cos(tphi[sm]-gphi[gi]))).min() < DR: m["eff"][cc][bb, 1] += 1
            for ri in range(len(tpt)):
                cc = tcls[ri]; bb = int(np.clip(np.digitize(tpt[ri], PTED)-1, 0, NB-1)); m["fake"][cc][bb, 0] += 1; m["fakeA"][cc][bb, 0] += 1
                anyok = sameok = False
                if len(gpt):
                    d = np.arctan2(np.sin(gphi-tphi[ri]), np.cos(gphi-tphi[ri])); dr = np.hypot(geta-teta[ri], d); j = int(dr.argmin())
                    if dr[j] < DR:
                        anyok = True
                        if gpt[j] > 0: m["rany"][cc].append(tpt[ri]/gpt[j])
                    sm = gcls == cc
                    if sm.any():
                        gs_pt, gs_eta, gs_phi = gpt[sm], geta[sm], gphi[sm]
                        d2 = np.arctan2(np.sin(gs_phi-tphi[ri]), np.cos(gs_phi-tphi[ri])); dr2 = np.hypot(gs_eta-teta[ri], d2); j2 = int(dr2.argmin())
                        if dr2[j2] < DR:
                            sameok = True
                            if gs_pt[j2] > 0: m["rsame"][cc].append(tpt[ri]/gs_pt[j2])
                if not anyok: m["fakeA"][cc][bb, 1] += 1
                if not sameok: m["fake"][cc][bb, 1] += 1
    return m


COL = {"ours": ("tab:red", "v2"), "moh": ("tab:blue", "v1")}   # v2=ours (links,one/particle,no cuts); v1=colleague


def make_plots(M, sample, outdir):
    os.makedirs(outdir, exist_ok=True); sl = SL.get(sample, sample); plt.rcParams.update({"font.size": 12})
    # jet response
    fig, ax = plt.subplots(figsize=(7.6, 5.6)); b = np.logspace(-1, 1, 140)
    for k in ("ours", "moh"):
        r = np.array(M[k]["resp"]); md = np.median(r); iq = (np.percentile(r, 75)-np.percentile(r, 25))/md
        ax.hist(r, bins=b, histtype="step", lw=2.4, color=COL[k][0], label=f"{COL[k][1]}  med {md:.2f}, IQR/med {iq:.2f}, std {np.std(r):.2f}")
    ax.axvline(1, color="k", ls="--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"target-jet $p_T$ / gen-jet $p_T$"); ax.set_ylabel("jets"); ax.set_title(f"Jet response vs gen — {sl}"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{outdir}/jet_response.pdf"); plt.close()
    # sumpt per class
    fig, ax = plt.subplots(figsize=(9, 5.2)); x = np.arange(len(CLASSES)); w = 0.38
    for j, k in enumerate(("ours", "moh")):
        ax.bar(x+(j-0.5)*w, [M[k]["sump"][c]/max(M[k]["gsump"][c], 1e-9) for c in CLASSES], w, color=COL[k][0], label=COL[k][1])
    ax.axhline(1, color="k", ls="--", lw=1); ax.set_xticks(x); ax.set_xticklabels([CLAB[c] for c in CLASSES], rotation=15)
    ax.set_ylabel(r"target $\Sigma p_T$ / gen"); ax.set_title(f"$\\Sigma p_T$ per class — {sl}"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(f"{outdir}/sumpt_per_class.pdf"); plt.close()
    # grids
    def grid(key, ttl, fn, hist=False):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); axs = axs.ravel()
        for i, c in enumerate(CLASSES):
            ax = axs[i]
            for k in ("ours", "moh"):
                if hist:
                    r = np.array(M[k][key][c])
                    if len(r): ax.hist(r, bins=np.logspace(-1, 1, 70), histtype="step", lw=2, density=True, color=COL[k][0], label=f"{COL[k][1]} (med {np.median(r):.2f}, std {np.std(r):.2f})")
                    ax.set_xscale("log")
                else:
                    d = M[k][key][c]
                    with np.errstate(divide="ignore", invalid="ignore"): y = np.where(d[:, 0] > 0, d[:, 1]/d[:, 0], np.nan)
                    ax.plot(PTC, y, "o-", lw=2, color=COL[k][0], label=COL[k][1].split("(")[0]); ax.set_xscale("log"); ax.set_ylim(-.03, 1.03)
            if hist: ax.axvline(1, color="k", ls="--", lw=1)
            ax.set_title(CLAB[c]); ax.set_xlabel(r"target $p_T$/gen $p_T$" if hist else r"$p_T$ [GeV]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        axs[5].axis("off"); fig.suptitle(f"{ttl} — {sl}", fontsize=14); plt.tight_layout(); plt.savefig(f"{outdir}/{fn}"); plt.close()
    grid("eff", "Efficiency vs gen (same-pid)", "efficiency_samepid.pdf")
    grid("effA", "Efficiency vs gen (any-pid)", "efficiency_anypid.pdf")
    grid("fake", "Fake rate (same-pid)", "fakerate_samepid.pdf")
    grid("fakeA", "Fake rate (any-pid)", "fakerate_anypid.pdf")
    grid("rsame", "Particle pT resolution (same-pid)", "particle_resolution_samepid.pdf", hist=True)
    grid("rany", "Particle pT resolution (any-pid)", "particle_resolution_anypid.pdf", hist=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=["ttbar_0pu", "qcd_0pu", "zll_0pu"])
    ap.add_argument("--outbase", default="/Users/fmokhtar/projects/particleflow/plots/phase2")
    a = ap.parse_args()
    for s in a.samples:
        M = {"ours": compute(sorted(glob.glob(f"{NANO}/{s}/pkl_links/*.pkl"))),
             "moh":  compute(sorted(glob.glob(f"{NANO}/{s}/pkl_mohamed/moh_*.pkl")))}
        outdir = f"{a.outbase}/final_target_{s}"
        make_plots(M, s, outdir)
        jr = {k: np.median(M[k]["resp"]) for k in M}
        print(f"{s}: jetResp ours={jr['ours']:.3f} colleague={jr['moh']:.3f}  ->  {outdir}")


if __name__ == "__main__":
    main()
