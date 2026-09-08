#!/usr/bin/env python
"""
Jet-level sanity/diagnostic plots to understand the jet pT response.
For each sample, three panels:
  (1) jet pT spectrum          -- gen / MLPF target / PF (TICLCandidates)
  (2) jet |eta| distribution   -- gen / MLPF target / PF
  (3) Delta R(gen, nearest jet)-- gen->target and gen->PF  (matching quality)

Jets:  gen  = stored genjet (matched-gen truth jets)
       tgt  = stored targetjet (clustered from ytarget particles)
       pf   = clustered here from ycand (TICLCandidates), anti-kt R=0.4, pt>3
All jets from the same anti-kt R=0.4, pt>3 definition.
"""
import os, glob, pickle, argparse, re
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import fastjet
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

JET_PT_MIN  = 3.0     # jet clustering threshold
GEN_PT_CUT  = 20.0    # gen-jet pT cut for the |eta| and dR panels
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
def sample_of(f): return re.sub(r"_[0-9]+_[0-9]+\.pkl$", "", os.path.basename(f))

def cluster_ycand(rec):
    """Cluster reco candidates (ycand, pid!=0) into jets -> Nx4 (pt,eta,phi,E)."""
    m = rec["pid"] != 0
    if not np.any(m): return np.zeros((0, 4))
    pt, eta = rec["pt"][m], rec["eta"][m]
    phi = np.arctan2(rec["sin_phi"][m], rec["cos_phi"][m]); E = rec["energy"][m]
    px, py = pt*np.cos(phi), pt*np.sin(phi)
    pz = pt*np.sinh(np.where(np.abs(eta) < 10, eta, 0.0))
    pjs = [fastjet.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(E[i]))
           for i in range(len(pt))]
    if not pjs: return np.zeros((0, 4))
    jets = fastjet.ClusterSequence(pjs, JETDEF).inclusive_jets(ptmin=JET_PT_MIN)
    return np.array([[j.pt(), j.eta(), j.phi(), j.e()] for j in jets]) if jets else np.zeros((0, 4))

def nearest_dR(genj, recj):
    """For each gen jet (pt>cut) return dR to the nearest reco jet (no dR cut)."""
    out = []
    if genj.size == 0 or recj.size == 0: return out
    for g in genj:
        if g[0] < GEN_PT_CUT: continue
        dphi = np.arctan2(np.sin(recj[:, 2]-g[2]), np.cos(recj[:, 2]-g[2]))
        out.append(float(np.min(np.hypot(recj[:, 1]-g[1], dphi))))
    return out

def collect(files):
    out = defaultdict(lambda: {"gen": [], "tgt": [], "pf": [], "dr_tgt": [], "dr_pf": []})
    for f in files:
        try: data = pickle.load(open(f, "rb"))
        except Exception: continue
        s = sample_of(f)
        for ev in data:
            gj = np.atleast_2d(ev["genjet"])
            if gj.size == 0 or gj.shape[-1] != 4: continue
            gj = gj.reshape(-1, 4)
            tj = np.atleast_2d(ev["targetjet"]); tj = tj.reshape(-1, 4) if tj.size and tj.shape[-1] == 4 else np.zeros((0, 4))
            pj = cluster_ycand(ev["ycand"])
            out[s]["gen"].append(gj[:, :2]); out[s]["tgt"].append(tj[:, :2]); out[s]["pf"].append(pj[:, :2])
            out[s]["dr_tgt"] += nearest_dR(gj, tj)
            out[s]["dr_pf"]  += nearest_dR(gj, pj)
    res = {}
    for s, v in out.items():
        res[s] = {k: (np.concatenate(v[k]) if v[k] else np.zeros((0, 2))) for k in ("gen", "tgt", "pf")}
        res[s]["dr_tgt"] = np.array(v["dr_tgt"]); res[s]["dr_pf"] = np.array(v["dr_pf"])
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--outdir", required=True)
    ap.add_argument("--per-sample", type=int, default=250)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count()-2))
    a = ap.parse_args()

    by = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.dir, "*.pkl"))): by[sample_of(f)].append(f)
    files = [f for fs in by.values() for f in fs[: a.per_sample]]
    print(f"{len(files)} files, {a.workers} workers")

    nch = min(a.workers*4, len(files)) or 1
    chunks = [files[i::nch] for i in range(nch)]
    agg = defaultdict(lambda: {"gen": [], "tgt": [], "pf": [], "dr_tgt": [], "dr_pf": []})
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(collect, c) for c in chunks]):
            for s, v in fu.result().items():
                for k in v: agg[s][k].append(v[k])
    res = {s: {k: (np.concatenate(v[k]) if len(v[k]) else np.zeros((0,)+ (('',)[:0]))) for k in v}
           for s, v in agg.items()}

    proc = {"ticl_qcd_nopu": "QCD multijet, 0 PU", "ticl_ttbar_nopu": r"$t\bar{t}$, 0 PU",
            "ticl_zll_nopu": r"$Z\to\ell\ell$, 0 PU"}
    styles = [("gen", "gen jet", "black"), ("tgt", "MLPF target", "tab:orange"), ("pf", "PF", "tab:blue")]
    plt.rcParams.update({"font.size": 13})

    for s in [x for x in ["ticl_qcd_nopu", "ticl_ttbar_nopu", "ticl_zll_nopu"] if x in res]:
        R = res[s]
        fig, (axp, axe, axd) = plt.subplots(1, 3, figsize=(17, 5.0))

        # (1) jet pT spectrum (all jets pt>3)
        ptbins = np.logspace(np.log10(3), np.log10(600), 50)
        for k, lab, c in styles:
            arr = R[k]
            if arr.size == 0: continue
            axp.hist(arr[:, 0], bins=ptbins, histtype="step", lw=1.8, color=c, label=lab)
        axp.set_xscale("log"); axp.set_yscale("log")
        axp.set_xlabel(r"jet $p_\mathrm{T}$ [GeV]"); axp.set_ylabel("Jets")
        axp.set_title("Jet $p_\\mathrm{T}$ spectrum"); axp.legend()

        # (2) jet |eta| (jets pt>GEN_PT_CUT)
        ebins = np.linspace(1.0, 4.5, 40)
        for k, lab, c in styles:
            arr = R[k]
            if arr.size == 0: continue
            sel = arr[:, 0] > GEN_PT_CUT
            axe.hist(np.abs(arr[sel, 1]), bins=ebins, histtype="step", lw=1.8, color=c, label=lab)
        axe.set_xlabel(r"jet $|\eta|$"); axe.set_ylabel(f"Jets ($p_\\mathrm{{T}}>{GEN_PT_CUT:.0f}$ GeV)")
        axe.set_title(r"Jet $|\eta|$ distribution"); axe.legend()

        # (3) dR(gen, nearest jet)
        drbins = np.linspace(0, 0.6, 60)
        for k, lab, c in [("dr_tgt", "gen→MLPF target", "tab:orange"), ("dr_pf", "gen→PF", "tab:blue")]:
            if R[k].size == 0: continue
            axd.hist(R[k], bins=drbins, histtype="step", lw=1.8, color=c,
                     label=f"{lab} (med {np.median(R[k]):.03f})")
        for x in (0.1, 0.2, 0.4):
            axd.axvline(x, color="gray", ls=":", lw=1)
        axd.set_xlabel(r"$\Delta R$(gen jet, nearest jet)"); axd.set_ylabel(f"Gen jets ($p_\\mathrm{{T}}>{GEN_PT_CUT:.0f}$ GeV)")
        axd.set_title(r"Matching $\Delta R$"); axd.legend()

        fig.suptitle(proc.get(s, s), fontsize=15, y=1.02)
        plt.tight_layout()
        fn = f"jet_diag_{s.replace('ticl_','').replace('_nopu','')}.pdf"
        plt.savefig(os.path.join(a.outdir, fn), bbox_inches="tight"); plt.close(fig)
        ng = R["gen"].shape[0]; nt = R["tgt"].shape[0]; npf = R["pf"].shape[0]
        print(f"{s}: genjets={ng} targetjets={nt} pfjets={npf} | "
              f"dR med tgt={np.median(R['dr_tgt']):.3f} pf={np.median(R['dr_pf']):.3f} -> {fn}")

if __name__ == "__main__": main()
