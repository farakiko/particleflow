#!/usr/bin/env python
"""CMS-style jet pT response (jet pT / genjet pT): baseline reco (PF=TICLCandidates,
ycand) vs MLPF target (ytarget). Matches the reference plot's axis conventions."""
import os, glob, pickle, argparse, re
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import fastjet
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

DR_MATCH = 0.1; PT_MIN_JET = 3.0; GEN_PT_MIN = 20.0
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
def sample_of(f): return re.sub(r"_[0-9]+_[0-9]+\.pkl$", "", os.path.basename(f))

def cluster(rec):
    """rec: structured array with pid,pt,eta,sin_phi,cos_phi,energy -> Nx4 (pt,eta,phi,E)."""
    m = rec["pid"] != 0
    if not np.any(m): return np.zeros((0,4))
    pt=rec["pt"][m]; eta=rec["eta"][m]
    phi=np.arctan2(rec["sin_phi"][m], rec["cos_phi"][m]); E=rec["energy"][m]
    px=pt*np.cos(phi); py=pt*np.sin(phi); pz=pt*np.sinh(np.where(np.abs(eta)<10,eta,0.0))
    pjs=[fastjet.PseudoJet(float(px[i]),float(py[i]),float(pz[i]),float(E[i])) for i in range(len(pt))]
    if not pjs: return np.zeros((0,4))
    jets=fastjet.ClusterSequence(pjs, JETDEF).inclusive_jets(ptmin=PT_MIN_JET)
    return np.array([[j.pt(),j.eta(),j.phi(),j.e()] for j in jets]) if jets else np.zeros((0,4))

def match(genj, recj):
    out=[]
    if genj.size==0 or recj.size==0: return out
    genj=genj.reshape(-1,4); recj=recj.reshape(-1,4)
    for g in genj:
        if g[0] < GEN_PT_MIN: continue
        dphi=np.arctan2(np.sin(recj[:,2]-g[2]), np.cos(recj[:,2]-g[2]))
        dR=np.hypot(recj[:,1]-g[1], dphi); j=int(np.argmin(dR))
        if dR[j]<DR_MATCH: out.append((g[0], g[1], recj[j,0]/g[0]))
    return out

def collect(files):
    out=defaultdict(lambda: {"pf":[], "tgt":[]})
    for f in files:
        try: data=pickle.load(open(f,"rb"))
        except Exception: continue
        s=sample_of(f)
        for ev in data:
            gj=np.atleast_2d(ev["genjet"])
            if gj.size==0 or gj.shape[-1]!=4: continue
            out[s]["tgt"] += match(gj, np.atleast_2d(ev["targetjet"]))
            out[s]["pf"]  += match(gj, cluster(ev["ycand"]))
    return {k:{kk:np.array(vv) for kk,vv in v.items()} for k,v in out.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir",required=True); ap.add_argument("--outdir",required=True)
    ap.add_argument("--per-sample",type=int,default=250)
    ap.add_argument("--workers",type=int,default=max(1,os.cpu_count()-2))
    a=ap.parse_args()
    by=defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.dir,"*.pkl"))): by[sample_of(f)].append(f)
    files=[f for fs in by.values() for f in fs[:a.per_sample]]
    print(f"{len(files)} files, {a.workers} workers")
    nch=min(a.workers*4,len(files)) or 1
    chunks=[files[i::nch] for i in range(nch)]
    res=defaultdict(lambda: {"pf":[], "tgt":[]})
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(collect,c) for c in chunks]):
            for k,v in fu.result().items():
                for kk in ("pf","tgt"):
                    if len(v[kk]): res[k][kk].append(v[kk])
    res={k:{kk:(np.concatenate(vv) if vv else np.zeros((0,3))) for kk,vv in v.items()} for k,v in res.items()}

    titles={"ticl_qcd_nopu":"QCD 0PU","ticl_ttbar_nopu":"ttbar 0PU","ticl_zll_nopu":"Z→ll 0PU"}
    bins=np.logspace(np.log10(0.1), np.log10(10), 200)
    plt.rcParams.update({"font.size":15})
    for s in [x for x in ["ticl_qcd_nopu","ticl_ttbar_nopu","ticl_zll_nopu"] if x in res]:
        fig,ax=plt.subplots(figsize=(7.6,6.4))
        for key,lab,col in [("pf","PF","tab:blue"),("tgt","MLPF target","tab:orange")]:
            d=res[s][key]
            if d.size==0: continue
            ax.hist(d[:,2], bins=bins, histtype="step", lw=1.6, color=col, label=lab)
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.1,10); ax.set_ylim(0.7, ax.get_ylim()[1])
        ax.set_xlabel(r"jet $p_\mathrm{T}$ / genjet $p_\mathrm{T}$")
        ax.set_ylabel("Counts")
        ax.grid(alpha=0.25, ls="--", which="both")
        ax.text(0.0,1.015,"CMS",transform=ax.transAxes,fontweight="bold",fontsize=19)
        ax.text(0.135,1.015,"Simulation",transform=ax.transAxes,style="italic",fontsize=16)
        ax.text(1.0,1.015,"Run4",transform=ax.transAxes,ha="right",fontsize=16)
        ax.legend(loc="upper left", frameon=False)
        proc = {"ticl_qcd_nopu": r"QCD multijet, 0 PU",
                "ticl_ttbar_nopu": r"$t\bar{t}$, 0 PU",
                "ticl_zll_nopu": r"$Z\to\ell\ell$, 0 PU"}.get(s, s)
        ax.text(0.97, 0.93, proc, transform=ax.transAxes, ha="right", va="top", fontsize=16)
        plt.tight_layout()
        fn=f"jet_response_cms_{s.replace('ticl_','').replace('_nopu','')}.pdf"
        plt.savefig(os.path.join(a.outdir,fn)); plt.close(fig)
        print(f"{s}: PF n={res[s]['pf'].shape[0]} med={np.median(res[s]['pf'][:,2]):.3f} | "
              f"target n={res[s]['tgt'].shape[0]} med={np.median(res[s]['tgt'][:,2]):.3f} -> {fn}")

if __name__=="__main__": main()
