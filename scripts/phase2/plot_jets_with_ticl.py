#!/usr/bin/env python
"""Target jet response + pT spectra vs gen, with the TICLCandidates (reco baseline)
curve overlaid. Reads targetjet/genjet from the run3style pkls and clusters TICL
jets from the matching raw NanoAOD (pkl run3style_<X>.pkl <-> <X>.root, same event order)."""
import glob, pickle, argparse, os, re
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import uproot, awkward as ak, fastjet
from concurrent.futures import ProcessPoolExecutor, as_completed

DR, GENPT, ELO, EHI = 0.2, 20.0, 1.5, 3.0
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
SL = {"ttbar_0pu": r"$t\bar{t}$, 0 PU", "qcd_0pu": "QCD multijet, 0 PU", "zll_0pu": r"$Z\to\ell\ell$, 0 PU"}

def cluster(pt, eta, phi, e):
    ok = pt > 0
    pt, eta, phi, e = pt[ok], eta[ok], phi[ok], e[ok]
    if len(pt) == 0: return np.zeros((0, 4))
    px = pt*np.cos(phi); py = pt*np.sin(phi); pz = pt*np.sinh(np.where(np.abs(eta) < 10, eta, 0.0))
    pjs = [fastjet.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(e[i])) for i in range(len(pt))]
    j = fastjet.ClusterSequence(pjs, JETDEF).inclusive_jets(ptmin=3.0)
    return np.array([[x.pt(), x.eta(), x.phi(), x.e()] for x in j]) if j else np.zeros((0, 4))

def resp(genj, recj):
    out = []
    if len(genj) == 0 or len(recj) == 0: return out
    for g in genj:
        if g[0] < GENPT or abs(g[1]) < ELO or abs(g[1]) > EHI: continue
        dphi = np.arctan2(np.sin(recj[:, 2]-g[2]), np.cos(recj[:, 2]-g[2]))
        b = int(np.argmin(np.hypot(recj[:, 1]-g[1], dphi)))
        d = np.hypot(recj[b, 1]-g[1], np.arctan2(np.sin(recj[b, 2]-g[2]), np.cos(recj[b, 2]-g[2])))
        if d < DR: out.append(recj[b, 0]/g[0])
    return out

def worker(args):
    pkl, rootdir = args
    base = re.sub(r"^run3style_", "", os.path.basename(pkl))[:-4]  # -> <X>
    root = os.path.join(rootdir, base + ".root")
    d = pickle.load(open(pkl, "rb"))
    A = uproot.open(root)["Events"].arrays(
        ["TICLCandidates_pt", "TICLCandidates_eta", "TICLCandidates_phi", "TICLCandidates_energy"])
    tr, cr = [], []          # target/gen, ticl/gen responses
    gjs, tjs, cjs = [], [], []  # pt,eta for gen/target/ticl jets
    for iev, ev in enumerate(d):
        gj = np.atleast_2d(ev["genjet"]); tj = np.atleast_2d(ev["targetjet"])
        gj = gj.reshape(-1, 4) if gj.size and gj.shape[-1] == 4 else np.zeros((0, 4))
        tj = tj.reshape(-1, 4) if tj.size and tj.shape[-1] == 4 else np.zeros((0, 4))
        cj = cluster(ak.to_numpy(A["TICLCandidates_pt"][iev]), ak.to_numpy(A["TICLCandidates_eta"][iev]),
                     ak.to_numpy(A["TICLCandidates_phi"][iev]), ak.to_numpy(A["TICLCandidates_energy"][iev]))
        tr += resp(gj, tj); cr += resp(gj, cj)
        if len(gj): gjs.append(gj[:, :2])
        if len(tj): tjs.append(tj[:, :2])
        if len(cj): cjs.append(cj[:, :2])
    def cat(x): return np.concatenate(x) if x else np.zeros((0, 2))
    return np.array(tr), np.array(cr), cat(gjs), cat(tjs), cat(cjs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkldir", required=True); ap.add_argument("--rootdir", required=True)
    ap.add_argument("--outdir", required=True); ap.add_argument("--sample", default="ttbar_0pu")
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    pkls = sorted(glob.glob(os.path.join(a.pkldir, "*.pkl")))
    TR, CR, GJ, TJ, CJ = [], [], [], [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(worker, (p, a.rootdir)) for p in pkls]):
            tr, cr, gj, tj, cj = fu.result()
            TR.append(tr); CR.append(cr); GJ.append(gj); TJ.append(tj); CJ.append(cj)
    tr = np.concatenate(TR); cr = np.concatenate(CR)
    gj = np.concatenate(GJ); tj = np.concatenate(TJ); cj = np.concatenate(CJ)
    lab = SL.get(a.sample, a.sample)
    plt.rcParams.update({"font.size": 14})

    # response overlay
    fig, ax = plt.subplots(figsize=(7.6, 5.7)); b = np.logspace(-1, 1, 150)
    for r, name, c in [(cr, "TICL (reco)", "tab:blue"), (tr, "MLPF target (Run3-style)", "tab:orange")]:
        m = np.median(r); iq = (np.percentile(r, 75)-np.percentile(r, 25))/m
        ax.hist(r, bins=b, histtype="step", lw=2, color=c, label=f"{name} (med {m:.2f}, IQR/med {iq:.2f})")
    ax.axvline(1, color="k", ls="--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_\mathrm{T}$ / genjet $p_\mathrm{T}$"); ax.set_ylabel("Jets")
    ax.set_title(r"Jet response vs gen (gen $p_\mathrm{T}>20$, $1.5<|\eta|<3.0$)")
    ax.text(0.97, 0.93, lab, transform=ax.transAxes, ha="right", va="top", fontsize=13)
    ax.legend(fontsize=11); plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, "run3style_response_with_ticl.pdf")); plt.close(fig)

    # pt spectra overlay
    fig, ax = plt.subplots(figsize=(7.6, 5.7)); bb = np.logspace(np.log10(3), np.log10(800), 50)
    for arr, name, c in [(gj, "gen (status 1, no ν)", "black"), (tj, "MLPF target", "tab:orange"), (cj, "TICL (reco)", "tab:blue")]:
        mm = np.abs(arr[:, 1]) > ELO
        ax.hist(arr[mm, 0], bins=bb, histtype="step", lw=2, color=c, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_\mathrm{T}$ [GeV]"); ax.set_ylabel("Jets")
    ax.set_title(r"Jet $p_\mathrm{T}$ spectra ($|\eta|>1.5$)")
    ax.text(0.97, 0.93, lab, transform=ax.transAxes, ha="right", va="top", fontsize=13)
    ax.legend(fontsize=11); plt.tight_layout()
    plt.savefig(os.path.join(a.outdir, "run3style_pt_spectra_with_ticl.pdf")); plt.close(fig)
    print(f"{a.sample}: target med={np.median(tr):.3f} (N={len(tr)}) | TICL med={np.median(cr):.3f} (N={len(cr)})")

if __name__ == "__main__":
    main()
