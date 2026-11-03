#!/usr/bin/env python3
"""
MLPF CMS Analysis for TICL Graph Data
"""

import sys, os, glob, pickle, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import mplhep
import awkward as ak
import vector, fastjet, tqdm

warnings.filterwarnings("ignore")

sys.path += ["../../mlpf/", "../../mlpf/plotting/"]
from plot_utils import cms_label, sample_label, pid_to_text
from plot_utils import ELEM_NAMES_CMS, save_img

mplhep.style.use("CMS")

SAMPLE_NAME  = "cms_pf_ttbar"
MIN_JET_PT   = 3.0
JET_RADIUS   = 0.4
PLOT_DIR     = "mlpf_analysis_plots"

_JET_DEF = fastjet.JetDefinition(fastjet.antikt_algorithm, JET_RADIUS)

TICL_TYPE_NAMES = {
    1: "Track",
    2: "EM Trackster",
    4: "Hadronic Trackster",
}

def _savefig(fig, name):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(name, dpi=150)
    plt.close(fig)


# Loading
def _load_one(path):
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d if isinstance(d, list) else [d]
    except Exception as e:
        print(f"  [WARN] {path}: {e}"); return []

def load_files_parallel(files, max_workers=None):
    max_workers = max_workers or min(len(files), os.cpu_count())
    out = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_load_one, p): p for p in files}
        for f in tqdm.tqdm(as_completed(futs), total=len(futs), desc="Loading"):
            out.extend(f.result())
    return out


# Conversion
def convert_events_vectorised(ticl_data):
    cols = {
        "Xelem":   {f: [] for f in ["typ","pt","eta","energy","phi","charge"]},
        "ytarget": {f: [] for f in ["pid","pt","eta","energy","sin_phi","cos_phi","phi","ispu","charge"]},
        "ycand":   {f: [] for f in ["pid","pt","eta","energy","sin_phi","cos_phi","phi","ispu"]},
        "pythia":  {f: [] for f in ["pid","pt","eta","phi","energy"]},
        "genmet":  [],
    }
    gj_pts, gj_etas, gj_phis, gj_ens = [], [], [], []

    def _get(ra, field, n):
        return ra[field].astype(np.float32) if field in ra.dtype.names \
               else np.zeros(n, np.float32)

    for ev in ticl_data:
        X, y, c = ev["Xelem"], ev["ytarget"], ev["ycand"]
        nX, ny, nc = len(X), len(y), len(c)

        for f in ["typ","pt","eta","energy","charge"]:
            cols["Xelem"][f].append(_get(X,f,nX))
        cols["Xelem"]["phi"].append(_get(X,"phi",nX))

        phi_y = np.arctan2(y["sin_phi"], y["cos_phi"]).astype(np.float32)
        cols["ytarget"]["pid"].append(np.abs(_get(y,"pid",ny)))
        for f in ["pt","eta","energy","sin_phi","cos_phi","ispu","charge"]:
            cols["ytarget"][f].append(_get(y,f,ny))
        cols["ytarget"]["phi"].append(phi_y)

        phi_c = (np.arctan2(c["sin_phi"],c["cos_phi"]).astype(np.float32)
                 if "sin_phi" in c.dtype.names else np.zeros(nc,np.float32))
        cols["ycand"]["pid"].append(np.abs(_get(c,"pid",nc)))
        for f in ["pt","eta","energy","sin_phi","cos_phi","ispu"]:
            cols["ycand"][f].append(_get(c,f,nc))
        cols["ycand"]["phi"].append(phi_c)

        p = ev.get("pythia", np.array([]))
        if len(p) > 0 and p.ndim == 2 and p.shape[1] >= 5:
            cols["pythia"]["pid"].append(np.abs(p[:,0]).astype(np.float32))
            for i,f in enumerate(["pt","eta","phi","energy"],1):
                cols["pythia"][f].append(p[:,i].astype(np.float32))
        else:
            for f in cols["pythia"]: cols["pythia"][f].append(np.array([],np.float32))

        # preprocessing saves genmet as np.array([met_magnitude, met_phi])
        # so gm[0] is the scalar MET magnitude
        gm = ev.get("genmet", None)
        if gm is None:
            cols["genmet"].append(0.0)
        elif hasattr(gm, '__len__'):
            cols["genmet"].append(float(gm[0]) if len(gm) >= 1 else 0.0)
        else:
            cols["genmet"].append(float(gm))

        gj = ev.get("genjet", np.array([]))
        if len(gj) > 0 and gj.ndim == 2 and gj.shape[1] >= 4:
            gj_pts.append(gj[:,0].tolist()); gj_etas.append(gj[:,1].tolist())
            gj_phis.append(gj[:,2].tolist()); gj_ens.append(gj[:,3].tolist())
        else:
            gj_pts.append([]); gj_etas.append([]); gj_phis.append([]); gj_ens.append([])

    xelem_masks = [a != 0 for a in cols["Xelem"]["typ"]]
    ytgt_masks  = [a != 0 for a in cols["ytarget"]["pid"]]
    ycand_masks = [a != 0 for a in cols["ycand"]["pid"]]

    def _masked(field_lists, masks):
        return {f: ak.Array([a[m] for a, m in zip(v, masks)])
                for f, v in field_lists.items()}

    arrs_awk = {
        "Xelem":   _masked(cols["Xelem"],   xelem_masks),
        "ytarget": _masked(cols["ytarget"], ytgt_masks),
        "ycand":   _masked(cols["ycand"],   ycand_masks),
        "pythia":  {f: ak.Array(v) for f, v in cols["pythia"].items()},
    }

    arrs_flat = {
        coll: {f: np.concatenate(v) if v else np.array([], np.float32)
               for f, v in fds.items()}
        for coll, fds in cols.items() if coll != "genmet"
    }

    genmet_arr = np.array(cols["genmet"], dtype=np.float32)
    genjet_cmssw = vector.awk(ak.zip({
        "pt": ak.Array(gj_pts), "eta": ak.Array(gj_etas),
        "phi": ak.Array(gj_phis), "energy": ak.Array(gj_ens),
    }))
    return arrs_awk, arrs_flat, genmet_arr, genjet_cmssw



def _compute_met(pt_awk, phi_awk):
    px = pt_awk * np.cos(phi_awk)
    py = pt_awk * np.sin(phi_awk)
    return ak.to_numpy(
        np.sqrt(ak.sum(px, axis=1)**2 + ak.sum(py, axis=1)**2)
    ).astype(np.float32)


def plot_met(arrs_awk, genmet_arr, plot_dir):
    print("   Computing MET from collections...")

    cand_met        = _compute_met(arrs_awk["ycand"]["pt"],   arrs_awk["ycand"]["phi"])
    target_met      = _compute_met(arrs_awk["ytarget"]["pt"], arrs_awk["ytarget"]["phi"])
    nopu_mask       = arrs_awk["ytarget"]["ispu"] < 0.5
    target_met_nopu = _compute_met(
        arrs_awk["ytarget"]["pt"][nopu_mask],
        arrs_awk["ytarget"]["phi"][nopu_mask]
    )

    print(f"   genMET          mean={genmet_arr.mean():.1f}  median={np.median(genmet_arr):.1f}  max={genmet_arr.max():.1f}")
    print(f"   targetMET       mean={target_met.mean():.1f}  median={np.median(target_met):.1f}")
    print(f"   targetMET(nopu) mean={target_met_nopu.mean():.1f}  median={np.median(target_met_nopu):.1f}")
    print(f"   candMET         mean={cand_met.mean():.1f}  median={np.median(cand_met):.1f}")

    all_met_vals = np.concatenate([
        genmet_arr[genmet_arr > 0],
        target_met[target_met > 0],
        cand_met[cand_met > 0]
    ])
    if len(all_met_vals):
        lo = max(np.log10(np.percentile(all_met_vals, 1)), -2)
        hi = min(np.log10(np.percentile(all_met_vals, 99)) + 0.5, 5)
    else:
        lo, hi = -2, 4
    b = np.logspace(lo, hi, 100)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(genmet_arr,      bins=b, histtype="step", lw=2, label="genMET")
    ax.hist(cand_met,        bins=b, histtype="step", lw=2, label="PF")
    ax.hist(target_met,      bins=b, histtype="step", lw=2, label="MLPF targets")
    #ax.hist(target_met_nopu, bins=b, histtype="step", lw=2, label="MLPF targets, no PU")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("MET (GeV)"); ax.set_ylabel("Events / bin")
    ax.legend(loc="upper left", fontsize=12); ax.grid(alpha=0.3)
    cms_label(ax)
    #sample_label(ax, SAMPLE_NAME)
    _savefig(fig, os.path.join(plot_dir, "met_distributions.png"))

    x = genmet_arr.astype(float)
    y = target_met_nopu.astype(float)
    valid = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    print(f"   hist2d: {valid.sum()} valid (x>0, y>0) event pairs out of {len(x)}")

    fig, ax = plt.subplots(figsize=(12, 10))
    if valid.sum() >= 2:
        x_lo = np.log10(np.percentile(x[valid], 1))
        x_hi = np.log10(np.percentile(x[valid], 99)) + 0.3
        y_lo = np.log10(np.percentile(y[valid], 1))
        y_hi = np.log10(np.percentile(y[valid], 99)) + 0.3
        bins2d = [np.logspace(x_lo, x_hi, 100), np.logspace(y_lo, y_hi, 100)]
        h = ax.hist2d(x[valid], y[valid], bins=bins2d,
                      cmap="hot_r", norm=mcolors.LogNorm())
        fig.colorbar(h[3], ax=ax, label="Counts")
        ref = [10**x_lo, 10**x_hi]
        ax.plot(ref, ref, color="black", ls="--", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "Insufficient data for hist2d\n(genMET may be zero in this sample)",
                transform=ax.transAxes, ha="center", va="center", fontsize=14, color="red")
        print("   WARNING: skipping hist2d — not enough valid (genMET>0, targetMET>0) pairs")
    ax.set_xlabel("Pythia MET (GeV)")
    ax.set_ylabel("Target MET, ispu < 0.5 (GeV)")
    cms_label(ax)
    #sample_label(ax, SAMPLE_NAME)
    _savefig(fig, os.path.join(plot_dir, "met_response_hist2d.png"))


# Jet clustering
def cluster_jets_batch(pts_awk, etas_awk, phis_awk, ens_awk, min_pt=MIN_JET_PT):
    all_pt, all_eta, all_phi, all_en = [], [], [], []
    for iev in range(len(pts_awk)):
        pts  = ak.to_numpy(pts_awk[iev]).astype(np.float64)
        etas = ak.to_numpy(etas_awk[iev]).astype(np.float64)
        phis = ak.to_numpy(phis_awk[iev]).astype(np.float64)
        ens  = ak.to_numpy(ens_awk[iev]).astype(np.float64)
        ok   = (pts > 0) & np.isfinite(etas) & np.isfinite(phis)
        pts, etas, phis, ens = pts[ok], etas[ok], phis[ok], ens[ok]
        if not len(pts):
            for lst in [all_pt,all_eta,all_phi,all_en]: lst.append([])
            continue
        px = pts*np.cos(phis); py = pts*np.sin(phis)
        pz = np.where(np.abs(etas)<10, pts*np.sinh(etas), 0.0)
        pjs  = [fastjet.PseudoJet(float(px[i]),float(py[i]),float(pz[i]),float(ens[i]))
                for i in range(len(pts))]
        jets = fastjet.ClusterSequence(pjs, _JET_DEF).inclusive_jets(ptmin=min_pt)
        all_pt.append([j.pt() for j in jets]);  all_eta.append([j.eta() for j in jets])
        all_phi.append([j.phi() for j in jets]); all_en.append([j.e()   for j in jets])
    return vector.awk(ak.zip({"pt":ak.Array(all_pt),"eta":ak.Array(all_eta),
                               "phi":ak.Array(all_phi),"energy":ak.Array(all_en)}))

def cluster_all(arrs_awk, genjet_cmssw):
    jets = {"cmssw": genjet_cmssw}
    for coll in ["ytarget","ycand","pythia"]:
        if coll not in arrs_awk: continue
        print(f"   Clustering {coll}...")
        jets[coll] = cluster_jets_batch(
            arrs_awk[coll]["pt"], arrs_awk[coll]["eta"],
            arrs_awk[coll]["phi"], arrs_awk[coll]["energy"])
        print(f"      -> {int(ak.sum(ak.num(jets[coll].pt)))} jets")
    if "ytarget" in arrs_awk:
        nopu = arrs_awk["ytarget"]["ispu"] < 0.5
        print("   Clustering ytarget_nopu...")
        jets["ytarget_nopu"] = cluster_jets_batch(
            arrs_awk["ytarget"]["pt"][nopu], arrs_awk["ytarget"]["eta"][nopu],
            arrs_awk["ytarget"]["phi"][nopu], arrs_awk["ytarget"]["energy"][nopu])
        print(f"      -> {int(ak.sum(ak.num(jets['ytarget_nopu'].pt)))} jets")
    return jets


# Jet matching
def match_jet_collections(jets_coll, ref_key, tgt_key, dR_max=0.1):
    ref_jets = jets_coll[ref_key]
    tgt_jets = jets_coll[tgt_key]
    ref_pts, tgt_pts = [], []
    for iev in range(len(ref_jets.pt)):
        rp = ak.to_numpy(ref_jets[iev].pt).astype(float)
        re = ak.to_numpy(ref_jets[iev].eta).astype(float)
        rf = ak.to_numpy(ref_jets[iev].phi).astype(float)
        tp = ak.to_numpy(tgt_jets[iev].pt).astype(float)
        te = ak.to_numpy(tgt_jets[iev].eta).astype(float)
        tf = ak.to_numpy(tgt_jets[iev].phi).astype(float)
        if not len(rp) or not len(tp): continue
        for i in range(len(rp)):
            deta = te - re[i]
            dphi = np.arctan2(np.sin(tf - rf[i]), np.cos(tf - rf[i]))
            dR   = np.hypot(deta, dphi)
            b    = np.argmin(dR)
            if dR[b] < dR_max:
                ref_pts.append(rp[i]); tgt_pts.append(tp[b])
    return np.array(ref_pts), np.array(tgt_pts)



def _frac_bins(vals, ispu, bins):
    out = []
    for lo,hi in zip(bins[:-1],bins[1:]):
        msk=(vals>=lo)&(vals<hi); n=msk.sum()
        out.append(float(np.sum(ispu[msk]>0.5)/n) if n else 0.0)
    return np.array(out)


def _worker_pid_pt_plot(args):
    v, py_pt, tg_pt, plot_dir = args
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, mplhep; mplhep.style.use("CMS")
    try:
        from plot_utils import pid_to_text
    except ImportError:
        pid_to_text = {}
    b = np.linspace(0, 1500, 201)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(py_pt): ax.hist(py_pt, bins=b, histtype="step", label="Pythia", lw=2)
    if len(tg_pt): ax.hist(tg_pt, bins=b, histtype="step", label="Target", lw=2)
    ax.set_yscale("log"); ax.legend()
    ax.set_xlabel("Particle pT (GeV)"); ax.set_ylabel("Counts")
    ax.set_title(f"PID {v} - {pid_to_text.get(v, str(v))}"); ax.grid(alpha=0.3)
    try: fig.tight_layout()
    except Exception: pass
    fig.savefig(os.path.join(plot_dir, f"pt_dist_pid_{v}.png"), dpi=150)
    plt.close(fig)


def _worker_pid_eta_plot(args):
    v, py_eta, tg_eta, plot_dir = args
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, mplhep; mplhep.style.use("CMS")
    try:
        from plot_utils import pid_to_text
    except ImportError:
        pid_to_text = {}
    b = np.linspace(-5, 5, 201)
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(py_eta): ax.hist(py_eta, bins=b, histtype="step", label="Pythia", lw=2)
    if len(tg_eta): ax.hist(tg_eta, bins=b, histtype="step", label="Target", lw=2)
    ax.set_yscale("log"); ax.legend()
    ax.set_xlabel("Particle eta"); ax.set_ylabel("Counts")
    ax.set_title(f"PID {v} - {pid_to_text.get(v, str(v))} eta"); ax.grid(alpha=0.3)
    try: fig.tight_layout()
    except Exception: pass
    fig.savefig(os.path.join(plot_dir, f"eta_dist_pid_{v}.png"), dpi=150)
    plt.close(fig)


def _worker_elem_matching(args):
    elem_type, en, gpid, cpid, total, plot_dir = args
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, mplhep; mplhep.style.use("CMS")
    ticl_names = {1:"Track",2:"EM Trackster e",3:"EM Trackster g",
                  4:"Hadronic Trackster",5:"Muon",6:"Electron Track"}
    tname = ticl_names.get(int(elem_type), f"Type {elem_type}")
    bins = np.logspace(-1, 3, 50); fg=[]; fc=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        b=(en>=lo)&(en<hi); n=b.sum()
        fg.append(float(np.sum(gpid[b]!=0)/n) if n else 0.0)
        fc.append(float(np.sum(cpid[b]!=0)/n) if n else 0.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bins[:-1], fg, ".-", lw=2, label="Truth-matched (target)", markersize=8)
    ax.plot(bins[:-1], fc, ".-", lw=2, label="PF-matched (cand)",      markersize=8)
    ax.set_xscale("log"); ax.set_ylim(0, 1.1)
    ax.set_xlabel("Element Energy E (GeV)"); ax.set_ylabel("Matched fraction")
    ax.set_title(f"TICL Element Type {int(elem_type)}: {tname}")
    ax.legend(fontsize=10); ax.grid(alpha=0.3, ls="--")
    try: fig.tight_layout()
    except Exception: pass
    fig.savefig(os.path.join(plot_dir, f"ticl_element_matching_type{int(elem_type)}.png"), dpi=150)
    plt.close(fig)
    nt=int((gpid!=0).sum()); np_=int((cpid!=0).sum())
    return f"Type {int(elem_type)} ({tname}): {total} | truth {nt} ({100*nt/total:.1f}%) | PF {np_} ({100*np_/total:.1f}%)"


def _worker_elem_ptratio(args):
    elem_type, xpt, tgt_pid, tgt_pt, cand_pid, cand_pt, plot_dir = args
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, mplhep; mplhep.style.use("CMS")
    ticl_names = {1:"Track",2:"EM Trackster e",3:"EM Trackster g",
                  4:"Hadronic Trackster",5:"Muon",6:"Electron Track"}
    tname = ticl_names.get(int(elem_type), f"Type {elem_type}")
    b = np.logspace(-3, 3, 100)
    fig, ax = plt.subplots(figsize=(8, 6))
    for pid_arr, pt_arr, lbl in [(tgt_pid, tgt_pt, "Truth-matched (target)"),
                                  (cand_pid, cand_pt, "PF-matched (cand)")]:
        msk = pid_arr != 0
        if msk.any():
            ep = xpt[msk]; pp = pt_arr[msk]
            ok = ep > 0; r = np.where(ok, pp/np.where(ok,ep,1.0), np.nan)
            r = r[np.isfinite(r)]
            if len(r): ax.hist(r, bins=b, histtype="step", lw=2, alpha=0.8, label=lbl)
    ax.axvline(1.0, color="k", ls="--", alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Particle pT / Element pT"); ax.set_ylabel("Counts")
    ax.set_title(f"TICL Type {int(elem_type)}: {tname}")
    ax.legend(fontsize=10); ax.grid(alpha=0.3, ls="--")
    try: fig.tight_layout()
    except Exception: pass
    fig.savefig(os.path.join(plot_dir, f"ticl_elem_ptratio_type{int(elem_type)}.png"), dpi=150)
    plt.close(fig)


# Standard plots
def plot_pu_fraction(arrs_flat, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(arrs_flat["ytarget"]["ispu"], bins=np.linspace(0,1,101), histtype="step", lw=2)
    ax.set_yscale("log"); ax.set_xlabel("PU fraction"); ax.set_ylabel("Counts"); ax.grid(alpha=0.3)
    cms_label(ax)
    #sample_label(ax, SAMPLE_NAME)
    _savefig(fig, os.path.join(plot_dir, "pu_frac.png"))


def plot_pu_fraction_vs_pt(arrs_flat, plot_dir):
    bins = np.logspace(-3, 3, 50); c = 0.5*(bins[:-1]+bins[1:])
    fr   = _frac_bins(arrs_flat["ytarget"]["pt"], arrs_flat["ytarget"]["ispu"], bins)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c, fr, ".-", lw=2); ax.set_xscale("log"); ax.set_ylim(0, 1.2)
    ax.axhline(1.0, color="k", ls="--", alpha=0.5); ax.axhline(0.5, color="r", ls=":", alpha=0.5)
    ax.set_xlabel("Particle pT (GeV)"); ax.set_ylabel("Fraction isPU > 0.5"); ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(plot_dir, "pu_frac_vs_pt.png"))


def plot_pu_fraction_vs_eta(arrs_flat, plot_dir):
    bins = np.linspace(-5, 5, 50); c = 0.5*(bins[:-1]+bins[1:])
    fr   = _frac_bins(arrs_flat["ytarget"]["eta"], arrs_flat["ytarget"]["ispu"], bins)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c, fr, ".-", lw=2); ax.set_ylim(0, 1.2)
    ax.axhline(1.0, color="k", ls="--", alpha=0.5); ax.axhline(0.5, color="r", ls=":", alpha=0.5)
    ax.set_xlabel("Particle eta"); ax.set_ylabel("Fraction isPU > 0.5"); ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(plot_dir, "pu_frac_vs_eta.png"))


def plot_jet_distributions(jets_coll, plot_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, jets in jets_coll.items():
        if name in ["cmssw", "ytarget_nopu"]:
            continue
        try:
            pts = ak.to_numpy(ak.flatten(jets.pt))
            if len(pts): ax.hist(pts, bins=np.logspace(0,4,100), histtype="step", lw=2, label=name, alpha=0.7)
        except Exception: pass
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Jet pT (GeV)"); ax.set_ylabel("Counts"); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(plot_dir, "jet_pt_distributions.png"))

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, jets in jets_coll.items():
        if name in ["cmssw", "ytarget_nopu"]:
            continue
        try:
            etas = ak.to_numpy(ak.flatten(jets.eta))
            if len(etas): ax.hist(np.abs(etas), bins=np.linspace(0,6,100), histtype="step", lw=2, label=name, alpha=0.7)
        except Exception: pass
    ax.set_xlabel("Jet |eta|"); ax.set_ylabel("Counts"); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(plot_dir, "jet_eta_distributions.png"))


def plot_event_display(arrs_awk, jets_coll, plot_dir, iev=2):
    fig, ax = plt.subplots(figsize=(12, 10))
    for coll, lbl, mk in [("pythia","Pythia ptcl","o"), ("ytarget","MLPF target","s")]:
        if coll in arrs_awk and len(arrs_awk[coll]["pt"][iev]) > 0:
            ax.scatter(ak.to_numpy(arrs_awk[coll]["eta"][iev]),
                       ak.to_numpy(arrs_awk[coll]["phi"][iev]),
                       s=5*ak.to_numpy(arrs_awk[coll]["pt"][iev]),
                       alpha=0.5, label=lbl, marker=mk)
    for jn, jl, jm in [("cmssw","genJets","v"), ("ytarget","target jets","^")]:
        if jn in jets_coll and len(jets_coll[jn][iev].pt) > 0:
            ax.scatter(ak.to_numpy(jets_coll[jn][iev].eta),
                       ak.to_numpy(jets_coll[jn][iev].phi),
                       s=5*ak.to_numpy(jets_coll[jn][iev].pt),
                       alpha=0.5, label=jl, marker=jm)
    ax.legend(ncols=2); ax.set_xlabel("eta"); ax.set_ylabel("phi")
    ax.set_xlim(-6,6); ax.set_ylim(-5,5); ax.grid(alpha=0.3)
    ax.set_title(f"Event Display (event {iev})")
    _savefig(fig, os.path.join(plot_dir, f"event_display_{iev}.png"))



def _dispatch(tasks, worker_fn, desc, max_workers=None):
    max_workers = max_workers or min(max(len(tasks),1), os.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker_fn, t) for t in tasks]
        for f in tqdm.tqdm(as_completed(futs), total=len(futs), desc=f"  {desc}"):
            result = f.result()
            if isinstance(result, str): print(f"      {result}")


def plot_pid_distributions_parallel(arrs_flat, plot_dir):
    pids = np.unique(arrs_flat["ytarget"]["pid"])
    pids = pids[pids > 0].astype(int)
    pt_tasks, eta_tasks = [], []
    for v in pids:
        py_msk = (arrs_flat["pythia"]["pid"] == v) & (arrs_flat["pythia"]["pt"] > 1)
        tg_msk = (arrs_flat["ytarget"]["pid"] == v) & (arrs_flat["ytarget"]["pt"] > 1)
        pt_tasks.append((int(v), arrs_flat["pythia"]["pt"][py_msk].copy(),
                         arrs_flat["ytarget"]["pt"][tg_msk].copy(), plot_dir))
        eta_tasks.append((int(v), arrs_flat["pythia"]["eta"][py_msk].copy(),
                          arrs_flat["ytarget"]["eta"][tg_msk].copy(), plot_dir))
    _dispatch(pt_tasks,  _worker_pid_pt_plot,  "PID pT histograms")
    _dispatch(eta_tasks, _worker_pid_eta_plot, "PID eta histograms")


def _worker_soft_pu(args):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, mplhep; mplhep.style.use("CMS")
    label, pt_lo, pt_hi, wscale, py_sumpt, tg_weighted, plot_dir = args
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(py_sumpt, tg_weighted, s=4, alpha=0.3, rasterized=True)
    lim = max(py_sumpt.max(), tg_weighted.max()) * 1.05 if len(py_sumpt) else 100
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5)
    ax.set_xlabel("Pythia sum pT (GeV)")
    ax.set_ylabel(f"Target sum(pT * {wscale}*(1-isPU)) (GeV)")
    ax.set_title(f"Soft PU subtraction  {label}"); ax.grid(alpha=0.3)
    try: fig.tight_layout()
    except Exception: pass
    fig.savefig(os.path.join(plot_dir, f"soft_pu_{label.replace(' ','_')}.png"), dpi=150)
    plt.close(fig)


def plot_soft_pu_parallel(arrs_awk, plot_dir):
    eta_min, eta_max = 0.0, 5.0
    configs = [("0.5<pT<2GeV", 0.5, 2.0, 1.3), ("pT>5GeV", 5.0, 1e9, 1.0)]

    tasks = []
    for label, pt_lo, pt_hi, wscale in configs:
        py_msk = ((np.abs(arrs_awk["pythia"]["eta"]) >= eta_min) &
                  (np.abs(arrs_awk["pythia"]["eta"]) <  eta_max) &
                  (arrs_awk["pythia"]["pt"] > pt_lo) &
                  (arrs_awk["pythia"]["pt"] < pt_hi))
        tg_msk = ((np.abs(arrs_awk["ytarget"]["eta"]) >= eta_min) &
                  (np.abs(arrs_awk["ytarget"]["eta"]) <  eta_max) &
                  (arrs_awk["ytarget"]["pt"] > pt_lo) &
                  (arrs_awk["ytarget"]["pt"] < pt_hi))
        py_sumpt    = ak.to_numpy(ak.sum(arrs_awk["pythia"]["pt"][py_msk], axis=1)).astype(np.float32)
        tg_pt       = arrs_awk["ytarget"]["pt"][tg_msk]
        tg_ispu     = arrs_awk["ytarget"]["ispu"][tg_msk]
        tg_weighted = ak.to_numpy(ak.sum(tg_pt * (wscale*(1.0 - tg_ispu)), axis=1)).astype(np.float32)
        tasks.append((label, pt_lo, pt_hi, wscale, py_sumpt, tg_weighted, plot_dir))
    _dispatch(tasks, _worker_soft_pu, "soft PU subtraction")


def plot_jet_response_single(jets_coll, plot_dir):
    if "cmssw" not in jets_coll or "ytarget" not in jets_coll:
        return
    print("   Matching jets for response plot...")

    ref_pt_tgt, tgt_pt = match_jet_collections(jets_coll, "cmssw", "ytarget", dR_max=0.1)
    print(f"      cmssw <-> ytarget : {len(ref_pt_tgt)} matched pairs")

    ref_pt_cand, cand_pt = np.array([]), np.array([])
    if "ycand" in jets_coll:
        ref_pt_cand, cand_pt = match_jet_collections(jets_coll, "cmssw", "ycand", dR_max=0.1)
        print(f"      cmssw <-> ycand   : {len(ref_pt_cand)} matched pairs")

    if not len(ref_pt_tgt):
        print("   No matched jet pairs, skipping response plot.")
        return

    b = np.linspace(0.5, 1.5, 100)
    fig, ax = plt.subplots(figsize=(8, 6))

    ratio_tgt = tgt_pt / ref_pt_tgt
    ax.hist(ratio_tgt, bins=b, histtype="bar", lw=1, label="MLPF target")

    if len(ref_pt_cand):
        ax.hist(cand_pt / ref_pt_cand, bins=b, histtype="step", lw=2, label="PF")

    ax.axvline(1.0, color="black", ls="--", lw=0.5)
    ax.set_xlabel("jet $p_T$ / genjet $p_T$")
    ax.set_ylabel("Counts")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(alpha=0.3, ls="--")
    cms_label(ax)
    #sample_label(ax, SAMPLE_NAME)

    mean_r = float(np.mean(ratio_tgt)); std_r = float(np.std(ratio_tgt))
    print(f"   Jet response (target): mean={mean_r:.3f}  std={std_r:.3f}")

    try: fig.tight_layout()
    except Exception: pass
    fig.savefig(os.path.join(plot_dir, "jet_response_total.png"), dpi=150)
    plt.close(fig)


def plot_overall_pt_distribution(arrs_awk, plot_dir):
    b     = np.logspace(-3, 4, 500)
    py_pt = ak.to_numpy(ak.flatten(arrs_awk["pythia"]["pt"])).astype(float)
    nopu  = arrs_awk["ytarget"]["ispu"] < 0.5
    tg_pt = ak.to_numpy(ak.flatten(arrs_awk["ytarget"]["pt"][nopu])).astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(py_pt, bins=b, histtype="step", lw=2, label="Pythia")
    ax.hist(tg_pt, bins=b, histtype="step", lw=2, label="MLPF target (ispu<0.5)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Particle $p_T$ (GeV)"); ax.set_ylabel("Counts")
    ax.legend(loc="best", fontsize=12); ax.grid(alpha=0.3)
    cms_label(ax)
    #sample_label(ax, SAMPLE_NAME)
    _savefig(fig, os.path.join(plot_dir, "overall_pt_distribution.png"))


def plot_sumpt_per_event(arrs_awk, plot_dir):
    # PID MAP 
    SUMPT_PID_MAP = {
        "pion":     [211],
        "kaon":     [321, 130, 310],
        "muon":     [13],
        "electron": [11],
        "photon":   [22],
        "all":      None,              # None = no PID filter
    }

    # GROUPS TO PLOT 
    SUMPT_GROUPS = [
        ["all"],
        ["pion"],
        ["kaon"],
        ["muon"],
        ["electron"],
        ["photon"],
        ["pion", "kaon"],           
        ["photon", "electron"],       
        ["photon", "electron","pion"],
        ["photon", "electron","pion","kaon"],
    ]

    def _get_pids(pid_names):
        for name in pid_names:
            if SUMPT_PID_MAP.get(name) is None:
                return None  # all particles
        pids = []
        for name in pid_names:
            pids.extend(SUMPT_PID_MAP.get(name, []))
        return pids

    def _pid_mask_awk(arr_pid, pids):
        if pids is None:
            return ak.ones_like(arr_pid, dtype=bool)
        mask = ak.zeros_like(arr_pid, dtype=bool)
        for pid in pids:
            mask = mask | (arr_pid == pid)
        return mask

    def _sumpt_awk(pt_arr, pid_arr, ispu_arr, pids):
        mask = _pid_mask_awk(pid_arr, pids) & (ispu_arr < 0.5)
        return ak.to_numpy(ak.sum(pt_arr[mask], axis=1)).astype(np.float32)

    def _sumpt_pythia(pt_arr, pid_arr, pids):
        mask = _pid_mask_awk(pid_arr, pids)
        return ak.to_numpy(ak.sum(pt_arr[mask], axis=1)).astype(np.float32)

    def _make_hist_plot(py_sumpt, tg_sumpt, label, title, fname):
        all_vals = np.concatenate([py_sumpt[py_sumpt > 0], tg_sumpt[tg_sumpt > 0]])
        if len(all_vals) < 2:
            print(f"     Skipping {title} no entries found")
            return
        lo = max(np.log10(np.percentile(all_vals,  1)), -1)
        hi = min(np.log10(np.percentile(all_vals, 99)) + 0.5, 6)
        b  = np.logspace(lo, hi, 100)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.hist(py_sumpt, bins=b, histtype="step", lw=2, label="Pythia")
        ax.hist(tg_sumpt, bins=b, histtype="step", lw=2, label="MLPF target (ispu<0.5)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(rf"$\sum p_T$ per event {title} (GeV)")
        ax.set_ylabel("Events / bin")
        ax.legend(loc="best", fontsize=12); ax.grid(alpha=0.3)
        cms_label(ax)
        #sample_label(ax, SAMPLE_NAME)
        _savefig(fig, os.path.join(plot_dir, fname))
        print(f"     Pythia mean={py_sumpt.mean():.1f}  target(noPU) mean={tg_sumpt.mean():.1f} GeV")

    # ---- loop over groups ----
    for group in SUMPT_GROUPS:
        if len(group) == 1:
            # single PID just one plot
            name  = group[0]
            pids  = _get_pids(group)
            title = name.capitalize()
            label = name
            py_s  = _sumpt_pythia(arrs_awk["pythia"]["pt"],
                                   arrs_awk["pythia"]["pid"], pids)
            tg_s  = _sumpt_awk(arrs_awk["ytarget"]["pt"],
                                arrs_awk["ytarget"]["pid"],
                                arrs_awk["ytarget"]["ispu"], pids)
            print(f"   [{title}]")
            _make_hist_plot(py_s, tg_s, label, title,
                            f"sumpt_per_event_{label}_dist.png")

        else:
            # multi-PID group: individual plot per PID + one combined
            print(f"   [{' + '.join(g.capitalize() for g in group)}]")

            # individual plots first
            for name in group:
                pids  = _get_pids([name])
                title = name.capitalize()
                py_s  = _sumpt_pythia(arrs_awk["pythia"]["pt"],
                                       arrs_awk["pythia"]["pid"], pids)
                tg_s  = _sumpt_awk(arrs_awk["ytarget"]["pt"],
                                    arrs_awk["ytarget"]["pid"],
                                    arrs_awk["ytarget"]["ispu"], pids)
                print(f"     individual: {title}")
                _make_hist_plot(py_s, tg_s, name, title,
                                f"sumpt_per_event_{name}_dist.png")

            # combined plot
            pids_combined = _get_pids(group)
            title_combined = " + ".join(g.capitalize() for g in group)
            label_combined = "_".join(group)
            py_s  = _sumpt_pythia(arrs_awk["pythia"]["pt"],
                                   arrs_awk["pythia"]["pid"], pids_combined)
            tg_s  = _sumpt_awk(arrs_awk["ytarget"]["pt"],
                                arrs_awk["ytarget"]["pid"],
                                arrs_awk["ytarget"]["ispu"], pids_combined)
            print(f"     combined: {title_combined}")
            _make_hist_plot(py_s, tg_s, label_combined, title_combined,
                            f"sumpt_per_event_{label_combined}_dist.png")

def plot_jet_response_loglog(jets_coll, plot_dir):
    if "cmssw" not in jets_coll or "ytarget" not in jets_coll:
        return
    print("   Matching jets for log-log response plot...")

    ref_tgt,  tgt_pt  = match_jet_collections(jets_coll, "cmssw", "ytarget",      dR_max=0.1)
    ref_nopu, nopu_pt = match_jet_collections(jets_coll, "cmssw", "ytarget_nopu", dR_max=0.1)
    ref_cand, cand_pt = np.array([]), np.array([])
    if "ycand" in jets_coll:
        ref_cand, cand_pt = match_jet_collections(jets_coll, "cmssw", "ycand", dR_max=0.1)

    b = np.logspace(-1, 1, 600)
    fig, ax = plt.subplots(figsize=(8, 7))
    if len(ref_cand):
        ax.hist(cand_pt / ref_cand, bins=b, histtype="step", lw=1, label="PF")
    if len(ref_tgt):
        ax.hist(tgt_pt  / ref_tgt,  bins=b, histtype="step", lw=1, label="MLPF target")
    #if len(ref_nopu):
    #    ax.hist(nopu_pt / ref_nopu, bins=b, histtype="step", lw=1, label="MLPF target, no PU")
    ax.axvline(1.0, color="black", ls="--", lw=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("jet $p_T$ / genjet $p_T$"); ax.set_ylabel("Counts")
    ax.legend(loc="upper left", fontsize=12); ax.grid(alpha=0.3, ls="--")
    cms_label(ax)
    #sample_label(ax, SAMPLE_NAME)
    _savefig(fig, os.path.join(plot_dir, "jet_response_loglog.png"))



def print_electron_elem_type_diagnostic(arrs_flat):
    msk = (arrs_flat["ytarget"]["pid"] == 11) & (arrs_flat["ytarget"]["pt"] > 5)
    types, counts = np.unique(arrs_flat["Xelem"]["typ"][msk], return_counts=True)
    total = counts.sum()
    print("\n   Electron element type breakdown (pid==11, pT > 5 GeV):")
    for t, n in zip(types, counts):
        print(f"      Type {int(t)} ({TICL_TYPE_NAMES.get(int(t), '?')}): {n}  ({100*n/total:.1f}%)")


def plot_element_plots_parallel(arrs_flat, plot_dir):
    elem_types = [t for t in [1, 2, 4]
                  if np.any(arrs_flat["Xelem"]["typ"] == t)]
    match_tasks, ratio_tasks = [], []
    for et in elem_types:
        msk  = arrs_flat["Xelem"]["typ"] == et
        en   = arrs_flat["Xelem"]["energy"][msk].copy()
        gpid = arrs_flat["ytarget"]["pid"][msk].copy()
        cpid = arrs_flat["ycand"]["pid"][msk].copy()
        xpt  = arrs_flat["Xelem"]["pt"][msk].copy()
        tpt  = arrs_flat["ytarget"]["pt"][msk].copy()
        cpt  = arrs_flat["ycand"]["pt"][msk].copy()
        match_tasks.append((int(et), en, gpid, cpid, int(msk.sum()), plot_dir))
        ratio_tasks.append((int(et), xpt, gpid, tpt, cpid, cpt, plot_dir))
    _dispatch(match_tasks, _worker_elem_matching, "element matching")
    _dispatch(ratio_tasks, _worker_elem_ptratio,  "element pT ratio")


# Main
def main():
    print("="*60); print("MLPF CMS Analysis - OPTIMISED v5"); print("="*60)

    file_pattern = "/afs/cern.ch/work/m/moanwar/private/mlpf/particleflow/mlpf/data/cms/raw/*.pkl"
    files = sorted(glob.glob(file_pattern))[:50]
    print(f"\nFound {len(files)} files")
    if not files: print("ERROR: no files found"); return

    print("\n1. Loading files in parallel...")
    ticl_data = load_files_parallel(files)
    print(f"   Loaded {len(ticl_data)} events")
    if not ticl_data: print("ERROR: no data"); return

    print("\n2. Vectorised conversion...")
    arrs_awk, arrs_flat, genmet_arr, genjet_cmssw = convert_events_vectorised(ticl_data)
    print(f"   ytarget (flat, unfiltered): {len(arrs_flat['ytarget']['pt'])}")
    print(f"   ytarget (awk, pid!=0):      {int(ak.sum(ak.num(arrs_awk['ytarget']['pt'])))}")
    print(f"   Events: {len(arrs_awk['ytarget']['pt'])}")

    print("\n3. Jet clustering...")
    jets_coll = cluster_all(arrs_awk, genjet_cmssw)

    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_dir = os.path.abspath(PLOT_DIR)

    print("\n4. Per-PID plots: pT + eta histograms...")
    plot_pid_distributions_parallel(arrs_flat, plot_dir)

    print("\n5. PU fraction plots...")
    plot_pu_fraction(arrs_flat, plot_dir)
    plot_pu_fraction_vs_pt(arrs_flat, plot_dir)
    plot_pu_fraction_vs_eta(arrs_flat, plot_dir)

    print("\n6. MET plots...")
    plot_met(arrs_awk, genmet_arr, plot_dir)

    print("\n7. Soft PU subtraction scatter plots...")
    plot_soft_pu_parallel(arrs_awk, plot_dir)

    print("\n8. Jet distributions...")
    plot_jet_distributions(jets_coll, plot_dir)

    print("\n9. Overall pT distribution (all PIDs)...")
    plot_overall_pt_distribution(arrs_awk, plot_dir)

    print("\n9b. Sum-pT per event (Pythia vs target vs PF)...")
    plot_sumpt_per_event(arrs_awk, plot_dir)

    print("\n10. Jet response (linear, target + PF)...")
    plot_jet_response_single(jets_coll, plot_dir)

    print("\n11. Jet response (log-log, 3 collections)...")
    plot_jet_response_loglog(jets_coll, plot_dir)

    print("\n12. Element matching + pT ratio plots (types 1,2,3,4,5,6)...")
    plot_element_plots_parallel(arrs_flat, plot_dir)

    print("\n13. Electron element type diagnostic...")
    print_electron_elem_type_diagnostic(arrs_flat)

    print("\n14. Event display...")
    plot_event_display(arrs_awk, jets_coll, plot_dir, iev=2)

    print(f"\n{'='*60}\nDone! Plots saved to {plot_dir}/  (steps 1-14)\n{'='*60}")


if __name__ == "__main__":
    main()
