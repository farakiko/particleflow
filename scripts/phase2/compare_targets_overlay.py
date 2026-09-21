#!/usr/bin/env python3
"""Overlay v1 (colleague) and v2 (run3style) targets on the SAME axes, using
moanwar's conversion/clustering/matching machinery verbatim (mlpf_ticl_analysis_plots).
Gen reference = the unfiltered pythia stored in the v2 pkls (same events for both;
NB the v1 pkls' own pythia is gen-filtered, so it is not used as reference here).

  pixi run python3 scripts/phase2/compare_targets_overlay.py \
      --v1-glob '...pkl_v1_local/*.pkl' --v2-glob '...pkl_links/*.pkl' \
      --outdir plots/phase2/target_validation/overlay [--max-files 50]
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH_CWD = os.getcwd()
os.chdir(HERE)
sys.path.insert(0, HERE)
import mlpf_ticl_analysis_plots as M  # noqa: E402
from run_target_validation import align_ycand, backfill_genmet  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import mplhep  # noqa: E402

ak = M.ak
mplhep.style.use("CMS")

PID_NAMES = {211: "charged hadron", 130: "neutral hadron", 22: "photon", 11: "electron", 13: "muon"}
C = {"gen": "black", "v1": "tab:orange", "v2": "tab:blue"}
LBL = {"v1": "v1", "v2": "v2", "gen": "gen"}  # legend names; overridden by --label1/--label2


def _from_launch(p):
    return p if os.path.isabs(p) else os.path.join(LAUNCH_CWD, p)


def load_version(pattern, max_files):
    files = sorted(glob.glob(_from_launch(pattern)))[:max_files]
    assert files, f"no files for {pattern}"
    data = [align_ycand(backfill_genmet(ev)) for ev in M.load_files_parallel(files)]
    arrs_awk, arrs_flat, genmet, genjet_cmssw = M.convert_events_vectorised(data)
    return arrs_awk, arrs_flat, genmet, genjet_cmssw


def _sv(fig, outdir, name):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(os.path.join(outdir, name), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-glob", required=True)
    ap.add_argument("--v2-glob", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--label1", default="v1", help="legend label for the --v1-glob target")
    ap.add_argument("--label2", default="v2", help="legend label for the --v2-glob target")
    ap.add_argument("--gen-eta-lo", type=float, default=1.5,
                    help="gen reference restricted to gen_eta_lo<|eta|<gen_eta_hi in particle-level "
                         "panels (targets are endcap-only; full-detector gen would be ~3.6x overpopulated)")
    ap.add_argument("--gen-eta-hi", type=float, default=3.0)
    a = ap.parse_args()
    global LBL
    LBL = {"v1": a.label1, "v2": a.label2, "gen": "gen"}
    outdir = _from_launch(a.outdir)
    os.makedirs(outdir, exist_ok=True)

    print("loading v1...")
    awk1, flat1, _, genjet1 = load_version(a.v1_glob, a.max_files)
    print("loading v2...")
    awk2, flat2, _, genjet2 = load_version(a.v2_glob, a.max_files)

    # gen reference for particle-level panels: endcap window matched to the targets
    # (response panels keep the FULL cmssw genjet list; matching restricts them there)
    gec = (abs(awk2["pythia"]["eta"]) > a.gen_eta_lo) & (abs(awk2["pythia"]["eta"]) < a.gen_eta_hi)
    pyec = {f: awk2["pythia"][f][gec] for f in ["pt", "eta", "phi", "energy", "pid"]}
    gec_flat = (np.abs(flat2["pythia"]["eta"]) > a.gen_eta_lo) & (np.abs(flat2["pythia"]["eta"]) < a.gen_eta_hi)
    gen_note = f"$({a.gen_eta_lo}<|\\eta|<{a.gen_eta_hi})$"

    # ---- jets: cluster both targets + endcap gen from v2 pythia
    print("clustering...")
    jets = {
        "cmssw": genjet2,  # CMSSW genjets, identical content in both files
        "v1": M.cluster_jets_batch(awk1["ytarget"]["pt"], awk1["ytarget"]["eta"],
                                   awk1["ytarget"]["phi"], awk1["ytarget"]["energy"]),
        "v2": M.cluster_jets_batch(awk2["ytarget"]["pt"], awk2["ytarget"]["eta"],
                                   awk2["ytarget"]["phi"], awk2["ytarget"]["energy"]),
        "gen": M.cluster_jets_batch(pyec["pt"], pyec["eta"], pyec["phi"], pyec["energy"]),
    }

    # endcap-only reference (moanwar's convention: his stored genjets are 1.5<|eta|<3.2)
    ec = (abs(jets["cmssw"].eta) > 1.5) & (abs(jets["cmssw"].eta) < 3.2)
    jets["cmssw_ec"] = jets["cmssw"][ec]

    # ---- 1. jet response overlay (his matching, ref = CMSSW genjets)
    matches = {k: M.match_jet_collections(jets, "cmssw", k, dR_max=0.1) for k in ["v1", "v2"]}
    matches_ec = {k: M.match_jet_collections(jets, "cmssw_ec", k, dR_max=0.1) for k in ["v1", "v2"]}
    fig, ax = plt.subplots(figsize=(10, 7))
    b = np.linspace(0, 2, 101)
    for k in ["v1", "v2"]:
        rp, tp = matches[k]
        r = tp / rp
        ax.hist(r, bins=b, histtype="step", lw=2, color=C[k], density=True,
                label=f"{LBL[k]} target  (n={len(r)}, mean={r.mean():.3f}, med={np.median(r):.3f}, std={r.std():.3f})")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("target jet $p_T$ / CMSSW genjet $p_T$  ($\\Delta R<0.1$)")
    ax.set_ylabel("density")
    ax.legend(fontsize=13)
    M.cms_label(ax)
    _sv(fig, outdir, "overlay_jet_response.png")

    # ---- 1b. response vs genjet pT: median + IQR band
    fig, ax = plt.subplots(figsize=(10, 7))
    ptbins = np.array([3, 5, 10, 20, 40, 60, 80, 100, 150, 200, 400], float)
    centers = np.sqrt(ptbins[:-1] * ptbins[1:])
    for k in ["v1", "v2"]:
        rp, tp = matches[k]
        r = tp / rp
        med, lo, hi = [], [], []
        for a_, b_ in zip(ptbins[:-1], ptbins[1:]):
            m = (rp >= a_) & (rp < b_)
            q = np.percentile(r[m], [25, 50, 75]) if m.sum() >= 10 else [np.nan] * 3
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        ax.plot(centers, med, "o-", color=C[k], lw=2, label=f"{LBL[k]} median")
        ax.fill_between(centers, lo, hi, color=C[k], alpha=0.18, label=f"{LBL[k]} IQR")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("CMSSW genjet $p_T$ (GeV)")
    ax.set_ylabel("target jet $p_T$ / genjet $p_T$")
    ax.legend(fontsize=12, ncol=2)
    M.cms_label(ax)
    _sv(fig, outdir, "overlay_jet_response_vs_pt.png")

    # ---- 1c. same two response plots with ENDCAP-ONLY reference jets
    fig, ax = plt.subplots(figsize=(10, 7))
    b = np.linspace(0, 2, 101)
    for k in ["v1", "v2"]:
        rp, tp = matches_ec[k]
        r = tp / rp
        ax.hist(r, bins=b, histtype="step", lw=2, color=C[k], density=True,
                label=f"{LBL[k]}  (n={len(r)}, mean={r.mean():.3f}, med={np.median(r):.3f}, std={r.std():.3f})")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("target jet $p_T$ / genjet $p_T$   (ref: $1.5<|\\eta|<3.2$ genjets)")
    ax.set_ylabel("density")
    ax.legend(fontsize=12)
    M.cms_label(ax)
    _sv(fig, outdir, "overlay_jet_response_endcapref.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    for k in ["v1", "v2"]:
        rp, tp = matches_ec[k]
        r = tp / rp
        med, lo, hi = [], [], []
        for a_, b_ in zip(ptbins[:-1], ptbins[1:]):
            m = (rp >= a_) & (rp < b_)
            q = np.percentile(r[m], [25, 50, 75]) if m.sum() >= 10 else [np.nan] * 3
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
        ax.plot(centers, med, "o-", color=C[k], lw=2, label=f"{LBL[k]} median")
        ax.fill_between(centers, lo, hi, color=C[k], alpha=0.18, label=f"{LBL[k]} IQR")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("CMSSW genjet $p_T$ (GeV)   (ref: $1.5<|\\eta|<3.2$)")
    ax.set_ylabel("target jet $p_T$ / genjet $p_T$")
    ax.legend(fontsize=12, ncol=2)
    M.cms_label(ax)
    _sv(fig, outdir, "overlay_jet_response_vs_pt_endcapref.png")

    # ---- 2. jet pt + eta spectra overlay
    for var, bins, xl in [("pt", np.logspace(np.log10(3), 3, 60), "jet $p_T$ (GeV)"),
                          ("eta", np.linspace(-5, 5, 101), "jet $\\eta$")]:
        fig, ax = plt.subplots(figsize=(10, 7))
        for k, lbl in [("cmssw", f"CMSSW genjets {gen_note}"), ("gen", f"pythia jets {gen_note}"),
                       ("v1", f"{LBL['v1']} target jets"), ("v2", f"{LBL['v2']} target jets")]:
            jv = jets[k]
            if k == "cmssw":  # full-detector list -> restrict the CURVE to the endcap window
                jv = jv[(abs(jv.eta) > a.gen_eta_lo) & (abs(jv.eta) < a.gen_eta_hi)]
            vals = ak.to_numpy(ak.flatten(getattr(jv, var)))
            ax.hist(vals, bins=bins, histtype="step", lw=2,
                    color=C.get(k, "gray"), ls="--" if k == "cmssw" else "-",
                    label=f"{lbl} (n={len(vals)})")
        if var == "pt":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("jets / bin")
        ax.legend(fontsize=12)
        M.cms_label(ax)
        _sv(fig, outdir, f"overlay_jet_{var}.png")

    # ---- 3. per-PID particle pt + eta spectra (gen restricted to the endcap window)
    for pid, pname in PID_NAMES.items():
        gm = (flat2["pythia"]["pid"] == pid) & gec_flat
        m1 = flat1["ytarget"]["pid"] == pid
        m2 = flat2["ytarget"]["pid"] == pid
        for var, bins, xl in [("pt", np.logspace(-2, 3, 80), f"{pname} $p_T$ (GeV)"),
                              ("eta", np.linspace(-5, 5, 101), f"{pname} $\\eta$")]:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.hist(flat2["pythia"][var][gm], bins=bins, histtype="step", lw=2, color=C["gen"],
                    ls="--", label=f"pythia {gen_note} (n={int(gm.sum())})")
            ax.hist(flat1["ytarget"][var][m1], bins=bins, histtype="step", lw=2, color=C["v1"],
                    label=f"v1 target (n={int(m1.sum())})")
            ax.hist(flat2["ytarget"][var][m2], bins=bins, histtype="step", lw=2, color=C["v2"],
                    label=f"v2 target (n={int(m2.sum())})")
            if var == "pt":
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel("particles / bin")
            ax.legend(fontsize=12)
            M.cms_label(ax)
            _sv(fig, outdir, f"overlay_{var}_pid{pid}.png")

    # ---- 4. per-event sum-pT ratio to gen (endcap-window gen)
    sums = {
        "gen": ak.to_numpy(ak.sum(pyec["pt"], axis=1)),
        "v1": ak.to_numpy(ak.sum(awk1["ytarget"]["pt"], axis=1)),
        "v2": ak.to_numpy(ak.sum(awk2["ytarget"]["pt"], axis=1)),
    }
    fig, ax = plt.subplots(figsize=(10, 7))
    b = np.linspace(0, 2, 101)
    ok = sums["gen"] > 0
    for k in ["v1", "v2"]:
        r = sums[k][ok] / sums["gen"][ok]
        ax.hist(r, bins=b, histtype="step", lw=2, color=C[k], density=True,
                label=f"{LBL[k]}  (med={np.median(r):.3f})")
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel(f"$\\Sigma p_T$(target) / $\\Sigma p_T$(pythia {gen_note}) per event")
    ax.set_ylabel("density")
    ax.legend(fontsize=13)
    M.cms_label(ax)
    _sv(fig, outdir, "overlay_sumpt_ratio.png")

    # ---- 5. particle multiplicity per event
    fig, ax = plt.subplots(figsize=(10, 7))
    b = np.linspace(0, 250, 126)
    for k, arr in [("gen", ak.num(pyec["pt"])), ("v1", ak.num(awk1["ytarget"]["pt"])),
                   ("v2", ak.num(awk2["ytarget"]["pt"]))]:
        vals = ak.to_numpy(arr)
        lbl = f"gen {gen_note}" if k == "gen" else LBL[k]
        ax.hist(vals, bins=b, histtype="step", lw=2, color=C[k],
                ls="--" if k == "gen" else "-", label=f"{lbl} (mean={vals.mean():.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("particles / event")
    ax.set_ylabel("events / bin")
    ax.legend(fontsize=13)
    M.cms_label(ax)
    _sv(fig, outdir, "overlay_multiplicity.png")

    # ---- 6. element truth-matched fraction vs E (tracks, hadronic tracksters)
    for typ, tname in [(1, "tracks"), (4, "hadronic tracksters")]:
        fig, ax = plt.subplots(figsize=(10, 7))
        bins = np.logspace(-1, 3, 40)
        for k, fl in [("v1", flat1), ("v2", flat2)]:
            msk = fl["Xelem"]["typ"] == typ
            if not msk.sum():
                continue
            en = fl["Xelem"]["energy"][msk]
            matched = fl["ytarget"]["pid"][msk] != 0
            frac = [float(matched[(en >= lo) & (en < hi)].mean()) if ((en >= lo) & (en < hi)).sum() else np.nan
                    for lo, hi in zip(bins[:-1], bins[1:])]
            ax.plot(bins[:-1], frac, ".-", lw=2, color=C[k],
                    label=f"{LBL[k]} ({int(msk.sum())} elements, {100*matched.mean():.1f}% matched)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.1)
        ax.set_xlabel(f"{tname}: element energy (GeV)")
        ax.set_ylabel("truth-matched fraction")
        ax.legend(fontsize=12)
        M.cms_label(ax)
        _sv(fig, outdir, f"overlay_elem_matched_type{typ}.png")

    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()
