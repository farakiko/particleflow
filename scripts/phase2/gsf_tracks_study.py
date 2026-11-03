"""GSF-track truth study on Phase-2 NanoAOD (0 PU): are the GSF tracks real electrons?

Question: the GSF tracks stored in our NanoAOD (`GSFTrack_*`) may be largely fake, which would
confuse MLPF. This script measures, per physics sample, how often a GSF track is truth-linked to
an electron SimTICLCandidate (efficiency, truth -> reco) and how often it is not (fake rate,
reco -> truth), and what the non-electron GSF tracks are instead.

Truth link: `SimTICLCandidates_gsftrackIdx` (index into `GSFTrack`; < 0 = no GSF track). Checked
in the data: it equals the first entry of the multi-valued `SimTICLCandidatesGsfTrackIdxs` table and
every SimTICLCandidate links at most one GSF track. A GSF track is almost always linked from at most one
SimTICLCandidate (ttbar/Zll: always; QCD: 8 of 58k linked GSF tracks have two charged-hadron owners);
when it has several, it counts as electron-linked if any owner is an electron (count in summary.json).

Region: HGCAL endcap, 1.5 < |eta| < 3.0, pT > 1 GeV. SimTICLCandidates are incomplete in the barrel,
so a barrel GSF track without a truth link is not evidence of a fake.

Definitions (all in the endcap acceptance; GSF kinematics = the "mode" pT/eta/phi):
  efficiency  = N(electron SimTICLCandidates with a linked GSF track) / N(electron SimTICLCandidates),
                selected and binned on the SimTICLCandidate pT/eta.
  fake rate   = N(GSF tracks NOT linked to an electron SimTICLCandidate) / N(GSF tracks), selected and
                binned on the GSF pT/eta; shown with the subset linked to no SimTICLCandidate at all.
  Only the truth links stored in the file are used; dR enters only the summary.json diagnostics.
  composition = fraction of GSF tracks linked to an electron / charged hadron / muon / nothing.

Usage:
  pixi run python3 scripts/phase2/gsf_tracks_study.py                     # ttbar_0pu + zll_0pu + qcd_0pu
  pixi run python3 scripts/phase2/gsf_tracks_study.py --max-files 5      # quick test
Outputs (PDF only): plots/GSF_tracks_study/<sample>/*.pdf, plus a summary.json of the key numbers.
"""

import argparse
import glob
import json
import os
from multiprocessing import Pool

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402
from scipy.stats import beta  # noqa: E402

ETA_MIN, ETA_MAX, PT_MIN = 1.5, 3.0, 1.0
DR_NEAR_GEN = 0.05  # diagnostics only: "is there a gen electron/photon near this unlinked GSF track?"
NO_MATCH = 9.0

SAMPLE_LABELS = {
    "ttbar_0pu": r"$t\bar{t}$",
    "zll_0pu": r"Z$\rightarrow\ell\ell$",
    "qcd_0pu": r"QCD multijet",
}

# truth categories of a GSF track = |pdgID| of its linked SimTICLCandidate (0 = no link); fixed CMS-palette colour each
CATEGORIES = [
    ("e", "Electron", lambda p: p == 11, "#5790fc"),
    ("had", "Charged hadron", lambda p: p == 211, "#f89c20"),
    ("mu", "Muon", lambda p: p == 13, "#e42536"),
    ("other", "Other", lambda p: (p != 0) & (p != 11) & (p != 211) & (p != 13), "#7a21dd"),
    ("none", "No SimTICLCandidate", lambda p: p == 0, "#964a8b"),
]

BLUE, RED = "#5790fc", "#e42536"  # CMS palette colours used for the curves
PT_BINS = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80, 150, 300], dtype=float)
ETA_BINS = np.round(np.arange(-ETA_MAX, ETA_MAX + 1e-6, 0.1), 2)  # signed eta, both endcaps; |eta| < ETA_MIN is outside the selection

BRANCHES = [
    "GSFTrack_ptMode",
    "GSFTrack_etaMode",
    "GSFTrack_phiMode",
    "GSFTrack_charge",
    "GSFTrack_nhits",
    "SimTICLCandidates_pdgID",
    "SimTICLCandidates_pt",
    "SimTICLCandidates_eta",
    "SimTICLCandidates_phi",
    "SimTICLCandidates_gsftrackIdx",
    "SimTICLCandidates_trackIdx",
    "SimTICLCandidates_nTrackIdxs",
    "SimTICLCandidatesTrackIdxs_trackIndex",
    "GeneralTrack_eta",
    "GeneralTrack_phi",
    "GeneralTrack_charge",
    "HGCalGenPart_pdgId",
    "HGCalGenPart_eta",
    "HGCalGenPart_phi",
]


def delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((eta1 - eta2) ** 2 + dphi**2)


def min_dr(a_eta, a_phi, b_eta, b_phi, b_mask=None):
    """Per object of collection a: dR to the closest object of collection b (NO_MATCH if none)."""
    a = ak.zip({"eta": a_eta, "phi": a_phi})
    b = ak.zip({"eta": b_eta, "phi": b_phi})
    if b_mask is not None:
        b = b[b_mask]
    pairs = ak.cartesian({"a": a, "b": b}, nested=True)
    dr = delta_r(pairs.a.eta, pairs.a.phi, pairs.b.eta, pairs.b.phi)
    return ak.flatten(ak.fill_none(ak.min(dr, axis=2), NO_MATCH)).to_numpy()


def argmin_dr_same_charge(a_eta, a_phi, a_q, b_eta, b_phi, b_q):
    """Per object of a: (dR, index in b) of the closest same-charge object of b."""
    a = ak.zip({"eta": a_eta, "phi": a_phi, "q": a_q})
    b = ak.zip({"eta": b_eta, "phi": b_phi, "q": b_q, "idx": ak.local_index(b_eta)})
    pairs = ak.cartesian({"a": a, "b": b}, nested=True)
    dr = ak.where(pairs.a.q == pairs.b.q, delta_r(pairs.a.eta, pairs.a.phi, pairs.b.eta, pairs.b.phi), NO_MATCH)
    best = ak.argmin(dr, axis=2, keepdims=True)
    dr_best = ak.flatten(ak.fill_none(ak.flatten(dr[best], axis=2), NO_MATCH)).to_numpy()
    idx_best = ak.flatten(ak.fill_none(ak.flatten(pairs.b.idx[best], axis=2), -1)).to_numpy()
    return dr_best, idx_best


def process_file(args):
    path, dr_match = args
    try:
        a = uproot.open(path)["Events"].arrays(BRANCHES)
    except Exception as e:  # a few NanoAOD files can be unreadable; report and skip
        return {"error": f"{path}: {e}"}
    nev = len(a)

    # ---- GSF track -> linked SimTICLCandidate |pdgID| (scatter the per-SimCand index into the flat GSF list) ----
    n_gsf = ak.num(a.GSFTrack_ptMode).to_numpy()
    gsf_offset = np.concatenate([[0], np.cumsum(n_gsf)[:-1]])
    sc_ev = np.repeat(np.arange(nev), ak.num(a.SimTICLCandidates_pdgID).to_numpy())
    sc_gsf = ak.flatten(a.SimTICLCandidates_gsftrackIdx).to_numpy()
    sc_pdg = np.abs(ak.flatten(a.SimTICLCandidates_pdgID).to_numpy())
    has = sc_gsf >= 0
    if np.any(sc_gsf[has] >= n_gsf[sc_ev[has]]):
        raise RuntimeError(f"{path}: gsftrackIdx out of range")
    gsf_global = gsf_offset[sc_ev[has]] + sc_gsf[has]
    n_multi_owner = int(np.sum(np.unique(gsf_global, return_counts=True)[1] > 1))
    gsf_link_pdg = np.zeros(n_gsf.sum(), dtype=np.int64)
    gsf_link_pdg[gsf_global] = sc_pdg[has]
    gsf_link_pdg[gsf_global[sc_pdg[has] == 11]] = 11  # several owners: electron wins if any owner is an electron

    # ---- GeneralTrack -> linked SimTICLCandidate |pdgID| (multi-valued table; first owner wins) ----
    n_trk = ak.num(a.GeneralTrack_eta).to_numpy()
    trk_offset = np.concatenate([[0], np.cumsum(n_trk)[:-1]])
    n_links = ak.flatten(a.SimTICLCandidates_nTrackIdxs).to_numpy()
    link_trk = ak.flatten(a.SimTICLCandidatesTrackIdxs_trackIndex).to_numpy().astype(np.int64)
    link_owner = np.repeat(np.arange(len(n_links)), n_links)
    link_ev = sc_ev[link_owner]
    trk_link_pdg = np.zeros(n_trk.sum(), dtype=np.int64)
    glob_trk = trk_offset[link_ev] + link_trk
    first = np.unique(glob_trk, return_index=True)[1]
    trk_link_pdg[glob_trk[first]] = sc_pdg[link_owner[first]]

    gsf_eta, gsf_phi, gsf_q = a.GSFTrack_etaMode, a.GSFTrack_phiMode, a.GSFTrack_charge
    sc_eta, sc_phi = a.SimTICLCandidates_eta, a.SimTICLCandidates_phi
    is_ele = np.abs(a.SimTICLCandidates_pdgID) == 11
    gen_pdg = a.HGCalGenPart_pdgId

    gsf = {
        "pt": ak.flatten(a.GSFTrack_ptMode).to_numpy(),
        "eta": ak.flatten(gsf_eta).to_numpy(),
        "nhits": ak.flatten(a.GSFTrack_nhits).to_numpy(),
        "link_pdg": gsf_link_pdg,
        "dr_ele_simcand": min_dr(gsf_eta, gsf_phi, sc_eta, sc_phi, b_mask=is_ele),
        "dr_gen_e": min_dr(gsf_eta, gsf_phi, a.HGCalGenPart_eta, a.HGCalGenPart_phi, b_mask=np.abs(gen_pdg) == 11),
        "dr_gen_gamma": min_dr(gsf_eta, gsf_phi, a.HGCalGenPart_eta, a.HGCalGenPart_phi, b_mask=gen_pdg == 22),
    }
    # duplicate check: dR to another GSF track that IS linked to an electron SimTICLCandidate
    gsf_is_e = ak.unflatten(gsf_link_pdg == 11, n_gsf)
    a_rec = ak.zip({"eta": gsf_eta, "phi": gsf_phi, "i": ak.local_index(gsf_eta)})
    b_rec = a_rec[gsf_is_e]
    pairs = ak.cartesian({"a": a_rec, "b": b_rec}, nested=True)
    dr = ak.where(pairs.a.i != pairs.b.i, delta_r(pairs.a.eta, pairs.a.phi, pairs.b.eta, pairs.b.phi), NO_MATCH)
    gsf["dr_other_e_gsf"] = ak.flatten(ak.fill_none(ak.min(dr, axis=2), NO_MATCH)).to_numpy()
    # overlap with a same-charge GeneralTrack, and that track's own truth link
    dr_trk, idx_trk = argmin_dr_same_charge(gsf_eta, gsf_phi, gsf_q, a.GeneralTrack_eta, a.GeneralTrack_phi, a.GeneralTrack_charge)
    gsf_ev = np.repeat(np.arange(nev), n_gsf)
    gsf["dr_gentrk"] = dr_trk
    gsf["gentrk_link_pdg"] = np.where(idx_trk >= 0, trk_link_pdg[trk_offset[gsf_ev] + np.maximum(idx_trk, 0)], -1)

    # ---- electron SimTICLCandidates (efficiency denominator) ----
    ele = {
        "pt": ak.flatten(a.SimTICLCandidates_pt[is_ele]).to_numpy(),
        "eta": ak.flatten(sc_eta[is_ele]).to_numpy(),
        "has_gsf": ak.flatten(a.SimTICLCandidates_gsftrackIdx[is_ele] >= 0).to_numpy(),
        "has_trk": ak.flatten(a.SimTICLCandidates_trackIdx[is_ele] >= 0).to_numpy(),
        "dr_gsf": min_dr(sc_eta[is_ele], sc_phi[is_ele], gsf_eta, gsf_phi),
    }
    return {"nev": nev, "gsf": gsf, "ele": ele, "dr_match": dr_match, "n_multi_owner": n_multi_owner}


def clopper_pearson(k, n, cl=0.6827):
    k, n = np.asarray(k, float), np.asarray(n, float)
    a = (1 - cl) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        lo = np.where(k > 0, beta.ppf(a, k, n - k + 1), 0.0)
        hi = np.where(k < n, beta.ppf(1 - a, k + 1, n - k), 1.0)
        r = np.where(n > 0, k / n, np.nan)
    return r, r - lo, hi - r


def binned_ratio(x, passed, bins):
    n, _ = np.histogram(x, bins)
    k, _ = np.histogram(x[passed], bins)
    return clopper_pearson(k, n), n


def cms_axes(sample, xlabel, ylabel, logx=False):
    fig, ax = plt.subplots(figsize=(10, 9))
    pileup = "200 PU" if sample.endswith("200pu") else "0 PU"
    mplhep.cms.label("Work in progress", data=False, rlabel=f"Phase-2, {pileup}", ax=ax, loc=0, fontsize=22)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    return fig, ax


def shade_excluded_barrel(ax):
    ax.axvspan(-ETA_MIN, ETA_MIN, color="grey", alpha=0.12, lw=0, zorder=0)
    ax.text(0, 0.04, rf"$|\eta| < {ETA_MIN}$" + "\nnot selected", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=16, color="dimgrey")


def annotate(ax, sample, selection):
    ax.text(0.04, 0.95, SAMPLE_LABELS.get(sample, sample), transform=ax.transAxes, fontsize=24, va="top")
    ax.text(0.04, 0.89, selection, transform=ax.transAxes, fontsize=19, va="top")


def plot_ratios(curves, bins, xlabel, ylabel, sample, selection, path, logx, shade=True, vlines=()):
    """curves: list of (label, x, passed, marker, color)."""
    fig, ax = cms_axes(sample, xlabel, ylabel, logx)
    centers = np.sqrt(bins[1:] * bins[:-1]) if logx else 0.5 * (bins[1:] + bins[:-1])
    xerr = [centers - bins[:-1], bins[1:] - centers]
    for i, (label, x, passed, marker, color) in enumerate(curves):
        (r, elo, ehi), n = binned_ratio(x, passed, bins)
        ok = n > 0
        # the first curve is drawn on top, so coincident points of a reference curve do not hide it
        ax.errorbar(
            centers[ok],
            r[ok],
            yerr=[elo[ok], ehi[ok]],
            xerr=[xerr[0][ok], xerr[1][ok]],
            fmt=marker,
            ms=9,
            capsize=0,
            lw=1.8,
            label=label,
            color=color,
            zorder=10 - i,
        )
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1.65)
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    if shade and not logx:
        shade_excluded_barrel(ax)
    for v in vlines:
        ax.axvline(v, color="grey", ls="--", lw=1)
    ax.legend(loc="upper left", fontsize=19, frameon=False, bbox_to_anchor=(0.01, 0.85))
    annotate(ax, sample, selection)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_composition(
    x, link_pdg, bins, xlabel, sample, selection, path, logx, categories=CATEGORIES, legend_title="GSF track linked to", shade=True, vlines=()
):
    fig, ax = cms_axes(sample, xlabel, "Fraction of GSF tracks", logx)
    n, _ = np.histogram(x, bins)
    fracs, labels, colors = [], [], []
    for _, label, sel, color in categories:
        k, _ = np.histogram(x[sel(link_pdg)], bins)
        if k.sum() == 0:
            continue
        fracs.append(np.divide(k, n, out=np.zeros(len(k)), where=n > 0))
        labels.append(label)
        colors.append(color)
    mplhep.histplot(fracs, bins, stack=True, histtype="fill", label=labels, color=colors, ax=ax, edgecolor="black", lw=0.6)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1.5 if len(fracs) <= 4 else 1.62)  # room for a third legend row
    if shade and not logx:
        shade_excluded_barrel(ax)
    for v in vlines:
        ax.axvline(v, color="black", ls="--", lw=1.2)
    ax.legend(loc="upper left", fontsize=18, frameon=False, ncol=2, bbox_to_anchor=(0.01, 0.85), title=legend_title, title_fontsize=18)
    annotate(ax, sample, selection)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def frac(k, n):
    r, elo, ehi = clopper_pearson(k, n)
    return {"value": float(r), "err_lo": float(elo), "err_hi": float(ehi), "k": int(k), "n": int(n)}


def run_sample(sample, args):
    files = sorted(glob.glob(os.path.join(args.nano_dir, sample, "*.root")))
    if args.max_files:
        files = files[: args.max_files]
    with Pool(args.nproc) as pool:
        results = pool.map(process_file, [(f, args.dr) for f in files])
    errors = [r["error"] for r in results if "error" in r]
    results = [r for r in results if "error" not in r]
    nev = sum(r["nev"] for r in results)
    gsf = {k: np.concatenate([r["gsf"][k] for r in results]) for k in results[0]["gsf"]}
    ele = {k: np.concatenate([r["ele"][k] for r in results]) for k in results[0]["ele"]}
    print(f"\n===== {sample}: {len(results)} files ({len(errors)} unreadable), {nev} events =====")
    for e in errors:
        print("  skipped:", e)

    outdir = os.path.join(args.outdir, sample)
    os.makedirs(outdir, exist_ok=True)
    dr = args.dr

    # ---- selections ----
    g_acc = (np.abs(gsf["eta"]) > ETA_MIN) & (np.abs(gsf["eta"]) < ETA_MAX) & (gsf["pt"] > PT_MIN)
    e_acc = (np.abs(ele["eta"]) > ETA_MIN) & (np.abs(ele["eta"]) < ETA_MAX) & (ele["pt"] > PT_MIN)
    G = {k: v[g_acc] for k, v in gsf.items()}
    E = {k: v[e_acc] for k, v in ele.items()}
    g_is_e = G["link_pdg"] == 11
    g_unlinked = G["link_pdg"] == 0

    # closure of the link: linked electron GSF tracks must sit on their electron
    closure = np.quantile(G["dr_ele_simcand"][g_is_e], [0.5, 0.9, 0.99]) if g_is_e.any() else [np.nan] * 3

    sel_txt = rf"${ETA_MIN} < |\eta| < {ETA_MAX}$, $p_\mathrm{{T}} > {PT_MIN:g}$ GeV"
    eta_lab_e = r"Electron SimTICLCandidate $\eta$"
    pt_lab_e = r"Electron SimTICLCandidate $p_\mathrm{T}$ [GeV]"
    eta_lab_g = r"GSF track $\eta$"
    pt_lab_g = r"GSF track $p_\mathrm{T}$ [GeV]"
    eff_passes = [("Linked GSF track", E["has_gsf"], "o", BLUE)]

    # ---- efficiency ----
    curves = [(lab, E["pt"], p, mk, c) for lab, p, mk, c in eff_passes]
    plot_ratios(curves, PT_BINS, pt_lab_e, "Efficiency", sample, sel_txt, f"{outdir}/efficiency_vs_pt.pdf", logx=True)
    for ptmin in args.eta_pt_min:
        m = E["pt"] > ptmin
        curves = [(lab, E["eta"][m], p[m], mk, c) for lab, p, mk, c in eff_passes]
        txt = rf"${ETA_MIN} < |\eta| < {ETA_MAX}$, $p_\mathrm{{T}} > {ptmin:g}$ GeV"
        suffix = "" if ptmin == PT_MIN else f"_ptgt{ptmin:g}"
        plot_ratios(curves, ETA_BINS, eta_lab_e, "Efficiency", sample, txt, f"{outdir}/efficiency_vs_eta{suffix}.pdf", logx=False)

    # ---- fake rate ----
    curves = [
        ("Not linked to an electron", G["pt"], ~g_is_e, "o", BLUE),
        ("Linked to no SimTICLCandidate", G["pt"], g_unlinked, "s", RED),
    ]
    plot_ratios(curves, PT_BINS, pt_lab_g, "Fake rate", sample, sel_txt, f"{outdir}/fakerate_vs_pt.pdf", logx=True)
    for ptmin in args.eta_pt_min:
        m = G["pt"] > ptmin
        ae = G["eta"][m]
        curves = [
            ("Not linked to an electron", ae, ~g_is_e[m], "o", BLUE),
            ("Linked to no SimTICLCandidate", ae, g_unlinked[m], "s", RED),
        ]
        txt = rf"${ETA_MIN} < |\eta| < {ETA_MAX}$, $p_\mathrm{{T}} > {ptmin:g}$ GeV"
        suffix = "" if ptmin == PT_MIN else f"_ptgt{ptmin:g}"
        plot_ratios(curves, ETA_BINS, eta_lab_g, "Fake rate", sample, txt, f"{outdir}/fakerate_vs_eta{suffix}.pdf", logx=False)

    # ---- composition ----
    plot_composition(G["pt"], G["link_pdg"], PT_BINS, pt_lab_g, sample, sel_txt, f"{outdir}/composition_vs_pt.pdf", logx=True)
    plot_composition(G["eta"], G["link_pdg"], ETA_BINS, eta_lab_g, sample, sel_txt, f"{outdir}/composition_vs_eta.pdf", logx=False)

    # ---- summary numbers ----
    U = g_unlinked
    trk_overlap = G["dr_gentrk"] < dr
    summary = {
        "sample": sample,
        "files": len(results),
        "unreadable_files": len(errors),
        "events": nev,
        "selection": f"{ETA_MIN} < |eta| < {ETA_MAX}, pT > {PT_MIN} GeV; dR match = {dr}",
        "gsf_per_event_all_eta": float(len(gsf["pt"]) / nev),
        "gsf_linked_from_several_simcands_all_eta": int(sum(r["n_multi_owner"] for r in results)),
        "gsf_fraction_in_selection": float(g_acc.mean()),
        "gsf_in_selection": int(g_acc.sum()),
        "electron_simcands_in_selection": int(e_acc.sum()),
        "link_closure_dR_q50_q90_q99": [float(c) for c in closure],
        "efficiency_gsf_link": frac(E["has_gsf"].sum(), len(E["pt"])),
        "efficiency_generaltrack_link": frac(E["has_trk"].sum(), len(E["pt"])),
        "efficiency_gsf_link_pt_gt_10": frac(E["has_gsf"][E["pt"] > 10].sum(), (E["pt"] > 10).sum()),
        "fakerate_link": frac((~g_is_e).sum(), len(G["pt"])),
        "fakerate_link_pt_gt_10": frac((~g_is_e)[G["pt"] > 10].sum(), (G["pt"] > 10).sum()),
        "composition": {key: frac(sel(G["link_pdg"]).sum(), len(G["pt"])) for key, _, sel, _ in CATEGORIES},
        "nhits_median_by_category": {key: float(np.median(G["nhits"][sel(G["link_pdg"])])) for key, _, sel, _ in CATEGORIES if sel(G["link_pdg"]).any()},
        "unlinked_diagnostics": {
            "n": int(U.sum()),
            "near_gen_electron_dR<0.05": frac((U & (G["dr_gen_e"] < DR_NEAR_GEN)).sum(), U.sum()),
            "near_gen_photon_dR<0.05": frac((U & (G["dr_gen_gamma"] < DR_NEAR_GEN)).sum(), U.sum()),
            "near_neither": frac((U & (G["dr_gen_e"] >= DR_NEAR_GEN) & (G["dr_gen_gamma"] >= DR_NEAR_GEN)).sum(), U.sum()),
            f"duplicate_of_electron_gsf_dR<{dr}": frac((U & (G["dr_other_e_gsf"] < dr)).sum(), U.sum()),
            f"overlaps_same_charge_generaltrack_dR<{dr}": frac((U & trk_overlap).sum(), U.sum()),
            "that_generaltrack_linked_to_|pdg|": {
                str(p): int(c) for p, c in zip(*np.unique(G["gentrk_link_pdg"][U & trk_overlap], return_counts=True))
            },
        },
        # trackless and truthless: no SimTICLCandidate link AND no same-charge GeneralTrack under it (closest to a genuine reco fake)
        "unlinked_and_no_generaltrack_fraction_of_all_gsf": frac((U & ~trk_overlap).sum(), len(G["pt"])),
        # baseline for the unlinked diagnostics: how often is ANY GSF category near a gen photon / electron (jet activity)?
        "near_gen_photon_dR<0.05_by_category": {
            key: frac((sel(G["link_pdg"]) & (G["dr_gen_gamma"] < DR_NEAR_GEN)).sum(), sel(G["link_pdg"]).sum())
            for key, _, sel, _ in CATEGORIES
            if sel(G["link_pdg"]).any()
        },
        "near_gen_electron_dR<0.05_by_category": {
            key: frac((sel(G["link_pdg"]) & (G["dr_gen_e"] < DR_NEAR_GEN)).sum(), sel(G["link_pdg"]).sum())
            for key, _, sel, _ in CATEGORIES
            if sel(G["link_pdg"]).any()
        },
        "hadron_linked_overlapping_generaltrack_linked_to_|pdg|": {
            str(p): int(c) for p, c in zip(*np.unique(G["gentrk_link_pdg"][(G["link_pdg"] == 211) & trk_overlap], return_counts=True))
        },
    }
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nano-dir", default="data/cms/phase2/offline/nano")
    p.add_argument("--samples", nargs="+", default=["ttbar_0pu", "zll_0pu", "qcd_0pu"])
    p.add_argument("--outdir", default="plots/GSF_tracks_study")
    p.add_argument("--dr", type=float, default=0.02, help="dR used only in the summary.json diagnostics (not in the eff / fake definitions)")
    p.add_argument("--eta-pt-min", type=float, nargs="+", default=[PT_MIN], help="pT thresholds for the eta plots (extra thresholds get a _ptgtX suffix)")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--nproc", type=int, default=8)
    args = p.parse_args()
    mplhep.style.use("CMS")
    for s in args.samples:
        run_sample(s, args)


if __name__ == "__main__":
    main()
