"""GSF tracks in the barrel, matched to GEN electrons by dR (Phase-2 NanoAOD, 0 PU).

Companion to gsf_tracks_study.py, which uses the SimTICLCandidate -> GSF truth link and is therefore
restricted to the HGCAL endcap: in the barrel SimTICLCandidates cover only ~10% of the visible gen
particles (vs > 100% in the endcap; numbers in summary.json). Here the truth is the generator record
`HGCalGenPart` (all status-1 particles, all eta) instead, matched to the GSF tracks by dR.

Matching: dR between the GSF track direction at the vertex ("mode" eta/phi) and the gen particle
direction, dR < --dr (default 0.02). Electron matching is ONE-TO-ONE: a gen electron claims only its
closest GSF track; any other GSF track within dR of it is not electron-matched and falls through to the
next class. In dense jets this removes most accidental hadron-GSF <-> soft-electron matches (QCD barrel:
1.3% -> 0.3% of hadron-linked GSF tracks), at the price of ~7% of genuine electron GSF tracks in QCD
whose gen electron has another GSF track closer. Validation against the truth link where it exists
(GSF tracks linked to electron / hadron / muon SimTICLCandidates) is written to summary.json.

Definitions (pT > 1 GeV; the eta/pT cuts and binning apply to the object being counted):
  efficiency  = N(gen electrons with a GSF track within dR) / N(gen electrons)
  fake rate   = N(GSF tracks with no gen electron within dR) / N(GSF tracks), shown with the subset
                that has no charged gen particle and no photon within dR at all
  composition = GSF tracks by matched gen class, priority electron (one-to-one) > muon > charged hadron > photon > none
pT plots: barrel |eta| < 1.5. eta plots: -3 < eta < 3, so barrel and endcap are compared with the same method.

Usage:
  pixi run python3 scripts/phase2/gsf_tracks_barrel_study.py      # ttbar_0pu + zll_0pu + qcd_0pu
Outputs (PDF only): plots/GSF_tracks_barrel_study/<sample>/*.pdf, plus a summary.json of the key numbers.
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

import gsf_tracks_study as base  # noqa: E402  (same directory: plotting helpers, binning, styling)

BARREL_ETA_MAX, ENDCAP_ETA_MAX, PT_MIN = 1.5, 3.0, 1.0
ETA_BINS_FULL = np.round(np.arange(-ENDCAP_ETA_MAX, ENDCAP_ETA_MAX + 1e-6, 0.1), 2)
CHARGED_HADRONS = [211, 321, 2212, 3222, 3112, 3312, 3334]  # all charged |pdgId| present in HGCalGenPart besides e, mu

# gen class codes used for the composition (priority order as listed)
MATCH_CATEGORIES = [
    ("e", "Electron", lambda c: c == 11, "#5790fc"),
    ("had", "Charged hadron", lambda c: c == 211, "#f89c20"),
    ("mu", "Muon", lambda c: c == 13, "#e42536"),
    ("gamma", "Photon only", lambda c: c == 22, "#9c9ca1"),
    ("none", "No gen particle", lambda c: c == 0, "#964a8b"),
]

BRANCHES = [
    "GSFTrack_ptMode",
    "GSFTrack_etaMode",
    "GSFTrack_phiMode",
    "HGCalGenPart_pdgId",
    "HGCalGenPart_pt",
    "HGCalGenPart_eta",
    "HGCalGenPart_phi",
    "SimTICLCandidates_pdgID",
    "SimTICLCandidates_pt",
    "SimTICLCandidates_eta",
    "SimTICLCandidates_gsftrackIdx",
]


def nearest(a_eta, a_phi, b_eta, b_phi):
    """Per object of a: (dR, index in b) of the closest object of b; (NO_MATCH, -1) if b is empty."""
    a = ak.zip({"eta": a_eta, "phi": a_phi})
    b = ak.zip({"eta": b_eta, "phi": b_phi, "i": ak.local_index(b_eta)})
    pairs = ak.cartesian({"a": a, "b": b}, nested=True)
    dr = base.delta_r(pairs.a.eta, pairs.a.phi, pairs.b.eta, pairs.b.phi)
    best = ak.argmin(dr, axis=2, keepdims=True)
    dr_best = ak.flatten(ak.fill_none(ak.flatten(dr[best], axis=2), base.NO_MATCH)).to_numpy()
    i_best = ak.flatten(ak.fill_none(ak.flatten(pairs.b.i[best], axis=2), -1)).to_numpy()
    return dr_best, i_best


def process_file(path):
    try:
        a = uproot.open(path)["Events"].arrays(BRANCHES)
    except Exception as e:  # a few NanoAOD files can be unreadable; report and skip
        return {"error": f"{path}: {e}"}
    nev = len(a)
    gen_pdg = np.abs(a.HGCalGenPart_pdgId)
    is_e = gen_pdg == 11
    is_charged_had = gen_pdg == CHARGED_HADRONS[0]
    for p in CHARGED_HADRONS[1:]:
        is_charged_had = is_charged_had | (gen_pdg == p)
    gsf_eta, gsf_phi = a.GSFTrack_etaMode, a.GSFTrack_phiMode
    gen_eta, gen_phi = a.HGCalGenPart_eta, a.HGCalGenPart_phi

    def dr_to(mask):
        return base.min_dr(gsf_eta, gsf_phi, gen_eta, gen_phi, b_mask=mask)

    # GSF -> nearest gen electron (also its index, for the duplicate count)
    dr_e, idx_e = nearest(gsf_eta, gsf_phi, gen_eta[is_e], gen_phi[is_e])
    # gen electron -> nearest GSF track
    ge_dr, ge_idx = nearest(gen_eta[is_e], gen_phi[is_e], gsf_eta, gsf_phi)

    # duplicate: GSF matched to a gen electron whose closest GSF track is a different one
    n_gsf = ak.num(a.GSFTrack_ptMode).to_numpy()
    n_ge = ak.num(gen_eta[is_e]).to_numpy()
    gsf_ev = np.repeat(np.arange(nev), n_gsf)
    ge_off = np.concatenate([[0], np.cumsum(n_ge)[:-1]])
    gsf_local = ak.flatten(ak.local_index(gsf_eta)).to_numpy()
    has_e = idx_e >= 0  # event has at least one gen electron
    owner_best = np.full(len(idx_e), -1)
    owner_best[has_e] = ge_idx[ge_off[gsf_ev[has_e]] + idx_e[has_e]]
    duplicate = has_e & (owner_best != gsf_local)

    # truth link of each GSF track (for validating the dR method where SimTICLCandidates exist)
    gsf_off = np.concatenate([[0], np.cumsum(n_gsf)[:-1]])
    sc_ev = np.repeat(np.arange(nev), ak.num(a.SimTICLCandidates_pdgID).to_numpy())
    sc_gsf = ak.flatten(a.SimTICLCandidates_gsftrackIdx).to_numpy()
    sc_pdg = np.abs(ak.flatten(a.SimTICLCandidates_pdgID).to_numpy())
    has = sc_gsf >= 0
    link_pdg = np.zeros(n_gsf.sum(), dtype=np.int64)
    link_pdg[gsf_off[sc_ev[has]] + sc_gsf[has]] = sc_pdg[has]
    link_pdg[(gsf_off[sc_ev[has]] + sc_gsf[has])[sc_pdg[has] == 11]] = 11

    # SimTICLCandidate coverage of the visible gen record, barrel vs endcap (why this study uses gen)
    visible = ~((gen_pdg == 12) | (gen_pdg == 14) | (gen_pdg == 16))
    coverage = {}
    for reg, lo, hi in [("barrel", 0.0, BARREL_ETA_MAX), ("endcap", BARREL_ETA_MAX, ENDCAP_ETA_MAX)]:
        g_in = (np.abs(gen_eta) >= lo) & (np.abs(gen_eta) < hi) & (a.HGCalGenPart_pt > PT_MIN) & visible
        s_in = (np.abs(a.SimTICLCandidates_eta) >= lo) & (np.abs(a.SimTICLCandidates_eta) < hi) & (a.SimTICLCandidates_pt > PT_MIN)
        coverage[reg] = (int(ak.sum(s_in)), int(ak.sum(g_in)))

    gsf = {
        "pt": ak.flatten(a.GSFTrack_ptMode).to_numpy(),
        "eta": ak.flatten(gsf_eta).to_numpy(),
        "dr_e": dr_e,
        "dr_mu": dr_to(gen_pdg == 13),
        "dr_had": dr_to(is_charged_had),
        "dr_gamma": dr_to(gen_pdg == 22),
        "duplicate": duplicate,
        "link_pdg": link_pdg,
    }
    ele = {
        "pt": ak.flatten(a.HGCalGenPart_pt[is_e]).to_numpy(),
        "eta": ak.flatten(gen_eta[is_e]).to_numpy(),
        "dr_gsf": ge_dr,
    }
    return {"nev": nev, "gsf": gsf, "ele": ele, "coverage": coverage}


def match_code(G, dr):
    """Gen class of each GSF track by priority electron (one-to-one) > muon > charged hadron > photon > none."""
    return np.select(
        [(G["dr_e"] < dr) & ~G["duplicate"], G["dr_mu"] < dr, G["dr_had"] < dr, G["dr_gamma"] < dr],
        [11, 13, 211, 22],
        default=0,
    )


def plot_dr_validation(G_barrel, dr, sample, path):
    """dR(GSF, nearest gen electron) for barrel GSF tracks: the electron peak and where the cut sits."""
    fig, ax = base.cms_axes(sample, r"$\Delta R$(GSF track, nearest gen electron)", "GSF tracks", logx=True)
    bins = np.logspace(-5, np.log10(5), 46)
    has_gen_e = G_barrel["dr_e"] < base.NO_MATCH  # GSF tracks in events without any gen electron are not shown
    x = np.clip(G_barrel["dr_e"][has_gen_e], bins[0], bins[-1] * 0.999)
    mplhep.histplot(np.histogram(x, bins)[0], bins, ax=ax, histtype="fill", color="#9c9ca1", alpha=0.6, label="All GSF tracks")
    linked = G_barrel["link_pdg"][has_gen_e] == 11
    mplhep.histplot(np.histogram(x[linked], bins)[0], bins, ax=ax, histtype="step", color="#5790fc", lw=2.5, label="Linked to electron SimTICLCandidate")
    ax.axvline(dr, color="#e42536", ls="--", lw=2, label=rf"Matching cut $\Delta R = {dr:g}$")
    ax.set_yscale("log")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0.5, ax.get_ylim()[1] * 50)
    ax.legend(loc="upper left", fontsize=18, frameon=False, bbox_to_anchor=(0.01, 0.85))
    base.annotate(ax, sample, rf"$|\eta| < {BARREL_ETA_MAX}$, $p_\mathrm{{T}} > {PT_MIN:g}$ GeV")
    ax.text(0.97, 0.95, "events with $\\geq$1 gen electron", transform=ax.transAxes, fontsize=16, ha="right", va="top", color="dimgrey")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def region_numbers(G, E, code, dr):
    out = {
        "efficiency": base.frac((E["dr_gsf"] < dr).sum(), len(E["pt"])),
        "efficiency_pt_gt_10": base.frac((E["dr_gsf"][E["pt"] > 10] < dr).sum(), (E["pt"] > 10).sum()),
        "fakerate_no_gen_electron": base.frac((code != 11).sum(), len(code)),
        "fakerate_no_gen_particle": base.frac((code == 0).sum(), len(code)),
        "fakerate_no_gen_electron_pt_gt_10": base.frac((code[G["pt"] > 10] != 11).sum(), (G["pt"] > 10).sum()),
        "composition": {key: base.frac(sel(code).sum(), len(code)) for key, _, sel, _ in MATCH_CATEGORIES},
        "gsf_near_gen_electron_but_not_its_closest": base.frac(((G["dr_e"] < dr) & G["duplicate"]).sum(), len(code)),
    }
    linked_e = G["link_pdg"] == 11
    linked_other = (G["link_pdg"] != 11) & (G["link_pdg"] != 0)
    linked = G["link_pdg"] != 0
    out["validation"] = {
        "electron_linked_gsf_electron_matched": base.frac((code[linked_e] == 11).sum(), linked_e.sum()),
        "hadron_or_muon_linked_gsf_electron_matched": base.frac((code[linked_other] == 11).sum(), linked_other.sum()),
        "agreement_with_link_on_is_electron": base.frac(((code[linked] == 11) == linked_e[linked]).sum(), linked.sum()),
    }
    return out


def run_sample(sample, args):
    files = sorted(glob.glob(os.path.join(args.nano_dir, sample, "*.root")))
    if args.max_files:
        files = files[: args.max_files]
    with Pool(args.nproc) as pool:
        results = pool.map(process_file, files)
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

    def select(d, lo, hi):
        m = (np.abs(d["eta"]) >= lo) & (np.abs(d["eta"]) < hi) & (d["pt"] > PT_MIN)
        return {k: v[m] for k, v in d.items()}

    Gb, Eb = select(gsf, 0.0, BARREL_ETA_MAX), select(ele, 0.0, BARREL_ETA_MAX)
    Ge, Ee = select(gsf, BARREL_ETA_MAX, ENDCAP_ETA_MAX), select(ele, BARREL_ETA_MAX, ENDCAP_ETA_MAX)
    Gf, Ef = select(gsf, 0.0, ENDCAP_ETA_MAX), select(ele, 0.0, ENDCAP_ETA_MAX)
    code_b, code_e, code_f = match_code(Gb, dr), match_code(Ge, dr), match_code(Gf, dr)

    sel_b = rf"$|\eta| < {BARREL_ETA_MAX}$, $p_\mathrm{{T}} > {PT_MIN:g}$ GeV"
    sel_f = rf"$|\eta| < {ENDCAP_ETA_MAX}$, $p_\mathrm{{T}} > {PT_MIN:g}$ GeV"
    pt_e, eta_e = r"Gen electron $p_\mathrm{T}$ [GeV]", r"Gen electron $\eta$"
    pt_g, eta_g = r"GSF track $p_\mathrm{T}$ [GeV]", r"GSF track $\eta$"
    eff_label = rf"GSF track within $\Delta R < {dr:g}$"
    fake_labels = (rf"No gen electron within $\Delta R < {dr:g}$", rf"No gen e, $\mu$, h$^\pm$, $\gamma$ within $\Delta R < {dr:g}$")
    edges = (-BARREL_ETA_MAX, BARREL_ETA_MAX)

    # ---- efficiency ----
    base.plot_ratios([(eff_label, Eb["pt"], Eb["dr_gsf"] < dr, "o", base.BLUE)], base.PT_BINS, pt_e, "Efficiency", sample, sel_b, f"{outdir}/efficiency_vs_pt.pdf", logx=True)
    base.plot_ratios(
        [(eff_label, Ef["eta"], Ef["dr_gsf"] < dr, "o", base.BLUE)],
        ETA_BINS_FULL,
        eta_e,
        "Efficiency",
        sample,
        sel_f,
        f"{outdir}/efficiency_vs_eta.pdf",
        logx=False,
        shade=False,
        vlines=edges,
    )
    # ---- fake rate ----
    curves = [(fake_labels[0], Gb["pt"], code_b != 11, "o", base.BLUE), (fake_labels[1], Gb["pt"], code_b == 0, "s", base.RED)]
    base.plot_ratios(curves, base.PT_BINS, pt_g, "Fake rate", sample, sel_b, f"{outdir}/fakerate_vs_pt.pdf", logx=True)
    curves = [(fake_labels[0], Gf["eta"], code_f != 11, "o", base.BLUE), (fake_labels[1], Gf["eta"], code_f == 0, "s", base.RED)]
    base.plot_ratios(curves, ETA_BINS_FULL, eta_g, "Fake rate", sample, sel_f, f"{outdir}/fakerate_vs_eta.pdf", logx=False, shade=False, vlines=edges)
    # ---- composition ----
    title = rf"Gen particle within $\Delta R < {dr:g}$"
    base.plot_composition(
        Gb["pt"], code_b, base.PT_BINS, pt_g, sample, sel_b, f"{outdir}/composition_vs_pt.pdf", logx=True, categories=MATCH_CATEGORIES, legend_title=title
    )
    base.plot_composition(
        Gf["eta"],
        code_f,
        ETA_BINS_FULL,
        eta_g,
        sample,
        sel_f,
        f"{outdir}/composition_vs_eta.pdf",
        logx=False,
        categories=MATCH_CATEGORIES,
        legend_title=title,
        shade=False,
        vlines=edges,
    )
    # ---- validation of the dR cut ----
    plot_dr_validation(Gb, dr, sample, f"{outdir}/dr_gsf_gen_electron.pdf")

    cov = {reg: [sum(r["coverage"][reg][i] for r in results) for i in (0, 1)] for reg in ("barrel", "endcap")}
    summary = {
        "sample": sample,
        "files": len(results),
        "unreadable_files": len(errors),
        "events": nev,
        "matching": f"dR(GSF mode eta/phi, gen eta/phi) < {dr}, not exclusive; pT > {PT_MIN} GeV",
        "simticlcandidates_per_visible_gen_particle_pt_gt_1": {reg: {"simcands": s, "gen": g, "ratio": s / g} for reg, (s, g) in cov.items()},
        "barrel": region_numbers(Gb, Eb, code_b, dr),
        "endcap_same_method": region_numbers(Ge, Ee, code_e, dr),
    }
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nano-dir", default="data/cms/phase2/offline/nano")
    p.add_argument("--samples", nargs="+", default=["ttbar_0pu", "zll_0pu", "qcd_0pu"])
    p.add_argument("--outdir", default="plots/GSF_tracks_barrel_study")
    p.add_argument("--dr", type=float, default=0.02, help="dR for GSF <-> gen matching")
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--nproc", type=int, default=8)
    args = p.parse_args()
    mplhep.style.use("CMS")
    for s in args.samples:
        run_sample(s, args)


if __name__ == "__main__":
    main()
