"""Candidate-level target vs gen for three pileup-free target definitions, read straight from the NanoAOD.

Target particle = one SimTICLCandidate, with the nano_v2 postprocessing kinematics: E = raw_energy for |pdgID| == 11
(SimTICLCandidates_energy is ~0.38x truth for electrons) else SimTICLCandidates_energy; pT = E sin(theta) from the
candidate eta; phi from the candidate. Three ways to call a candidate "not pileup":
  A: SimTICLCandidates_isPU == 0            (drops the -1 lookup failures)
  B: SimTICLCandidates_isPU in {0, -1}      (keeps the -1 lookup failures, signal or pileup alike)
  C: ticlSimTrackstersfromCPs_isPU == 0     (index-aligned 1:1 with SimTICLCandidates, never -1)
Gen = HGCalGenPart (status 1, pT > 0.5 GeV, hard scatter only), neutrinos removed.
Everything in the HGCAL endcap 1.5 < |eta| < 3.0; jets are anti-kT R=0.4, pT > 15 GeV, clustered from endcap particles.

Plots (PDF) under plots/phase2/target_isPU_defs_vs_gen/<sample>/:
  particle_pt.pdf, jet_pt.pdf, jet_pt_response.pdf

Usage: pixi run python3 scripts/phase2/target_isPU_defs_vs_gen.py [--samples ttbar_0pu ttbar_200pu] [--nfiles 50]
"""

import argparse
import glob
import os

import awkward as ak
import fastjet
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

ETA_MIN, ETA_MAX = 1.5, 3.0
JET_PT_MIN = 15.0
DR_MATCH = 0.1
NU = [12, 14, 16]
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
RLABEL = {"ttbar_0pu": r"$t\bar{t}$, 0 PU", "ttbar_200pu": r"$t\bar{t}$, 200 PU"}

DEFS = [
    ("A", "SimTICLCandidates isPU = 0", "#d62728", "-"),
    ("B", "SimTICLCandidates isPU = 0 or -1", "#1f77b4", "--"),
    ("C", "ticlSimTrackstersfromCPs isPU = 0", "#2ca02c", ":"),
]
GEN_COLOR = "black"

BRANCHES = [
    "SimTICLCandidates_pdgID",
    "SimTICLCandidates_eta",
    "SimTICLCandidates_phi",
    "SimTICLCandidates_energy",
    "SimTICLCandidates_raw_energy",
    "SimTICLCandidates_isPU",
    "ticlSimTrackstersfromCPs_isPU",
    "HGCalGenPart_pdgId",
    "HGCalGenPart_pt",
    "HGCalGenPart_eta",
    "HGCalGenPart_phi",
    "HGCalGenPart_energy",
]


def load(sample, nfiles):
    files = sorted(glob.glob(f"data/cms/phase2/offline/nano/{sample}/*.root"))[:nfiles]
    return ak.concatenate([uproot.open(f)["Events"].arrays(BRANCHES) for f in files]), len(files)


def in_endcap(eta):
    return (np.abs(eta) > ETA_MIN) & (np.abs(eta) < ETA_MAX)


def particles(a):
    """Per-event endcap particle collections (pt, eta, phi, E) for gen and each target definition."""
    is_e = np.abs(a.SimTICLCandidates_pdgID) == 11
    energy = ak.where(is_e, a.SimTICLCandidates_raw_energy, a.SimTICLCandidates_energy)
    eta, phi = a.SimTICLCandidates_eta, a.SimTICLCandidates_phi
    pt = energy / np.cosh(eta)
    cand = ak.zip({"pt": pt, "eta": eta, "phi": phi, "E": energy})
    old, new = a.SimTICLCandidates_isPU, a.ticlSimTrackstersfromCPs_isPU
    sel = {"A": old == 0, "B": old != 1, "C": new == 0}
    out = {k: cand[m & in_endcap(eta)] for k, m in sel.items()}
    g = ak.zip({"pt": a.HGCalGenPart_pt, "eta": a.HGCalGenPart_eta, "phi": a.HGCalGenPart_phi, "E": a.HGCalGenPart_energy})
    apid = np.abs(a.HGCalGenPart_pdgId)
    gm = (apid != NU[0]) & (apid != NU[1]) & (apid != NU[2]) & in_endcap(a.HGCalGenPart_eta)
    out["gen"] = g[gm]
    return out


def cluster(p):
    vec = ak.zip(
        {"px": p.pt * np.cos(p.phi), "py": p.pt * np.sin(p.phi), "pz": p.pt * np.sinh(p.eta), "E": p.E}
    )
    j = fastjet.ClusterSequence(vec, JETDEF).inclusive_jets(min_pt=JET_PT_MIN)
    pt = np.hypot(j.px, j.py)
    eta = np.arcsinh(j.pz / np.maximum(pt, 1e-9))
    phi = np.arctan2(j.py, j.px)
    jets = ak.zip({"pt": pt, "eta": eta, "phi": phi})
    return jets[in_endcap(jets.eta)]


def match_response(gen_jets, tgt_jets):
    """Greedy dR < DR_MATCH matching of target jets to gen jets (highest gen pT first); returns gen pT, response."""
    gpt, resp = [], []
    for g, t in zip(gen_jets, tgt_jets):
        gp, ge, gf = g.pt.to_numpy(), g.eta.to_numpy(), g.phi.to_numpy()
        tp, te, tf = t.pt.to_numpy(), t.eta.to_numpy(), t.phi.to_numpy()
        used = np.zeros(len(tp), bool)
        for k in np.argsort(-gp):
            if not len(tp):
                break
            d = np.hypot(te - ge[k], np.arctan2(np.sin(tf - gf[k]), np.cos(tf - gf[k])))
            d[used] = 99
            j = int(np.argmin(d))
            if d[j] < DR_MATCH:
                used[j] = True
                gpt.append(gp[k])
                resp.append(tp[j] / gp[k])
    return np.array(gpt), np.array(resp)


def label(ax, sample, nev):
    mplhep.cms.label("Work in progress", data=False, rlabel=f"{RLABEL.get(sample, sample)} ({nev} events)", ax=ax, fontsize=16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=["ttbar_0pu", "ttbar_200pu"])
    ap.add_argument("--nfiles", type=int, default=50)
    args = ap.parse_args()
    mplhep.style.use("CMS")
    note = rf"HGCAL endcap ${ETA_MIN} < |\eta| < {ETA_MAX}$"

    for sample in args.samples:
        a, nf = load(sample, args.nfiles)
        nev = len(a)
        out = f"plots/phase2/target_isPU_defs_vs_gen/{sample}"
        os.makedirs(out, exist_ok=True)
        print(f"\n===== {sample}: {nf} files, {nev} events")
        P = particles(a)

        # particle pT
        fig, ax = plt.subplots(figsize=(10, 8))
        b = np.logspace(-1, 2.7, 60)
        h = ax.hist(ak.flatten(P["gen"].pt).to_numpy(), bins=b, histtype="step", lw=2.5, color=GEN_COLOR, label=f"Gen (N={ak.sum(ak.num(P['gen']))})")
        for k, lab, c, ls in DEFS:
            ax.hist(ak.flatten(P[k].pt).to_numpy(), bins=b, histtype="step", lw=2.5, color=c, ls=ls, label=f"{k}: {lab} (N={ak.sum(ak.num(P[k]))})")
        ax.axvline(0.5, color="0.5", lw=1, ls="-.")
        ax.text(0.52, 0.04, "gen $p_T$ > 0.5 GeV", transform=ax.get_xaxis_transform(), fontsize=13, color="0.4")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 30)
        ax.set_xlabel(r"particle $p_T$ [GeV]")
        ax.set_ylabel("particles")
        ax.legend(fontsize=13, title=note, title_fontsize=13, loc="upper right")
        label(ax, sample, nev)
        fig.tight_layout()
        fig.savefig(f"{out}/particle_pt.pdf")
        plt.close(fig)

        # jets
        J = {k: cluster(P[k]) for k in ["gen"] + [d[0] for d in DEFS]}
        fig, ax = plt.subplots(figsize=(10, 8))
        b = np.logspace(np.log10(JET_PT_MIN), 2.8, 34)
        ax.hist(ak.flatten(J["gen"].pt).to_numpy(), bins=b, histtype="step", lw=2.5, color=GEN_COLOR, label=f"Gen (N={ak.sum(ak.num(J['gen']))})")
        for k, lab, c, ls in DEFS:
            ax.hist(ak.flatten(J[k].pt).to_numpy(), bins=b, histtype="step", lw=2.5, color=c, ls=ls, label=f"{k}: {lab} (N={ak.sum(ak.num(J[k]))})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 30)
        ax.set_xticks([15, 20, 30, 50, 100, 200, 400])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"jet $p_T$ [GeV]")
        ax.set_ylabel("jets")
        ax.legend(fontsize=13, title=note + "\n" + rf"anti-$k_T$ R=0.4, $p_T$ > {JET_PT_MIN:.0f} GeV", title_fontsize=13, loc="upper right")
        label(ax, sample, nev)
        fig.tight_layout()
        fig.savefig(f"{out}/jet_pt.pdf")
        plt.close(fig)

        # jet response
        fig, ax = plt.subplots(figsize=(10, 8))
        b = np.linspace(0, 3, 61)
        ngen = ak.sum(ak.num(J["gen"]))
        for k, lab, c, ls in DEFS:
            gpt, r = match_response(J["gen"], J[k])
            med, iqr = np.median(r), np.subtract(*np.percentile(r, [75, 25]))
            ax.hist(np.clip(r, b[0], b[-1] - 1e-6), bins=b, histtype="step", lw=2.5, color=c, ls=ls, label=f"{k}: median {med:.3f}, IQR {iqr:.3f}, matched {len(r)}/{ngen}")
            print(f"  {k} ({lab}): particles {ak.sum(ak.num(P[k]))}, jets {ak.sum(ak.num(J[k]))}, matched {len(r)}/{ngen}, response median {med:.3f} IQR {iqr:.3f}")
        print(f"  gen: particles {ak.sum(ak.num(P['gen']))}, jets {ngen}")
        ax.axvline(1, color="k", ls=":")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
        ax.set_xlabel(r"jet $p_T^{\mathrm{target}} / p_T^{\mathrm{gen}}$ (overflow in last bin)")
        ax.set_ylabel("matched jets")
        ax.legend(fontsize=13, title=note + "\n" + rf"$\Delta R$ < {DR_MATCH} to gen jet", title_fontsize=13, loc="upper right")
        label(ax, sample, nev)
        fig.tight_layout()
        fig.savefig(f"{out}/jet_pt_response.pdf")
        plt.close(fig)
        print("  saved", out)


if __name__ == "__main__":
    main()
