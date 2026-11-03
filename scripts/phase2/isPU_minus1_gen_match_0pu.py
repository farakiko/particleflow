"""0 PU: do the SimTICLCandidates with isPU = -1 (the CMSSW lookup failure) correspond to a gen particle?

At 0 PU every candidate is signal, so isPU = -1 vs isPU = 0 only says whether the CMSSW parent-CaloParticle lookup
failed. Here we ask whether the -1 candidates are "real" particles in the sense of matching a gen particle, compared
with the isPU = 0 candidates.

Target candidate: SimTICLCandidate in the endcap 1.5 < |eta| < 3.0 with pT > 0.5 GeV, nano_v2 kinematics
(E = raw_energy for |pdgID| == 11, else energy; pT = E / cosh(eta)).
Gen: HGCalGenPart (all status 1, stored only for pT > 0.5 GeV), neutrinos removed, |eta| in 1.3-3.2 (margin so that
candidates near the endcap edges can still find their gen particle).

Only SimTICLCandidates (incl. their isPU flag) and gen particles are used. Kinematic match, per candidate (not one-to-one):
is there a gen particle with the same charge, dR < 0.05 and 0.5 < pT_cand / pT_gen < 2 (and a loose dR < 0.2 version)?
Also a one-to-one version (greedy over all candidates, highest pT first) to see whether the -1 candidates only match gen
particles that another candidate already took (fragments / duplicates).

Plots (PDF) under plots/phase2/isPU_minus1_gen_match_0pu/<sample>/:
  match_fraction.pdf          matched fraction vs pT, |eta|, PID, number of sim tracksters (-1 vs 0)
  match_fraction_by_ncons.pdf matched fraction vs pT for 1 vs >= 2 sim tracksters (tight and loose dR)
  match_variables.pdf         dR to the nearest same-charge gen, pT ratio to it (-1 vs 0)

Usage: pixi run python3 scripts/phase2/isPU_minus1_gen_match_0pu.py [--samples ttbar_0pu qcd_0pu zll_0pu] [--nfiles 20]
"""

import argparse
import glob
import os

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402

ETA_MIN, ETA_MAX = 1.5, 3.0
GEN_ETA_MIN, GEN_ETA_MAX = 1.3, 3.2
PT_MIN = 0.5
DR_MATCH = 0.05
DR_LOOSE = 0.2  # neutral candidates take their direction from the sim-trackster barycenter, which is broad
PT_RATIO = (0.5, 2.0)
NU = (12, 14, 16)
GEN_CHARGED = (11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334)
PID_GROUPS = [("photon", 22), ("electron", 11), ("muon", 13), ("ch. hadron", 211), ("neu. hadron", 130)]
GROUPS = [(-1, "isPU = -1", "#d62728"), (0, "isPU = 0", "#1f77b4")]
RLABEL = {"ttbar_0pu": r"$t\bar{t}$, 0 PU", "qcd_0pu": "QCD, 0 PU", "zll_0pu": r"Z$\to\ell\ell$, 0 PU"}

BR = [
    "SimTICLCandidates_pdgID",
    "SimTICLCandidates_charge",
    "SimTICLCandidates_eta",
    "SimTICLCandidates_phi",
    "SimTICLCandidates_energy",
    "SimTICLCandidates_raw_energy",
    "SimTICLCandidates_isPU",
    "SimTICLCandidates_nSimTICLCandidatesExtra",
    "HGCalGenPart_pdgId",
    "HGCalGenPart_pt",
    "HGCalGenPart_eta",
    "HGCalGenPart_phi",
]


def in_range(eta, lo, hi):
    return (np.abs(eta) > lo) & (np.abs(eta) < hi)


def dphi(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def load(sample, nfiles):
    files = sorted(glob.glob(f"data/cms/phase2/offline/nano/{sample}/*.root"))[:nfiles]
    a = ak.concatenate([uproot.open(f)["Events"].arrays(BR) for f in files])
    pid = np.abs(a.SimTICLCandidates_pdgID)
    energy = ak.where(pid == 11, a.SimTICLCandidates_raw_energy, a.SimTICLCandidates_energy)
    eta = a.SimTICLCandidates_eta
    t = ak.zip(
        {
            "pt": energy / np.cosh(eta),
            "eta": eta,
            "phi": a.SimTICLCandidates_phi,
            "pid": pid,
            "q": a.SimTICLCandidates_charge != 0,
            "ncons": a.SimTICLCandidates_nSimTICLCandidatesExtra,
            "ispu": a.SimTICLCandidates_isPU,
        }
    )
    t = t[in_range(t.eta, ETA_MIN, ETA_MAX) & (t.pt > PT_MIN)]
    gpid = np.abs(a.HGCalGenPart_pdgId)
    g = ak.zip({"pt": a.HGCalGenPart_pt, "eta": a.HGCalGenPart_eta, "phi": a.HGCalGenPart_phi, "pid": gpid})
    g = g[(gpid != NU[0]) & (gpid != NU[1]) & (gpid != NU[2]) & in_range(g.eta, GEN_ETA_MIN, GEN_ETA_MAX)]
    return t, g, len(files), len(a)


def match(t, g):
    """Per candidate: dR and pT ratio to the nearest same-charge gen; per-candidate match; one-to-one greedy match."""
    out = {k: [] for k in ("dr", "ratio", "match", "loose", "unique")}
    for ev in range(len(t)):
        tt, gg = t[ev], g[ev]
        tpt, teta, tphi, tq = (tt[k].to_numpy() for k in ("pt", "eta", "phi", "q"))
        gpt, geta, gphi = gg.pt.to_numpy(), gg.eta.to_numpy(), gg.phi.to_numpy()
        gq = np.isin(gg.pid.to_numpy(), GEN_CHARGED)
        n = len(tpt)
        if len(gpt) == 0:
            out["dr"].append(np.full(n, 9.0))
            out["ratio"].append(np.full(n, np.nan))
            out["match"].append(np.zeros(n, bool))
            out["loose"].append(np.zeros(n, bool))
            out["unique"].append(np.zeros(n, bool))
            continue
        d = np.hypot(teta[:, None] - geta[None, :], dphi(tphi[:, None], gphi[None, :]))
        d = np.where(tq[:, None] == gq[None, :], d, 9.0)
        r = tpt[:, None] / gpt[None, :]
        rok = (r > PT_RATIO[0]) & (r < PT_RATIO[1])
        ok = (d < DR_MATCH) & rok
        near = np.argmin(d, axis=1)
        out["dr"].append(d[np.arange(n), near])
        out["ratio"].append(r[np.arange(n), near])
        out["match"].append(ok.any(axis=1))
        out["loose"].append(((d < DR_LOOSE) & rok).any(axis=1))
        used = np.zeros(len(gpt), bool)
        uniq = np.zeros(n, bool)
        for i in np.argsort(-tpt):
            c = ok[i] & ~used
            if c.any():
                j = np.where(c)[0][np.argmin(d[i][c])]
                used[j] = True
                uniq[i] = True
        out["unique"].append(uniq)
    return {k: np.concatenate(v) for k, v in out.items()}


def flat(x):
    return ak.to_numpy(ak.flatten(x))


def frac_err(m, sel):
    n = sel.sum()
    f = m[sel].mean() if n else np.nan
    return f, np.sqrt(f * (1 - f) / max(n, 1)), n


def binned(x, m, sel, bins):
    f, e = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = sel & (x >= lo) & (x < hi)
        ff, ee, n = frac_err(m, s)
        f.append(ff if n >= 20 else np.nan)
        e.append(ee if n >= 20 else np.nan)
    return np.array(f), np.array(e)


def cms(ax, rl):
    mplhep.cms.label("Work in progress", data=False, rlabel=rl, ax=ax, fontsize=15)


def cut_text(ax):
    ax.text(
        0.02,
        0.03,
        rf"cand. $p_T$ > {PT_MIN} GeV, {ETA_MIN} < $|\eta|$ < {ETA_MAX}" + "\n" + rf"match: same charge, $\Delta R$ < {DR_MATCH}, {PT_RATIO[0]} < $p_T$ ratio < {PT_RATIO[1]}",
        transform=ax.transAxes,
        fontsize=11,
        color="0.35",
        va="bottom",
    )


def report(sample, c, M):
    print(f"\n===== {sample}")
    for v, name, _ in GROUPS:
        s = c["ispu"] == v
        print(f"  {name}: N={s.sum()}  matched {M['match'][s].mean():.4f}  one-to-one {M['unique'][s].mean():.4f}  loose (dR<{DR_LOOSE}) {M['loose'][s].mean():.4f}")
        for nc_name, nc in (("ncons=1", c["ncons"] == 1), ("ncons>=2", c["ncons"] >= 2)):
            ss = s & nc
            print(f"     {nc_name:8s} N={ss.sum():7d} ({ss.sum() / max(s.sum(), 1):.3f})  matched {M['match'][ss].mean():.4f}  one-to-one {M['unique'][ss].mean():.4f}  loose {M['loose'][ss].mean():.4f}")
        for pname, p in PID_GROUPS:
            ss = s & (c["pid"] == p)
            if ss.sum():
                print(f"     {pname:12s} N={ss.sum():7d} ({ss.sum() / s.sum():.3f})  matched {M['match'][ss].mean():.4f}  one-to-one {M['unique'][ss].mean():.4f}")
        for lo, hi in ((0.5, 1), (1, 2), (2, 5), (5, 20), (20, 1e9)):
            ss = s & (c["pt"] >= lo) & (c["pt"] < hi)
            print(f"     pT {lo:>4}-{hi:<6g} N={ss.sum():7d}  matched {M['match'][ss].mean():.4f}")


def plot_match_fraction(c, M, out, rl):
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    specs = [
        (axs[0, 0], c["pt"], np.logspace(np.log10(PT_MIN), 2.3, 18), True, r"candidate $p_T$ [GeV]"),
        (axs[0, 1], np.abs(c["eta"]), np.linspace(ETA_MIN, ETA_MAX, 13), False, r"candidate $|\eta|$"),
    ]
    for ax, x, b, logx, xl in specs:
        xc = np.sqrt(b[1:] * b[:-1]) if logx else 0.5 * (b[1:] + b[:-1])
        for v, name, col in GROUPS:
            s = c["ispu"] == v
            f, e = binned(x, M["match"], s, b)
            ax.errorbar(xc, f, yerr=e, fmt="o-", color=col, label=f"{name} (N={s.sum()})")
            fu, _ = binned(x, M["loose"], s, b)
            ax.plot(xc, fu, "--", color=col, lw=1.5, label=rf"{name}, $\Delta R$ < {DR_LOOSE}")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xl)
    for ax, keys, xl in (
        (axs[1, 0], [(n, c["pid"] == p) for n, p in PID_GROUPS], "candidate type"),
        (axs[1, 1], [("1", c["ncons"] == 1), ("2", c["ncons"] == 2), ("3-4", (c["ncons"] >= 3) & (c["ncons"] <= 4)), (r"$\geq$5", c["ncons"] >= 5)], "sim tracksters in the candidate"),
    ):
        x = np.arange(len(keys))
        for k, (v, name, col) in enumerate(GROUPS):
            s = c["ispu"] == v
            fe = [frac_err(M["match"], s & m) for _, m in keys]
            f = np.array([q[0] if q[2] >= 20 else np.nan for q in fe])
            ax.bar(x + (k - 0.5) * 0.4, f, 0.4, yerr=[q[1] for q in fe], color=col, alpha=0.85, label=name)
            for xi, (ff, _, n) in zip(x, fe):
                if n:
                    ax.text(xi + (k - 0.5) * 0.4, 0.02, f"N={n}", rotation=90, ha="center", va="bottom", fontsize=10, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels([kk for kk, _ in keys])
        ax.set_xlabel(xl)
    for ax in axs.flat:
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("candidates matched to a gen particle")
        ax.legend(fontsize=12, loc="upper left", ncol=2)
        cut_text(ax)
        cms(ax, rl)
    fig.tight_layout()
    fig.savefig(f"{out}/match_fraction.pdf")
    plt.close(fig)


def plot_by_ncons(c, M, out, rl):
    """Matched fraction vs pT, split by the number of sim tracksters: separates the flag from the candidate composition."""
    fig, axs = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
    b = np.logspace(np.log10(PT_MIN), 2.3, 18)
    xc = np.sqrt(b[1:] * b[:-1])
    for ax, (key, tag) in zip(axs, (("match", rf"$\Delta R$ < {DR_MATCH}"), ("loose", rf"$\Delta R$ < {DR_LOOSE}"))):
        for v, name, col in GROUPS:
            for nname, nm, ls in (("1 sim trackster", c["ncons"] == 1, "o-"), (r"$\geq$2 sim tracksters", c["ncons"] >= 2, "s--")):
                s = (c["ispu"] == v) & nm
                f, e = binned(c["pt"], M[key], s, b)
                ax.errorbar(xc, f, yerr=e, fmt=ls, color=col, mfc="white" if "--" in ls else col, label=f"{name}, {nname} (N={s.sum()})")
        ax.set_xscale("log")
        ax.set_xlabel(r"candidate $p_T$ [GeV]")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=12, loc="upper left", ncol=2, title=f"match: same charge, {tag}, {PT_RATIO[0]} < $p_T$ ratio < {PT_RATIO[1]}", title_fontsize=12)
        cms(ax, rl)
    axs[0].set_ylabel("candidates matched to a gen particle")
    fig.tight_layout()
    fig.savefig(f"{out}/match_fraction_by_ncons.pdf")
    plt.close(fig)


def plot_variables(c, M, out, rl):
    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    specs = [
        (M["dr"], np.logspace(-4, 0.5, 46), r"$\Delta R$ to nearest same-charge gen particle (overflow in last bin)", DR_MATCH),
        (M["ratio"], np.logspace(-1.3, 1.3, 53), r"$p_T^{\mathrm{cand}} / p_T^{\mathrm{gen}}$, nearest same-charge gen", PT_RATIO),
    ]
    for ax, (x, b, xl, cut) in zip(axs, specs):
        for v, name, col in GROUPS:
            s = (c["ispu"] == v) & np.isfinite(x)
            ax.hist(np.clip(x[s], b[0], b[-1] * 0.999), bins=b, histtype="step", lw=2.5, weights=np.full(s.sum(), 1.0 / max(s.sum(), 1)), color=col, label=name)
        for cc in np.atleast_1d(cut):
            ax.axvline(cc, color="0.4", ls=":")
        ax.set_xscale("log")
        ax.set_xlabel(xl, fontsize=17)
        ax.set_ylabel("fraction of candidates per bin")
        ax.legend(fontsize=13, loc="upper right")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
        cms(ax, rl)
    fig.tight_layout()
    fig.savefig(f"{out}/match_variables.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", default=["ttbar_0pu", "qcd_0pu", "zll_0pu"])
    ap.add_argument("--nfiles", type=int, default=20)
    args = ap.parse_args()
    mplhep.style.use("CMS")
    for sample in args.samples:
        t, g, nf, nev = load(sample, args.nfiles)
        M = match(t, g)
        c = {k: flat(t[k]) for k in t.fields}
        rl = f"{RLABEL.get(sample, sample)} ({nev} events)"
        out = f"plots/phase2/isPU_minus1_gen_match_0pu/{sample}"
        os.makedirs(out, exist_ok=True)
        report(f"{sample}: {nf} files, {nev} events", c, M)
        plot_match_fraction(c, M, out, rl)
        plot_variables(c, M, out, rl)
        plot_by_ncons(c, M, out, rl)
        print("  saved", out)


if __name__ == "__main__":
    main()
