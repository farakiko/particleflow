"""Where does pileup-free target A lose hard-scatter particles that target C keeps? (200 PU, HGCAL endcap)

  A: SimTICLCandidates_isPU == 0           (current postprocessing label; -1 = CMSSW lookup failure, dropped)
  C: ticlSimTrackstersfromCPs_isPU == 0    (index-aligned 1:1 with SimTICLCandidates, never -1)

Target particle = one SimTICLCandidate with the nano_v2 kinematics (E = raw_energy for |pdgID| == 11, else energy;
pT = E / cosh(eta)). Gen = HGCalGenPart (status 1, hard scatter), no neutrinos. Everything: 1.5 < |eta| < 3.0 and
pT > 0.5 GeV (target and gen alike; softer particles are irrelevant here).

Plots under plots/phase2/isPU_A_vs_C_corners/<sample>/:
  lost_fraction.pdf        fraction of C particles that A drops, vs pT, |eta|, PID, N constituent sim tracksters
  lost_fraction_2d.pdf     same in (pT, |eta|) per PID group
  gen_efficiency.pdf       gen particles with a matching target particle (dR, pT ratio, charge), A vs C, + ratio
  jet_response.pdf         matched-jet response median / IQR vs gen-jet pT, |eta| and gen-jet photon pT fraction
  event_sumpt.pdf          per-event endcap sum pT, target / gen

Usage: pixi run python3 scripts/phase2/isPU_A_vs_C_corners.py [--sample ttbar_200pu] [--nfiles 100]
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

ETA_MIN, ETA_MAX = 1.5, 3.0
PT_MIN = 0.5
JET_PT_MIN = 15.0
DR_JET = 0.1
DR_PART = 0.05
PT_RATIO = (0.5, 2.0)
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
NU = (12, 14, 16)
GEN_CHARGED = (11, 13, 211, 321, 2212, 3222, 3112, 3312, 3334)
COL = {"A": "#d62728", "C": "#2ca02c", "gen": "black"}
LAB = {"A": "A: SimTICLCandidates isPU = 0", "C": "C: ticlSimTrackstersfromCPs isPU = 0"}
# target PID groups (SimTICLCandidates only carries 11, 13, 22, 130, 211)
PID_GROUPS = [("photon", (22,)), ("electron", (11,)), ("muon", (13,)), ("ch. hadron", (211,)), ("neu. hadron", (130,))]

BR = [
    "SimTICLCandidates_pdgID",
    "SimTICLCandidates_charge",
    "SimTICLCandidates_eta",
    "SimTICLCandidates_phi",
    "SimTICLCandidates_energy",
    "SimTICLCandidates_raw_energy",
    "SimTICLCandidates_isPU",
    "SimTICLCandidates_nSimTICLCandidatesExtra",
    "ticlSimTrackstersfromCPs_isPU",
    "HGCalGenPart_pdgId",
    "HGCalGenPart_pt",
    "HGCalGenPart_eta",
    "HGCalGenPart_phi",
    "HGCalGenPart_energy",
]


def in_endcap(eta):
    return (np.abs(eta) > ETA_MIN) & (np.abs(eta) < ETA_MAX)


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
            "E": energy,
            "pid": pid,
            "q": a.SimTICLCandidates_charge,
            "ncons": a.SimTICLCandidates_nSimTICLCandidatesExtra,
            "old": a.SimTICLCandidates_isPU,
            "new": a.ticlSimTrackstersfromCPs_isPU,
        }
    )
    t = t[in_endcap(t.eta) & (t.pt > PT_MIN)]
    gpid = np.abs(a.HGCalGenPart_pdgId)
    g = ak.zip(
        {"pt": a.HGCalGenPart_pt, "eta": a.HGCalGenPart_eta, "phi": a.HGCalGenPart_phi, "E": a.HGCalGenPart_energy, "pid": gpid}
    )
    gm = (gpid != NU[0]) & (gpid != NU[1]) & (gpid != NU[2]) & in_endcap(g.eta) & (g.pt > PT_MIN)
    g = g[gm]
    return t, g, len(files)


def dphi(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def binned_frac(x, num_mask, den_mask, bins, w=None):
    w = np.ones(len(x)) if w is None else w
    n, _ = np.histogram(x[num_mask & den_mask], bins, weights=w[num_mask & den_mask])
    d, _ = np.histogram(x[den_mask], bins, weights=w[den_mask])
    nn, _ = np.histogram(x[num_mask & den_mask], bins)
    dd, _ = np.histogram(x[den_mask], bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = n / d
        err = np.sqrt(np.clip(nn / np.maximum(dd, 1) * (1 - nn / np.maximum(dd, 1)), 0, None) / np.maximum(dd, 1))
    f[dd < 10] = np.nan
    return f, err, dd


def cms(ax, rl):
    mplhep.cms.label("Work in progress", data=False, rlabel=rl, ax=ax, fontsize=15)
    ax.text(0.99, 0.985, rf"$p_T$ > {PT_MIN} GeV, {ETA_MIN} < $|\eta|$ < {ETA_MAX}", transform=ax.transAxes, ha="right", va="top", fontsize=12, color="0.35")


# ------------------------------------------------------------------------------------------- particle level
def plot_lost_fraction(t, out, rl):
    """Fraction of C (signal) particles that A drops; split by why (A's flag is -1 or 1)."""
    c = ak.to_numpy(ak.flatten(t.new == 0))
    old = ak.to_numpy(ak.flatten(t.old))
    pt, aeta, pid, ncons, E = (ak.to_numpy(ak.flatten(x)) for x in (t.pt, np.abs(t.eta), t.pid, t.ncons, t.E))
    lost = c & (old != 0)
    print(f"  A subset of C: {not np.any((old == 0) & ~c)};  C particles {c.sum()}, lost by A {lost.sum()} ({100 * lost.mean() / c.mean():.2f}%),"
          f" of which A=-1 {np.sum(c & (old == -1))}, A=1 {np.sum(c & (old == 1))};  energy lost {100 * E[lost].sum() / E[c].sum():.2f}%")
    for name, ids in PID_GROUPS:
        m = c & np.isin(pid, ids)
        if m.sum():
            print(f"    {name:12s}: N={m.sum():6d} lost {100 * np.mean(lost[m]):5.1f}%  (E-weighted {100 * E[lost & m].sum() / E[m].sum():5.1f}%)")
    for lo, hi in [(1, 2), (2, 3), (3, 5), (5, 99)]:
        m = c & (ncons >= lo) & (ncons < hi)
        print(f"    ncons [{lo},{hi}): frac of C {m.sum() / c.sum():.3f}  lost {100 * np.mean(lost[m]):5.1f}%")

    fig, axs = plt.subplots(2, 2, figsize=(18, 15))
    # vs pT
    ax = axs[0, 0]
    b = np.logspace(np.log10(PT_MIN), 2.5, 25)
    xc = np.sqrt(b[1:] * b[:-1])
    for why, wm, ls in (("A = -1 or 1", old != 0, "-"), ("A = -1", old == -1, "--"), ("A = 1", old == 1, ":")):
        f, e, _ = binned_frac(pt, wm, c, b)
        ax.errorbar(xc, f, yerr=e, fmt="o" + ls, color="#d62728" if why.startswith("A = -1 or") else None, label=f"{why}")
    for name, ids in PID_GROUPS[:1] + PID_GROUPS[3:]:
        f, e, _ = binned_frac(pt, old != 0, c & np.isin(pid, ids), b)
        ax.plot(xc, f, "-", lw=1.5, alpha=0.7, label=f"{name} (A = -1 or 1)")
    ax.set_xscale("log")
    ax.set_xlabel(r"target particle $p_T$ [GeV]")
    # vs |eta|
    ax = axs[0, 1]
    b = np.linspace(ETA_MIN, ETA_MAX, 16)
    xc = 0.5 * (b[1:] + b[:-1])
    f, e, _ = binned_frac(aeta, old != 0, c, b)
    ax.errorbar(xc, f, yerr=e, fmt="o-", color="#d62728", label="all")
    for name, ids in PID_GROUPS[:1] + PID_GROUPS[3:]:
        f, e, _ = binned_frac(aeta, old != 0, c & np.isin(pid, ids), b)
        ax.errorbar(xc, f, yerr=e, fmt="o-", ms=4, lw=1.5, alpha=0.8, label=name)
    ax.set_xlabel(r"target particle $|\eta|$")
    # vs PID (bar)
    ax = axs[1, 0]
    names, fr, efr, cnt = [], [], [], []
    for name, ids in PID_GROUPS:
        m = c & np.isin(pid, ids)
        if m.sum():
            names.append(name)
            fr.append(np.mean(lost[m]))
            efr.append(E[lost & m].sum() / E[m].sum())
            cnt.append(m.sum())
    x = np.arange(len(names))
    ax.bar(x - 0.2, fr, 0.4, color="#d62728", label="by count")
    ax.bar(x + 0.2, efr, 0.4, color="#ff9896", label="by energy")
    for i in range(len(names)):
        ax.text(x[i], max(fr[i], efr[i]) + 0.005, f"N={cnt[i]}", ha="center", fontsize=12)
    ax.set_xticks(x, names, fontsize=15)
    ax.minorticks_off()
    ax.set_ylim(0, max(max(fr), max(efr)) * 1.4)
    ax.set_xlabel("target particle type")
    # vs number of constituent sim tracksters
    ax = axs[1, 1]
    b = np.arange(0.5, 11.5, 1)
    xc = 0.5 * (b[1:] + b[:-1])
    f, e, d = binned_frac(np.minimum(ncons, 10), old != 0, c, b)
    ax.errorbar(xc, f, yerr=e, fmt="o-", color="#d62728", label="all")
    for name, ids in PID_GROUPS[:1] + PID_GROUPS[3:]:
        f, e, _ = binned_frac(np.minimum(ncons, 10), old != 0, c & np.isin(pid, ids), b)
        ax.errorbar(xc, f, yerr=e, fmt="o-", ms=4, lw=1.5, alpha=0.8, label=name)
    ax2 = ax.twinx()
    ax2.step(xc, d / d.sum(), where="mid", color="0.6", lw=1.5)
    ax2.set_ylabel("share of C particles (grey)", color="0.5", fontsize=16)
    ax2.set_yscale("log")
    ax.set_xlabel("N sim tracksters in the candidate (10 = overflow)")
    for ax in axs.flat:
        ax.set_ylabel("fraction of C particles dropped by A")
        ax.set_ylim(0, None)
        ax.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0, 0.95))
        cms(ax, rl)
    fig.tight_layout()
    fig.savefig(f"{out}/lost_fraction.pdf")
    plt.close(fig)

    # 2D (pT, |eta|) per PID group
    groups = [("all", None), ("photon", (22,)), ("ch. hadron", (211,)), ("neu. hadron", (130,))]
    fig, axs = plt.subplots(1, 4, figsize=(32, 8))
    bx, by = np.logspace(np.log10(PT_MIN), 2, 9), np.linspace(ETA_MIN, ETA_MAX, 7)
    for ax, (name, ids) in zip(axs, groups):
        m = c if ids is None else c & np.isin(pid, ids)
        num, _, _ = np.histogram2d(pt[m & lost], aeta[m & lost], [bx, by])
        den, _, _ = np.histogram2d(pt[m], aeta[m], [bx, by])
        with np.errstate(invalid="ignore", divide="ignore"):
            fr = np.where(den >= 20, num / den, np.nan)
        im = ax.pcolormesh(bx, by, fr.T, cmap="Reds", vmin=0, vmax=0.4)
        for i in range(len(bx) - 1):
            for j in range(len(by) - 1):
                if den[i, j] >= 20:
                    ax.text(np.sqrt(bx[i] * bx[i + 1]), 0.5 * (by[j] + by[j + 1]), f"{100 * fr[i, j]:.0f}%\n{int(den[i, j])}", ha="center", va="center", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlabel(r"$p_T$ [GeV]")
        ax.set_ylabel(r"$|\eta|$")
        ax.set_title(f"{name}: dropped by A (% , N in C)", fontsize=17)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{out}/lost_fraction_2d.pdf")
    plt.close(fig)


def gen_match(t, g, sel):
    """For each gen particle: is there an unused target particle (sel) with dR < DR_PART, compatible charge and pT ratio?"""
    matched = []
    for ev in range(len(g)):
        gg, tt = g[ev], t[ev][sel[ev]]
        gpt, geta, gphi = gg.pt.to_numpy(), gg.eta.to_numpy(), gg.phi.to_numpy()
        gq = np.isin(gg.pid.to_numpy(), GEN_CHARGED)
        tpt, teta, tphi, tq = tt.pt.to_numpy(), tt.eta.to_numpy(), tt.phi.to_numpy(), tt.q.to_numpy() != 0
        used = np.zeros(len(tpt), bool)
        m = np.zeros(len(gpt), bool)
        for k in np.argsort(-gpt):
            if not len(tpt):
                break
            d = np.hypot(teta - geta[k], dphi(tphi, gphi[k]))
            r = tpt / gpt[k]
            ok = (~used) & (tq == gq[k]) & (r > PT_RATIO[0]) & (r < PT_RATIO[1]) & (d < DR_PART)
            if ok.any():
                j = np.where(ok)[0][np.argmin(d[ok])]
                used[j] = True
                m[k] = True
        matched.append(m)
    return np.concatenate(matched)


def plot_gen_efficiency(t, g, out, rl):
    selA, selC = t.old == 0, t.new == 0
    mA, mC = gen_match(t, g, selA), gen_match(t, g, selC)
    gpt, gaeta, gpid = (ak.to_numpy(ak.flatten(x)) for x in (g.pt, np.abs(g.eta), g.pid))
    print(f"  gen particles {len(gpt)}: matched A {mA.mean():.4f}, C {mC.mean():.4f}")
    gen_groups = [("photon", (22,)), ("electron", (11,)), ("ch. hadron", (211, 321, 2212, 3222, 3112, 3312, 3334)), ("neu. hadron", (130, 2112, 310, 3122, 3322))]
    for name, ids in gen_groups:
        m = np.isin(gpid, ids)
        print(f"    gen {name:12s}: N={m.sum():6d}  eff A {mA[m].mean():.4f}  C {mC[m].mean():.4f}  (C-A {mC[m].mean() - mA[m].mean():+.4f})")

    fig, axs = plt.subplots(2, 3, figsize=(27, 13), gridspec_kw={"height_ratios": [3, 1.3]}, sharex="col")
    specs = [
        (gpt, np.logspace(np.log10(PT_MIN), 2.3, 20), True, r"gen particle $p_T$ [GeV]", None),
        (gaeta, np.linspace(ETA_MIN, ETA_MAX, 13), False, r"gen particle $|\eta|$", None),
    ]
    for col, (x, b, logx, xl, _) in enumerate(specs):
        xc = np.sqrt(b[1:] * b[:-1]) if logx else 0.5 * (b[1:] + b[:-1])
        ax, rx = axs[0, col], axs[1, col]
        effs = {}
        for k, mm in (("A", mA), ("C", mC)):
            f, e, _ = binned_frac(x, mm, np.ones(len(x), bool), b)
            effs[k] = f
            ax.errorbar(xc, f, yerr=e, fmt="o-", color=COL[k], label=LAB[k])
        rx.plot(xc, effs["A"] / effs["C"], "o-", color=COL["A"])
        for name, ids in gen_groups:
            m = np.isin(gpid, ids)
            fa, _, _ = binned_frac(x, mA, m, b)
            fc, _, _ = binned_frac(x, mC, m, b)
            rx.plot(xc, fa / fc, "-", lw=1.3, alpha=0.8, label=name)
        rx.axhline(1, color="k", ls=":")
        rx.set_ylabel("A / C")
        rx.set_ylim(0.6, 1.05)
        rx.set_xlabel(xl)
        if logx:
            ax.set_xscale("log")
        ax.set_ylabel("gen particles matched to a target particle")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=13, loc="lower right", title=rf"$\Delta R$ < {DR_PART}, {PT_RATIO[0]} < $p_T$ ratio < {PT_RATIO[1]}, same charge", title_fontsize=12)
        rx.legend(fontsize=10, loc="lower left", ncol=2)
        cms(ax, rl)
    # by gen type
    ax, rx = axs[0, 2], axs[1, 2]
    x = np.arange(len(gen_groups))
    ea = [mA[np.isin(gpid, ids)].mean() for _, ids in gen_groups]
    ec = [mC[np.isin(gpid, ids)].mean() for _, ids in gen_groups]
    ax.bar(x - 0.2, ea, 0.4, color=COL["A"], label=LAB["A"])
    ax.bar(x + 0.2, ec, 0.4, color=COL["C"], label=LAB["C"])
    rx.plot(x, np.array(ea) / np.array(ec), "o", color=COL["A"], ms=10)
    rx.axhline(1, color="k", ls=":")
    rx.set_ylim(0.6, 1.05)
    rx.set_ylabel("A / C")
    rx.set_xticks(x, [n for n, _ in gen_groups], fontsize=15)
    rx.minorticks_off()
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("gen particles matched to a target particle")
    ax.legend(fontsize=13, loc="lower right")
    cms(ax, rl)
    fig.tight_layout()
    fig.savefig(f"{out}/gen_efficiency.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- jet / event level
def cluster(p):
    vec = ak.zip({"px": p.pt * np.cos(p.phi), "py": p.pt * np.sin(p.phi), "pz": p.pt * np.sinh(p.eta), "E": p.E})
    cs = fastjet.ClusterSequence(vec, JETDEF)
    j = cs.inclusive_jets(min_pt=JET_PT_MIN)
    cons = cs.constituents(min_pt=JET_PT_MIN)
    pt = np.hypot(j.px, j.py)
    eta = np.arcsinh(j.pz / np.maximum(pt, 1e-9))
    return pt, eta, np.arctan2(j.py, j.px), cons


def plot_jets(t, g, out, rl):
    gp = g
    gjpt, gjeta, gjphi, gcons = cluster(gp)
    # photon pT fraction of each gen jet, from the gen constituents (match back by px)
    gpx = gp.pt * np.cos(gp.phi)
    gpho = ak.where(gp.pid == 22, gp.pt, 0.0)
    jets = {}
    for k, sel in (("A", t.old == 0), ("C", t.new == 0)):
        jets[k] = cluster(t[sel])[:3]
    rows = []
    for ev in range(len(g)):
        GP, GE, GF = gjpt[ev].to_numpy(), gjeta[ev].to_numpy(), gjphi[ev].to_numpy()
        pxs, phos = gpx[ev].to_numpy(), gpho[ev].to_numpy()
        for k in np.where(in_endcap(GE))[0]:
            cpx = gcons[ev][k].px.to_numpy()
            sel = np.isin(pxs, cpx)
            fpho = phos[sel].sum() / max(GP[k], 1e-9)
            row = [GP[k], abs(GE[k]), fpho]
            for key in ("A", "C"):
                TP, TE, TF = (x[ev].to_numpy() for x in jets[key])
                if len(TP):
                    d = np.hypot(TE - GE[k], dphi(TF, GF[k]))
                    j = np.argmin(d)
                    row.append(TP[j] / GP[k] if d[j] < DR_JET else np.nan)
                else:
                    row.append(np.nan)
            rows.append(row)
    R = np.array(rows)
    gpt_, geta_, fpho_, rA, rC = R.T
    print(f"  gen jets (endcap) {len(R)}: matched A {np.mean(~np.isnan(rA)):.4f}, C {np.mean(~np.isnan(rC)):.4f}")
    print(f"    response median A {np.nanmedian(rA):.3f}  C {np.nanmedian(rC):.3f};  jets with A response < 0.9: {np.nanmean(rA < 0.9):.3f}, C {np.nanmean(rC < 0.9):.3f}")

    fig, axs = plt.subplots(2, 3, figsize=(27, 13), gridspec_kw={"height_ratios": [3, 1.3]}, sharex="col")
    specs = [
        (gpt_, np.array([15, 20, 30, 45, 70, 110, 200, 500]), True, r"gen jet $p_T$ [GeV]"),
        (geta_, np.linspace(ETA_MIN, ETA_MAX, 7), False, r"gen jet $|\eta|$"),
        (fpho_, np.array([0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.01]), False, r"gen jet photon $p_T$ fraction"),
    ]
    for col, (x, b, logx, xl) in enumerate(specs):
        ax, rx = axs[0, col], axs[1, col]
        xc = np.sqrt(b[1:] * b[:-1]) if logx else 0.5 * (b[1:] + b[:-1])
        for key, r in (("A", rA), ("C", rC)):
            med, lo, hi, eff = [], [], [], []
            for i in range(len(b) - 1):
                m = (x >= b[i]) & (x < b[i + 1])
                rr = r[m & ~np.isnan(r)]
                eff.append(np.mean(~np.isnan(r[m])) if m.sum() >= 10 else np.nan)
                if len(rr) >= 10:
                    q = np.percentile(rr, [25, 50, 75])
                else:
                    q = [np.nan] * 3
                lo.append(q[0])
                med.append(q[1])
                hi.append(q[2])
            ax.plot(xc, med, "o-", color=COL[key], label=f"{LAB[key]}: median")
            ax.fill_between(xc, lo, hi, color=COL[key], alpha=0.2, label="IQR")
            rx.plot(xc, eff, "o-", color=COL[key])
        ax.axhline(1, color="k", ls=":")
        ax.set_ylabel(r"jet $p_T^{\mathrm{target}} / p_T^{\mathrm{gen}}$")
        ax.set_ylim(0.6, 1.25)
        rx.set_ylabel("matched frac.")
        rx.set_ylim(0.7, 1.02)
        rx.set_xlabel(xl)
        if logx:
            ax.set_xscale("log")
            rx.set_xticks([20, 30, 50, 100, 200])
            rx.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            rx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim(b[0], min(b[-1], np.nanmax(x) * 1.05))
        n, _ = np.histogram(x, b)
        for i in range(len(xc)):
            ax.text(xc[i], 0.63, f"{n[i]}", ha="center", fontsize=11, color="0.4", clip_on=True)
        ax.legend(fontsize=12, loc="upper left", title=rf"anti-$k_T$ R=0.4 > {JET_PT_MIN:.0f} GeV, $\Delta R$ < {DR_JET}; grey = N gen jets", title_fontsize=11)
        cms(ax, rl)
    fig.tight_layout()
    fig.savefig(f"{out}/jet_response.pdf")
    plt.close(fig)

    # per-event sum pT
    sg = ak.to_numpy(ak.sum(g.pt, axis=1))
    fig, ax = plt.subplots(figsize=(10, 8))
    b = np.linspace(0.4, 1.6, 61)
    for key, sel in (("A", t.old == 0), ("C", t.new == 0)):
        r = ak.to_numpy(ak.sum(t.pt[sel], axis=1)) / sg
        ax.hist(np.clip(r, b[0], b[-1] - 1e-6), b, histtype="step", lw=2.5, color=COL[key], label=f"{LAB[key]}: median {np.median(r):.3f}, IQR {np.subtract(*np.percentile(r, [75, 25])):.3f}")
        print(f"    event sum pT ratio {key}: median {np.median(r):.3f}")
    ax.axvline(1, color="k", ls=":")
    ax.set_xlabel(r"event $\sum p_T^{\mathrm{target}} / \sum p_T^{\mathrm{gen}}$ (endcap, $p_T$ > 0.5 GeV)")
    ax.set_ylabel("events")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.4)
    ax.legend(fontsize=12, loc="upper left")
    cms(ax, rl)
    fig.tight_layout()
    fig.savefig(f"{out}/event_sumpt.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="ttbar_200pu")
    ap.add_argument("--nfiles", type=int, default=100)
    args = ap.parse_args()
    mplhep.style.use("CMS")
    t, g, nf = load(args.sample, args.nfiles)
    out = f"plots/phase2/isPU_A_vs_C_corners/{args.sample}"
    os.makedirs(out, exist_ok=True)
    rl = {"ttbar_200pu": r"$t\bar{t}$ 200 PU", "ttbar_0pu": r"$t\bar{t}$ 0 PU"}.get(args.sample, args.sample) + f" ({len(t)} ev)"
    print(f"===== {args.sample}: {nf} files, {len(t)} events")
    plot_lost_fraction(t, out, rl)
    plot_gen_efficiency(t, g, out, rl)
    plot_jets(t, g, out, rl)
    print("saved", out)


if __name__ == "__main__":
    main()
