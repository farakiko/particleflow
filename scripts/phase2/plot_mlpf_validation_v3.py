#!/usr/bin/env python3
"""Paper-ready MLPF validation plots from the evaluation parquets (preds_test):
the EXACT panel suite of plot_target_validation_v3.py with one extra curve —
MLPF (green) — alongside gen (dashed black), target (blue) and TICL (red).

Reads the run_predictions output (particles.target/cand/pred + pythia + genmet);
jets are clustered HERE from the particle lists (anti-kT 0.4, pT>3) so every
convention matches the target-validation suite exactly:
  - particle-level panels: pT > --pt-min on EVERY curve (spectra uncut)
  - gen denominators windowed to --gen-window (default 1.5<|eta|<3.0)
  - jet-eta panel: jet pT > --jet-pt-min on every curve (jet-pT spectrum uncut)
  - response/resolution: DR<0.2, gen-jet pT>20, axis 1.5<|eta|<3.0
  - eff/fake per class: DR<0.2 same-class; curves for target, TICL and MLPF
  - CMS Simulation + Work in progress; pdf only

  python3 scripts/phase2/plot_mlpf_validation_v3.py \
      --preds-glob '<exp>/preds_test/cms_pf_ticl_nopu/*.parquet' \
      --outdir <dir> [--max-files N]
"""
import argparse
import glob
import os

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import mplhep
import numpy as np

mplhep.style.use("CMS")

CLASSES = [("chad", "Charged hadrons"), ("nhad", "Neutral hadrons"),
           ("photon", "Photons"), ("electron", "Electrons"), ("muon", "Muons")]
CLS_IDX = {1: "chad", 2: "nhad", 3: "photon", 4: "electron", 5: "muon"}
NHAD_PIDS = {130, 310, 2112, 3122, 3322, 3212, 421, 511}
NEUTRINOS = {12, 14, 16}
C = {"gen": "black", "target": "#1f77b4", "cand": "#d62728", "pred": "#2ca02c"}
LBL = {"gen": "Gen", "target": "Target", "cand": "TICL", "pred": "MLPF"}
KEYS = ["gen", "target", "cand", "pred"]
RECO = ["target", "cand", "pred"]
DR_PART, DR_JET, JET_PTMIN, JET_ETA = 0.2, 0.2, 20.0, (1.5, 3.0)
PT_BINS = np.array([1, 2, 3, 5, 8, 13, 20, 35, 60, 100, 200.0])
JPT_BINS = np.array([20, 30, 40, 60, 80, 100, 150, 200, 400.0])


def pclass(pid):
    a = abs(int(pid))
    if a == 22: return "photon"
    if a == 11: return "electron"
    if a == 13: return "muon"
    return "nhad" if a in NHAD_PIDS else "chad"


def cms(ax, proc):
    mplhep.cms.label(ax=ax, data=False, label="Work in progress", rlabel=f"{proc} (14 TeV)", fontsize=20)


def sv(fig, outdir, name, formats):
    fig.tight_layout()
    for f in formats:
        fig.savefig(os.path.join(outdir, f"{name}.{f}"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def load(files):
    P = {k: {f: [] for f in ["pt", "eta", "phi", "cls", "energy"]} for k in KEYS}
    met = {k: [] for k in KEYS}
    nev = 0
    for fn in files:
        try:
            d = ak.from_parquet(fn, columns=["particles", "pythia"])
        except Exception as e:
            print(f"  [WARN] {fn}: {e}")
            continue
        for ie in range(len(d)):
            nev += 1
            py = np.asarray(d["pythia"][ie], np.float32).reshape(-1, 5)
            real = ~((py[:, 0] == 0) & (py[:, 1] == 0))          # strip batch padding
            py = py[real]
            vis = ~np.isin(np.abs(py[:, 0]).astype(int), list(NEUTRINOS))
            py = py[vis]
            P["gen"]["pt"].append(py[:, 1]); P["gen"]["eta"].append(py[:, 2])
            P["gen"]["phi"].append(py[:, 3]); P["gen"]["energy"].append(py[:, 4])
            P["gen"]["cls"].append(np.array([pclass(p) for p in py[:, 0]]))
            for key in RECO:
                t = d["particles"][key][ie]
                cls = np.asarray(t["cls_id"], np.int32)
                v = cls != 0
                pt = np.asarray(t["pt"], np.float32)[v]
                eta = np.asarray(t["eta"], np.float32)[v]
                phi = np.arctan2(np.asarray(t["sin_phi"], np.float32)[v],
                                 np.asarray(t["cos_phi"], np.float32)[v])
                en = np.asarray(t["energy"], np.float32)[v]
                P[key]["pt"].append(pt); P[key]["eta"].append(eta)
                P[key]["phi"].append(phi); P[key]["energy"].append(en)
                P[key]["cls"].append(np.array([CLS_IDX.get(c, "chad") for c in cls[v]]))
            for key in KEYS:
                px = np.sum(P[key]["pt"][-1] * np.cos(P[key]["phi"][-1]))
                pyy = np.sum(P[key]["pt"][-1] * np.sin(P[key]["phi"][-1]))
                met[key].append(float(np.hypot(px, pyy)))
    return P, met, nev


def cluster(P, key, nev):
    import fastjet
    jd = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    out = []
    for ie in range(nev):
        pt, eta, phi, en = P[key]["pt"][ie], P[key]["eta"][ie], P[key]["phi"][ie], P[key]["energy"][ie]
        ok = (pt > 0) & np.isfinite(eta) & np.isfinite(phi)
        pt, eta, phi, en = pt[ok], eta[ok], phi[ok], en[ok]
        if not len(pt):
            out.append(np.zeros((0, 4), np.float32))
            continue
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(np.clip(eta, -9, 9))
        pjs = [fastjet.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(en[i])) for i in range(len(pt))]
        js = fastjet.ClusterSequence(pjs, jd).inclusive_jets(ptmin=3.0)
        out.append(np.array([[j.pt(), j.eta(), j.phi(), j.e()] for j in js], np.float32).reshape(-1, 4))
    return out


def match_ratio(ref_jets, jets_):
    rp, tp = [], []
    for r, t in zip(ref_jets, jets_):
        if not len(r):
            continue
        sel = (r[:, 0] > JET_PTMIN) & (np.abs(r[:, 1]) > JET_ETA[0]) & (np.abs(r[:, 1]) < JET_ETA[1])
        for j in r[sel]:
            if not len(t):
                continue
            d = np.hypot(t[:, 1] - j[1], np.arctan2(np.sin(t[:, 2] - j[2]), np.cos(t[:, 2] - j[2])))
            k = int(np.argmin(d))
            if d[k] < DR_JET:
                rp.append(j[0]); tp.append(t[k, 0])
    return np.array(rp), np.array(tp)


def step(ax, vals, bins, key, extra="", density=False):
    ax.hist(vals, bins=bins, histtype="step", lw=2.5, color=C[key],
            ls="--" if key == "gen" else "-", density=density,
            label=f"{LBL[key]}{extra}")


def logticks(ax, ticks):
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(ticks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-glob", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-files", type=int, default=300)
    ap.add_argument("--proc-label", default=r"$\mathrm{t}\bar{\mathrm{t}}$+QCD+DY, 0 PU")
    ap.add_argument("--formats", nargs="+", default=["pdf"])
    ap.add_argument("--pt-min", type=float, default=1.0)
    ap.add_argument("--jet-pt-min", type=float, default=10.0)
    ap.add_argument("--gen-window", nargs=2, type=float, default=[1.5, 3.0])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    files = sorted(glob.glob(a.preds_glob))[: a.max_files]
    assert files, f"no files match {a.preds_glob}"
    print(f"{len(files)} parquet files")
    P, met, nev = load(files)
    print(f"{nev} events; clustering jets for {KEYS}...")
    jets = {k: cluster(P, k, nev) for k in KEYS}
    proc = a.proc_label
    glo, ghi = a.gen_window
    genw_note = rf"gen: ${glo}<|\eta|<{ghi}$"
    ptm = a.pt_min
    ptm_note = rf"$p_\mathrm{{T}}>{ptm:g}$ GeV"
    PTM = {k: [P[k]["pt"][ie] > ptm for ie in range(nev)] for k in KEYS}
    GW = [(np.abs(P["gen"]["eta"][ie]) > glo) & (np.abs(P["gen"]["eta"][ie]) < ghi) for ie in range(nev)]
    F = {k: {f: np.concatenate(P[k][f]) for f in ["pt", "eta", "cls"]} for k in KEYS}
    FPT = {k: np.concatenate(PTM[k]) for k in KEYS}

    # ---- 1. eta panels (floored)
    fig, ax = plt.subplots(figsize=(11, 9))
    b = np.linspace(-4, 4, 81)
    for k in KEYS:
        vals = F[k]["eta"][FPT[k]]
        step(ax, vals, b, k, extra=f"  ({len(vals)/nev:.1f}/event)")
    ax.text(0.03, 0.85, ptm_note, transform=ax.transAxes, fontsize=17)
    ax.set_xlabel(r"particle $\eta$")
    ax.set_ylabel("Particles / bin")
    ax.legend(loc="upper center", fontsize=17)
    cms(ax, proc)
    sv(fig, a.outdir, "particle_eta_all", a.formats)

    for cl, cname in CLASSES:
        fig, ax = plt.subplots(figsize=(11, 9))
        for k in KEYS:
            m = (F[k]["cls"] == cl) & FPT[k]
            step(ax, F[k]["eta"][m], b, k, extra=f"  (n={int(m.sum())})")
        ax.text(0.03, 0.85, ptm_note, transform=ax.transAxes, fontsize=17)
        ax.set_xlabel(rf"{cname}: $\eta$")
        ax.set_ylabel("Particles / bin")
        ax.set_yscale("log")
        ax.legend(loc="lower center", fontsize=15, ncol=2)
        cms(ax, proc)
        sv(fig, a.outdir, f"particle_eta_{cl}", a.formats)

    # ---- 2. per-class pT spectra (uncut)
    bpt = np.logspace(-1.3, 2.7, 60)
    for cl, cname in CLASSES:
        fig, ax = plt.subplots(figsize=(11, 9))
        for k in KEYS:
            m = F[k]["cls"] == cl
            step(ax, F[k]["pt"][m], bpt, k, extra=f"  (n={int(m.sum())})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(rf"{cname}: $p_\mathrm{{T}}$ [GeV]")
        ax.set_ylabel("Particles / bin")
        ax.legend(loc="upper right", fontsize=15)
        cms(ax, proc)
        sv(fig, a.outdir, f"particle_pt_{cl}", a.formats)

    # ---- 3. multiplicity + sumpt ratios (floored; gen windowed in denominators)
    fig, ax = plt.subplots(figsize=(11, 9))
    b = np.linspace(0, 160, 81)
    for k in KEYS:
        arr = np.array([int(m.sum()) for m in PTM[k]])
        step(ax, arr, b, k, extra=f"  (mean {arr.mean():.1f})")
    ax.text(0.7, 0.6, ptm_note, transform=ax.transAxes, fontsize=17)
    ax.set_xlabel("Particles / event")
    ax.set_ylabel("Events / bin")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=17)
    cms(ax, proc)
    sv(fig, a.outdir, "multiplicity", a.formats)

    fig, ax = plt.subplots(figsize=(11, 9))
    b = np.linspace(0, 2, 81)
    g = np.array([float(P["gen"]["pt"][ie][GW[ie] & PTM["gen"][ie]].sum()) for ie in range(nev)])
    ok = g > 10
    for k in RECO:
        sk = np.array([float(P[k]["pt"][ie][PTM[k][ie]].sum()) for ie in range(nev)])
        r = sk[ok] / g[ok]
        step(ax, r, b, k, extra=rf"  (median {np.median(r):.3f})", density=True)
    ax.axvline(1, color="gray", ls=":", lw=1.5)
    ax.set_xlabel(rf"$\Sigma p_\mathrm{{T}}$ / $\Sigma p_\mathrm{{T}}^\mathrm{{gen}}$  per event   ({genw_note}, {ptm_note})")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=17)
    cms(ax, proc)
    sv(fig, a.outdir, "sumpt_ratio", a.formats)

    # ---- 4. jets
    fig, ax = plt.subplots(figsize=(11, 9))
    bj = np.logspace(np.log10(3), 2.9, 50)
    for k in KEYS:
        arr = np.concatenate([j[:, 0] for j in jets[k] if len(j)] or [np.zeros(0)])
        step(ax, arr, bj, k, extra=f"  ({len(arr)/nev:.1f}/event)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_\mathrm{T}$ [GeV]")
    ax.set_ylabel("Jets / bin")
    ax.legend(loc="upper right", fontsize=17)
    cms(ax, proc)
    sv(fig, a.outdir, "jet_pt", a.formats)

    fig, ax = plt.subplots(figsize=(11, 9))
    b = np.linspace(-4, 4, 81)
    jptm = a.jet_pt_min
    for k in KEYS:
        arr = np.concatenate([j[j[:, 0] > jptm, 1] for j in jets[k] if len(j)] or [np.zeros(0)])
        step(ax, arr, b, k, extra=f"  ({len(arr)/nev:.2f}/event)")
    ax.text(0.03, 0.85, rf"jet $p_\mathrm{{T}}>{jptm:g}$ GeV", transform=ax.transAxes, fontsize=17)
    ax.set_xlabel(r"jet $\eta$")
    ax.set_ylabel("Jets / bin")
    ax.legend(loc="upper center", fontsize=15, ncol=2)
    cms(ax, proc)
    sv(fig, a.outdir, "jet_eta", a.formats)

    matches = {k: match_ratio(jets["gen"], jets[k]) for k in RECO}
    fig, ax = plt.subplots(figsize=(11, 9))
    b = np.linspace(0, 2, 101)
    for k in RECO:
        rp, tp = matches[k]
        r = tp / rp
        q1, q2, q3 = np.percentile(r, [25, 50, 75])
        step(ax, r, b, k, extra=rf"  (med {q2:.3f}, IQR {q3-q1:.3f})", density=True)
    ax.axvline(1, color="gray", ls=":", lw=1.5)
    ax.set_xlabel(rf"jet $p_\mathrm{{T}}$ / gen-jet $p_\mathrm{{T}}$   ($p_\mathrm{{T}}^\mathrm{{gen}}>{JET_PTMIN:.0f}$ GeV, ${JET_ETA[0]}<|\eta|<{JET_ETA[1]}$)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left", fontsize=15)
    cms(ax, proc)
    sv(fig, a.outdir, "jet_response", a.formats)

    centers = np.sqrt(JPT_BINS[:-1] * JPT_BINS[1:])
    for name, ylab in [("jet_response_vs_pt", r"jet $p_\mathrm{T}$ / gen-jet $p_\mathrm{T}$"),
                       ("jet_resolution_vs_pt", "jet response IQR / median")]:
        fig, ax = plt.subplots(figsize=(11, 9))
        for k in RECO:
            rp, tp = matches[k]
            r = tp / rp
            xs, med, lo, hi, res = [], [], [], [], []
            for l_, h_ in zip(JPT_BINS[:-1], JPT_BINS[1:]):
                m = (rp >= l_) & (rp < h_)
                if m.sum() < 20:
                    continue
                q1, q2, q3 = np.percentile(r[m], [25, 50, 75])
                xs.append(np.sqrt(l_ * h_)); med.append(q2); lo.append(q1); hi.append(q3)
                res.append((q3 - q1) / q2 if q2 > 0 else np.nan)
            if name == "jet_response_vs_pt":
                ax.plot(xs, med, "o-", lw=2.5, color=C[k], label=f"{LBL[k]} median")
                ax.fill_between(xs, lo, hi, color=C[k], alpha=0.15, label=f"{LBL[k]} IQR")
            else:
                ax.plot(xs, res, "o-", lw=2.5, color=C[k], label=LBL[k])
        if name == "jet_response_vs_pt":
            ax.axhline(1, color="gray", ls=":", lw=1.5)
            ax.set_ylim(0, 1.6)
        else:
            ax.set_ylim(0, None)
        ax.set_xscale("log")
        logticks(ax, [20, 30, 50, 100, 200, 400])
        ax.set_xlabel(r"gen-jet $p_\mathrm{T}$ [GeV]")
        ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=14, ncol=2)
        ax.grid(alpha=0.3)
        cms(ax, proc)
        sv(fig, a.outdir, name, a.formats)

    # ---- 5. MET
    fig, ax = plt.subplots(figsize=(11, 9))
    b = np.logspace(-0.5, 2.7, 60)
    for k in KEYS:
        arr = np.array(met[k])
        step(ax, arr, b, k, extra=f"  (median {np.median(arr):.1f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_\mathrm{T}^\mathrm{miss}$ (visible sum) [GeV]")
    ax.set_ylabel("Events / bin")
    ax.legend(loc="upper right", fontsize=15)
    cms(ax, proc)
    sv(fig, a.outdir, "met", a.formats)

    fig, ax = plt.subplots(figsize=(11, 9))
    b = np.linspace(0, 3, 76)
    gmet = np.array([float(np.hypot(np.sum(P["gen"]["pt"][ie][GW[ie]] * np.cos(P["gen"]["phi"][ie][GW[ie]])),
                                    np.sum(P["gen"]["pt"][ie][GW[ie]] * np.sin(P["gen"]["phi"][ie][GW[ie]])))) for ie in range(nev)])
    okm = gmet > 5
    for k in RECO:
        r = np.array(met[k])[okm] / gmet[okm]
        step(ax, r, b, k, extra=rf"  (median {np.median(r):.2f})", density=True)
    ax.axvline(1, color="gray", ls=":", lw=1.5)
    ax.set_xlabel(rf"$p_\mathrm{{T}}^\mathrm{{miss}}$ / $p_\mathrm{{T}}^\mathrm{{miss,gen}}$   ($>5$ GeV, {genw_note})")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=15)
    cms(ax, proc)
    sv(fig, a.outdir, "met_ratio", a.formats)

    # ---- 6. efficiency & fake rate per class (gen windowed+floored; 3 curves)
    for cl, cname in CLASSES:
        res = {}
        for key in RECO:
            num = np.zeros(len(PT_BINS) - 1); den = np.zeros(len(PT_BINS) - 1)
            fnum = np.zeros(len(PT_BINS) - 1); fden = np.zeros(len(PT_BINS) - 1)
            for ie in range(nev):
                gm = (P["gen"]["cls"][ie] == cl) & GW[ie] & PTM["gen"][ie]
                tm = (P[key]["cls"][ie] == cl) & PTM[key][ie]
                ge, gp_, gpt = P["gen"]["eta"][ie][gm], P["gen"]["phi"][ie][gm], P["gen"]["pt"][ie][gm]
                te, tp_, tpt = P[key]["eta"][ie][tm], P[key]["phi"][ie][tm], P[key]["pt"][ie][tm]
                for i in range(len(gpt)):
                    bidx = np.clip(np.digitize(gpt[i], PT_BINS) - 1, 0, len(PT_BINS) - 2)
                    den[bidx] += 1
                    if len(te):
                        d = np.hypot(te - ge[i], np.arctan2(np.sin(tp_ - gp_[i]), np.cos(tp_ - gp_[i])))
                        if d.min() < DR_PART:
                            num[bidx] += 1
                for i in range(len(tpt)):
                    bidx = np.clip(np.digitize(tpt[i], PT_BINS) - 1, 0, len(PT_BINS) - 2)
                    fden[bidx] += 1
                    if len(ge):
                        d = np.hypot(ge - te[i], np.arctan2(np.sin(gp_ - tp_[i]), np.cos(gp_ - tp_[i])))
                        if d.min() < DR_PART:
                            continue
                    fnum[bidx] += 1
            res[key] = (num, den, fnum, fden)
        centers_p = np.sqrt(PT_BINS[:-1] * PT_BINS[1:])
        for kind, idx, ylab in [("efficiency", (0, 1), "Efficiency"), ("fakerate", (2, 3), "Fake rate")]:
            fig, ax = plt.subplots(figsize=(11, 9))
            for key in RECO:
                num, den = res[key][idx[0]], res[key][idx[1]]
                ok = den >= 25
                ax.errorbar(centers_p[ok], num[ok] / den[ok],
                            yerr=np.sqrt(np.clip(num[ok], 1, None)) / den[ok],
                            fmt="o-", lw=2.5, ms=8, color=C[key], label=LBL[key])
            ax.set_xscale("log")
            logticks(ax, [1, 2, 5, 10, 20, 50, 100, 200])
            ax.set_ylim(0, 1.15)
            ax.set_xlabel(rf"{cname}: $p_\mathrm{{T}}$ [GeV]")
            ax.set_ylabel(f"{ylab}  ($\\Delta R<{DR_PART}$, same class)")
            note = f"{genw_note}, all curves {ptm_note}" if kind == "efficiency" else f"all curves {ptm_note}"
            ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=14, va="top")
            ax.legend(loc="best", fontsize=16)
            ax.grid(alpha=0.3)
            cms(ax, proc)
            sv(fig, a.outdir, f"{kind}_{cl}", a.formats)

    print(f"done -> {a.outdir}")


if __name__ == "__main__":
    main()
