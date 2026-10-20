#!/usr/bin/env python
"""
Phase-2 200PU MLPF evaluation: reconstruction vs generator, after pileup mitigation.

Reads the per-event inference parquet files (produced by `mlpf/pipeline.py ... test`, which
MUST contain the `pythia` gen-particle branch) and produces two directories of paper-ready
plots comparing the reconstruction -- after pileup mitigation -- to the generator (pythia):

  <out>/ideal_pu_removal/   pileup removed using the TRUTH isPU flag. This is the ideal
                            ceiling; every plot is annotated "optimal pileup removal using
                            truth info".
  <out>/model_pu_removal/   pileup removed by each model's own output:
                              "Baseline MLPF"              -> no removal
                              "MLPF + PU head"             -> drop predicted P(pileup) >= 0.5
                              "MLPF + pt weighted PU head" -> drop predicted P(pileup) >= 0.5

Plots produced in each directory (one curve per model + a gen reference where relevant):
  distributions/   particle pT and eta spectra -- all particles, and split per type
  sumpt_response/  per-event  sum-pT(reco) / sum-pT(gen)  per particle type
  efficiency/      efficiency vs pT and vs eta, per type; two matching variants (same-PID or not)
  fake_rate/       fake rate  vs pT and vs eta, per type; two matching variants
  jets/            jet pT spectrum, inclusive response (median+IQR in legend),
                   response vs pT, resolution (IQR/median) vs pT

Matching (particles): greedy nearest reco<->gen within dR < DR (default 0.1). The "same-PID"
variant additionally requires the reconstructed type to equal the gen type. Everything is
restricted to the HGCAL endcap (|eta| in ENDCAP), since MLPF reconstructs the endcap only.
Jets: anti-kt R=0.4, pT>JET_PTCUT, matched to gen jets within dR<DR.

Usage (run inside the mlpf venv, which has fastjet/vector/awkward/mplhep):
  python eval_phase2_plots.py \
      --preds-root /shared/mlpf-phase2/experiments \
      --out        /shared/mlpf-phase2/eval_plots/phase2_physics \
      --nfiles 1000

To adapt to new models/samples, edit the MODELS / SAMPLE block below. The preds for each model
are read from <preds-root>/<exp>/preds_test/<SAMPLE>/pred_*.parquet.
"""
import os, glob, argparse
import numpy as np
import awkward as ak
import fastjet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

# ----------------------------------------------------------------------------- configuration
SAMPLE = "cms_pf_phase2_ttbar200pu"
RLABEL = r"$\mathrm{t}\bar{\mathrm{t}}$, 200 PU (14 TeV)"   # top-right CMS label
# (display label, experiment subdir, colour, mitigation: "none"=keep all, "head"=drop pred P(pu)>=cut)
MODELS = [
    ("Baseline MLPF",              "ttbar200-heptv2",         "#2ca02c", "none"),
    ("MLPF + PU head",             "ttbar200-heptv2-pu",      "#1f77b4", "head"),
    ("MLPF + pt weighted PU head", "ttbar200-heptv2-pu-rew",  "#d62728", "head"),
    ("MLPF + soft PU head (T2)",   "ttbar200-heptv2-pu-soft", "#9467bd", "head"),
    ("MLPF + PU head + ctx (T3)",  "ttbar200-heptv2-pu-ctx",  "#8c564b", "head"),
    ("MLPF + PU head + feats (T1)", "ttbar200-heptv2-pu-feats", "#e377c2", "head"),
]
PU_CUT = 0.5
DR = 0.1
ENDCAP = (1.48, 3.0)
NU = {12, 14, 16}
JET_PTCUT = 15.0
# reco class index (cms_phase2 CLASS_LABELS = [0,211,130,22,11,13]) -> name
TYPES = [(1, "chrg_had"), (2, "neut_had"), (3, "photon"), (4, "electron"), (5, "muon")]
NEUT_HAD = {130, 2112, 310, 311, 3122, 3212, 3322}  # |pdgid| of neutral hadrons
PT_BINS = np.array([1, 2, 3, 5, 8, 12, 20, 35, 60, 120])      # particle pT bins (eff/fake)
ETA_BINS = np.linspace(ENDCAP[0], ENDCAP[1], 9)               # particle eta bins
JET_PT_BINS = np.array([15, 30, 50, 90, 160, 300])            # gen-jet pT bins
_jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)


# ----------------------------------------------------------------------------- helpers
def cms_label(ax, note=None):
    # Optional descriptor (e.g. the ideal-mitigation note) as a subtitle line above the frame,
    # so it never overlaps the curves or legend. bbox_inches="tight" keeps it in the saved PDF.
    rlabel = RLABEL
    hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=rlabel, fontsize=16)
    if note:
        ax.text(0.5, 1.13, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=13, style="italic", color="0.30")


def classify_gen(pdgid):
    """pythia pdgid -> reco type index (1..5); 0 = drop (neutrino / unknown)."""
    a = np.abs(pdgid).astype(int)
    out = np.zeros_like(a)
    out[a == 22] = 3
    out[a == 11] = 4
    out[a == 13] = 5
    isnu = np.isin(a, list(NU))
    rest = (a != 22) & (a != 11) & (a != 13) & (~isnu)
    isneu = np.isin(a, list(NEUT_HAD))
    out[rest & isneu] = 2          # neutral hadron
    out[rest & (~isneu)] = 1       # charged hadron
    out[isnu] = 0
    return out


def load(exp, preds_root, nfiles):
    fs = sorted(glob.glob(f"{preds_root}/{exp}/preds_test/{SAMPLE}/pred_*.parquet"))[:nfiles]
    if not fs:
        raise FileNotFoundError(f"no preds for {exp} under {preds_root}")
    arrs = []
    for f in fs:
        try:
            arrs.append(ak.from_parquet(f))
        except Exception as e:  # noqa
            print(f"  skip {f}: {e}")
    return ak.concatenate(arrs)


def reco_particles(a, mitig):
    """Return per-event reco particle dict (pt,eta,phi,type) after pileup mitigation.
    mitig: 'none' | 'head' | 'truth'. Masks are per-element (pred & target are aligned)."""
    pr = a["particles"]["pred"]
    keep = (pr["cls_id"] != 0) & (np.abs(pr["eta"]) >= ENDCAP[0]) & (np.abs(pr["eta"]) <= ENDCAP[1])
    if mitig == "head":
        keep = keep & (pr["ispu"] < PU_CUT)
    elif mitig == "truth":
        keep = keep & (a["particles"]["target"]["ispu"] < PU_CUT)
    return dict(pt=pr["pt"][keep], eta=pr["eta"][keep], phi=pr["phi"][keep], type=pr["cls_id"][keep])


def gen_particles(a):
    """Per-event gen (pythia) particle dict (pt,eta,phi,type), endcap, non-neutrino."""
    py = a["pythia"]
    pdg = py[..., 0]; pt = py[..., 1]; eta = py[..., 2]; phi = py[..., 3]
    typ = ak.Array([classify_gen(ak.to_numpy(p)) for p in pdg])
    keep = (pt > 0) & (typ > 0) & (np.abs(eta) >= ENDCAP[0]) & (np.abs(eta) <= ENDCAP[1])
    return dict(pt=pt[keep], eta=eta[keep], phi=phi[keep], type=typ[keep])


def ec_flat(coll):
    """Flatten a per-event particle dict to numpy arrays, endcap only."""
    eta = ak.to_numpy(ak.flatten(coll["eta"]))
    m = (np.abs(eta) >= ENDCAP[0]) & (np.abs(eta) <= ENDCAP[1])
    return {k: ak.to_numpy(ak.flatten(v))[m] for k, v in coll.items()}


def jets_from(coll):
    """Cluster anti-kt R=0.4 jets (pT>JET_PTCUT) from a per-event particle dict."""
    pt, eta, phi = coll["pt"], coll["eta"], coll["phi"]
    vec = ak.zip({"px": pt * np.cos(phi), "py": pt * np.sin(phi),
                  "pz": pt * np.sinh(eta), "E": np.sqrt((pt * np.cosh(eta)) ** 2)})
    j = fastjet.ClusterSequence(vec, _jetdef).inclusive_jets(min_pt=JET_PTCUT)
    return j.px, j.py, j.pz


def match_event(ge, gp, gt, re, rp, rt, samepid):
    """Greedy nearest reco<->gen match within DR. Returns (gen_matched, reco_matched) bool."""
    gm = np.zeros(len(ge), bool); rm = np.zeros(len(re), bool); used = np.zeros(len(re), bool)
    for i in range(len(ge)):
        if not len(re):
            break
        dphi = np.arctan2(np.sin(rp - gp[i]), np.cos(rp - gp[i]))
        d = np.hypot(re - ge[i], dphi)
        if samepid:
            d = np.where(rt == gt[i], d, 99.0)
        d = np.where(used, 99.0, d)
        j = int(np.argmin(d))
        if d[j] < DR:
            gm[i] = True; rm[j] = True; used[j] = True
    return gm, rm


# ----------------------------------------------------------------------------- plot routines
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  ->", os.path.relpath(path))


def plot_spectrum(reco, gen, which, outdir, note, type_sel=None, type_name="all"):
    """pT or eta spectrum; which in {'pt','eta'}. type_sel: reco/gen type index or None (all)."""
    if which == "pt":
        bins = np.logspace(0, 2.4, 45); xlabel = r"particle $p_T$ [GeV]"; logx = True
    else:
        bins = np.linspace(ENDCAP[0], ENDCAP[1], 40); xlabel = r"particle $|\eta|$"; logx = False
    fig, ax = plt.subplots(figsize=(8, 7))

    def sel(d, t):
        v = d[which] if which == "pt" else np.abs(d["eta"])
        return v if t is None else v[d["type"] == t]

    g = sel(gen, type_sel)
    ax.hist(g, bins=bins, histtype="step", lw=2.6, color="k", ls="--", label=f"gen ({len(g)})")
    for label, _, c, _m in MODELS:
        d = reco[label]; v = sel(d, type_sel)
        ax.hist(v, bins=bins, histtype="step", lw=2.3, color=c, label=f"{label} ({len(v)})")
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel("particles")
    ax.legend(fontsize=11, title=None if type_name == "all" else type_name)
    cms_label(ax, note)
    _save(fig, f"{outdir}/particle_{which}_{type_name}.pdf")


def plot_sumpt_response(reco_ev, gen_ev, outdir, note):
    """per-event sum-pT(reco)/sum-pT(gen), per type. gen_ev is per-model (events are
    ordered differently in each model's preds, so reco and gen must come from the same model)."""
    for cid, nm in TYPES:
        fig, ax = plt.subplots(figsize=(8, 7)); b = np.linspace(0, 2.5, 50)
        for label, _, c, _m in MODELS:
            r = reco_ev[label]; g = gen_ev[label]
            num = ak.to_numpy(ak.sum(r["pt"][r["type"] == cid], axis=1))
            den = ak.to_numpy(ak.sum(g["pt"][g["type"] == cid], axis=1))
            ok = den > 0
            ratio = num[ok] / den[ok]
            md = np.median(ratio) if len(ratio) else np.nan
            ax.hist(ratio, bins=b, histtype="step", lw=2.3, color=c, label=f"{label} (med {md:.2f})")
        ax.axvline(1, color="k", ls=":")
        ax.set_xlabel(r"$\sum p_T^{\mathrm{reco}} / \sum p_T^{\mathrm{gen}}$ per event")
        ax.set_ylabel("events"); ax.legend(fontsize=11, title=nm); cms_label(ax, note)
        _save(fig, f"{outdir}/sumpt_response_{nm}.pdf")


def plot_eff_fake(reco_ev, gen_ev, outdir, note, samepid):
    """Efficiency (gen-based) and fake rate (reco-based) vs pT and vs eta, per type."""
    cond = "samePID" if samepid else "anyPID"
    cond_txt = r"Matching criterion: $\Delta R < %.2f$ with Gen%s" % (DR, ", same PID" if samepid else "")
    PT_MIN = 1.0  # require particle pT > 1 GeV
    # accumulate per model: counts[model][type][obs] = (num, den) arrays over bins
    PT, ETA = PT_BINS, ETA_BINS
    acc = {lbl: {cid: dict(eff_pt_n=np.zeros(len(PT) - 1), eff_pt_d=np.zeros(len(PT) - 1),
                           eff_eta_n=np.zeros(len(ETA) - 1), eff_eta_d=np.zeros(len(ETA) - 1),
                           fk_pt_n=np.zeros(len(PT) - 1), fk_pt_d=np.zeros(len(PT) - 1),
                           fk_eta_n=np.zeros(len(ETA) - 1), fk_eta_d=np.zeros(len(ETA) - 1))
                 for cid, _ in TYPES} for lbl, *_ in MODELS}
    # events are ordered differently per model, so pair each model's reco with its own gen
    for label, _, _c, _m in MODELS:
        g = gen_ev[label]; r = reco_ev[label]; A = acc[label]
        for ev in range(len(g["pt"])):
            ge = ak.to_numpy(g["eta"][ev]); gp = ak.to_numpy(g["phi"][ev])
            gt = ak.to_numpy(g["type"][ev]); gpt = ak.to_numpy(g["pt"][ev])
            re = ak.to_numpy(r["eta"][ev]); rp = ak.to_numpy(r["phi"][ev])
            rt = ak.to_numpy(r["type"][ev]).astype(int); rpt = ak.to_numpy(r["pt"][ev])
            # pT > 1 GeV cut on both gen and reco particles
            gk = gpt > PT_MIN; ge, gp, gt, gpt = ge[gk], gp[gk], gt[gk], gpt[gk]; gae = np.abs(ge)
            rk = rpt > PT_MIN; re, rp, rt, rpt = re[rk], rp[rk], rt[rk], rpt[rk]; rae = np.abs(re)
            gm, rm = match_event(ge, gp, gt, re, rp, rt, samepid)
            for cid, _ in TYPES:
                gsel = gt == cid
                A[cid]["eff_pt_d"] += np.histogram(gpt[gsel], PT)[0]
                A[cid]["eff_pt_n"] += np.histogram(gpt[gsel & gm], PT)[0]
                A[cid]["eff_eta_d"] += np.histogram(gae[gsel], ETA)[0]
                A[cid]["eff_eta_n"] += np.histogram(gae[gsel & gm], ETA)[0]
                rsel = rt == cid
                A[cid]["fk_pt_d"] += np.histogram(rpt[rsel], PT)[0]
                A[cid]["fk_pt_n"] += np.histogram(rpt[rsel & (~rm)], PT)[0]
                A[cid]["fk_eta_d"] += np.histogram(rae[rsel], ETA)[0]
                A[cid]["fk_eta_n"] += np.histogram(rae[rsel & (~rm)], ETA)[0]

    def ratio(n, d):
        r = np.full(len(d), np.nan); m = d > 0; r[m] = n[m] / d[m]; return r

    for metric, ylabel, sub in [("eff", "efficiency", "efficiency"), ("fk", "fake rate", "fake_rate")]:
        for obs, bins, xlabel in [("pt", PT, r"particle $p_T$ [GeV]"), ("eta", ETA, r"particle $|\eta|$")]:
            ctr = np.sqrt(bins[:-1] * bins[1:]) if obs == "pt" else 0.5 * (bins[:-1] + bins[1:])
            for cid, nm in TYPES:
                fig, ax = plt.subplots(figsize=(8, 7))
                for label, _, c, _m in MODELS:
                    A = acc[label][cid]
                    y = ratio(A[f"{metric}_{obs}_n"], A[f"{metric}_{obs}_d"])
                    ax.plot(ctr, y, "o-", color=c, lw=2.2, ms=6, label=label)
                if obs == "pt":
                    ax.set_xscale("log")
                ax.set_ylim(-0.03, 1.08); ax.set_xlabel(xlabel); ax.set_ylabel(f"{ylabel} ({nm})")
                ax.legend(fontsize=10, title=cond_txt, title_fontsize=9); ax.grid(alpha=0.3)
                cms_label(ax)
                if note:  # ideal-mode note goes inside the frame (not as a subtitle)
                    ax.text(0.5, 0.045, note, transform=ax.transAxes, ha="center", va="bottom",
                            fontsize=11, style="italic", color="0.30")
                _save(fig, f"{outdir}/{sub}/{metric}_vs_{obs}_{nm}_{cond}.pdf")


def _endcap_jet_pt(px, py, pz):
    """Flatten per-event jet (px,py,pz) to a 1D array of endcap jet pT."""
    fx = ak.to_numpy(ak.flatten(px)); fy = ak.to_numpy(ak.flatten(py)); fz = ak.to_numpy(ak.flatten(pz))
    pt = np.hypot(fx, fy); eta = np.abs(np.arcsinh(fz / np.clip(pt, 1e-6, None)))
    m = (eta >= ENDCAP[0]) & (eta <= ENDCAP[1])
    return pt[m]


def plot_jets(reco_ev, gen_ev, outdir, note):
    """Jet pT spectrum, inclusive response (med+IQR), response vs pT, resolution (IQR/med) vs pT.
    Gen jets are clustered from each model's own generator (pythia) particles, and reco and gen
    jets are matched event-by-event within the SAME model, because the event ordering differs
    across models' preds."""

    def match(reco_coll, gen_coll):
        rpx, rpy, rpz = jets_from(reco_coll)
        gpx, gpy, gpz = jets_from(gen_coll)
        og, op = [], []
        for ev in range(len(gpx)):
            Gx = ak.to_numpy(gpx[ev]); Gy = ak.to_numpy(gpy[ev]); Gz = ak.to_numpy(gpz[ev])
            Rx = ak.to_numpy(rpx[ev]); Ry = ak.to_numpy(rpy[ev]); Rz = ak.to_numpy(rpz[ev])
            G = np.hypot(Gx, Gy); Ge = np.arcsinh(Gz / np.clip(G, 1e-6, None)); Gp = np.arctan2(Gy, Gx)
            P = np.hypot(Rx, Ry); Pe = np.arcsinh(Rz / np.clip(P, 1e-6, None)); Pp = np.arctan2(Ry, Rx)
            inec = (np.abs(Ge) >= ENDCAP[0]) & (np.abs(Ge) <= ENDCAP[1])
            used = np.zeros(len(P), bool)
            for k in np.where(inec)[0]:
                if not len(P):
                    break
                dphi = np.arctan2(np.sin(Pp - Gp[k]), np.cos(Pp - Gp[k]))
                d = np.hypot(Pe - Ge[k], dphi); d[used] = 99
                j = int(np.argmin(d))
                if d[j] < DR:
                    used[j] = True; og.append(G[k]); op.append(P[j])
        return np.array(og), np.array(op)

    jr = {}
    # 1) jet pT spectrum (gen + reco jets pT, endcap)
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.logspace(np.log10(JET_PTCUT), 2.7, 35)
    g0px, g0py, g0pz = jets_from(gen_ev[MODELS[0][0]])
    ax.hist(_endcap_jet_pt(g0px, g0py, g0pz), bins=b, histtype="step", lw=2.6, color="k", ls="--", label="gen jets")
    for label, _, c, _m in MODELS:
        og, op = match(reco_ev[label], gen_ev[label]); jr[label] = (og, op)
        rpx, rpy, rpz = jets_from(reco_ev[label])
        ax.hist(_endcap_jet_pt(rpx, rpy, rpz), bins=b, histtype="step", lw=2.3, color=c, label=label)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"jet $p_T$ [GeV]"); ax.set_ylabel("jets")
    ax.legend(fontsize=11); cms_label(ax, note); _save(fig, f"{outdir}/jet_pt_spectrum.pdf")

    # 2) inclusive response (med + IQR in legend)
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.linspace(0, 2.5, 55)
    for label, _, c, _m in MODELS:
        og, op = jr[label]; r = op / np.clip(og, 1e-6, None)
        md = np.median(r); iq = np.percentile(r, 75) - np.percentile(r, 25)
        ax.hist(r, bins=b, histtype="step", lw=2.3, color=c, label=f"{label}: M={md:.2f}, IQR={iq:.2f}")
    ax.axvline(1, color="k", ls=":"); ax.set_xlabel(r"jet $p_T^{\mathrm{reco}} / p_T^{\mathrm{gen}}$")
    ax.set_ylabel("jets"); ax.legend(fontsize=11); cms_label(ax, note)
    _save(fig, f"{outdir}/jet_response_inclusive.pdf")

    # 3) response vs pT  &  4) resolution vs pT
    ctr = np.sqrt(JET_PT_BINS[:-1] * JET_PT_BINS[1:])
    figM, axM = plt.subplots(figsize=(8, 7)); figR, axR = plt.subplots(figsize=(8, 7))
    for label, _, c, _m in MODELS:
        og, op = jr[label]; r = op / np.clip(og, 1e-6, None)
        med, reso = [], []
        for lo, hi in zip(JET_PT_BINS[:-1], JET_PT_BINS[1:]):
            m = (og >= lo) & (og < hi)
            if m.sum() > 6:
                rr = r[m]; md = np.median(rr); iq = np.percentile(rr, 75) - np.percentile(rr, 25)
                med.append(md); reso.append(iq / md if md > 0 else np.nan)
            else:
                med.append(np.nan); reso.append(np.nan)
        axM.plot(ctr, med, "o-", color=c, lw=2.2, ms=6, label=label)
        axR.plot(ctr, reso, "o-", color=c, lw=2.2, ms=6, label=label)
    axM.axhline(1, color="k", ls=":"); axM.set_xscale("log"); axM.set_xlabel(r"gen jet $p_T$ [GeV]")
    axM.set_ylabel("median jet response"); axM.legend(fontsize=11); axM.grid(alpha=0.3); cms_label(axM, note)
    _save(figM, f"{outdir}/jet_response_vs_pt.pdf")
    axR.set_xscale("log"); axR.set_xlabel(r"gen jet $p_T$ [GeV]"); axR.set_ylabel("jet resolution  IQR/median")
    axR.legend(fontsize=11); axR.grid(alpha=0.3); cms_label(axR, note)
    _save(figR, f"{outdir}/jet_resolution_vs_pt.pdf")


# ----------------------------------------------------------------------------- driver
def run_dir(a_by_model, out, mode):
    """mode: 'ideal' (truth isPU mitigation) or 'model' (each model's own strategy)."""
    note = "optimal pileup removal using truth info" if mode == "ideal" else None
    outdir = f"{out}/{'ideal_pu_removal' if mode == 'ideal' else 'model_pu_removal'}"
    for sub in ["distributions", "sumpt_response", "efficiency", "fake_rate", "jets"]:
        os.makedirs(f"{outdir}/{sub}", exist_ok=True)
    print(f"\n==================== {mode} -> {outdir} ====================")

    reco_ev = {}
    for label, exp, _c, mitig in MODELS:
        mm = "truth" if mode == "ideal" else mitig
        reco_ev[label] = reco_particles(a_by_model[label], mm)
    # gen is per-model: each model's preds store events in a different order, so per-event
    # comparisons must pair a model's reco with that same model's gen.
    gen_ev = {lbl: gen_particles(a_by_model[lbl]) for lbl, *_ in MODELS}

    # flattened endcap views for spectra (order-independent, so one gen curve suffices)
    reco_flat = {lbl: ec_flat(reco_ev[lbl]) for lbl in reco_ev}
    gen_flat = ec_flat(gen_ev[MODELS[0][0]])

    print("distributions ...")
    for which in ("pt", "eta"):
        plot_spectrum(reco_flat, gen_flat, which, f"{outdir}/distributions", note)
        for cid, nm in TYPES:
            plot_spectrum(reco_flat, gen_flat, which, f"{outdir}/distributions", note, type_sel=cid, type_name=nm)

    print("sum-pT response ...")
    plot_sumpt_response(reco_ev, gen_ev, f"{outdir}/sumpt_response", note)

    print("efficiency / fake rate ...")
    for samepid in (False, True):
        plot_eff_fake(reco_ev, gen_ev, outdir, note, samepid)

    print("jets ...")
    plot_jets(reco_ev, gen_ev, f"{outdir}/jets", note)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-root", default="/shared/mlpf-phase2/experiments")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--nfiles", type=int, default=1000, help="pred files (events) per model to read")
    ap.add_argument("--models", default="", help="comma-separated MODELS indices to keep (default: all)")
    ap.add_argument("--modes", default="ideal,model", help="comma-separated modes to run: ideal,model")
    args = ap.parse_args()
    if args.models.strip():
        keep = [int(i) for i in args.models.split(",")]
        MODELS[:] = [MODELS[i] for i in keep]
    for label, exp, _c, _m in MODELS:
        print(f"using model: {label} ({exp})")
    a_by_model = {}
    present = []
    for label, exp, _c, _m in MODELS:
        print(f"loading {label} ({exp}) ...")
        try:
            a_by_model[label] = load(exp, args.preds_root, args.nfiles)
        except FileNotFoundError:
            print(f"  skip {label}: no preds yet (not trained/evaluated)")
            continue
        present.append((label, exp, _c, _m))
    MODELS[:] = present  # restrict to models whose preds are present
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        run_dir(a_by_model, args.out, mode)
    print("\nDONE ->", args.out)


if __name__ == "__main__":
    main()
