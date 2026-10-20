#!/usr/bin/env python
"""
Phase-2 200PU -- model-based vs ideal (truth) pileup removal, for a SINGLE model.

For one model (default: the pt-weighted PU head), overlay the physics AFTER pileup
mitigation using (a) the model's own PU head [drop pred P(pileup) >= PU_CUT] and
(b) the TRUTH isPU flag [the ideal ceiling], with the generator (pythia) as reference
where relevant. This isolates "how close does our algorithm get to the truth-based optimum".

Everything is restricted to the HGCAL endcap, and reuses the machinery (loading, particle
selection, jet clustering) in eval_phase2_plots.py so the definitions match the main eval.

Output (under <out>/model_vs_ideal_<model>/):
  particle_pt_spectrum.pdf   endcap particle pT spectrum: gen, ideal, model
  jet_pt_spectrum.pdf        endcap jet pT spectrum:      gen, ideal, model
  jet_response_inclusive.pdf inclusive jet pT(reco)/pT(gen) (median+IQR in legend): ideal, model
  jet_response_vs_pt.pdf     median jet response vs gen-jet pT: ideal, model
  sumpt_response.pdf         per-event sum pT(reco)/sum pT(gen): ideal, model

Usage (inside the mlpf venv):
  python model_vs_ideal_pu.py --preds-root /shared/mlpf-phase2/experiments \
      --out /shared/mlpf-phase2/eval_plots/phase2_physics --model 2 --nfiles 1000
"""
import os, sys, re, argparse
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_phase2_plots import (  # noqa: E402
    MODELS, RLABEL, ENDCAP, JET_PTCUT, JET_PT_BINS, DR,
    load, reco_particles, gen_particles, jets_from, ec_flat, _endcap_jet_pt,
)

hep.style.use("CMS")

C_MODEL, C_IDEAL, C_GEN = "#d62728", "#2ca02c", "k"   # our algo / truth ceiling / gen
IDEAL_LABEL = "Truth PU removal (ideal)"
ENDCAP_NOTE = r"HGCAL endcap (%.2f < $|\eta|$ < %.1f)" % ENDCAP


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def cms_label(ax):
    hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=16)


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("  ->", os.path.relpath(path))


def match_jets(reco_coll, gen_coll):
    """Greedy endcap jet match within DR; returns matched (gen_pt, reco_pt) arrays."""
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


def sumpt_ratio(reco_coll, gen_coll):
    """Per-event sum pT(reco) / sum pT(gen), all endcap particles."""
    num = ak.to_numpy(ak.sum(reco_coll["pt"], axis=1))
    den = ak.to_numpy(ak.sum(gen_coll["pt"], axis=1))
    ok = den > 0
    return num[ok] / den[ok]


# ----------------------------------------------------------------------------- plots
def plot_particle_pt(reco_model, reco_ideal, gen, outdir, mlabel, type_sel=None, type_name="all"):
    """Particle pT spectrum (gen, ideal, model). type_sel: class id (1=chrg_had,3=photon) or None."""
    rm, ri, g = ec_flat(reco_model), ec_flat(reco_ideal), ec_flat(gen)
    bins = np.logspace(0, 2.4, 45)

    def sel(d):
        return d["pt"] if type_sel is None else d["pt"][d["type"] == type_sel]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.hist(sel(g), bins=bins, histtype="step", lw=2.6, color=C_GEN, ls="--", label="Gen")
    ax.hist(sel(ri), bins=bins, histtype="step", lw=2.3, color=C_IDEAL, label=IDEAL_LABEL)
    ax.hist(sel(rm), bins=bins, histtype="step", lw=2.3, color=C_MODEL, label=mlabel)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"particle $p_T$ [GeV]"); ax.set_ylabel("particles")
    ttl = ENDCAP_NOTE if type_name == "all" else f"{type_name}, {ENDCAP_NOTE}"
    ax.legend(fontsize=12, title=ttl, title_fontsize=11); cms_label(ax)
    _save(fig, f"{outdir}/particle_pt_spectrum_{type_name}.pdf")


def plot_jet_pt(reco_model, reco_ideal, gen, outdir, mlabel):
    bins = np.logspace(np.log10(JET_PTCUT), 2.7, 35)
    fig, ax = plt.subplots(figsize=(8, 7))
    for coll, c, lab, ls in [(gen, C_GEN, "Gen jets", "--"),
                             (reco_ideal, C_IDEAL, IDEAL_LABEL, "-"),
                             (reco_model, C_MODEL, mlabel, "-")]:
        px, py, pz = jets_from(coll)
        ax.hist(_endcap_jet_pt(px, py, pz), bins=bins, histtype="step", lw=2.4, color=c, ls=ls, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_T$ [GeV]"); ax.set_ylabel("jets")
    ax.legend(fontsize=12, title=ENDCAP_NOTE + r", anti-$k_T$ R=0.4", title_fontsize=10); cms_label(ax)
    _save(fig, f"{outdir}/jet_pt_spectrum.pdf")


def plot_jet_response(jr, outdir, mlabel):
    # inclusive
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.linspace(0, 2.5, 55)
    for (og, op), c, lab in [(jr["ideal"], C_IDEAL, IDEAL_LABEL), (jr["model"], C_MODEL, mlabel)]:
        r = op / np.clip(og, 1e-6, None)
        md = np.median(r); iq = np.percentile(r, 75) - np.percentile(r, 25)
        ax.hist(r, bins=b, histtype="step", lw=2.4, color=c, label=f"{lab}: M={md:.2f}, IQR={iq:.2f}")
    ax.axvline(1, color="k", ls=":"); ax.set_xlabel(r"jet $p_T^{\mathrm{reco}} / p_T^{\mathrm{gen}}$")
    ax.set_ylabel("jets"); ax.legend(fontsize=11, title=ENDCAP_NOTE, title_fontsize=10); cms_label(ax)
    _save(fig, f"{outdir}/jet_response_inclusive.pdf")
    # median vs gen-jet pT
    ctr = np.sqrt(JET_PT_BINS[:-1] * JET_PT_BINS[1:])
    fig, ax = plt.subplots(figsize=(8, 7))
    for (og, op), c, lab in [(jr["ideal"], C_IDEAL, IDEAL_LABEL), (jr["model"], C_MODEL, mlabel)]:
        r = op / np.clip(og, 1e-6, None); med = []
        for lo, hi in zip(JET_PT_BINS[:-1], JET_PT_BINS[1:]):
            m = (og >= lo) & (og < hi)
            med.append(np.median(r[m]) if m.sum() > 6 else np.nan)
        ax.plot(ctr, med, "o-", color=c, lw=2.3, ms=7, label=lab)
    ax.axhline(1, color="k", ls=":"); ax.set_xscale("log")
    ax.set_xlabel(r"gen jet $p_T$ [GeV]"); ax.set_ylabel("median jet response")
    ax.legend(fontsize=12, title=ENDCAP_NOTE, title_fontsize=10); ax.grid(alpha=0.3); cms_label(ax)
    _save(fig, f"{outdir}/jet_response_vs_pt.pdf")


def plot_sumpt(reco_model, reco_ideal, gen, outdir, mlabel):
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.linspace(0, 2.5, 55)
    for coll, c, lab in [(reco_ideal, C_IDEAL, IDEAL_LABEL), (reco_model, C_MODEL, mlabel)]:
        r = sumpt_ratio(coll, gen)
        md = np.median(r) if len(r) else np.nan
        ax.hist(r, bins=b, histtype="step", lw=2.4, color=c, label=f"{lab} (med {md:.2f})")
    ax.axvline(1, color="k", ls=":")
    ax.set_xlabel(r"$\sum p_T^{\mathrm{reco}} / \sum p_T^{\mathrm{gen}}$ per event")
    ax.set_ylabel("events"); ax.legend(fontsize=12, title=ENDCAP_NOTE, title_fontsize=10); cms_label(ax)
    _save(fig, f"{outdir}/sumpt_response.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-root", default="/shared/mlpf-phase2/experiments")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--model", type=int, default=2, help="MODELS index to compare (default 2 = pt weighted PU head)")
    ap.add_argument("--nfiles", type=int, default=1000)
    args = ap.parse_args()

    mlabel, exp, _c, mitig = MODELS[args.model]
    if mitig != "head":
        raise SystemExit(f"model {args.model} ({mlabel}) has no PU head; pick a head model")
    outdir = f"{args.out}/model_vs_ideal_{slug(mlabel)}"
    os.makedirs(outdir, exist_ok=True)
    print(f"comparing {mlabel} ({exp}) model-PU-removal vs truth-PU-removal -> {outdir}")

    a = load(exp, args.preds_root, args.nfiles)
    reco_model = reco_particles(a, "head")    # drop pred P(pileup) >= cut
    reco_ideal = reco_particles(a, "truth")   # drop truth isPU (ideal ceiling)
    gen = gen_particles(a)                     # same event order (same parquet)

    print("particle spectrum ...")
    plot_particle_pt(reco_model, reco_ideal, gen, outdir, mlabel)
    plot_particle_pt(reco_model, reco_ideal, gen, outdir, mlabel, type_sel=1, type_name="chrg_had")
    plot_particle_pt(reco_model, reco_ideal, gen, outdir, mlabel, type_sel=3, type_name="photon")
    print("jet spectrum ...");      plot_jet_pt(reco_model, reco_ideal, gen, outdir, mlabel)
    print("jet response ...")
    jr = {"model": match_jets(reco_model, gen), "ideal": match_jets(reco_ideal, gen)}
    print(f"  matched jets: ideal {len(jr['ideal'][0])}, model {len(jr['model'][0])}")
    plot_jet_response(jr, outdir, mlabel)
    print("sum-pT response ...");   plot_sumpt(reco_model, reco_ideal, gen, outdir, mlabel)
    print("\nDONE ->", outdir)


if __name__ == "__main__":
    main()
