#!/usr/bin/env python
"""
Phase-2 200PU -- T5: hard-mask vs soft-weight pileup removal, for one model (default pt-weighted).

The PU head gives P(pileup) per element. Two ways to use it:
  hard : drop elements with P(pileup) >= PU_CUT         (the current T0 strategy)
  soft : keep all elements, scale momentum by w=1-P(pileup)  (down-weight instead of drop)
with the TRUTH-isPU removal as the ideal ceiling and the generator as reference.

Soft weighting is a pure inference-time change on an already-trained model (no retraining): a
leaked pileup element at P=0.4 contributes 0.6x instead of 1x, so it should shave the diffuse
residual-pileup pedestal that the hard 0.5 cut leaves (see the model_vs_ideal sum-pT result).

Output (under <out>/hard_vs_soft_<model>/):
  sumpt_response.pdf          per-event sum pT(reco)/sum pT(gen): hard, soft, ideal
  jet_response_inclusive.pdf  inclusive jet response (median+IQR in legend)
  jet_response_vs_pt.pdf      median jet response vs gen-jet pT
  jet_pt_spectrum.pdf         endcap jet pT spectrum: gen, ideal, soft, hard

Usage (inside the mlpf venv):
  python hard_vs_soft_pu.py --model 2 --nfiles 1000
"""
import os, sys, argparse
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_phase2_plots import MODELS, ENDCAP, PU_CUT, JET_PTCUT, JET_PT_BINS, load, reco_particles, gen_particles, jets_from, _endcap_jet_pt  # noqa: E402
from model_vs_ideal_pu import match_jets, sumpt_ratio, cms_label, _save, slug, ENDCAP_NOTE, C_IDEAL, C_GEN, IDEAL_LABEL  # noqa: E402

C_HARD, C_SOFT = "#d62728", "#1f77b4"   # hard mask (T0) / soft weight


def reco_soft(a):
    """Keep all endcap real particles, scale pT by w = 1 - P(pileup) (soft down-weighting)."""
    pr = a["particles"]["pred"]
    keep = (pr["cls_id"] != 0) & (np.abs(pr["eta"]) >= ENDCAP[0]) & (np.abs(pr["eta"]) <= ENDCAP[1])
    w = 1.0 - pr["ispu"]  # P(pileup) already in [0,1]
    return dict(pt=(pr["pt"] * w)[keep], eta=pr["eta"][keep], phi=pr["phi"][keep], type=pr["cls_id"][keep])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-root", default="/shared/mlpf-phase2/experiments")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--model", type=int, default=2, help="MODELS index (default 2 = pt weighted PU head)")
    ap.add_argument("--nfiles", type=int, default=1000)
    args = ap.parse_args()

    mlabel, exp, _c, mitig = MODELS[args.model]
    if mitig != "head":
        raise SystemExit(f"model {args.model} ({mlabel}) has no PU head")
    outdir = f"{args.out}/hard_vs_soft_{slug(mlabel)}"
    os.makedirs(outdir, exist_ok=True)
    print(f"{mlabel} ({exp}): hard-mask vs soft-weight -> {outdir}")

    a = load(exp, args.preds_root, args.nfiles)
    rhard = reco_particles(a, "head")    # drop P(pu) >= cut
    rsoft = reco_soft(a)                  # keep all, weight by 1-P(pu)
    rideal = reco_particles(a, "truth")  # truth removal (ceiling)
    gen = gen_particles(a)
    series = [("hard mask (cut %.1f)" % PU_CUT, rhard, C_HARD),
              ("soft weight (1-P)", rsoft, C_SOFT),
              (IDEAL_LABEL, rideal, C_IDEAL)]

    # 1) sum-pT response
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.linspace(0, 2.5, 55)
    for lab, coll, c in series:
        r = sumpt_ratio(coll, gen); md = np.median(r) if len(r) else np.nan
        ax.hist(r, bins=b, histtype="step", lw=2.4, color=c, label=f"{lab} (med {md:.2f})")
    ax.axvline(1, color="k", ls=":")
    ax.set_xlabel(r"$\sum p_T^{\mathrm{reco}} / \sum p_T^{\mathrm{gen}}$ per event"); ax.set_ylabel("events")
    ax.legend(fontsize=12, title=f"{mlabel}\n{ENDCAP_NOTE}", title_fontsize=9); cms_label(ax)
    _save(fig, f"{outdir}/sumpt_response.pdf")

    # jets
    jr = {lab: match_jets(coll, gen) for lab, coll, _c in series}

    # 2) inclusive response
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.linspace(0, 2.5, 55)
    for lab, _coll, c in series:
        og, op = jr[lab]; r = op / np.clip(og, 1e-6, None)
        md = np.median(r); iq = np.percentile(r, 75) - np.percentile(r, 25)
        ax.hist(r, bins=b, histtype="step", lw=2.4, color=c, label=f"{lab}: M={md:.2f}, IQR={iq:.2f}")
    ax.axvline(1, color="k", ls=":"); ax.set_xlabel(r"jet $p_T^{\mathrm{reco}} / p_T^{\mathrm{gen}}$")
    ax.set_ylabel("jets"); ax.legend(fontsize=11, title=ENDCAP_NOTE, title_fontsize=10); cms_label(ax)
    _save(fig, f"{outdir}/jet_response_inclusive.pdf")

    # 3) median response vs gen-jet pT
    ctr = np.sqrt(JET_PT_BINS[:-1] * JET_PT_BINS[1:])
    fig, ax = plt.subplots(figsize=(8, 7))
    for lab, _coll, c in series:
        og, op = jr[lab]; r = op / np.clip(og, 1e-6, None); med = []
        for lo, hi in zip(JET_PT_BINS[:-1], JET_PT_BINS[1:]):
            m = (og >= lo) & (og < hi)
            med.append(np.median(r[m]) if m.sum() > 6 else np.nan)
        ax.plot(ctr, med, "o-", color=c, lw=2.3, ms=7, label=lab)
    ax.axhline(1, color="k", ls=":"); ax.set_xscale("log")
    ax.set_xlabel(r"gen jet $p_T$ [GeV]"); ax.set_ylabel("median jet response")
    ax.legend(fontsize=11, title=ENDCAP_NOTE, title_fontsize=10); ax.grid(alpha=0.3); cms_label(ax)
    _save(fig, f"{outdir}/jet_response_vs_pt.pdf")

    # 4) jet pT spectrum
    bins = np.logspace(np.log10(JET_PTCUT), 2.7, 35)
    fig, ax = plt.subplots(figsize=(8, 7))
    gpx, gpy, gpz = jets_from(gen)
    ax.hist(_endcap_jet_pt(gpx, gpy, gpz), bins=bins, histtype="step", lw=2.6, color=C_GEN, ls="--", label="Gen jets")
    for lab, coll, c in [series[2], series[1], series[0]]:  # ideal, soft, hard
        px, py, pz = jets_from(coll)
        ax.hist(_endcap_jet_pt(px, py, pz), bins=bins, histtype="step", lw=2.3, color=c, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_T$ [GeV]"); ax.set_ylabel("jets")
    ax.legend(fontsize=11, title=ENDCAP_NOTE, title_fontsize=10); cms_label(ax)
    _save(fig, f"{outdir}/jet_pt_spectrum.pdf")

    print("DONE ->", outdir)


if __name__ == "__main__":
    main()
