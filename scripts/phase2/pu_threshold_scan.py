#!/usr/bin/env python
"""
Phase-2 200PU -- PU-head operating-point scan.

The main eval applies a single hard cut (keep if pred P(pileup) < 0.5). For the
pt-weighted binary head that 0.5 is roughly calibrated, but for the *soft*
energy-fraction head (T2) the predicted score is a continuous hard-energy
fraction, so 0.5 is not necessarily the right operating point. This scans the
cut and reports, per model, how sum-pT/gen and jet response move with it, so a
soft head can be judged at its own best threshold rather than a borrowed 0.5.

For each head model and each cut c, keep reco particles with P(pileup) < c, then
measure (endcap): median per-event sum pT(reco)/sum pT(gen) and median jet
pT response. The truth-isPU removal is the ideal ceiling (cut-independent).

Output (under <out>/pu_threshold_scan/):
  sumpt_vs_cut.pdf      median sum-pT/gen vs cut, one curve per model + ideal line
  jetresp_vs_cut.pdf    median jet response vs cut, one curve per model + ideal line
and a printed table (cut x model) to stdout.

Usage (inside the mlpf venv):
  python pu_threshold_scan.py --preds-root /shared/mlpf-phase2/experiments \
      --out /shared/mlpf-phase2/eval_plots/phase2_physics --models 2 3 4 --nfiles 1000
"""
import os, sys, argparse
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_phase2_plots import (  # noqa: E402
    MODELS, RLABEL, ENDCAP, load, gen_particles,
)
from model_vs_ideal_pu import match_jets, sumpt_ratio  # noqa: E402

hep.style.use("CMS")
ENDCAP_NOTE = r"HGCAL endcap (%.2f < $|\eta|$ < %.1f)" % ENDCAP


def reco_at_cut(a, cut, truth=False):
    """Per-event reco particle dict keeping P(pileup) < cut (or truth isPU < 0.5)."""
    pr = a["particles"]["pred"]
    keep = (pr["cls_id"] != 0) & (np.abs(pr["eta"]) >= ENDCAP[0]) & (np.abs(pr["eta"]) <= ENDCAP[1])
    score = a["particles"]["target"]["ispu"] if truth else pr["ispu"]
    keep = keep & (score < (0.5 if truth else cut))
    return dict(pt=pr["pt"][keep], eta=pr["eta"][keep], phi=pr["phi"][keep], type=pr["cls_id"][keep])


def jet_resp_median(reco_coll, gen):
    og, op = match_jets(reco_coll, gen)
    if not len(og):
        return np.nan, 0
    r = op / np.clip(og, 1e-6, None)
    return float(np.median(r)), len(og)


def cms_label(ax):
    hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-root", default="/shared/mlpf-phase2/experiments")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--models", type=int, nargs="+", default=[2, 3, 4], help="MODELS indices (head models)")
    ap.add_argument("--nfiles", type=int, default=1000)
    ap.add_argument("--cuts", type=float, nargs="+",
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    args = ap.parse_args()
    outdir = f"{args.out}/pu_threshold_scan"
    os.makedirs(outdir, exist_ok=True)
    cuts = np.array(args.cuts)

    sum_curves, jet_curves, labels, colors = {}, {}, {}, {}
    ideal_sumpt, ideal_jet = [], []
    for idx in args.models:
        lab, exp, col, mitig = MODELS[idx]
        if mitig != "head":
            print(f"skip {lab}: no PU head"); continue
        a = load(exp, args.preds_root, args.nfiles)
        gen = gen_particles(a)
        labels[idx], colors[idx] = lab, col
        sum_curves[idx] = np.array([np.median(sumpt_ratio(reco_at_cut(a, c), gen)) for c in cuts])
        jet_curves[idx] = np.array([jet_resp_median(reco_at_cut(a, c), gen)[0] for c in cuts])
        ri = reco_at_cut(a, 0.5, truth=True)
        ideal_sumpt.append(np.median(sumpt_ratio(ri, gen)))
        ideal_jet.append(jet_resp_median(ri, gen)[0])
    ideal_sumpt = float(np.mean(ideal_sumpt)); ideal_jet = float(np.mean(ideal_jet))

    # table
    print("\n=== median sum-pT/gen vs cut ===")
    print("cut      " + "".join(f"{labels[i][:16]:>18s}" for i in sum_curves))
    for j, c in enumerate(cuts):
        print(f"{c:4.2f}   " + "".join(f"{sum_curves[i][j]:18.3f}" for i in sum_curves))
    print(f"ideal  " + "".join(f"{ideal_sumpt:18.3f}" for _ in sum_curves))
    print("\n=== median jet response vs cut ===")
    print("cut      " + "".join(f"{labels[i][:16]:>18s}" for i in jet_curves))
    for j, c in enumerate(cuts):
        print(f"{c:4.2f}   " + "".join(f"{jet_curves[i][j]:18.3f}" for i in jet_curves))
    print(f"ideal  " + "".join(f"{ideal_jet:18.3f}" for _ in jet_curves))
    # best cut (closest sum-pT/gen to 1.0) per model
    print("\n=== cut minimizing |sum-pT/gen - 1| ===")
    for i in sum_curves:
        k = int(np.nanargmin(np.abs(sum_curves[i] - 1.0)))
        print(f"  {labels[i]:30s} best cut={cuts[k]:.2f}  sumpt={sum_curves[i][k]:.3f}  jet={jet_curves[i][k]:.3f}")

    for curves, ideal, ylab, ttl, fn in [
        (sum_curves, ideal_sumpt, r"median $\sum p_T^{\mathrm{reco}}/\sum p_T^{\mathrm{gen}}$",
         "sum-pT response", "sumpt_vs_cut.pdf"),
        (jet_curves, ideal_jet, r"median jet $p_T^{\mathrm{reco}}/p_T^{\mathrm{gen}}$",
         "jet response", "jetresp_vs_cut.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 7))
        for i in curves:
            ax.plot(cuts, curves[i], "o-", color=colors[i], lw=2.3, ms=7, label=labels[i])
        ax.axhline(1.0, color="k", ls=":", label="target (gen)")
        ax.axhline(ideal, color="0.5", ls="--", label="truth PU removal (ideal)")
        ax.axvline(0.5, color="0.8", ls="-", lw=1)
        ax.set_xlabel(r"PU-head cut (keep if $P_{\mathrm{PU}}$ < cut)"); ax.set_ylabel(ylab)
        ax.legend(fontsize=11, title=ENDCAP_NOTE + f", {ttl}", title_fontsize=10)
        ax.grid(alpha=0.3); cms_label(ax)
        fig.tight_layout(); fig.savefig(f"{outdir}/{fn}", bbox_inches="tight"); plt.close(fig)
        print("  ->", os.path.relpath(f"{outdir}/{fn}"))
    print("\nDONE ->", outdir)


if __name__ == "__main__":
    main()
