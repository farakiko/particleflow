#!/usr/bin/env python3
"""Learnability comparison of the neutral-target schemes (docs/phase2.md 4A/9B):
argmax (one target per particle on its best trackster) vs fragment (moanwar-style
proportional split, truth-energy conserving, no gen filter).

Per TRACKSTER element with a neutral label, the regression target is
log(E_target / E_trackster_raw); a learnable target is O(1) with small tails.
Also reports the trackster label composition (null fraction etc.).

  pixi run python3 scripts/phase2/frag_learnability.py \
      --glob-a '...pkl_links/*.pkl' --glob-b '...pkl_links_frag/*.pkl' \
      --outdir plots/phase2/frag_comparison
"""
import argparse
import glob
import pickle
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
import numpy as np

mplhep.style.use("CMS")
TRACKSTER_TYP = 4


def collect(pattern, max_files):
    files = sorted(glob.glob(pattern))[:max_files]
    assert files, f"no files for {pattern}"
    out = {"logr_nhad": [], "logr_photon": [], "labels": {"null": 0, "nhad": 0, "photon": 0, "other": 0}, "nts": 0}
    for f in files:
        for ev in pickle.load(open(f, "rb")):
            X, y = ev["Xelem"], ev["ytarget"]
            ts = X["typ"] == TRACKSTER_TYP
            out["nts"] += int(ts.sum())
            pid = np.abs(y["pid"][ts]).astype(int)
            eelem = X["energy"][ts]
            etgt = y["energy"][ts]
            out["labels"]["null"] += int((pid == 0).sum())
            out["labels"]["nhad"] += int((pid == 130).sum())
            out["labels"]["photon"] += int((pid == 22).sum())
            out["labels"]["other"] += int(((pid != 0) & (pid != 130) & (pid != 22)).sum())
            for key, p in [("logr_nhad", 130), ("logr_photon", 22)]:
                m = (pid == p) & (eelem > 0) & (etgt > 0)
                out[key].append(np.log(etgt[m] / eelem[m]))
    out["logr_nhad"] = np.concatenate(out["logr_nhad"])
    out["logr_photon"] = np.concatenate(out["logr_photon"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob-a", required=True)
    ap.add_argument("--glob-b", required=True)
    ap.add_argument("--label-a", default="argmax")
    ap.add_argument("--label-b", default="fragment")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-files", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    A = collect(a.glob_a, a.max_files)
    B = collect(a.glob_b, a.max_files)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    bins = np.linspace(-3, 5, 120)
    for ax, key, title in [(axes[0], "logr_nhad", "neutral hadron"), (axes[1], "logr_photon", "photon")]:
        for d, lbl, c in [(A, a.label_a, "tab:orange"), (B, a.label_b, "tab:blue")]:
            r = d[key]
            q1, q2, q3 = np.percentile(r, [25, 50, 75])
            big = float((np.abs(r) > np.log(5)).mean())
            ax.hist(r, bins=bins, histtype="step", lw=2, color=c, density=True,
                    label=f"{lbl}: n={len(r)}, med={q2:.2f}, IQR={q3 - q1:.2f}, |ratio|>5: {100 * big:.0f}%")
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel(r"$\log(E_{\mathrm{target}} / E_{\mathrm{trackster}}^{\mathrm{raw}})$")
        ax.set_ylabel("density")
        ax.set_title(f"{title} regression target per trackster", fontsize=15)
        ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "learnability_logratio.png"), dpi=140)
    plt.close(fig)

    print(f"\n{'':>10} {'tracksters':>11} {'null%':>7} {'nhad%':>7} {'photon%':>8} {'other%':>7}")
    for d, lbl in [(A, a.label_a), (B, a.label_b)]:
        n = d["nts"]
        L = d["labels"]
        print(f"{lbl:>10} {n:>11} {100 * L['null'] / n:>6.1f} {100 * L['nhad'] / n:>6.1f} "
              f"{100 * L['photon'] / n:>7.1f} {100 * L['other'] / n:>6.1f}")
    for key, t in [("logr_nhad", "nhad"), ("logr_photon", "photon")]:
        for d, lbl in [(A, a.label_a), (B, a.label_b)]:
            r = d[key]
            q1, q2, q3 = np.percentile(r, [25, 50, 75])
            print(f"{t:>8} {lbl:>9}: med={q2:.3f} IQR={q3 - q1:.3f} frac|ratio|>5={100 * float((np.abs(r) > np.log(5)).mean()):.1f}%")
    print(f"done -> {a.outdir}")


if __name__ == "__main__":
    main()
