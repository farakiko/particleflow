#!/usr/bin/env python
"""
Phase-2 200PU MLPF -- pileup-classification head performance.

Evaluates the PU output head of the two models that have one ("MLPF + PU head" and
"MLPF + pt weighted PU head"): how well it separates pileup from hard-scatter particles,
and in particular how robust that separation is against particle pT (the "pt-shortcut"
failure mode, where a naive head simply learns low-pT -> pileup).

The head output is the per-element predicted P(pileup), stored as `particles.pred.ispu`
in the inference parquet (a probability in [0,1]); the truth label is `particles.target.ispu`
(0 = hard scatter, 1 = pileup); the truth type is `particles.target.cls_id`. Only elements
with a real target particle (cls_id != 0) in the HGCAL endcap are used. The baseline model has
no head (ispu is a constant 0.5) and is excluded.

Produces, under <out>/pu_head_perf/:
  puscore_vs_pt_<model>.pdf   median P(pileup) vs truth pT, signal vs pileup (IQR band) -- per model
  puscore_vs_pt_signal.pdf    median P(pileup) vs truth pT for SIGNAL only, both models (pt-shortcut)
  roc_by_type_<model>.pdf     ROC (pileup vs signal) split by truth particle type, AUC in legend
  pu_prob_<model>.pdf         P(pileup) distribution, truth signal vs truth pileup

Usage (inside the mlpf venv):
  python pu_head_perf.py --preds-root /shared/mlpf-phase2/experiments \
                         --out /shared/mlpf-phase2/eval_plots/phase2_physics --nfiles 1000
"""
import os, glob, argparse, re
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

# ----------------------------------------------------------------------------- configuration
SAMPLE = "cms_pf_phase2_ttbar200pu"
RLABEL = r"$\mathrm{t}\bar{\mathrm{t}}$, 200 PU (14 TeV)"
# (display label, experiment subdir, colour)  -- only models that HAVE a PU head
MODELS = [
    ("MLPF + PU head",             "ttbar200-heptv2-pu",     "#1f77b4"),
    ("MLPF + pt weighted PU head", "ttbar200-heptv2-pu-rew", "#d62728"),
    ("MLPF + soft PU head (T2)",   "ttbar200-heptv2-pu-soft", "#9467bd"),
    ("MLPF + PU head + ctx (T3)",  "ttbar200-heptv2-pu-ctx",  "#8c564b"),
    ("MLPF + PU head + feats (T1)", "ttbar200-heptv2-pu-feats", "#e377c2"),
]
# CLEAN_EXP=1 -> evaluate the clean-flag retrains (append "-clean" to each exp subdir at runtime).
if os.environ.get("CLEAN_EXP"):
    MODELS = [(lbl, exp + "-clean", c) for (lbl, exp, c) in MODELS]
PU_CUT = 0.5
ENDCAP = (1.48, 3.0)
# truth class index (cms_phase2 CLASS_LABELS = [0,211,130,22,11,13]) -> (name, colour)
TYPES = [(1, "chrg_had", "#1f77b4"), (2, "neut_had", "#ff7f0e"), (3, "photon", "#2ca02c"),
         (4, "electron", "#d62728"), (5, "muon", "#9467bd")]
PT_BINS = np.logspace(0, 2.2, 16)   # truth pT profile bins
PT_2D = np.logspace(0, 2.2, 25)     # 2D map pT bins (from 1 GeV)
SCORE_BINS = np.linspace(0, 1, 41)  # 2D map P(pileup) bins
SIG_C, PU_C = "#1f77b4", "#d62728"  # signal / pileup colours in per-model plots


# ----------------------------------------------------------------------------- helpers
def cms_label(ax, note=None):
    hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=16)
    if note:
        ax.text(0.5, 1.13, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=13, style="italic", color="0.30")


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def load(exp, preds_root, nfiles):
    fs = sorted(glob.glob(f"{preds_root}/{exp}/preds_test/{SAMPLE}/pred_*.parquet"))[:nfiles]
    if not fs:
        raise FileNotFoundError(f"no preds for {exp} under {preds_root}")
    return ak.concatenate([ak.from_parquet(f) for f in fs])


def extract(a):
    """Flat per-element arrays for real target particles in the endcap:
    score=P(pileup), label=truth ispu (1=pileup), pt=truth pT, typ=truth class id."""
    pr = a["particles"]["pred"]; tg = a["particles"]["target"]
    score = ak.to_numpy(ak.flatten(pr["ispu"]))
    label = ak.to_numpy(ak.flatten(tg["ispu"]))
    pt = ak.to_numpy(ak.flatten(tg["pt"]))
    eta = ak.to_numpy(ak.flatten(tg["eta"]))
    typ = ak.to_numpy(ak.flatten(tg["cls_id"])).astype(int)
    m = (typ != 0) & (np.abs(eta) >= ENDCAP[0]) & (np.abs(eta) <= ENDCAP[1])
    return dict(score=score[m], label=label[m], pt=pt[m], typ=typ[m])


def roc(score, label):
    """Manual ROC for a binary score (positive class = pileup, label==1). Returns fpr, tpr, auc."""
    order = np.argsort(-score, kind="mergesort")
    s = label[order].astype(float)
    P = s.sum(); N = len(s) - P
    if P == 0 or N == 0:
        return np.array([0, 1.0]), np.array([0, 1.0]), np.nan
    tpr = np.concatenate([[0.0], np.cumsum(s) / P])
    fpr = np.concatenate([[0.0], np.cumsum(1 - s) / N])
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy>=2 renamed trapz
    return fpr, tpr, float(_trapz(tpr, fpr))


def profile(x, y, bins):
    """Median y (+ IQR) of y in log-x bins; returns centres, med, q25, q75."""
    idx = np.digitize(x, bins) - 1
    cx, md, lo, hi = [], [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() > 20:
            yy = y[m]
            cx.append(np.sqrt(bins[b] * bins[b + 1]))
            md.append(np.median(yy)); lo.append(np.percentile(yy, 25)); hi.append(np.percentile(yy, 75))
    return np.array(cx), np.array(md), np.array(lo), np.array(hi)


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("  ->", os.path.relpath(path))


# ----------------------------------------------------------------------------- plots
def plot_score_vs_pt_permodel(d, label, outdir):
    """median P(pileup) vs truth pT, split signal vs pileup, for one model."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for lab, name, c in [(0, "hard scatter", SIG_C), (1, "pileup", PU_C)]:
        sel = d["label"] == lab
        cx, md, lo, hi = profile(d["pt"][sel], d["score"][sel], PT_BINS)
        ax.plot(cx, md, "o-", color=c, lw=2.4, ms=6, label=f"{name} (truth)")
        ax.fill_between(cx, lo, hi, color=c, alpha=0.18, lw=0)
    ax.axhline(PU_CUT, color="k", ls=":", lw=1.5, label=f"cut = {PU_CUT}")
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"truth particle $p_T$ [GeV]"); ax.set_ylabel(r"predicted $P(\mathrm{pileup})$")
    ax.legend(fontsize=12, title=label, title_fontsize=12, loc="center right"); ax.grid(alpha=0.3)
    cms_label(ax)
    _save(fig, f"{outdir}/puscore_vs_pt_{slug(label)}.pdf")


def plot_score_vs_pt_signal(data, outdir):
    """median P(pileup) vs truth pT for SIGNAL particles, both models (pt-shortcut comparison)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, _, c in MODELS:
        d = data[label]; sel = d["label"] == 0
        cx, md, lo, hi = profile(d["pt"][sel], d["score"][sel], PT_BINS)
        ax.plot(cx, md, "o-", color=c, lw=2.4, ms=6, label=label)
        ax.fill_between(cx, lo, hi, color=c, alpha=0.15, lw=0)
    ax.axhline(PU_CUT, color="k", ls=":", lw=1.5, label=f"cut = {PU_CUT}")
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"truth particle $p_T$ [GeV]")
    ax.set_ylabel(r"predicted $P(\mathrm{pileup})$,  hard-scatter particles")
    ax.legend(fontsize=12, loc="upper right"); ax.grid(alpha=0.3)
    cms_label(ax, note="lower = fewer hard-scatter particles tagged as pileup")
    _save(fig, f"{outdir}/puscore_vs_pt_signal.pdf")


def plot_score_vs_pt_pileup(data, outdir):
    """median P(pileup) vs truth pT for PILEUP particles, all models (rejection vs pT)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, _, c in MODELS:
        d = data[label]; sel = d["label"] == 1
        cx, md, lo, hi = profile(d["pt"][sel], d["score"][sel], PT_BINS)
        ax.plot(cx, md, "o-", color=c, lw=2.4, ms=6, label=label)
        ax.fill_between(cx, lo, hi, color=c, alpha=0.15, lw=0)
    ax.axhline(PU_CUT, color="k", ls=":", lw=1.5, label=f"cut = {PU_CUT}")
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"truth particle $p_T$ [GeV]")
    ax.set_ylabel(r"predicted $P(\mathrm{pileup})$,  pileup particles")
    ax.legend(fontsize=12, loc="lower left"); ax.grid(alpha=0.3)
    cms_label(ax, note="higher = more pileup correctly tagged as pileup")
    _save(fig, f"{outdir}/puscore_vs_pt_pileup.pdf")


def plot_roc_by_type(d, label, outdir):
    """ROC (pileup vs signal) per truth particle type, for one model."""
    fig, ax = plt.subplots(figsize=(8, 7))
    for cid, name, c in TYPES:
        sel = d["typ"] == cid
        fpr, tpr, auc = roc(d["score"][sel], d["label"][sel])
        ax.plot(fpr, tpr, "-", color=c, lw=2.4,
                label=f"{name} (AUC {auc:.3f})" if auc == auc else f"{name} (n/a)")
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("false positive rate (signal tagged pileup)")
    ax.set_ylabel("true positive rate (pileup tagged pileup)")
    ax.legend(fontsize=12, title=label, title_fontsize=12, loc="lower right"); ax.grid(alpha=0.3)
    cms_label(ax)
    _save(fig, f"{outdir}/roc_by_type_{slug(label)}.pdf")


def _colnorm(H):
    """Normalise each pT column (axis 0) of a 2D histogram to unit area."""
    s = H.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return H / s


def plot_score_vs_pt_2d(d, label, outdir):
    """2D P(pileup) vs truth pT, truth-signal | truth-pileup side by side.

    Each pT column is normalised to unit area, so colour = P(score | pT, class): this shows,
    at fixed pT, how the head's score is distributed -- directly revealing whether the ~0.5
    population is low-pT signal (left panel) and/or high-pT pileup (right panel), and the
    low-pT score spread (the IQR of the median-score curve). White curve = median score."""
    pu = d["label"] >= 0.5
    panels = [(~pu, "truth hard scatter"), (pu, "truth pileup")]
    Hs = [_colnorm(np.histogram2d(d["pt"][sel], d["score"][sel], [PT_2D, SCORE_BINS])[0])
          for sel, _ in panels]
    vmax = max(H.max() for H in Hs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    pc = None
    for ax, (sel, ttl), H in zip(axes, panels, Hs):
        pc = ax.pcolormesh(PT_2D, SCORE_BINS, H.T, cmap="viridis", vmin=0, vmax=vmax)
        cx, md, lo, hi = profile(d["pt"][sel], d["score"][sel], PT_2D)
        ax.plot(cx, md, "o-", color="w", lw=2.0, ms=4, label="median")
        ax.fill_between(cx, lo, hi, color="w", alpha=0.0)  # IQR shown via band below
        ax.plot(cx, lo, "--", color="w", lw=1.0, alpha=0.7)
        ax.plot(cx, hi, "--", color="w", lw=1.0, alpha=0.7, label="25-75%")
        ax.axhline(PU_CUT, color="r", ls=":", lw=1.6)
        ax.set_xscale("log"); ax.set_xlim(PT_2D[0], PT_2D[-1]); ax.set_ylim(0, 1)
        ax.set_xlabel(r"truth particle $p_T$ [GeV]")
        ax.text(0.04, 0.96, ttl, transform=ax.transAxes, fontsize=15, color="w",
                va="top", fontweight="bold")
    axes[0].set_ylabel(r"predicted $P(\mathrm{pileup})$")
    axes[0].legend(fontsize=11, loc="upper right", framealpha=0.3)
    fig.colorbar(pc, ax=list(axes), label=r"$P(\mathrm{score}\,|\,p_T)$  [column-normalised]", pad=0.02)
    hep.cms.label(ax=axes[0], data=False, label="Work in Progress", rlabel=RLABEL, fontsize=13)
    axes[1].text(1.0, 1.01, label, transform=axes[1].transAxes, ha="right", va="bottom",
                 fontsize=12, style="italic", color="0.30")
    path = f"{outdir}/puscore_vs_pt_2d_{slug(label)}.pdf"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("  ->", os.path.relpath(path))


def plot_prob_dist(d, label, outdir):
    """P(pileup) distribution, truth signal vs truth pileup, for one model."""
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.linspace(0, 1, 51)
    for lab, name, c in [(0, "hard scatter", SIG_C), (1, "pileup", PU_C)]:
        sel = d["label"] == lab
        ax.hist(d["score"][sel], bins=b, histtype="step", lw=2.4, color=c, density=True,
                label=f"{name} (truth, {int(sel.sum())})")
    ax.axvline(PU_CUT, color="k", ls=":", lw=1.5)
    ax.set_yscale("log"); ax.set_xlabel(r"predicted $P(\mathrm{pileup})$")
    ax.set_ylabel("particles (normalised)")
    ax.legend(fontsize=12, title=label, title_fontsize=12); cms_label(ax)
    _save(fig, f"{outdir}/pu_prob_{slug(label)}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-root", default="/shared/mlpf-phase2/experiments")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--nfiles", type=int, default=1000)
    args = ap.parse_args()

    outdir = f"{args.out}/pu_head_perf"
    os.makedirs(outdir, exist_ok=True)

    data = {}
    present = []
    for label, exp, _c in MODELS:
        print(f"loading {label} ({exp}) ...")
        try:
            d = extract(load(exp, args.preds_root, args.nfiles))
        except FileNotFoundError:
            print(f"  skip {label}: no preds yet (not trained/evaluated)")
            continue
        data[label] = d
        present.append((label, exp, _c))
        _, _, auc = roc(d["score"], d["label"])
        print(f"  {label}: {len(d['score'])} endcap particles, "
              f"pileup frac {d['label'].mean():.3f}, overall AUC {auc:.3f}")
    MODELS[:] = present  # restrict plots to models whose preds are present

    print("plots ...")
    for label, _, _c in MODELS:
        plot_score_vs_pt_permodel(data[label], label, outdir)
        plot_score_vs_pt_2d(data[label], label, outdir)
        plot_roc_by_type(data[label], label, outdir)
        plot_prob_dist(data[label], label, outdir)
    plot_score_vs_pt_signal(data, outdir)
    plot_score_vs_pt_pileup(data, outdir)

    print("\nDONE ->", outdir)


if __name__ == "__main__":
    main()
