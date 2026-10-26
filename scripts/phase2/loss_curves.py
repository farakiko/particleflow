#!/usr/bin/env python
"""
Phase-2 200PU MLPF -- training/validation loss curves vs epoch.

Reads the per-checkpoint loss history written by `mlpf/pipeline.py ... train`
(<exp>/history/step_<N>.json, each with a "train" and a "valid" loss dict) and draws,
for each model, the total loss and every individual loss component as a function of epoch,
with two curves per plot: training (solid) and validation (dashed).

Step -> epoch uses STEPS_PER_EPOCH (default 4500: the epoch 0->1 boundary falls at ~step 4500,
so 50000 steps ~= 11 epochs; all models share the same tfds split).

Output, under <out>/loss_curves/<model>/ :
  total.pdf                 total loss
  <component>.pdf           one per loss component (classification, regression_pt, ...; the
                            PU-head models also have classification_pu)

Usage (inside the mlpf venv):
  python loss_curves.py --exp-root /shared/mlpf-phase2/experiments \
                        --out /shared/mlpf-phase2/eval_plots/phase2_physics --steps-per-epoch 4500
"""
import os, glob, json, argparse, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

# ----------------------------------------------------------------------------- configuration
RLABEL = r"$\mathrm{t}\bar{\mathrm{t}}$, 200 PU (14 TeV)"
# (display label, experiment subdir, overlay colour)
MODELS = [
    ("MLPF + pt weighted PU head (T0)", "ttbar200-heptv2-pu-rew",  "#d62728"),
    ("MLPF + PU head + feats (T1)",     "ttbar200-heptv2-pu-feats", "#e377c2"),
    ("MLPF + soft PU head (T2)",        "ttbar200-heptv2-pu-soft", "#9467bd"),
    ("MLPF + PU head + ctx (T3)",       "ttbar200-heptv2-pu-ctx",  "#8c564b"),
]
# raw history key -> (pretty title, output filename)
COMPONENTS = {
    "Total":                ("Total loss", "total"),
    "Classification_binary": ("Binary classification loss (element has a particle)", "classification_binary"),
    "Classification":       ("PID classification loss", "classification_pid"),
    "Regression_pt":        (r"$p_T$ regression loss", "regression_pt"),
    "Regression_energy":    ("Energy regression loss", "regression_energy"),
    "Regression_eta":       (r"$\eta$ regression loss", "regression_eta"),
    "Regression_sin_phi":   (r"$\sin\phi$ regression loss", "regression_sin_phi"),
    "Regression_cos_phi":   (r"$\cos\phi$ regression loss", "regression_cos_phi"),
    "Classification_pu":    ("Pileup classification loss", "classification_pu"),
}
TRAIN_C, VAL_C = "#1f77b4", "#d62728"


def cms_label(ax, note=None):
    hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=16)
    if note:
        ax.text(0.5, 1.13, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=13, style="italic", color="0.30")


def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def load_history(exp_dir, steps_per_epoch):
    """Return (epochs, {component: {'train': arr, 'valid': arr}}) sorted by step."""
    files = glob.glob(f"{exp_dir}/history/step_*.json")
    steps = sorted(int(re.search(r"step_(\d+)\.json", f).group(1)) for f in files)
    epochs = np.array(steps) / steps_per_epoch
    comps = {}
    for step in steps:
        d = json.load(open(f"{exp_dir}/history/step_{step}.json"))
        for key in d["train"]:
            comps.setdefault(key, {"train": [], "valid": []})
            comps[key]["train"].append(d["train"][key])
            comps[key]["valid"].append(d["valid"].get(key, np.nan))
    for key in comps:
        comps[key]["train"] = np.array(comps[key]["train"], dtype=float)
        comps[key]["valid"] = np.array(comps[key]["valid"], dtype=float)
    return epochs, comps


def plot_component(epochs, series, title, label, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(epochs, series["train"], "o-", color=TRAIN_C, lw=2.4, ms=6, label="training")
    ax.plot(epochs, series["valid"], "s--", color=VAL_C, lw=2.4, ms=6, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel(title)
    # log-y when the values span a wide positive range (keeps flat small losses readable)
    both = np.concatenate([series["train"], series["valid"]])
    both = both[np.isfinite(both) & (both > 0)]
    if both.size and both.max() / both.min() > 50:
        ax.set_yscale("log")
    ax.legend(fontsize=13, title=label, title_fontsize=12); ax.grid(alpha=0.3)
    cms_label(ax)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("  ->", os.path.relpath(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-root", default="/shared/mlpf-phase2/experiments")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--steps-per-epoch", type=float, default=4500.0)
    args = ap.parse_args()

    overlay = []  # (label, colour, epochs, comps) for the cross-model comparison
    for label, exp, colour in MODELS:
        exp_dir = f"{args.exp_root}/{exp}"
        if not os.path.isdir(f"{exp_dir}/history"):
            print(f"skip {label}: no history dir"); continue
        print(f"{label} ({exp}) ...")
        epochs, comps = load_history(exp_dir, args.steps_per_epoch)
        outdir = f"{args.out}/loss_curves/{slug(label)}"
        os.makedirs(outdir, exist_ok=True)
        # total first, then the rest in the configured order, then any unknown keys
        order = [k for k in COMPONENTS if k in comps] + [k for k in comps if k not in COMPONENTS]
        for key in order:
            title, fname = COMPONENTS.get(key, (key, slug(key)))
            plot_component(epochs, comps[key], title, label, f"{outdir}/{fname}.pdf")
        overlay.append((label, colour, epochs, comps))

    # cross-model validation-loss overlays (the one-glance comparisons)
    for comp_key, ylabel, fname in [
        ("Total", "total validation loss", "total_comparison.pdf"),
        ("Classification_pu", "pileup-head validation loss", "pu_comparison.pdf"),
    ]:
        if not any(comp_key in comps for _, _, _, comps in overlay):
            continue
        fig, ax = plt.subplots(figsize=(8, 7))
        for label, colour, epochs, comps in overlay:
            if comp_key in comps:
                ax.plot(epochs, comps[comp_key]["valid"], "o-", color=colour, lw=2.4, ms=5, label=label)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        ax.legend(fontsize=11, title="validation (50k steps)", title_fontsize=11)
        cms_label(ax)
        path = f"{args.out}/loss_curves/{fname}"
        fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        print("  ->", os.path.relpath(path))

    print("\nDONE ->", f"{args.out}/loss_curves")


if __name__ == "__main__":
    main()
