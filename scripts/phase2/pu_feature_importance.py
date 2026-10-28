#!/usr/bin/env python
"""
Phase-2 200PU MLPF -- which INPUT FEATURES the pileup head relies on (permutation importance).

Answers "what does the trained PU head actually lean on to separate pileup from hard scatter?"
by *permutation importance*: for each input feature, shuffle that feature's values across the
real elements of each event (destroying its information while keeping its marginal distribution),
re-run inference, and measure how much the per-element PU-classification AUC drops. A large drop
=> the model relies on that feature; ~0 => it is ignored (or fully redundant with another feature).

This measures the *existing* model (no retraining). It is the cheap first pass; confirm the top
features afterwards with leave-one-out retrains (model can re-adapt => "needs" vs merely "uses").

Metric: per-element AUC of P(pileup) vs the truth pileup label (binarised `ispu`), over real target
particles (cls_id != 0) in the HGCAL endcap (1.48 < |eta| < 3.0) -- the same selection as
pu_head_perf.py. P(pileup) = softmax(model PU logits)[..., 1], exactly as mlpf.py produces it.

Correlated features split credit (permuting one alone underestimates both because the model falls
back on the other). Use --groups to also permute predefined correlated groups jointly.

Outputs under <out>/pu_feat_importance/:
  importance_<model>.pdf        horizontal bar of AUC-drop per feature (sorted), with repeat-std error bars
  importance_heatmap.pdf        features x models heatmap of AUC-drop (base features)
  importance_groups.pdf         grouped-permutation AUC-drop per model (only with --groups)
  importance.json               raw numbers (baseline AUC, per-feature mean/std drop, per model)

Usage (inside the mlpf venv, 1 GPU):
  python scripts/phase2/pu_feature_importance.py \
      --experiments-root /shared/mlpf-phase2/experiments \
      --data-dir /shared/mlpf-phase2/tfds_clean \
      --out /shared/mlpf-phase2/eval_plots/phase2_physics \
      --nevents 500 --nrepeat 3 --groups
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import re
import json
import glob
import pickle
import argparse

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

from mlpf.conf import X_FEATURES
from mlpf.model.mlpf import MLPF
from mlpf.model.utils import unpack_target
from mlpf.model.PFDataset import PFDataset, Collater

hep.style.use("CMS")

# ----------------------------------------------------------------------------- configuration
SAMPLE = "cms_pf_phase2_ttbar200pu"
RLABEL = r"$\mathrm{t}\bar{\mathrm{t}}$, 200 PU (14 TeV)"
ENDCAP = (1.48, 3.0)
BASE_FEATS = X_FEATURES["cms_phase2"]  # 38 base input features (order == model input columns)
# T1 appends these 5 engineered PUPPI-style columns (compute_engineered_feats in PFDataset.py)
ENG_FEATS = ["eng_z_lv", "eng_dz_to_lv", "eng_alpha_F", "eng_alpha_F2", "eng_alpha_C"]

# (display label, experiment subdir, uses engineered feats). Only models WITH a PU head.
# Defaults point at the clean-flag (-clean) retrains; override with --experiments-root / edit here.
MODELS = [
    ("pu (unweighted)", "ttbar200-heptv2-pu-clean", False),
    ("T0 pt-weighted", "ttbar200-heptv2-pu-rew-clean", False),
    ("T1 engineered feats", "ttbar200-heptv2-pu-feats-clean", True),
    ("T2 soft target", "ttbar200-heptv2-pu-soft-clean", False),
    ("T3 global context", "ttbar200-heptv2-pu-ctx-clean", False),
]

# correlated groups (by base-feature name) -- permuted jointly under --groups so redundancy
# inside a group does not hide the group's true importance.
GROUPS = {
    "kinematics (pt,e,p)": ["pt", "e", "px", "py", "pz", "em_energy"],
    "position (eta,phi,z)": ["eta", "sin_phi", "cos_phi", "bary_z"],
    "track-density": ["min_dR_track", "near_track_pt", "sum_pt_dR10",
                      "n_trk_dR01", "n_trk_dR02", "n_trk_dR03", "n_trk_dR04", "n_trk_dR05"],
    "trackster-shape": ["ev1", "ev2", "ev3", "nhits"],
    "timing": ["time", "timeerror"],
    "track-quality": ["pterror", "etaerror", "phierror", "lambdaerror", "qoverperror"],
    "vertex": ["vx", "vy", "vz"],
    "muon/gsf-id": ["muon_type", "muon_dt_hits", "muon_csc_hits", "gsf_type"],
    "elem-type": ["typ_idx", "charge"],
}


# ----------------------------------------------------------------------------- helpers
def slug(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def cms_label(ax, note=None):
    hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=16)
    if note:
        ax.text(0.5, 1.13, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=13, style="italic", color="0.30")


def auc_score(score, label):
    """AUC of a binary score (positive class = pileup, label==1). Rank-based, no sklearn dep."""
    label = label.astype(float)
    P = label.sum()
    N = len(label) - P
    if P == 0 or N == 0:
        return np.nan
    # Mann-Whitney / AUC via tie-averaged ranks (positive class = pileup)
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_rank = csum - (counts - 1) / 2.0  # mean 1-indexed rank of each unique value
    ranks = avg_rank[inv]
    return float((ranks[label == 1].sum() - P * (P + 1) / 2.0) / (P * N))


def load_model(exp_dir, device):
    """Rebuild the model from the experiment's saved config pickle + latest checkpoint."""
    with open(os.path.join(exp_dir, "model_kwargs.pkl"), "rb") as f:
        config = pickle.load(f)
    model = MLPF(config).to(device)
    ckpt = sorted(glob.glob(os.path.join(exp_dir, "checkpoints", "checkpoint-*.pth")),
                  key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))[-1]
    state = torch.load(ckpt, map_location=device)
    msg = model.load_state_dict(state["model_state_dict"], strict=False)
    if msg.missing_keys:
        print(f"    [warn] missing keys: {msg.missing_keys[:4]}{'...' if len(msg.missing_keys) > 4 else ''}")
    model.eval()
    step = int(re.search(r"checkpoint-(\d+)", ckpt).group(1))
    return model, config, step


def cache_batches(data_dir, config, engineered, nevents, batch_size, nconfigs):
    """Decode up to nevents test events into a list of CPU PFBatch (X, ytarget)."""
    pad = config.pad_to_multiple_elements
    datasets = []
    for i in range(1, nconfigs + 1):
        # builder config i (1..N); no version pin -> tfds resolves the latest built version
        ds = PFDataset(data_dir, f"{SAMPLE}/{i}", "test", pad_to_multiple=pad, engineered_feats=engineered).ds
        datasets.append(ds)
    ds = torch.utils.data.ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         collate_fn=Collater(["X", "ytarget"], []), num_workers=0)
    batches, seen = [], 0
    for b in loader:
        batches.append(b)
        seen += b.X.shape[0]
        if seen >= nevents:
            break
    return batches


@torch.no_grad()
def infer_scores(model, batches, device, perm_cols=None, rng=None):
    """Run inference over cached batches (optionally permuting perm_cols jointly within each event's
    real elements) and return flat endcap target-particle arrays: P(pileup), truth ispu, pt, eta, typ."""
    sc, lab, pt, eta, typ = [], [], [], [], []
    for b in batches:
        X = b.X.clone()
        mask = X[:, :, 0] != 0  # padding mask (real elements)
        if perm_cols is not None:
            for r in range(X.shape[0]):
                idx = torch.nonzero(mask[r], as_tuple=False).flatten()
                if idx.numel() < 2:
                    continue
                perm = idx[torch.as_tensor(rng.permutation(idx.numel()))]
                for c in perm_cols:
                    X[r, idx, c] = X[r, perm, c]
        Xd, md = X.to(device), mask.to(device)
        preds_pu = model(Xd, md)[3]  # (B, N, 2) raw logits
        prob = torch.softmax(preds_pu.float(), dim=-1)[:, :, 1].cpu()  # P(pileup)

        tg = unpack_target(b.ytarget, model)
        cls = tg["cls_id"]
        te = tg["eta"]
        sel = mask & (cls != 0) & (te.abs() >= ENDCAP[0]) & (te.abs() <= ENDCAP[1])
        sc.append(prob[sel].numpy())
        lab.append(tg["ispu"][sel].numpy())
        pt.append(tg["pt"][sel].numpy())
        eta.append(te[sel].numpy())
        typ.append(cls[sel].numpy())
    return (np.concatenate(sc), np.concatenate(lab), np.concatenate(pt),
            np.concatenate(eta), np.concatenate(typ))


# ----------------------------------------------------------------------------- per-model importance
def run_model(label, exp_dir, engineered, args, device):
    print(f"\n===== {label}  ({os.path.basename(exp_dir)}) =====")
    model, config, step = load_model(exp_dir, device)
    feats = list(BASE_FEATS) + (ENG_FEATS if engineered else [])
    n_in = len(feats)
    print(f"    loaded step {step}, {n_in} input features, engineered={engineered}")

    batches = cache_batches(args.data_dir, config, engineered, args.nevents, args.batch_size, args.nconfigs)
    nev = sum(b.X.shape[0] for b in batches)
    base = infer_scores(model, batches, device)
    base_auc = auc_score(base[0], base[1])
    npart = len(base[0])
    print(f"    {nev} events, {npart} endcap target particles, baseline AUC = {base_auc:.4f}")

    # per-feature permutation importance (AUC drop), averaged over nrepeat shuffles
    drop_mean = np.full(n_in, np.nan)
    drop_std = np.full(n_in, np.nan)
    for j in range(n_in):
        drops = []
        for r in range(args.nrepeat):
            rng = np.random.default_rng(args.seed + 1000 * j + r)
            s = infer_scores(model, batches, device, perm_cols=[j], rng=rng)
            drops.append(base_auc - auc_score(s[0], s[1]))
        drop_mean[j], drop_std[j] = np.mean(drops), np.std(drops)
        print(f"    [{j:2d}] {feats[j]:<18s}  drop = {drop_mean[j]:+.4f} +- {drop_std[j]:.4f}")

    # grouped permutation (optional)
    group_drop = {}
    if args.groups:
        name2col = {f: i for i, f in enumerate(feats)}
        for gname, members in GROUPS.items():
            cols = [name2col[m] for m in members if m in name2col]
            if not cols:
                continue
            drops = []
            for r in range(args.nrepeat):
                rng = np.random.default_rng(args.seed + 7777 + r)
                s = infer_scores(model, batches, device, perm_cols=cols, rng=rng)
                drops.append(base_auc - auc_score(s[0], s[1]))
            group_drop[gname] = [float(np.mean(drops)), float(np.std(drops))]
            print(f"    (group) {gname:<22s}  drop = {np.mean(drops):+.4f} +- {np.std(drops):.4f}")

    return dict(label=label, exp=os.path.basename(exp_dir), step=step, engineered=engineered,
                feats=feats, base_auc=base_auc, n_particles=int(npart),
                drop_mean=drop_mean.tolist(), drop_std=drop_std.tolist(), group_drop=group_drop)


# ----------------------------------------------------------------------------- plots
def plot_model_bars(res, outdir, topn):
    feats = np.array(res["feats"])
    dm = np.array(res["drop_mean"])
    ds = np.array(res["drop_std"])
    order = np.argsort(dm)[::-1][:topn]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(order) + 1.6))
    y = np.arange(len(order))[::-1]
    ax.barh(y, dm[order], xerr=ds[order], color="#1f77b4", alpha=0.85, error_kw=dict(lw=1, ecolor="0.3"))
    ax.set_yticks(y)
    ax.set_yticklabels(feats[order], fontsize=11)
    ax.set_xlabel("AUC drop when feature permuted (higher = more relied on)")
    ax.axvline(0, color="0.5", lw=0.8)
    cms_label(ax, f"{res['label']} -- baseline AUC {res['base_auc']:.3f}")
    _save(fig, os.path.join(outdir, f"importance_{slug(res['label'])}.pdf"))


def plot_heatmap(results, outdir):
    # base features only (common across all models), ranked by mean importance across models
    mats, labels = [], []
    for r in results:
        name2d = {f: d for f, d in zip(r["feats"], r["drop_mean"])}
        mats.append([name2d.get(f, np.nan) for f in BASE_FEATS])
        labels.append(r["label"])
    M = np.array(mats).T  # (nfeat, nmodel)
    rank = np.argsort(np.nanmean(M, axis=1))[::-1]
    M, feats = M[rank], np.array(BASE_FEATS)[rank]
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 3, 0.30 * len(feats) + 2))
    vmax = np.nanpercentile(np.abs(M), 98) or 1.0
    im = ax.imshow(M, aspect="auto", cmap="magma_r", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=9)
    fig.colorbar(im, ax=ax, label="AUC drop", fraction=0.046, pad=0.04)
    ax.set_title("PU-head feature importance (permutation)", fontsize=13)
    _save(fig, os.path.join(outdir, "importance_heatmap.pdf"))


def plot_groups(results, outdir):
    gnames = list(GROUPS.keys())
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(gnames) + 1.8))
    y = np.arange(len(gnames))[::-1]
    w = 0.8 / max(1, len(results))
    for k, r in enumerate(results):
        vals = [r["group_drop"].get(g, [np.nan, 0])[0] for g in gnames]
        err = [r["group_drop"].get(g, [np.nan, 0])[1] for g in gnames]
        ax.barh(y + (k - len(results) / 2) * w + w / 2, vals, height=w, xerr=err,
                label=r["label"], error_kw=dict(lw=0.8, ecolor="0.4"))
    ax.set_yticks(y)
    ax.set_yticklabels(gnames, fontsize=11)
    ax.set_xlabel("AUC drop when feature GROUP permuted jointly")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.legend(fontsize=9, loc="lower right")
    cms_label(ax, "grouped permutation (correlated features together)")
    _save(fig, os.path.join(outdir, "importance_groups.pdf"))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  ->", path)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments-root", default="/shared/mlpf-phase2/experiments")
    ap.add_argument("--data-dir", default="/shared/mlpf-phase2/tfds_clean")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--nevents", type=int, default=500, help="test events to evaluate on")
    ap.add_argument("--nconfigs", type=int, default=2, help="tfds builder configs (1..N) to draw events from")
    ap.add_argument("--nrepeat", type=int, default=3, help="permutation shuffles to average per feature")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--topn", type=int, default=25, help="features shown in per-model bar chart")
    ap.add_argument("--groups", action="store_true", help="also run grouped (correlated) permutation")
    ap.add_argument("--models", default="", help="comma-separated MODELS indices to keep (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    models = MODELS
    if args.models:
        keep = [int(i) for i in args.models.split(",")]
        models = [MODELS[i] for i in keep]

    outdir = os.path.join(args.out, "pu_feat_importance")
    os.makedirs(outdir, exist_ok=True)
    device = torch.device(args.device)
    print(f"device={device}  data_dir={args.data_dir}  out={outdir}")

    results = []
    for label, exp, eng in models:
        exp_dir = os.path.join(args.experiments_root, exp)
        if not os.path.isdir(os.path.join(exp_dir, "checkpoints")):
            print(f"  [skip] no checkpoints for {exp}")
            continue
        try:
            results.append(run_model(label, exp_dir, eng, args, device))
        except Exception as e:  # keep going if one model fails
            print(f"  [error] {label}: {e}")

    if not results:
        print("no models evaluated"); return

    for r in results:
        plot_model_bars(r, outdir, args.topn)
    plot_heatmap(results, outdir)
    if args.groups and any(r["group_drop"] for r in results):
        plot_groups(results, outdir)

    with open(os.path.join(outdir, "importance.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  ->", os.path.join(outdir, "importance.json"))

    # stdout summary: top-8 features per model
    print("\n================= TOP FEATURES (AUC drop) =================")
    for r in results:
        feats, dm = np.array(r["feats"]), np.array(r["drop_mean"])
        order = np.argsort(dm)[::-1][:8]
        top = ", ".join(f"{feats[i]} ({dm[i]:+.3f})" for i in order)
        print(f"  {r['label']:<22s} baseAUC={r['base_auc']:.3f} | {top}")


if __name__ == "__main__":
    main()
