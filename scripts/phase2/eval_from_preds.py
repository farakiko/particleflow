#!/usr/bin/env python3
"""Per-class evaluation of MLPF from the pipeline's prediction parquets
(preds_test/<sample>/*.parquet). Complements the jet-level make_plots suite with
the physics-level breakdown of docs/phase2.md §8: per-element confusion matrix,
per-class efficiency & fake rate vs pT, per-class regression response.

  pixi run python3 scripts/phase2/eval_from_preds.py \
      --preds data/cms/phase2/preds_test --outdir plots/phase2/eval_perclass
"""
import argparse
import glob
import os

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
import numpy as np

mplhep.style.use("CMS")

CLASSES = ["null", "ch.had", "n.had", r"$\gamma$", r"$e^\pm$", r"$\mu^\pm$"]
NCLS = len(CLASSES)
PT_BINS = np.logspace(-1, 3, 25)


def load_flat(pattern, max_files=None):
    files = sorted(glob.glob(pattern))
    if max_files:
        files = files[:max_files]
    assert files, f"no files for {pattern}"
    cols = {f"{t}_{v}": [] for t in ["target", "pred"] for v in ["cls_id", "pt", "energy", "eta"]}
    nev = 0
    for f in files:
        d = ak.from_parquet(f, columns=["particles.target.cls_id", "particles.target.pt",
                                        "particles.target.energy", "particles.target.eta",
                                        "particles.pred.cls_id", "particles.pred.pt",
                                        "particles.pred.energy", "particles.pred.eta"])
        nev += len(d)
        for t in ["target", "pred"]:
            for v in ["cls_id", "pt", "energy", "eta"]:
                cols[f"{t}_{v}"].append(ak.to_numpy(ak.flatten(d["particles"][t][v])))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    print(f"{pattern}: {len(files)} files, {nev} events, {len(out['target_cls_id'])} elements")
    return out, nev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="preds dir containing <sample>/ subdirs")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-files", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    samples = sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.preds, "*")) if os.path.isdir(p))
    print("samples:", samples)
    flats = {}
    for s in samples:
        flats[s], _ = load_flat(os.path.join(a.preds, s, "*.parquet"), a.max_files)
    allf = {k: np.concatenate([flats[s][k] for s in samples]) for k in flats[samples[0]]}

    tc = allf["target_cls_id"].astype(int)
    pc = allf["pred_cls_id"].astype(int)

    # ---- 1. per-element confusion matrix (row-normalized over target class)
    cm = np.zeros((NCLS, NCLS))
    for i in range(NCLS):
        m = tc == i
        n = m.sum()
        for j in range(NCLS):
            cm[i, j] = (pc[m] == j).sum() / max(n, 1)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NCLS):
        for j in range(NCLS):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=13)
    ax.set_xticks(range(NCLS), CLASSES, fontsize=13)
    ax.set_yticks(range(NCLS), CLASSES, fontsize=13)
    ax.set_xlabel("MLPF class")
    ax.set_ylabel("target class")
    fig.colorbar(im, ax=ax, label="fraction of target class")
    ax.set_title("all samples, per element", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "confusion_matrix.png"), dpi=140)
    plt.close(fig)
    print("\nconfusion (rows=target, cols=pred):")
    for i in range(NCLS):
        print(f"  {CLASSES[i]:>8}: " + " ".join(f"{cm[i, j]:.2f}" for j in range(NCLS)) + f"   (diag {cm[i, i]:.2f})")

    # ---- 2. efficiency (same-class and any-class) + fake rate vs pT
    for i in range(1, NCLS):
        fig, ax = plt.subplots(figsize=(10, 7))
        mt = tc == i
        pt_t = allf["target_pt"][mt]
        same = (pc[mt] == i)
        anyp = (pc[mt] != 0)
        mp = pc == i
        pt_p = allf["pred_pt"][mp]
        fake = (tc[mp] == 0)
        for vals, sel, lbl, sty in [(pt_t, same, "efficiency (same class)", "o-"),
                                    (pt_t, anyp, "efficiency (any class)", "s--"),
                                    (pt_p, fake, "fake rate", "^:")]:
            ys, xs = [], []
            for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:]):
                m = (vals >= lo) & (vals < hi)
                if m.sum() >= 20:
                    xs.append(np.sqrt(lo * hi))
                    ys.append(sel[m].mean())
            ax.plot(xs, ys, sty, lw=2, label=lbl)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.15)
        ax.set_xlabel(f"{CLASSES[i]} $p_T$ (GeV)")
        ax.set_ylabel("fraction")
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        mplhep.cms.label(ax=ax, data=False, label="Preliminary", rlabel="Phase-2 (14 TeV)")
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, f"eff_fake_cls{i}.png"), dpi=140)
        plt.close(fig)
        print(f"{CLASSES[i]:>8}: eff(same)={same.mean():.3f} eff(any)={anyp.mean():.3f} "
              f"fake={fake.mean():.3f} (n_target={mt.sum()}, n_pred={mp.sum()})")

    # ---- 3. per-class regression response (pred/target, same element, both nonzero)
    fig, ax = plt.subplots(figsize=(10, 7))
    b = np.linspace(0, 2, 101)
    for i in range(1, NCLS):
        m = (tc == i) & (pc != 0) & (allf["target_pt"] > 0) & (allf["pred_pt"] > 0)
        if m.sum() < 20:
            print(f"{CLASSES[i]:>8}: too few matched elements ({m.sum()}), skipping response")
            continue
        r = allf["pred_pt"][m] / allf["target_pt"][m]
        q1, q2, q3 = np.percentile(r, [25, 50, 75])
        ax.hist(r, bins=b, histtype="step", lw=2, density=True,
                label=f"{CLASSES[i]}  (med={q2:.2f}, IQR={q3 - q1:.2f})")
        print(f"{CLASSES[i]:>8}: pt response med={q2:.3f} iqr={q3 - q1:.3f} (n={m.sum()})")
    ax.axvline(1, color="gray", ls=":", lw=1)
    ax.set_xlabel("MLPF $p_T$ / target $p_T$ (same element)")
    ax.set_ylabel("density")
    ax.legend(fontsize=11)
    mplhep.cms.label(ax=ax, data=False, label="Preliminary", rlabel="Phase-2 (14 TeV)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "pt_response_per_class.png"), dpi=140)
    plt.close(fig)

    print(f"done -> {a.outdir}")


if __name__ == "__main__":
    main()
