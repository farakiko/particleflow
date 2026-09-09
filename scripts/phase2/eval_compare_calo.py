#!/usr/bin/env python
"""Overlay the two trained calo variants (clue3d vs links) w.r.t. gen in one plot directory.
Each model is evaluated on its OWN val split (pkl_clue3d / pkl_links). Reuses the helpers and
val-split logic from eval_local_mps so the matching/binning is identical to the single-model eval.

  clue3d = blue, links = red, gen = black reference.
Plots: jet response, per-class efficiency, fake rate, ΣpT ratio to gen, per-particle pT response.
"""
import os, sys, argparse
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "mlpf/heptfds/cms_pf_phase2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlpf.model.mlpf import MLPF
from mlpf.model.utils import unpack_predictions
from eval_local_mps import (build_config, cluster, pdg_cls, load_val, CLS, CLASSES, CLABEL,
                            PTED, PTC, NB, ELO, EHI, DR, GENPT)

VAR = [("clue3d", "pkl_clue3d", "tab:blue"), ("links", "pkl_links", "tab:red")]


def eval_variant(ckpt, pkl_subdir, device):
    ck = torch.load(ckpt, map_location="cpu")
    c = ck.get("config", {"num_convs": 3, "num_heads": 16, "head_dim": 16})
    model = MLPF(build_config(c["num_convs"], c["num_heads"], c["head_dim"]))
    model.load_state_dict(ck["model_state_dict"]); model.eval().to(device)
    val = load_val(pkl_subdir)
    resp = []; eff = {c: np.zeros((NB, 2)) for c in CLASSES}; fake = {c: np.zeros((NB, 2)) for c in CLASSES}
    sump = {c: 0.0 for c in CLASSES}; gsump = {c: 0.0 for c in CLASSES}; respp = {c: [] for c in CLASSES}
    B = 16
    with torch.no_grad():
        for bi in range(0, len(val), B):
            evb = val[bi:bi+B]; N = max(len(e["X"]) for e in evb); nf = evb[0]["X"].shape[1]
            X = np.zeros((len(evb), N, nf), np.float32); mask = np.zeros((len(evb), N), bool)
            for i, e in enumerate(evb):
                X[i, :len(e["X"])] = e["X"]; mask[i, :len(e["X"])] = True
            pred = unpack_predictions(model(torch.tensor(X, device=device), torch.tensor(mask, device=device)))
            cid = pred["cls_id"].cpu().numpy(); mom = pred["momentum"].cpu().numpy()
            for i, e in enumerate(evb):
                n = len(e["X"]); xe = e["X"]; ci = cid[i, :n]; m = mom[i, :n]; sel = ci != 0
                p_pt = np.exp(m[sel, 0])*xe[sel, 1]; p_eta = m[sel, 1]
                p_phi = np.arctan2(m[sel, 2], m[sel, 3]); p_e = np.exp(m[sel, 4])*xe[sel, 5]
                p_cls = np.array([pdg_cls(CLS[k]) for k in ci[sel]])
                gj = e["gj"].reshape(-1, 4); mj = cluster(p_pt, p_eta, p_phi, p_e)
                for g in gj:
                    if g[0] < GENPT or abs(g[1]) < ELO or abs(g[1]) > EHI or len(mj) == 0: continue
                    dphi = np.arctan2(np.sin(mj[:, 2]-g[2]), np.cos(mj[:, 2]-g[2]))
                    b = int(np.argmin(np.hypot(mj[:, 1]-g[1], dphi)))
                    if np.hypot(mj[b, 1]-g[1], dphi[b]) < DR: resp.append(mj[b, 0]/g[0])
                yp = e["yp"].reshape(-1, 5); gm = (np.abs(yp[:, 2]) > ELO) & (np.abs(yp[:, 2]) < EHI)
                g_pid, g_pt, g_eta, g_phi = yp[gm, 0], yp[gm, 1], yp[gm, 2], yp[gm, 3]
                g_cls = np.array([pdg_cls(x) for x in g_pid])
                rmask = (np.abs(p_eta) > ELO) & (np.abs(p_eta) < EHI)
                re, rp, rf, rc = p_eta[rmask], p_pt[rmask], p_phi[rmask], p_cls[rmask]
                for cc in CLASSES: sump[cc] += rp[rc == cc].sum()
                for gi in range(len(g_pt)):
                    cc = g_cls[gi]; bb = int(np.clip(np.digitize(g_pt[gi], PTED)-1, 0, NB-1)); eff[cc][bb, 0] += 1
                    same = rc == cc
                    if same.any():
                        dphi = np.arctan2(np.sin(rf[same]-g_phi[gi]), np.cos(rf[same]-g_phi[gi]))
                        if np.min(np.hypot(re[same]-g_eta[gi], dphi)) < DR: eff[cc][bb, 1] += 1
                for ri in range(len(rp)):
                    cc = rc[ri]; bb = int(np.clip(np.digitize(rp[ri], PTED)-1, 0, NB-1)); fake[cc][bb, 0] += 1
                    same = g_cls == cc; ok = False
                    if same.any():
                        dphi = np.arctan2(np.sin(g_phi[same]-rf[ri]), np.cos(g_phi[same]-rf[ri]))
                        j = int(np.argmin(np.hypot(g_eta[same]-re[ri], dphi)))
                        if np.hypot(g_eta[same][j]-re[ri], dphi[j]) < DR:
                            ok = True
                            if g_pt[same][j] > 0: respp[cc].append(rp[ri]/g_pt[same][j])
                    if not ok: fake[cc][bb, 1] += 1
                for cc in CLASSES: gsump[cc] += g_pt[g_cls == cc].sum()
    return dict(resp=np.array(resp), eff=eff, fake=fake, sump=sump, gsump=gsump, respp=respp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(REPO, "plots/phase2/mlpf_eval_clue3d_vs_links"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
    res = {}
    for name, sub, _ in VAR:
        ck = os.path.join(REPO, f"experiments/phase2_{name}/checkpoints/best.pth")
        print(f"eval {name} ({ck})"); res[name] = eval_variant(ck, sub, a.device)

    # Fig 1: jet response
    fig, ax = plt.subplots(figsize=(7.6, 5.6)); b = np.logspace(-1, 1, 120)
    for name, _, col in VAR:
        r = res[name]["resp"]; ax.hist(r, bins=b, histtype="step", lw=2.4, color=col,
                                       label=f"{name} (med {np.median(r):.3f})")
    ax.axvline(1, color="k", ls="--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_\mathrm{T}$ / gen-jet $p_\mathrm{T}$"); ax.set_ylabel("jets")
    ax.set_title(r"MLPF jet response vs gen: clue3d vs links ($1.5<|\eta|<3$, gen $p_T>20$)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "calo_jet_response.pdf")); plt.close(fig)

    # Fig 2/3: efficiency + fake grids
    def grid(key, title, fn):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); axs = axs.ravel()
        for k, c in enumerate(CLASSES):
            ax = axs[k]
            for name, _, col in VAR:
                d = res[name][key][c]
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = np.where(d[:, 0] > 0, d[:, 1]/d[:, 0], np.nan)
                ax.plot(PTC, y, "o-", color=col, lw=2, label=name)
            ax.set_xscale("log"); ax.set_ylim(-.03, 1.03); ax.set_title(CLABEL[c])
            ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]"); ax.grid(alpha=.3); ax.legend(fontsize=9)
        axs[5].axis("off"); fig.suptitle(title + r" vs gen (same-class, $\Delta R<0.2$, $1.5<|\eta|<3$)", fontsize=14)
        plt.tight_layout(); plt.savefig(os.path.join(a.outdir, fn)); plt.close(fig)
    grid("eff", "Efficiency", "calo_efficiency_vs_pt.pdf")
    grid("fake", "Fake rate", "calo_fakerate_vs_pt.pdf")

    # Fig 4: ΣpT ratio to gen per class
    fig, ax = plt.subplots(figsize=(9, 5.2)); x = np.arange(len(CLASSES)); w = 0.38
    for j, (name, _, col) in enumerate(VAR):
        vals = [res[name]["sump"][c]/max(res[name]["gsump"][c], 1e-9) for c in CLASSES]
        ax.bar(x + (j-0.5)*w, vals, w, color=col, label=f"{name}/gen")
    ax.axhline(1, color="k", ls="--", lw=1); ax.set_xticks(x); ax.set_xticklabels([CLABEL[c] for c in CLASSES], rotation=20)
    ax.set_ylabel(r"$\Sigma p_\mathrm{T}$ ratio to gen"); ax.set_title(r"$\Sigma p_\mathrm{T}$ per class: clue3d vs links ($1.5<|\eta|<3$)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "calo_sumpt_per_class.pdf")); plt.close(fig)

    # Fig 5: per-particle pT response (same-pid)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); axs = axs.ravel(); bb = np.logspace(-1, 1, 80)
    for k, c in enumerate(CLASSES):
        ax = axs[k]
        for name, _, col in VAR:
            r = np.array(res[name]["respp"][c])
            if len(r): ax.hist(r, bins=bb, histtype="step", lw=2, color=col, density=True,
                               label=f"{name} (med {np.median(r):.2f})")
        ax.axvline(1, color="k", ls="--", lw=1); ax.set_xscale("log"); ax.set_title(CLABEL[c])
        ax.set_xlabel(r"reco $p_\mathrm{T}$/gen $p_\mathrm{T}$"); ax.grid(alpha=.3); ax.legend(fontsize=9)
    axs[5].axis("off"); fig.suptitle(r"Per-particle pT response vs gen (same-class): clue3d vs links", fontsize=14)
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "calo_particle_ptresponse.pdf")); plt.close(fig)

    print("\nsummary (MLPF vs gen):")
    print(f"  jet response median:  " + "  ".join(f"{n}={np.median(res[n]['resp']):.3f}" for n, _, _ in VAR))
    for c in CLASSES:
        line = f"  {c:9s}"
        for n, _, _ in VAR:
            e = res[n]["eff"][c][:, 1].sum()/max(res[n]["eff"][c][:, 0].sum(), 1)
            s = res[n]["sump"][c]/max(res[n]["gsump"][c], 1e-9)
            line += f"  {n}: eff={e:.2f} ΣpT={s:.2f}"
        print(line)
    print("plots ->", a.outdir)


if __name__ == "__main__":
    main()
