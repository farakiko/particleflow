#!/usr/bin/env python
"""(2) Read-only separation check: do the data-available trackster shape features carry
signal for the two neutral problems, BEFORE we pay a schema change + re-postprocess?

For each neutral SimTICLCandidate (photon/neutral-had) we anchor to its argmax-shared
CLUE3DHigh trackster (same rule as postprocessing) and record that trackster's reco
features + the energy closure (rawE/truthE). We then ask:
  (i)  CLASSIFICATION: do features separate photon-anchored vs nhad-anchored tracksters?
       (attacks the 47% nhad->photon confusion). Metric = AUC (0.5=none, ->0/1 = strong).
  (ii) REGRESSION: within nhad, do features separate under-captured (closure<0.4) from
       well-captured (closure>0.8) tracksters? (would help predict the energy correction).
Features: EV eigenvalues + ratios, EM fraction, |barycenter_z| depth, time, nhits, rawE.
"""
import glob, argparse, os
import numpy as np, awkward as ak, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

TS  = "ticlTrackstersCLUE3DHigh"
S2R = "SimCP2ticlTrackstersCLUE3DHighByHits"
NANO = "/Users/fmokhtar/projects/particleflow/data/cms/phase2/offline/Aug31/nano"
BR = ["SimTICLCandidates_pdgID", "SimTICLCandidates_energy",
      f"{S2R}_n{S2R}Links", f"{S2R}Links_index", f"{S2R}Links_sharedEnergy",
      f"{TS}_EV1", f"{TS}_EV2", f"{TS}_EV3", f"{TS}_raw_energy", f"{TS}_raw_em_energy",
      f"{TS}_barycenter_z", f"{TS}_time", f"{TS}_timeError",
      f"{TS}_nticlTrackstersCLUE3DHighvertices"]
FEATS = ["rawE", "em_frac", "EV1", "EV2/EV1", "EV3/EV1", "|z| depth", "time", "nhits"]


def rankdata(a):
    a = np.asarray(a, float); n = len(a)
    order = np.argsort(a, kind="mergesort"); sa = a[order]
    out = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        out[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r = np.empty(n); r[order] = out
    return r


def auc(pos, neg):
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def roc_curve(pos, neg):
    """ROC treating pos as the positive class; higher score = more positive."""
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort"); labels = labels[order]
    tpr = np.concatenate([[0], np.cumsum(labels) / len(pos)])
    fpr = np.concatenate([[0], np.cumsum(1 - labels) / len(neg)])
    return fpr, tpr


LOGX = {0, 2, 7}   # rawE, EV1, nhits


def make_plots(P, N, outdir):
    os.makedirs(outdir, exist_ok=True)
    # ---- Fig A: feature distributions, photon vs nhad ----
    fig, axs = plt.subplots(2, 4, figsize=(18, 8.2)); axs = axs.ravel()
    for k, f in enumerate(FEATS):
        ax = axs[k]; pv = P[:, k]; nv = N[:, k]
        pv = pv[np.isfinite(pv)]; nv = nv[np.isfinite(nv)]
        allv = np.concatenate([pv, nv]); lo, hi = np.percentile(allv, [1, 99])
        if k in LOGX:
            posv = allv[allv > 0]; lo = np.percentile(posv, 1) if len(posv) else 1e-3
            if not (hi > lo): hi = lo * 10
            bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 40); ax.set_xscale("log")
        else:
            if not (hi > lo): hi = lo + 1.0
            bins = np.linspace(lo, hi, 40)
        ax.hist(pv, bins=bins, histtype="step", lw=2, color="tab:blue", density=True, label=f"photon ({len(pv)})")
        ax.hist(nv, bins=bins, histtype="step", lw=2, color="tab:red", density=True, label=f"nhad ({len(nv)})")
        ax.set_title(f); ax.set_xlabel(f); ax.set_ylabel("density"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle(r"Trackster shape features by anchored class ($\gamma$ vs neutral hadron), ttbar 0 PU", fontsize=15)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "ts_feature_distributions.pdf")); plt.close(fig)

    # ---- Fig B: ROC curves, gamma vs nhad, per feature ----
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    order = sorted(range(len(FEATS)), key=lambda k: -abs(auc(P[:, k], N[:, k]) - 0.5))
    for k in order:
        A = auc(P[:, k], N[:, k]); sign = 1.0 if A >= 0.5 else -1.0
        fpr, tpr = roc_curve(sign * P[:, k], sign * N[:, k])
        ax.plot(fpr, tpr, lw=2, label=f"{FEATS[k]:9s} AUC {max(A, 1-A):.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="no separation")
    ax.set_xlabel("neutral-hadron mis-ID rate (FPR)"); ax.set_ylabel(r"photon efficiency (TPR)")
    ax.set_title(r"$\gamma$ vs neutral-hadron trackster separation (per feature)")
    ax.legend(fontsize=10, loc="lower right", prop={"family": "monospace"}); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "ts_feature_roc.pdf")); plt.close(fig)
    print(f"\nplots -> {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="ttbar_0pu"); ap.add_argument("--max-files", type=int, default=30)
    ap.add_argument("--outdir", default="/Users/fmokhtar/projects/particleflow/plots/phase2/ts_features")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{NANO}/{a.sample}/*.root"))[:a.max_files] if a.max_files > 0 \
        else sorted(glob.glob(f"{NANO}/{a.sample}/*.root"))

    rows = {"photon": [], "nhad": []}   # per anchored trackster: [feat..., closure]
    for fn in files:
        E = uproot.open(fn)["Events"].arrays(BR)
        for iev in range(len(E)):
            g = lambda b: ak.to_numpy(E[b][iev])
            pid = g("SimTICLCandidates_pdgID"); en = g("SimTICLCandidates_energy")
            EV1 = g(f"{TS}_EV1"); EV2 = g(f"{TS}_EV2"); EV3 = g(f"{TS}_EV3")
            rawE = g(f"{TS}_raw_energy"); emE = g(f"{TS}_raw_em_energy")
            bz = g(f"{TS}_barycenter_z"); tm = g(f"{TS}_time"); nh = g(f"{TS}_nticlTrackstersCLUE3DHighvertices")
            n_ts = len(rawE)
            cnt = g(f"{S2R}_n{S2R}Links"); off = np.concatenate([[0], np.cumsum(cnt)]).astype(int)
            idx = g(f"{S2R}Links_index"); she = g(f"{S2R}Links_sharedEnergy")
            for i in range(len(pid)):
                ap_ = abs(int(pid[i]))
                if ap_ not in (22, 130) or en[i] <= 0:
                    continue
                ii = idx[off[i]:off[i + 1]]; ss = she[off[i]:off[i + 1]]
                if not (len(ss) and ss.max() > 0):
                    continue
                t = int(ii[int(np.argmax(ss))])
                if not (0 <= t < n_ts) or rawE[t] <= 0:
                    continue
                feat = [rawE[t], emE[t] / rawE[t], EV1[t],
                        EV2[t] / EV1[t] if EV1[t] > 0 else np.nan,
                        EV3[t] / EV1[t] if EV1[t] > 0 else np.nan,
                        abs(bz[t]), tm[t] if tm[t] > -50 else np.nan, nh[t]]
                rows["photon" if ap_ == 22 else "nhad"].append(feat + [rawE[t] / en[i]])

    P = np.array(rows["photon"]); N = np.array(rows["nhad"])
    print(f"sample={a.sample}  files={len(files)}   photon-anchored={len(P)}  nhad-anchored={len(N)}\n")

    print("(i) CLASSIFICATION — photon vs nhad trackster (median photon | median nhad | AUC photon>nhad):")
    print(f"    {'feature':10s}{'med γ':>10}{'med nhad':>11}{'AUC':>8}   (|AUC-0.5| = separation)")
    for k, f in enumerate(FEATS):
        pv, nv = P[:, k], N[:, k]
        mp = np.nanmedian(pv); mn = np.nanmedian(nv); A = auc(pv, nv)
        star = " <<" if abs(A - 0.5) > 0.15 else ""
        print(f"    {f:10s}{mp:>10.2f}{mn:>11.2f}{A:>8.2f}{star}")

    print("\n(ii) REGRESSION — within nhad, under-captured (closure<0.4) vs well-captured (>0.8):")
    clo = N[:, -1]; lo = N[clo < 0.4]; hi = N[clo > 0.8]
    print(f"    under-captured N={len(lo)}   well-captured N={len(hi)}")
    print(f"    {'feature':10s}{'med low':>10}{'med high':>11}{'AUC low>high':>14}")
    for k, f in enumerate(FEATS):
        A = auc(lo[:, k], hi[:, k])
        star = " <<" if abs(A - 0.5) > 0.15 else ""
        print(f"    {f:10s}{np.nanmedian(lo[:, k]):>10.2f}{np.nanmedian(hi[:, k]):>11.2f}{A:>14.2f}{star}")

    make_plots(P, N, a.outdir)


if __name__ == "__main__":
    main()
