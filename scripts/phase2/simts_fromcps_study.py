"""ticlSimTrackstersfromCPs: isPU distribution and correspondence with SimTICLCandidates and the input tracksters.

(1) ticlSimTrackstersfromCPs_isPU vs SimTICLCandidates_isPU, at 0 PU and 200 PU (counts and raw-energy weighted).
(2) Is ticlSimTrackstersfromCPs 1:1 with the input tracksters (ticlTrackstersCLUE3DHigh)? Per-event counts and the
    hit-based associator link multiplicities (RecoticlTrackstersCLUE3DHigh2SimCPByHits / SimCP2ticlTrackstersCLUE3DHighByHits).
(3) Is it 1:1 with SimTICLCandidates? Per-event counts and per-index agreement of eta / energy.
(4) Where SimTICLCandidates_isPU == -1, what does ticlSimTrackstersfromCPs_isPU say? 2D correlation of the two flags
    per sample (endcap, |barycenter eta| in [1.5, 3.0]), and the fromCPs flag for the -1 candidates alone.
(5) Why the counts differ: fraction of CPs sharing energy with any CLUE3DHigh trackster vs CP energy.

Usage: pixi run python3 scripts/phase2/simts_fromcps_study.py [--nfiles 10]
"""

import argparse
import glob
import os

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep  # noqa: E402
import numpy as np  # noqa: E402
import uproot  # noqa: E402

SAMPLES = [("ttbar_0pu", r"$t\bar{t}$ 0 PU"), ("zll_0pu", r"Z$\to\ell\ell$ 0 PU"), ("qcd_0pu", "QCD 0 PU"), ("ttbar_200pu", r"$t\bar{t}$ 200 PU")]
OUT = "plots/phase2/simts_fromcps"

CP = "ticlSimTrackstersfromCPs"
TS = "ticlTrackstersCLUE3DHigh"
BRANCHES = [
    f"n{CP}",
    "nSimTICLCandidates",
    f"n{TS}",
    f"{CP}_isPU",
    f"{CP}_raw_energy",
    f"{CP}_barycenter_eta",
    "SimTICLCandidates_isPU",
    "SimTICLCandidates_charge",
    "SimTICLCandidates_eta",
    "SimTICLCandidates_raw_energy",
    f"{TS}_raw_energy",
    f"{TS}_barycenter_eta",
    f"Reco{TS}2SimCPByHits_nReco{TS}2SimCPByHitsLinks",
    f"Reco{TS}2SimCPByHitsLinks_index",
    f"Reco{TS}2SimCPByHitsLinks_score",
    f"Reco{TS}2SimCPByHitsLinks_sharedEnergy",
    f"SimCP2{TS}ByHits_nSimCP2{TS}ByHitsLinks",
    f"SimCP2{TS}ByHitsLinks_sharedEnergy",
]


def load(sample, nfiles):
    files = sorted(glob.glob(f"data/cms/phase2/offline/nano/{sample}/*.root"))[:nfiles]
    return ak.concatenate([uproot.open(f)["Events"].arrays(BRANCHES) for f in files]), len(files)


def nested(a, counts_branch, value_branch):
    return ak.unflatten(a[value_branch], ak.flatten(a[counts_branch]), axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nfiles", type=int, default=10)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mplhep.style.use("CMS")

    res = {}
    for sample, label in SAMPLES:
        a, nf = load(sample, args.nfiles)
        r = {"label": label}
        print(f"\n===== {sample}: {nf} files, {len(a)} events")

        # (3) alignment with SimTICLCandidates
        same_n = np.mean(a[f"n{CP}"] == a["nSimTICLCandidates"])
        cp_eta = ak.flatten(a[f"{CP}_barycenter_eta"]).to_numpy()
        cp_e = ak.flatten(a[f"{CP}_raw_energy"]).to_numpy()
        c_eta = ak.flatten(a["SimTICLCandidates_eta"]).to_numpy()
        c_e = ak.flatten(a["SimTICLCandidates_raw_energy"]).to_numpy()
        c_q = ak.flatten(a["SimTICLCandidates_charge"]).to_numpy()
        neu = c_q == 0
        print(f"  (3) events with n({CP}) == n(SimTICLCandidates): {same_n:.6f}")
        print(f"      index-wise |raw_energy diff| < 1e-3 rel: all {np.mean(np.isclose(cp_e, c_e, rtol=1e-3)):.6f}")
        print(f"      neutral candidates: |eta diff|<1e-3 {np.mean(np.abs(cp_eta[neu] - c_eta[neu]) < 1e-3):.6f} (n={neu.sum()})")
        print(f"      charged candidates: |eta diff|<0.05 {np.mean(np.abs(cp_eta[~neu] - c_eta[~neu]) < 0.05):.4f} (eta from track, not barycenter)")

        # (1) isPU distributions
        r["new"] = ak.flatten(a[f"{CP}_isPU"]).to_numpy()
        r["old"] = ak.flatten(a["SimTICLCandidates_isPU"]).to_numpy()
        r["e"] = cp_e
        r["abseta"] = np.abs(cp_eta)
        r["endcap"] = (r["abseta"] > 1.5) & (r["abseta"] < 3.0)
        for name in ("new", "old"):
            v, c = np.unique(r[name], return_counts=True)
            ew = [cp_e[r[name] == x].sum() / cp_e.sum() for x in v]
            print(f"  (1) {name:3s} isPU values {v.tolist()} frac {np.round(c / c.sum(), 5).tolist()} E-frac {np.round(ew, 5).tolist()}")
        print(f"      fromCPs |barycenter eta| in [1.5, 3.0]: {np.mean((r['abseta'] > 1.5) & (r['abseta'] < 3.0)):.4f}")

        # (2) correspondence with the input tracksters
        n_ts = a[f"n{TS}"].to_numpy()
        n_cp = a[f"n{CP}"].to_numpy()
        r["n_ts"], r["n_cp"] = n_ts, n_cp
        se_r = nested(a, f"Reco{TS}2SimCPByHits_nReco{TS}2SimCPByHitsLinks", f"Reco{TS}2SimCPByHitsLinks_sharedEnergy")
        sc_r = nested(a, f"Reco{TS}2SimCPByHits_nReco{TS}2SimCPByHitsLinks", f"Reco{TS}2SimCPByHitsLinks_score")
        se_c = nested(a, f"SimCP2{TS}ByHits_nSimCP2{TS}ByHitsLinks", f"SimCP2{TS}ByHitsLinks_sharedEnergy")
        ts_e = ak.flatten(a[f"{TS}_raw_energy"]).to_numpy()
        # per input trackster: number of CPs sharing energy, and the energy fraction held by the largest one
        nlink_ts = ak.flatten(ak.sum(se_r > 0, axis=2)).to_numpy()
        top_frac = ak.flatten(ak.fill_none(ak.max(se_r, axis=2), 0)).to_numpy() / np.maximum(ts_e, 1e-9)
        best_score = ak.flatten(ak.fill_none(ak.min(sc_r, axis=2), 1)).to_numpy()
        # per CP: number of input tracksters sharing energy
        nlink_cp = ak.flatten(ak.sum(se_c > 0, axis=2)).to_numpy()
        r.update(nlink_ts=nlink_ts, top_frac=top_frac, nlink_cp=nlink_cp, ts_e=ts_e)
        print(f"  (2) mean per event: n({TS}) {n_ts.mean():.1f}, n({CP}) {n_cp.mean():.1f}; events with equal counts {np.mean(n_ts == n_cp):.4f}")
        print(f"      per input trackster: n linked CPs  0:{np.mean(nlink_ts == 0):.4f} 1:{np.mean(nlink_ts == 1):.4f} >=2:{np.mean(nlink_ts >= 2):.4f}")
        print(f"      per input trackster: top CP holds >=90% of energy {np.mean(top_frac >= 0.9):.4f}; best score <0.2 {np.mean(best_score < 0.2):.4f}")
        print(f"      per CP: n linked input tracksters  0:{np.mean(nlink_cp == 0):.4f} 1:{np.mean(nlink_cp == 1):.4f} >=2:{np.mean(nlink_cp >= 2):.4f}")
        res[sample] = r

    # ---- plot 1: isPU distributions (count and energy weighted) ----
    bins = np.arange(-1.5, 2.0, 1.0)
    for weighted in (False, True):
        fig, axs = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
        for ax, name, title in ((axs[0], "new", f"{CP}_isPU"), (axs[1], "old", "SimTICLCandidates_isPU")):
            for sample, _ in SAMPLES:
                r = res[sample]
                w = r["e"] / r["e"].sum() if weighted else np.full(len(r[name]), 1 / len(r[name]))
                ax.hist(r[name], bins=bins, weights=w, histtype="step", lw=2, label=r["label"])
            lines = []
            for sample, _ in SAMPLES:
                r = res[sample]
                w = r["e"] if weighted else np.ones(len(r[name]))
                fr = [w[r[name] == v].sum() / w.sum() for v in (-1, 0, 1)]
                lines.append(f"{r['label']}: " + ", ".join(f"{v:+d}: {100 * f:.1f}%" for v, f in zip((-1, 0, 1), fr)))
            ax.text(0.03, 0.62, "\n".join(lines), transform=ax.transAxes, fontsize=15, va="top")
            ax.set_xticks([-1, 0, 1])
            ax.set_xlabel(title, fontsize=22)
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 3)
            mplhep.cms.label("Work in progress", data=False, rlabel="Phase-2 HGCAL", ax=ax, fontsize=20)
        axs[0].set_ylabel("Fraction of raw energy" if weighted else "Fraction of CaloParticles")
        axs[0].legend(loc="upper left", fontsize=18)
        fig.tight_layout()
        fn = f"{OUT}/ispu_fromcps_vs_candidates{'_energy' if weighted else ''}.pdf"
        fig.savefig(fn)
        plt.close(fig)
        print("saved", fn)

    # ---- plot 2: per-event counts, input tracksters vs fromCPs sim tracksters ----
    fig, axs = plt.subplots(1, 2, figsize=(20, 9))
    for ax, samples in ((axs[0], [s for s, _ in SAMPLES if s.endswith("_0pu")]), (axs[1], ["ttbar_200pu"])):
        hi = max(max(res[s]["n_cp"].max(), res[s]["n_ts"].max()) for s in samples) * 1.05
        for s in samples:
            ax.scatter(res[s]["n_cp"], res[s]["n_ts"], s=8, alpha=0.4, label=res[s]["label"])
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="1:1")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel(f"N {CP} per event", fontsize=20)
        ax.set_ylabel(f"N {TS} per event", fontsize=20)
        ax.legend(loc="upper left", fontsize=18, markerscale=3)
        mplhep.cms.label("Work in progress", data=False, rlabel="Phase-2 HGCAL", ax=ax, fontsize=20)
    fig.tight_layout()
    fig.savefig(f"{OUT}/counts_per_event.pdf")
    plt.close(fig)
    print("saved", f"{OUT}/counts_per_event.pdf")

    # ---- plot 3: associator link multiplicities ----
    fig, axs = plt.subplots(1, 3, figsize=(30, 9))
    mb = np.arange(-0.5, 10.5, 1)
    for sample, _ in SAMPLES:
        r = res[sample]
        axs[0].hist(np.minimum(r["nlink_ts"], 9), bins=mb, histtype="step", lw=2, density=True, label=r["label"])
        axs[1].hist(np.minimum(r["nlink_cp"], 9), bins=mb, histtype="step", lw=2, density=True, label=r["label"])
        axs[2].hist(np.clip(r["top_frac"], 0, 1), bins=np.linspace(0, 1, 41), histtype="step", lw=2, density=True, label=r["label"])
    axs[0].set_xlabel(f"N CPs sharing energy with one {TS}\n(9 = overflow)", fontsize=20)
    axs[1].set_xlabel(f"N {TS} sharing energy with one CP\n(9 = overflow)", fontsize=20)
    axs[2].set_xlabel(f"Energy fraction of a {TS}\nfrom its largest CP", fontsize=20)
    for ax in axs:
        ax.set_yscale("log")
        ax.set_ylabel("Normalized")
        mplhep.cms.label("Work in progress", data=False, rlabel="Phase-2 HGCAL", ax=ax, fontsize=20)
    axs[0].legend(loc="upper right", fontsize=18)
    fig.tight_layout()
    fig.savefig(f"{OUT}/association_multiplicity.pdf")
    plt.close(fig)
    print("saved", f"{OUT}/association_multiplicity.pdf")

    plot_ispu_correlation(res)
    plot_linked_vs_energy(res)


def plot_ispu_correlation(res):
    old_vals, new_vals = (-1, 0, 1), (0, 1)
    for weighted in (False, True):
        fig, axs = plt.subplots(1, len(SAMPLES), figsize=(9 * len(SAMPLES), 9))
        for ax, (sample, _) in zip(axs, SAMPLES):
            r = res[sample]
            m = r["endcap"]
            o, n = r["old"][m], r["new"][m]
            w = r["e"][m] if weighted else np.ones(m.sum())
            mat = np.array([[w[(o == ov) & (n == nv)].sum() for nv in new_vals] for ov in old_vals]) / w.sum()
            ax.imshow(mat, origin="lower", cmap="Blues", norm=matplotlib.colors.LogNorm(1e-4, 1), aspect="auto")
            for i, ov in enumerate(old_vals):
                for j, nv in enumerate(new_vals):
                    cnt = np.sum((o == ov) & (n == nv))
                    ax.text(j, i, f"{100 * mat[i, j]:.2f}%\n(N={cnt})", ha="center", va="center", fontsize=17, color="white" if mat[i, j] > 0.1 else "black")
            ax.set_xticks(range(len(new_vals)), [str(v) for v in new_vals])
            ax.set_yticks(range(len(old_vals)), [str(v) for v in old_vals])
            ax.minorticks_off()
            ax.set_xlabel(f"{CP}_isPU", fontsize=18)
            ax.set_ylabel("SimTICLCandidates_isPU", fontsize=18)
            ax.set_title(r["label"] + (" (raw-energy weighted)" if weighted else ""), fontsize=20)
        fig.tight_layout()
        fn = f"{OUT}/ispu_correlation{'_energy' if weighted else ''}.pdf"
        fig.savefig(fn)
        plt.close(fig)
        print("saved", fn)

    # fromCPs flag of the candidates with SimTICLCandidates_isPU == -1, against the rest
    fig, axs = plt.subplots(1, len(SAMPLES), figsize=(9 * len(SAMPLES), 9), sharey=True)
    for ax, (sample, _) in zip(axs, SAMPLES):
        r = res[sample]
        m = r["endcap"]
        for sel, lab, ls in ((r["old"] == -1, "SimTICLCandidates_isPU = -1", "-"), (r["old"] != -1, "SimTICLCandidates_isPU = 0/1", "--")):
            v = r["new"][m & sel]
            h = ax.hist(v, bins=[-0.5, 0.5, 1.5], weights=np.full(len(v), 1 / max(len(v), 1)), histtype="step", lw=2, ls=ls, label=f"{lab} (N={len(v)})")
            for x, y in zip((0, 1), h[0]):
                ax.text(x + (-0.2 if ls == "-" else 0.2), max(y, 2e-4) * 1.3, f"{100 * y:.2f}%", ha="center", fontsize=15)
        ax.set_xticks([0, 1])
        ax.minorticks_off()
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 10)
        ax.set_xlabel(f"{CP}_isPU", fontsize=18)
        ax.set_title(r["label"], fontsize=20)
        ax.legend(loc="upper left", fontsize=14)
    axs[0].set_ylabel("Fraction of candidates (endcap)")
    fig.tight_layout()
    fn = f"{OUT}/ispu_fromcps_for_candidate_minus1.pdf"
    fig.savefig(fn)
    plt.close(fig)
    print("saved", fn)


def plot_linked_vs_energy(res):
    fig, ax = plt.subplots(figsize=(12, 9))
    bins = np.logspace(-2, 3, 26)
    for sample, _ in SAMPLES:
        r = res[sample]
        e, linked = r["e"], r["nlink_cp"] > 0
        idx = np.digitize(e, bins) - 1
        ok = np.array([np.sum(idx == i) >= 20 for i in range(len(bins) - 1)])
        frac = np.array([linked[idx == i].mean() if c else np.nan for i, c in enumerate(ok)])
        ax.plot(np.sqrt(bins[1:] * bins[:-1]), frac, "o-", label=r["label"])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"{CP} raw energy [GeV]")
    ax.set_ylabel(f"Fraction sharing energy with a {TS}", fontsize=20)
    ax.legend(loc="lower right", fontsize=18)
    mplhep.cms.label("Work in progress", data=False, rlabel="Phase-2 HGCAL", ax=ax, fontsize=20)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cp_linked_vs_energy.pdf")
    plt.close(fig)
    print("saved", f"{OUT}/cp_linked_vs_energy.pdf")


if __name__ == "__main__":
    main()
