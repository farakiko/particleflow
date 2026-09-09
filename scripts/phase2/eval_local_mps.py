#!/usr/bin/env python
"""Evaluate a trained MLPF checkpoint on the val split and overlay THREE reco curves
on the target-validation metrics, all w.r.t. gen:
    target   (Run3-style truth target, orange)
    baseline (TICLCandidates, the CMSSW reco baseline, blue)
    MLPF     (this model's prediction, red)
gen = black (reference / denominator).

The baseline (candjet + ycand) is not surfaced by the training adapter (it stores
zeros for ycand), so we read it straight from the raw pkls, aligned to the adapter's
event order. Plots: jet response, pT spectra, efficiency, fake rate, sum-pT per class."""
import os, sys, glob, argparse, random, pickle
import numpy as np, torch, fastjet
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "mlpf/heptfds/cms_pf_phase2"))
from mlpf.conf import MLPFConfig, Dataset, X_FEATURES, CLASS_LABELS, ELEM_TYPES_NONZERO
from mlpf.model.mlpf import MLPF
from mlpf.model.utils import unpack_predictions
import cms_phase2_utils as adapter

NANO = os.path.join(REPO, "data/cms/phase2/offline/Aug31/nano")
CLS = CLASS_LABELS[Dataset.CMS_PHASE2.value]          # [0,211,130,22,11,13]
NHAD = {130, 310, 2112, 3122, 3322, 3212, 421, 511}
CLASSES = ["photon", "nhad", "chad", "electron", "muon"]
CLABEL = {"photon": "photons", "nhad": "neutral hadrons", "chad": "charged hadrons",
          "electron": "electrons", "muon": "muons"}
# reco curves overlaid on every plot (name -> color, legend label)
RECO = [("target", "tab:orange", "target"),
        ("baseline", "tab:blue", "baseline (TICL)"),
        ("mlpf", "red", "MLPF")]
JETDEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
ELO, EHI, DR, GENPT = 1.5, 3.0, 0.2, 20.0
PTED = np.array([1, 2, 3, 5, 8, 13, 20, 35, 60, 100, 200.]); PTC = np.sqrt(PTED[:-1]*PTED[1:]); NB = len(PTC)


def pdg_cls(apid):
    a = int(abs(apid))
    return "photon" if a == 22 else "electron" if a == 11 else "muon" if a == 13 else ("nhad" if a in NHAD else "chad")


def build_config(nc, nh, hd):
    ds = Dataset.CMS_PHASE2
    return MLPFConfig(dataset=ds, data_dir="/tmp/x", conv_type="attention",
                      input_dim=len(X_FEATURES[ds.value]), num_classes=len(CLASS_LABELS[ds.value]),
                      elemtypes_nonzero=ELEM_TYPES_NONZERO[ds.value],
                      model={"type": "attention", "input_encoding": "split",
                             "attention": {"num_convs": nc, "head_dim": hd, "num_heads": nh, "attention_type": "math"}})


def cluster(pt, eta, phi, e):
    ok = pt > 0; pt, eta, phi, e = pt[ok], eta[ok], phi[ok], e[ok]
    if len(pt) == 0: return np.zeros((0, 4))
    px, py = pt*np.cos(phi), pt*np.sin(phi); pz = pt*np.sinh(np.where(np.abs(eta) < 10, eta, 0))
    pjs = [fastjet.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(e[i])) for i in range(len(pt))]
    j = fastjet.ClusterSequence(pjs, JETDEF).inclusive_jets(ptmin=3.0)
    return np.array([[x.pt(), x.eta(), x.phi(), x.e()] for x in j]) if j else np.zeros((0, 4))


def load_val(pkl_subdir="pkl_run3style"):
    """Replicate training's val split: same order + shuffle(seed 0), first 10%.
    Also pull the TICL baseline (candjet + ycand) straight from the raw pkl, which
    the adapter does not expose. raw[i] aligns with the adapter's i-th event."""
    evs = []
    for s in ["ttbar_0pu", "qcd_0pu", "zll_0pu"]:
        for f in sorted(glob.glob(f"{NANO}/{s}/{pkl_subdir}/*.pkl")):
            Xs, ytg, yc, gm, gj, tj, yp = adapter.prepare_data_phase2(f)
            raw = pickle.load(open(f, "rb"))
            for i in range(len(Xs)):
                ev = raw[i]
                evs.append({"X": Xs[i].astype(np.float32), "yt": ytg[i], "gj": gj[i], "tj": tj[i],
                            "yp": yp[i], "cj": np.asarray(ev["candjet"], np.float32).reshape(-1, 4),
                            "yc": ev["ycand"]})
    random.Random(0).shuffle(evs)
    return evs[:max(1, len(evs)//10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "experiments/phase2_local/checkpoints/best.pth"))
    ap.add_argument("--outdir", default=os.path.join(REPO, "plots/phase2/mlpf_eval"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--pkl-subdir", default="pkl_run3style", help="per-sample pkl subdir (e.g. pkl_clue3d / pkl_links)")
    a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu")
    c = ck.get("config", {"num_convs": 3, "num_heads": 8, "head_dim": 16})
    model = MLPF(build_config(c["num_convs"], c["num_heads"], c["head_dim"]))
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    model.eval().to(a.device)
    print(f"loaded {a.ckpt} (epoch {ck.get('epoch','?')}, val {ck.get('val_loss','?')})")

    val = load_val(a.pkl_subdir); print(f"val events: {len(val)}")
    NAMES = [r[0] for r in RECO]

    # accumulators
    resp = {k: [] for k in NAMES}
    jets_all = {"gen": []} | {k: [] for k in NAMES}                          # (pt,eta) for spectra
    eff = {k: {c: np.zeros((NB, 2)) for c in CLASSES} for k in NAMES}        # [tot, matched_same]
    fake = {k: {c: np.zeros((NB, 2)) for c in CLASSES} for k in NAMES}       # [tot, fake_same]
    eff_any = {k: {c: np.zeros((NB, 2)) for c in CLASSES} for k in NAMES}    # [tot, matched_any-pid]
    fake_any = {k: {c: np.zeros((NB, 2)) for c in CLASSES} for k in NAMES}   # [tot, fake_any-pid]
    sump = {k: {c: 0.0 for c in CLASSES} for k in NAMES + ["gen"]}
    conf = np.zeros((6, 6), np.int64)   # per-element target class (row) vs MLPF predicted class (col)
    sumpt_pt = {k: {c: np.zeros(NB) for c in CLASSES} for k in NAMES + ["gen"]}   # ΣpT in bins of particle pT
    resp_part = {k: {c: {"any": [], "same": []} for c in CLASSES} for k in NAMES}  # per-particle pT/genpT

    B = 16
    with torch.no_grad():
        for bi in range(0, len(val), B):
            evb = val[bi:bi+B]
            N = max(len(e["X"]) for e in evb); nf = evb[0]["X"].shape[1]
            X = np.zeros((len(evb), N, nf), np.float32); mask = np.zeros((len(evb), N), bool)
            for i, e in enumerate(evb):
                X[i, :len(e["X"])] = e["X"]; mask[i, :len(e["X"])] = True
            Xt = torch.tensor(X, device=a.device); mk = torch.tensor(mask, device=a.device)
            pred = unpack_predictions(model(Xt, mk))
            cls_id = pred["cls_id"].cpu().numpy(); mom = pred["momentum"].cpu().numpy()
            for i, e in enumerate(evb):
                n = len(e["X"]); xe = e["X"]
                cid = cls_id[i, :n]; m = mom[i, :n]; sel = cid != 0
                np.add.at(conf, (e["yt"][:, 0].astype(int), cid), 1)   # target vs predicted class, per element
                # ---- MLPF predicted particles ----
                p_pt = np.exp(m[sel, 0]) * xe[sel, 1]; p_eta = m[sel, 1]
                p_phi = np.arctan2(m[sel, 2], m[sel, 3]); p_e = np.exp(m[sel, 4]) * xe[sel, 5]
                p_pid = np.array([CLS[cc] for cc in cid[sel]])
                # ---- TARGET particles (from pkl ytarget) ----
                yt = e["yt"]; tsel = yt[:, 0] != 0
                t_pt = yt[tsel, 2]; t_eta = yt[tsel, 3]; t_phi = np.arctan2(yt[tsel, 4], yt[tsel, 5]); t_e = yt[tsel, 6]
                t_pid = np.array([CLS[int(k)] for k in yt[tsel, 0]])
                # ---- BASELINE particles (TICLCandidates from pkl ycand) ----
                yc = e["yc"]
                if len(yc):
                    c_pt = np.asarray(yc["pt"], np.float32); okc = c_pt > 0
                    c_pt = c_pt[okc]; c_eta = np.asarray(yc["eta"], np.float32)[okc]
                    c_phi = np.asarray(yc["phi"], np.float32)[okc]; c_pid = np.asarray(yc["pid"], np.float32)[okc]
                else:
                    c_pt = c_eta = c_phi = c_pid = np.zeros(0, np.float32)
                # ---- jets ----
                gj = e["gj"].reshape(-1, 4); tj = e["tj"].reshape(-1, 4); cjb = e["cj"].reshape(-1, 4)
                mj = cluster(p_pt, p_eta, p_phi, p_e)
                pj = {"target": tj, "baseline": cjb, "mlpf": mj}
                jets_all["gen"].append(gj[:, :2] if len(gj) else np.zeros((0, 2)))
                for name in NAMES:
                    jj = pj[name]; jets_all[name].append(jj[:, :2] if len(jj) else np.zeros((0, 2)))
                # ---- jet response vs gen ----
                for name in NAMES:
                    jets = pj[name]
                    for g in gj:
                        if g[0] < GENPT or abs(g[1]) < ELO or abs(g[1]) > EHI or len(jets) == 0: continue
                        dphi = np.arctan2(np.sin(jets[:, 2]-g[2]), np.cos(jets[:, 2]-g[2]))
                        b = int(np.argmin(np.hypot(jets[:, 1]-g[1], dphi)))
                        if np.hypot(jets[b, 1]-g[1], dphi[b]) < DR: resp[name].append(jets[b, 0]/g[0])
                # ---- gen particles (pythia), endcap only ----
                yp = e["yp"].reshape(-1, 5); gm2 = (np.abs(yp[:, 2]) > ELO) & (np.abs(yp[:, 2]) < EHI)
                g_pid = yp[gm2, 0]; g_pt = yp[gm2, 1]; g_eta = yp[gm2, 2]; g_phi = yp[gm2, 3]
                g_cls = np.array([pdg_cls(x) for x in g_pid])
                # ---- per-class eff / fake / sumpt for each reco ----
                for name, (rp, re, rf, rpid) in [
                    ("target",   (t_pt, t_eta, t_phi, t_pid)),
                    ("baseline", (c_pt, c_eta, c_phi, c_pid)),
                    ("mlpf",     (p_pt, p_eta, p_phi, p_pid))]:
                    rc = np.array([pdg_cls(x) for x in rpid])
                    rmask = (np.abs(re) > ELO) & (np.abs(re) < EHI)
                    re2, rp2, rf2, rc2 = re[rmask], rp[rmask], rf[rmask], rc[rmask]
                    for c in CLASSES: sump[name][c] += rp2[rc2 == c].sum()
                    # efficiency: gen -> reco within DR (same-class AND any-class)
                    for gi in range(len(g_pt)):
                        cc = g_cls[gi]; bb = int(np.clip(np.digitize(g_pt[gi], PTED)-1, 0, NB-1))
                        eff[name][cc][bb, 0] += 1; eff_any[name][cc][bb, 0] += 1
                        same = rc2 == cc
                        if same.any():
                            dphi = np.arctan2(np.sin(rf2[same]-g_phi[gi]), np.cos(rf2[same]-g_phi[gi]))
                            if np.min(np.hypot(re2[same]-g_eta[gi], dphi)) < DR: eff[name][cc][bb, 1] += 1
                        if len(re2):     # any-pid: match to nearest reco regardless of class
                            dpha = np.arctan2(np.sin(rf2-g_phi[gi]), np.cos(rf2-g_phi[gi]))
                            if np.min(np.hypot(re2-g_eta[gi], dpha)) < DR: eff_any[name][cc][bb, 1] += 1
                    # fake: reco with no gen within DR (same-class AND any-class)
                    for ri in range(len(rp2)):
                        cc = rc2[ri]; bb = int(np.clip(np.digitize(rp2[ri], PTED)-1, 0, NB-1))
                        fake[name][cc][bb, 0] += 1; fake_any[name][cc][bb, 0] += 1
                        same = g_cls == cc; ok = False
                        if same.any():
                            dphi = np.arctan2(np.sin(g_phi[same]-rf2[ri]), np.cos(g_phi[same]-rf2[ri]))
                            ok = np.min(np.hypot(g_eta[same]-re2[ri], dphi)) < DR
                        if not ok: fake[name][cc][bb, 1] += 1
                        oka = False      # any-pid: is there ANY gen particle nearby?
                        if len(g_pt):
                            dpha = np.arctan2(np.sin(g_phi-rf2[ri]), np.cos(g_phi-rf2[ri]))
                            oka = np.min(np.hypot(g_eta-re2[ri], dpha)) < DR
                        if not oka: fake_any[name][cc][bb, 1] += 1
                    # (3a) ΣpT in bins of the particle's own pT
                    for c in CLASSES:
                        sel = rc2 == c
                        if sel.any():
                            bb = np.clip(np.digitize(rp2[sel], PTED) - 1, 0, NB - 1)
                            np.add.at(sumpt_pt[name][c], bb, rp2[sel])
                    # (3b) per-particle pT response: reco -> nearest gen within DR (any-pid & same-pid)
                    for ri in range(len(rp2)):
                        cc = rc2[ri]
                        if len(g_pt):
                            dR = np.hypot(g_eta - re2[ri], np.arctan2(np.sin(g_phi - rf2[ri]), np.cos(g_phi - rf2[ri])))
                            j = int(np.argmin(dR))
                            if dR[j] < DR and g_pt[j] > 0: resp_part[name][cc]["any"].append(rp2[ri] / g_pt[j])
                        sm = g_cls == cc
                        if sm.any():
                            gp, ge, gf = g_pt[sm], g_eta[sm], g_phi[sm]
                            dR = np.hypot(ge - re2[ri], np.arctan2(np.sin(gf - rf2[ri]), np.cos(gf - rf2[ri])))
                            j = int(np.argmin(dR))
                            if dR[j] < DR and gp[j] > 0: resp_part[name][cc]["same"].append(rp2[ri] / gp[j])
                for c in CLASSES:
                    sump["gen"][c] += g_pt[g_cls == c].sum()
                    sel = g_cls == c
                    if sel.any():
                        bb = np.clip(np.digitize(g_pt[sel], PTED) - 1, 0, NB - 1)
                        np.add.at(sumpt_pt["gen"][c], bb, g_pt[sel])

    for k in jets_all: jets_all[k] = np.concatenate(jets_all[k]) if jets_all[k] else np.zeros((0, 2))

    # ---- Fig 1: jet response ----
    fig, ax = plt.subplots(figsize=(7.6, 5.6)); b = np.logspace(-1, 1, 120)
    for name, col, lab in RECO:
        r = np.array(resp[name])
        if len(r): ax.hist(r, bins=b, histtype="step", lw=2, color=col, label=f"{lab} (med {np.median(r):.2f})")
    ax.axvline(1, color="k", ls="--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_\mathrm{T}$ / genjet $p_\mathrm{T}$"); ax.set_ylabel("Jets")
    ax.set_title(r"Jet response vs gen ($1.5<|\eta|<3$, gen $p_\mathrm{T}>20$)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "mlpf_jet_response.pdf")); plt.close(fig)

    # ---- Fig 2: jet pT spectra ----
    fig, ax = plt.subplots(figsize=(7.6, 5.6)); bb = np.logspace(np.log10(3), np.log10(800), 50)
    for arr, col, lab in [(jets_all["gen"], "black", "gen (status 1, no ν)")] + \
                          [(jets_all[n], col, lab) for n, col, lab in RECO]:
        mm = np.abs(arr[:, 1]) > ELO
        ax.hist(arr[mm, 0], bins=bb, histtype="step", lw=2, color=col, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_\mathrm{T}$ [GeV]"); ax.set_ylabel("Jets")
    ax.set_title(r"Jet $p_\mathrm{T}$ spectra ($|\eta|>1.5$)"); ax.legend(fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "mlpf_pt_spectra.pdf")); plt.close(fig)

    # ---- Fig 3/4: efficiency + fake (per class) ----
    def grid(metric, title, fn, match):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); axs = axs.ravel()
        for k, c in enumerate(CLASSES):
            ax = axs[k]
            for name, col, lab in RECO:
                d = metric[name][c]
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = np.where(d[:, 0] > 0, d[:, 1]/d[:, 0], np.nan)
                ax.plot(PTC, y, "o-", color=col, lw=2, label=lab)
            ax.set_xscale("log"); ax.set_ylim(-.03, 1.03); ax.set_title(CLABEL[c])
            ax.set_xlabel(r"$p_\mathrm{T}$ [GeV]"); ax.grid(alpha=.3); ax.legend(fontsize=9)
        axs[0].set_ylabel(title); axs[3].set_ylabel(title); axs[5].axis("off")
        fig.suptitle(title + r" vs pT (" + match + r", $\Delta R<0.2$, $1.5<|\eta|<3$)", fontsize=14)
        plt.tight_layout(); plt.savefig(os.path.join(a.outdir, fn)); plt.close(fig)
    grid(eff, "Efficiency", "mlpf_efficiency_vs_pt.pdf", "same-class")
    grid(fake, "Fake rate", "mlpf_fakerate_vs_pt.pdf", "same-class")
    grid(eff_any, "Efficiency", "mlpf_efficiency_vs_pt_anypid.pdf", "any-class")
    grid(fake_any, "Fake rate", "mlpf_fakerate_vs_pt_anypid.pdf", "any-class")

    # ---- Fig 5: sum-pt per class ----
    fig, ax = plt.subplots(figsize=(9.5, 5.2)); x = np.arange(len(CLASSES)); w = 0.27
    for j, (name, col, lab) in enumerate(RECO):
        off = (j - 1) * w
        ax.bar(x + off, [sump[name][c]/max(sump["gen"][c], 1e-9) for c in CLASSES], w, color=col, label=f"{lab}/gen")
    ax.axhline(1, color="k", ls="--", lw=1); ax.set_xticks(x); ax.set_xticklabels([CLABEL[c] for c in CLASSES], rotation=20)
    ax.set_ylabel(r"$\Sigma p_\mathrm{T}$ ratio to gen"); ax.set_title(r"$\Sigma p_\mathrm{T}$ per class ($1.5<|\eta|<3$)"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "mlpf_sumpt_per_class.pdf")); plt.close(fig)

    # ---- Fig 6: confusion matrix (per-element target class vs MLPF predicted class) ----
    LBL = ["null", "chad", "nhad", r"$\gamma$", "e", r"$\mu$"]
    rown = conf / np.clip(conf.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    im = ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6)); ax.set_xticklabels(LBL); ax.set_yticks(range(6)); ax.set_yticklabels(LBL)
    ax.set_xlabel("MLPF predicted class"); ax.set_ylabel("target class")
    ax.set_title("Confusion matrix (per element, row-normalized)")
    for r in range(6):
        for cc in range(6):
            ax.text(cc, r, f"{100*rown[r, cc]:.0f}%\n{int(conf[r, cc])}", ha="center", va="center",
                    fontsize=8, color="white" if rown[r, cc] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row fraction")
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "mlpf_confusion.pdf")); plt.close(fig)

    # ---- Fig 6b: classification ceiling (found-but-mislabeled), MLPF ----
    # per class: any-pid eff = spatial reconstruction ceiling; same-pid eff = correctly classified.
    ceil = [eff_any["mlpf"][c][:, 1].sum()/max(eff_any["mlpf"][c][:, 0].sum(), 1) for c in CLASSES]
    got  = [eff["mlpf"][c][:, 1].sum()/max(eff["mlpf"][c][:, 0].sum(), 1) for c in CLASSES]
    cause = {"photon": "→γ ok", "nhad": "→γ 47%", "chad": "—", "electron": "→chad 71%", "muon": "→chad 67%"}
    fig, ax = plt.subplots(figsize=(9.5, 6)); x = np.arange(len(CLASSES))
    ax.bar(x, ceil, 0.62, color="#d0d0d0", label="reconstruction ceiling (spatially found, any class)")
    ax.bar(x, got, 0.62, color="tab:red", label="MLPF (found AND correctly classified)")
    for i, c in enumerate(CLASSES):
        gap = ceil[i] - got[i]
        if gap > 0.02:
            ax.annotate("", xy=(i, ceil[i]), xytext=(i, got[i]),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
            ax.text(i + 0.34, (ceil[i]+got[i])/2, f"-{gap:.0%}\nmis-ID", va="center", fontsize=9, color="black")
        ax.text(i, ceil[i] + 0.02, cause[c], ha="center", fontsize=8, color="dimgray")
    ax.set_xticks(x); ax.set_xticklabels([CLABEL[c] for c in CLASSES]); ax.set_ylim(0, 1.12)
    ax.set_ylabel(r"efficiency (gen $\to$ MLPF, $\Delta R<0.2$, $1.5<|\eta|<3$)")
    ax.set_title("Why the rare classes underperform: found spatially, lost to mis-classification\n"
                 "(gap = classification ceiling the current inputs can't reach — no muon-ID / depth features)",
                 fontsize=11)
    ax.legend(loc="lower center"); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "mlpf_classification_ceiling.pdf")); plt.close(fig)

    # ---- Fig 7: ΣpT vs pT, one figure per class (gen/target/baseline/mlpf + ratio) ----
    COL = {"gen": "black", "target": "tab:orange", "baseline": "tab:blue", "mlpf": "red"}
    LAB = {"gen": "gen", "target": "target", "baseline": "baseline (TICL)", "mlpf": "MLPF"}
    for c in CLASSES:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for k in ["gen"] + NAMES:
            a1.plot(PTC, sumpt_pt[k][c], "o-", color=COL[k], lw=2, label=LAB[k])
        a1.set_xscale("log"); a1.set_yscale("log"); a1.set_ylabel(r"$\Sigma p_\mathrm{T}$ [GeV]")
        a1.set_title(f"{CLABEL[c]}: $\\Sigma p_T$ vs $p_T$ ($1.5<|\\eta|<3$)"); a1.legend(); a1.grid(alpha=.3)
        gv = sumpt_pt["gen"][c]
        for k in NAMES:
            with np.errstate(divide="ignore", invalid="ignore"):
                a2.plot(PTC, np.where(gv > 0, sumpt_pt[k][c]/gv, np.nan), "o-", color=COL[k], lw=2)
        a2.axhline(1, color="k", ls="--", lw=1); a2.set_ylim(0, 2); a2.set_xscale("log")
        a2.set_ylabel("reco / gen"); a2.set_xlabel(r"particle $p_\mathrm{T}$ [GeV]"); a2.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(a.outdir, f"mlpf_sumpt_vs_pt_{c}.pdf")); plt.close(fig)

    # ---- Fig 8: per-particle pT response vs gen (ΔR<0.2), any-pid and same-pid ----
    def resp_grid(mode, fn):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.5)); axs = axs.ravel(); bb = np.logspace(-1, 1, 80)
        for k, c in enumerate(CLASSES):
            ax = axs[k]
            for name, col, lab in RECO:
                r = np.array(resp_part[name][c][mode])
                if len(r):
                    ax.hist(r, bins=bb, histtype="step", lw=2, color=col, density=True,
                            label=f"{lab} (med {np.median(r):.2f}, N{len(r)})")
            ax.axvline(1, color="k", ls="--", lw=1); ax.set_xscale("log"); ax.set_title(CLABEL[c])
            ax.set_xlabel(r"reco $p_\mathrm{T}$ / gen $p_\mathrm{T}$"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        axs[5].axis("off")
        fig.suptitle(f"Per-particle pT response vs gen (ΔR<0.2, {mode}-pid match, $1.5<|\\eta|<3$)", fontsize=14)
        plt.tight_layout(); plt.savefig(os.path.join(a.outdir, fn)); plt.close(fig)
    resp_grid("any", "mlpf_particle_ptresponse_anypid.pdf")
    resp_grid("same", "mlpf_particle_ptresponse_samepid.pdf")

    # ---- summary ----
    print("jet response median: " + "  ".join(f"{n}={np.median(resp[n]):.3f}" for n in NAMES if len(resp[n])))
    print(f"{'class':10s}" + "".join(f"{lab:>22}" for _, _, lab in RECO))
    for c in CLASSES:
        row = f"{c:10s}"
        for name, _, _ in RECO:
            e_ = eff[name][c][:, 1].sum()/max(eff[name][c][:, 0].sum(), 1)
            ea = eff_any[name][c][:, 1].sum()/max(eff_any[name][c][:, 0].sum(), 1)
            s_ = sump[name][c]/max(sump["gen"][c], 1e-9)
            row += f"   eff={e_:.2f}(any {ea:.2f}) ΣpT={s_:.2f}"
        print(row)
    print("\nconfusion (row=target, col=MLPF pred, row-normalized %):")
    print(f"{'':8s}" + "".join(f"{l:>8}" for l in LBL))
    for r in range(6):
        print(f"{LBL[r]:8s}" + "".join(f"{100*rown[r, cc]:>7.0f}%" for cc in range(6)))
    print("\nper-particle pT response median [IQR/med]  (reco/gen, ΔR<0.2):")
    print(f"{'class':10s}" + "".join(f"{lab+' any/same':>26}" for _, _, lab in RECO))
    for c in CLASSES:
        row = f"{c:10s}"
        for name, _, _ in RECO:
            cell = ""
            for mode in ("any", "same"):
                r = np.array(resp_part[name][c][mode])
                if len(r):
                    md = np.median(r); iqr = (np.percentile(r, 75)-np.percentile(r, 25))/md if md else 0
                    cell += f"{md:.2f}[{iqr:.2f}] "
                else:
                    cell += "  -   "
            row += f"{cell:>26}"
        print(row)
    print("plots ->", a.outdir)


if __name__ == "__main__":
    main()
