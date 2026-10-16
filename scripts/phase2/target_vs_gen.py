#!/usr/bin/env python
"""
Phase-2 200PU -- target-level validation of the pileup-free target vs the generator.

Defines the "pileup-free target" in the 200PU sample as the MLPF target particles that carry
no pileup flag (per-element target ispu < 0.5, the same threshold used in training) and compares
it to the generator (pythia) particles. This checks that, once pileup is removed at truth level,
the target we train MLPF against reproduces the hard-scatter generator -- before any model is
involved. Reads the tfds directly (ytarget + pythia branches), using ALL configs / both splits
for maximum statistics (~10k ttbar 200PU events). Everything is restricted to the HGCAL endcap.

Produces, under <out>/target_vs_gen/ :
  particle_pt_spectrum.pdf   particle pT spectrum, Target (pileup-free) vs Gen (all types)
  jet_pt_spectrum.pdf        anti-kt R=0.4 jet pT spectrum, Target (pileup-free) vs Gen
  jet_pt_response.pdf        matched target-jet / gen-jet pT response (median + IQR)

Usage (inside the mlpf venv):
  python target_vs_gen.py --tfds-base /shared/mlpf-phase2/tfds/cms_pf_phase2_ttbar200pu \
                          --out /shared/mlpf-phase2/eval_plots/phase2_physics
"""
import os, argparse
import numpy as np
import awkward as ak
import fastjet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
import mplhep as hep
import tensorflow_datasets as tfds

hep.style.use("CMS")

# ----------------------------------------------------------------------------- configuration
RLABEL = r"$\mathrm{t}\bar{\mathrm{t}}$, 200 PU (14 TeV)"
ENDCAP = (1.48, 3.0)
PU_CUT = 0.5
NU = {12, 14, 16}
JET_PTCUT = 15.0
DR = 0.1
# ytarget column indices (mlpf ParticleFeatures ordering)
C_CLS, C_PT, C_ETA, C_SPHI, C_CPHI, C_E, C_ISPU = 0, 2, 3, 4, 5, 6, 7
TGT_C, GEN_C = "#d62728", "#444444"   # Target (pileup-free) / Gen colours
_jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)


def cms_label(ax, note=None):
    hep.cms.label(ax=ax, data=False, label="Work in Progress", rlabel=RLABEL, fontsize=16)
    if note:
        ax.text(0.5, 1.13, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=13, style="italic", color="0.30")


def endcap(eta):
    return (np.abs(eta) >= ENDCAP[0]) & (np.abs(eta) <= ENDCAP[1])


def load(tfds_base, configs):
    """Iterate the tfds configs/splits; return flat particle pT arrays and per-event jet inputs."""
    t_pt, g_pt = [], []                                   # flat particle pT (endcap)
    all_ispu, all_tpt = [], []                            # all endcap target particles: ispu, pT
    t_px, t_py, t_pz, t_e = [], [], [], []                # per-event target-pf jet inputs
    g_px, g_py, g_pz, g_e = [], [], [], []                # per-event gen jet inputs
    nev = 0
    for v in configs:
        bdir = f"{tfds_base}/{v}/1.0.0"
        if not os.path.isdir(bdir):
            continue
        b = tfds.builder_from_directory(bdir)
        for split in b.info.splits:
            ds = b.as_data_source(split=split)
            for i in range(len(ds)):
                e = ds[i]; yt = np.asarray(e["ytarget"]); py = np.asarray(e["pythia"])
                # all real target particles in the endcap (for the isPU-flag plot / diagnostics)
                am = (yt[:, C_CLS] != 0) & endcap(yt[:, C_ETA])
                all_ispu.append(yt[am, C_ISPU]); all_tpt.append(yt[am, C_PT])
                # pileup-free target: real particle, no pileup flag, endcap
                tm = (yt[:, C_CLS] != 0) & (yt[:, C_ISPU] < PU_CUT) & endcap(yt[:, C_ETA])
                tpt = yt[tm, C_PT]; teta = yt[tm, C_ETA]
                tphi = np.arctan2(yt[tm, C_SPHI], yt[tm, C_CPHI]); ten = yt[tm, C_E]
                # generator (pythia): visible (non-neutrino), endcap
                gm = (~np.isin(np.abs(py[:, 0]).astype(int), list(NU))) & (py[:, 1] > 0) & endcap(py[:, 2])
                gpt = py[gm, 1]; geta = py[gm, 2]; gphi = py[gm, 3]; gen = py[gm, 4]
                t_pt.append(tpt); g_pt.append(gpt)
                t_px.append(tpt * np.cos(tphi)); t_py.append(tpt * np.sin(tphi))
                t_pz.append(tpt * np.sinh(teta)); t_e.append(ten)
                g_px.append(gpt * np.cos(gphi)); g_py.append(gpt * np.sin(gphi))
                g_pz.append(gpt * np.sinh(geta)); g_e.append(gen)
                nev += 1
        print(f"  config {v}: cumulative {nev} events")
    jets_in = lambda px, py, pz, en: ak.zip(
        {"px": ak.Array(px), "py": ak.Array(py), "pz": ak.Array(pz), "E": ak.Array(en)})
    return dict(
        t_pt=np.concatenate(t_pt), g_pt=np.concatenate(g_pt),
        all_ispu=np.concatenate(all_ispu), all_tpt=np.concatenate(all_tpt),
        tgt_jets_in=jets_in(t_px, t_py, t_pz, t_e),
        gen_jets_in=jets_in(g_px, g_py, g_pz, g_e), nev=nev)


def cluster(jets_in):
    j = fastjet.ClusterSequence(jets_in, _jetdef).inclusive_jets(min_pt=JET_PTCUT)
    return j.px, j.py, j.pz


def endcap_jet_pt(px, py, pz):
    fx = ak.to_numpy(ak.flatten(px)); fy = ak.to_numpy(ak.flatten(py)); fz = ak.to_numpy(ak.flatten(pz))
    pt = np.hypot(fx, fy)
    return pt[endcap(np.arcsinh(fz / np.clip(pt, 1e-6, None)))]


def match_response(gpx, gpy, gpz, tpx, tpy, tpz):
    """Greedy dR match of target jets to gen jets (endcap); returns gen_pt, response."""
    og, resp = [], []
    for ev in range(len(gpx)):
        Gx = ak.to_numpy(gpx[ev]); Gy = ak.to_numpy(gpy[ev]); Gz = ak.to_numpy(gpz[ev])
        Tx = ak.to_numpy(tpx[ev]); Ty = ak.to_numpy(tpy[ev]); Tz = ak.to_numpy(tpz[ev])
        G = np.hypot(Gx, Gy); Ge = np.arcsinh(Gz / np.clip(G, 1e-6, None)); Gp = np.arctan2(Gy, Gx)
        T = np.hypot(Tx, Ty); Te = np.arcsinh(Tz / np.clip(T, 1e-6, None)); Tp = np.arctan2(Ty, Tx)
        inec = endcap(Ge); used = np.zeros(len(T), bool)
        for k in np.where(inec)[0]:
            if not len(T):
                break
            dphi = np.arctan2(np.sin(Tp - Gp[k]), np.cos(Tp - Gp[k]))
            d = np.hypot(Te - Ge[k], dphi); d[used] = 99
            j = int(np.argmin(d))
            if d[j] < DR:
                used[j] = True; og.append(G[k]); resp.append(T[j] / G[k])
    return np.array(og), np.array(resp)


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("  ->", os.path.relpath(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tfds-base", default="/shared/mlpf-phase2/tfds/cms_pf_phase2_ttbar200pu")
    ap.add_argument("--out", default="/shared/mlpf-phase2/eval_plots/phase2_physics")
    ap.add_argument("--configs", default="1,2,3,4,5,6,7,8,9,10")
    args = ap.parse_args()
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    outdir = f"{args.out}/target_vs_gen"
    os.makedirs(outdir, exist_ok=True)
    note = r"HGCAL endcap  (1.48 < $|\eta|$ < 3.0)"

    print("loading tfds ...")
    D = load(args.tfds_base, configs)
    print(f"loaded {D['nev']} events")

    # 1) particle pT spectrum
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.logspace(0, 2.5, 50)
    ax.hist(D["g_pt"], bins=b, histtype="step", lw=2.6, color=GEN_C, label="Gen")
    ax.hist(D["t_pt"], bins=b, histtype="step", lw=2.6, color=TGT_C, label="Target (pileup-free)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"particle $p_T$ [GeV]"); ax.set_ylabel("particles")
    ax.legend(fontsize=13, title=note, title_fontsize=11); cms_label(ax)
    _save(fig, f"{outdir}/particle_pt_spectrum.pdf")

    # cluster jets
    gpx, gpy, gpz = cluster(D["gen_jets_in"])
    tpx, tpy, tpz = cluster(D["tgt_jets_in"])

    # 2) jet pT spectrum
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.logspace(np.log10(JET_PTCUT), 2.8, 34)
    gjpt = endcap_jet_pt(gpx, gpy, gpz); tjpt = endcap_jet_pt(tpx, tpy, tpz)
    ax.hist(gjpt, bins=b, histtype="step", lw=2.6, color=GEN_C, label="Gen")
    ax.hist(tjpt, bins=b, histtype="step", lw=2.6, color=TGT_C, label="Target (pileup-free)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"jet $p_T$ [GeV]"); ax.set_ylabel("jets")
    ax.set_xticks([15, 20, 30, 50, 100, 200, 400])
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=13, title=note + "\n" + r"anti-$k_T$ R=0.4, $p_T>15$ GeV", title_fontsize=11)
    cms_label(ax)
    _save(fig, f"{outdir}/jet_pt_spectrum.pdf")

    # 3) jet pT response (target / gen) -- legend upper-left to avoid the curve
    og, resp = match_response(gpx, gpy, gpz, tpx, tpy, tpz)
    md = np.median(resp); iq = np.percentile(resp, 75) - np.percentile(resp, 25)
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.linspace(0, 2, 60)
    ax.hist(resp, bins=b, histtype="step", lw=2.6, color=TGT_C, label=f"M={md:.2f}, IQR={iq:.2f}")
    ax.axvline(1, color="k", ls=":")
    ax.set_xlabel(r"jet $p_T^{\mathrm{target}} / p_T^{\mathrm{gen}}$"); ax.set_ylabel("jets")
    ax.legend(fontsize=13, loc="upper left", title=note, title_fontsize=11); cms_label(ax)
    _save(fig, f"{outdir}/jet_pt_response.pdf")

    # 4) isPU flag distribution of the target particles (continuous pileup fraction in [0,1])
    ispu = D["all_ispu"]
    nsig = int((ispu < PU_CUT).sum()); npu = int((ispu >= PU_CUT).sum()); ntot = nsig + npu
    fig, ax = plt.subplots(figsize=(8, 7)); b = np.linspace(0, 1, 51)
    ax.hist(ispu, bins=b, histtype="stepfilled", lw=2.4, color=TGT_C, alpha=0.85,
            label="target particles (endcap)")
    ax.axvline(PU_CUT, color="k", ls=":", lw=1.8)
    ax.text(PU_CUT + 0.02, 0.92, "pileup-free cut", transform=ax.get_xaxis_transform(),
            fontsize=12, style="italic", va="top")
    ax.set_yscale("log"); ax.set_xlim(0, 1)
    ax.set_xlabel("target particle isPU (pileup fraction)"); ax.set_ylabel("target particles")
    ax.legend(fontsize=12, title=note, title_fontsize=11); cms_label(ax)
    _save(fig, f"{outdir}/ispu_flag.pdf")

    print(f"\nparticle pT median: gen {np.median(D['g_pt']):.2f}  target {np.median(D['t_pt']):.2f}")
    print(f"jet response: median {md:.3f}  IQR {iq:.3f}  on {len(resp)} matched jets")
    print(f"isPU flag: hard-scatter {nsig} ({100*nsig/ntot:.1f}%)  pileup {npu} ({100*npu/ntot:.1f}%)")
    # diagnostic for high-pT non-closure: how much of the high-pT target is flagged pileup?
    for cut in (20, 50):
        hi = D["all_tpt"] >= cut
        if hi.sum():
            print(f"  target particles pT>={cut}: {int(hi.sum())}, flagged pileup "
                  f"{100*(ispu[hi] >= PU_CUT).mean():.1f}%")
    print("DONE ->", outdir)


if __name__ == "__main__":
    main()
