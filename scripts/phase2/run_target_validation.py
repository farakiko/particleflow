#!/usr/bin/env python3
"""Run moanwar's target-validation plots (mlpf_ticl_analysis_plots.py, used VERBATIM)
on any target pkls, so v1 (his script) and v2 (run3style) can be compared on his
metrics. Only the file selection, output dir, and a genmet backfill are added here
(v2 pkls don't store genmet; it is recomputed from pythia exactly as in
mlpf/heptfds/cms_pf_phase2/cms_phase2_utils.py).

  pixi run python3 scripts/phase2/run_target_validation.py \
    --glob 'data/.../pkl_v1_local/*.pkl' --outdir plots/phase2/target_validation/v1 [--max-files 50]
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH_CWD = os.getcwd()
os.chdir(HERE)  # moanwar's module resolves ../../mlpf relative to the cwd
sys.path.insert(0, HERE)
import mlpf_ticl_analysis_plots as M  # noqa: E402


def _from_launch(path):
    return path if os.path.isabs(path) else os.path.join(LAUNCH_CWD, path)


def align_ycand(ev):
    """His format has ycand per-element (aligned with Xelem). v2 pkls store the
    TICLCandidates as a separate list instead -> replace with an aligned all-zero
    recarray (v2 has no per-element PF assignment; 'PF' curves read as empty)."""
    X, c = ev["Xelem"], ev.get("ycand")
    if c is None or len(c) != len(X):
        fields = ["pid", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "ispu"]
        ev["ycand"] = np.zeros(len(X), dtype=[(f, np.float32) for f in fields]).view(np.recarray)
    return ev


def backfill_genmet(ev):
    if "genmet" not in ev:
        p = np.asarray(ev.get("pythia", np.zeros((0, 5))), np.float32).reshape(-1, 5)
        if len(p):
            met = float(np.hypot(np.sum(p[:, 1] * np.cos(p[:, 3])), np.sum(p[:, 1] * np.sin(p[:, 3]))))
        else:
            met = 0.0
        ev["genmet"] = np.array([met, 0.0], np.float32)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="pkl glob (quote it)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-files", type=int, default=50)
    a = ap.parse_args()

    files = sorted(glob.glob(_from_launch(a.glob)))[: a.max_files]
    print(f"{len(files)} files from {a.glob}")
    assert files, "no input files"

    data = M.load_files_parallel(files)
    data = [align_ycand(backfill_genmet(ev)) for ev in data]
    print(f"{len(data)} events")

    arrs_awk, arrs_flat, genmet_arr, genjet_cmssw = M.convert_events_vectorised(data)
    print(f"target particles (pid!=0): {int(M.ak.sum(M.ak.num(arrs_awk['ytarget']['pt'])))}")

    # v2 has an all-empty per-element ycand; clustering it yields untyped empty
    # awkward lists that vector rejects -> skip it and inject typed empty jets.
    skip_cand = int(M.ak.sum(M.ak.num(arrs_awk["ycand"]["pt"]))) == 0
    if skip_cand:
        ycand_saved = arrs_awk.pop("ycand")
    jets = M.cluster_all(arrs_awk, genjet_cmssw)
    if skip_cand:
        arrs_awk["ycand"] = ycand_saved
        e2d = M.ak.Array(np.empty((len(genmet_arr), 0), np.float32))
        jets["ycand"] = M.vector.awk(M.ak.zip({"pt": e2d, "eta": e2d, "phi": e2d, "energy": e2d}))

    outdir = _from_launch(a.outdir)
    os.makedirs(outdir, exist_ok=True)

    M.plot_pid_distributions_parallel(arrs_flat, outdir)
    M.plot_pu_fraction(arrs_flat, outdir)
    M.plot_pu_fraction_vs_pt(arrs_flat, outdir)
    M.plot_pu_fraction_vs_eta(arrs_flat, outdir)
    M.plot_met(arrs_awk, genmet_arr, outdir)
    M.plot_soft_pu_parallel(arrs_awk, outdir)
    M.plot_jet_distributions(jets, outdir)
    M.plot_overall_pt_distribution(arrs_awk, outdir)
    M.plot_sumpt_per_event(arrs_awk, outdir)
    M.plot_jet_response_single(jets, outdir)
    M.plot_jet_response_loglog(jets, outdir)
    M.plot_element_plots_parallel(arrs_flat, outdir)
    M.print_electron_elem_type_diagnostic(arrs_flat)
    M.plot_event_display(arrs_awk, jets, outdir, iev=2)
    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()
