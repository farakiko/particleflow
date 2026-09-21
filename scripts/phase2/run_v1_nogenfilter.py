#!/usr/bin/env python3
"""Run moanwar's postprocessing (postprocessing_ticl_ttbar_nopu.py, UNMODIFIED) with
the gen-matching turned off — the surgical isolation of the gen filter.

The only change, injected at runtime: `build_gen_to_simcand_map` is replaced by an
ACCEPTANCE-ONLY version that reproduces his in-acceptance logic exactly (matched
position |eta|>=1.5 via track-HGCAL extrapolation for charged, simcand eta otherwise;
simcand pt >= SIMCAND_PT_MIN) but skips the gen deltaR matching:
  - matched_simcand_indices = ALL in-acceptance simcands  (target side: no gen gating)
  - matched_gen_indices     = ALL gens                    (reference side: unfiltered)
Everything else — score cuts, fragmentation, element selection, kinematics — is his
code verbatim, so (filter ON) vs (this) differs ONLY by the gen matching.

  pixi run python3 scripts/phase2/run_v1_nogenfilter.py --input X.root --output Y.pkl
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import postprocessing_ticl_ttbar_nopu as V1  # noqa: E402


def acceptance_only(ev):
    simcan_eta = ev["simcan_eta"]
    simcan_pid = ev["simcan_pdgid"]
    simcan_trkId = ev["simcan_trkId"]
    track_hgcal_eta = ev["track_hgcal_eta"]
    n_tracks = len(track_hgcal_eta)
    n_sc = len(simcan_eta)

    sc_match_eta = np.array([float(simcan_eta[j]) for j in range(n_sc)])
    for j in range(n_sc):
        if abs(int(simcan_pid[j])) not in V1._CHARGED_PIDS:
            continue
        trk_ids = simcan_trkId[j] if j < len(simcan_trkId) else []
        for tid in trk_ids:
            tid = int(tid)
            if 0 <= tid < n_tracks and abs(float(track_hgcal_eta[tid])) > 1.5:
                sc_match_eta[j] = float(track_hgcal_eta[tid])
                break

    ok = (np.abs(sc_match_eta) >= 1.5) & (np.asarray(ev["simcan_pt"]) >= V1.SIMCAND_PT_MIN)
    matched_simcands = set(np.where(ok)[0].tolist())
    matched_gens = set(range(len(ev["genpar_pdgid"])))
    return matched_simcands, matched_gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-events", type=int, default=-1)
    a = ap.parse_args()

    V1.build_gen_to_simcand_map = acceptance_only
    # call the per-file worker IN-PROCESS (his main uses a ProcessPoolExecutor whose
    # spawned children re-import the module and would lose the patch)
    import pickle
    data = V1.process_file_no_progress(a.input, a.num_events, 0, use_superclustering=False)
    with open(a.output, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    npart = sum(int((ev["ytarget"]["pid"] != 0).sum()) for ev in data)
    print(f"saved {a.output}: {len(data)} events, {npart} target particles (gen filter OFF)")


if __name__ == "__main__":
    main()
