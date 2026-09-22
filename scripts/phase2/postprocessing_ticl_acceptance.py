import uproot
import numpy as np
import networkx as nx
import argparse
import os
import math
import pickle
from tqdm import tqdm
from collections import defaultdict
import fastjet
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

"""
   python3 postprocessing_ticl_ttbar_nopu_nanoaod.py --input NanoAOD.root --output tmp/test_nano.pkl --num-events 100 --max-files 5

"""

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

elem_branches = [
    "typ", "pt", "eta", "phi", "energy", "charge", "px", "py", "pz",
    "em_energy", "bary_z", "nhits",
    "min_dR_track", "near_track_pt", "shower_depth", "sum_pt_dR10",
    "n_trk_dR01", "n_trk_dR02", "n_trk_dR03", "n_trk_dR04", "n_trk_dR05",
    # trackster features
    "n_clusters", "ts_time", "ts_time_err",
    # track features
    "track_muon_type", "track_muon_dt_hits", "track_muon_csc_hits",
    "track_gsf_type",
    "track_pt_err", "track_eta_err", "track_phi_err",
    "track_lambda_err", "track_qoverp_err",
    "track_vx", "track_vy", "track_vz",
]

particle_feature_order = [
    "pid", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy",
    "ispu", "simulatorStatus", "cp_to_track", "cp_to_cluster", "jet_idx"
]

_NEUTRAL_PIDS  = frozenset([130, 22, 310])
_CHARGED_PIDS  = frozenset([11, 13, 211, 321])
_NEUTRINO_PIDS = frozenset([12, 14, 16])

# -- particle categories for matching ------------------------------------------------
_CAT_EM    = frozenset([11, 22])          # electrons, photons
_CAT_CHAD  = frozenset([211, 321])        # charged hadrons
_CAT_NHAD  = frozenset([130, 310])        # neutral hadrons
_CAT_MU    = frozenset([13])              # muons

# -- tunable thresholds ---------------------------------------------------------------
GEN_PT_MIN     = 1.0  # min gen pT (GeV)
SIMCAND_PT_MIN = 1.0  # min SimTICLCandidate pT (GeV) -- applied directly to the simCand's
                       # own pt, independent of the matching gen particle's pT
DR_MAX_EM    = 0.1    # EM:            electrons, photons
DR_MAX_CHAD  = 0.1    # charged hadron: pi+/-, K+/-
DR_MAX_NHAD  = 0.1    # neutral hadron: K0L, K0S
DR_MAX_MU    = 0.1    # muons
DR_MAX_OTHER = 0.1    # everything else

TRACK_ETA_MIN = 1.48
TRACK_ETA_MAX = 3.0


def _dr_threshold(abs_pid):
    """Return dR matching threshold for a given |PID|."""
    if abs_pid in _CAT_EM:    return DR_MAX_EM
    if abs_pid in _CAT_CHAD:  return DR_MAX_CHAD
    if abs_pid in _CAT_NHAD:  return DR_MAX_NHAD
    if abs_pid in _CAT_MU:    return DR_MAX_MU
    return DR_MAX_OTHER


def get_charge(pid):
    abs_pid = abs(pid)
    if pid in _NEUTRAL_PIDS:
        return 0.0
    if abs_pid in _CHARGED_PIDS:
        return -math.copysign(1.0, pid)
    return 0.0


def is_charged_particle(pid):
    return abs(pid) in _CHARGED_PIDS


# =====================================================================================
# NanoAOD ragged-collection helper
# =====================================================================================
def _split_by_counts(flat, counts):
    counts = np.asarray(counts, dtype=np.int64)
    flat = np.asarray(flat)
    if len(counts) == 0:
        return []
    if len(flat) == 0:
        return [np.array([], dtype=flat.dtype) for _ in counts]
    edges = np.cumsum(counts)[:-1]
    return np.split(flat, edges)


# -- Jet / MET --------------------------------------------------------------------------
def _make_pseudojets(pts, etas, phis, energies):
    px = pts * np.cos(phis)
    py = pts * np.sin(phis)
    safe_eta = np.where(np.abs(etas) < 10, etas, 0.0)
    pz = pts * np.sinh(safe_eta)
    return [fastjet.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(energies[i]))
            for i in range(len(pts))]


_JET_DEF = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)


def compute_truth_jets(g, matched_gen_indices, pt_min=3.0):
    stable = [
        n for n in g.nodes
        if n[0] == "gen"
        and g.nodes[n].get("status", 0) == 1
        and g.nodes[n].get("num_daughters", 0) == 0
        and g.nodes[n].get("pid", 0) not in _NEUTRINO_PIDS
        and n[1] in matched_gen_indices   # <- only matched gen
    ]
    if not stable:
        return np.array([])
    attrs    = g.nodes
    pts      = np.array([attrs[n]["pt"]     for n in stable], dtype=np.float64)
    etas     = np.array([attrs[n]["eta"]    for n in stable], dtype=np.float64)
    phis     = np.array([attrs[n]["phi"]    for n in stable], dtype=np.float64)
    energies = np.array([attrs[n]["energy"] for n in stable], dtype=np.float64)
    try:
        pjs     = _make_pseudojets(pts, etas, phis, energies)
        cluster = fastjet.ClusterSequence(pjs, _JET_DEF)
        jets    = cluster.inclusive_jets(ptmin=pt_min)
        return np.array([[j.pt(), j.eta(), j.phi(), j.e()] for j in jets]) \
               if jets else np.array([])
    except Exception as e:
        print(f"Warning: truth jet clustering failed: {e}")
        return np.array([])


def compute_target_jets(ytarget, pt_min=3.0):
    valid = ytarget["pid"] != 0
    if not np.any(valid):
        return np.zeros((0, 4), dtype=np.float32)
    pts      = ytarget["pt"][valid].astype(np.float64)
    etas     = ytarget["eta"][valid].astype(np.float64)
    phis     = np.arctan2(ytarget["sin_phi"][valid], ytarget["cos_phi"][valid]).astype(np.float64)
    energies = ytarget["energy"][valid].astype(np.float64)
    try:
        pjs     = _make_pseudojets(pts, etas, phis, energies)
        cluster = fastjet.ClusterSequence(pjs, _JET_DEF)
        jets    = cluster.inclusive_jets(ptmin=pt_min)
        return np.array([[j.pt(), j.eta(), j.phi(), j.e()] for j in jets], dtype=np.float32) \
               if jets else np.zeros((0, 4), dtype=np.float32)
    except Exception as e:
        print(f"Warning: target jet clustering failed: {e}")
        return np.zeros((0, 4), dtype=np.float32)


def compute_genmet(g, matched_gen_indices=None):
    """GenMET from matched stable neutrinos."""
    gen_nodes = [n for n in g.nodes if n[0] == "gen"]
    attrs = g.nodes
    neutrinos = [
        n for n in gen_nodes
        if attrs[n].get("status", 0) == 1
        and attrs[n].get("pid", 0) in _NEUTRINO_PIDS
        and attrs[n].get("num_daughters", 0) == 0
        and (matched_gen_indices is None or n[1] in matched_gen_indices)
    ]
    if not neutrinos:
        return np.array([0.0, 0.0])
    pts  = np.array([attrs[n]["pt"]  for n in neutrinos])
    phis = np.array([attrs[n]["phi"] for n in neutrinos])
    sum_px = np.sum(pts * np.cos(phis))
    sum_py = np.sum(pts * np.sin(phis))
    return np.array([math.hypot(sum_px, sum_py), math.atan2(sum_py, sum_px)])


def assign_jet_indices(ytarget, targetjets, pt_min=3.0):
    ytarget_copy = ytarget.copy()
    ytarget_copy["jet_idx"] = -1
    valid = ytarget["pid"] != 0
    if not np.any(valid) or len(targetjets) == 0:
        return ytarget_copy
    pts           = ytarget["pt"][valid].astype(np.float64)
    etas          = ytarget["eta"][valid].astype(np.float64)
    phis          = np.arctan2(ytarget["sin_phi"][valid], ytarget["cos_phi"][valid]).astype(np.float64)
    energies      = ytarget["energy"][valid].astype(np.float64)
    valid_indices = np.where(valid)[0]
    try:
        pjs     = _make_pseudojets(pts, etas, phis, energies)
        cluster = fastjet.ClusterSequence(pjs, _JET_DEF)
        jets    = cluster.inclusive_jets(ptmin=pt_min)
        if hasattr(cluster, 'constituent_index'):
            indices = cluster.constituent_index(ptmin=pt_min)
        else:
            indices = []
            for jet in jets:
                consts      = jet.constituents()
                jet_indices = [i for i, p in enumerate(pjs) if any(
                    abs(p.pt() - c.pt()) < 1e-6 for c in consts)]
                indices.append(jet_indices)
        for jet_idx, const_indices in enumerate(indices):
            for ci in const_indices:
                if ci < len(valid_indices):
                    ytarget_copy["jet_idx"][valid_indices[ci]] = jet_idx
    except Exception as e:
        print(f"Warning: jet index assignment failed: {e}")
    return ytarget_copy


# -- Gen -> SimCand matching -----------------------------------------------------------
def build_gen_to_simcand_map(ev):
    """ACCEPTANCE-ONLY replacement of the original gen<->simCand deltaR matching.
    This is the ONLY change vs postprocessing_ticl_ttbar_nopu.py (diff the two files).

    Reco/target side -- UNCHANGED acceptance, matching REMOVED:
      a SimTICLCandidate is kept iff it satisfies the original script's own
      in-acceptance criterion: matching position |eta| >= 1.5 (position = the linked
      track EXTRAPOLATED to the HGCAL surface for charged candidates when it lands
      at |eta|>1.5, else the candidate direction) AND pt >= SIMCAND_PT_MIN.
      The deltaR-to-gen requirement is dropped -> every in-acceptance candidate
      contributes to the target regardless of gen matchability.

    Gen side -- the SAME endcap boundary + pt floor applied DIRECTLY to gen:
      a gen particle is kept iff |eta| >= 1.5 and pt >= GEN_PT_MIN (status-1 /
      neutrino handling stays in the callers, exactly as before). This keeps the
      truth references (truth_jets, stored pythia) endcap-consistent with the
      target without conditioning either side on the other.
    """
    gen_eta = ev["genpar_eta"]
    gen_pt  = ev["genpar_pt"]

    simcan_eta   = ev["simcan_eta"]
    simcan_pid   = ev["simcan_pdgid"]
    simcan_trkId = ev["simcan_trkId"]

    track_hgcal_eta = ev["track_hgcal_eta"]
    n_tracks = len(track_hgcal_eta)

    n_sc = len(simcan_eta)
    sc_match_eta = np.array([float(simcan_eta[j]) for j in range(n_sc)])
    for j in range(n_sc):
        abs_pid = abs(int(simcan_pid[j]))
        if abs_pid not in _CHARGED_PIDS:
            continue   # neutrals stay with their own direction
        trk_ids = simcan_trkId[j] if j < len(simcan_trkId) else []
        for tid in trk_ids:
            tid = int(tid)
            if not (0 <= tid < n_tracks):
                continue
            heta = float(track_hgcal_eta[tid])
            if abs(heta) > 1.5:
                sc_match_eta[j] = heta
                break

    simcan_pt = ev["simcan_pt"]
    sc_in_acceptance = (np.abs(sc_match_eta) >= 1.5) & (simcan_pt >= SIMCAND_PT_MIN)
    matched_simcand_indices = set(int(j) for j in np.where(sc_in_acceptance)[0])

    # gen side: keep ALL gens -> stored references (genjet, pythia) are UNCUT;
    # every gen-side acceptance/fiducial choice is applied at evaluation time instead.
    matched_gen_indices = set(range(len(gen_eta)))

    return matched_simcand_indices, matched_gen_indices


# =====================================================================================
# NanoAOD event reading
# =====================================================================================
_BRANCHES = [
    # reco tracksters
    "ticlTracksterLinks_raw_energy", "ticlTracksterLinks_raw_pt",
    "ticlTracksterLinks_barycenter_eta", "ticlTracksterLinks_barycenter_phi",
    "ticlTracksterLinks_barycenter_z", "ticlTracksterLinks_raw_em_energy",
    "ticlTracksterLinks_time", "ticlTracksterLinks_timeError",
    "ticlTracksterLinks_nticlTracksterLinksvertices", "ticlTracksterLinksvertices_vertices",

    # EG-supercluster tracksters
    "ticlTracksterLinksSuperclusteringDNN_raw_energy", "ticlTracksterLinksSuperclusteringDNN_raw_pt",
    "ticlTracksterLinksSuperclusteringDNN_barycenter_eta", "ticlTracksterLinksSuperclusteringDNN_barycenter_phi",
    "ticlTracksterLinksSuperclusteringDNN_barycenter_z", "ticlTracksterLinksSuperclusteringDNN_raw_em_energy",
    "ticlTracksterLinksSuperclusteringDNN_time", "ticlTracksterLinksSuperclusteringDNN_timeError",

    # layer clusters 
    "HGCalLayerClusters_position_z", "HGCalLayerClusters_energy",

    # reco TICL candidates
    "TICLCandidates_pt", "TICLCandidates_eta", "TICLCandidates_phi", "TICLCandidates_energy",
    "TICLCandidates_pdgID",
    "TICLCandidates_nLinkedTracksters", "linkedTracksters_tracksterIndex",
    "TICLCandidates_nTrackIdxs", "TICLCandidatesTrackIdxs_trackIndex",
    "TICLCandidates_nGsfTrackIdxs", "TICLCandidatesGsfTrackIdxs_trackIndex",

    # sim TICL candidates
    "SimTICLCandidates_pdgID", "SimTICLCandidates_pt", "SimTICLCandidates_eta", "SimTICLCandidates_phi",
    "SimTICLCandidates_energy", "SimTICLCandidates_raw_energy", "SimTICLCandidates_isPU",
    "SimTICLCandidates_nTrackIdxs", "SimTICLCandidatesTrackIdxs_trackIndex",
    "SimTICLCandidates_nGsfTrackIdxs", "SimTICLCandidatesGsfTrackIdxs_trackIndex",
    "SimTICLCandidates_nSimTICLCandidatesExtra", "SimTICLCandidatesExtra_tracksterIndex",

    # gen particles
    "HGCalGenPart_pdgId", "HGCalGenPart_mass", "HGCalGenPart_eta", "HGCalGenPart_phi",
    "HGCalGenPart_status", "HGCalGenPart_pt", "HGCalGenPart_energy",

    # general tracks
    "GeneralTrack_pt", "GeneralTrack_p", "GeneralTrack_eta", "GeneralTrack_phi", "GeneralTrack_charge",
    "GeneralTrack_missingOuterHits", "GeneralTrack_nhits",
    "GeneralTrack_muon_type", "GeneralTrack_muon_dt_hits", "GeneralTrack_muon_csc_hits",
    "GeneralTrack_etaErr", "GeneralTrack_phiErr", "GeneralTrack_lambdaErr",
    "GeneralTrack_ptErr", "GeneralTrack_qoverpErr",
    "GeneralTrack_vx", "GeneralTrack_vy", "GeneralTrack_vz",
    "GeneralTrack_hgcal_eta", "GeneralTrack_hgcal_phi",

    # GSF tracks
    "GSFTrack_ptMode", "GSFTrack_pMode", "GSFTrack_pxMode", "GSFTrack_pyMode", "GSFTrack_pzMode",
    "GSFTrack_etaMode", "GSFTrack_etaModeError", "GSFTrack_phiMode", "GSFTrack_phiModeError",
    "GSFTrack_ptModeError", "GSFTrack_vx", "GSFTrack_vy", "GSFTrack_vz",
    "GSFTrack_qoverpModeError", "GSFTrack_nhits", "GSFTrack_charge",

    # associations 
    "RecoticlTracksterLinks2SimCPByHits_nRecoticlTracksterLinks2SimCPByHitsLinks",
    "RecoticlTracksterLinks2SimCPByHitsLinks_index", "RecoticlTracksterLinks2SimCPByHitsLinks_score",
    "SimCP2ticlTracksterLinksByHits_nSimCP2ticlTracksterLinksByHitsLinks",
    "SimCP2ticlTracksterLinksByHitsLinks_index", "SimCP2ticlTracksterLinksByHitsLinks_score",
    "SimCP2ticlTracksterLinksByHitsLinks_sharedEnergy",
    "RecoticlTracksterLinksSuperclusteringDNN2SimCPByHits_nRecoticlTracksterLinksSuperclusteringDNN2SimCPByHitsLinks",
    "RecoticlTracksterLinksSuperclusteringDNN2SimCPByHitsLinks_index",
    "RecoticlTracksterLinksSuperclusteringDNN2SimCPByHitsLinks_score",
    "SimCP2ticlTracksterLinksSuperclusteringDNNByHits_nSimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks",
    "SimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks_index",
    "SimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks_score",
    "SimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks_sharedEnergy",
]


def read_all_events(events_tree, start_event=0, num_events=-1):
    total = events_tree.num_entries
    if num_events == -1:
        num_events = total - start_event
    else:
        num_events = min(num_events, total - start_event)

    a = events_tree.arrays(
        _BRANCHES, entry_start=start_event, entry_stop=start_event + num_events, library="np"
    )

    events = []
    for i in range(num_events):
        ev = {}

        # -- reco tracksters (post-linking) -------------------------------------------
        ev["ts_energy"]    = a["ticlTracksterLinks_raw_energy"][i]
        ev["ts_pt"]        = a["ticlTracksterLinks_raw_pt"][i]
        ev["ts_eta"]       = a["ticlTracksterLinks_barycenter_eta"][i]
        ev["ts_phi"]       = a["ticlTracksterLinks_barycenter_phi"][i]
        ev["ts_z"]         = a["ticlTracksterLinks_barycenter_z"][i]
        ev["ts_em_energy"] = a["ticlTracksterLinks_raw_em_energy"][i]
        ev["ts_time"]      = a["ticlTracksterLinks_time"][i]
        ev["ts_time_err"]  = a["ticlTracksterLinks_timeError"][i]

        ts_vtx_idx_lists = _split_by_counts(
            a["ticlTracksterLinksvertices_vertices"][i],
            a["ticlTracksterLinks_nticlTracksterLinksvertices"][i],
        )
        ev["ts_vertices_idx"] = ts_vtx_idx_lists

        lc_z = a["HGCalLayerClusters_position_z"][i]
        lc_e = a["HGCalLayerClusters_energy"][i]
        n_lc = len(lc_z)
        ev["ts_vertices_z"] = [
            lc_z[idxs[(idxs >= 0) & (idxs < n_lc)].astype(np.int64)] for idxs in ts_vtx_idx_lists
        ]
        ev["ts_vertices_e"] = [
            lc_e[idxs[(idxs >= 0) & (idxs < n_lc)].astype(np.int64)] for idxs in ts_vtx_idx_lists
        ]

        # -- EG-supercluster tracksters -------------------------------------------------
        ev["tsEG_energy"]    = a["ticlTracksterLinksSuperclusteringDNN_raw_energy"][i]
        ev["tsEG_pt"]        = a["ticlTracksterLinksSuperclusteringDNN_raw_pt"][i]
        ev["tsEG_eta"]       = a["ticlTracksterLinksSuperclusteringDNN_barycenter_eta"][i]
        ev["tsEG_phi"]       = a["ticlTracksterLinksSuperclusteringDNN_barycenter_phi"][i]
        ev["tsEG_z"]         = a["ticlTracksterLinksSuperclusteringDNN_barycenter_z"][i]
        ev["tsEG_em_energy"] = a["ticlTracksterLinksSuperclusteringDNN_raw_em_energy"][i]
        ev["tsEG_time"]      = a["ticlTracksterLinksSuperclusteringDNN_time"][i]
        ev["tsEG_time_err"]  = a["ticlTracksterLinksSuperclusteringDNN_timeError"][i]

        # -- reco TICL candidates --------------------------------------------------------
        ev["tcan_pt"]         = a["TICLCandidates_pt"][i]
        ev["tcan_eta"]        = a["TICLCandidates_eta"][i]
        ev["tcan_phi"]        = a["TICLCandidates_phi"][i]
        ev["tcan_energy"]     = a["TICLCandidates_energy"][i]
        ev["candidate_pdgId"] = a["TICLCandidates_pdgID"][i]

        ev["trkst_indcies"] = _split_by_counts(
            a["linkedTracksters_tracksterIndex"][i], a["TICLCandidates_nLinkedTracksters"][i]
        )
        ev["trks_indcies"] = _split_by_counts(
            a["TICLCandidatesTrackIdxs_trackIndex"][i], a["TICLCandidates_nTrackIdxs"][i]
        )
        ev["tcan_gsftrk_indices"] = _split_by_counts(
            a["TICLCandidatesGsfTrackIdxs_trackIndex"][i], a["TICLCandidates_nGsfTrackIdxs"][i]
        )

        # -- sim TICL candidates ----------------------------------------------------------
        ev["simcan_pdgid"]      = a["SimTICLCandidates_pdgID"][i]
        ev["simcan_pt"]         = a["SimTICLCandidates_pt"][i]
        ev["simcan_eta"]        = a["SimTICLCandidates_eta"][i]
        ev["simcan_phi"]        = a["SimTICLCandidates_phi"][i]
        ev["simcan_reg_energy"] = a["SimTICLCandidates_energy"][i]
        ev["simcan_raw_energy"] = a["SimTICLCandidates_raw_energy"][i]
        ev["simcan_ispu"]       = a["SimTICLCandidates_isPU"][i]

        ev["simcan_trkId"] = _split_by_counts(
            a["SimTICLCandidatesTrackIdxs_trackIndex"][i], a["SimTICLCandidates_nTrackIdxs"][i]
        )
        ev["simcan_gsftrkId"] = _split_by_counts(
            a["SimTICLCandidatesGsfTrackIdxs_trackIndex"][i], a["SimTICLCandidates_nGsfTrackIdxs"][i]
        )
        ev["simcan_simTracksterCPIndex"] = _split_by_counts(
            a["SimTICLCandidatesExtra_tracksterIndex"][i], a["SimTICLCandidates_nSimTICLCandidatesExtra"][i]
        )

        # -- gen particles ---------------------------------------------------------------
        ev["genpar_pdgid"]  = a["HGCalGenPart_pdgId"][i]
        ev["genpar_mass"]   = a["HGCalGenPart_mass"][i]
        ev["genpar_eta"]    = a["HGCalGenPart_eta"][i]
        ev["genpar_phi"]    = a["HGCalGenPart_phi"][i]
        ev["genpar_status"] = a["HGCalGenPart_status"][i]
        ev["genpar_pt"]     = a["HGCalGenPart_pt"][i]
        ev["genpar_energy"] = a["HGCalGenPart_energy"][i] 
        # -- general tracks --------------------------------------------------------------
        ev["track_pt"]        = a["GeneralTrack_pt"][i]
        ev["track_p"]         = a["GeneralTrack_p"][i]
        ev["track_eta"]       = a["GeneralTrack_eta"][i]
        ev["track_phi"]       = a["GeneralTrack_hgcal_phi"][i]
        ev["track_charge"]    = a["GeneralTrack_charge"][i]
        ev["track_hits"]      = a["GeneralTrack_missingOuterHits"][i]
        ev["track_nhits"]     = a["GeneralTrack_nhits"][i]
        ev["track_hgcal_eta"] = a["GeneralTrack_hgcal_eta"][i]
        ev["track_hgcal_phi"] = a["GeneralTrack_hgcal_phi"][i]
        ev["track_muon_type"]     = a["GeneralTrack_muon_type"][i]
        ev["track_muon_dt_hits"]  = a["GeneralTrack_muon_dt_hits"][i]
        ev["track_muon_csc_hits"] = a["GeneralTrack_muon_csc_hits"][i]
        ev["track_pt_err"]     = a["GeneralTrack_ptErr"][i]
        ev["track_eta_err"]    = a["GeneralTrack_etaErr"][i]
        ev["track_phi_err"]    = a["GeneralTrack_phiErr"][i]
        ev["track_lambda_err"] = a["GeneralTrack_lambdaErr"][i]
        ev["track_qoverp_err"] = a["GeneralTrack_qoverpErr"][i]
        ev["track_vx"] = a["GeneralTrack_vx"][i]
        ev["track_vy"] = a["GeneralTrack_vy"][i]
        ev["track_vz"] = a["GeneralTrack_vz"][i]

        # -- GSF tracks --------------------------------------------------------------------
        ev["gsf_track_ptMode"]       = a["GSFTrack_ptMode"][i]
        ev["gsf_track_ptModeError"]  = a["GSFTrack_ptModeError"][i]
        ev["gsf_track_pxMode"]       = a["GSFTrack_pxMode"][i]
        ev["gsf_track_pyMode"]       = a["GSFTrack_pyMode"][i]
        ev["gsf_track_pzMode"]       = a["GSFTrack_pzMode"][i]
        ev["gsf_track_etaMode"]      = a["GSFTrack_etaMode"][i]
        ev["gsf_track_etaModeError"] = a["GSFTrack_etaModeError"][i]
        ev["gsf_track_phiMode"]      = a["GSFTrack_phiMode"][i]
        ev["gsf_track_phiModeError"] = a["GSFTrack_phiModeError"][i]
        ev["gsf_track_pMode"]        = a["GSFTrack_pMode"][i]
        ev["gsf_track_vx"] = a["GSFTrack_vx"][i]
        ev["gsf_track_vy"] = a["GSFTrack_vy"][i]
        ev["gsf_track_vz"] = a["GSFTrack_vz"][i]
        ev["gsf_track_qoverpError"] = a["GSFTrack_qoverpModeError"][i]
        ev["gsf_track_nhits"]  = a["GSFTrack_nhits"][i]
        ev["gsf_track_charge"] = a["GSFTrack_charge"][i]

        # -- associations (CP flavor) ------------------------------------------------------
        r2s_idx_lists = _split_by_counts(
            a["RecoticlTracksterLinks2SimCPByHitsLinks_index"][i],
            a["RecoticlTracksterLinks2SimCPByHits_nRecoticlTracksterLinks2SimCPByHitsLinks"][i],
        )
        r2s_score_lists = _split_by_counts(
            a["RecoticlTracksterLinks2SimCPByHitsLinks_score"][i],
            a["RecoticlTracksterLinks2SimCPByHits_nRecoticlTracksterLinks2SimCPByHitsLinks"][i],
        )
        ev["ticlTracksterLinks_recoToSim_CP"]       = r2s_idx_lists
        ev["ticlTracksterLinks_recoToSim_CP_score"] = r2s_score_lists

        s2r_idx_lists = _split_by_counts(
            a["SimCP2ticlTracksterLinksByHitsLinks_index"][i],
            a["SimCP2ticlTracksterLinksByHits_nSimCP2ticlTracksterLinksByHitsLinks"][i],
        )
        s2r_score_lists = _split_by_counts(
            a["SimCP2ticlTracksterLinksByHitsLinks_score"][i],
            a["SimCP2ticlTracksterLinksByHits_nSimCP2ticlTracksterLinksByHitsLinks"][i],
        )
        s2r_sharedE_lists = _split_by_counts(
            a["SimCP2ticlTracksterLinksByHitsLinks_sharedEnergy"][i],
            a["SimCP2ticlTracksterLinksByHits_nSimCP2ticlTracksterLinksByHitsLinks"][i],
        )
        ev["ticlTracksterLinks_simToReco_CP"]         = s2r_idx_lists
        ev["ticlTracksterLinks_simToReco_CP_score"]   = s2r_score_lists
        ev["ticlTracksterLinks_simToReco_CP_sharedE"] = s2r_sharedE_lists

        r2s_dnn_idx_lists = _split_by_counts(
            a["RecoticlTracksterLinksSuperclusteringDNN2SimCPByHitsLinks_index"][i],
            a["RecoticlTracksterLinksSuperclusteringDNN2SimCPByHits_nRecoticlTracksterLinksSuperclusteringDNN2SimCPByHitsLinks"][i],
        )
        r2s_dnn_score_lists = _split_by_counts(
            a["RecoticlTracksterLinksSuperclusteringDNN2SimCPByHitsLinks_score"][i],
            a["RecoticlTracksterLinksSuperclusteringDNN2SimCPByHits_nRecoticlTracksterLinksSuperclusteringDNN2SimCPByHitsLinks"][i],
        )
        ev["ticlTracksterLinksDNN_recoToSim_CP"]       = r2s_dnn_idx_lists
        ev["ticlTracksterLinksDNN_recoToSim_CP_score"] = r2s_dnn_score_lists

        s2r_dnn_idx_lists = _split_by_counts(
            a["SimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks_index"][i],
            a["SimCP2ticlTracksterLinksSuperclusteringDNNByHits_nSimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks"][i],
        )
        s2r_dnn_score_lists = _split_by_counts(
            a["SimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks_score"][i],
            a["SimCP2ticlTracksterLinksSuperclusteringDNNByHits_nSimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks"][i],
        )
        s2r_dnn_sharedE_lists = _split_by_counts(
            a["SimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks_sharedEnergy"][i],
            a["SimCP2ticlTracksterLinksSuperclusteringDNNByHits_nSimCP2ticlTracksterLinksSuperclusteringDNNByHitsLinks"][i],
        )
        ev["ticlTracksterLinksDNN_simToReco_CP"]         = s2r_dnn_idx_lists
        ev["ticlTracksterLinksDNN_simToReco_CP_score"]   = s2r_dnn_score_lists
        ev["ticlTracksterLinksDNN_simToReco_CP_sharedE"] = s2r_dnn_sharedE_lists

        events.append(ev)
    return events, num_events


def collect_hadronic_connections(ev, matched_simcand_indices):
    connections = []
    r2s_scores  = ev["ticlTracksterLinks_recoToSim_CP_score"]
    s2r_scores  = ev["ticlTracksterLinks_simToReco_CP_score"]
    s2r_sharedE = ev["ticlTracksterLinks_simToReco_CP_sharedE"]
    s2r_index   = ev["ticlTracksterLinks_simToReco_CP"]
    simcan_energy = ev["simcan_reg_energy"]
    simcan_pdgid  = ev["simcan_pdgid"]
    simcan_eta    = ev["simcan_eta"]
    ts_energy     = ev["ts_energy"]

    for sim_idx, (shared_arr, idx_arr) in enumerate(zip(s2r_sharedE, s2r_index)):
        if sim_idx not in matched_simcand_indices:
            continue
        if len(shared_arr) == 0:
            continue
        cp_energy = simcan_energy[sim_idx]
        cp_pid    = simcan_pdgid[sim_idx]
        if abs(cp_pid) == 11 or abs(cp_pid) == 22:
            continue
        cp_eta = simcan_eta[sim_idx]
        for idx2, (trackster_idx, shared_energy) in enumerate(zip(idx_arr, shared_arr)):
            trackster_idx = int(trackster_idx)
            if shared_energy <= 0:
                continue
            r_score = r2s_scores[trackster_idx][0] if trackster_idx < len(r2s_scores) and len(r2s_scores[trackster_idx]) > 0 else 1.0
            s_score = s2r_scores[sim_idx][idx2]    if sim_idx < len(s2r_scores) and idx2 < len(s2r_scores[sim_idx]) else 1.0
            if cp_pid in [211, 321]:
                if r_score > 0.6 or s_score > 0.9:
                    continue
            elif cp_pid in [130, 310]:
                if r_score > 0.6 or s_score > 0.9:
                    continue
            connections.append({
                'cp_idx': sim_idx, 'cp_pid': cp_pid, 'cp_energy': cp_energy,
                'cp_eta': cp_eta, 'element_idx': trackster_idx, 'element_type': 3,
                'shared_energy': ts_energy[trackster_idx],
                'element_energy': ts_energy[trackster_idx],
                'is_charged': is_charged_particle(cp_pid)
            })
    return connections


def collect_em_connections(ev, matched_simcand_indices, use_superclustering=False):
    connections = []
    n_ts     = len(ev["ts_energy"])
    n_tracks = len(ev["track_pt"])

    if use_superclustering:
        r2s_scores  = ev["ticlTracksterLinksDNN_recoToSim_CP_score"]
        s2r_scores  = ev["ticlTracksterLinksDNN_simToReco_CP_score"]
        s2r_sharedE = ev["ticlTracksterLinksDNN_simToReco_CP_sharedE"]
        s2r_index   = ev["ticlTracksterLinksDNN_simToReco_CP"]
        if s2r_sharedE is None or len(s2r_sharedE) == 0:
            return connections
        simcan_pdgid      = ev["simcan_pdgid"]
        simcan_raw_energy = ev["simcan_raw_energy"]
        simcan_reg_energy = ev["simcan_reg_energy"]
        simcan_eta        = ev["simcan_eta"]

        tsEG_energy       = ev["tsEG_energy"]
        for sim_idx, (shared_arr, idx_arr) in enumerate(zip(s2r_sharedE, s2r_index)):
            if sim_idx not in matched_simcand_indices:
                continue
            if len(shared_arr) == 0:
                continue
            cp_pid  = simcan_pdgid[sim_idx]
            abs_pid = abs(cp_pid)
            if abs_pid not in (11, 22):
                continue
            cp_eta    = simcan_eta[sim_idx]
            cp_energy = simcan_raw_energy[sim_idx] if abs_pid == 11 else simcan_reg_energy[sim_idx]
            for idx2, (trackster_idx, shared_energy) in enumerate(zip(idx_arr, shared_arr)):
                trackster_idx = int(trackster_idx)
                if shared_energy <= 0:
                    continue
                r_score = r2s_scores[trackster_idx][0] if trackster_idx < len(r2s_scores) and len(r2s_scores[trackster_idx]) > 0 else 1.0
                s_score = s2r_scores[sim_idx][idx2] if sim_idx < len(s2r_scores) and idx2 < len(s2r_scores[sim_idx]) else 1.0
                if r_score > 0.6 or s_score > 0.9:
                    continue
                connections.append({
                    'cp_idx': sim_idx, 'cp_pid': cp_pid, 'cp_energy': cp_energy,
                    'cp_eta': cp_eta, 'element_idx': n_ts + n_tracks + trackster_idx,
                    'element_type': 2, 'shared_energy': tsEG_energy[trackster_idx],
                    'element_energy': tsEG_energy[trackster_idx],
                    'is_charged': is_charged_particle(cp_pid)
                })
    else:
        r2s_scores  = ev["ticlTracksterLinks_recoToSim_CP_score"]
        s2r_scores  = ev["ticlTracksterLinks_simToReco_CP_score"]
        s2r_sharedE = ev["ticlTracksterLinks_simToReco_CP_sharedE"]
        s2r_index   = ev["ticlTracksterLinks_simToReco_CP"]
        simcan_pdgid      = ev["simcan_pdgid"]
        simcan_raw_energy = ev["simcan_raw_energy"]
        simcan_reg_energy = ev["simcan_reg_energy"]
        simcan_eta        = ev["simcan_eta"]
        ts_energy         = ev["ts_energy"]

        for sim_idx, (shared_arr, idx_arr) in enumerate(zip(s2r_sharedE, s2r_index)):
            if sim_idx not in matched_simcand_indices:
                continue
            if len(shared_arr) == 0:
                continue
            cp_pid  = simcan_pdgid[sim_idx]
            abs_pid = abs(cp_pid)
            if abs_pid not in (11, 22):
                continue
            cp_eta    = simcan_eta[sim_idx]
            cp_energy = simcan_raw_energy[sim_idx] if abs_pid == 11 else simcan_reg_energy[sim_idx]
            for idx2, (trackster_idx, shared_energy) in enumerate(zip(idx_arr, shared_arr)):
                trackster_idx = int(trackster_idx)
                if shared_energy <= 0:
                    continue
                r_score = r2s_scores[trackster_idx][0] if trackster_idx < len(r2s_scores) and len(r2s_scores[trackster_idx]) > 0 else 1.0
                s_score = s2r_scores[sim_idx][idx2] if sim_idx < len(s2r_scores) and idx2 < len(s2r_scores[sim_idx]) else 1.0
                if r_score > 0.6 or s_score > 0.9:
                    continue
                connections.append({
                    'cp_idx': sim_idx, 'cp_pid': cp_pid, 'cp_energy': cp_energy,
                    'cp_eta': cp_eta, 'element_idx': trackster_idx, 'element_type': 2,
                    'shared_energy': ts_energy[trackster_idx],
                    'element_energy': ts_energy[trackster_idx],
                    'is_charged': is_charged_particle(cp_pid)
                })
    return connections


def collect_track_connections(ev, matched_simcand_indices):
    connections = []
    n_ts = len(ev["ts_energy"])
    simcan_trkId      = ev["simcan_trkId"]
    simcan_pdgid      = ev["simcan_pdgid"]
    simcan_reg_energy = ev["simcan_reg_energy"]
    simcan_raw_energy = ev["simcan_raw_energy"]
    simcan_eta        = ev["simcan_eta"]
    track_p           = ev["track_p"]
    n_tracks          = len(track_p)

    for cp_idx, track_indices in enumerate(simcan_trkId):
        if cp_idx not in matched_simcand_indices:
            continue
        if len(track_indices) == 0:
            continue
        cp_pid    = simcan_pdgid[cp_idx]
        cp_energy = simcan_raw_energy[cp_idx] if abs(cp_pid) == 11 else simcan_reg_energy[cp_idx]
        cp_eta    = simcan_eta[cp_idx]
        for track_idx in track_indices:
            track_idx = int(track_idx)
            if not (0 <= track_idx < n_tracks):
                continue
            tp = track_p[track_idx]
            connections.append({
                'cp_idx': cp_idx, 'cp_pid': cp_pid, 'cp_energy': cp_energy,
                'cp_eta': cp_eta, 'element_idx': n_ts + track_idx, 'element_type': 1,
                'shared_energy': tp, 'element_energy': tp,
                'is_charged': is_charged_particle(cp_pid),
            })
    return connections


def collect_gsf_track_connections(ev, matched_simcand_indices, use_superclustering=False):
    connections = []
    n_ts = len(ev["ts_energy"])
    n_tracks = len(ev["track_p"])
    n_ts_eg = len(ev["tsEG_energy"])
    simcan_gsftrkId   = ev["simcan_gsftrkId"]
    simcan_pdgid      = ev["simcan_pdgid"]
    simcan_raw_energy = ev["simcan_raw_energy"]
    simcan_eta        = ev["simcan_eta"]
    gsf_track_p       = ev["gsf_track_pMode"]
    n_gsftracks       = len(gsf_track_p)
    gsf_offset        = n_ts + n_tracks + (n_ts_eg if use_superclustering else 0)

    for cp_idx, gsf_track_indices in enumerate(simcan_gsftrkId):
        if cp_idx not in matched_simcand_indices:
            continue
        if len(gsf_track_indices) == 0:
            continue
        cp_pid = simcan_pdgid[cp_idx]
        if abs(cp_pid) != 11:
            continue
        cp_energy = simcan_raw_energy[cp_idx]
        cp_eta    = simcan_eta[cp_idx]
        for gsftrack_idx in gsf_track_indices:
            gsftrack_idx = int(gsftrack_idx)
            if not (0 <= gsftrack_idx < n_gsftracks):
                continue
            tp = gsf_track_p[gsftrack_idx]
            connections.append({
                'cp_idx': cp_idx, 'cp_pid': cp_pid, 'cp_energy': cp_energy,
                'cp_eta': cp_eta, 'element_idx': gsf_offset + gsftrack_idx, 'element_type': 4,
                'shared_energy': tp, 'element_energy': tp,
                'is_charged': is_charged_particle(cp_pid),
            })
    return connections


def split_caloparticles(connections, ev, use_superclustering=False):
    cp_groups = defaultdict(list)
    for conn in connections:
        cp_groups[conn['cp_idx']].append(conn)

    split_cps  = []
    new_cp_idx = len(ev["simcan_raw_energy"])

    simcan_ispu    = ev["simcan_ispu"]
    n_ts           = len(ev["ts_energy"])
    n_ts_eg        = len(ev["tsEG_energy"])
    n_tracks       = len(ev["track_pt"])
    track_eta      = ev["track_eta"]
    ts_eta         = ev["ts_eta"]
    tsEG_eta       = ev["tsEG_eta"]
    gsf_track_eta  = ev["gsf_track_etaMode"]
    n_ts_eg_offset = n_ts_eg if use_superclustering else 0

    for cp_idx, conns in cp_groups.items():
        if not conns:
            continue
        cp_ispu    = float(simcan_ispu[cp_idx]) if cp_idx < len(simcan_ispu) else 0.0
        abs_pid    = abs(conns[0]['cp_pid'])
        is_charged = conns[0]['is_charged']

        valid_elements = []
        for conn in conns:
            elem_idx  = conn['element_idx']
            elem_type = conn['element_type']
            cp_eta    = conn['cp_eta']
            elem_eta  = None
            if elem_type == 1:
                ti = elem_idx - n_ts
                if 0 <= ti < len(track_eta):
                    elem_eta = track_eta[ti]
            elif elem_type == 4:
                ti = elem_idx - (n_ts + n_tracks + n_ts_eg_offset)
                if 0 <= ti < len(gsf_track_eta):
                    elem_eta = gsf_track_eta[ti]
            elif elem_type == 3:
                if 0 <= elem_idx < n_ts:
                    elem_eta = ts_eta[elem_idx]
            elif elem_type == 2:
                if use_superclustering:
                    if elem_idx < n_ts:
                        elem_eta = ts_eta[elem_idx]
                    else:
                        ti = elem_idx - (n_ts + n_tracks)
                        if 0 <= ti < n_ts_eg:
                            elem_eta = tsEG_eta[ti]
                else:
                    if 0 <= elem_idx < n_ts:
                        elem_eta = ts_eta[elem_idx]
            if elem_eta is not None and cp_eta * elem_eta > 0:
                valid_elements.append(conn)

        if not valid_elements:
            continue

        tracks         = [e for e in valid_elements if e['element_type'] == 1]
        had_tracksters = [e for e in valid_elements if e['element_type'] == 3]
        em_tracksters  = [e for e in valid_elements if e['element_type'] == 2]
        gsf_tracks     = [e for e in valid_elements if e['element_type'] == 4]

        if abs_pid in (11, 22, 13):
            if abs_pid == 11:
                if not tracks and not em_tracksters and not gsf_tracks:
                    continue
                if gsf_tracks:
                    total_shared_gsftracks = sum(t['shared_energy'] for t in gsf_tracks) or 1.0
                    for gsft in gsf_tracks:
                        sf = gsft['shared_energy'] / total_shared_gsftracks
                        split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                            'element_idx': gsft['element_idx'], 'element_type': 4,
                            'cp_energy': gsft['cp_energy'] * sf, 'cp_pid': gsft['cp_pid'],
                            'cp_eta': gsft['cp_eta'], 'weight': 1, 'is_winner': True, 'ispu': cp_ispu})
                if tracks:
                    total_shared_tracks = sum(t['shared_energy'] for t in tracks) or 1.0
                    for t in tracks:
                        sf = t['shared_energy'] / total_shared_tracks
                        split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                            'element_idx': t['element_idx'], 'element_type': 1,
                            'cp_energy': t['cp_energy'] * sf, 'cp_pid': t['cp_pid'],
                            'cp_eta': t['cp_eta'], 'weight': 1, 'is_winner': True, 'ispu': cp_ispu})
                if not gsf_tracks and not tracks and em_tracksters:
                    total_shared_em = sum(e['shared_energy'] for e in em_tracksters) or 1.0
                    for em in em_tracksters:
                        sf = em['shared_energy'] / total_shared_em
                        split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                            'element_idx': em['element_idx'], 'element_type': 2,
                            'cp_energy': em['cp_energy'] * sf, 'cp_pid': em['cp_pid'],
                            'cp_eta': em['cp_eta'], 'weight': 1, 'is_winner': True, 'ispu': cp_ispu})
                new_cp_idx += 1
            elif abs_pid == 22:
                if not em_tracksters:
                    continue
                total_shared = sum(e['shared_energy'] for e in em_tracksters) or 1.0
                for em in em_tracksters:
                    sf = em['shared_energy'] / total_shared
                    split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                        'element_idx': em['element_idx'], 'element_type': 2,
                        'cp_energy': em['cp_energy'] * sf, 'cp_pid': em['cp_pid'],
                        'cp_eta': em['cp_eta'], 'weight': 1.0, 'is_winner': True, 'ispu': cp_ispu})
                    new_cp_idx += 1
            elif abs_pid == 13:
                if not tracks:
                    continue
                total_shared = sum(m['shared_energy'] for m in tracks) or 1.0
                for m in tracks:
                    sf = m['shared_energy'] / total_shared
                    split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                        'element_idx': m['element_idx'], 'element_type': 1,
                        'cp_energy': m['cp_energy'] * sf, 'cp_pid': m['cp_pid'],
                        'cp_eta': m['cp_eta'], 'weight': 1, 'is_winner': True, 'ispu': cp_ispu})
                    new_cp_idx += 1

        elif is_charged and len(tracks) == 1:
            t = tracks[0]
            split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                'element_idx': t['element_idx'], 'element_type': 1,
                'cp_energy': t['cp_energy'], 'cp_pid': t['cp_pid'],
                'cp_eta': t['cp_eta'], 'cp_fraction': 1.0, 'weight': 1,
                'is_winner': True, 'ispu': cp_ispu})
            new_cp_idx += 1
            for had in had_tracksters:
                split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                    'element_idx': had['element_idx'], 'element_type': 3,
                    'cp_energy': 0.0, 'cp_pid': had['cp_pid'],
                    'cp_eta': had['cp_eta'], 'weight': 0.0,
                    'is_winner': False, 'ispu': cp_ispu})
                new_cp_idx += 1

        elif is_charged and len(tracks) > 1:
            total_shared = sum(t['shared_energy'] for t in tracks) or 1.0
            for track in tracks:
                sf = track['shared_energy'] / total_shared
                split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                    'element_idx': track['element_idx'], 'element_type': 1,
                    'cp_energy': track['cp_energy'] * sf, 'cp_pid': track['cp_pid'],
                    'cp_eta': track['cp_eta'], 'weight': 1, 'is_winner': True, 'ispu': cp_ispu})
                new_cp_idx += 1
            for had in had_tracksters:
                split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                    'element_idx': had['element_idx'], 'element_type': 3,
                    'cp_energy': 0.0, 'cp_pid': had['cp_pid'],
                    'cp_eta': had['cp_eta'], 'weight': 0.0,
                    'is_winner': False, 'ispu': cp_ispu})
                new_cp_idx += 1

        else:
            total_shared = sum(e['shared_energy'] for e in had_tracksters) or 1.0
            for elem in had_tracksters:
                sf = elem['shared_energy'] / total_shared
                split_cps.append({'original_idx': cp_idx, 'new_idx': new_cp_idx,
                    'element_idx': elem['element_idx'], 'element_type': elem['element_type'],
                    'cp_energy': elem['cp_energy'] * sf, 'cp_pid': elem['cp_pid'],
                    'cp_eta': elem['cp_eta'], 'weight': 1.0, 'is_winner': True, 'ispu': cp_ispu})
                new_cp_idx += 1

    return split_cps


def compute_trackster_features(ev):
    trk_pt  = ev["track_pt"]
    trk_eta = ev["track_eta"]
    good = (trk_pt >= 1) & (np.abs(trk_eta) > TRACK_ETA_MIN) & (np.abs(trk_eta) < TRACK_ETA_MAX)
    tk_eta = ev["track_hgcal_eta"][good]
    tk_phi = ev["track_hgcal_phi"][good]
    tk_pt  = trk_pt[good]

    def get_min_dR(ts_eta, ts_phi):
        if len(tk_eta) == 0:
            return 99.0, 0.0
        deta = tk_eta - ts_eta
        dphi = np.arctan2(np.sin(tk_phi - ts_phi), np.cos(tk_phi - ts_phi))
        dR   = np.sqrt(deta**2 + dphi**2)
        idx  = np.argmin(dR)
        return float(dR[idx]), float(tk_pt[idx])

    n_ts = len(ev["ts_energy"])
    ts_min_dR  = np.zeros(n_ts)
    ts_near_pt = np.zeros(n_ts)
    ts_depth   = np.zeros(n_ts)

    for i in range(n_ts):
        ts_min_dR[i], ts_near_pt[i] = get_min_dR(
            float(ev["ts_eta"][i]), float(ev["ts_phi"][i]))
        vz = np.abs(np.asarray(ev["ts_vertices_z"][i], dtype=np.float64))
        ve = np.asarray(ev["ts_vertices_e"][i], dtype=np.float64)
        if len(vz) > 0 and ve.sum() > 0:
            ts_depth[i] = float(np.sum(vz * ve) / np.sum(ve))
        else:
            ts_depth[i] = float(np.abs(ev["ts_z"][i]))

    ts_sum_pt10 = np.zeros(n_ts)
    ts_n_trk01  = np.zeros(n_ts)
    ts_n_trk02  = np.zeros(n_ts)
    ts_n_trk03  = np.zeros(n_ts)
    ts_n_trk04  = np.zeros(n_ts)
    ts_n_trk05  = np.zeros(n_ts)

    for i in range(n_ts):
        ts_eta_i = float(ev["ts_eta"][i])
        ts_phi_i = float(ev["ts_phi"][i])
        if len(tk_eta) > 0:
            deta = tk_eta - ts_eta_i
            dphi = np.arctan2(np.sin(tk_phi - ts_phi_i), np.cos(tk_phi - ts_phi_i))
            dR   = np.sqrt(deta**2 + dphi**2)
            ts_sum_pt10[i] = float(np.sum(tk_pt[dR < 0.10]))
            ts_n_trk01[i]  = float(np.sum(dR < 0.01))
            ts_n_trk02[i]  = float(np.sum(dR < 0.02))
            ts_n_trk03[i]  = float(np.sum(dR < 0.03))
            ts_n_trk04[i]  = float(np.sum(dR < 0.04))
            ts_n_trk05[i]  = float(np.sum(dR < 0.05))

    return ts_min_dR, ts_near_pt, ts_depth, ts_sum_pt10, \
           ts_n_trk01, ts_n_trk02, ts_n_trk03, ts_n_trk04, ts_n_trk05


def make_graph(ev, iev, use_superclustering=False):
    g = nx.DiGraph()

    n_ts     = len(ev["ts_energy"])
    n_tracks = len(ev["track_pt"])
    n_ts_eg  = len(ev["tsEG_energy"])

    matched_simcand_indices, matched_gen_indices = build_gen_to_simcand_map(ev)

    ts_min_dR, ts_near_pt, ts_depth, ts_sum_pt10, \
        ts_n_trk01, ts_n_trk02, ts_n_trk03, ts_n_trk04, ts_n_trk05 = \
        compute_trackster_features(ev)

    ts_eta = ev["ts_eta"]; ts_phi = ev["ts_phi"]
    ts_pt  = ev["ts_pt"];  ts_en  = ev["ts_energy"]
    _theta = 2.0 * np.arctan(np.exp(-ts_eta.astype(np.float64)))
    _px    = ts_pt * np.cos(ts_phi)
    _py    = ts_pt * np.sin(ts_phi)
    _pz    = ts_en * np.cos(_theta)

    ts_vx     = ev["ts_vertices_idx"]
    ts_time   = ev["ts_time"]
    ts_time_e = ev["ts_time_err"]

    g.add_nodes_from(
        (("elem", i), dict(
            typ=3, pt=float(ts_pt[i]), energy=float(ts_en[i]),
            eta=float(ts_eta[i]), phi=float(ts_phi[i]), charge=0.0,
            px=float(_px[i]), py=float(_py[i]), pz=float(_pz[i]),
            em_energy=float(ev["ts_em_energy"][i]),
            bary_z=float(ev["ts_z"][i]), nhits=0.0,
            min_dR_track=float(ts_min_dR[i]),
            near_track_pt=float(ts_near_pt[i]),
            shower_depth=float(ts_depth[i]),
            sum_pt_dR10=float(ts_sum_pt10[i]),
            n_trk_dR01=float(ts_n_trk01[i]),
            n_trk_dR02=float(ts_n_trk02[i]),
            n_trk_dR03=float(ts_n_trk03[i]),
            n_trk_dR04=float(ts_n_trk04[i]),
            n_trk_dR05=float(ts_n_trk05[i]),
            isolation=float(ts_min_dR[i] * float(ts_en[i])),
            n_clusters=float(len(ts_vx[i])),
            ts_time=float(ts_time[i])     if len(ts_time)   > i else 0.0,
            ts_time_err=float(ts_time_e[i]) if len(ts_time_e) > i else 0.0,
        ))
        for i in range(n_ts)
    )

    trk_pt  = ev["track_pt"];  trk_p   = ev["track_p"]
    trk_eta = ev["track_eta"]; trk_phi = ev["track_phi"]
    trk_chg = ev["track_charge"]
    safe_eta = np.where(np.abs(trk_eta) < 10, trk_eta, 0.0)
    _pz_trk  = trk_pt * np.sinh(safe_eta.astype(np.float64))

    track_nodes = []
    for itrk in range(n_tracks):
        if not (TRACK_ETA_MIN < abs(float(trk_eta[itrk])) < TRACK_ETA_MAX):
            continue
        track_nodes.append((
            ("elem", n_ts + itrk),
            dict(
                typ=1,
                pt=float(trk_pt[itrk]), energy=float(trk_p[itrk]),
                eta=float(trk_eta[itrk]), phi=float(trk_phi[itrk]),
                layer=0, charge=float(trk_chg[itrk]),
                px=float(trk_pt[itrk] * math.cos(trk_phi[itrk])),
                py=float(trk_pt[itrk] * math.sin(trk_phi[itrk])),
                pz=float(_pz_trk[itrk]),
                em_energy=0.0, bary_z=0.0,
                nhits=float(ev["track_nhits"][itrk]),
                min_dR_track=0.0, near_track_pt=0.0, shower_depth=0.0,
                sum_pt_dR10=0.0, n_trk_dR01=0.0, n_trk_dR02=0.0,
                n_trk_dR03=0.0, n_trk_dR04=0.0, n_trk_dR05=0.0,
                isolation=0.0,
                n_clusters=0.0, ts_time=0.0, ts_time_err=0.0,
                track_muon_type=float(ev["track_muon_type"][itrk]),
                track_muon_dt_hits=float(ev["track_muon_dt_hits"][itrk]),
                track_muon_csc_hits=float(ev["track_muon_csc_hits"][itrk]),
                # so this per-GeneralTrack match flag is no longer computed.
                track_gsf_type=0.0,
                track_pt_err=float(ev["track_pt_err"][itrk]),
                track_eta_err=float(ev["track_eta_err"][itrk]),
                track_phi_err=float(ev["track_phi_err"][itrk]),
                track_lambda_err=float(ev["track_lambda_err"][itrk]),
                track_qoverp_err=float(ev["track_qoverp_err"][itrk]),
                track_vx=float(ev["track_vx"][itrk]),
                track_vy=float(ev["track_vy"][itrk]),
                track_vz=float(ev["track_vz"][itrk]),
            )
        ))
    g.add_nodes_from(track_nodes)

    n_gsftracks = len(ev["gsf_track_ptMode"])
    gsf_offset = n_ts + n_tracks + (n_ts_eg if use_superclustering else 0)
    for igsf in range(n_gsftracks):
        g.add_node(
            ("elem", gsf_offset + igsf),
            typ=2,  
            pt=float(ev["gsf_track_ptMode"][igsf]),
            energy=float(ev["gsf_track_pMode"][igsf]),
            eta=float(ev["gsf_track_etaMode"][igsf]),
            phi=float(ev["gsf_track_phiMode"][igsf]),
            charge=float(ev["gsf_track_charge"][igsf]),
            px=float(ev["gsf_track_pxMode"][igsf]),
            py=float(ev["gsf_track_pyMode"][igsf]),
            pz=float(ev["gsf_track_pzMode"][igsf]),
            em_energy=0.0, bary_z=0.0, nhits=float(ev["gsf_track_nhits"][igsf]),
            min_dR_track=0.0, near_track_pt=0.0, shower_depth=0.0,
            sum_pt_dR10=0.0, n_trk_dR01=0.0, n_trk_dR02=0.0,
            n_trk_dR03=0.0, n_trk_dR04=0.0, n_trk_dR05=0.0,
            isolation=0.0, n_clusters=0.0,
            ts_time=0.0, ts_time_err=0.0,
            track_muon_type=0.0, track_muon_dt_hits=0.0, track_muon_csc_hits=0.0,
            track_gsf_type=1.0,
            track_pt_err=float(ev["gsf_track_ptModeError"][igsf]),
            track_eta_err=float(ev["gsf_track_etaModeError"][igsf]),
            track_phi_err=float(ev["gsf_track_phiModeError"][igsf]),
            track_lambda_err=0.0,
            track_qoverp_err=float(ev["gsf_track_qoverpError"][igsf]),
            track_vx=float(ev["gsf_track_vx"][igsf]),
            track_vy=float(ev["gsf_track_vy"][igsf]),
            track_vz=float(ev["gsf_track_vz"][igsf]),
        )

    eg_eta = ev["tsEG_eta"]; eg_phi = ev["tsEG_phi"]
    eg_pt  = ev["tsEG_pt"];  eg_en  = ev["tsEG_energy"]
    _theta_eg = 2.0 * np.arctan(np.exp(-eg_eta.astype(np.float64)))
    _px_eg    = eg_pt * np.cos(eg_phi)
    _py_eg    = eg_pt * np.sin(eg_phi)
    _pz_eg    = eg_en * np.cos(_theta_eg)

    if use_superclustering:
        g.add_nodes_from(
            (("elem", n_ts + n_tracks + i), dict(
                typ=4, is_em=True, pt=float(eg_pt[i]), energy=float(eg_en[i]),
                eta=float(eg_eta[i]), phi=float(eg_phi[i]), charge=0.0,
                px=float(_px_eg[i]), py=float(_py_eg[i]), pz=float(_pz_eg[i]),
                em_energy=float(ev["tsEG_em_energy"][i]),
                bary_z=float(ev["tsEG_z"][i]), nhits=0.0,
                min_dR_track=0.0, near_track_pt=0.0, shower_depth=0.0,
                sum_pt_dR10=0.0, n_trk_dR01=0.0, n_trk_dR02=0.0,
                n_trk_dR03=0.0, n_trk_dR04=0.0, n_trk_dR05=0.0,
                isolation=0.0,
            ))
            for i in range(n_ts_eg)
        )

    em_connections  = collect_em_connections(ev, matched_simcand_indices,
                                             use_superclustering=use_superclustering)
    all_connections = (collect_hadronic_connections(ev, matched_simcand_indices) +
                       em_connections +
                       collect_track_connections(ev, matched_simcand_indices)
                       + collect_gsf_track_connections(ev, matched_simcand_indices, use_superclustering))
    split_cps = split_caloparticles(all_connections, ev, use_superclustering)

    em_ts_indices = set(conn["element_idx"] for conn in em_connections)
    for ts_idx in em_ts_indices:
        node_id = ("elem", ts_idx)
        if node_id in g.nodes:
            g.nodes[node_id]["is_em"] = True

    cp_totals = defaultdict(lambda: {'track': 0.0, 'cluster': 0.0})
    for sc in split_cps:
        key = 'track' if sc['element_type'] in (1, 4) else 'cluster'
        cp_totals[sc['original_idx']][key] += sc['weight']

    simcan_phi = ev["simcan_phi"]

    cp_nodes = []
    cp_edges = []
    seen_cp_idx = {}

    for sc in split_cps:
        original_idx = sc['original_idx']
        if original_idx >= len(ev["simcan_raw_energy"]):
            continue
        cp_idx  = sc['new_idx']
        node_id = ("cp", cp_idx)

        if cp_idx not in seen_cp_idx:
            cp_eta    = float(sc['cp_eta'])
            cp_phi    = float(simcan_phi[original_idx])
            cp_pid_v  = int(sc['cp_pid'])
            cp_energy = float(sc['cp_energy'])
            theta     = 2.0 * math.atan(math.exp(-cp_eta))
            cp_pt     = cp_energy * math.sin(theta)
            track_total   = cp_totals[original_idx]['track']
            cluster_total = cp_totals[original_idx]['cluster']
            cp_nodes.append((node_id, dict(
                pid=abs(int(cp_pid_v)), pt=cp_pt, eta=cp_eta,
                sin_phi=math.sin(cp_phi), cos_phi=math.cos(cp_phi),
                energy=cp_energy, ispu=float(sc['ispu']),
                generatorStatus=0, simulatorStatus=1,
                cp_to_track=float(track_total), cp_to_cluster=float(cluster_total),
                jet_idx=-1,
                px=cp_pt * math.cos(cp_phi), py=cp_pt * math.sin(cp_phi),
                pz=cp_energy * math.cos(theta),
                charge=get_charge(cp_pid_v), original_cp_idx=original_idx
            )))
            seen_cp_idx[cp_idx] = True

        elem_node = ("elem", sc['element_idx'])
        cp_edges.append((node_id, elem_node, sc['element_type'], sc['weight'], sc['is_winner']))

    g.add_nodes_from(cp_nodes)
    for (node_id, elem_node, element_type, weight, is_winner) in cp_edges:
        if elem_node not in g.nodes:
            continue
        g.add_edge(node_id, elem_node, weight=weight, is_winner=is_winner)

    n_tcan = len(ev["tcan_pt"])
    cand_nodes = []
    for ipf in range(n_tcan):
        pid = ev["candidate_pdgId"][ipf] if ipf < len(ev["candidate_pdgId"]) else 211
        cand_nodes.append((
            ("pfcand", ipf),
            dict(pid=abs(int(pid)), charge=float(get_charge(pid)),
                 pt=float(ev["tcan_pt"][ipf]), eta=float(ev["tcan_eta"][ipf]),
                 sin_phi=math.sin(ev["tcan_phi"][ipf]),
                 cos_phi=math.cos(ev["tcan_phi"][ipf]),
                 energy=float(ev["tcan_energy"][ipf]),
                 ispu=0.0, generatorStatus=0, simulatorStatus=0,
                 cp_to_track=0.0, cp_to_cluster=0.0, jet_idx=-1)
        ))
    g.add_nodes_from(cand_nodes)

    cand_edges = []
    for cand_idx in range(n_tcan):
        cand_node    = ("pfcand", cand_idx)
        ts_indices   = ev["trkst_indcies"][cand_idx] if cand_idx < len(ev["trkst_indcies"]) else []
        trk_idx_list = ev["trks_indcies"][cand_idx] if cand_idx < len(ev["trks_indcies"]) else []
        gsf_idx_list = ev["tcan_gsftrk_indices"][cand_idx] if cand_idx < len(ev["tcan_gsftrk_indices"]) else []
        for ts_idx in ts_indices:
            ts_idx = int(ts_idx)
            if 0 <= ts_idx < n_ts:
                en = ("elem", ts_idx)
                if en in g.nodes:
                    cand_edges.append((en, cand_node, 1.0))
        for trk_idx in trk_idx_list:
            trk_idx = int(trk_idx)
            if 0 <= trk_idx < n_tracks:
                en = ("elem", n_ts + trk_idx)
                if en in g.nodes:
                    cand_edges.append((en, cand_node, 1.0))

        for gsf_idx in gsf_idx_list:
            gsf_idx = int(gsf_idx)
            if 0 <= gsf_idx < n_gsftracks:
                en = ("elem", gsf_offset + gsf_idx)
                if en in g.nodes:
                    cand_edges.append((en, cand_node, 1.0))
    g.add_edges_from((u, v, {"weight": w}) for u, v, w in cand_edges)

    if "genpar_pdgid" in ev and len(ev["genpar_pdgid"]) > 0:
        n_gen  = len(ev["genpar_pdgid"])
        pts    = ev["genpar_pt"];   etas   = ev["genpar_eta"]
        phis   = ev["genpar_phi"];  ens    = ev["genpar_energy"]
        masses = ev["genpar_mass"]; stats  = ev["genpar_status"]
        safe_eta_gen = np.where(np.abs(etas) < 10, etas, 0.0)
        px_gen = pts * np.cos(phis)
        py_gen = pts * np.sin(phis)
        pz_gen = pts * np.sinh(safe_eta_gen.astype(np.float64))
        gen_nodes_list = [
            (("gen", i), dict(
                pid=abs(int(ev["genpar_pdgid"][i])),
                pt=float(pts[i]), eta=float(etas[i]), phi=float(phis[i]),
                energy=float(ens[i]), mass=float(masses[i]),
                status=int(stats[i]),
                px=float(px_gen[i]), py=float(py_gen[i]), pz=float(pz_gen[i]),
                num_daughters=0, is_stable=(int(stats[i]) == 1)
            ))
            for i in range(n_gen)
        ]
        g.add_nodes_from(gen_nodes_list)

    g.graph["matched_gen_indices"] = matched_gen_indices

    return g


def find_representative_elements(g, elem_to_cp, cp_to_elem, elem_type, pid_type=0):
    elems = [(g.nodes[e]["pt"], e) for e in g.nodes
             if e[0] == "elem" and g.nodes[e]["typ"] == elem_type]
    for _, elem in sorted(elems, key=lambda x: x[0], reverse=True):
        cps = list(g.predecessors(elem))
        if pid_type:
            cps = [cp for cp in cps if abs(g.nodes[cp]["pid"]) == pid_type]
        available = [cp for cp in cps if cp not in cp_to_elem and cp[0] == "cp"]
        if not available:
            continue
        if g.nodes[elem].get("is_em", False):
            cp = max(available, key=lambda cp: g.nodes[cp]["energy"])
        else:
            cps_weight = [(g.edges[(cp, elem)]["weight"], cp) for cp in available]
            cp = max(cps_weight, key=lambda x: x[0])[1]
        elem_to_cp[elem] = cp
        cp_to_elem[cp]   = elem


def prepare_normalized_table(g):
    all_elements = [n for n in g.nodes if n[0] == "elem"]
    all_elements.sort(key=lambda x: (
        0 if g.nodes[x]["typ"] == 2 else
        1 if g.nodes[x]["typ"] == 1 else
        2 if g.nodes[x]["typ"] == 4 else
        3 if g.nodes[x]["typ"] == 3 else 4,
        x[1]
    ))

    elem_to_primary_cp = {}
    cp_to_primary_elem = {}
    find_representative_elements(g, elem_to_primary_cp, cp_to_primary_elem, 2, pid_type=11)
    find_representative_elements(g, elem_to_primary_cp, cp_to_primary_elem, 1, pid_type=11)
    for et in (1, 3, 4):
        find_representative_elements(g, elem_to_primary_cp, cp_to_primary_elem, et)

    primary_pairs = set()
    for elem, cp in elem_to_primary_cp.items():
        primary_pairs.add((cp, elem))

    elem_to_all_cps = defaultdict(list)
    elem_to_cand    = {}
    for elem in all_elements:
        for pred in g.predecessors(elem):
            if pred[0] == "cp":
                if (pred, elem) in primary_pairs:
                    elem_to_all_cps[elem].append(pred)
                elif (g.nodes[elem].get("is_em", False)
                      and abs(g.nodes[pred]["pid"]) == 22
                      and pred not in elem_to_all_cps[elem]):
                    elem_to_all_cps[elem].append(pred)
        for succ in g.successors(elem):
            if succ[0] == "pfcand":
                elem_to_cand[elem] = succ
                break

    n       = len(all_elements)
    Xelem   = np.recarray((n,), dtype=[(name, np.float32) for name in elem_branches])
    ytarget = np.recarray((n,), dtype=[(name, np.float32) for name in particle_feature_order])
    ycand   = np.recarray((n,), dtype=[(name, np.float32) for name in particle_feature_order])
    Xelem.fill(0.0); ytarget.fill(0.0); ycand.fill(0.0)

    attrs = g.nodes

    for ielem, elem in enumerate(all_elements):
        nd = attrs[elem]
        for branch in elem_branches:
            if branch in nd:
                Xelem[branch][ielem] = nd[branch]

    for ielem, elem in enumerate(all_elements):
        cps = elem_to_all_cps.get(elem, [])
        if cps:
            highest_energy_cp = max(cps, key=lambda cp: attrs[cp]["energy"])
            pid = attrs[highest_energy_cp]["pid"]
            total_energy = 0.0
            ws = defaultdict(float)
            for cp in cps:
                w         = g.edges[(cp, elem)].get('weight', 1.0)
                cp_energy = attrs[cp]["energy"] * w
                total_energy += cp_energy
                for key in ('charge', 'pt', 'eta', 'sin_phi', 'cos_phi',
                            'energy', 'ispu', 'cp_to_track', 'cp_to_cluster'):
                    if key in attrs[cp]:
                        ws[key] += attrs[cp][key] * cp_energy
            if total_energy > 0:
                inv = 1.0 / total_energy
                ytarget["pid"][ielem]          = pid
                ytarget["charge"][ielem]        = ws['charge']       * inv
                ytarget["pt"][ielem]            = ws['pt']            * inv
                ytarget["eta"][ielem]           = ws['eta']           * inv
                ytarget["sin_phi"][ielem]       = ws['sin_phi']       * inv
                ytarget["cos_phi"][ielem]       = ws['cos_phi']       * inv
                ytarget["energy"][ielem]        = total_energy
                ytarget["ispu"][ielem]          = ws['ispu']          * inv
                ytarget["cp_to_track"][ielem]   = ws['cp_to_track']   * inv
                ytarget["cp_to_cluster"][ielem] = ws['cp_to_cluster'] * inv
                if (pid == 22
                    and g.nodes[elem].get("is_em", False)):
                    eta_val = float(ws['eta'] * inv)
                    theta   = 2.0 * math.atan(math.exp(-abs(eta_val)))
                    ytarget["pt"][ielem] = total_energy * math.sin(theta)
                else:
                    ytarget["pt"][ielem] = ws['pt'] * inv
        else:
            nd = attrs[elem]
            ytarget["charge"][ielem]  = nd.get("charge", 0.0)
            ytarget["eta"][ielem]     = nd["eta"]
            ytarget["sin_phi"][ielem] = math.sin(nd["phi"])
            ytarget["cos_phi"][ielem] = math.cos(nd["phi"])
            ytarget["ispu"][ielem]    = 1.0

        ytarget["simulatorStatus"][ielem] = 1 if cps else 0
        ytarget["jet_idx"][ielem]         = -1

    for ielem, elem in enumerate(all_elements):
        cand = elem_to_cand.get(elem)
        if cand is not None:
            nd = attrs[cand]
            for feature in particle_feature_order:
                if feature in nd:
                    ycand[feature][ielem] = nd[feature]

    return Xelem, ycand, ytarget


def _process_file_worker(args):
    input_file, num_events, start_event, use_superclustering = args
    return process_file_no_progress(input_file, num_events, start_event,
                                    use_superclustering=use_superclustering)


def process_file_no_progress(input_file, num_events=-1, start_event=0,
                              use_superclustering=False):
    tf = uproot.open(input_file)
    if "Events" not in tf:
        return []
    events_tree = tf["Events"]

    total_events = events_tree.num_entries
    if num_events == -1:
        num_events = total_events - start_event
    else:
        num_events = min(num_events, total_events - start_event)

    all_data = []
    pt_min   = 3.0

    all_events, num_events = read_all_events(events_tree, start_event, num_events)

    for iev, ev in enumerate(all_events):
        try:
            g     = make_graph(ev, iev, use_superclustering=use_superclustering)
            Xelem, ycand, ytarget = prepare_normalized_table(g)

            matched_gen_indices = g.graph.get("matched_gen_indices", set())

            truth_jets = compute_truth_jets(g, matched_gen_indices, pt_min=pt_min)
            genmet      = compute_genmet(g, None)
            targetjets = compute_target_jets(ytarget, pt_min=pt_min)
            ytarget_ji = assign_jet_indices(ytarget, targetjets, pt_min=pt_min)

            stable_gen = []
            for node in g.nodes:
                if node[0] != "gen":
                    continue
                if g.nodes[node].get("status", 0) != 1:
                    continue
                pid = g.nodes[node]["pid"]
                eta = g.nodes[node]["eta"]
                pt  = g.nodes[node]["pt"]

                if abs(pid) in _NEUTRINO_PIDS:
                    continue   # stored pythia = visible gens only (genMET keeps neutrinos separately)

                # UNCUT storage: no eta window, no pt floor (eval applies acceptance)
                stable_gen.append([pid, pt, eta,
                                   g.nodes[node]["phi"],
                                   g.nodes[node]["energy"]])
            stable_gen_array = np.array(stable_gen, dtype=np.float32) \
                               if stable_gen else np.array([], dtype=np.float32)

            all_data.append({
                "Xelem":            Xelem,
                "ycand":            ycand,
                "ytarget":          ytarget_ji,
                "genjet":           truth_jets,
                "targetjet":        targetjets,
                "genmet":           genmet,
                "pythia":           stable_gen_array,
                "event_idx":        start_event + iev,
                "file_name":        os.path.basename(input_file),
                "global_event_idx": len(all_data)
            })
        except Exception as e:
            print(f"\nERROR event {start_event + iev} in {input_file}: {e}")
            traceback.print_exc()
    return all_data


def process_files(input_list, output_file, num_events=-1, events_per_file=-1,
                  num_workers=None, events_per_pkl=-1, use_superclustering=False):
    input_list = [f.strip() for f in input_list if f.strip() and not f.startswith('#')]

    if num_workers is None:
        num_workers = min(len(input_list), max(1, multiprocessing.cpu_count() - 1))

    tasks     = []
    remaining = num_events
    for f in input_list:
        if num_events > 0 and remaining <= 0:
            break
        if events_per_file > 0:
            n = events_per_file
        elif num_events > 0:
            n = remaining
        else:
            n = -1
        tasks.append((f, n, 0, use_superclustering))
        if num_events > 0:
            remaining -= (n if n > 0 else 9999999)

    all_data   = []
    total_done = 0
    chunk_idx  = 0

    print(f"Launching {num_workers} parallel workers for {len(tasks)} files ...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_file_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="Files", unit="file") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    file_data = future.result()
                    all_data.extend(file_data)
                    total_done += len(file_data)
                    pbar.set_postfix(events=total_done)
                    if events_per_pkl > 0:
                        while len(all_data) >= events_per_pkl:
                            chunk = all_data[:events_per_pkl]
                            all_data = all_data[events_per_pkl:]
                            base = output_file.replace('.pkl', '')
                            out_path = f"{base}_{chunk_idx:04d}.pkl"
                            print(f"\n  Auto-saving {len(chunk)} events to {out_path} ...")
                            with open(out_path, 'wb') as f:
                                pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
                            print(f"  Saved {out_path}")
                            chunk_idx += 1
                except Exception as e:
                    print(f"\nERROR in worker for {task[0]}: {e}")
                pbar.update(1)

    return all_data, chunk_idx


def main():
    parser = argparse.ArgumentParser(
        description='TICL postprocessing (NanoAOD) -- gen->simCand matching, no eta cuts')
    parser.add_argument('--input',           type=str, required=True)
    parser.add_argument('--output',          type=str, default='ticl_graph_data.pkl')
    parser.add_argument('--num-events',      type=int, default=-1)
    parser.add_argument('--events-per-file', type=int, default=-1)
    parser.add_argument('--max-files',       type=int, default=-1)
    parser.add_argument('--num-workers',     type=int, default=None)
    parser.add_argument('--events-per-pkl',  type=int, default=-1)
    parser.add_argument('--use-superclustering',
                        dest='use_superclustering',
                        action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found!"); return

    if args.input.endswith('.root'):
        input_files = [args.input]
    elif args.input.endswith('.txt'):
        with open(args.input) as f:
            input_files = [l.strip() for l in f
                           if l.strip() and not l.startswith('#')
                           and l.strip().endswith('.root')]
        if args.max_files > 0:
            input_files = input_files[:args.max_files]
        print(f"Found {len(input_files)} ROOT files")
    else:
        print("ERROR: input must be .root or .txt"); return

    missing = [f for f in input_files if not os.path.exists(f)]
    if missing:
        print(f"ERROR: {len(missing)} files missing (first: {missing[0]})"); return

    all_data, chunk_idx = process_files(
        input_files, args.output,
        args.num_events, args.events_per_file,
        args.num_workers, args.events_per_pkl,
        use_superclustering=args.use_superclustering)

    if not all_data and chunk_idx == 0:
        print("ERROR: no events processed!"); return

    if args.events_per_pkl > 0:
        if all_data:
            base = args.output.replace('.pkl', '')
            out_path = f"{base}_{chunk_idx:04d}.pkl"
            print(f"\nSaving remaining {len(all_data)} events to {out_path} ...")
            with open(out_path, 'wb') as f:
                pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            chunk_idx += 1
        print(f"\nDone! Total {chunk_idx} PKL files saved.")
    else:
        print(f"\nSaving {len(all_data)} events to {args.output} ...")
        with open(args.output, 'wb') as f:
            pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if all_data:
        total_elements = sum(len(e['Xelem']) for e in all_data)
        total_tracks   = sum(np.sum(e['Xelem']['typ'] == 1) for e in all_data)
        total_gsf      = sum(np.sum(e['Xelem']['typ'] == 2) for e in all_data)
        total_tracksterlinks_ts = sum(np.sum(e['Xelem']['typ'] == 3) for e in all_data)
        total_supercluster_em   = sum(np.sum(e['Xelem']['typ'] == 4) for e in all_data)
        total_cp    = sum(np.sum(e['ytarget']['pid'] > 0) for e in all_data)
        genmet_vals = np.array([e['genmet'][0] for e in all_data])
        n = len(all_data)
        print(f"\nDone!  Events: {n}  |  Files: {len(input_files)}")
        print(f"  avg elements/event:              {total_elements/n:.1f}")
        print(f"  avg tracks/event:                {total_tracks/n:.1f}")
        print(f"  avg GSF tracks/event:             {total_gsf/n:.1f}")
        print(f"  avg CaloParticles/ev:             {total_cp/n:.1f}")
        print(f"  genMET mean={genmet_vals.mean():.2f}  "
              f"median={np.median(genmet_vals):.2f}  "
              f"max={genmet_vals.max():.2f} GeV")
        print("\n  Matching: gen->simCand (one gen -> many simCands)")
        print(f"  dR thresholds: EM={DR_MAX_EM}  CHAD={DR_MAX_CHAD}  "
              f"NHAD={DR_MAX_NHAD}  MU={DR_MAX_MU}  other={DR_MAX_OTHER}")


if __name__ == "__main__":
    main()