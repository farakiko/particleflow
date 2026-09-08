#!/usr/bin/env python
"""
Run3-paper-style target postprocessing for the CMS Phase-2 TICL NanoAOD.

Philosophy (mirrors mlpf/data/cms/postprocessing2.py):
  - target = truth particles (SimTICLCandidates, one per CaloParticle) with TRUTH energy
  - each truth particle assigned to ONE primary input element:
        charged (has trackIdx)  -> its GeneralTrack
        neutral (or no track)   -> its highest-shared-energy CLUE3DHigh trackster
    (assoc: SimCP2ticlTrackstersCLUE3DHighByHits; candidate index == SimCP index [VERIFIED])
  - one target particle per element; if several truth particles share an element, merge them
  - NO fragmentation of a particle across tracksters (unlike the SimTICLCandidate-native flow)

Elements: all GeneralTrack (typ=1) + all ticlTrackstersCLUE3DHigh (typ=4).
Gen jets: HGCalGenPart (all status==1), excluding neutrinos, anti-kt R=0.4, pt>3.
"""
import math, pickle, argparse
import numpy as np, awkward as ak, uproot, fastjet
from collections import defaultdict

CHARGED_PIDS = {11, 13, 211, 321}
NEUTRINOS    = {12, 14, 16}
SENTINEL     = -2147483648
JETDEF       = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
JET_PT_MIN   = 3.0

elem_branches = ["typ", "pt", "eta", "phi", "energy", "charge", "px", "py", "pz",
                 "em_energy", "nhits",
                 # trackster shape/timing (zero on tracks): depth, timing, PCA eigenvalues
                 "bary_z", "time", "timeerror", "ev1", "ev2", "ev3",
                 # track quality / muon-ID / vertex (zero on tracksters)
                 "muon_type", "muon_dt_hits", "muon_csc_hits",
                 "pterror", "etaerror", "phierror", "lambdaerror", "qoverperror",
                 "vx", "vy", "vz"]
particle_feature_order = ["pid", "charge", "pt", "eta", "sin_phi", "cos_phi",
                          "energy", "ispu", "jet_idx"]

TS  = "ticlTrackstersCLUE3DHigh"
S2R = "SimCP2ticlTrackstersCLUE3DHighByHits"   # sim CaloParticle -> reco trackster

BRANCHES = [
    "SimTICLCandidates_pdgID", "SimTICLCandidates_charge", "SimTICLCandidates_pt",
    "SimTICLCandidates_eta", "SimTICLCandidates_phi", "SimTICLCandidates_energy",
    "SimTICLCandidates_raw_energy", "SimTICLCandidates_trackIdx", "SimTICLCandidates_isPU",
    "GeneralTrack_pt", "GeneralTrack_eta", "GeneralTrack_phi", "GeneralTrack_p",
    "GeneralTrack_charge", "GeneralTrack_nhits",
    "GeneralTrack_muon_type", "GeneralTrack_muon_dt_hits", "GeneralTrack_muon_csc_hits",
    "GeneralTrack_ptErr", "GeneralTrack_etaErr", "GeneralTrack_phiErr",
    "GeneralTrack_lambdaErr", "GeneralTrack_qoverpErr",
    "GeneralTrack_vx", "GeneralTrack_vy", "GeneralTrack_vz",
    f"{TS}_raw_pt", f"{TS}_barycenter_eta", f"{TS}_barycenter_phi", f"{TS}_barycenter_z",
    f"{TS}_raw_energy", f"{TS}_regressed_energy", f"{TS}_raw_em_energy",
    f"{TS}_nticlTrackstersCLUE3DHighvertices", f"{TS}_time", f"{TS}_timeError",
    f"{TS}_EV1", f"{TS}_EV2", f"{TS}_EV3",
    f"{S2R}_n{S2R}Links", f"{S2R}Links_index", f"{S2R}Links_sharedEnergy",
    "HGCalGenPart_pdgId", "HGCalGenPart_status", "HGCalGenPart_pt",
    "HGCalGenPart_eta", "HGCalGenPart_phi", "HGCalGenPart_energy",
    "TICLCandidates_pdgID", "TICLCandidates_charge", "TICLCandidates_pt",
    "TICLCandidates_eta", "TICLCandidates_phi", "TICLCandidates_energy",
]
ycand_order = ["pid", "charge", "pt", "eta", "phi", "energy"]

def cluster_jets(pt, eta, phi, energy):
    """anti-kt R=0.4 jets from (pt,eta,phi,energy) arrays -> Nx4 (pt,eta,phi,energy)."""
    if len(pt) == 0:
        return np.zeros((0, 4), np.float32)
    px = pt*np.cos(phi); py = pt*np.sin(phi)
    pz = pt*np.sinh(np.where(np.abs(eta) < 10, eta, 0.0))
    pjs = [fastjet.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(energy[i]))
           for i in range(len(pt))]
    jets = fastjet.ClusterSequence(pjs, JETDEF).inclusive_jets(ptmin=JET_PT_MIN)
    return (np.array([[j.pt(), j.eta(), j.phi(), j.e()] for j in jets], np.float32)
            if jets else np.zeros((0, 4), np.float32))

def process_event(E, iev):
    g = lambda b: ak.to_numpy(E[b][iev])

    # ---- input elements: tracks (typ 1) then tracksters (typ 4) ----
    tpt, teta, tphi = g("GeneralTrack_pt"), g("GeneralTrack_eta"), g("GeneralTrack_phi")
    tp, tchg, tnh   = g("GeneralTrack_p"), g("GeneralTrack_charge"), g("GeneralTrack_nhits")
    tmut, tmdt, tmcsc = g("GeneralTrack_muon_type"), g("GeneralTrack_muon_dt_hits"), g("GeneralTrack_muon_csc_hits")
    tpte, tetae, tphie = g("GeneralTrack_ptErr"), g("GeneralTrack_etaErr"), g("GeneralTrack_phiErr")
    tlame, tqpe = g("GeneralTrack_lambdaErr"), g("GeneralTrack_qoverpErr")
    tvx, tvy, tvz = g("GeneralTrack_vx"), g("GeneralTrack_vy"), g("GeneralTrack_vz")
    n_trk = len(tpt)
    spt  = g(f"{TS}_raw_pt");  seta = g(f"{TS}_barycenter_eta"); sphi = g(f"{TS}_barycenter_phi")
    # regressed_energy is all-zeros in this NanoAOD (not filled for CLUE3DHigh) -> use raw_energy
    sreg = g(f"{TS}_raw_energy"); sem = g(f"{TS}_raw_em_energy")
    snh  = g(f"{TS}_nticlTrackstersCLUE3DHighvertices")
    sbz  = g(f"{TS}_barycenter_z"); stime = g(f"{TS}_time"); sterr = g(f"{TS}_timeError")
    sev1, sev2, sev3 = g(f"{TS}_EV1"), g(f"{TS}_EV2"), g(f"{TS}_EV3")
    n_ts = len(spt)
    n_el = n_trk + n_ts

    Xelem = np.recarray((n_el,), dtype=[(n, np.float32) for n in elem_branches]); Xelem.fill(0.0)
    # tracks
    Xelem["typ"][:n_trk] = 1
    Xelem["pt"][:n_trk] = tpt; Xelem["eta"][:n_trk] = teta; Xelem["phi"][:n_trk] = tphi
    Xelem["energy"][:n_trk] = tp; Xelem["charge"][:n_trk] = tchg; Xelem["nhits"][:n_trk] = tnh
    Xelem["px"][:n_trk] = tpt*np.cos(tphi); Xelem["py"][:n_trk] = tpt*np.sin(tphi)
    Xelem["pz"][:n_trk] = tpt*np.sinh(np.where(np.abs(teta) < 10, teta, 0.0))
    # track quality / muon-ID / vertex (trackster shape features stay 0 for tracks)
    Xelem["muon_type"][:n_trk] = tmut; Xelem["muon_dt_hits"][:n_trk] = tmdt; Xelem["muon_csc_hits"][:n_trk] = tmcsc
    Xelem["pterror"][:n_trk] = tpte; Xelem["etaerror"][:n_trk] = tetae; Xelem["phierror"][:n_trk] = tphie
    Xelem["lambdaerror"][:n_trk] = tlame; Xelem["qoverperror"][:n_trk] = tqpe
    Xelem["vx"][:n_trk] = tvx; Xelem["vy"][:n_trk] = tvy; Xelem["vz"][:n_trk] = tvz
    # tracksters
    th = 2.0*np.arctan(np.exp(-seta))
    Xelem["typ"][n_trk:] = 4
    Xelem["pt"][n_trk:] = spt; Xelem["eta"][n_trk:] = seta; Xelem["phi"][n_trk:] = sphi
    Xelem["energy"][n_trk:] = sreg; Xelem["em_energy"][n_trk:] = sem; Xelem["nhits"][n_trk:] = snh
    Xelem["px"][n_trk:] = spt*np.cos(sphi); Xelem["py"][n_trk:] = spt*np.sin(sphi)
    Xelem["pz"][n_trk:] = sreg*np.cos(th)
    # trackster shape/timing (track quality features stay 0 for tracksters)
    # bary_z -> |z| (longitudinal depth; endcap sign already carried by eta)
    # time == -99 is the "no valid timing" sentinel -> zero it (and its error) so it isn't a fake input
    tvalid = stime > -50
    Xelem["bary_z"][n_trk:] = np.abs(sbz)
    Xelem["time"][n_trk:] = np.where(tvalid, stime, 0.0)
    Xelem["timeerror"][n_trk:] = np.where(tvalid, sterr, 0.0)
    Xelem["ev1"][n_trk:] = sev1; Xelem["ev2"][n_trk:] = sev2; Xelem["ev3"][n_trk:] = sev3

    # ---- truth particles (SimTICLCandidates) ----
    pid = g("SimTICLCandidates_pdgID"); chg = g("SimTICLCandidates_charge")
    cpt = g("SimTICLCandidates_pt"); ceta = g("SimTICLCandidates_eta")
    cphi = g("SimTICLCandidates_phi"); cen = g("SimTICLCandidates_energy")
    craw = g("SimTICLCandidates_raw_energy")
    ctrk = g("SimTICLCandidates_trackIdx"); cpu = g("SimTICLCandidates_isPU")

    # per-particle (pt, energy): electrons use raw_energy (SimTICLCandidate energy/pt are
    # ~0.4x truth for e; raw_energy=calo deposit ~0.97x truth) [VERIFIED]. Others: energy.
    def pkin(i):
        if abs(int(pid[i])) == 11:
            th = 2.0*math.atan(math.exp(-ceta[i])); e = float(craw[i]); return e*math.sin(th), e
        return float(cpt[i]), float(cen[i])

    # associator SimCP->trackster (per-candidate, since candidate idx == SimCP idx)
    cnt = g(f"{S2R}_n{S2R}Links"); off = np.concatenate([[0], np.cumsum(cnt)]).astype(int)
    aidx = g(f"{S2R}Links_index"); ashe = g(f"{S2R}Links_sharedEnergy")

    elem_to_parts = defaultdict(list)
    for i in range(len(pid)):
        charged = (abs(int(pid[i])) in CHARGED_PIDS) and (int(ctrk[i]) != SENTINEL)
        if charged:
            ti = int(ctrk[i])
            if 0 <= ti < n_trk:
                elem_to_parts[ti].append(i)
        else:
            ii = aidx[off[i]:off[i+1]]; ss = ashe[off[i]:off[i+1]]
            if len(ss) and ss.max() > 0:
                best = int(ii[int(np.argmax(ss))])
                if 0 <= best < n_ts:
                    elem_to_parts[n_trk + best].append(i)

    ytarget = np.recarray((n_el,), dtype=[(n, np.float32) for n in particle_feature_order])
    ytarget.fill(0.0); ytarget["jet_idx"] = -1
    for e, parts in elem_to_parts.items():
        # merge (usually a single particle); pick pid/charge from highest-energy particle
        kin = {i: pkin(i) for i in parts}   # (pt, energy) per particle, e- corrected
        parts_sorted = sorted(parts, key=lambda i: kin[i][1], reverse=True)
        lead = parts_sorted[0]
        px = np.sum([kin[i][0]*np.cos(cphi[i]) for i in parts])
        py = np.sum([kin[i][0]*np.sin(cphi[i]) for i in parts])
        pz = np.sum([kin[i][0]*np.sinh(ceta[i]) if abs(ceta[i]) < 10 else 0.0 for i in parts])
        en = np.sum([kin[i][1] for i in parts])
        pt = math.hypot(px, py); phi = math.atan2(py, px)
        eta = np.arcsinh(pz/pt) if pt > 0 else 0.0
        ytarget["pid"][e] = abs(int(pid[lead])); ytarget["charge"][e] = chg[lead]
        ytarget["pt"][e] = pt; ytarget["eta"][e] = eta
        ytarget["sin_phi"][e] = math.sin(phi); ytarget["cos_phi"][e] = math.cos(phi)
        ytarget["energy"][e] = en
        ytarget["ispu"][e] = max(0.0, float(cpu[lead]))   # isPU in {0,-1}; -1(undet)->0

    # ---- jets ----
    vt = ytarget["pid"] != 0
    targetjet = cluster_jets(ytarget["pt"][vt], ytarget["eta"][vt],
                             np.arctan2(ytarget["sin_phi"][vt], ytarget["cos_phi"][vt]),
                             ytarget["energy"][vt])
    gpid = g("HGCalGenPart_pdgId"); gpt = g("HGCalGenPart_pt")
    geta = g("HGCalGenPart_eta"); gphi = g("HGCalGenPart_phi"); gen_e = g("HGCalGenPart_energy")
    gmask = ~np.isin(np.abs(gpid), list(NEUTRINOS))     # status is always 1 here
    genjet = cluster_jets(gpt[gmask], geta[gmask], gphi[gmask], gen_e[gmask])
    pythia = np.stack([np.abs(gpid[gmask]).astype(np.float32), gpt[gmask], geta[gmask],
                       gphi[gmask], gen_e[gmask]], axis=-1) if gmask.sum() else np.zeros((0, 5), np.float32)

    # ---- baseline reco: TICLCandidates (raw list + clustered jets) ----
    cpid = g("TICLCandidates_pdgID"); cchg = g("TICLCandidates_charge")
    ctpt = g("TICLCandidates_pt"); cteta = g("TICLCandidates_eta")
    ctphi = g("TICLCandidates_phi"); cte = g("TICLCandidates_energy")
    ycand = np.recarray((len(cpid),), dtype=[(n, np.float32) for n in ycand_order]); ycand.fill(0.0)
    if len(cpid):
        ycand["pid"] = np.abs(cpid); ycand["charge"] = cchg; ycand["pt"] = ctpt
        ycand["eta"] = cteta; ycand["phi"] = ctphi; ycand["energy"] = cte
    cok = ctpt > 0
    candjet = cluster_jets(ctpt[cok], cteta[cok], ctphi[cok], cte[cok])

    return {"Xelem": Xelem, "ytarget": ytarget, "ycand": ycand, "genjet": genjet,
            "targetjet": targetjet, "candjet": candjet, "pythia": pythia}

def process(infile, outfile, num_events=-1):
    print(f"opening {infile}")
    E = uproot.open(infile)["Events"].arrays(BRANCHES)
    n = len(E) if num_events < 0 else min(num_events, len(E))
    out = []
    for iev in range(n):
        out.append(process_event(E, iev))
        if (iev+1) % 50 == 0: print(f"  {iev+1}/{n}")
    with open(outfile, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    # quick summary
    ne = sum(len(d["Xelem"]) for d in out); npart = sum(int((d["ytarget"]["pid"] != 0).sum()) for d in out)
    print(f"saved {outfile}: {n} events, {ne} elements, {npart} target particles "
          f"({npart/n:.1f}/ev), targetjets {sum(len(d['targetjet']) for d in out)}, "
          f"genjets {sum(len(d['genjet']) for d in out)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-events", type=int, default=-1)
    a = ap.parse_args()
    process(a.input, a.output, a.num_events)

if __name__ == "__main__":
    main()
