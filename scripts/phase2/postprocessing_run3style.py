#!/usr/bin/env python
"""
Run3-paper-style target postprocessing for the CMS Phase-2 TICL NanoAOD.

Settled design (see docs/phase2_target_comparison.md), common to both calo variants:
  - target = truth particles (SimTICLCandidates) with TRUTH energy (e- use raw_energy)
  - charged (has trackIdx) -> its GeneralTrack ; neutral/photon -> its highest-shared-energy trackster
  - ONE target per particle (primary element); multiple particles on one element -> merged
  - NO gen-match filter, NO associator score cuts  (both discard real energy and degrade jet scale;
    verified: charged filter 0.97->0.89, neutral cut 0.974->0.954 jet response)

The ONE design axis left open (to settle by full-scale training) is the calo collection:
  --calo clue3d : ticlTrackstersCLUE3DHigh  (pre-linking raw tracksters; MLPF learns the linking)
  --calo links  : ticlTracksterLinks        (post-linking, CMSSW-merged tracksters)
Everything else is identical between the two -> a clean apples-to-apples comparison.

Elements: all GeneralTrack (typ=1) + all GSFTrack (typ=2) + all <calo> tracksters (typ=4).
Electrons prefer their GSF track (as the colleague's script does); other charged -> GeneralTrack.
Gen jets: HGCalGenPart (status==1), excluding neutrinos, anti-kt R=0.4, pt>3.
"""
import math, pickle, argparse
import numpy as np, awkward as ak, uproot, fastjet
from collections import defaultdict

CHARGED_PIDS = {11, 13, 211, 321}
NEUTRINOS    = {12, 14, 16}
SENTINEL     = -2147483648
JETDEF       = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
JET_PT_MIN   = 3.0
ENDCAP_LO, ENDCAP_HI = 1.5, 3.0   # HGCAL acceptance: target particles restricted to the endcap

# calo collection choice (the only thing that differs between the two variants)
CALO = {"clue3d": "ticlTrackstersCLUE3DHigh", "links": "ticlTracksterLinks"}

elem_branches = ["typ", "pt", "eta", "phi", "energy", "charge", "px", "py", "pz",
                 "em_energy", "nhits",
                 # trackster shape/timing (zero on tracks): depth, timing, PCA eigenvalues
                 "bary_z", "time", "timeerror", "ev1", "ev2", "ev3",
                 # track quality / muon-ID / vertex (zero on tracksters)
                 "muon_type", "muon_dt_hits", "muon_csc_hits", "gsf_type",
                 "pterror", "etaerror", "phierror", "lambdaerror", "qoverperror",
                 "vx", "vy", "vz",
                 # track-density around trackster (zero on tracks): colleague's proximity features
                 "min_dR_track", "near_track_pt", "sum_pt_dR10",
                 "n_trk_dR01", "n_trk_dR02", "n_trk_dR03", "n_trk_dR04", "n_trk_dR05"]
particle_feature_order = ["pid", "charge", "pt", "eta", "sin_phi", "cos_phi",
                          "energy", "ispu", "jet_idx"]
ycand_order = ["pid", "charge", "pt", "eta", "phi", "energy"]


def branches_for(ts):
    """Branch list for a given calo collection name (ts)."""
    s2r = f"SimCP2{ts}ByHits"
    return [
        "SimTICLCandidates_pdgID", "SimTICLCandidates_charge", "SimTICLCandidates_pt",
        "SimTICLCandidates_eta", "SimTICLCandidates_phi", "SimTICLCandidates_energy",
        "SimTICLCandidates_raw_energy", "SimTICLCandidates_trackIdx", "SimTICLCandidates_isPU",
        "GeneralTrack_pt", "GeneralTrack_eta", "GeneralTrack_phi", "GeneralTrack_p",
        "GeneralTrack_charge", "GeneralTrack_nhits",
        "GeneralTrack_hgcal_eta", "GeneralTrack_hgcal_phi",
        "GeneralTrack_muon_type", "GeneralTrack_muon_dt_hits", "GeneralTrack_muon_csc_hits",
        "GeneralTrack_ptErr", "GeneralTrack_etaErr", "GeneralTrack_phiErr",
        "GeneralTrack_lambdaErr", "GeneralTrack_qoverpErr",
        "GeneralTrack_vx", "GeneralTrack_vy", "GeneralTrack_vz",
        "GSFTrack_ptMode", "GSFTrack_pMode", "GSFTrack_etaMode", "GSFTrack_phiMode",
        "GSFTrack_pxMode", "GSFTrack_pyMode", "GSFTrack_pzMode",
        "GSFTrack_charge", "GSFTrack_nhits",
        "GSFTrack_ptModeError", "GSFTrack_etaModeError", "GSFTrack_phiModeError",
        "GSFTrack_lambdaModeError", "GSFTrack_qoverpModeError",
        "GSFTrack_vx", "GSFTrack_vy", "GSFTrack_vz",
        "SimTICLCandidates_nGsfTrackIdxs", "SimTICLCandidatesGsfTrackIdxs_trackIndex",
        f"{ts}_raw_pt", f"{ts}_barycenter_eta", f"{ts}_barycenter_phi", f"{ts}_barycenter_z",
        f"{ts}_raw_energy", f"{ts}_raw_em_energy", f"{ts}_n{ts}vertices",
        f"{ts}_time", f"{ts}_timeError", f"{ts}_EV1", f"{ts}_EV2", f"{ts}_EV3",
        f"{s2r}_n{s2r}Links", f"{s2r}Links_index", f"{s2r}Links_sharedEnergy",
        f"{s2r}Links_score",
        f"Reco{ts}2SimCPByHits_nReco{ts}2SimCPByHitsLinks", f"Reco{ts}2SimCPByHitsLinks_score",
        "HGCalGenPart_pdgId", "HGCalGenPart_status", "HGCalGenPart_pt",
        "HGCalGenPart_eta", "HGCalGenPart_phi", "HGCalGenPart_energy",
        "TICLCandidates_pdgID", "TICLCandidates_charge", "TICLCandidates_pt",
        "TICLCandidates_eta", "TICLCandidates_phi", "TICLCandidates_energy",
    ]


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


def process_event(E, iev, ts, neutral_split="argmax", frag_select="score",
                  frag_min_share=0.05, frag_r_max=0.6, frag_s_max=0.9, acceptance="simcand"):
    s2r = f"SimCP2{ts}ByHits"
    g = lambda b: ak.to_numpy(E[b][iev])

    # ---- input elements: tracks (typ 1) then tracksters (typ 4) ----
    tpt, teta, tphi = g("GeneralTrack_pt"), g("GeneralTrack_eta"), g("GeneralTrack_phi")
    tp, tchg, tnh   = g("GeneralTrack_p"), g("GeneralTrack_charge"), g("GeneralTrack_nhits")
    tke, tkp        = g("GeneralTrack_hgcal_eta"), g("GeneralTrack_hgcal_phi")
    tmut, tmdt, tmcsc = g("GeneralTrack_muon_type"), g("GeneralTrack_muon_dt_hits"), g("GeneralTrack_muon_csc_hits")
    tpte, tetae, tphie = g("GeneralTrack_ptErr"), g("GeneralTrack_etaErr"), g("GeneralTrack_phiErr")
    tlame, tqpe = g("GeneralTrack_lambdaErr"), g("GeneralTrack_qoverpErr")
    tvx, tvy, tvz = g("GeneralTrack_vx"), g("GeneralTrack_vy"), g("GeneralTrack_vz")
    n_trk = len(tpt)
    spt  = g(f"{ts}_raw_pt");  seta = g(f"{ts}_barycenter_eta"); sphi = g(f"{ts}_barycenter_phi")
    # regressed_energy is all-zeros in this NanoAOD -> use raw_energy as the trackster element energy
    sreg = g(f"{ts}_raw_energy"); sem = g(f"{ts}_raw_em_energy")
    snh  = g(f"{ts}_n{ts}vertices")
    sbz  = g(f"{ts}_barycenter_z"); stime = g(f"{ts}_time"); sterr = g(f"{ts}_timeError")
    sev1, sev2, sev3 = g(f"{ts}_EV1"), g(f"{ts}_EV2"), g(f"{ts}_EV3")
    n_ts = len(spt)
    # ---- GSF tracks (typ 2): electron-momentum elements, appended after tracksters ----
    gpt_m = g("GSFTrack_ptMode"); gpm = g("GSFTrack_pMode")
    geta_m = g("GSFTrack_etaMode"); gphi_m = g("GSFTrack_phiMode")
    gpx = g("GSFTrack_pxMode"); gpy = g("GSFTrack_pyMode"); gpz = g("GSFTrack_pzMode")
    gchg = g("GSFTrack_charge"); gnh = g("GSFTrack_nhits")
    gpte = g("GSFTrack_ptModeError"); getae = g("GSFTrack_etaModeError"); gphie = g("GSFTrack_phiModeError")
    glame = g("GSFTrack_lambdaModeError"); gqpe = g("GSFTrack_qoverpModeError")
    gvx = g("GSFTrack_vx"); gvy = g("GSFTrack_vy"); gvz = g("GSFTrack_vz")
    n_gsf = len(gpt_m); gbase = n_trk + n_ts; n_el = n_trk + n_ts + n_gsf

    Xelem = np.recarray((n_el,), dtype=[(n, np.float32) for n in elem_branches]); Xelem.fill(0.0)
    # tracks
    Xelem["typ"][:n_trk] = 1
    Xelem["pt"][:n_trk] = tpt; Xelem["eta"][:n_trk] = teta; Xelem["phi"][:n_trk] = tphi
    Xelem["energy"][:n_trk] = tp; Xelem["charge"][:n_trk] = tchg; Xelem["nhits"][:n_trk] = tnh
    Xelem["px"][:n_trk] = tpt*np.cos(tphi); Xelem["py"][:n_trk] = tpt*np.sin(tphi)
    Xelem["pz"][:n_trk] = tpt*np.sinh(np.where(np.abs(teta) < 10, teta, 0.0))
    Xelem["muon_type"][:n_trk] = tmut; Xelem["muon_dt_hits"][:n_trk] = tmdt; Xelem["muon_csc_hits"][:n_trk] = tmcsc
    Xelem["pterror"][:n_trk] = tpte; Xelem["etaerror"][:n_trk] = tetae; Xelem["phierror"][:n_trk] = tphie
    Xelem["lambdaerror"][:n_trk] = tlame; Xelem["qoverperror"][:n_trk] = tqpe
    Xelem["vx"][:n_trk] = tvx; Xelem["vy"][:n_trk] = tvy; Xelem["vz"][:n_trk] = tvz
    # tracksters
    th = 2.0*np.arctan(np.exp(-seta))
    Xelem["typ"][n_trk:gbase] = 4
    Xelem["pt"][n_trk:gbase] = spt; Xelem["eta"][n_trk:gbase] = seta; Xelem["phi"][n_trk:gbase] = sphi
    Xelem["energy"][n_trk:gbase] = sreg; Xelem["em_energy"][n_trk:gbase] = sem; Xelem["nhits"][n_trk:gbase] = snh
    Xelem["px"][n_trk:gbase] = spt*np.cos(sphi); Xelem["py"][n_trk:gbase] = spt*np.sin(sphi)
    Xelem["pz"][n_trk:gbase] = sreg*np.cos(th)
    # bary_z -> |z| (depth; endcap sign is in eta); time==-99 sentinel -> 0 (with its error)
    tvalid = stime > -50
    Xelem["bary_z"][n_trk:gbase] = np.abs(sbz)
    Xelem["time"][n_trk:gbase] = np.where(tvalid, stime, 0.0)
    Xelem["timeerror"][n_trk:gbase] = np.where(tvalid, sterr, 0.0)
    Xelem["ev1"][n_trk:gbase] = sev1; Xelem["ev2"][n_trk:gbase] = sev2; Xelem["ev3"][n_trk:gbase] = sev3
    # track-density around each trackster (tracks with pt>=1, at the HGCAL surface)
    if n_ts:
        good = tpt >= 1.0
        gke, gkp, gkpt = tke[good], tkp[good], tpt[good]
        if len(gke):
            dphi = np.arctan2(np.sin(gkp[None, :] - sphi[:, None]), np.cos(gkp[None, :] - sphi[:, None]))
            dR = np.hypot(gke[None, :] - seta[:, None], dphi)            # (n_ts, n_good)
            jmin = dR.argmin(1)
            Xelem["min_dR_track"][n_trk:gbase] = dR[np.arange(n_ts), jmin]
            Xelem["near_track_pt"][n_trk:gbase] = gkpt[jmin]
            Xelem["sum_pt_dR10"][n_trk:gbase] = (gkpt[None, :] * (dR < 0.10)).sum(1)
            for col, c in [("n_trk_dR01", 0.01), ("n_trk_dR02", 0.02), ("n_trk_dR03", 0.03),
                           ("n_trk_dR04", 0.04), ("n_trk_dR05", 0.05)]:
                Xelem[col][n_trk:gbase] = (dR < c).sum(1)
        else:
            Xelem["min_dR_track"][n_trk:gbase] = 99.0
    # else: no tracksters; min_dR_track stays 0 (no trackster rows)

    # GSF tracks (typ 2, gsf_type=1): kinematics from *Mode; muon-ID / shape / density stay 0
    Xelem["typ"][gbase:] = 2; Xelem["gsf_type"][gbase:] = 1.0
    Xelem["pt"][gbase:] = gpt_m; Xelem["eta"][gbase:] = geta_m; Xelem["phi"][gbase:] = gphi_m
    Xelem["energy"][gbase:] = gpm; Xelem["charge"][gbase:] = gchg; Xelem["nhits"][gbase:] = gnh
    Xelem["px"][gbase:] = gpx; Xelem["py"][gbase:] = gpy; Xelem["pz"][gbase:] = gpz
    Xelem["pterror"][gbase:] = gpte; Xelem["etaerror"][gbase:] = getae; Xelem["phierror"][gbase:] = gphie
    Xelem["lambdaerror"][gbase:] = glame; Xelem["qoverperror"][gbase:] = gqpe
    Xelem["vx"][gbase:] = gvx; Xelem["vy"][gbase:] = gvy; Xelem["vz"][gbase:] = gvz

    # ---- truth particles (SimTICLCandidates) ----
    pid = g("SimTICLCandidates_pdgID"); chg = g("SimTICLCandidates_charge")
    cpt = g("SimTICLCandidates_pt"); ceta = g("SimTICLCandidates_eta")
    cphi = g("SimTICLCandidates_phi"); cen = g("SimTICLCandidates_energy")
    craw = g("SimTICLCandidates_raw_energy")
    ctrk = g("SimTICLCandidates_trackIdx"); cpu = g("SimTICLCandidates_isPU")

    def pkin(i):  # (pt, energy); electrons use raw_energy (their 'energy'/'pt' are ~0.4x truth) [VERIFIED]
        if abs(int(pid[i])) == 11:
            th = 2.0*math.atan(math.exp(-ceta[i])); e = float(craw[i]); return e*math.sin(th), e
        return float(cpt[i]), float(cen[i])

    cnt = g(f"{s2r}_n{s2r}Links"); off = np.concatenate([[0], np.cumsum(cnt)]).astype(int)
    aidx = g(f"{s2r}Links_index"); ashe = g(f"{s2r}Links_sharedEnergy")
    asco = g(f"{s2r}Links_score")          # simToReco score, parallel to ashe (0=perfect, 1=no match)
    r2s = f"Reco{ts}2SimCPByHits"
    rcnt = g(f"{r2s}_n{r2s}Links"); rsco = g(f"{r2s}Links_score")
    roff = np.concatenate([[0], np.cumsum(rcnt)]).astype(int)
    # per-trackster leading recoToSim score (moanwar takes entry [0]; 1.0 = unassociated)
    ts_rscore = np.ones(n_ts, np.float32)
    has_r = rcnt > 0
    ts_rscore[has_r] = rsco[roff[:-1][has_r]]
    # electron -> GSF-track association (ragged), the same branch his script uses
    gcnt = g("SimTICLCandidates_nGsfTrackIdxs"); goff = np.concatenate([[0], np.cumsum(gcnt)]).astype(int)
    gflat = g("SimTICLCandidatesGsfTrackIdxs_trackIndex")

    def best_trackster(i):  # argmax-sharedEnergy trackster element for particle i, or None
        ii = aidx[off[i]:off[i+1]]; ss = ashe[off[i]:off[i+1]]
        if len(ss) and ss.max() > 0:
            b = int(ii[int(np.argmax(ss))])
            if 0 <= b < n_ts:
                return n_trk + b
        return None

    def trackster_fragments(i):
        """All associated tracksters of particle i with shared-energy weights
        (moanwar-style fragmentation, NO gen filter). Fragment selection:
          score (default, synchronized with moanwar): keep a trackster iff
            r_score<=frag_r_max AND s_score<=frag_s_max (his reject: r>0.6 or s>0.9);
          share: keep tracksters carrying >=frag_min_share of the total shared energy.
        In BOTH modes the weights are renormalized over the KEPT fragments — the
        redistribution happens AFTER the filtering, so the particle's full truth
        energy is conserved as long as >=1 fragment survives (else it is dropped)."""
        ii = aidx[off[i]:off[i+1]]; ss = ashe[off[i]:off[i+1]]; sc = asco[off[i]:off[i+1]]
        ok = (ss > 0) & (ii >= 0) & (ii < n_ts)
        if not ok.any():
            return []
        ii, ss, sc = ii[ok], ss[ok], sc[ok]
        if frag_select == "score":
            keep = (sc <= frag_s_max) & (ts_rscore[ii.astype(int)] <= frag_r_max)
            if not keep.any():
                return []
        else:
            w0 = ss / ss.sum()
            keep = w0 >= frag_min_share
            if not keep.any():
                keep = w0 == w0.max()
        ii, ss = ii[keep], ss[keep]
        w = ss / ss.sum()
        return [(n_trk + int(t), float(x)) for t, x in zip(ii, w)]

    def neutral_elems(i):
        # argmax: whole particle on its best trackster; fragment: split across tracksters
        if neutral_split == "fragment":
            return trackster_fragments(i)
        e = best_trackster(i)
        return [(e, 1.0)] if e is not None else []

    def track_in_hgcal(ti):
        # detector-landing acceptance for track anchors: the track EXTRAPOLATED to the
        # HGCAL surface must land in the endcap window (moanwar's criterion, minus gen match)
        return ENDCAP_LO <= abs(float(tke[ti])) <= ENDCAP_HI

    def in_acceptance(i, ti=None):
        """simcand mode: truth direction in 1.5<|eta|<3 (rigid, momentum-level).
        anchor mode: decided by where the ANCHOR lands — tracks via their HGCAL-surface
        extrapolation; trackster-anchored particles are in by construction (tracksters
        only exist in HGCAL), so no extra test there."""
        if acceptance == "anchor":
            if ti is not None:
                return track_in_hgcal(ti)
            return True  # trackster-anchored: element existence IS the acceptance
        return ENDCAP_LO <= abs(float(ceta[i])) <= ENDCAP_HI

    elem_to_parts = defaultdict(list)   # elem -> [(particle index, energy weight)]
    for i in range(len(pid)):
        apid = abs(int(pid[i]))
        if apid == 11:
            # electron: prefer its GSF track (as his script does), then general track, then trackster
            gids = gflat[goff[i]:goff[i+1]]
            gid = next((int(x) for x in gids if 0 <= int(x) < n_gsf), None)
            ti = int(ctrk[i])
            has_trk = ti != SENTINEL and 0 <= ti < n_trk
            if gid is not None:
                # GSF has no stored HGCAL extrapolation -> use the general track's when
                # available; a GSF-only electron counts as landed (HGCAL-seeded object)
                if in_acceptance(i, ti if has_trk else None):
                    elem_to_parts[gbase + gid].append((i, 1.0))
                continue
            if has_trk:
                if in_acceptance(i, ti):
                    elem_to_parts[ti].append((i, 1.0))
                continue
            if in_acceptance(i):
                for e, w in neutral_elems(i):
                    elem_to_parts[e].append((i, w))
            continue
        charged = (apid in CHARGED_PIDS) and (int(ctrk[i]) != SENTINEL)
        if charged:
            ti = int(ctrk[i])
            if 0 <= ti < n_trk and in_acceptance(i, ti):
                elem_to_parts[ti].append((i, 1.0))
        else:
            if in_acceptance(i):
                for e, w in neutral_elems(i):
                    elem_to_parts[e].append((i, w))

    ytarget = np.recarray((n_el,), dtype=[(n, np.float32) for n in particle_feature_order])
    ytarget.fill(0.0); ytarget["jet_idx"] = -1
    for e, parts in elem_to_parts.items():
        # each contribution = weight * particle kinematics (weight=1 except fragments);
        # fragments keep the PARTICLE direction so they re-sum to the particle in jets
        kin = {}
        for i, w in parts:
            p, en_ = pkin(i)
            kin[i] = (w * p, w * en_)
        lead = sorted(parts, key=lambda iw: kin[iw[0]][1], reverse=True)[0][0]
        px = np.sum([kin[i][0]*np.cos(cphi[i]) for i, _ in parts])
        py = np.sum([kin[i][0]*np.sin(cphi[i]) for i, _ in parts])
        pz = np.sum([kin[i][0]*np.sinh(ceta[i]) if abs(ceta[i]) < 10 else 0.0 for i, _ in parts])
        en = np.sum([kin[i][1] for i, _ in parts])
        pt = math.hypot(px, py); phi = math.atan2(py, px)
        eta = np.arcsinh(pz/pt) if pt > 0 else 0.0
        ytarget["pid"][e] = abs(int(pid[lead])); ytarget["charge"][e] = chg[lead]
        ytarget["pt"][e] = pt; ytarget["eta"][e] = eta
        ytarget["sin_phi"][e] = math.sin(phi); ytarget["cos_phi"][e] = math.cos(phi)
        ytarget["energy"][e] = en
        ytarget["ispu"][e] = max(0.0, float(cpu[lead]))

    # ---- jets ----
    vt = ytarget["pid"] != 0
    targetjet = cluster_jets(ytarget["pt"][vt], ytarget["eta"][vt],
                             np.arctan2(ytarget["sin_phi"][vt], ytarget["cos_phi"][vt]),
                             ytarget["energy"][vt])
    gpid = g("HGCalGenPart_pdgId"); gpt = g("HGCalGenPart_pt")
    geta = g("HGCalGenPart_eta"); gphi = g("HGCalGenPart_phi"); gen_e = g("HGCalGenPart_energy")
    gmask = ~np.isin(np.abs(gpid), list(NEUTRINOS))
    genjet = cluster_jets(gpt[gmask], geta[gmask], gphi[gmask], gen_e[gmask])
    pythia = np.stack([np.abs(gpid[gmask]).astype(np.float32), gpt[gmask], geta[gmask],
                       gphi[gmask], gen_e[gmask]], axis=-1) if gmask.sum() else np.zeros((0, 5), np.float32)

    # ---- baseline reco: TICLCandidates ----
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


def process(infile, outfile, num_events=-1, calo="clue3d", neutral_split="argmax",
            frag_select="score", frag_min_share=0.05, frag_r_max=0.6, frag_s_max=0.9,
            acceptance="simcand"):
    ts = CALO[calo]
    print(f"opening {infile}  (calo={calo} -> {ts}, neutral_split={neutral_split}"
          + (f", frag_select={frag_select}" if neutral_split == "fragment" else "")
          + f", acceptance={acceptance})")
    try:
        tree = uproot.open(infile)["Events"]
    except uproot.KeyInFileError:
        print("  WARNING: no 'Events' tree (empty/corrupt file) -> skipping, no output written")
        return
    E = tree.arrays(branches_for(ts))
    n = len(E) if num_events < 0 else min(num_events, len(E))
    if n == 0:
        print("  WARNING: 0 events -> skipping, no output written")
        return
    out = []
    for iev in range(n):
        out.append(process_event(E, iev, ts, neutral_split, frag_select, frag_min_share, frag_r_max, frag_s_max, acceptance))
        if (iev+1) % 50 == 0: print(f"  {iev+1}/{n}")
    with open(outfile, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    ne = sum(len(d["Xelem"]) for d in out); npart = sum(int((d["ytarget"]["pid"] != 0).sum()) for d in out)
    print(f"saved {outfile}: {n} events, {ne} elements, {npart} target particles "
          f"({npart/n:.1f}/ev), targetjets {sum(len(d['targetjet']) for d in out)}, "
          f"genjets {sum(len(d['genjet']) for d in out)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-events", type=int, default=-1)
    ap.add_argument("--calo", choices=list(CALO), default="clue3d",
                    help="calo collection: clue3d (pre-linking) or links (CMSSW-merged)")
    ap.add_argument("--neutral-split", choices=["argmax", "fragment"], default="argmax",
                    help="neutral->trackster target: argmax (one per particle, run3-style) or "
                         "fragment (moanwar-style proportional split, no gen filter; truth energy "
                         "renormalized over the fragments that pass the selection)")
    ap.add_argument("--frag-select", choices=["score", "share"], default="score",
                    help="fragment selection: score (moanwar-synchronized r/s cuts) or share (energy floor)")
    ap.add_argument("--frag-min-share", type=float, default=0.05,
                    help="share mode: drop tracksters below this fraction of the total shared energy")
    ap.add_argument("--frag-r-max", type=float, default=0.6,
                    help="score mode: keep fragment iff trackster recoToSim score <= this (moanwar 0.6)")
    ap.add_argument("--frag-s-max", type=float, default=0.9,
                    help="score mode: keep fragment iff simToReco score <= this (moanwar 0.9)")
    ap.add_argument("--acceptance", choices=["simcand", "anchor"], default="simcand",
                    help="endcap acceptance: simcand (truth direction in 1.5<|eta|<3) or anchor "
                         "(detector-landing: track HGCAL extrapolation / trackster existence)")
    a = ap.parse_args()
    process(a.input, a.output, a.num_events, a.calo, a.neutral_split,
            a.frag_select, a.frag_min_share, a.frag_r_max, a.frag_s_max, a.acceptance)


if __name__ == "__main__":
    main()
