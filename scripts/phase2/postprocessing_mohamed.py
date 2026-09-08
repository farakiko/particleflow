#!/usr/bin/env python
"""
Port of the colleague's (Mohamed's) target logic
(.staging/cms/postprocessing_ticl_ttbar_nopu.py) to the CMS Phase-2 NanoAOD.

Faithfully reproduces his SimTICLCandidate-based target:
  - gen->simcand matching filter (keep simcands matched to a stable gen in HGCAL)
  - connections via TICL associators with score cuts (r_score>0.6 or s_score>0.9 drop)
  - split_caloparticles: neutrals FRAGMENT into one target per matched trackster
  - element types: track=1, EM trackster=2, HAD trackster=3, GSF=4
  - one target per input element via find_representative_elements

Calo input = ticlTracksterLinks (linked tracksters), as in his script (no superclustering).
Truth energy uses SimTICLCandidates 'energy' (NanoAOD has no 'regressed_energy'); electrons use 'raw_energy'.
"""
import math, pickle, argparse
import numpy as np, awkward as ak, uproot, fastjet
from collections import defaultdict

NEUTRINOS = {12, 14, 16}
CHARGED   = {11, 13, 211, 321}
SENT      = -2147483648
JETDEF    = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
JET_PT_MIN = 3.0
GEN_PT_MIN = 1.0
DR_MAX = {tuple(): 0.1}  # pid-dependent; EM/CHAD/NHAD/MU=0.1, else 0.001
R_CUT, S_CUT = 0.6, 0.9
TS = "ticlTracksterLinks"

elem_branches = ["typ", "pt", "eta", "phi", "energy", "charge", "px", "py", "pz", "em_energy"]
particle_feature_order = ["pid", "charge", "pt", "eta", "sin_phi", "cos_phi", "energy", "ispu", "jet_idx"]

def dr_thresh(apid):
    return 0.1 if apid in (11, 22, 211, 321, 130, 310, 13) else 0.001

def cluster_jets(pt, eta, phi, energy):
    if len(pt) == 0:
        return np.zeros((0, 4), np.float32)
    px = pt*np.cos(phi); py = pt*np.sin(phi)
    pz = pt*np.sinh(np.where(np.abs(eta) < 10, eta, 0.0))
    pjs = [fastjet.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(energy[i])) for i in range(len(pt))]
    jets = fastjet.ClusterSequence(pjs, JETDEF).inclusive_jets(ptmin=JET_PT_MIN)
    return (np.array([[j.pt(), j.eta(), j.phi(), j.e()] for j in jets], np.float32)
            if jets else np.zeros((0, 4), np.float32))

BRANCHES = [
    f"{TS}_raw_energy", f"{TS}_regressed_energy", f"{TS}_raw_pt", f"{TS}_raw_em_energy",
    f"{TS}_barycenter_eta", f"{TS}_barycenter_phi", f"{TS}_barycenter_z",
    "SimTICLCandidates_pdgID", "SimTICLCandidates_pt", "SimTICLCandidates_eta",
    "SimTICLCandidates_phi", "SimTICLCandidates_energy", "SimTICLCandidates_raw_energy",
    "SimTICLCandidates_isPU", "SimTICLCandidates_nTrackIdxs", "SimTICLCandidatesTrackIdxs_trackIndex",
    "GeneralTrack_pt", "GeneralTrack_p", "GeneralTrack_eta", "GeneralTrack_phi",
    "GeneralTrack_hgcal_eta", "GeneralTrack_hgcal_phi", "GeneralTrack_charge",
    # simToReco (SimCP -> tracksterLinks) and recoToSim (tracksterLinks -> SimCP)
    "SimCP2ticlTracksterLinksByHits_nSimCP2ticlTracksterLinksByHitsLinks",
    "SimCP2ticlTracksterLinksByHitsLinks_index", "SimCP2ticlTracksterLinksByHitsLinks_score",
    "SimCP2ticlTracksterLinksByHitsLinks_sharedEnergy",
    "RecoticlTracksterLinks2SimCPByHits_nRecoticlTracksterLinks2SimCPByHitsLinks",
    "RecoticlTracksterLinks2SimCPByHitsLinks_score",
    "HGCalGenPart_pdgId", "HGCalGenPart_status", "HGCalGenPart_pt",
    "HGCalGenPart_eta", "HGCalGenPart_phi", "HGCalGenPart_energy",
]

def build_gen_to_simcand(g, ev, iev):
    gpid = g("HGCalGenPart_pdgId"); gst = g("HGCalGenPart_status")
    geta = g("HGCalGenPart_eta"); gphi = g("HGCalGenPart_phi"); gpt = g("HGCalGenPart_pt")
    spid = g("SimTICLCandidates_pdgID"); seta = g("SimTICLCandidates_eta"); sphi = g("SimTICLCandidates_phi")
    strk_n = g("SimTICLCandidates_nTrackIdxs"); strk_flat = g("SimTICLCandidatesTrackIdxs_trackIndex")
    tke = g("GeneralTrack_hgcal_eta"); tkp = g("GeneralTrack_hgcal_phi")
    n_sc = len(spid)
    off = np.concatenate([[0], np.cumsum(strk_n)]).astype(int)
    match_eta = seta.astype(float).copy(); match_phi = sphi.astype(float).copy()
    for j in range(n_sc):
        if abs(int(spid[j])) in CHARGED:
            tids = strk_flat[off[j]:off[j+1]]
            for tid in tids:
                if 0 <= tid < len(tke) and abs(tke[tid]) > 1.5:
                    match_eta[j] = tke[tid]; match_phi[j] = tkp[tid]; break
    in_acc = np.abs(match_eta) >= 1.5
    matched = set()
    for gi in range(len(gpid)):
        if int(gst[gi]) != 1: continue
        apid = abs(int(gpid[gi]))
        if apid in NEUTRINOS or float(gpt[gi]) < GEN_PT_MIN: continue
        deta = match_eta - float(geta[gi])
        dphi = np.arctan2(np.sin(match_phi - float(gphi[gi])), np.cos(match_phi - float(gphi[gi])))
        dR = np.hypot(deta, dphi)
        hit = np.where((dR < dr_thresh(apid)) & in_acc)[0]
        matched.update(int(k) for k in hit)
    return matched

def process_event(E, iev):
    g = lambda b: ak.to_numpy(E[b][iev])
    ts_e = g(f"{TS}_raw_energy"); ts_reg = g(f"{TS}_regressed_energy")
    ts_pt = g(f"{TS}_raw_pt"); ts_em = g(f"{TS}_raw_em_energy")
    ts_eta = g(f"{TS}_barycenter_eta"); ts_phi = g(f"{TS}_barycenter_phi")
    n_ts = len(ts_e)
    tpt = g("GeneralTrack_pt"); tp = g("GeneralTrack_p"); teta = g("GeneralTrack_eta")
    tphi = g("GeneralTrack_phi"); tchg = g("GeneralTrack_charge")
    n_trk = len(tpt)
    spid = g("SimTICLCandidates_pdgID"); spt = g("SimTICLCandidates_pt")
    seta = g("SimTICLCandidates_eta"); sphi = g("SimTICLCandidates_phi")
    sen = g("SimTICLCandidates_energy"); sraw = g("SimTICLCandidates_raw_energy")
    spu = g("SimTICLCandidates_isPU")
    strk_n = g("SimTICLCandidates_nTrackIdxs"); strk_flat = g("SimTICLCandidatesTrackIdxs_trackIndex")
    strk_off = np.concatenate([[0], np.cumsum(strk_n)]).astype(int)

    # associators (per SimCP)
    s2r_n = g("SimCP2ticlTracksterLinksByHits_nSimCP2ticlTracksterLinksByHitsLinks")
    s2r_off = np.concatenate([[0], np.cumsum(s2r_n)]).astype(int)
    s2r_idx = g("SimCP2ticlTracksterLinksByHitsLinks_index")
    s2r_sc = g("SimCP2ticlTracksterLinksByHitsLinks_score")
    s2r_she = g("SimCP2ticlTracksterLinksByHitsLinks_sharedEnergy")
    r2s_n = g("RecoticlTracksterLinks2SimCPByHits_nRecoticlTracksterLinks2SimCPByHitsLinks")
    r2s_off = np.concatenate([[0], np.cumsum(r2s_n)]).astype(int)
    r2s_sc = g("RecoticlTracksterLinks2SimCPByHitsLinks_score")
    def rbest(ti):  # best (min) recoToSim score of trackster ti
        s = r2s_sc[r2s_off[ti]:r2s_off[ti+1]]
        return float(np.min(s)) if len(s) else 1.0

    matched = build_gen_to_simcand(g, E, iev)

    # ---- collect connections (element_idx in: ts [0,n_ts), track [n_ts, n_ts+n_trk)) ----
    conns = []
    for i in range(len(spid)):
        if i not in matched: continue
        apid = abs(int(spid[i]))
        cE = sraw[i] if apid == 11 else sen[i]     # electrons use raw_energy, else 'energy'
        # calo (trackster) connections via simToReco
        js = s2r_idx[s2r_off[i]:s2r_off[i+1]]; ss = s2r_sc[s2r_off[i]:s2r_off[i+1]]; sh = s2r_she[s2r_off[i]:s2r_off[i+1]]
        for k in range(len(js)):
            if sh[k] <= 0: continue
            ti = int(js[k])
            if ti >= n_ts: continue
            if rbest(ti) > R_CUT or (float(ss[k]) if k < len(ss) else 1.0) > S_CUT: continue
            etype = 2 if apid in (11, 22) else 3     # EM vs HAD trackster
            conns.append(dict(cp=i, pid=int(spid[i]), cE=cE, ceta=seta[i],
                              eidx=ti, etype=etype, w=ts_e[ti], ispu=max(0.0, float(spu[i]))))
        # track connections
        if apid != 22:  # photons never get tracks
            for tid in strk_flat[strk_off[i]:strk_off[i+1]]:
                if 0 <= tid < n_trk:
                    conns.append(dict(cp=i, pid=int(spid[i]), cE=cE, ceta=seta[i],
                                      eidx=n_trk_off(n_ts, tid), etype=1, w=tp[tid],
                                      ispu=max(0.0, float(spu[i]))))

    # ---- split_caloparticles (Mohamed's fragmenting logic) ----
    groups = defaultdict(list)
    for c in conns: groups[c["cp"]].append(c)
    split = []; new_idx = len(spid)
    for cp, cs in groups.items():
        apid = abs(cs[0]["pid"])
        # eta-sign (endcap side) consistency
        valid = []
        for c in cs:
            e_eta = teta[c["eidx"]-n_ts] if c["etype"] == 1 else ts_eta[c["eidx"]]
            if c["ceta"]*e_eta > 0: valid.append(c)
        if not valid: continue
        tracks = [c for c in valid if c["etype"] == 1]
        em = [c for c in valid if c["etype"] == 2]
        had = [c for c in valid if c["etype"] == 3]
        if apid == 11:      # electron: tracks+EM share ONE new_idx
            if not tracks and not em: continue
            tot_t = sum(c["w"] for c in tracks) or 1.0; tot_e = sum(c["w"] for c in em) or 1.0
            for c in tracks: split.append((cp, new_idx, c, c["cE"]*c["w"]/tot_t, True))
            for c in em:     split.append((cp, new_idx, c, c["cE"]*c["w"]/tot_e, True))
            new_idx += 1
        elif apid == 22:    # photon: FRAGMENT, one new_idx per EM trackster
            if not em: continue
            tot = sum(c["w"] for c in em) or 1.0
            for c in em:
                split.append((cp, new_idx, c, c["cE"]*c["w"]/tot, True)); new_idx += 1
        elif apid == 13:    # muon: tracks only
            if not tracks: continue
            tot = sum(c["w"] for c in tracks) or 1.0
            for c in tracks:
                split.append((cp, new_idx, c, c["cE"]*c["w"]/tot, True)); new_idx += 1
        elif apid in CHARGED and len(tracks) >= 1:  # charged hadron: track(s) carry energy, tracksters zeroed
            if len(tracks) == 1:
                split.append((cp, new_idx, tracks[0], tracks[0]["cE"], True)); new_idx += 1
            else:
                tot = sum(c["w"] for c in tracks) or 1.0
                for c in tracks:
                    split.append((cp, new_idx, c, c["cE"]*c["w"]/tot, True)); new_idx += 1
            for c in had:
                split.append((cp, new_idx, c, 0.0, False)); new_idx += 1
        else:               # neutral hadron: FRAGMENT among HAD tracksters
            tot = sum(c["w"] for c in had) or 1.0
            for c in had:
                split.append((cp, new_idx, c, c["cE"]*c["w"]/tot, True)); new_idx += 1

    # ---- build graph nodes: elements + cp; find primary element per cp ----
    n_el = n_ts + n_trk
    typ = np.zeros(n_el, np.int32); typ[:n_ts] = 3; typ[n_ts:] = 1
    # relabel EM-matched tracksters -> typ 2
    for cp, ni, c, e, win in split:
        if c["etype"] == 2 and c["eidx"] < n_ts: typ[c["eidx"]] = 2

    # cp nodes: energy/eta/phi/pid/charge (from first split entry per new_idx)
    cp_node = {}
    cp_edges = defaultdict(list)  # elem -> list of (new_idx, weight)
    for cp, ni, c, energy, win in split:
        if ni not in cp_node:
            eta = float(c["ceta"]); th = 2*math.atan(math.exp(-eta))
            cp_node[ni] = dict(pid=abs(c["pid"]), energy=energy, eta=eta,
                               phi=float(sphi[cp]), ispu=c["ispu"], charge=_charge(c["pid"]))
        cp_edges[c["eidx"]].append((ni, c["w"] if win else 0.0))

    # find_representative_elements: assign each elem its highest-weight cp (unique)
    used_cp = set(); elem_cp = {}
    order = sorted(range(n_el), key=lambda e: (0 if typ[e]==1 else 1 if typ[e]==2 else 2, -_ept(e, ts_pt, tpt, n_ts)))
    for e in order:
        cands = [(w, ni) for (ni, w) in cp_edges.get(e, []) if ni not in used_cp]
        if cands:
            ni = max(cands, key=lambda x: x[0])[1]
            elem_cp[e] = ni; used_cp.add(ni)

    # ---- ytarget ----
    ytarget = np.recarray((n_el,), dtype=[(n, np.float32) for n in particle_feature_order])
    ytarget.fill(0.0); ytarget["jet_idx"] = -1
    for e, ni in elem_cp.items():
        nd = cp_node[ni]
        if nd["energy"] <= 0: continue     # zeroed (charged-hadron trackster) -> no particle
        eta = nd["eta"]; th = 2*math.atan(math.exp(-eta)); pt = nd["energy"]*math.sin(th)
        ytarget["pid"][e] = nd["pid"]; ytarget["charge"][e] = nd["charge"]
        ytarget["pt"][e] = pt; ytarget["eta"][e] = eta
        ytarget["sin_phi"][e] = math.sin(nd["phi"]); ytarget["cos_phi"][e] = math.cos(nd["phi"])
        ytarget["energy"][e] = nd["energy"]; ytarget["ispu"][e] = nd["ispu"]

    # ---- Xelem ----
    Xelem = np.recarray((n_el,), dtype=[(n, np.float32) for n in elem_branches]); Xelem.fill(0.0)
    Xelem["typ"] = typ
    Xelem["pt"][:n_ts] = ts_pt; Xelem["eta"][:n_ts] = ts_eta; Xelem["phi"][:n_ts] = ts_phi
    Xelem["energy"][:n_ts] = ts_reg; Xelem["em_energy"][:n_ts] = ts_em
    Xelem["pt"][n_ts:] = tpt; Xelem["eta"][n_ts:] = teta; Xelem["phi"][n_ts:] = tphi
    Xelem["energy"][n_ts:] = tp; Xelem["charge"][n_ts:] = tchg

    # ---- jets ----
    vt = ytarget["pid"] != 0
    targetjet = cluster_jets(ytarget["pt"][vt], ytarget["eta"][vt],
                             np.arctan2(ytarget["sin_phi"][vt], ytarget["cos_phi"][vt]), ytarget["energy"][vt])
    gpid = g("HGCalGenPart_pdgId"); gpt = g("HGCalGenPart_pt"); geta = g("HGCalGenPart_eta")
    gphi = g("HGCalGenPart_phi"); gen_e = g("HGCalGenPart_energy")
    gm = ~np.isin(np.abs(gpid), list(NEUTRINOS))
    genjet = cluster_jets(gpt[gm], geta[gm], gphi[gm], gen_e[gm])
    pythia = np.stack([np.abs(gpid[gm]).astype(np.float32), gpt[gm], geta[gm], gphi[gm], gen_e[gm]], -1) \
             if gm.sum() else np.zeros((0, 5), np.float32)
    return {"Xelem": Xelem, "ytarget": ytarget, "genjet": genjet, "targetjet": targetjet, "pythia": pythia}

def n_trk_off(n_ts, tid):
    return n_ts + int(tid)
def _charge(pid):
    a = abs(pid)
    if a in (130, 22, 310): return 0.0
    if a in (11, 13): return -math.copysign(1.0, pid)
    if a in (211, 321): return math.copysign(1.0, pid)
    return 0.0
def _ept(e, ts_pt, tpt, n_ts):
    return float(ts_pt[e]) if e < n_ts else float(tpt[e-n_ts])

def process(infile, outfile, num_events=-1):
    print(f"opening {infile}")
    E = uproot.open(infile)["Events"].arrays(BRANCHES)
    n = len(E) if num_events < 0 else min(num_events, len(E))
    out = [process_event(E, i) for i in range(n)]
    with open(outfile, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    npart = sum(int((d["ytarget"]["pid"] != 0).sum()) for d in out)
    print(f"saved {outfile}: {n} ev, {npart} target particles ({npart/n:.1f}/ev), "
          f"targetjets {sum(len(d['targetjet']) for d in out)}, genjets {sum(len(d['genjet']) for d in out)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--num-events", type=int, default=-1); a = ap.parse_args()
    process(a.input, a.output, a.num_events)

if __name__ == "__main__":
    main()
