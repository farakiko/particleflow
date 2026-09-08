#!/usr/bin/env python
"""Aggregate event counts and target-particle (type x pT-range) statistics
over the TICL postprocessed pkl files. Parallel over files."""
import os, sys, glob, pickle, argparse, re
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

PT_EDGES  = [0, 1, 5, 10, 20, 50, 100, 200, np.inf]
PT_LABELS = ["0-1", "1-5", "5-10", "10-20", "20-50", "50-100", "100-200", "200+"]
TYPES     = ["photon", "neut_had", "chrg_had", "electron", "muon", "other"]
PIDMAP    = {22: "photon", 130: "neut_had", 310: "neut_had",
             211: "chrg_had", 321: "chrg_had", 11: "electron", 13: "muon"}

def sample_of(fname):
    return re.sub(r"_[0-9]+_[0-9]+\.pkl$", "", os.path.basename(fname))

def blank():
    return {"n_events": 0, "n_particles": 0,
            "hist": np.zeros((len(TYPES), len(PT_LABELS)), dtype=np.int64)}

def process_files(files):
    out = defaultdict(blank)
    tidx = {t: i for i, t in enumerate(TYPES)}
    for f in files:
        try:
            data = pickle.load(open(f, "rb"))
        except Exception as e:
            sys.stderr.write(f"skip {f}: {e}\n"); continue
        s = out[sample_of(f)]
        s["n_events"] += len(data)
        for ev in data:
            yt  = ev["ytarget"]
            pid = yt["pid"]
            m   = pid != 0
            if not np.any(m):
                continue
            apid = np.abs(pid[m]).astype(np.int64)
            pt   = yt["pt"][m]
            s["n_particles"] += int(m.sum())
            ptbin = np.digitize(pt, PT_EDGES) - 1
            ptbin = np.clip(ptbin, 0, len(PT_LABELS) - 1)
            for p, b in zip(apid, ptbin):
                s["hist"][tidx.get(PIDMAP.get(int(p), "other"), tidx["other"]), b] += 1
    return dict(out)

def merge(dst, src):
    for k, v in src.items():
        d = dst.setdefault(k, blank())
        d["n_events"]    += v["n_events"]
        d["n_particles"] += v["n_particles"]
        d["hist"]        += v["hist"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-files", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.pkl")))
    if a.max_files > 0:
        # spread across sample types for a representative POC
        by = defaultdict(list)
        for f in files: by[sample_of(f)].append(f)
        files = [f for fs in by.values() for f in fs[: a.max_files]]
    n = len(files)
    print(f"processing {n} files with {a.workers} workers", flush=True)

    # chunk files across workers
    nchunks = min(a.workers * 4, n) or 1
    chunks = [files[i::nchunks] for i in range(nchunks)]
    result = {}
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process_files, c) for c in chunks]
        for fu in as_completed(futs):
            merge(result, fu.result())
            done += 1
            print(f"  chunk {done}/{len(chunks)} done", flush=True)

    # ---- report ----
    print("\n" + "=" * 70)
    print(f"{'sample':18s} {'files':>7} {'events':>10} {'target particles':>18}")
    tot_e = tot_p = 0
    for s in sorted(result):
        r = result[s]
        print(f"{s:18s} {'':>7} {r['n_events']:>10,d} {r['n_particles']:>18,d}")
        tot_e += r["n_events"]; tot_p += r["n_particles"]
    print(f"{'TOTAL':18s} {n:>7,d} {tot_e:>10,d} {tot_p:>18,d}")

    for s in sorted(result):
        h = result[s]["hist"]
        print("\n" + "-" * 70)
        print(f"[{s}]  target particles by type x pT (GeV)")
        hdr = f"{'type':10s}" + "".join(f"{l:>10s}" for l in PT_LABELS) + f"{'TOTAL':>12s}"
        print(hdr)
        for i, t in enumerate(TYPES):
            row = h[i]
            if row.sum() == 0: continue
            print(f"{t:10s}" + "".join(f"{v:>10,d}" for v in row) + f"{row.sum():>12,d}")
        col = h.sum(0)
        print(f"{'ALL':10s}" + "".join(f"{v:>10,d}" for v in col) + f"{h.sum():>12,d}")

    # save machine-readable
    outp = os.path.join(a.dir, ".sample_stats.npz")
    np.savez(outp, **{f"{s}__hist": result[s]["hist"] for s in result},
             **{f"{s}__nev": result[s]["n_events"] for s in result},
             types=np.array(TYPES), pt_labels=np.array(PT_LABELS))
    print(f"\nsaved -> {outp}")

if __name__ == "__main__":
    main()
