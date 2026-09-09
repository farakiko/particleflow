#!/usr/bin/env python
"""Sharded driver for Run3-style postprocessing over a whole NanoAOD tree.

Discovers every *.root under --input-dir, deterministically shards the sorted list,
and processes files[shard_index::num_shards] -> one pkl each under --output-dir,
mirroring the input subdirectory structure.

Designed for embarrassingly-parallel batch systems (k8s Indexed Job, HTCondor,
or plain xargs): every shard writes a disjoint set of outputs, skips files already
done, and writes each pkl atomically (tmp + rename) so a killed pod never leaves a
half-written file that a later run would wrongly skip. Fully resumable: re-run the
same job and only missing/failed files are redone.

Example (k8s Indexed Job passes JOB_COMPLETION_INDEX / total as the shard args):
  python postprocess_shard.py \
    --input-dir  /eos/cms/store/group/dpg_hgcal/comm_hgcal/moanwar/mlpf/nano \
    --output-dir /eos/user/f/fmokhtar/mlpf/phase2/pkl_run3style \
    --shard-index "$JOB_COMPLETION_INDEX" --num-shards 100
"""
import os, sys, glob, argparse, traceback, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocessing_run3style import process


def find_roots(input_dir):
    return sorted(glob.glob(os.path.join(input_dir, "**", "*.root"), recursive=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="root of the NanoAOD tree (searched recursively)")
    ap.add_argument("--output-dir", required=True, help="where pkls go (input subdir structure is mirrored)")
    ap.add_argument("--shard-index", type=int, default=0, help="this shard's index in [0, num-shards)")
    ap.add_argument("--num-shards", type=int, default=1, help="total number of shards")
    ap.add_argument("--num-events", type=int, default=-1, help="limit events/file (default all; use for a smoke test)")
    ap.add_argument("--max-files", type=int, default=-1, help="cap files this shard processes (for a quick test pod)")
    ap.add_argument("--calo", choices=["clue3d", "links"], default="clue3d",
                    help="calo collection passed to postprocessing (clue3d pre-linking / links CMSSW-merged)")
    ap.add_argument("--overwrite", action="store_true", help="reprocess even if the output pkl already exists")
    a = ap.parse_args()

    files = find_roots(a.input_dir)
    if not files:
        print(f"ERROR: no *.root found under {a.input_dir}", flush=True)
        sys.exit(1)
    mine = files[a.shard_index::a.num_shards]
    if a.max_files > 0:
        mine = mine[:a.max_files]
    print(f"shard {a.shard_index}/{a.num_shards}: {len(mine)} of {len(files)} total files", flush=True)

    ok = fail = skip = 0
    t0 = time.time()
    for i, root in enumerate(mine):
        rel = os.path.relpath(root, a.input_dir)
        base = os.path.splitext(os.path.basename(root))[0]
        outdir = os.path.join(a.output_dir, os.path.dirname(rel))
        outfile = os.path.join(outdir, f"run3style_{base}.pkl")
        if os.path.exists(outfile) and not a.overwrite:
            skip += 1
            continue
        os.makedirs(outdir, exist_ok=True)
        tmp = outfile + f".tmp.{a.shard_index}"
        try:
            process(root, tmp, a.num_events, a.calo)   # writes the pkl to tmp
            os.replace(tmp, outfile)           # atomic publish on the same filesystem
            ok += 1
            print(f"[{i+1}/{len(mine)}] OK   {rel}", flush=True)
        except Exception as e:
            fail += 1
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            print(f"[{i+1}/{len(mine)}] FAIL {rel}: {e}", flush=True)
            traceback.print_exc()

    print(f"shard {a.shard_index} done in {time.time()-t0:.0f}s: ok={ok} skip={skip} fail={fail}", flush=True)
    sys.exit(2 if fail else 0)


if __name__ == "__main__":
    main()
