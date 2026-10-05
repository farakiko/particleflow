#!/bin/bash
# Print the resumable training trend: latest step + recent validations (valid loss / jet IQR /
# match). Run inside any pod that mounts /shared (the util pod): reads history JSONs, no GPU.
#   EXP / LOG env override the paths (default s3ls-prod).
EXP=${EXP:-/shared/mlpf-phase2/experiments/s3ls-prod}
LOG=${LOG:-/shared/mlpf-phase2/logs/s3ls_prod.log}
echo "latest: $(grep -hE 'Step [0-9]+/|start/resume|exit=' "$LOG" 2>/dev/null | tail -1)"
echo "recent validations:"
source /shared/envs/mlpf/bin/activate 2>/dev/null
EXP="$EXP" python - <<'PY'
import glob, json, os
exp = os.environ["EXP"]
B = "step/cms_pf_ticl_nopu/jet_ratio/jet_ratio_target_to_pred_pt/"
fs = sorted(glob.glob(exp + "/history/step_*.json"),
            key=lambda f: int(f.split("_")[-1].split(".")[0]))
if not fs:
    print("  (no validations yet)")
for f in fs[-10:]:
    d = json.load(open(f)); v = d["valid"]; tv = v["Total"] if isinstance(v, dict) else v
    st = f.split("_")[-1].split(".")[0]
    try:
        print("  step %8s: valid %.4f  jet med %.3f iqr %.3f match %.3f" % (
            st, tv, d[B + "med"], d[B + "iqr"], d[B + "match_frac"]))
    except KeyError:
        print("  step %8s: valid %.4f" % (st, tv))
PY
