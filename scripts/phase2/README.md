# Phase-2 target/sample analysis scripts

Exploratory analysis of the TICL postprocessed pkl outputs (per-event dicts with
`Xelem`, `ycand`, `ytarget`, `genjet`, `targetjet`, `genmet`, `pythia`).
Requires `uproot numpy awkward matplotlib fastjet` (fastjet only for the jet
clustering of `ycand`). All take `--dir <pkl_output>` and parallelize over files.

Data: `data/cms/phase2/offline/Aug31/nano/pkl_output/` (QCD / ttbar / Z→ll, 0 PU).

| script | what it does |
|---|---|
| `sample_stats.py` | event counts per sample + target-particle counts by type × pT range. Saves `.sample_stats.npz`. |
| `jet_response.py` | target-jet pT response vs truth (matched-gen) jets: response histogram + median/resolution vs pT. `--per-sample N`. |
| `jet_response_cms.py` | CMS-style PF (TICLCandidates) vs MLPF-target jet response (`jet pT / genjet pT`, log-log). |
| `jet_diagnostics.py` | per-sample sanity plots: jet pT spectrum, \|η\|, and gen→nearest-jet ΔR (matching quality). |

Example:
```
python scripts/phase2/jet_response_cms.py \
    --dir data/cms/phase2/offline/Aug31/nano/pkl_output \
    --outdir plots/phase2 --per-sample 250
```

Key knobs live at the top of each file (`GEN_PT_MIN`, `DR_MATCH`, pT bins, etc.).
Outputs go to `plots/phase2/` (gitignored).
