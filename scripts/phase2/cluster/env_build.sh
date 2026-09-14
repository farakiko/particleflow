#!/bin/bash
# One-time build of the persistent MLPF training env at /shared/envs/mlpf
# (uv-managed python 3.11 on /shared; wheel cache on node-local /scratch)
set -uo pipefail
echo "=== env build start $(date) on $(hostname) ==="
# EOS home is over quota (even 0-byte writes fail) -> keep ALL dotfiles/caches on /shared
export HOME=/shared/mlpf-phase2/home
mkdir -p "$HOME"
export UV_INSTALL_DIR=/shared/uv/bin
export UV_PYTHON_INSTALL_DIR=/shared/uv/python
export UV_CACHE_DIR=/scratch/fmokhtar-uv-cache
if [ ! -x /shared/uv/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "UV INSTALL FAILED"; exit 1; }
fi
export PATH=/shared/uv/bin:$PATH
uv --version
cd /shared/particleflow || exit 1
export UV_PROJECT_ENVIRONMENT=/shared/envs/mlpf
echo "=== uv sync (deps only) ==="
uv sync --no-install-project --python 3.11 2>&1 || { echo "UV SYNC FAILED"; exit 1; }
echo "=== editable install of the repo ==="
# NB: `uv pip` ignores UV_PROJECT_ENVIRONMENT -> target the venv explicitly
uv pip install --python /shared/envs/mlpf/bin/python --no-deps -e . 2>&1 || { echo "EDITABLE INSTALL FAILED"; exit 1; }
echo "=== sanity ==="
source /shared/envs/mlpf/bin/activate
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())"
python - <<'EOF'
import torch
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(512, 512, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda matmul ok:", tuple(y.shape))
else:
    print("WARNING: no cuda")
EOF
python -c "import tensorflow_datasets, array_record, fastjet, comet_ml, awkward, vector; print('deps ok')"
python -c "
from mlpf.conf import MLPFConfig
from mlpf.model.mlpf import MLPF
c = MLPFConfig.from_spec('particleflow_spec.yaml','pyg-cms-phase2-v1','cms_phase2_ngt',None,None)
m = MLPF(c)
print('MLPF params: %.2fM' % (sum(p.numel() for p in m.parameters())/1e6))
"
echo "=== env build done $(date) ==="
