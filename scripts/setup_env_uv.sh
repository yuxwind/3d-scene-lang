#!/bin/bash
# Build the sr_vggt Python env with uv from envs/sr_vggt.lock.txt (gandalf's env, Python 3.11, torch 2.11 + cu128),
# for hosts without the IDRIS modules (gandalf, wukong, cleps). jz uses scripts/setup_env.sh instead.
#   scripts/setup_env_uv.sh            -> ${SR_ENV_ROOT}/sr_vggt ; set SR_PYTHON_BIN=${SR_ENV_ROOT}/sr_vggt/bin in site.env
# Needs internet (login node) and uv (~/.local/bin/uv or $UV). The uv cache and the downloaded CPython go under
# ${SR_ENV_ROOT} (not $HOME: cleps' home is at its quota). Rerunnable: an existing env is synced to the lock.
set -euo pipefail
SR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "${SR_ROOT}/site.env" ]] || { echo "ERROR: ${SR_ROOT}/site.env missing" >&2; exit 1; }
set -a; source "${SR_ROOT}/site.env"; set +a
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
VENV="${SR_ENV_ROOT:?set SR_ENV_ROOT in site.env}/sr_vggt"
export UV_CACHE_DIR="${SR_ENV_ROOT}/.uv-cache" UV_PYTHON_INSTALL_DIR="${SR_ENV_ROOT}/.uv-python" UV_LINK_MODE=copy
mkdir -p "${SR_ENV_ROOT}"
echo "=== ${VENV} (uv $("${UV}" --version | cut -d' ' -f2)) ==="
[[ -x "${VENV}/bin/python" ]] || "${UV}" venv --python 3.11 "${VENV}"
"${UV}" pip sync --python "${VENV}/bin/python" "${SR_ROOT}/envs/sr_vggt.lock.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
PYTHONPATH="${SR_ROOT}" "${VENV}/bin/python" - <<'PY'
import torch, transformers, trl, peft, accelerate, decord, numpy, pandas
import sr_vggt.model  # vendored deps import
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__, "trl", trl.__version__,
      "peft", peft.__version__, "| GPU visible here:", torch.cuda.is_available())
PY
du -sh "${VENV}" | sed 's/^/size /'
