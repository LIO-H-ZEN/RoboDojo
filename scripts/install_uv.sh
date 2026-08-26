#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${ROBODOJO_UV_ENV:-/tmp/robodojo-uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-robodojo}"
BUILD_TMP_DIR="${ROBODOJO_BUILD_TMP:-/tmp/robodojo-build}"

[[ "${VENV_PATH}" = /tmp/* ]] || {
    echo "[install_uv] ROBODOJO_UV_ENV must be under /tmp: ${VENV_PATH}" >&2
    exit 2
}

command -v uv >/dev/null 2>&1 || {
    echo "[install_uv] uv is required" >&2
    exit 1
}

export UV_CACHE_DIR
export TMPDIR="${BUILD_TMP_DIR}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-32}"
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-32}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-32}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PIP_USER=0
export TERM=xterm-256color

mkdir -p "${UV_CACHE_DIR}" "${BUILD_TMP_DIR}"
uv venv --python 3.11 "${VENV_PATH}"
PYTHON="${VENV_PATH}/bin/python"

uv pip install --python "${PYTHON}" pip setuptools wheel
uv pip install --python "${PYTHON}" -r "${ROOT_DIR}/scripts/requirements.txt"
uv pip install --python "${PYTHON}" \
    opencv-python-headless==4.11.0.86 pillow matplotlib \
    scipy==1.15.3 scikit-learn numpy==1.26.0
uv pip install --python "${PYTHON}" \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${PYTHON}" \
    'isaacsim[all,extscache]==5.1.0' \
    --extra-index-url https://pypi.nvidia.com

# IsaacLab's installer uses the active interpreter and pip entry point.
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
(cd "${ROOT_DIR}/third_party/IsaacLab" && ./isaaclab.sh --install none)

uv pip uninstall --python "${PYTHON}" nvidia-curobo curobo
uv pip install --python "${PYTHON}" \
    -e "${ROOT_DIR}/third_party/curobo[cu12]" --no-build-isolation

uv pip install --python "${PYTHON}" \
    numpy==1.26.0 packaging==23.0 typing_extensions==4.12.2 \
    filelock==3.13.1 websockets==12.0 click==8.1.7 psutil==5.9.8 \
    wheel==0.45.1 starlette==0.45.3 scipy==1.15.3 warp-lang==1.11.0 \
    'onnx>=1.18,<1.22' 'ipython<9' virtualenv==20.30.0
uv pip uninstall --python "${PYTHON}" python-discovery

"${PYTHON}" - <<'PY'
import importlib.metadata

import isaaclab  # noqa: F401
import isaacsim  # noqa: F401
import torch

print(f"isaacsim={importlib.metadata.version('isaacsim')}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

echo "[install_uv] ready: ${VENV_PATH}"
