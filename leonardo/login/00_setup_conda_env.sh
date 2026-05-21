#!/usr/bin/env bash
# Login-node only: bootstrap miniforge (inside the repo if $HOME is full) and
# create the parler-tts conda env at $CONDA_ENV_PREFIX, then `pip install -e
# .[train]`. Idempotent: skips steps that are already done.
#
# Usage:
#   bash leonardo/login/00_setup_conda_env.sh
#
# Run on a login node (compute nodes have no internet).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"

mkdir -p "$REPO_ROOT/.conda" "$CACHE_ROOT" "$REPO_ROOT/leonardo/logs"

# ---- 1. miniforge (always install in repo, never reuse system conda) -------
# A fresh miniforge with conda-forge as the ONLY channel avoids surprises from
# an inherited $HOME/.condarc (e.g. the `defaults` channel shadowing packages).
MINIFORGE_DIR="$REPO_ROOT/.conda/miniforge"
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  echo "[setup] installing miniforge into $MINIFORGE_DIR"
  INSTALLER="$REPO_ROOT/.conda/miniforge_installer.sh"
  curl -fsSL -o "$INSTALLER" \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$INSTALLER" -b -p "$MINIFORGE_DIR"
  rm -f "$INSTALLER"
else
  echo "[setup] miniforge already present at $MINIFORGE_DIR (reusing)"
fi

# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
# Ignore any $HOME/.condarc for this shell.
export CONDARC=/dev/null

# ---- 2. env -----------------------------------------------------------------
if [ ! -d "$CONDA_ENV_PREFIX" ] || [ ! -x "$CONDA_ENV_PREFIX/bin/python" ]; then
  echo "[setup] creating env at $CONDA_ENV_PREFIX"
  "$MINIFORGE_DIR/bin/conda" create -y -p "$CONDA_ENV_PREFIX" \
    --override-channels -c conda-forge \
    python=3.10 \
    libsndfile sox ffmpeg \
    pip git
else
  echo "[setup] env already exists at $CONDA_ENV_PREFIX (reusing)"
fi

conda activate "$CONDA_ENV_PREFIX"

# ---- 3. pytorch (cu121 wheels) ---------------------------------------------
# cu121 wheels are forward-compatible with the `module load cuda/12.2` runtime
# used on Leonardo compute nodes. 2.5.1 is the highest cu121 wheel published.
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
echo "[setup] installing torch/torchaudio $TORCH_VERSION (cu121 wheels)"
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION"

# ---- 4. parler-tts + training extras ---------------------------------------
# Installs transformers==4.46.1, datasets[audio], accelerate, wandb, jiwer,
# evaluate, descript-audio-codec (DAC), sentencepiece, protobuf.
echo "[setup] pip install -e .[train]"
pip install -e "$REPO_ROOT[train]"
pip check || echo "[setup] pip check reported issues (often benign for the git audiotools pin)"

# ---- 5. sanity import -------------------------------------------------------
python - <<'PY'
import torch, torchaudio, transformers, datasets, accelerate
import parler_tts  # noqa: F401
print("[setup] parler_tts:", parler_tts.__version__ if hasattr(parler_tts, "__version__") else "ok")
print("[setup] torch:", torch.__version__,
      "torchaudio:", torchaudio.__version__,
      "transformers:", transformers.__version__,
      "datasets:", datasets.__version__)
print("[setup] CUDA available (login node, may be False):", torch.cuda.is_available())
PY

echo "[setup] DONE. Activate with:"
echo "        source \"$REPO_ROOT/leonardo/env.sh\" && activate_conda_env"
