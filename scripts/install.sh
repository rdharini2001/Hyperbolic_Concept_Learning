#!/bin/bash
# One-time installation, no root needed. Workstation or cluster login node (needs internet).
#
#   bash scripts/install.sh
#
# Creates a venv at $HT_ENV (default ~/envs/hypertrace) with:
#   * PyTorch + torchvision built for CUDA 12.8 (required for Blackwell / RTX 50xx / RTX PRO 6000;
#     also runs on older GPUs, Turing and newer). Override with TORCH_INDEX=... if needed.
#   * this package (hypertrace) in editable mode
#   * CT-CLIP from source, with only the dependencies its encoder needs
#   * MONAI, for the FMCIB lesion encoder
#   * the Hugging Face CLI (`hf`), used by the download scripts
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
HT_ENV="${HT_ENV:-$HOME/envs/hypertrace}"
HT_WORK="${HT_WORK:-$HOME/hypertrace_work}"
HT_SRC="${HT_SRC:-$HT_WORK/src}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python >= 3.10 needed, found {sys.version.split()[0]}. Set PYTHON=/path/to/python3.11 "
             "(or use conda/uv to get one without root).")
EOF

echo ">> venv at $HT_ENV"
"$PY" -m venv "$HT_ENV"
source "$HT_ENV/bin/activate"
pip install --upgrade pip wheel

echo ">> PyTorch from $TORCH_INDEX"
pip install torch torchvision --index-url "$TORCH_INDEX"

echo ">> hypertrace"
pip install -e ".[ctclip,fmcib,dev,viz]"

echo ">> CT-CLIP (source) into $HT_SRC"
mkdir -p "$HT_SRC"
[ -d "$HT_SRC/CT-CLIP" ] || git clone https://github.com/ibrahimethemhamamci/CT-CLIP "$HT_SRC/CT-CLIP"
# --no-deps: its setup.py pulls packages we don't need (wilds, ImageNetV2 from git, ...) and could
# replace the CUDA 12.8 torchvision. The encoder's real dependencies are installed explicitly below.
pip install --no-deps -e "$HT_SRC/CT-CLIP/transformer_maskgit"
pip install --no-deps -e "$HT_SRC/CT-CLIP/CT_CLIP"
pip install accelerate beartype "einops>=0.6" "ema-pytorch>=0.2.2" sentencepiece torchtyping tqdm \
            "vector-quantize-pytorch==1.1.2" opencv-python-headless pillow ftfy regex
# vector-quantize-pytorch is pinned like CT-CLIP's setup.py: its parameter names are part of the checkpoint.

echo ">> Hugging Face CLI"
pip install -U huggingface_hub

echo ">> checks"
python scripts/check_env.py
echo
echo "Installed. Next: edit HT_WORK in scripts/env.sh, then"
echo "  source scripts/env.sh"
echo "  bash scripts/download_weights.sh      # CT-CLIP + CXR-BERT (accept CT-RATE terms first) + FMCIB"
echo "  ONLY_CSV=1 bash scripts/download_bucket.sh && NEEDED_ONLY=1 bash scripts/download_bucket.sh"
