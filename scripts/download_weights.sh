#!/bin/bash
# Download the pretrained weights Part 2 needs (needs internet):
#   1. CT-CLIP image+text model: models/CT-CLIP-Related/CT-CLIP_v2.pt from the CT-RATE dataset repo
#   2. CXR-BERT tokenizer/text model used by CT-CLIP: microsoft/BiomedVLP-CXR-BERT-specialized
#   3. FMCIB lesion encoder (Pai et al. 2024): model_weights.torch from Zenodo record 10528450
#
#   WHAT=fmcib bash scripts/download_weights.sh     # only FMCIB (WHAT=ctclip: only CT-CLIP + CXR-BERT)
#
# CT-RATE is GATED (licence CC BY-NC-SA, non-commercial):
#   a. open https://huggingface.co/datasets/ibrahimhamamci/CT-RATE while logged in and accept the terms
#   b. create a read token at https://huggingface.co/settings/tokens and run:  hf auth login
#
#   source scripts/env.sh && bash scripts/download_weights.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
mkdir -p "$HT_MODELS"
WHAT="${WHAT:-all}"

if [ "$WHAT" = all ] || [ "$WHAT" = ctclip ]; then
echo ">> CT-CLIP_v2.pt -> $HT_MODELS/models/CT-CLIP-Related/"
hf download ibrahimhamamci/CT-RATE models/CT-CLIP-Related/CT-CLIP_v2.pt \
  --repo-type dataset --local-dir "$HT_MODELS" \
  || { echo "!! Download failed. Did you accept the CT-RATE terms and run 'hf auth login'?"; exit 1; }

echo ">> CXR-BERT -> $HT_CXR_BERT"
hf download microsoft/BiomedVLP-CXR-BERT-specialized --local-dir "$HT_CXR_BERT"

ls -lh "$HT_CTCLIP_CKPT"
fi

if [ "$WHAT" = all ] || [ "$WHAT" = fmcib ]; then
  echo ">> FMCIB -> /storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/models/fmcib/model_weights.torch"
  mkdir -p "$(dirname "$HT_FMCIB_CKPT")"
  [ -s "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/models/fmcib/model_weights.torch" ] || curl -L --fail -o "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/models/fmcib/model_weights.torch" \
    "https://zenodo.org/records/10528450/files/model_weights.torch?download=1" \
    || { echo "!! FMCIB download failed (is zenodo.org reachable from this machine?)"; exit 1; }
  ls -lh "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/models/fmcib/model_weights.torch"
fi
echo "Done."
