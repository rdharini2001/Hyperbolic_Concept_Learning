#!/bin/bash
# Environment for every HYPER-TRACE command, on a workstation or a cluster:
#   source scripts/env.sh
# Edit HT_WORK (and, on a cluster, the module lines) once. Nothing here needs root.

# --- where everything big lives (data ~50-70 GB, cache ~1 GB, runs ~1 GB). Not a small $HOME quota.
export HT_WORK="${HT_WORK:-$HOME/hypertrace_work}"          # cluster: e.g. /scratch/$USER/hypertrace

# --- clusters only: load modules if the `module` command exists (harmless on a workstation)
if command -v module >/dev/null 2>&1; then
  module load python/3.11 2>/dev/null || true             # adjust names: `module avail python cuda`
  module load cuda/12.8 2>/dev/null || true
fi

# --- Python environment created by scripts/install.sh
export HT_ENV="${HT_ENV:-$HOME/envs/hypertrace}"
[ -f "$HT_ENV/bin/activate" ] && source "$HT_ENV/bin/activate"

# --- data / outputs
export HT_BUCKET="$HT_WORK/LIDC-IDRI-bucket"   # local copy of the bucket: images/, masks_instance/, *.csv
export HT_DATA_ROOT="$HT_WORK/data"            # manifest.csv, manifest_folds.csv
export HT_CACHE_ROOT="$HT_WORK/cache"          # frozen CT-CLIP readouts
export HT_OUTPUT_ROOT="$HT_WORK/outputs"       # training runs
export HT_MODELS="$HT_WORK/models"

# --- pretrained weights (scripts/download_weights.sh puts them here)
export HT_CTCLIP_CKPT="$HT_MODELS/models/CT-CLIP-Related/CT-CLIP_v2.pt"
export HT_CXR_BERT="$HT_MODELS/BiomedVLP-CXR-BERT-specialized"
export HT_FMCIB_CKPT="$HT_MODELS/fmcib/model_weights.torch"   # lesion-level encoder (Pai et al. 2024)

# --- runs never download anything (cluster compute nodes are offline; makes runs reproducible).
#     The download_* scripts switch this back on for themselves.
export HF_HOME="$HT_WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-8}}"
export PYTHONUNBUFFERED=1
