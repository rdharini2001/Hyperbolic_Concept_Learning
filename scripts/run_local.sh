#!/bin/bash
# The whole Part-2 pipeline on ONE machine with one GPU (no SLURM). Every step is resumable:
# re-running skips what is already done, so you can stop it at any time.
#
#   source scripts/env.sh
#   bash scripts/run_local.sh              # all steps
#   bash scripts/run_local.sh manifest     # one step: manifest | radiomics | extract | probe | tune | train | viz | summary
#   SEEDS="0" FOLDS="0 1" bash scripts/run_local.sh train   # a smaller grid
#   DIMS="2 4 8 16 32 64" bash scripts/run_local.sh train   # dimension sweep (Poincaré vs Euclidean per dim)
#   TAG=v3 bash scripts/run_local.sh train                  # run-name suffix _tv3: a new set of run folders
#   GEOMETRIES="euclidean poincare sphere kstereo product:matched product:swapped" DIMS="20 40" ...
#                                    # product:<assignment> = one factor per concept group (geometry.product)
#   TUNE_DIMS="4 16" bash scripts/run_local.sh tune          # matched tuning grid (hypertrace-tune; not in "all")
#   VIZ_DIMS="2 64" VIZ_FOLDS="0 3" VIZ_SEEDS="0" bash scripts/run_local.sh viz   # one figure per combination
#
# HT_CACHE_DIR is the one cache that extract writes and probe / train read
# (default: CT-CLIP 64 mm crops, $HT_CACHE_ROOT/ctclip_crop64). Examples:
#   EXTRACT_SET="input_mode=crop crop_size_mm=96" HT_CACHE_DIR=$HT_CACHE_ROOT/ctclip_crop96 bash scripts/run_local.sh extract
#   HT_CACHE_DIR=$HT_CACHE_ROOT/ctclip_crop96 bash scripts/run_local.sh probe
#   export HT_CACHE_DIR=$HT_CACHE_ROOT/fmcib_lesion
#   EXTRACT_SET="encoder.name=fmcib" bash scripts/run_local.sh extract
#   bash scripts/run_local.sh probe
#   CONFIG=src/hypertrace/configs/experiments/fmcib.yaml bash scripts/run_local.sh train
set -euo pipefail
cd "$(dirname "$0")/.."
# [ -n "${HT_WORK:-}" ] || source scripts/env.sh

# --- data / outputs
export HT_BUCKET="/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/lidc_data"   # local copy of the bucket: images/, masks_instance/, *.csv
export HT_DATA_ROOT="/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/data"            # manifest.csv, manifest_folds.csv
export HT_CACHE_ROOT="/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/cache"          # frozen CT-CLIP readouts
export HT_OUTPUT_ROOT="/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/outputs"       # training runs
export HT_MODELS="/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/models"

# --- pretrained weights (scripts/download_weights.sh puts them here)
export HT_CTCLIP_CKPT="$HT_MODELS/CT-CLIP-Related/CT-CLIP_v2.pt"
export HT_CXR_BERT="$HT_MODELS/CXR-BERT"
export HT_FMCIB_CKPT="$HT_MODELS/fmcib/model_weights.torch"

# --- runs never download anything (cluster compute nodes are offline; makes runs reproducible).
#     The download_* scripts switch this back on for themselves.
export HF_HOME="/storage/home/hcoda1/5/nshanbhogue6/scratch/cache/htrace_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-8}}"
export PYTHONUNBUFFERED=1

STEP="${1:-all}"
CONFIG="${CONFIG:-src/hypertrace/configs/train.yaml}"
GEOMETRIES="${GEOMETRIES:-poincare euclidean}"
FOLDS="${FOLDS:-0 1 2 3 4}"
SEEDS="${SEEDS:-0 1 2}"
DIMS="${DIMS:-64}"
TAG="${TAG:-}"                       # optional run-name suffix (tag=...); empty = none
# cache the probe and training read; must be the folder hypertrace-extract logged ("cache ...")
# (named HT_CACHE_DIR, not CACHE: a generic CACHE variable in the shell must not redirect it)
HT_CACHE_DIR="${HT_CACHE_DIR:-/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/cache/ctclip_crop64}"
RADIOMICS="/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/cache/radiomics.csv"
DEVICE="${DEVICE:-cuda}"          # extraction device; set DEVICE=cpu only for tests (very slow with CT-CLIP)
EXTRACT_SET="${EXTRACT_SET:-input_mode=crop crop_size_mm=64}"   # extraction overrides; cache_dir=$HT_CACHE_DIR is added
TRAIN_SET="${TRAIN_SET:-}"        # extra training overrides, e.g. "trainer.max_epochs=5"
mkdir -p "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/logs"
run() { [ "$STEP" = all ] || [ "$STEP" = "$1" ]; }

if run manifest; then
  echo "== manifest + folds"
  [ -f "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/data/manifest.csv" ] || hypertrace-build-manifest --bucket-dir "$HT_BUCKET" --out "$HT_DATA_ROOT/manifest.csv"
  [ -f "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/data/manifest_folds.csv" ] || hypertrace-make-folds --manifest "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/data/manifest.csv" --out "$HT_DATA_ROOT/manifest_folds.csv"
fi

if run radiomics; then
  echo "== hand-crafted baseline features (CPU, ~10 min)"
  [ -f "$RADIOMICS" ] || hypertrace-radiomics --out "$RADIOMICS" 2>&1 | tee -a "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/logs/radiomics.log"
fi

if run extract; then
  echo "== extract (GPU; resumable)"
  # shellcheck disable=SC2086
  hypertrace-extract --set device="$DEVICE" $EXTRACT_SET cache_dir="$HT_CACHE_DIR" 2>&1 \
    | tee -a "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/logs/extract_$(basename "$HT_CACHE_DIR").log"
fi

if run probe; then
  echo "== feature check on $HT_CACHE_DIR"
  RAD=(); [ -f "$RADIOMICS" ] && RAD=(--radiomics "$RADIOMICS")
  hypertrace-zeroshot --cache-dir "$HT_CACHE_DIR" --manifest "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/data/manifest_folds.csv" --probe ${RAD[@]+"${RAD[@]}"} \
    | tee "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/logs/probe_$(basename "$HT_CACHE_DIR").txt"
fi

TAG_SET=""; [ -n "$TAG" ] && TAG_SET="tag=$TAG"
# a GEOMETRIES entry -> (--set arguments, run-name part): product:matched -> product-matched
geo_set() { case "$1" in product:*) echo "geometry.name=product geometry.product.assignment=${1#product:}";;
                         *) echo "geometry.name=$1";; esac; }
geo_name() { case "$1" in product:*) echo "product-${1#product:}";; *) echo "$1";; esac; }

if [ "$STEP" = "tune" ]; then  # only on request: it trains many runs
  echo "== matched tuning grid: geometries [$GEOMETRIES] x dims [${TUNE_DIMS:-16}] x folds [$FOLDS] x seeds [${TUNE_SEEDS:-0}]"
  # shellcheck disable=SC2086
  hypertrace-tune --config "$CONFIG" --geometries $GEOMETRIES --dims ${TUNE_DIMS:-16} --folds $FOLDS \
    --seeds ${TUNE_SEEDS:-0} --shard "${TUNE_SHARD:-0/1}" ${TUNE_SPACE:+--space "$TUNE_SPACE"} \
    --set paths.cache_dir="$HT_CACHE_DIR" $TRAIN_SET 2>&1 | tee -a "$HT_WORK/logs/tune_$(basename "$CONFIG" .yaml).log"
fi

if run train; then
  echo "== train grid: geometries [$GEOMETRIES] x dims [$DIMS] x folds [$FOLDS] x seeds [$SEEDS], config $CONFIG, cache $HT_CACHE_DIR${TAG:+, tag $TAG}"
  CTAG=$(basename "$CONFIG" .yaml)
  for d in $DIMS; do for g in $GEOMETRIES; do for f in $FOLDS; do for s in $SEEDS; do
    # hypertrace-train itself resumes an interrupted run and finishes a completed one instantly
    LOG="/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/logs/train_${CTAG}_$(geo_name "$g")_d${d}${TAG:+_t$TAG}_f${f}_s${s}.log"
    # shellcheck disable=SC2086
    hypertrace-train --config "$CONFIG" --set $(geo_set "$g") data.test_fold="$f" seed="$s" \
        model.projector.embed_dim="$d" paths.cache_dir="$HT_CACHE_DIR" $TAG_SET $TRAIN_SET > "$LOG" 2>&1 \
      && tail -n 1 "$LOG" || echo "!! failed: $g d$d fold $f seed $s (see $LOG)"
  done; done; done; done
fi

if run viz; then
  # default: last dim, first fold, first seed. VIZ_DIMS / VIZ_FOLDS / VIZ_SEEDS: one figure per combination
  VD="${VIZ_DIMS:-${DIMS##* }}"; VF="${VIZ_FOLDS:-${FOLDS%% *}}"; VS="${VIZ_SEEDS:-${SEEDS%% *}}"
  echo "== concept-space figures: dims [$VD] x folds [$VF] x seeds [$VS], geometries side by side"
  EXP=$(python -c "import sys,yaml;from hypertrace.config import load_config;print(load_config(sys.argv[1])['experiment'])" "$CONFIG")
  FEAT=$(python -c "import sys;from hypertrace.config import load_config;print(load_config(sys.argv[1])['data']['image_feature'])" "$CONFIG")
  for D in $VD; do for F in $VF; do for S in $VS; do
    ARGS=()
    for g in $GEOMETRIES; do
      r="/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/outputs/runs/${EXP}_$(geo_name "$g")_${FEAT}_d${D}${TAG:+_t$TAG}_f${F}_s${S}"
      [ -f "$r/checkpoints/best.pt" ] && ARGS+=(--run "$r")
    done
    if [ ${#ARGS[@]} -gt 0 ]; then
      hypertrace-visualize "${ARGS[@]}" --color-by "${COLOR_BY:-margin_spiculated}" --out "$HT_OUTPUT_ROOT/figures" \
        --name "concept_space_${EXP}_d${D}${TAG:+_t$TAG}_f${F}_s${S}"
    else
      echo "   d$D fold $F seed $S: no finished runs to plot yet"
    fi
  done; done; done
fi

if [ "$STEP" = "best" ]; then
  # one figure per dim: the (fold, seed) with the best mean VALIDATION AUROC over the arms; all arms drawn on it
  echo "== figures for the best split per dim (chosen on validation)"
  EXP=$(python -c "import sys;from hypertrace.config import load_config;print(load_config(sys.argv[1])['experiment'])" "$CONFIG")
  FEAT=$(python -c "import sys;from hypertrace.config import load_config;print(load_config(sys.argv[1])['data']['image_feature'])" "$CONFIG")
  NARMS=$(echo $GEOMETRIES | wc -w)
  python scripts/best_runs.py --runs "$HT_OUTPUT_ROOT/runs" --tag "$TAG" --dims "$DIMS" --n-arms "$NARMS" | while read -r D F S; do
    ARGS=()
    for g in $GEOMETRIES; do
      r="$HT_OUTPUT_ROOT/runs/${EXP}_$(geo_name "$g")_${FEAT}_d${D}${TAG:+_t$TAG}_f${F}_s${S}"
      [ -f "$r/checkpoints/best.pt" ] && ARGS+=(--run "$r")
    done
    echo "   d$D: fold $F seed $S"
    hypertrace-visualize "${ARGS[@]}" --color-by "${COLOR_BY:-margin_spiculated}" --out "$HT_OUTPUT_ROOT/figures" \
      --name "best_${EXP}_d${D}${TAG:+_t$TAG}_f${F}_s${S}"
  done
fi

if run summary; then
  echo "== summary"
  hypertrace-summarize --runs "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/outputs/runs" --out "/storage/home/hcoda1/5/nshanbhogue6/scratch/hypertrace/outputs/all_runs.csv" --per-concept --groups
fi
