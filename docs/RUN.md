# Running HYPER-TRACE Part 2 (short guide)

One workstation, one GPU, command line, no root. Every step resumes where it stopped, so you can
stop and restart any time. Run long jobs in `tmux` or `nohup ... &`. Logs: `$HT_WORK/logs/`.
(Cluster: `slurm/setup_env.sh`, then `slurm/pipeline.sh`; the old SLURM section is in your previous RUNNING.md.)

## 0. Once

```bash
bash scripts/install.sh                       # venv in ~/envs/hypertrace (Python >= 3.10, CUDA 12.8 driver)
source scripts/env.sh                         # EVERY new terminal. Edit HT_WORK in it once (needs ~70 GB)
hf auth login                                 # after accepting the terms at huggingface.co/datasets/ibrahimhamamci/CT-RATE
bash scripts/download_weights.sh              # CT-CLIP + CXR-BERT + FMCIB (WHAT=fmcib: FMCIB only)
ONLY_CSV=1 bash scripts/download_bucket.sh && NEEDED_ONLY=1 bash scripts/download_bucket.sh   # data
python scripts/check_env.py && pytest -q      # "all good"; ~146 tests
bash scripts/run_local.sh manifest            # 1390 nodules, 701 scans; folds
```

**"CXR-BERT" is not a separate case.** It is the text encoder inside CT-CLIP (`HT_CXR_BERT`, downloaded
above). It only embeds the concept prompts, used for zero-shot and `model.concepts.init=text`. You run
two encoders: **CT-CLIP** and **FMCIB**.

## 1. Extract features (once per encoder; ~15 min FMCIB, longer for CT-CLIP)

`HT_CACHE_DIR` selects the cache that `extract`, `probe` and `train` all use.

```bash
# FMCIB (lesion encoder; 4096-d)
export HT_CACHE_DIR=$HT_CACHE_ROOT/fmcib_lesion
EXTRACT_SET="encoder.name=fmcib" bash scripts/run_local.sh extract

# CT-CLIP, 64 mm nodule crops (the default cache)
export HT_CACHE_DIR=$HT_CACHE_ROOT/ctclip_crop64
bash scripts/run_local.sh extract
# CT-CLIP 96 mm crops, if you want that variant
export HT_CACHE_DIR=$HT_CACHE_ROOT/ctclip_crop96
EXTRACT_SET="input_mode=crop crop_size_mm=96" bash scripts/run_local.sh extract
# a CT-CLIP cache made before the v3 ontology: refresh only the prompt embeddings (seconds)
hypertrace-extract --set only_text=true cache_dir=$HT_CACHE_DIR
```

Check the features before training on them. A feature is worth training only if its probe beats
`diameter_only` and `radiomics` on the morphology concepts (margin, shape, type, calcification):

```bash
bash scripts/run_local.sh radiomics && bash scripts/run_local.sh probe     # table + $HT_CACHE_DIR/zeroshot.csv
```

## 2. The three training setups

Pick the encoder with `HT_CACHE_DIR` and `CONFIG`. Set `TAG` to a new value for every new set of
settings (a run folder is never reused with different settings).

| encoder | `HT_CACHE_DIR` | `CONFIG` |
|---|---|---|
| FMCIB | `$HT_CACHE_ROOT/fmcib_lesion` | `src/hypertrace/configs/experiments/fmcib.yaml` |
| CT-CLIP 64 mm | `$HT_CACHE_ROOT/ctclip_crop64` | `src/hypertrace/configs/train.yaml` (default; `token_mean`) |
| CT-CLIP 96 mm | `$HT_CACHE_ROOT/ctclip_crop96` | `src/hypertrace/configs/train.yaml` |

`fmcib.yaml` is 4 lines (create it if your checkout lacks it):
```yaml
base: ../train.yaml
experiment: fmcib
data:
  image_feature: lesion_embedding
```

Below, set the two variables once per terminal (FMCIB shown; swap for CT-CLIP):

```bash
export HT_CACHE_DIR=$HT_CACHE_ROOT/fmcib_lesion
export CONFIG=src/hypertrace/configs/experiments/fmcib.yaml
```

### 2a. Smoke test (1 run, ~30 s)

```bash
GEOMETRIES="poincare euclidean" FOLDS=0 SEEDS=0 DIMS=16 TAG=smoke bash scripts/run_local.sh train summary
```

### 2b. Tune, same budget for every arm (optional but recommended; ~1 h per dim)

```bash
TUNE_DIMS="16" GEOMETRIES="poincare euclidean" bash scripts/run_local.sh tune
```
Prints per-fold (use in the paper) and global selections; the global one gives a `--set` string for
`TRAIN_SET="..."` below. Tables: `$HT_OUTPUT_ROOT/tuning/*/`.

### 2c. Dimension sweep: Poincaré vs Euclidean (single space)

```bash
TAG=v3 DIMS="2 4 8 16 32 64" GEOMETRIES="poincare euclidean" bash scripts/run_local.sh train summary
```
2 geometries x 6 dims x 5 folds x 3 seeds = 180 runs, ~1.5 h. Add `sphere kstereo` to GEOMETRIES to
include them (they fit one space like the others).

### 2d. One geometry per concept group (product space)

`embed_dim` is the TOTAL, split across the 5 groups, so use dims >= 10:

```bash
TAG=v3 DIMS="10 20 40" \
GEOMETRIES="euclidean poincare sphere kstereo product:matched product:matched_no_sphere product:swapped product:all_euclidean product:learned" \
  bash scripts/run_local.sh train summary
```
9 arms x 3 dims x 5 folds x 3 seeds = 405 runs, ~3 h. Tune first with the same arms
(`TUNE_DIMS=20 ... run_local.sh tune`) if you will report tuned numbers.

Same commands for CT-CLIP: only change the two `export` lines.

## 3. Visualise the best ones

```bash
# the best split per dim (chosen on VALIDATION AUROC, same split for all arms), one figure per dim
TAG=v3 DIMS="10 20 40" GEOMETRIES="euclidean poincare product:matched product:swapped" bash scripts/run_local.sh best
# or a chosen fold/seed/dim
VIZ_DIMS="2 64" VIZ_FOLDS="0 3" VIZ_SEEDS="0" TAG=v3 bash scripts/run_local.sh viz
```
Output in `$HT_OUTPUT_ROOT/figures/`: `*.png` (rows = arms; product runs: one row per factor),
`*.html` (interactive, hover), `*_coords.csv`. Colour = concept group, marker = concept, names in the
legend. Nodules are grey by the label of `COLOR_BY` (default `margin_spiculated`, or `malignancy`).
Never choose the figure by test AUROC; that is cherry-picking.

## 4. Interpreting the results

`summary` prints mean +/- SD per arm and writes `$HT_OUTPUT_ROOT/all_runs.csv` (one row per run).
Send me `all_runs.csv` and the printed group table.

1. **Is the model learning?** Compare `test/auroc_macro` (the bottleneck) with the two probes.
   embedding ≈ input > bottleneck: projection fine, concept head loses information (tune `per_concept_tau`,
   `w_hierarchy`, `concept_lr`). embedding ≈ bottleneck < input: the projection is the bottleneck
   (raise `weight_decay`, `input_dropout`, `pca_dim`, or the dimension). `train_auroc − test_auroc` = overfitting.
2. **Poincaré vs Euclidean.** Per dim, take the paired difference (same fold and seed) over the 15 runs.
   Report it only if the sign is the same in most folds and the mean exceeds its SD. Expect the largest
   gap at the smallest dimensions; if it vanishes at 16+, say so (that is a finding, not a failure).
3. **Does the geometry fit the structure?** The group table, on each group's own metric:
   hierarchy groups (nodule type): `parent_child_violation`, `tree_map`, `edge_in_cone`, `depth_spearman`
   (Poincaré should win); exclusive groups (shape): `exclusive_violation`; size: `ordinal_spearman`.
4. **Product space.** `matched` should beat `swapped` and `all_euclidean` on the groups it assigns
   differently. `matched` ≈ `all_euclidean` = factor splitting helps, curvature does not. `matched` ≈
   `swapped` = the assignment hypothesis is not supported for this encoder.
5. **Learned κ.** `train/curvature/<group>` in `metrics_per_epoch.csv` (negative = hyperbolic). Trust the
   sign only if it is the same across folds and seeds and does not depend on `optim.curvature_lr`.
6. **Failed or missing runs.** `hypertrace-train` logs `!! failed` and the loop continues; re-run the
   same command to retry. Compare arms only on (dim, fold, seed) that finished for all of them.

## 5. Concepts: Lung-RADS and RadLex

The tree is `configs/ontology/lung_nodule.yaml`. Lung-RADS v2022 groups and thresholds (size, nodule
type, calcification) are already in it, with a `lungrads` note per node. RadLex IDs (`radlex.id`) are
**empty on purpose** and do not affect training. Fill them (RadLex needs a free RSNA licence):

```bash
# 1. download RadLex: BioPortal -> RADLEX -> Download -> CSV (free login), or the OWL file at radlex.org
hypertrace-radlex --source ~/Downloads/RADLEX.csv            # dry run -> radlex_report.csv
hypertrace-radlex --source ~/Downloads/RADLEX.csv --write    # fills unique exact matches only
# or, with a free BioPortal API key:  hypertrace-radlex --bioportal-key $BIOPORTAL_API_KEY --write
```
Open `radlex_report.csv`: pick ambiguous nodes by hand, check each filled ID. Also verify the
Lung-RADS notes against the official ACR v2022 document before citing them.

## 6. Troubleshooting

| message | fix |
|---|---|
| run folder refused (different settings than the saved config) | new `TAG=...` or new `HT_OUTPUT_ROOT` |
| `embed_dim = N is too small for 5 factors` | total dim >= 5 (use 10, 20, 40) |
| `No module named 'monai'` | `pip install -e ".[fmcib]"` |
| `FMCIB checkpoint ... does not match` | wrong file; re-download with `WHAT=fmcib bash scripts/download_weights.sh` |
| `only N% of the nodule lies inside the 50 mm cube` | nodule > 50 mm; FMCIB sees its centre only |
| CUDA out of memory in extract | `DEVICE=cuda` and close other jobs |