# Running HYPER-TRACE Part 2

Part A is one machine with one GPU, from the command line. Part B is the same on a SLURM cluster.
No step needs root.

```
install ─► download weights + data ─► manifest ─► extract (GPU, once) ─► probe ─► train grid ─► summarize
```

## Hardware

| step | needs | rough cost (estimates, not measured on real CT-CLIP) |
|---|---|---|
| extraction | GPU, frozen CT-CLIP **inference only**, batch 1, bf16 | a few GB of GPU memory. About 700 scans once; mostly limited by reading/resampling the NIfTI files (tens of minutes to a couple of hours) |
| training | GPU or CPU, ~33 k trainable parameters on cached vectors | seconds to a minute per run; the 30-run grid in well under an hour |

**A single RTX Blackwell GPU is enough.** No step trains CT-CLIP (its authors used an 80 GB A100 for
*training* CT-CLIP; inference is far lighter). The first real run should be
`EXTRACT_SET="limit_scans=3"`, which prints the actual peak GPU memory (see A4).
Blackwell needs a PyTorch build for **CUDA 12.8 or newer** (`cu128`). Older builds fail with
"no kernel image is available for execution". `scripts/install.sh` installs cu128 by default and
`scripts/check_env.py` runs a real kernel on the GPU to confirm it works.

---

## Part A: one machine, command line

### A1. Install (once, ~10 min)

Requirements: Linux, an NVIDIA driver recent enough for CUDA 12.8 (`nvidia-smi` shows "CUDA Version: 12.8"
or higher), Python ≥ 3.10, git, and ~70 GB free disk.

```bash
git clone <your repo> hypertrace && cd hypertrace     # or unzip hypertrace_part2.zip
nvidia-smi                                            # driver / CUDA version, GPU name and memory
bash scripts/install.sh                               # venv in ~/envs/hypertrace
```

If the system Python is older than 3.10, use `PYTHON=/path/to/python3.11 bash scripts/install.sh`, or
get one without root via conda/uv.

Then set where data and results go, once, in `scripts/env.sh`:

```bash
export HT_WORK="${HT_WORK:-$HOME/hypertrace_work}"    # the only line most people need to change
```

and **in every new terminal**:

```bash
cd hypertrace && source scripts/env.sh
```

`scripts/env.sh` activates the venv and sets these variables:

| variable | value |
|---|---|
| `HT_BUCKET` | `$HT_WORK/LIDC-IDRI-bucket` |
| `HT_DATA_ROOT` | `$HT_WORK/data` |
| `HT_CACHE_ROOT` | `$HT_WORK/cache` |
| `HT_OUTPUT_ROOT` | `$HT_WORK/outputs` |
| `HT_CTCLIP_CKPT`, `HT_CXR_BERT`, `HT_FMCIB_CKPT` | the weight files (below) |

It also turns Hugging Face downloads **off** for normal runs, so a run never fetches anything
behind your back. The download scripts switch them back on for themselves.

### A2. Download the weights (once)

CT-CLIP's weights are in the **gated** CT-RATE dataset repository (licence CC BY-NC-SA, non-commercial):

1. Log in to Hugging Face, open <https://huggingface.co/datasets/ibrahimhamamci/CT-RATE> and accept the terms.
2. Create a read token at <https://huggingface.co/settings/tokens>, then run `hf auth login`.
3. `bash scripts/download_weights.sh`

That fetches:

| file | where it goes | what it is |
|---|---|---|
| `models/CT-CLIP-Related/CT-CLIP_v2.pt` from `ibrahimhamamci/CT-RATE` | `$HT_WORK/models/models/CT-CLIP-Related/CT-CLIP_v2.pt` (= `$HT_CTCLIP_CKPT`) | CT-CLIP image + text model (the base contrastive model, not the fine-tuned VocabFine/ClassFine variants) |
| `microsoft/BiomedVLP-CXR-BERT-specialized` | `$HT_WORK/models/BiomedVLP-CXR-BERT-specialized` (= `$HT_CXR_BERT`) | tokenizer + BERT architecture CT-CLIP's text encoder is built on |

| `model_weights.torch` from Zenodo record 10528450 | `$HT_WORK/models/fmcib/model_weights.torch` (= `$HT_FMCIB_CKPT`) | FMCIB lesion encoder (Pai et al. 2024), not gated |

By hand:

```bash
hf download ibrahimhamamci/CT-RATE models/CT-CLIP-Related/CT-CLIP_v2.pt --repo-type dataset --local-dir $HT_MODELS
hf download microsoft/BiomedVLP-CXR-BERT-specialized --local-dir $HT_CXR_BERT
mkdir -p $(dirname $HT_FMCIB_CKPT) && curl -L -o $HT_FMCIB_CKPT \
  "https://zenodo.org/records/10528450/files/model_weights.torch?download=1"
```

`WHAT=fmcib bash scripts/download_weights.sh` fetches only FMCIB. It needs MONAI
(`pip install -e ".[fmcib]"`, included in `scripts/install.sh`).

### A3. Download the data (once)

```bash
ONLY_CSV=1    bash scripts/download_bucket.sh    # ~5 MB of CSVs
NEEDED_ONLY=1 bash scripts/download_bucket.sh    # + images and instance masks of the 701 scans used
```

Re-running resumes. Details and the raw `hf buckets` commands are in [DATA.md](DATA.md#2-downloading).

### A4. Check, then run

```bash
python scripts/check_env.py      # GPU + bf16 kernel, CT-CLIP / MONAI imports, weights present → "all good"
pytest -q                        # ~110 tests, ~1 min, no data/weights needed

bash scripts/run_local.sh manifest                               # → 1390 nodules, 701 scans, 694 patients
bash scripts/run_local.sh radiomics                              # hand-crafted baseline features (CPU, ~10 min)
EXTRACT_SET="input_mode=crop crop_size_mm=64 limit_scans=3" bash scripts/run_local.sh extract   # first real test
bash scripts/run_local.sh extract                                # CT-CLIP, 64 mm nodule crops; resumable
bash scripts/run_local.sh probe                                  # what the features carry vs diameter / radiomics
bash scripts/run_local.sh train                                  # {poincare, euclidean} × 5 folds × 3 seeds
bash scripts/run_local.sh summary                                # mean ± SD per arm, per-concept AUROC
bash scripts/run_local.sh viz                                    # concept-space figures (fold 0, seed 0)
```

**`HT_CACHE_DIR` decides which cache `extract` writes and `probe` / `train` read** (default
`$HT_CACHE_ROOT/ctclip_crop64`). `bash scripts/run_local.sh` with no argument runs all steps. It skips
work already done, so you can stop and restart it any time. Logs go to `$HT_WORK/logs/`. Use
`nohup bash scripts/run_local.sh > run.log 2>&1 &` or `tmux` for long runs.

The lesion-level encoder (FMCIB) end to end:

```bash
export HT_CACHE_DIR=$HT_CACHE_ROOT/fmcib_lesion
EXTRACT_SET="encoder.name=fmcib" bash scripts/run_local.sh extract     # 50 mm cube per nodule, ~15 min
bash scripts/run_local.sh probe
CONFIG=src/hypertrace/configs/experiments/fmcib.yaml bash scripts/run_local.sh train
```

Variants:

```bash
SEEDS="0" FOLDS="0" bash scripts/run_local.sh train                     # one quick run per geometry
DIMS="2 4 8 16 32 64" SEEDS="0" bash scripts/run_local.sh train         # dimension sweep (the key hyperbolic test)
EXTRACT_SET="input_mode=crop crop_size_mm=96" HT_CACHE_DIR=$HT_CACHE_ROOT/ctclip_crop96 bash scripts/run_local.sh extract
CONFIG=src/hypertrace/configs/experiments/tangent_concepts.yaml bash scripts/run_local.sh train   # ablation
CONFIG=src/hypertrace/configs/experiments/no_hierarchy_loss.yaml bash scripts/run_local.sh train
```

**Caches made before this version** (e.g. `ctclip_crop64` / `ctclip_crop96` from the old code) work as
they are: free concept points don't need prompt embeddings. Only the zero-shot column of the probe
and `model.concepts.init=text` need prompts for the new 26-node tree; refresh them in seconds with
`hypertrace-extract --set only_text=true cache_dir=$HT_CACHE_DIR`.

### A5. The individual commands (what `run_local.sh` calls)

```bash
hypertrace-build-manifest --bucket-dir $HT_BUCKET --out $HT_DATA_ROOT/manifest.csv
hypertrace-make-folds --manifest $HT_DATA_ROOT/manifest.csv --out $HT_DATA_ROOT/manifest_folds.csv
hypertrace-radiomics                                          # → $HT_CACHE_ROOT/radiomics.csv
hypertrace-extract --set input_mode=crop crop_size_mm=64 cache_dir=$HT_CACHE_ROOT/ctclip_crop64
hypertrace-extract --set encoder.name=fmcib                   # → $HT_CACHE_ROOT/fmcib_lesion/
hypertrace-zeroshot --cache-dir $HT_CACHE_ROOT/ctclip_crop64 --manifest $HT_DATA_ROOT/manifest_folds.csv \
                    --probe --radiomics $HT_CACHE_ROOT/radiomics.csv
hypertrace-train                                              # Poincaré, d = 64, fold 0, seed 0
hypertrace-train --set geometry.name=euclidean data.test_fold=3 seed=2 model.projector.embed_dim=8
hypertrace-train --config src/hypertrace/configs/experiments/fmcib.yaml
hypertrace-train --print-config --set optim.lr=1e-3           # show the resolved config, don't run
hypertrace-summarize --runs $HT_OUTPUT_ROOT/runs --out $HT_OUTPUT_ROOT/all_runs.csv --per-concept
```

Every command takes `--help`.

### A6. Visualising the concept space

```bash
R=$HT_OUTPUT_ROOT/runs
hypertrace-visualize --run $R/main_poincare_token_mean_d64_f0_s0 --run $R/main_euclidean_token_mean_d64_f0_s0
hypertrace-visualize --run <run> --color-by malignancy --split test --out figs/
hypertrace-visualize --run <run> --full-disk --labels          # concept names on the plot too
VIZ_DIMS="2 64" VIZ_FOLDS="0 3" VIZ_SEEDS="0" bash scripts/run_local.sh viz   # one figure per combination
```

It rebuilds each model from `config.yaml` + `checkpoints/best.pt` and the embedding cache, and embeds
**every** nodule (`--split` filters to train/val/test). Output in `--out` (default `./concept_space_figures`):

| file | content |
|---|---|
| `concept_space.png` | one row per run, four panels (below) |
| `<run>.html` | interactive radial and disk views; hover shows nodule id, split, depth and label (self-contained, works offline) |
| `<run>_coords.csv` | 2-D coordinates of every concept and nodule, for your own figures |

The four panels in `concept_space.png`:

1. **Radial view.** Radius = the point's *exact* depth: hyperbolic d(0, x) for Poincaré, ‖x‖ for Euclidean. Angle = 2-D PCA of its direction in the tangent space at the origin.
2. **Disk view.** Poincaré: tangent PCA, then the exp map into a 2-D Poincaré disk. Opens zoomed to the data, or `--full-disk`. Euclidean: plain PCA.
3. **Concept depth vs depth in the tree**, with Spearman ρ.
4. **Lesion depth** for concept present vs absent.

Concepts: hue = concept group (size, shape, margin, nodule type, calcification), marker shape and
shade = the concept within its group, names in the legend below the figure (`--labels` also writes them
on the plot). Hue alone cannot separate five groups for every reader, so a concept is always identified
by hue *and* marker. Hollow markers are unscored nodes; grey hexagons are the trunk (chest → nodule).
In the HTML, clicking a legend entry hides that concept in both panels.
Nodules: grey ramp of the soft label of `--color-by` (any scored concept, default `margin_spiculated`,
or `malignancy`), so they do not compete with the concept colours.
Plain PCA/t-SNE on raw ball coordinates is avoided on purpose: it treats the ball as flat and hides
the hierarchy. Edges are straight chords; in the disk the true geodesics are arcs.

### A7. Reading the probe (step `probe`)

One row per (feature, concept), printed as tables and saved to `<cache>/zeroshot.csv`:

- `probe`: AUROC of a patient-grouped logistic-regression probe, for every readout in the cache
  (`token_mean`, `masked_latent`, `global_latent` for CT-CLIP; `lesion_embedding` for FMCIB);
- baseline rows: `diameter_only` (AUROC of the Part-1 diameter itself, either direction) and
  `radiomics` (probe on the 21 hand-crafted features from `hypertrace-radiomics`);
- `zeroshot`: AUROC of raw CT-CLIP image–text similarity (CT-CLIP caches whose prompts match the tree).

The last table is the mean probe AUROC per feature. **A learned feature is only worth training on if it
beats `diameter_only` and `radiomics` on the morphology concepts** (margin, shape, texture,
calcification). The size bins are defined by the diameter, so `diameter_only` is 1.0 there by construction.

### A8. Training diagnostics and the matched tuning grid

Every run now reports, in `metrics.json` and the log line:

| number | meaning |
|---|---|
| `train/auroc_macro` | the model on its own training split: **train − test = overfitting** |
| `test/auroc_macro` | the concept bottleneck (distance head) |
| `test/probe_embedding/auroc_macro` | logistic probe on the learned lesion embedding (tangent coordinates) |
| `test/probe_input/auroc_macro` | the same probe on the input features, same split: the ceiling (cached per fold in `$HT_OUTPUT_ROOT/reference/`) |

How to read them: **embedding ≈ input > head** → the projection keeps the information and the
distance-to-concept head loses it (tune the head: `per_concept_tau`, `w_hierarchy`, `concept_lr`).
**embedding ≈ head < input** → the projection is the bottleneck (regularise: `weight_decay`,
`input_dropout`, `pca_dim`; or raise `embed_dim`). `metrics_per_epoch.csv` has train and val AUROC per
epoch, plus each loss part (`val/loss_concept`, `val/loss_hierarchy`, ...).

Model selection now uses `trainer.select_metric: val/loss_concept` (concept BCE only) from
`select_from_epoch: 20` on. The old behaviour (total loss incl. tree terms) is
`--set trainer.select_metric=val/loss`. The Euclidean twin is no longer capped (`clip_r.euclidean: null`).

**These defaults change results, so new runs need new run folders.** `hypertrace-train` refuses to resume
a folder trained with different settings. Use `TAG=v3` (run-name suffix `_tv3`) or a new
`HT_OUTPUT_ROOT`.

Tuning, with the same budget for every geometry (`hypertrace-tune`; not part of `all`):

```bash
export HT_CACHE_DIR=$HT_CACHE_ROOT/fmcib_lesion
CONFIG=src/hypertrace/configs/experiments/fmcib.yaml TUNE_DIMS="4 16" bash scripts/run_local.sh tune
# two GPUs / two shells:  TUNE_SHARD=0/2 ...   and   TUNE_SHARD=1/2 ...
```

16 trials from `configs/tune/default.yaml` (weight decay, input dropout, PCA, lr, concept lr, hierarchy
weight, per-concept τ; trial 0 = the base config) × 2 geometries × dims × 5 folds. At ~25 s per run,
`TUNE_DIMS=16` is about 1 h 10 min. Selection uses validation only:

- **per-fold** (use these numbers in the paper): for each fold, the trial with the best validation
  AUROC on that fold; its test AUROC is reported;
- **global**: one trial per geometry and dim, for a final sweep. It prints the `--set` string:
  `TAG=tuned TRAIN_SET="<that string>" DIMS="2 4 8 16 32 64" bash scripts/run_local.sh train`.
  Its test numbers are slightly optimistic (a fold's validation split is another fold's test split).

Tables: `$HT_OUTPUT_ROOT/tuning/<name>/{trials,runs,selected_per_fold,selected_global}.csv`.

### A9. Which geometry for which concept group (step 3)

New arms, all on the same cache and the same total dimension:

```bash
export HT_CACHE_DIR=$HT_CACHE_ROOT/fmcib_lesion
CONFIG=src/hypertrace/configs/experiments/fmcib.yaml TAG=v3 DIMS="10 20 40" \
GEOMETRIES="euclidean poincare sphere kstereo product:matched product:matched_no_sphere product:swapped product:all_euclidean product:learned" \
  bash scripts/run_local.sh train viz summary
```

- `sphere`, `kstereo` (learned κ from 0): one space, like `euclidean` / `poincare`.
- `product:<assignment>`: one factor per concept group (size, nodule type, shape, margin, calcification),
  `embed_dim` split equally (so at least 5; use 10, 20, 40 for all arms). Assignments are in
  `train.yaml` → `geometry.product.assignments` (IMPLEMENTATION.md §4.6).
- 9 arms × 3 dims × 5 folds × 3 seeds = 405 runs, about 3 h at ~25 s each. Tune first with the same
  arms: `GEOMETRIES="... product:matched ..." TUNE_DIMS=20 bash scripts/run_local.sh tune`.
- `summary` now prints a per-group table (`--groups`): read each group on the metric that matches its
  structure (hierarchy → `parent_child_violation`, `tree_map`; exclusive → `exclusive_violation`;
  size → `ordinal_spearman`, `ordinal_dist_spearman`).
- Figures of a product run have one row per factor, each in its own geometry (Poincaré disk, sphere in
  stereographic view with the equator dashed, or flat).
- `metrics_per_epoch.csv` logs the curvature of every factor (`train/curvature/<group>`; signed:
  −1 = Poincaré c = 1). For learned κ, also try `--set optim.curvature_lr=1e-3`.

---

## Part B: SLURM cluster

The same steps as batch jobs. Setup on a login node: `bash slurm/setup_env.sh` (loads modules, then
runs `scripts/install.sh`). Set `HT_WORK` to scratch and fix the module names in `scripts/env.sh`.
Downloads (A2, A3) also run on the login node, because compute nodes are usually offline.

```bash
sbatch slurm/extract.sbatch                                   # 8 GPU tasks sharded by scan (resumable)
sbatch slurm/extract.sbatch --set input_mode=crop crop_size_mm=64 cache_dir=$HT_CACHE_ROOT/ctclip_crop64
sbatch slurm/extract.sbatch --set encoder.name=fmcib
sbatch slurm/train_array.sbatch                               # 30 tasks: geometry × fold × seed
CONFIG=src/hypertrace/configs/experiments/fmcib.yaml sbatch slurm/train_array.sbatch
bash slurm/pipeline.sh                                        # manifest → folds → extract → train, chained
```

Requeued tasks resume exactly from `checkpoints/last.pt`.

---

## Configs

Plain YAML in `src/hypertrace/configs/`. No framework, three conventions:

- **`base: ../train.yaml`**: inherit and change only what differs. Mappings merge; lists and values replace.
- **`${VAR:-default}`**: environment variable with a fallback. An unset variable without a default is an error.
- **`--set key.sub=value`** on the command line, parsed as YAML (`1e-3`, `true`, `[1, 2]`).
  **Unknown keys are rejected**, so a typo can't silently do nothing.

| file | purpose |
|---|---|
| `train.yaml` | one training run: CT-CLIP 64 mm crops, `token_mean`, free concept points + Riemannian Adam, d = 64 |
| `extract.yaml` | encoders (CT-CLIP, FMCIB), preprocessing, input mode, `limit_scans`, sharding |
| `experiments/fmcib.yaml` | lesion-level encoder cache, `lesion_embedding` |
| `experiments/tangent_concepts.yaml` | ablation: concept points as tangent vectors + AdamW instead of Riemannian Adam |
| `experiments/text_init.yaml` | ablation: concept points initialised from CT-CLIP prompt embeddings |
| `experiments/no_hierarchy_loss.yaml` | ablation: no entailment losses |
| `experiments/learn_curvature.yaml` | ablation: learn the ball curvature |
| `experiments/global_latent.yaml` | ablation: no grounding (whole-scan CT-CLIP embedding) |
| `experiments/v1_text_anchored.yaml` | the first real-data setup, kept for reproducibility |
| `experiments/smoke_fake.yaml` | dry run with the fake-encoder cache |
| `experiments/tree_v2.yaml` | the v2 concept tree, for comparison with earlier runs |
| `ontology/lung_nodule.yaml` | v3 concept ontology: 50 nodes, clinical terms, Lung-RADS roles, RadLex IDs |
| `labels/lidc.yaml` | which ontology nodes LIDC labels, and how (17 scored); the default `concepts_file` |
| `tune/default.yaml` | search space for `hypertrace-tune` |
| `concepts/lidc.yaml` | concept tree v2 (26 nodes, 17 scored), prompts, label rules |
| `concepts/lidc_v1.yaml` | concept tree v1 (18 nodes, 10 scored) |
| `bucket/lidc_hf.yaml` | bucket CSV column map and term → code tables |

To add an experiment, copy one of the small files and give it a new `experiment:` name. Run folders
are named after it, so arms never overwrite each other. Editing the ontology or a label mapping never
needs re-extraction; for CT-CLIP caches, `hypertrace-extract --set only_text=true cache_dir=...` refreshes
the prompt embeddings (only used for zero-shot and `init: text`; needed once for the v3 tree).

A new dataset needs only a new file in `labels/` (`ontology: ../ontology/lung_nodule.yaml` plus its
`labels:`) and `--set concepts_file=<it>`; a new organ, a new ontology file.

**RadLex IDs.** Download RadLex after accepting RSNA's licence (BioPortal → RADLEX → Download → CSV, or
the OWL file from radlex.org), then:

```bash
hypertrace-radlex --source ~/Downloads/RADLEX.csv               # dry run: radlex_report.csv
hypertrace-radlex --source ~/Downloads/RADLEX.csv --write       # fill unique exact matches
# or: hypertrace-radlex --bioportal-key $BIOPORTAL_API_KEY --write
```

Check every filled ID and pick the ambiguous ones by hand from the report before citing them.

## Outputs of one run

`$HT_OUTPUT_ROOT/runs/<experiment>_<geometry>_<feature>_d<dim>[_t<tag>]_f<fold>_s<seed>/`

| file | content |
|---|---|
| `config.yaml` | resolved config + concept tree + `_meta` (command, git commit, library versions, GPU, SLURM job id) |
| `metrics_per_epoch.csv` | training curves: loss parts, train and val AUROC per epoch |
| `metrics.json` | train, val and test metrics of the best checkpoint, selection metric, best epoch, epochs run, probe diagnostics, hierarchy metrics of the initial concept points (`init`) |
| `predictions_test.csv` | per test-fold nodule: `prob_<concept>`, `soft_<concept>`, `lesion_depth` (input to Part 3) |
| `concept_space.pt` | Poincaré positions and depths of all concept nodes, curvature |
| `feature_norm.pt` | training-split mean / SD used to standardise the inputs (and PCA axes if `data.pca_dim`) |
| `checkpoints/best.pt`, `last.pt` | weights; `last.pt` also holds optimiser/scheduler/RNG state |

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `no kernel image is available` / `sm_120 is not compatible` | PyTorch without Blackwell support: reinstall with `TORCH_INDEX=https://download.pytorch.org/whl/cu128` |
| `check_env.py`: CUDA GPU not visible | driver too old for CUDA 12.8 (`nvidia-smi`), or `CUDA_VISIBLE_DEVICES` hides it |
| `CT-CLIP checkpoint ... does not match the model` | wrong file, or `vector-quantize-pytorch` isn't 1.1.2 (`pip install vector-quantize-pytorch==1.1.2`) |
| `to_visual_latent expects ... inputs` | CT-CLIP internals differ from the wrapper's assumptions ([IMPLEMENTATION.md §7](IMPLEMENTATION.md#7-open-checks)) |
| download 401/403 for CT-RATE | accept the dataset terms on the website, then `hf auth login` |
| `hf: command not found` / no `hf buckets` | `pip install -U huggingface_hub` inside the venv |
| CUDA out of memory in extraction | send the `limit_scans=3` log; options are `input_mode=crop` or reducing the canvas |
| `ConfigError: ... unknown key` | typo in `--set`; the message lists valid keys |
| `ConfigError: environment variable $X is not set` | `source scripts/env.sh` first |
| `IntensityError` | scan already normalised, or `hu_already_normalized` set wrongly |
| `InstanceLabelError` | instance-mask ids don't match `nodules.csv` for that scan |
| `Cache ... was built with different settings` | use a new cache folder for new extraction settings |
| `text.pt was built for a different concept tree` | only for `concepts.mode=text` / `init=text`: `hypertrace-extract --set only_text=true cache_dir=...` |
| `FMCIB checkpoint ... does not match` | wrong file or a different MONAI ResNet layout; send the message (lists missing keys) |
| `No module named 'monai'` | `pip install -e ".[fmcib]"` |
| `only N% of the nodule lies inside the 50x50x50-voxel cube` | nodule > 50 mm; FMCIB sees only its centre (as in the original method) |
| `Non-finite loss` | lower `optim.lr`; check `geometry.poincare.curvature_bounds` |
