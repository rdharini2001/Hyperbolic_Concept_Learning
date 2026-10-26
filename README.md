# HYPER-TRACE · Part 2: Poincaré concept bottleneck for 3D chest CT

Code for the middle row of the overview figure: a frozen 3D CT encoder (CT-CLIP on zoomed
nodule crops, or the lesion-level FMCIB model) is read out at each lung nodule, and the result is
placed in a **Poincaré-ball concept space** together with a radiology concept tree (26 nodes,
17 scored concepts: Lung-RADS size bins, shape, margin, nodule type, calcification). Every tree
node is a free point on the ball, trained with Riemannian Adam; concept scores are read off as
geodesic distances. Neuro-symbolic rules and the malignancy-risk head (Part 3) consume this
module's outputs.

```
bucket scan (raw HU) ─► reorient · HU window · resample ─► nodule crop ─► frozen encoder ─► lesion embedding
masks_instance == nodule id ──► crop centre / token weights ───────────────┘                  │ (cached once)
                                                                                               │
                  standardise → learnable linear map → tangent vector → exp map onto the Poincaré ball
                                                                                               │
concept tree (26 nodes) ─► one free point per node on the ball (Riemannian Adam) ──────────────┤
             hyperbolic entailment cones on the tree   +   score_k = σ(−d_ℍ(lesion, concept_k)/τ + b_k)
                                                        │
                          predictions_test.csv · concept_space.pt  ──►  Part 3 (rules, risk head)
```

**Main experiment:** an otherwise identical Euclidean concept space (`--set geometry.name=euclidean`) vs. the
Poincaré one, across concept-space dimensions (`DIMS="2 4 8 16 32 64"`). Every learned feature is compared
with a diameter-only and a hand-crafted radiomics baseline (`hypertrace-radiomics`, `hypertrace-zeroshot`).
Both geometries are tuned with the same budget and selected on validation only (`hypertrace-tune`), and
every run reports linear probes on its input and its learned embedding next to the concept head.
Beyond one space: a sphere, a learned curvature of either sign (`kstereo`), and a product space with one
factor per concept group (`product:matched` vs `product:swapped`), read on per-group metrics.

## Documentation

| file | read it for |
|---|---|
| [docs/DATA.md](docs/DATA.md) | downloading the LIDC-IDRI bucket, what Part 2 uses, how the manifest and labels are built |
| [docs/RUNNING.md](docs/RUNNING.md) | short guide: setup, extract, train (FMCIB / CT-CLIP, dims, product arms), figures, interpreting results, RadLex |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | how each part works (preprocessing, grounding, Poincaré geometry, Riemannian Adam, losses, training loop), design decisions, verification, open checks |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | manuscript-style methods and references |

## Quick start (one machine, one GPU)

```bash
bash scripts/install.sh                       # once: venv, PyTorch cu128 (Blackwell-ready), CT-CLIP, hf CLI
$EDITOR scripts/env.sh                        # set HT_WORK (where data/outputs go)
source scripts/env.sh                         # every new terminal
bash scripts/download_weights.sh              # CT-CLIP_v2.pt (gated: accept CT-RATE terms, `hf auth login`) + CXR-BERT + FMCIB
ONLY_CSV=1 bash scripts/download_bucket.sh    # CSVs (~5 MB)
NEEDED_ONLY=1 bash scripts/download_bucket.sh # + images/masks of the 701 scans used
python scripts/check_env.py && pytest -q      # GPU kernel, imports, weights; ~110 tests

bash scripts/run_local.sh manifest
bash scripts/run_local.sh                     # radiomics → extract (CT-CLIP 64 mm crops) → probe → train → viz → summary

export HT_CACHE_DIR=$HT_CACHE_ROOT/fmcib_lesion                                  # lesion-level encoder
EXTRACT_SET="encoder.name=fmcib" bash scripts/run_local.sh extract
bash scripts/run_local.sh probe && CONFIG=src/hypertrace/configs/experiments/fmcib.yaml bash scripts/run_local.sh train
```

On a SLURM cluster: `bash slurm/setup_env.sh`, then `bash slurm/pipeline.sh`.

## Layout

```
src/hypertrace/
  config.py        plain-YAML configs: base: inheritance, ${ENV:-default}, strict --set overrides
  training.py      plain PyTorch training loop (early stopping, best/last checkpoints, exact resume)
  optim.py         AdamW + Riemannian Adam for concept points on the ball
  configs/         train.yaml, extract.yaml, experiments/*.yaml, concepts/lidc.yaml, bucket/lidc_hf.yaml
  geometry/        poincare.py (closed-form ball operations incl. exp/transport), manifolds.py (Poincaré / Euclidean)
  concepts/        hierarchy.py (concept tree), labels.py (reader ratings → soft / binary labels)
  encoders/        base.py (VLM + lesion-encoder interfaces), ctclip.py, fmcib.py, fake.py, preprocess.py, pooling.py
  data/            manifest, volumes (NIfTI), folds, embedding_cache, dataset, synthetic (bucket-format test data)
  models/          projector, concept_head, losses, model (ConceptSpaceModel)
  eval/            metrics
  viz.py           geometry-aware 2-D layouts of a trained concept space
  cli/             build_manifest, make_folds, radiomics, extract, zeroshot, train, summarize, visualize
scripts/           env.sh, install.sh, check_env.py, download_weights.sh, download_bucket.sh, run_local.sh
slurm/             setup_env.sh, extract.sbatch, train_array.sbatch, pipeline.sh
tests/             config, geometry, Riemannian Adam (vs geoopt), Poincaré-model checks, FMCIB loading, labels,
                   preprocessing, manifest, pooling, end to end
```

## Status

- First real run (CT-CLIP whole-scan `masked_latent`, text-anchored concepts, v1 tree): concept
  AUROC at chance in both geometries. The CT-CLIP features carried no nodule information at
  15 mm token scale (probe below the diameter alone), and text-anchored concepts collapsed to one
  point. This version addresses both (zoomed crops or a lesion-level encoder; free concept points); see
  "Lessons from the first real runs" in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).
- Tested end to end on synthetic data in the exact bucket layout, with stand-in encoders.
  FMCIB has not been run with its real weights yet; its loader fails loudly on a mismatch.
- The manifest built from the real bucket CSVs has 1,390 nodules on 701 scans from 694 patients.
- The CT-CLIP wrapper's assumptions about its internals are listed under "Open checks" in
  [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md). A load-time self-check fails loudly if they are wrong.

Data: LIDC-IDRI (Armato et al., Med. Phys. 2011), CC BY 3.0, re-hosted as
[Dharini24/LIDC-IDRI-bucket](https://huggingface.co/buckets/Dharini24/LIDC-IDRI-bucket) from the
QIICR DICOM SEG/SR re-encoding (Fedorov et al., Med. Phys. 2020). Cite both when publishing.