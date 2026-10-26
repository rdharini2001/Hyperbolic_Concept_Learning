# Data: the LIDC-IDRI bucket

Source: [`Dharini24/LIDC-IDRI-bucket`](https://huggingface.co/buckets/Dharini24/LIDC-IDRI-bucket) on
Hugging Face. It is a re-hosted NIfTI version of LIDC-IDRI: 1,010 patients, 1,018 chest CT scans, about
71 GB, 10,945 files. The annotations come from the QIICR DICOM SEG/SR re-encoding of the four-reader LIDC reads.

## 1. What is in the bucket, and what Part 2 uses

| path | content | used by Part 2 |
|---|---|---|
| `images/<case>.nii.gz` | CT, **int16 Hounsfield units, native geometry** (not normalised, not resampled) | yes |
| `masks_instance/<case>.nii.gz` | uint16, one id per clustered nodule (50 % consensus); id = `nodule_id` in `nodules.csv` | yes |
| `annotations.csv` | one row per reader annotation (6,859): SR characteristics, diameter, mask path | yes |
| `nodules.csv` | one row per clustered nodule (2,651): `n_annotations`, `in_default_gt`, `consensus_voxels`, median characteristics | yes |
| `scans.csv` | one row per scan (1,018): spacing, slice gaps, manufacturer, LUNA16 membership | yes |
| `train.jsonl` | loader manifest, one row per scan | no (same information as `scans.csv`) |
| `masks/`, `masks_annotation/`, `masks_annotation_count/` | default-GT, per-annotation and count masks | no |
| `aux/`, `preview/` | original XML/metadata from TCIA, PNG previews | no |

`<case>` is `<PatientID>_<last 10 digits of the SeriesInstanceUID>`, e.g. `LIDC-IDRI-0001_9720603192`.
Eight patients have two scans, so **patient** (not scan) is the unit for data splits.

## 2. Downloading

Buckets are a newer Hub storage type, downloaded with the `hf buckets` commands of a recent
`huggingface_hub` ([Hub docs: buckets](https://huggingface.co/docs/huggingface_hub/en/guides/buckets)).
`scripts/install.sh` installs it. Check with `hf buckets --help`.

On a cluster, run downloads on a **login or data-transfer node**, because compute nodes are usually offline.
`scripts/download_bucket.sh` wraps everything below and writes to `$HT_BUCKET` (set in `scripts/env.sh`).

```bash
source scripts/env.sh

# (a) metadata only, ~5 MB: enough to build the manifest and look at labels
ONLY_CSV=1 bash scripts/download_bucket.sh

# (b) recommended: metadata + only the scans the manifest uses (701 of 1,018 scans)
NEEDED_ONLY=1 bash scripts/download_bucket.sh

# (c) everything Part 2 could use: all of images/ and masks_instance/
bash scripts/download_bucket.sh
```

The same commands by hand:

```bash
hf buckets cp   hf://buckets/Dharini24/LIDC-IDRI-bucket/nodules.csv      $HT_BUCKET/nodules.csv
hf buckets cp   hf://buckets/Dharini24/LIDC-IDRI-bucket/annotations.csv  $HT_BUCKET/annotations.csv
hf buckets cp   hf://buckets/Dharini24/LIDC-IDRI-bucket/scans.csv        $HT_BUCKET/scans.csv
hf buckets sync hf://buckets/Dharini24/LIDC-IDRI-bucket/images           $HT_BUCKET/images
hf buckets sync hf://buckets/Dharini24/LIDC-IDRI-bucket/masks_instance   $HT_BUCKET/masks_instance
hf buckets list Dharini24/LIDC-IDRI-bucket -R | head      # browse
```

`sync` is incremental: re-running it after an interruption only fetches what is missing. If the bucket
is private or gated for your account, run `hf auth login` first.

> I could not reach Hugging Face from the environment where this code was written. The commands
> follow the documented `hf buckets` interface but have not been run against this bucket. The
> `NEEDED_ONLY` path uses `huggingface_hub.download_bucket_files` from the same docs.

Expected layout:

```
$HT_BUCKET/
  annotations.csv  nodules.csv  scans.csv
  images/LIDC-IDRI-0001_9720603192.nii.gz ...
  masks_instance/LIDC-IDRI-0001_9720603192.nii.gz ...
```

## 3. From CSVs to the manifest

```bash
hypertrace-build-manifest --bucket-dir $HT_BUCKET --out $HT_DATA_ROOT/manifest.csv
hypertrace-make-folds     --manifest $HT_DATA_ROOT/manifest.csv --out $HT_DATA_ROOT/manifest_folds.csv
```

The column map lives in `src/hypertrace/configs/bucket/lidc_hf.yaml` and was verified against the real
CSVs on 2026-10-13. The builder does five things:

1. **Reads ratings from the `*_term` columns, not `*_score`.** The score columns are empty whenever the SR
   stored a named concept instead of "N out of 5":

   | characteristic | `*_score` empty | named terms |
   |---|---|---|
   | margin | 47 % | Indistinct margin = 1, Circumscribed margin = 5 |
   | texture | 86 % | non-solid = 1, part-solid = 3, solid = 5 |
   | sphericity | 58 % | linear = 1, ovoid = 3, round = 5 |
   | calcification | 100 % | Popcorn 1, Laminated 2, Solid 3, Non-central 4, Central 5, Absent 6 |
   | internal structure | 100 % | Soft tissue 1, Fluid 2, Adipose 3, Air 4 |

   Using `*_score` would silently drop every extreme category. Terms are mapped to the LIDC integer codes,
   "N out of 5 (...)" is parsed from its leading number, an unknown term stops the build, and every parsed
   term is cross-checked against `*_score` where one exists. All 6,859 annotations agree.
2. **Pivots to one row per nodule** with reader columns `spiculation_r1 … spiculation_rK` (K up to 8,
   because the clustering merged a few extra annotations). The label code averages over whatever is
   present.
3. **Keeps the bucket's default ground truth**: nodules with ≥ 3 annotations (`in_default_gt`).
4. **Drops scans with irregular slice gaps** (`z_gap_max − z_gap_min > 0.1 mm`). A single NIfTI affine
   misplaces slices there, so resampled geometry would be wrong.
5. **Writes paths** to `images/` and `masks_instance/` relative to the bucket root, plus `spacing_*` and
   `consensus_voxels`. Extraction cross-checks both against the image files.

Result on the current release:

| | count |
|---|---|
| clustered nodules | 2,651 |
| default GT (≥ 3 annotations) | 1,392 |
| dropped (irregular slice gaps, 2 scans) | 2 |
| **manifest** | **1,390 nodules · 701 scans · 694 patients** |
| annotations per nodule | 3: 480 · 4: 896 · 5–8: 14 |
| patients with GT nodules on two scans | 7 |

`hypertrace-make-folds` assigns 5 patient-level folds, stratified on reader-mean malignancy ≥ 3
(727 / 1,390 nodules). That gives 277–280 nodules per fold. Build the folds once and keep the file:
every arm and seed must use the same split.

## 4. Concept labels

Defined per node in `src/hypertrace/configs/concepts/lidc.yaml`:

| concept | source | soft label (training target) |
|---|---|---|
| size_large / size_small | `diameter_mm` ≥ 8 mm (placeholder cut-off) | 0/1 |
| margin_smooth / margin_irregular | `margin` 1–5 | reader mean rescaled to [0, 1] (inverted for irregular) |
| margin_lobulated | `lobulation` 1–5 | reader mean rescaled |
| margin_spiculated | `spiculation` 1–5 | reader mean rescaled |
| texture_solid / texture_subsolid | `texture` 1–5 | reader mean rescaled (inverted for subsolid) |
| calcification_present / absent | `calcification` 1–6 | fraction of readers coding 1–5 / coding 6 |

The binary view used for AUROC/F1 is soft ≥ 0.5 (reader mean ≥ 3). Prevalence in the manifest:
large 63 %, spiculated 10 %, lobulated 11 %, irregular margin 16 %, subsolid 9 %, calcified 13 %. The rare
concepts are re-weighted in the loss (`use_pos_weight`); report AUPRC alongside AUROC.

## 5. What happens to the images (summary)

Extraction does all normalisation, because the bucket images are raw:
1. Reorient each scan to canonical RAS and store it as (z, y, x).
2. Check the intensity range against `hu_already_normalized`.
3. Clip HU to [−1000, 1000] and divide by 1000. This happens before resampling, so out-of-FOV padding doesn't smear into tissue.
4. Resample to CT-CLIP spacing (z 1.5 / y 0.75 / x 0.75 mm).
5. Place the scan in CT-CLIP's 240×480×480 canvas.
6. Build each nodule's mask from `masks_instance == nodule id` and check its voxel count against `consensus_voxels`.

Details: [IMPLEMENTATION.md §2](IMPLEMENTATION.md#2-preprocessing-and-grounding-extraction).

## 6. Caveats to state in the paper

- LIDC malignancy and characteristics are **radiologist ratings, not pathology**.
- **LUNA16 is a subset of LIDC-IDRI**, so it can't serve as external validation.
- The 8 mm size cut-off is a placeholder; align it with the Lung-RADS category you report.
- Reader identity is not consistent across nodules in LIDC, so "reader k" just means the k-th annotation of that nodule.
