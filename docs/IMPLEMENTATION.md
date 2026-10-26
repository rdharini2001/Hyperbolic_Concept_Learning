# Implementation notes

What the code does, where it lives, and why it is built this way. Formulas use curvature −c (c > 0),
a ball of radius 1/√c, and the Euclidean norm ‖·‖.

## 0. Lessons from the first real runs (what changed in v2)

The first real run (CT-CLIP, whole-scan window, `masked_latent`, concepts = CT-CLIP prompt embeddings
through the image map, 10-concept tree) gave test macro AUROC 0.51 ± 0.01 in **both** geometries
(Brier 0.25, the value of predicting 0.5 everywhere). Two independent causes:

1. **The features did not see the nodules.** Grounding was correct (mask coverage vs. diameter,
   Spearman 0.94), but patient-grouped linear probes on every CT-CLIP readout were near chance and
   below the diameter alone (spiculation: diameter 0.86, best probe 0.65). At CT-CLIP's native scale a
   token is a ~15 mm cube, so a 6–9 mm nodule fills a few percent of one token. Lesion depth
   separated concept labels on the training split (AUROC ≈ 0.9) and not on test (≈ 0.5): memorisation.
2. **Text-anchored concepts collapsed.** CT-CLIP embeddings of the prompts are nearly identical, so
   after the shared linear map all ten concepts sat within 0.015 of the same depth (0.185–0.200)
   while lesions spread over 0.25–1.37: each lesion was equally far from every concept.

v2 changes: inputs at nodule scale (CT-CLIP on 64/96 mm crops, and a lesion-level encoder, FMCIB);
features z-scored on the training split; **every concept is a free point** on the ball, initialised
from the tree and trained with **Riemannian Adam**; a 26-node tree with 17 clinically used concepts;
diameter and radiomics baselines in the probe; tree-fidelity and consistency metrics; run names carry
the dimension for the dimension sweep. `experiments/v1_text_anchored.yaml` reproduces v1.

## 1. Overview and code map

```
                 ┌──────────────── extraction (once, GPU) ────────────────┐   ┌──── training (per run, minutes) ────┐
bucket CSVs ─► manifest ─► folds        scan ─► normalise ─► CT-CLIP ─► tokens ─► readout ─► cache ─► concept space ─► scores
             build_manifest  make_folds   volumes  preprocess   encoders    pooling   embedding_cache   model + training
```

| stage | module | entry point |
|---|---|---|
| bucket CSVs → manifest | `cli/build_manifest.py`, `configs/bucket/lidc_hf.yaml` | `hypertrace-build-manifest` |
| patient-level folds | `data/folds.py` | `hypertrace-make-folds` |
| load scans / masks | `data/volumes.py` | — |
| normalise, resample, window | `encoders/preprocess.py` | — |
| frozen encoders (CT-CLIP, FMCIB, fakes) | `encoders/base.py`, `ctclip.py`, `fmcib.py`, `fake.py` | — |
| hand-crafted baseline features | `cli/radiomics.py` | `hypertrace-radiomics` |
| mask → lesion readout | `encoders/pooling.py` | — |
| extraction loop + cache | `cli/extract.py`, `data/embedding_cache.py` | `hypertrace-extract` |
| feature check | `cli/zeroshot.py` | `hypertrace-zeroshot` |
| concept tree, labels | `concepts/hierarchy.py`, `concepts/labels.py`, `configs/concepts/lidc.yaml` | — |
| Poincaré / Euclidean geometry | `geometry/poincare.py`, `geometry/manifolds.py` | — |
| Riemannian Adam | `optim.py` | — |
| projector, head, losses | `models/projector.py`, `concept_head.py`, `losses.py`, `model.py` | — |
| training loop, evaluation | `training.py`, `eval/metrics.py`, `data/dataset.py` | `hypertrace-train` |
| aggregation | `cli/summarize.py` | `hypertrace-summarize` |
| figures | `viz.py`, `cli/visualize.py` | `hypertrace-visualize` |
| configs | `config.py`, `configs/*.yaml` | — |

The encoders stay **frozen**. They run once per nodule (or scan window) and their outputs are cached.
Everything that is trained (a linear map, 26 concept points, scales, biases, temperature: ≈ 34 k
parameters for CT-CLIP `token_mean` at d = 64, ≈ 264 k for FMCIB's 4,096-d input) runs on cached vectors.

## 2. Preprocessing and grounding (extraction)

Per scan (`cli/extract.py`):

1. **Load** `images/<case>` and `masks_instance/<case>` with nibabel and reorient both to the closest
   canonical orientation (RAS+). Arrays become (z, y, x) and spacing (z, y, x) mm, so every scan has
   the same anatomical axis order whatever the scanner wrote. Header spacing must match `scans.csv`
   (tolerance 0.02 mm).
2. **Intensity guard.** If `hu_already_normalized=false` but the CT range spans < 50 units (or the
   reverse), extraction stops. This catches double or missing normalisation.
3. **HU window**: clip to [−1000, 1000] and divide by 1000, *before* resampling, so −2048/−3024
   out-of-FOV padding can't bleed into tissue during interpolation.
4. **Resample** trilinearly to CT-CLIP's spacing (z 1.5, y 0.75, x 0.75 mm). The new size is
   `round(n · spacing / target)` per axis.

Per nodule:

5. **Mask** = `masks_instance == instance_label`. Its voxel count must be within 5 % of
   `consensus_voxels` from `nodules.csv`, otherwise the run stops (`InstanceLabelError`). This is the
   runtime check that instance ids equal `nodule_id`. The mask is resampled onto exactly the CT's grid
   and kept soft (fractional), since it is only used as weights.
6. **Input window.** CT-CLIP's canvas is 240×480×480 voxels, about 360 mm at its spacing, close to an
   LIDC field of view.
   - `scan` (default): the whole scan is centre-cropped/padded into the canvas, as CT-RATE was. If a
     nodule would fall outside, the window shifts along that axis to contain it. Nodules sharing a
     window share one encoder pass (on synthetic data, 612 nodules from 358 scans needed 391 passes).
   - `crop`: a cube of `crop_size_mm` centred on the nodule is resized to the canvas. This is the
     figure's "grounded lesion crop": the nodule spans many tokens, but at an unfamiliar scale.
7. **Tokens.** CT-CLIP's image encoder (CTViT) returns a token map of shape (t, h, w, C) = (24, 24, 24, 512)
   for the canvas (patch 10×20×20 voxels ≈ 15 mm per side).
8. **Occupancy.** The mask is average-pooled to the token grid: occupancy ∈ [0, 1] is the fraction of each
   token's voxels inside the nodule. Optional `dilate_tokens=k` grows the region by k tokens.
9. **Readouts** (all cached, chosen at training time):
   - `token_mean`: occupancy-weighted mean of the nodule's tokens, 512-d. It doesn't depend on position,
     and it lives in the encoder's token space.
   - `masked_latent`: tokens are depth-averaged with occupancy weights, columns are gated by occupancy,
     and the result goes through CT-CLIP's own image projection, then is L2-normalised. It lives in CT-CLIP's
     image–text space. With a full mask it reduces exactly to CT-CLIP's global embedding (tested).
   - `global_latent`: CT-CLIP's embedding of the whole window, with no grounding (ablation / arm A input).

The cache (`data/embedding_cache.py`) stores one `.pt` per nodule, written atomically. It records its
settings in `meta.json` and refuses to mix runs with different settings. On first load during training
it is consolidated into `images.pt`. Concept prompts are encoded once into `text.pt`.

**Scan vs. crop.** `scan` keeps CT-CLIP in distribution, but a median 9 mm nodule covers 1–3 of its
15 mm tokens, and on the real data that carried no concept information (§0). `crop` (now the default
in `run_local.sh`: 64 mm, ≈ 5.6× zoom, ≈ 40 tokens for a 9 mm nodule; 96 mm, ≈ 3.8×, ≈ 10 tokens) puts
many tokens on the nodule at an unfamiliar scale. Whether that helps is measured by the probe.

### 2b. Lesion-level encoder (FMCIB)

`encoders/fmcib.py` wraps the foundation model of Pai et al. (Nat Mach Intell 2024): a MONAI 3D
ResNet-50 (widen factor 2, 4,096-d output) pretrained self-supervised on 11,467 DeepLesion lesions (no
LIDC data). Preprocessing follows its package (`fmcib` 1.0.0, read from the source):

- resample to 1 mm isotropic (trilinear);
- intensity (HU + 1024) / 3072; we clip to [−1024, 3071] HU first (the original does not clip; this
  changes only out-of-FOV padding values);
- 50 × 50 × 50 voxel cube centred on the seed point, start = int(centre) − 25, zero (= −1024 HU)
  outside the volume. The paper used the centre of mass of the 3D contour for LUNA16; we use the
  centre of mass of the consensus mask;
- orientation LPS, transposed to (z, y, x): our RAS (z, y, x) arrays are flipped along y and x
  (checked numerically).

Readout `lesion_embedding` (4,096-d) plus `occupancy_sum` = fraction of the nodule inside the cube
(a warning is logged below 0.9, i.e. nodules > 50 mm). Unlike the original loader (`strict=False` + a
warning), ours loads strictly and stops on any missing or unexpected key. No text encoder: concepts are
initialised from the tree.

### 2c. Baselines

`hypertrace-radiomics` computes 21 hand-crafted features per nodule from the raw HU scan and the
consensus mask at native spacing (size, extents, surface, sphericity, PCA elongation/flatness, HU
statistics and percentiles, ground-glass and calcified fractions, a 2-voxel outer shell and the
inside–shell contrast). `hypertrace-zeroshot` adds `diameter_only` and `radiomics` rows next to the
encoder readouts. A learned feature that doesn't beat them is not carrying extra concept information.

## 3. Concept tree and labels

**v3 (default): an ontology plus a per-dataset label mapping.**

- `configs/ontology/lung_nodule.yaml` is the clinical tree, independent of any dataset: 50 nodes,
  depth 8. Each node has a stable `name`, a `parent`, a clinical `display` term (Fleischner glossary
  2024), a `radlex: {id, term}` entry, a `lungrads` note on how it enters a Lung-RADS v2022 assessment,
  and prompts. Group nodes carry `lungrads_feature` (size, nodule_type, solid_component, benign_feature,
  additional_feature, location).
- `configs/labels/lidc.yaml` says which nodes LIDC can label and how. Every other node stays in the tree
  **unscored**: it is embedded and shaped by the hierarchy loss, and a later dataset can score it.
- `concepts/ontology.py` resolves the pair into the one tree format the rest of the code uses (and that
  every run saves in its `config.yaml`). An ontology node with a `label`, or a mapping label for a node
  that is not in the ontology, is an error.

Tree (scored for LIDC in **bold**):

| group | nodes |
|---|---|
| size (mean diameter, Lung-RADS solid bins) | **<6**, **6–<8**, **8–<15**, **≥15 mm** |
| nodule type | **solid**, **subsolid** → **part-solid** → solid component <6 / 6–<8 / ≥8 mm; **ground-glass** |
| shape | **round**, **oval**, lentiform/triangular, **linear** |
| margin | **smooth**, **irregular**, **lobulated**, **spiculated** (Lung-RADS 4X) |
| calcification | **calcified** → benign pattern (complete, central, concentric, popcorn), eccentric; **non-calcified** |
| internal features | fat-containing, cavitation, atypical cyst → thick-walled, multilocular |
| location | juxtapleural → perifissural; airway → segmental-or-proximal, subsegmental; upper lobe |

The 17 scored concepts and their label rules are exactly v2's (tested column by column); only names
follow the clinical vocabulary: `texture_*` → `type_*`, `texture` → `nodule_type`, `shape_ovoid` →
`shape_oval`. Positives (binary view): size 134 / 378 / 575 / 303; shape 978 / 448 / 68; margin
1233 / 223 / 153 / 136; type 1179 / 312 / 176 / 101; calcification 178 / 1234. The exclusive-group
metric uses the scored children of each `exclusive` node, which gives the same five groups as v2.
Left unscored from LIDC, with the reason in the mapping file: solid-component size (not rated),
calcification patterns (6 non-central), internal structure (0–4 nodules per code), location (needs the
lung and lobe masks), lentiform shape.

**RadLex IDs** are never typed in by hand. `hypertrace-radlex` matches each node's `radlex.term` list
against an official RadLex release (BioPortal CSV export or the OWL file, which you download after
accepting RSNA's licence) or the BioPortal API. Matching is exact, ignoring case, spaces and hyphens.
Only a unique match fills an ID (with `--write`); ambiguous terms are listed with all candidates in
`radlex_report.csv` for you to choose. Size and solid-component bins are thresholds, not RadLex concepts
(`radlex: null`). The IDs are not filled in this release: RadLex was not reachable from the build machine.

The v2 tree (`concepts/lidc.yaml`, 26 nodes, labels inside the tree) is
`experiments/tree_v2.yaml`; v1 (10 concepts) is `concepts/lidc_v1.yaml`. Prompts are kept for
zero-shot checks and `init: text`.

Labels (`concepts/labels.py`): the soft label is the reader mean rescaled to [0, 1] (or the fraction of
readers, for calcification). It is the BCE training target, so reader disagreement stays in the label.
The binary view (soft ≥ 0.5) is used only for AUROC/AUPRC/F1. Missing ratings are NaN and masked out.

## 4. The Poincaré concept space

### 4.1 Geometry (`geometry/poincare.py`, closed form, no approximations)

| quantity | formula |
|---|---|
| exp map at 0 | exp₀(v) = tanh(√c‖v‖) · v / (√c‖v‖) |
| log map at 0 | log₀(x) = artanh(√c‖x‖) · x / (√c‖x‖) |
| Möbius addition | x ⊕ y = ((1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 − c‖x‖²)y) / (1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²) |
| geodesic distance | d(x,y) = (2/√c) artanh(√c ‖(−x) ⊕ y‖) = (1/√c) arcosh(1 + 2c‖x−y‖² / ((1−c‖x‖²)(1−c‖y‖²))) |
| depth (specificity) | d(0, x) = (2/√c) artanh(√c‖x‖) |
| entailment cone (Ganea et al. 2018) | half-aperture ψ(x) = arcsin(K(1−‖x‖²)/‖x‖); exterior angle Ξ(x,y) (closed form in code); y ∈ cone(x) ⇔ Ξ(x,y) ≤ ψ(x) |

Cone formulas are for the unit ball, so points are scaled by √c first. Pairwise distances use the arcosh
form, with ‖x−y‖² computed from explicit differences. The expanded ‖x‖² + ‖y‖² − 2⟨x,y⟩ form loses
precision near the boundary: it gave d(x, x) ≈ 0.15 in float32, which a test now guards against. Points
are kept 10⁻⁵ inside the boundary, and training runs in float32.

### 4.2 Model (`models/projector.py`, `concept_head.py`, `model.py`)

```
image readout x (frozen, z-scored) ──► v = clip_r( α · W · x ) ──► lesion = exp₀(v) ∈ ball
concept k: a free point p_k ∈ ball (Riemannian Adam)
score_k = σ( −d(lesion, p_k) / τ + b_k )
```

- `W` is a learnable linear map to a d-dimensional tangent space (d = 64 by default; swept 2–64).
- Inputs are z-scored with training-split statistics (`data.standardize`). (v1: `√d_in` undid the L2
  normalisation of CLIP latents instead.)
- **Concept points** (`model.concepts`): every node is its own parameter. `param: riemannian` stores
  the point on the ball and updates it with Riemannian Adam (§4.4); `param: tangent` stores a tangent
  vector at the origin and maps it with exp₀ (ablation). Initialisation `tree`: radius in the tangent
  space r(depth) = 2 (depth + 1)/(max depth + 2), the root gets a random direction, each child its
  parent's direction plus noise of relative size 0.5. `text`: directions from the principal axes of the
  *centred* prompt embeddings (centring removes what all radiology prompts share), same radii. Both
  arms start from the same tangent vectors (Poincaré: exp₀ of them).
- v1 (`concepts.mode: text`) placed concepts at clip_r(α_txt W √d t); with `masked_latent` image and
  text shared W. That is what collapsed on real data (§0).
- `α` is a learnable scale initialised to 1/√64, as in MERU.
- `clip_r = 2` caps the tangent norm (feature clipping, Guo et al. CVPR 2022). The maximum radius is
  tanh(2) ≈ 0.964, away from the boundary where gradients vanish. This is a regulariser, not a change of geometry.
- `τ` (temperature, init 0.2) and `b_k` (per-concept bias) are learnable. Curvature is fixed at c = 1
  by default (`experiments/learn_curvature.yaml` learns it, bounded to [0.1, 10]).

### 4.3 Is this a genuine Poincaré representation?

Yes:
- Every lesion and concept is a **point inside the ball**.
- Scores are a function of the **exact geodesic distance** between those points.
- The hierarchy losses and metrics use the **exact hyperbolic entailment cones** and the hyperbolic
  distance from the origin.

Lesions: the network outputs a tangent vector at the origin and exp₀ maps it into the ball. exp₀ is a
smooth bijection from ℝᵈ onto the ball (log₀ is its exact inverse), so this is an exact
reparameterisation, and the trainable weights (W, α) are ordinary Euclidean tensors for which AdamW is
the right optimiser (as in MERU). Concepts: free points on the ball, updated with Riemannian Adam (§4.4),
because there the parameter itself lives on the manifold.

`tests/test_poincare_model.py` checks this on the model's actual outputs:
- all points stay inside the ball for c ∈ {0.5, 1, 2}, even for extreme inputs;
- the logits equal −d/τ + b when d is recomputed with the independent Möbius-addition formula;
- the geodesic distance orders pairs differently from Euclidean distance: points near the boundary are
  far apart in hyperbolic terms even when Euclidean-close, which no flat shortcut can do;
- depth equals 2·artanh(‖x‖);
- exp₀ ∘ log₀ is the identity;
- the Euclidean twin has identical parameters and initialisation (concept points: the same tangent
  vectors) and differs only in geometry.

`tests/test_riemannian.py` checks the optimiser: exp_x moves by exactly λ_x‖u‖ in geodesic distance,
parallel transport preserves the Riemannian norm, the Euclidean group is exactly `torch.optim.Adam`,
plain groups are exactly `torch.optim.AdamW`, one step moves a point by lr·√d in Riemannian length,
points converge to a target and stay in the ball, and both second-moment variants match **geoopt**
(Kochurov et al., 2020) to 10⁻⁶ when geoopt is given the exact exponential map.

**Euclidean twin (arm B).** The same projector, head, losses and initialisation, with `to_space` = identity,
Euclidean distance, and Euclidean entailment cones (ψ(x) = arcsin(K/‖x‖)). Switch with `geometry.name=euclidean`.

### 4.4 Riemannian Adam for the concept points (`optim.py`)

Concept points are parameters *on* the ball, so they are updated along the manifold (Bécigneul &
Ganea, ICLR 2019), with the exact operations of the ball (`geometry/poincare.py`):

| step | formula |
|---|---|
| Riemannian gradient | r = g / λ_x², with λ_x = 2 / (1 − c‖x‖²) |
| first moment | m ← β₁ P(m) + (1 − β₁) r (P = parallel transport from the previous point) |
| second moment | v ← β₂ v + (1 − β₂) λ_x² r² (per coordinate; `point`: λ_x²‖r‖², one scalar per point) |
| update | x ← exp_x(−lr · m̂ / (√v̂ + ε)), exp_x(u) = x ⊕ tanh(√c λ_x‖u‖/2) u / (√c‖u‖) |
| transport | P_{x→y}(m) = gyr[y, −x] m · λ_x / λ_y |

**Second moment.** The ball's metric is conformal (λ_x² times the identity), so the coordinate axes
are orthogonal tangent directions and a per-coordinate preconditioner is well defined. With it, the
Euclidean twin (r = g, exp_x(u) = x + u, identity transport) is *exactly* Adam: both arms run the same
algorithm and each step moves a point by ≈ lr·√d in Riemannian length. The per-point variant (geoopt's
convention) makes Poincaré steps √d times shorter than the twin's; on synthetic data it left the
concept points at their initial positions (test AUROC 0.58 vs 0.81). It is kept as
`optim.riemannian_second_moment: point` and matches geoopt to 10⁻⁶ in the tests.

Ordinary weights use decoupled AdamW (same formula as `torch.optim.AdamW`, tested). One optimiser
holds all groups, so the warm-up/cosine schedule, clipping and exact resume are unchanged. The
`tangent` ablation (exp₀ of free tangent vectors, plain AdamW) answers whether the Riemannian update
matters beyond the reparameterisation.

### 4.5 Curvature of either sign: the κ-stereographic model (`geometry/kstereo.py`)

One closed-form geometry for every constant curvature κ (Bachmann, Bécigneul & Ganea, ICML 2020;
Skopek et al., ICLR 2020): κ < 0 is the Poincaré ball of radius 1/√−κ, κ = 0 is flat, κ > 0 is the
stereographic projection of the sphere of radius 1/√κ (origin = one pole, ‖x‖ = 1/√κ = equator,
antipode at infinity). Möbius addition, distance, exp/log maps, conformal factor, gyration and
parallel transport are the standard κ-stereographic formulas. Near κ = 0 the tan/tanh forms switch to
their Taylor series, so a learned κ passes through 0 with finite values and gradients.

- `sphere`: fixed κ > 0 (default 1). `kstereo`: any κ, by default learned from 0, clamped to [−4, 4];
  the sign it ends with is a result. `curvature_lr` gives κ its own learning rate (Adam moves a lone
  scalar by about lr per step whatever the gradient size).
- Conventions: at κ = 0 distances are 2‖x − y‖ (the stereographic scale; the temperature absorbs it).
  Cone half-aperture ψ(x) = asin(K (1 + κ‖x‖²)/‖x‖), continuous through κ = 0; it equals Ganea's
  hyperbolic cone at κ = −1 and the Euclidean cone at κ = 0. The exterior angle at x towards y is the
  angle between x and (−x) ⊕ y, the direction of log_x(y); the model is conformal, so this is the
  Riemannian angle. **On the sphere cones are only formal** (no nested-cone structure), so sphere
  factors in the product model skip the cone losses by default.
- On the sphere, exp_0 caps tangent vectors at 0.95 π/(2√κ) and points are kept within 0.95 π/√κ of
  the origin, so nothing reaches the antipode.
- Verified (`tests/test_kstereo_product.py`): κ = −c reproduces `geometry/poincare.py` (distance, maps,
  transport, angles); κ > 0 gives great-circle distances on the sphere; geoopt's `Stereographic`
  agrees for κ ∈ {1, −0.7, 0.3}; Riemannian Adam matches geoopt's on κ = ±; κ has finite non-zero
  gradients at and around 0.

### 4.6 Product concept space: one factor per concept group (`models/product.py`)

`geometry.name=product` gives every concept group (the children of the trunk: size, nodule type,
shape, margin, calcification) its own factor M_g with its own geometry. The lesion projection
(total dimension `embed_dim`, split equally across factors) is cut into per-factor blocks; concept k of
group g is scored only from g's block: logit_k = −d_{M_g}(x_g, p_k)/τ + b_k. Each factor has its own
concept points (Riemannian Adam in that factor's geometry, `geometry_index`), and its own hierarchy and
lesion-entailment losses on its group's subtree (skipped on the sphere). The trunk (chest … nodule)
holds for every input and is not embedded; groups with no scored concept are skipped by default.

The assignment is a named table (`geometry.product.assignments`, part of the run name, e.g.
`product-matched`):

| assignment | size | nodule type | shape | margin | calcification | tests |
|---|---|---|---|---|---|---|
| `matched` | Euclidean | Poincaré | sphere | Euclidean | Poincaré | the hypothesis |
| `matched_no_sphere` | Euclidean | Poincaré | Euclidean | Euclidean | Poincaré | hyperbolic part alone |
| `swapped` | sphere | Euclidean | Poincaré | Poincaré | sphere | wrong geometry per group (control) |
| `all_euclidean` | E | E | E | E | E | factorisation without curvature |
| `all_poincare`, `all_sphere` | one geometry everywhere | | | | | |
| `learned` | κ learned per factor from 0 | | | | | does each κ take the predicted sign? |

Reading the result: a geometry helps a group if `matched` beats `swapped` and `all_euclidean` **on the
metric that matches the group's structure** (below), at equal total dimension, under the matched tuning
protocol (§6.1). `all_euclidean` vs a single Euclidean space separates the effect of factorising from
the effect of curvature.

### 4.7 Per-group metrics (`eval/metrics.py: group_metrics`)

For every model (single space or product), per concept group, `group/<g>/<metric>` in `metrics.json`,
computed in the space where the group's concepts live (`model.group_view`):

| structure | metric | meaning |
|---|---|---|
| any | `auroc` | mean AUROC of the group's scored concepts |
| hierarchy (nodule type, calcification) | `parent_child_violation`, `edge_in_cone`, `depth_spearman`, `tree_map` | consistency and tree structure inside the group subtree |
| exclusive categories (size, shape, nodule type, calcification) | `exclusive_violation` | nodules without exactly one predicted category |
| ordered bins (`ordinal: true`: size, solid component) | `ordinal_spearman`, `ordinal_dist_spearman` | predicted vs true bin; concept points laid out in bin order |

Product runs also report the tree metrics of each factor (`<group>/tree_map`, ...) and their mean (the
plain `tree_map`, ...). `hypertrace-summarize --groups` prints the group table per arm.

## 5. Losses (`models/losses.py`)

L = L_concept + w_h · L_hier + w_img · L_img (defaults w_h = 1, w_img = 0.1)

- **L_concept**: BCE with soft targets on the 10 scored concepts. NaN targets are masked. Positive
  classes are weighted by `pos_weight` = neg/pos on the training split (capped at 50).
- **L_hier**: on the concept tree, positives are all ancestor→descendant pairs:
  mean max(0, Ξ(parent, child) − ψ(parent)), which puts the child inside the parent's cone. Negatives are
  non-ancestor pairs: mean max(0, m − violation), with m = 0.1, which keeps unrelated nodes out of each
  other's cones (otherwise one wide cone could swallow everything).
- **L_img**: the lesion should lie inside the cones of the concepts it has (weighted by the soft label) and
  of the always-true `nodule` node (MERU-style text-entails-image). This ties the bottleneck to the tree.

## 6. Training and evaluation (`training.py`, `cli/train.py`)

- **Split**: test = `test_fold`, val = next fold, train = the other three. All patient-level, and asserted
  disjoint at load time.
- **Optimiser**: AdamW, lr 5e-3. Weight decay 1e-4 applies to matrices only (none on scales, biases, τ, curvature).
- **Schedule**: linear warm-up over 5 % of steps, then cosine to 0, stepped per batch.
- **Gradient clipping**: global norm 1.0, computed over parameters in optimiser-group order.
- **Selection and stopping**: after every epoch the full validation split is evaluated (and, with
  `eval_train`, the training split in eval mode, for train/val curves). From `select_from_epoch` (20) on,
  `best.pt` is saved on a strict improvement of `select_metric`, and training stops after 30 epochs
  without one (max 200). Default `val/loss_concept`, the concept BCE alone. Until the training-fix patch
  it was the total validation loss, which includes the tree terms: they can keep falling while concept
  prediction overfits, so the chosen checkpoint mixed the two goals. The warm-up keeps the reason the
  total loss was chosen originally: concept AUROC alone peaked at epoch 0 in v1, before the tree was
  learned. Metrics named `loss...` are minimised, others maximised; test metrics are refused.
- **Regularisation knobs** (all off by default, searched by `hypertrace-tune`): `optim.weight_decay`,
  `model.projector.input_dropout` (dropout on the input features), `data.pca_dim` (top-k principal axes
  of the training split), `model.head.per_concept_tau`.
- **Per-geometry cap**: `model.projector.clip_r` is `{poincare: 2.0, euclidean: null}`. The cap keeps
  Poincaré points off the boundary; the Euclidean space has none, and the shared 2.0 cap pinned every
  Euclidean lesion to a ring at d = 2 (seen in the FMCIB d = 2 runs).
- **Diagnostics** (`eval/probes.py`): after training, logistic probes (C = 0.1) fitted on the training
  split, on (a) the learned lesion embedding in tangent coordinates and (b) the input features.
  Comparing them with the head says whether the projection or the distance head loses information.
- **Run folders**: `tag` adds `_t<tag>` to the run name; a folder that holds a run with other settings
  is refused instead of being resumed.
- **Resume**: `last.pt` holds model, optimiser, scheduler, epoch counters and all RNG states. A pre-empted
  job resumes and ends with the same weights as an uninterrupted run (tested).
- **Determinism**: seeds for Python, NumPy and torch; deterministic kernels; batch order from the seeded
  global torch RNG.
- **Metrics** (`eval/metrics.py`), on the best checkpoint for val and test:
  - per-concept and macro AUROC, AUPRC, F1 (at 0.5), Brier (against soft labels);
  - hierarchy metrics: `edge_in_cone` (fraction of parent→child edges satisfied), `non_edge_in_cone`
    (unrelated pairs wrongly nested), `depth_spearman` (tree depth vs hyperbolic depth), `tree_map`
    (mean average precision of recovering each node's tree neighbours from distances, Nickel & Kiela
    2017) and `dist_spearman` (embedded distance vs tree path length over all pairs). The same metrics
    for the *initial* points are stored under `init` in `metrics.json`, since tree initialisation
    already encodes part of the tree;
  - consistency: `parent_child_violation` (child predicted present, parent absent: part-solid without
    subsolid) and `exclusive_violation` (not exactly one category of an exclusive group predicted).
    Both threshold scores at 0.5, and positive-class weighting shifts scores upwards, so compare arms
    rather than reading them as absolute rates.
- **Exports**: `predictions_test.csv` (out-of-fold, for Part 3) and `concept_space.pt`.

### 6.1 Tuning protocol (`cli/tune.py`)

Trials are drawn once from a fixed search space with a fixed seed; trial 0 is the base config. Every
geometry × dimension × fold × seed runs the same trials, so the arms get an equal budget. Selection is on
validation AUROC only. **Per-fold selection** (each outer fold picks on its own validation split, test
reported) is the estimate to publish; **global selection** (one setting per arm) is for convenience and
slightly optimistic, because each fold's validation split is the next fold's test split. A geometry gap
smaller than what tuning moves (~0.01 AUROC here) is only meaningful under this matched protocol.

## 7. Open checks

These can't be verified without the real CT-CLIP checkout and scans. Each fails loudly or is measurable:

1. **CT-CLIP internals** (`encoders/ctclip.py`, written from memory of the public repo, which I couldn't
   open from the build environment):
   - input 240×480×480;
   - CTViT patch 20 (spatial) × 10 (depth), dim 512;
   - image latent = `to_visual_latent` applied to the depth-averaged, flattened (24·24·512 = 294,912) token map;
   - text latent = projection of BERT's CLS token.

   A load-time self-check compares `to_visual_latent`'s input size with the token grid. Also confirm the
   HU window and voxel spacing in CT-CLIP's preprocessing.

   Checkpoint loading (`load_ctclip_state`) is made robust to newer libraries (PyTorch ≥ 2.6, needed
   for Blackwell, and newer `transformers`). It loads with `weights_only=False`, so only use the official
   file. It ignores only the BERT `position_ids` buffer that newer `transformers` no longer stores, and
   it fails on any other missing or unexpected key. `vector-quantize-pytorch` is pinned to 1.1.2 as in
   CT-CLIP's own setup, because its parameter names are part of the checkpoint.
2. **Orientation.** Scans are canonicalised to RAS and stored as (z, y, x). Save one canvas slice and
   compare it with a CT-RATE input. If CT-CLIP expects flipped axes, flip in `data/volumes.py`, re-extract,
   and compare probe AUROCs.
3. **Instance ids.** Checked at runtime against `consensus_voxels`. Nothing to do unless
   `InstanceLabelError` appears.
4. **Input scale.** Whole scan vs 64 / 96 mm crops vs FMCIB: decide with the probe against
   `diameter_only` and `radiomics` (§2c).
5. **FMCIB weights.** The loader was tested on checkpoints in both layouts the original code accepts
   (`trunk_state_dict`, `state_dict` with `model.trunk.`), not on the Zenodo file itself, which the
   build environment could not download. It stops on any key mismatch.

## 8. Verification

- **107 unit and end-to-end tests** (`pytest -q`, about 40 s on CPU):
  - config loader;
  - Riemannian Adam (vs. torch Adam/AdamW and geoopt), exp map and parallel transport;
  - FMCIB: architecture, both checkpoint layouts, strict rejection of a wrong file, intensity / crop conventions;
  - CT-CLIP checkpoint loading (tolerates version-only buffer differences, rejects wrong checkpoints);
  - Poincaré formulas and gradients near the boundary;
  - model-level Poincaré checks;
  - label building on the real bucket schema;
  - preprocessing (spacing, HU, windows, orientation);
  - manifest adapter (term parsing, score cross-check, filters);
  - pooling;
  - the full pipeline on a synthetic bucket: extraction idempotence, cache guards, both geometries,
    exact resume, experiment inheritance, instance-label check, probe with baselines, radiomics,
    lesion-encoder extraction and training, concept points staying in the ball, v1 reproduction.
- **v1 is reproduced bit for bit.** `experiments/v1_text_anchored.yaml` with the new code and the
  v1 configuration with the previous code, on the same synthetic cache: identical validation loss at
  every one of 40 epochs (max |Δ| = 0) and identical test AUROC.
- **v1 vs v2 on synthetic data** (287 nodules, 200 patients, fold 0, seed 0, 60 epochs; fake encoders,
  so absolute numbers mean nothing, the comparison of variants does):

| variant | geometry | test AUROC | edge in cone | depth ρ | tree MAP |
|---|---|---|---|---|---|
| v1 text-anchored, `masked_latent` | Poincaré / Euclidean | 0.53 / 0.44 | 1.00 / 1.00 | 0.91 / 0.86 | 0.49 / 0.43 |
| v2 free points + Riemannian Adam, `token_mean` | Poincaré / Euclidean | 0.81 / 0.80 | 0.96 / 1.00 | 0.86 / 0.77 | 0.29 / 0.31 |
| v2 free points, tangent + AdamW | Poincaré / Euclidean | 0.81 / 0.81 | 1.00 / 0.92 | 0.76 / 0.73 | 0.31 / 0.32 |
| v2 Riemannian, per-point second moment | Poincaré / Euclidean | 0.58 / 0.58 | 0.04 / 0.16 | 0.90 / 0.88 | 0.83 / 0.74 |
| v2 fake lesion encoder | Poincaré / Euclidean | 0.74 / 0.73 | 0.92 / 0.80 | 0.87 / 0.78 | 0.26 / 0.35 |

The v1 collapse reproduces even here. With the per-point second moment the points barely moved
(tree MAP stayed at its initial 0.83 and the cones were never satisfied), which is why the
per-coordinate form is the default. Training lowers tree MAP from the tree initialisation (0.83) as
concepts move towards the lesions that carry them; whether that differs between geometries on real
features is one of the things the dimension sweep measures.
- **Equivalence of the plain loop with the earlier Lightning/Hydra version.** Both were run on the same
  synthetic cache (612 nodules), 2 geometries × 2 features × 5 folds, seed 0. Per-epoch validation losses
  agree to ≤ 1.5 × 10⁻⁷ for the whole run, so the same epochs are selected and stopped on. Results:
  see the table below.

Test-set concept AUROC (macro), mean ± SD over 5 folds. The synthetic data and fake encoder make the
absolute numbers meaningless; what matters is that the two columns agree.

| arm | Lightning + Hydra | plain loop + YAML | max per-run \|Δ AUROC\| | AUPRC (both) | edge in cone (both) |
|---|---|---|---|---|---|
| Poincaré · token_mean | 0.8258 ± 0.0090 | 0.8258 ± 0.0090 | 3 × 10⁻⁸ | 0.8073 | 0.965 |
| Euclidean · token_mean | 0.8302 ± 0.0125 | 0.8302 ± 0.0125 | 3 × 10⁻⁸ | 0.8142 | 0.953 |
| Poincaré · masked_latent | 0.6436 ± 0.0224 | 0.6436 ± 0.0224 | 3 × 10⁻⁸ | 0.6770 | 0.965 |
| Euclidean · masked_latent | 0.5801 ± 0.0827 | 0.5801 ± 0.0827 | 3 × 10⁻⁸ | 0.6092 | 1.000 |

Across all 20 runs: max |Δ test AUROC| = 3.0 × 10⁻⁸, max |Δ test AUPRC| = 3.0 × 10⁻⁸, and max per-epoch
|Δ validation loss| = 5.4 × 10⁻⁷ (float32 rounding). Every run trained the same number of epochs and
selected the same best epoch. (Lightning's log has one extra row, which is its post-training
`validate()` call, not an extra training epoch.) The plain loop is also about 25 % faster per run (≈ 25 s vs ≈ 36 s on CPU).

Getting bit-level agreement needed two details. Batches are shuffled from the seeded global torch RNG,
in the same order of RNG use as before. The clipping norm is summed over parameters in
optimiser-group order: with `model.parameters()` order the runs drifted apart by ~10⁻⁶ per epoch after
epoch 10, ending ~0.006 AUPRC apart.

## 9. Changes from the previous version

v2 (this version): see §0. In short: CT-CLIP crop inputs and the FMCIB lesion encoder; z-scored
features; free concept points with Riemannian Adam (tree or text initialisation); concept tree v2;
diameter and radiomics baselines; tree-fidelity and consistency metrics; `_d<dim>` in run names and a
`DIMS` loop; `HT_CACHE_DIR` is the single cache path in `run_local.sh`.

Earlier (plain YAML + loop):

- Hydra replaced by plain YAML (`config.py`, ~175 lines), and PyTorch Lightning by `training.py`
  (~250 lines). Same model, losses, optimiser, schedule, clipping, early stopping and selection.
- Every run saves its resolved config plus metadata: command, git commit, dirty flag, library versions,
  SLURM job id.
- Run names start with the experiment name, so ablations can't overwrite the main runs. Under Hydra,
  `+experiment=...` runs collided.
- Console commands now exit with status 0 on success. The Hydra version returned the metrics dict from
  `main`, which `sys.exit` turned into exit status 1 under SLURM.
- `.gitignore` entries anchored to the repo root: `data/` also matched `src/hypertrace/data/`, so that
  package was never committed.
