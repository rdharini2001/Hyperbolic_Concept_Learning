---
title: "HYPER-TRACE Part 2: Methodology"
subtitle: "A Poincaré-ball concept bottleneck over frozen 3D chest-CT encoders (v2)"
---

*Manuscript-oriented description of what is implemented in this repository. Section 12 lists what has and has not been verified, and Section 13 lists references. Implementation details (file by file) are in `docs/IMPLEMENTATION.md`; commands are in `docs/RUNNING.md`.*

## 1. Problem and contribution

**Task.** For each lung nodule on a chest CT, predict a set of radiologist-defined semantic concepts (size, margin, texture, calcification) through an interpretable bottleneck. The concept scores are the input to a downstream malignancy-risk head and to neuro-symbolic rules (Part 3, not covered here).

**Core idea.** Radiology concepts are hierarchical (chest → lung → pulmonary lesion → nodule → margin → spiculated). Trees embed with low distortion in hyperbolic space (Sarkar 2011; Nickel & Kiela 2017; Sala et al. 2018). We therefore place the lesion and the concepts of a concept tree in a **Poincaré ball** and read each concept score off the **geodesic distance** between the lesion and the concept.

**What is held fixed and what is learned.** The 3D encoder is frozen: CT-CLIP (a CT vision-language model) read out on nodule-centred crops, or FMCIB (a lesion-level CT foundation model). Trained on cached embeddings: a linear map into the concept space, one free point per concept-tree node (on the ball, updated with Riemannian Adam), a temperature and per-concept biases (~34 k parameters for CT-CLIP features at d = 64; ~264 k for FMCIB's 4,096-d features).

**Controlled comparison.** The Poincaré model has an otherwise identical **Euclidean twin** (same parameters, initialisation, losses, data and schedule; only the geometry changes). Any difference is attributable to the geometry.

## 2. Pipeline overview

```
bucket scan (raw HU) -> reorient, HU window, resample -> nodule-centred crop -> frozen encoder -> lesion embedding (cached)
masks_instance == nodule id -> crop centre (and, for CT-CLIP, token weights) --^
        z-score -> learnable linear map -> tangent vector -> exp map onto the Poincaré ball
concept tree (26 nodes) -> one free point per node on the ball (Riemannian Adam)
        entailment cones on the tree  +  score_k = sigmoid(-d(lesion, concept_k)/tau + b_k)
        -> predictions_test.csv, concept_space.pt (inputs to Part 3)
```

Stage 1 (extraction, GPU, once) and stage 2 (training, minutes per run, CPU or GPU) are separate. Stage 2 never touches CT-CLIP.

## 3. Data

**Source.** LIDC-IDRI (Armato et al. 2011): 1,010 patients, up to four radiologists per nodule, ratings of nine characteristics plus malignancy. We use the Hugging Face re-hosting `Dharini24/LIDC-IDRI-bucket`, derived from the QIICR DICOM SEG/SR re-encoding of the LIDC annotations (Fedorov et al. 2020). It provides NIfTI scans, per-nodule instance masks, and per-nodule/per-annotation tables (`scans.csv`, `nodules.csv`, `annotations.csv`).

**Ground-truth rule.** We follow the bucket default: a clustered nodule is kept if ≥ 3 radiologists annotated it. This gives 2,651 → 1,392 nodules; two nodules on scans with irregular slice spacing (slice-gap spread > 0.1 mm) are dropped, leaving **1,390 nodules on 701 scans from 694 patients**. Seven patients have nodules on two scans, so folds are defined per patient.

**Rating encoding.** The `*_score` columns in the bucket are not the ordinal ratings (found by profiling the real CSVs). Ratings are therefore parsed from the `*_term` columns with fixed value maps (margin: Indistinct = 1 … Circumscribed = 5; texture: non-solid = 1, part-solid = 3, solid = 5; calcification: Popcorn 1, Laminated 2, Solid 3, Non-central 4, Central 5, Absent 6) and cross-checked against the scores.

**Splits.** Five patient-level folds, stratified on reader-mean malignancy ≥ 3 (727/1,390 positive). For a test fold k, validation is fold k + 1 and training is the remaining three. A leakage assertion checks that no patient appears in two splits. Test predictions are therefore out-of-fold for every nodule across the five folds.

## 4. Preprocessing and lesion grounding

1. **Orientation.** Scan and mask are reoriented to RAS+ and stored as (z, y, x) arrays with (z, y, x) spacing, so axis order is identical across scanners. Header spacing is checked against `scans.csv`.
2. **Intensity guard.** If the range of the loaded CT contradicts the `hu_already_normalized` flag, extraction stops (catches double or missing normalisation, which the Part-1 outputs make possible).
3. **Windowing.** HU clipped to [−1000, 1000] and divided by 1000. This is done **before resampling** so that out-of-field-of-view padding values (−2048, −3024) cannot bleed into tissue during interpolation.
4. **Resampling.** Trilinear to CT-CLIP's spacing (1.5 mm in z, 0.75 mm in x and y).
5. **Nodule mask.** `masks_instance == instance_label`. Its voxel count must agree with `consensus_voxels` within 5 %; otherwise extraction stops. This is the runtime check that the instance id equals the nodule id. The mask is resampled onto the CT grid and kept soft, since it is used only as weights.
6. **Input window (CT-CLIP).** `crop` (used in v2): a cube of 64 or 96 mm centred on the nodule is resized to CT-CLIP's 240 × 480 × 480 canvas (≈ 5.6× / 3.8× zoom; a 9 mm nodule covers ≈ 40 / ≈ 10 tokens). `scan` (v1): the whole scan in the canvas at native CT-CLIP scale, where a median nodule covers 1–3 tokens.
7. **Token occupancy (CT-CLIP).** The image encoder returns a 24 × 24 × 24 token map (patch 10 × 20 × 20 voxels). The mask is average-pooled to this grid: occupancy ∈ [0, 1] is the fraction of each token's voxels inside the nodule, and serves as readout weights.
8. **Lesion cube (FMCIB).** Resample to 1 mm isotropic, intensity (HU + 1024)/3072 (clipped to [−1024, 3071] HU), a 50 × 50 × 50 voxel cube centred on the mask's centre of mass, reoriented to LPS, as in the FMCIB package.

## 5. Frozen encoders and lesion readouts

**CT-CLIP** (Hamamci et al. 2024) pairs a 3D ViT image encoder (CTViT, in the style of Phenaki's C-ViViT) with a CXR-BERT text encoder (Boecking et al. 2022) trained contrastively on CT-RATE, as in CLIP (Radford et al. 2021). Three readouts are cached per nodule and chosen at training time:

| readout | definition | space |
|---|---|---|
| `token_mean` | occupancy-weighted mean of the nodule's tokens (512-d) | encoder token space; position-invariant |
| `masked_latent` | tokens depth-averaged with occupancy weights, columns gated by occupancy, passed through CT-CLIP's own image projection, L2-normalised | CT-CLIP image–text space |
| `global_latent` | CT-CLIP's embedding of the whole window | no grounding (ablation; plain-CT-CLIP baseline input) |

With a full mask, `masked_latent` reduces exactly to CT-CLIP's global embedding (unit-tested).

On nodule crops the main readout is `token_mean`. `masked_latent` passes through CT-CLIP's projection, a linear map over the flattened token grid, so it is position-specific.

**FMCIB** (Pai et al. 2024) is a 3D ResNet-50 (MONAI, widen factor 2) pretrained with self-supervised contrastive learning on 11,467 lesions from DeepLesion; LIDC was not used for pretraining. Its 4,096-d embedding of the lesion cube is the readout `lesion_embedding`. It has no text encoder.

**Baselines.** (i) `diameter_only`: the Part-1 consensus diameter alone. (ii) `radiomics`: 21 hand-crafted features from the raw HU scan and the consensus mask (volume, equivalent diameter, extents, surface area, sphericity, PCA elongation and flatness, HU mean/SD/percentiles, ground-glass (−750 to −300 HU) and calcified (> 200 HU) fractions, mean HU of a 2-voxel outer shell and inside–shell contrast). Every readout is compared with both in patient-grouped linear probes (logistic regression, C = 0.1, 5-fold GroupKFold by patient); a learned feature that doesn't beat them carries no additional concept information.

**First real result (motivates v2).** With whole-scan `masked_latent`, every CT-CLIP probe was near chance and below the diameter alone (e.g. spiculation: diameter 0.86, best CT-CLIP probe 0.65), although the mask-to-token grounding was verified (coverage vs diameter, Spearman 0.94). The concept model on these features reached macro AUROC 0.51 ± 0.01 in both geometries.

## 6. Concept hierarchy and labels

**Tree (v2).** 26 nodes, depth 6: chest → lung → pulmonary lesion → nodule → {size, shape, margin, texture, calcification}, with **17 scored concepts** chosen to follow the descriptors used in Lung-RADS v2022 and the Fleischner 2017 guidelines and in benign/malignant reasoning:

| group | scored concepts | label construction | positives (binary view, n = 1,390) |
|---|---|---|---|
| size (Lung-RADS bins) | <6, 6–<8, 8–<15, ≥15 mm | range of the Part-1 diameter | 134 / 378 / 575 / 303 |
| shape (LIDC sphericity) | round (codes 4–5), ovoid (3), linear (1–2) | fraction of readers | 978 / 448 / 68 |
| margin | smooth, irregular (margin rating), lobulated, spiculated | reader mean, rescaled | 1,233 / 223 / 153 / 136 |
| nodule type (texture) | solid (5); subsolid (1–4) → part-solid (3–4), ground-glass (1–2) | fraction of readers | 1,179 / 312 / 176 / 101 |
| calcification | present (1–5), absent (6) | fraction of readers | 178 / 1,234 |

`subsolid` is a scored internal node, so parent–child consistency can be measured. Size, shape, nodule type and calcification are groups of mutually exclusive categories; margin descriptors are rated separately in LIDC and are not exclusive. All scores are independent sigmoids. Left out for lack of cases: calcification pattern (6 nodules with majority non-central), fat / fluid / air internal structure (0–4). v1 had 10 concepts (one 8 mm size threshold; solid/subsolid; no shape).

**Soft labels and binary view.** Training uses the soft label (reader mean rescaled to [0, 1], or fraction of readers choosing the category), so reader disagreement stays in the target. A binary view (soft ≥ 0.5) is used only for AUROC, AUPRC and F1. Missing ratings are NaN and masked out of the loss. The rare concepts (linear 4.9 %, ground-glass 7.3 %, part-solid 12.7 %, spiculated 9.8 %) motivate AUPRC and positive-class weighting.

## 7. The Poincaré concept space

**Model.** The Poincaré ball with curvature −c (c > 0, radius 1/√c, default c = 1) is

$$\mathbb{B}^n_c=\{x\in\mathbb{R}^n:\ c\|x\|^2<1\}.$$

All operations are closed form (Ungar's gyrovector formalism; Ganea et al. 2018a):

- exponential map at the origin: $\exp_0(v)=\tanh(\sqrt c\,\|v\|)\,v/(\sqrt c\,\|v\|)$, with $\log_0$ its exact inverse;
- Möbius addition: $x\oplus y=\dfrac{(1+2c\langle x,y\rangle+c\|y\|^2)\,x+(1-c\|x\|^2)\,y}{1+2c\langle x,y\rangle+c^2\|x\|^2\|y\|^2}$;
- geodesic distance: $d(x,y)=\tfrac{2}{\sqrt c}\operatorname{artanh}\!\big(\sqrt c\,\|(-x)\oplus y\|\big)=\tfrac{1}{\sqrt c}\operatorname{arcosh}\!\Big(1+\tfrac{2c\|x-y\|^2}{(1-c\|x\|^2)(1-c\|y\|^2)}\Big)$;
- depth (specificity): $d(0,x)=\tfrac{2}{\sqrt c}\operatorname{artanh}(\sqrt c\,\|x\|)$.

**Lesions.** The frozen feature $x$, z-scored with training-split statistics, is mapped as

$$v=\operatorname{clip}_{r}\big(\alpha\,W\,x\big),\qquad \ell=\exp_0(v)\in\mathbb{B}^{d}_c .$$

- The network outputs a *tangent vector at the origin*; $\exp_0$ is a smooth bijection from $\mathbb{R}^d$ onto the ball, so this is an **exact reparameterisation** and the trainable weights ($W$, $\alpha$) are ordinary Euclidean tensors trained with AdamW, as in MERU (Desai et al. 2023).
- $\alpha$ is a learnable scale initialised at $1/\sqrt{d}$ (MERU).
- $\operatorname{clip}_r$ with $r=2$ caps the tangent norm so points stay at radius ≤ tanh 2 ≈ 0.964, away from the boundary where gradients vanish (feature clipping, Guo et al. 2022). This regularises; it does not change the geometry.
- $d$ = 64 by default and swept over {2, 4, 8, 16, 32, 64}.

**Concepts: free points on the ball.** Each node $k$ of the tree has its own parameter $p_k\in\mathbb{B}^d_c$. Initialisation (`tree`): tangent radius $r(\text{depth})=2(\text{depth}+1)/(\text{max depth}+2)$; the root receives a random direction and each child its parent's direction plus Gaussian noise of relative size 0.5, then $p_k=\exp_0(\cdot)$. Alternative (`text`): directions from the principal axes of the *centred* prompt embeddings (centring removes the component shared by all radiology prompts), same radii. In v1, concepts were the prompt embeddings passed through the image map; because CT-CLIP's prompt embeddings are nearly identical, all ten concepts collapsed to one point (depth range 0.185–0.200 vs lesions 0.25–1.37).

**Riemannian Adam** (Bécigneul & Ganea 2019). For a point $x$ with Euclidean gradient $g$, with $\lambda_x = 2/(1-c\|x\|^2)$:

$$r=g/\lambda_x^2,\quad m\leftarrow\beta_1 P(m)+(1-\beta_1)r,\quad v\leftarrow\beta_2 v+(1-\beta_2)\lambda_x^2 r^{2},\quad x\leftarrow\exp_x\!\big(-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)\big),$$

with $\exp_x(u)=x\oplus\tanh(\sqrt c\,\lambda_x\|u\|/2)\,u/(\sqrt c\|u\|)$ and parallel transport $P_{x\to y}(m)=\operatorname{gyr}[y,-x]\,m\;\lambda_x/\lambda_y$. The second moment is per coordinate, which is well defined because the ball's metric is conformal; for the Euclidean twin ($r=g$, $\exp_x(u)=x+u$, identity transport) the update is then exactly Adam, so both arms run the same algorithm with the same hyperparameters. (The per-point variant used by geoopt, $\lambda_x^2\|r\|^2$, gives Poincaré steps $\sqrt d$ times shorter than the twin's; it is available as an option.) Ablation `tangent`: free concept tangent vectors mapped by $\exp_0$ and trained with AdamW.

**Concept score.** $s_k=\sigma\!\big(-d(\text{lesion},\text{concept}_k)/\tau+b_k\big)$, with learnable temperature τ (init 0.2) and per-concept bias $b_k$. Closer to a concept means higher score.

**Entailment cones.** Hierarchy is enforced with the hyperbolic entailment cones of Ganea et al. (2018b). For a parent $x$ the cone half-aperture is $\psi(x)=\arcsin\!\big(K(1-\|x\|^2)/\|x\|\big)$ (K = 0.1) and a child $y$ lies in the cone iff its exterior angle $\Xi(x,y)\le\psi(x)$. Violation is $\max(0,\Xi-\psi)$. MERU uses the same idea for text-entails-image.

**Numerical care.** Pairwise distances use the arcosh form with $\|x-y\|^2$ from explicit differences; the expanded form loses precision near the boundary (it gave $d(x,x)\approx0.15$ in float32; a regression test guards this). Points are kept 10⁻⁵ inside the boundary.

**Euclidean twin.** Identical parameters and initialisation (concept points: the same tangent vectors); $\exp_0$ is replaced by identity, distance is Euclidean, cones are Euclidean ($\psi(x)=\arcsin(K/\|x\|)$), and the concept points are updated with Adam.

**Is it really Poincaré?** Yes: every lesion and concept is a point in the ball, concepts are optimised on the manifold, scores are functions of exact geodesic distance, and hierarchy losses use exact hyperbolic cones and origin distance. Tests verify the model's logits against an independent Möbius-addition implementation, that hyperbolic and Euclidean distance rank pairs differently near the boundary, and the optimiser against geoopt (Kochurov et al. 2020).

## 8. Objective

$$\mathcal{L}=\mathcal{L}_{concept}+w_h\,\mathcal{L}_{hier}+w_{img}\,\mathcal{L}_{img},\qquad w_h=1,\ w_{img}=0.1.$$

- $\mathcal{L}_{concept}$: BCE with soft targets on the 10 scored concepts, NaN targets masked, positive-class weight neg/pos on the training split (capped at 50).
- $\mathcal{L}_{hier}$: on the concept tree, all ancestor→descendant pairs (transitive closure) should satisfy child ∈ cone(parent) (mean violation); non-ancestor pairs are pushed out of each other's cones with margin 0.1 (up to 512 sampled negatives per step), preventing one wide cone from swallowing everything.
- $\mathcal{L}_{img}$: the lesion should lie inside the cones of the concepts it has (weighted by the soft label) and of the always-true `nodule` node.

## 9. Training and evaluation protocol

- AdamW (Loshchilov & Hutter 2019) for the weights and Riemannian Adam for the concept points (one optimiser), lr 5·10⁻³, weight decay 10⁻⁴ on weight matrices only; 5 % linear warm-up then cosine decay (Goyal et al. 2017; Loshchilov & Hutter 2017) stepped per batch; global gradient-norm clipping at 1.0 (Pascanu et al. 2013).
- Up to 200 epochs; model selection and early stopping (patience 30) on **validation loss** with strict improvement. AUROC alone peaked at epoch 0, before the hierarchy was learned, so it is not used for selection.
- Exact resume (model, optimiser, scheduler, all RNG states) and deterministic kernels; a pre-empted run ends with the same weights as an uninterrupted one (tested).
- **Concept metrics** (on binary view): per-concept and macro AUROC, AUPRC, F1 (threshold 0.5); Brier against soft labels.
- **Hierarchy metrics**: `edge_in_cone` (fraction of parent→child edges satisfied), `non_edge_in_cone` (unrelated pairs wrongly nested), `depth_spearman` (tree depth vs hyperbolic depth), `tree_map` (mean average precision of recovering each node's tree neighbours from distances; Nickel & Kiela 2017), `dist_spearman` (embedded distance vs tree path length over all pairs). Reported also for the initial points, since tree initialisation encodes part of the tree.
- **Consistency metrics**: `parent_child_violation` (child predicted present while its scored parent is absent, e.g. part-solid without subsolid) and `exclusive_violation` (not exactly one category of an exclusive group predicted present).
- **Planned experimental grid**: encoder/input {CT-CLIP 64 mm, CT-CLIP 96 mm, FMCIB} (chosen by the probe against the baselines) × geometry {Poincaré, Euclidean} × dimension {2, 4, 8, 16, 32, 64} × 5 folds × 3 seeds, plus ablations (tangent vs Riemannian concepts, text vs tree initialisation, no hierarchy loss, learned curvature, no grounding). The expected signature of a hyperbolic advantage is a Poincaré arm that holds concept AUROC and tree fidelity at low $d$ while the Euclidean arm degrades. Report mean ± SD; paired comparisons on the same folds and seeds.

## 10. Visualisation

`hypertrace-visualize` draws the learned space with two geometry-aware layouts: a *radial* view (radius = exact hyperbolic depth, angle = PCA of tangent-space directions) and a *disk* view (tangent-space PCA mapped back with $\exp_0$). It also plots concept depth against tree depth and lesion depth by concept, and exports an interactive HTML and a coordinates CSV. These are 2-D projections of a 64-d space and are descriptive, not evidence.

## 11. Software engineering

Plain YAML configs with `base:` inheritance, `${ENV:-default}` and strict `--set key=value` overrides; each run stores its resolved config plus command, git commit, dirty flag, library versions and SLURM job id. The cache stores its settings and refuses to mix incompatible runs. Hydra and PyTorch Lightning were removed after showing equivalence on 20 paired runs (Section 12). 107 unit and end-to-end tests run on a synthetic bucket in the exact real layout. The v1 configuration is kept (`experiments/v1_text_anchored.yaml`) and reproduces the earlier code bit for bit.

## 12. Verification status and limitations (state these in the paper)

**Verified.**

- 146 tests pass: Poincaré formulas and gradients near the boundary; exp map and parallel transport; Riemannian Adam against torch Adam/AdamW and geoopt; model-level Poincaré checks; FMCIB loading and preprocessing conventions; label parsing on the real bucket schema; preprocessing; manifest; pooling; the pipeline end to end including exact resume.
- Plain loop vs earlier Lightning/Hydra implementation: 20 runs, max |Δ test AUROC| = 3.0·10⁻⁸, max |Δ test AUPRC| = 3.0·10⁻⁸, max per-epoch |Δ val loss| = 5.4·10⁻⁷, same best epochs. (Synthetic data; absolute AUROCs are meaningless.)
- The real manifest (1,390 nodules) was built from the bucket tables; whole-scan CT-CLIP extraction, probe and the v1 training grid ran on the real data (Section 5, "First real result").
- On synthetic data, v2 (free concept points) reached macro AUROC 0.81 where v1 (text-anchored) stayed at 0.53, with the same features.

**Not yet verified / open.**

1. The only real-data results so far are v1 (chance level). v2 has not yet been run on real features: crop extractions are in progress, FMCIB has not been run with its released weights (its loader is strict and was tested on synthetic checkpoints in both supported layouts).
2. CT-CLIP internals (input shape, patch size, spacing, HU window, orientation, projection path) were written from memory; a load-time self-check catches shape mismatches only. Orientation and HU window should be confirmed against the official preprocessing.
3. Zoomed crops put CT-CLIP outside its training scale; whether that helps is an empirical question answered by the probe.
4. The size bins are defined by the diameter, so they are trivially predicted by `diameter_only`; they are kept as concepts for the tree and for Part 3, and morphology concepts are the informative comparison.
5. Concept labels come from radiologist ratings, not pathology; LIDC malignancy is a radiologist rating. Pathology (diagnosis sheet in `aux/`) is for Part 3. Some concepts are rare (68 linear, 101 ground-glass nodules), so their per-fold AUROC is noisy.
6. LUNA16 is a subset of LIDC-IDRI and cannot serve as external validation.
7. Nodules are very small relative to CT-CLIP's 15 mm tokens (1–3 tokens for the median nodule), which may limit any readout.
8. CT-CLIP weights are CC BY-NC-SA; the checkpoint is loaded with `weights_only=False`, so only the official file should be used.

## 13. References

*Reconstructed from memory while the build environment could not reach the web. **Verify authors, venue, year, pages and DOI against the publisher or arXiv before citing.** Entries marked † are the ones I am least sure of.*

**Data and foundation models**

1. Armato SG III, McLennan G, Bidaut L, et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans. *Medical Physics* 38(2):915–931, 2011.
2. Fedorov A, Hancock M, Clunie D, et al. Standardized representation, visualization and sharing of the LIDC-IDRI annotations using DICOM SEG and SR (DICOM re-encoding of the LIDC-IDRI collection). *Medical Physics*, 2020.† (check exact title and volume)
3. Clark K, Vendt B, Smith K, et al. The Cancer Imaging Archive (TCIA): maintaining and operating a public information repository. *Journal of Digital Imaging* 26:1045–1057, 2013.
4. Hamamci IE, Er S, Almas F, et al. Developing generalist foundation models from a multimodal dataset for 3D computed tomography (CT-RATE, CT-CLIP, CT-CHAT). arXiv:2403.17834, 2024.† (check for the journal version)
5. Radford A, Kim JW, Hallacy C, et al. Learning transferable visual models from natural language supervision (CLIP). *ICML*, 2021.
6. Boecking B, Usuyama N, Bannur S, et al. Making the most of text semantics to improve biomedical vision-language processing (CXR-BERT, BioViL). *ECCV*, 2022.
7. Villegas R, Babaeizadeh M, Kindermans P-J, et al. Phenaki: variable length video generation from open domain textual description. *ICLR*, 2023. (basis of the C-ViViT-style CTViT)
8. Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in CT images: the LUNA16 challenge. *Medical Image Analysis* 42:1–13, 2017.
9. Pai S, Bontempi D, Hadzic I, Prudente V, Sokač M, Chaunzwa TL, Bernatz S, Hosny A, Mak RH, Birkbak NJ, Aerts HJWL. Foundation model for cancer imaging biomarkers. *Nature Machine Intelligence* 6(3):354–367, 2024. doi:10.1038/s42256-024-00807-9. (FMCIB; verified on PMC.) Code: AIM-Harvard/foundation-cancer-image-biomarker; weights: Zenodo record 10528450.
10. Yan K, Wang X, Lu L, Summers RM. DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning. *Journal of Medical Imaging* 5(3):036501, 2018. (FMCIB's pretraining data)
11. Cardoso MJ, Li W, Brown R, et al. MONAI: an open-source framework for deep learning in healthcare. arXiv:2211.02701, 2022.

**Hyperbolic geometry and hyperbolic representation learning**

12. Nickel M, Kiela D. Poincaré embeddings for learning hierarchical representations. *NeurIPS*, 2017.
13. Ganea O-E, Bécigneul G, Hofmann T. Hyperbolic entailment cones for learning hierarchical embeddings. *ICML*, 2018. (entailment cones, half-aperture and exterior angle)
14. Ganea O-E, Bécigneul G, Hofmann T. Hyperbolic neural networks. *NeurIPS*, 2018. (Möbius operations, exp/log maps)
15. Ungar AA. *Analytic Hyperbolic Geometry and Albert Einstein's Special Theory of Relativity.* World Scientific, 2008. (gyrovector formalism)
16. Sarkar R. Low distortion Delaunay embedding of trees in hyperbolic plane. *Graph Drawing*, 2011.
17. Sala F, De Sa C, Gu A, Ré C. Representation tradeoffs for hyperbolic embeddings. *ICML*, 2018.
18. Desai K, Nickel M, Rajpurohit T, Johnson J, Vedantam R. Hyperbolic image-text representations (MERU). *ICML*, 2023. (tangent-space exp map, learnable scales, text-entails-image)
19. Guo Y, Wang X, Chen Y, Yu SX. Clipped hyperbolic classifiers are super-hyperbolic classifiers. *CVPR*, 2022. (feature clipping)
20. Khrulkov V, Mirvakhabova L, Ustinova E, Oseledets I, Lempitsky V. Hyperbolic image embeddings. *CVPR*, 2020.
21. Mathieu E, Le Lan C, Maddison CJ, Tomioka R, Teh YW. Continuous hierarchical representations with Poincaré variational auto-encoders. *NeurIPS*, 2019. (tangent-space parameterisation)
22. Bécigneul G, Ganea O-E. Riemannian adaptive optimization methods. *ICLR*, 2019. (Riemannian Adam used for the concept points)
23. Kochurov M, Karimov R, Kozlukov S. Geoopt: Riemannian optimization in PyTorch. arXiv:2005.02819, 2020. (reference implementation used in the tests)
24. Peng W, Varanka T, Mostafa A, Shi H, Zhao G. Hyperbolic deep neural networks: a survey. *IEEE TPAMI* 44(12):10023–10044, 2022.
25. Pal A, van Spengler M, Di Melo GM, et al. Compositional entailment learning for hyperbolic vision-language models (HyCoCLIP). *ICLR*, 2025.†
26. Yu Z, Nguyen T, Gal Y, et al. Skin lesion recognition with class-hierarchy regularized hyperbolic embeddings. *MICCAI*, 2022.†

**Clinical guidelines behind the concept tree**

27. American College of Radiology. Lung-RADS® v2022. ACR, 2022. (size bins 6 / 8 / 15 mm; solid, part-solid, ground-glass nodule types)
28. MacMahon H, Naidich DP, Goo JM, et al. Guidelines for management of incidental pulmonary nodules detected on CT images: from the Fleischner Society 2017. *Radiology* 284(1):228–243, 2017.

**Concept bottlenecks and interpretable nodule assessment**

29. Koh PW, Nguyen T, Tang YS, et al. Concept bottleneck models. *ICML*, 2020.
30. Shen S, Han SX, Aberle DR, Bui AA, Hsu W. An interpretable deep hierarchical semantic convolutional neural network for lung nodule malignancy classification. *Expert Systems with Applications* 128:84–95, 2019.
31. Hancock MC, Magnan JF. Lung nodule malignancy classification using only radiologist-quantified image features as inputs to statistical learning algorithms: probing the Lung Image Database Consortium dataset with two statistical learning methods. *Journal of Medical Imaging* 3(4):044504, 2016.†

**Optimisation and training practice**

32. Loshchilov I, Hutter F. Decoupled weight decay regularization (AdamW). *ICLR*, 2019.
33. Loshchilov I, Hutter F. SGDR: stochastic gradient descent with warm restarts. *ICLR*, 2017. (cosine schedule)
34. Goyal P, Dollár P, Girshick R, et al. Accurate, large minibatch SGD: training ImageNet in 1 hour. arXiv:1706.02677, 2017. (linear warm-up)
35. Pascanu R, Mikolov T, Bengio Y. On the difficulty of training recurrent neural networks. *ICML*, 2013. (gradient-norm clipping)
36. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4(9):100804, 2023. (motivation for patient-level splits and the leakage assertion)

**Software and platforms**

37. Paszke A, Gross S, Massa F, et al. PyTorch: an imperative style, high-performance deep learning library. *NeurIPS*, 2019.
38. Brett M, et al. nibabel (software, Zenodo). Pedregosa F, et al. Scikit-learn: machine learning in Python. *JMLR* 12:2825–2830, 2011.
39. Wolf T, et al. Transformers: state-of-the-art natural language processing. *EMNLP (Demos)*, 2020. Hugging Face Hub and Buckets documentation, <https://huggingface.co/docs/hub>.
40. Wang P. `vector-quantize-pytorch` v1.1.2 (GitHub, lucidrains), pinned because its parameter names are part of the CT-CLIP checkpoint.

**Project context (from the proposal; fill in exact citations from your proposal draft)**

41. Concept-based vision-language chest-CT reasoning (CLEAR), the chest-CT foundation model comparison (MARCUS), and CineMA, as cited in the HYPER-TRACE proposal.

## 14. Suggested mapping to manuscript sections

- *Introduction / related work*: refs 12–26 (hyperbolic), 29–31 (concept bottlenecks, LIDC interpretability), 4–6 and 9 (CT foundation models), 27–28 (clinical descriptors).
- *Method*: Sections 4–8 above; formulas in Section 7 (Riemannian Adam: refs 22–23).
- *Experiments*: Section 9 protocol and Section 3 data; ablations listed in Section 9.
- *Limitations*: Section 12.
