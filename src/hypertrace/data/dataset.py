"""Cached encoder readouts + LIDC concept labels for one fold (everything held in memory).

``standardize``: z-score every feature with the mean / SD of the TRAINING split only
(no information from val/test). Use it for features that are not L2-normalised CLIP
latents (token_mean, lesion-encoder embeddings). The statistics are kept in
``feature_mean`` / ``feature_std`` and saved with each run.

``pca_dim``: after standardising, project every nodule onto the top-k principal axes of
the TRAINING split (centred on the training mean). Fewer input weights for 4,096-d
lesion embeddings; the axes are kept in ``pca_components`` and saved with the run.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hypertrace.concepts import ConceptHierarchy, LabelBuilder
from hypertrace.data.embedding_cache import EmbeddingCache
from hypertrace.data.folds import assert_no_patient_leakage, split_indices
from hypertrace.data.manifest import load_manifest

log = logging.getLogger(__name__)


class _Split(Dataset):
    def __init__(self, x: torch.Tensor, soft: torch.Tensor, rows: np.ndarray) -> None:
        self.x, self.soft, self.rows = x, soft, torch.as_tensor(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return {"x": self.x[i], "soft": self.soft[i], "row": self.rows[i]}


class ConceptData:
    """Loads the manifest, labels and cached features once; hands out patient-level splits.

    test = ``test_fold``, val = the next fold, train = the remaining folds.
    """

    def __init__(
        self,
        manifest_path: str,
        cache_dir: str,
        hierarchy: ConceptHierarchy,
        test_fold: int = 0,
        n_folds: int = 5,
        image_feature: str = "masked_latent",
        binary_threshold: float = 0.5,
        reader_column_pattern: str = "{attribute}_r{reader}",
        batch_size: int = 64,
        min_occupancy: float = 1e-3,
        standardize: bool = False,
        pca_dim: int | None = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.batch_size = batch_size
        cache = EmbeddingCache(cache_dir)

        df = load_manifest(manifest_path)
        if "fold" not in df.columns:
            raise KeyError("Manifest has no `fold` column. Run `hypertrace-make-folds` first.")
        imgs = cache.load_images(df["nodule_id"].tolist())
        if image_feature not in imgs:
            raise KeyError(f"image_feature '{image_feature}' not in cache ({list(imgs)})")

        keep = (imgs["occupancy_sum"] > min_occupancy).numpy()
        if (~keep).any():
            log.warning("Dropping %d nodules with empty token occupancy: %s",
                        (~keep).sum(), df.loc[~keep, "nodule_id"].head().tolist())
        self.df = df.loc[keep].reset_index(drop=True)
        self.x = imgs[image_feature][torch.from_numpy(keep)].float()

        labels = LabelBuilder(hierarchy, binary_threshold, reader_column_pattern).build(self.df)
        self.concept_names = labels.names
        self.soft = torch.from_numpy(labels.soft)
        self.binary = torch.from_numpy(labels.binary)

        self.splits = split_indices(self.df["fold"].to_numpy(), test_fold, n_folds)
        assert_no_patient_leakage(self.df, self.splits)
        log.info("Split sizes (nodules): %s", {k: len(v) for k, v in self.splits.items()})

        self.feature_mean = self.feature_std = None
        if standardize:
            tr = self.x[torch.as_tensor(self.splits["train"])]
            self.feature_mean, self.feature_std = tr.mean(0), tr.std(0).clamp_min(1e-6)
            self.x = (self.x - self.feature_mean) / self.feature_std

        self.pca_mean = self.pca_components = None
        if pca_dim:
            tr = self.x[torch.as_tensor(self.splits["train"])].double()
            k = min(int(pca_dim), tr.shape[0] - 1, tr.shape[1])
            self.pca_mean = tr.mean(0)
            _, s, vt = torch.linalg.svd(tr - self.pca_mean, full_matrices=False)
            # sign convention (largest |loading| positive) so the axes do not flip between runs
            sign = torch.sign(vt[:k].gather(1, vt[:k].abs().argmax(1, keepdim=True))).squeeze(1)
            self.pca_components = (vt[:k] * sign.unsqueeze(1)).float()
            self.pca_mean = self.pca_mean.float()
            self.x = (self.x - self.pca_mean) @ self.pca_components.T
            kept = float((s[:k] ** 2).sum() / (s ** 2).sum())
            log.info("PCA: %d -> %d features (%.1f%% of training variance)", tr.shape[1], k, 100 * kept)

        # Prompt embeddings: needed for concepts.mode=text or init=text; absent for encoders without text.
        self.text_latents, self.text_problem = None, None
        if not cache.has_text():
            self.text_problem = "this cache has no text.pt (encoder without a text encoder)"
        else:
            text = cache.read_text()
            if text["node_names"] != hierarchy.names:
                self.text_problem = ("text.pt was built for a different concept tree; re-run "
                                     "`hypertrace-extract --set only_text=true cache_dir=<this cache>`")
            else:
                self.text_latents = prompt_ensemble(text["latents"], text["owner"], len(hierarchy))

    @property
    def image_dim(self) -> int:
        return self.x.shape[1]

    @property
    def text_dim(self) -> int | None:
        return None if self.text_latents is None else self.text_latents.shape[1]

    def pos_weight(self) -> torch.Tensor:
        """neg/pos ratio per concept on the training split (rare concepts get more weight)."""
        b = self.binary[self.splits["train"]]
        pos = torch.nansum(b, dim=0)
        n = (~torch.isnan(b)).sum(dim=0)
        return ((n - pos) / pos.clamp_min(1)).clamp(max=50.0)

    def loader(self, split: str, shuffle: bool = False, generator: torch.Generator | None = None) -> DataLoader:
        idx = self.splits[split]
        return DataLoader(_Split(self.x[idx], self.soft[idx], idx), batch_size=self.batch_size,
                          shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)


def prompt_ensemble(latents: torch.Tensor, owner: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Average L2-normalised prompt latents per node, then renormalise -> (n_nodes, D)."""
    lat = F.normalize(latents.float(), dim=-1)
    out = torch.zeros(n_nodes, lat.shape[1]).index_add_(0, owner, lat)
    counts = torch.bincount(owner, minlength=n_nodes).clamp_min(1).unsqueeze(1)
    return F.normalize(out / counts, dim=-1)
