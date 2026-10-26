"""On-disk cache of frozen-VLM outputs.

CT-CLIP is frozen, so it runs exactly once per nodule. Every training run
(both geometries, all folds, all seeds) then reads small tensors from here and
fits on one GPU in minutes.

Layout::

    <cache_dir>/
      meta.json               encoder + preprocessing settings (guards against mixing caches)
      text.pt                 prompts, owner index, L2-normalised text latents
      shards/<nodule_id>.pt   per-nodule readouts (written by possibly many SLURM array tasks)
      images.pt               consolidated readouts (built on first load)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch

IMAGE_KEYS = ("token_mean", "masked_latent", "global_latent", "occupancy_sum")  # VLM caches (v1)


def _safe(nodule_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", nodule_id)


class EmbeddingCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir).expanduser()
        self.shard_dir = self.dir / "shards"

    # ------------------------------------------------------------- metadata
    def write_meta(self, meta: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / "meta.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old != meta:
                raise RuntimeError(
                    f"Cache {self.dir} was built with different settings.\n old: {old}\n new: {meta}\n"
                    "Use a new cache_dir (or delete the old one)."
                )
            return
        _atomic_write_text(path, json.dumps(meta, indent=2, sort_keys=True))

    def read_meta(self) -> dict[str, Any]:
        return json.loads((self.dir / "meta.json").read_text())

    # ----------------------------------------------------------------- text
    def write_text(self, prompts: list[str], owner: torch.Tensor, latents: torch.Tensor, node_names: list[str]) -> None:
        _atomic_save(
            {"prompts": prompts, "owner": owner, "latents": latents.cpu(), "node_names": node_names},
            self.dir / "text.pt",
        )

    def has_text(self) -> bool:
        return (self.dir / "text.pt").exists()

    def read_text(self) -> dict[str, Any]:
        return torch.load(self.dir / "text.pt", map_location="cpu", weights_only=False)

    # ---------------------------------------------------------------- image
    def has(self, nodule_id: str) -> bool:
        return (self.shard_dir / f"{_safe(nodule_id)}.pt").exists()

    def write_image(self, nodule_id: str, readout: dict[str, torch.Tensor]) -> None:
        """Store every readout given (VLM: IMAGE_KEYS; lesion encoders: lesion_embedding + occupancy_sum)."""
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        _atomic_save({k: v.cpu() for k, v in readout.items()}, self.shard_dir / f"{_safe(nodule_id)}.pt")

    def load_images(self, nodule_ids: list[str]) -> dict[str, torch.Tensor]:
        """Stacked readouts in the order of ``nodule_ids``. Consolidates shards on first call."""
        consolidated = self.dir / "images.pt"
        if consolidated.exists():
            data = torch.load(consolidated, map_location="cpu", weights_only=False)
            if set(nodule_ids) <= set(data["ids"]):
                pos = {n: i for i, n in enumerate(data["ids"])}
                idx = torch.tensor([pos[n] for n in nodule_ids])
                return {k: v[idx] for k, v in data.items() if k != "ids"}
        if not (self.dir / "meta.json").exists():
            raise FileNotFoundError(
                f"{self.dir} is not an embedding cache (no meta.json). Pass the folder that "
                "hypertrace-extract logged as 'cache ...' (it contains meta.json, text.pt and shards/)."
            )
        missing = [n for n in nodule_ids if not self.has(n)]
        if missing:
            found = sorted(p.name for p in self.shard_dir.glob("*.pt")) if self.shard_dir.is_dir() else []
            raise FileNotFoundError(
                f"{len(missing)} of {len(nodule_ids)} nodules have no cached embedding in {self.shard_dir} "
                f"(e.g. {missing[:3]}). That folder holds {len(found)} shard files (e.g. {found[:3]}). "
                "If it holds none, extraction wrote elsewhere or has not run; if it holds others, the "
                "manifest differs from the one used for extraction."
            )
        rows = [torch.load(self.shard_dir / f"{_safe(n)}.pt", map_location="cpu") for n in nodule_ids]
        out = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
        _atomic_save({"ids": list(nodule_ids), **out}, consolidated)
        return out


def _atomic_save(obj: Any, path: Path) -> None:
    """Write to a temp file then rename, so a job killed mid-write never leaves a corrupt file."""
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)
