"""Run a frozen encoder over the bucket scans and cache one lesion readout per nodule.

    hypertrace-extract                                                   # packaged configs/extract.yaml (CT-CLIP)
    hypertrace-extract --set input_mode=crop crop_size_mm=64 cache_dir=$HT_CACHE_ROOT/ctclip_crop64
    hypertrace-extract --set encoder.name=fmcib                          # lesion-level encoder (FMCIB)
    hypertrace-extract --set encoder.name=fake device=cpu                # dry run, no weights needed
    hypertrace-extract --set only_text=true cache_dir=...                # re-encode prompts after editing the tree
    sbatch slurm/extract.sbatch                                          # SLURM array, sharded by scan

Each scan (``images/<case>.nii.gz``) is loaded once, normalised (HU, the encoder's own
settings) and resampled to the encoder's spacing. For each of its nodules,
``masks_instance == instance_label`` gives the grounding mask, cross-checked against
``consensus_voxels`` from nodules.csv. Re-running skips nodules already cached, so a
pre-empted job resumes where it stopped.

CT-CLIP (token map): the mask selects tokens (readouts token_mean / masked_latent /
global_latent). Lesion encoders (FMCIB): a fixed cube centred on the mask's centre of mass
is embedded (readout lesion_embedding); ``occupancy_sum`` is then the fraction of the
nodule inside the cube (< 1 means the nodule is larger than the cube).
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hypertrace.concepts import ConceptHierarchy
from hypertrace.concepts.ontology import load_concept_tree
from hypertrace.config import ConfigError, default_config, load_config, run_metadata, save_config
from hypertrace.data.embedding_cache import EmbeddingCache
from hypertrace.data.manifest import load_manifest
from hypertrace.data.volumes import check_spacing, load_volume
from hypertrace.encoders import build_encoder
from hypertrace.encoders.pooling import readout_from_tokens
from hypertrace.encoders.preprocess import (
    center_of_mass,
    centered_crop,
    crop_window,
    lesion_crop,
    mask_bbox,
    normalize_hu,
    prepare_ct,
    prepare_mask,
    scan_window,
)

log = logging.getLogger("hypertrace.extract")
READOUT_KEYS = ("token_mean", "masked_latent", "global_latent", "occupancy_sum")
LESION_ENCODERS = ("fmcib", "fake_lesion")


class InstanceLabelError(RuntimeError):
    pass


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                        ("cpu" if name == "auto" else name))


HU_KEYS = ("hu_window", "hu_scale", "hu_shift")


def cache_dir_for(cfg: dict) -> str:
    if cfg.get("cache_dir"):
        return cfg["cache_dir"]
    if cfg["encoder"]["name"] in LESION_ENCODERS:
        return str(Path(cfg["paths"]["cache_root"]) / f"{cfg['encoder']['name']}_lesion")
    name = f"{cfg['encoder']['name']}_{cfg['input_mode']}_dil{cfg['dilate_tokens']}"
    return str(Path(cfg["paths"]["cache_root"]) / name)


def _check_label(inst: np.ndarray, label: int, expected: float | None, name: str, rel_tol: float) -> np.ndarray:
    """Binary mask for ``label``; fail if its size disagrees with nodules.csv (wrong label mapping)."""
    binary = inst == label
    n = int(binary.sum())
    if n == 0:
        present = np.unique(inst[inst > 0])
        raise InstanceLabelError(f"{name}: label {label} absent from instance mask (labels present: {present[:20]})")
    if expected is not None and np.isfinite(expected) and expected > 0:
        if abs(n - expected) / expected > rel_tol:
            labels, counts = np.unique(inst[inst > 0], return_counts=True)
            raise InstanceLabelError(
                f"{name}: label {label} has {n} voxels but consensus_voxels={int(expected)}. "
                f"Label sizes in this scan: {dict(zip(labels.tolist(), counts.tolist(), strict=True))}. "
                "masks_instance ids may not equal nodules.csv nodule_id."
            )
    return binary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=default_config("extract"))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(a.config, a.set)
    except ConfigError as e:
        raise SystemExit(f"config error: {e}") from None

    shard_id, num_shards = int(cfg["shard_id"]), int(cfg["num_shards"])
    dev = _device(cfg["device"])
    enc = dict(cfg["encoder"][cfg["encoder"]["name"]])
    ts = enc.pop("target_spacing_mm", None)
    target_spacing = [float(v) for v in ts] if ts is not None else None
    # HU settings: top level (CT-CLIP), overridable per encoder (FMCIB: (HU + 1024) / 3072)
    hu = {k: cfg.get(k, 0.0 if k == "hu_shift" else None) for k in HU_KEYS}
    hu.update({k: enc.pop(k) for k in HU_KEYS if k in enc})
    hu_window = tuple(float(v) for v in hu["hu_window"])
    hu_scale, hu_shift = float(hu["hu_scale"]), float(hu["hu_shift"])
    lesion_mode = cfg["encoder"]["name"] in LESION_ENCODERS
    cache_dir = cache_dir_for(cfg)
    log.info("Shard %d/%d on %s, %s, cache %s", shard_id, num_shards, dev,
             "lesion-centred cube" if lesion_mode else f"input_mode={cfg['input_mode']}", cache_dir)

    tree = load_concept_tree(cfg["concepts_file"])
    hierarchy = ConceptHierarchy.from_config(tree)
    cache = EmbeddingCache(cache_dir)
    meta = {
        "encoder": cfg["encoder"]["name"],
        "encoder_kwargs": {k: v for k, v in enc.items() if k not in ("checkpoint", "text_model")},
        "target_spacing_mm": target_spacing,
        "input_mode": "lesion" if lesion_mode else cfg["input_mode"],
        "crop_size_mm": float(cfg["crop_size_mm"]) if (cfg["input_mode"] == "crop" and not lesion_mode) else None,
        "dilate_tokens": int(cfg["dilate_tokens"]),
        "hu_already_normalized": bool(cfg["hu_already_normalized"]),
        "hu_window": list(hu_window),
    }
    if hu_shift:  # only recorded when used, so caches built before this option still match
        meta["hu_shift"] = hu_shift
    if hu_scale != float(cfg["hu_scale"]):
        meta["hu_scale"] = hu_scale
    cache.write_meta(meta)
    logs = Path(cache_dir) / "extract_logs"
    logs.mkdir(parents=True, exist_ok=True)
    save_config({**cfg, "concepts": tree}, logs / f"config_shard{shard_id}.yaml", run_metadata(argv))

    vlm = build_encoder(cfg["encoder"]["name"], **enc).to(dev)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if (cfg["amp"] and dev.type == "cuda") else nullcontext()

    if lesion_mode:
        if cfg["only_text"]:
            raise SystemExit(f"encoder '{cfg['encoder']['name']}' has no text encoder (nothing to re-encode).")
    elif shard_id == 0 or cfg["only_text"]:
        prompts, owner = hierarchy.all_prompts()
        lat = []
        bs = int(cfg["text_batch_size"])
        with torch.inference_mode(), amp:
            for i in range(0, len(prompts), bs):
                lat.append(vlm.encode_text(prompts[i : i + bs]).float().cpu())
        cache.write_text(prompts, owner, F.normalize(torch.cat(lat), dim=-1), hierarchy.names)
        log.info("Encoded %d prompts for %d concept nodes", len(prompts), len(hierarchy))
    if cfg["only_text"]:
        return

    df = load_manifest(cfg["paths"]["manifest"], cfg["paths"]["bucket_root"])
    cases = sorted(df["case_id"].unique())
    mine = [c for i, c in enumerate(cases) if i % num_shards == shard_id]
    if cfg.get("limit_scans"):
        mine = mine[: int(cfg["limit_scans"])]  # quick real-data check (e.g. GPU memory) before the full run
    df = df[df["case_id"].isin(mine)]
    todo = df[[not cache.has(n) for n in df["nodule_id"]]]
    log.info("%d scans / %d nodules in shard, %d nodules left", len(mine), len(df), len(todo))

    canvas = tuple(vlm.crop_size) if lesion_mode else tuple(vlm.input_shape)
    air = normalize_hu(torch.tensor(-1024.0 if lesion_mode else -1000.0), *hu_window, scale=hu_scale,
                       shift=hu_shift).item()
    n_partial = 0
    t0, n_done, n_windows, n_empty = time.time(), 0, 0, 0
    for case_id, rows in todo.groupby("case_id", sort=True):
        r0 = rows.iloc[0]
        img = load_volume(r0["image_path"])
        inst = load_volume(r0["instance_mask_path"])
        if img.array.shape != inst.array.shape:
            raise ValueError(f"{case_id}: image {img.array.shape} vs instance mask {inst.array.shape}")
        spacing_cols = [r0.get(c) for c in ("spacing_z", "spacing_y", "spacing_x")]
        check_spacing(img, None if any(v is None for v in spacing_cols) else tuple(map(float, spacing_cols)), case_id)
        ct = prepare_ct(img.array, img.spacing, target_spacing, dev, cfg["hu_already_normalized"],
                        hu_window, hu_scale, cfg["check_intensity"], case_id, hu_shift)
        grid = tuple(ct.shape[2:])
        token_cache: dict[tuple, torch.Tensor] = {}  # window start -> tokens (scan mode)

        for row in rows.itertuples():
            binary = _check_label(inst.array, int(row.instance_label), getattr(row, "consensus_voxels", None),
                                  row.nodule_id, float(cfg["label_size_tolerance"]))
            mask = prepare_mask(binary, grid, dev)
            bbox = mask_bbox(mask)
            if bbox is None:
                raise InstanceLabelError(f"{row.nodule_id}: mask vanished after resampling")
            if lesion_mode:
                com = center_of_mass(mask)
                crop = centered_crop(ct, com, canvas, air)
                inside = float(centered_crop(mask, com, canvas, 0.0).sum() / mask.sum().clamp_min(1e-12))
                if vlm.flip_axes:
                    crop = crop.flip([2 + a for a in vlm.flip_axes])
                with torch.inference_mode(), amp:
                    emb = vlm.encode(crop).float()
                if inside < 0.9:
                    n_partial += 1
                    log.warning("%s: only %.0f%% of the nodule lies inside the %s-voxel cube", row.nodule_id,
                                100 * inside, "x".join(map(str, canvas)))
                cache.write_image(row.nodule_id, {"lesion_embedding": emb[0].cpu(),
                                                  "occupancy_sum": torch.tensor(inside)})
                n_done += 1
                n_windows += 1
                continue
            with torch.inference_mode(), amp:
                if cfg["input_mode"] == "scan":
                    start = scan_window(grid, canvas, bbox)
                    if start not in token_cache:
                        token_cache[start] = vlm.encode_tokens(crop_window(ct, start, canvas, air))
                        n_windows += 1
                    tokens = token_cache[start]
                    m_in = crop_window(mask, start, canvas, 0.0)
                elif cfg["input_mode"] == "crop":
                    size = [max(1, round(float(cfg["crop_size_mm"]) / s)) for s in (target_spacing or img.spacing)]
                    tokens = vlm.encode_tokens(lesion_crop(ct, bbox, size, canvas, air))
                    m_in = lesion_crop(mask, bbox, size, canvas, 0.0)
                    n_windows += 1
                else:
                    raise ValueError(f"input_mode must be 'scan' or 'crop', got {cfg['input_mode']!r}")
                r = readout_from_tokens(vlm, tokens, m_in, int(cfg["dilate_tokens"]))
            if float(r.occupancy_sum[0]) <= 1e-3:
                n_empty += 1
                log.warning("%s: nodule covers no tokens (mask outside window?)", row.nodule_id)
            cache.write_image(row.nodule_id, {k: getattr(r, k)[0] for k in READOUT_KEYS})
            n_done += 1
        if n_done and (n_done % 50 < len(rows) or n_done == len(todo)):
            log.info("%d/%d nodules, %d encoder passes, %.1f s/nodule", n_done, len(todo), n_windows,
                     (time.time() - t0) / n_done)
    log.info("Shard finished: %d nodules, %d encoder passes, %d with empty occupancy%s", n_done, n_windows, n_empty,
             f", {n_partial} larger than the cube" if lesion_mode else "")
    if dev.type == "cuda":
        log.info("Peak GPU memory: %.1f GB allocated (%.1f GB reserved) on %s",
                 torch.cuda.max_memory_allocated(dev) / 2**30, torch.cuda.max_memory_reserved(dev) / 2**30,
                 torch.cuda.get_device_name(dev))


if __name__ == "__main__":
    main()
