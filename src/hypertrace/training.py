"""Plain PyTorch training loop for the concept space (replaces PyTorch Lightning).

Per epoch: one pass over the training split (AdamW, per-step warm-up + cosine LR,
gradient-norm clipping), then validation (and, with ``eval_train``, a scoring pass over
the training split for train/val curves). ``select_metric`` drives model selection
(``best.pt``) and early stopping (stop after ``patience`` epochs without a strict
improvement). Default ``val/loss_concept``: the concept BCE only. The total loss also
contains the tree terms, which can keep falling while concept prediction overfits.
Metrics ending in ``loss...`` are minimised, everything else (AUROC...) maximised.
``last.pt`` holds everything needed to resume a pre-empted SLURM job exactly: model,
optimiser, scheduler, epoch counters and RNG states.

Data are a few hundred cached vectors per split, so batches are tiny and one device
is plenty. Metrics are computed on the full split at the end of each pass.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hypertrace.eval.metrics import concept_metrics, consistency_metrics, group_metrics
from hypertrace.models.model import ConceptSpaceModel
from hypertrace.optim import RiemannianAdamW

log = logging.getLogger("hypertrace.training")


# ---------------------------------------------------------------- setup helpers
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cuBLAS
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def warmup_cosine(total_steps: int, warmup_frac: float) -> Callable[[int], float]:
    warm = max(1, int(warmup_frac * total_steps))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total_steps - warm))))

    return f


def make_optimizer(model: ConceptSpaceModel, optim: dict, total_steps: int):
    """v1 (text-anchored concepts): torch AdamW, unchanged. Free concept points: AdamW for the
    weights + Riemannian Adam for the points (``hypertrace/optim.py``), one optimiser, one schedule."""
    groups = model.param_groups(optim["weight_decay"], optim.get("concept_lr"), optim.get("curvature_lr"))
    if any(g.get("manifold") for g in groups) or hasattr(model, "concept_param"):
        opt = RiemannianAdamW(groups, model.manifold_geometries(), lr=optim["lr"],
                              second_moment=optim.get("riemannian_second_moment", "coordinate"))
    else:
        opt = torch.optim.AdamW(groups, lr=optim["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total_steps, optim.get("warmup_frac", 0.05)))
    return opt, sched


# ------------------------------------------------------------------ one pass
def _to(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(model: ConceptSpaceModel, loader: DataLoader, opt, sched, device: torch.device,
                    grad_clip_norm: float | None) -> dict[str, float]:
    model.train()
    sums: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = _to(batch, device)
        out = model(batch["x"])
        parts = model.loss(out, batch["soft"])
        if not torch.isfinite(parts["total"]):
            raise FloatingPointError("Non-finite loss; check curvature bounds / clip_r / learning rate.")
        opt.zero_grad(set_to_none=True)
        parts["total"].backward()
        if grad_clip_norm:
            # Norm over parameters in optimiser-group order (float summation order matters for
            # bit-level reproducibility; this matches the original Lightning implementation).
            params = [p for g in opt.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
        opt.step()
        sched.step()
        b = len(batch["x"])
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach()) * b
        n += b
    return {f"train/loss_{k}": v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model: ConceptSpaceModel, loader: DataLoader, device: torch.device, stage: str,
             binary_threshold: float = 0.5) -> dict[str, float]:
    """Batch-size-weighted mean losses (total and each part) + concept metrics on the full split +
    hierarchy metrics. Keys: ``<stage>/loss`` (total), ``<stage>/loss_<part>``, ``<stage>/auroc_macro``, ..."""
    model.eval()
    sums, n, probs, soft = {}, 0, [], []
    for batch in loader:
        batch = _to(batch, device)
        out = model(batch["x"])
        for k, v in model.loss(out, batch["soft"]).items():
            sums[k] = sums.get(k, 0.0) + float(v) * len(batch["x"])
        n += len(batch["x"])
        probs.append(torch.sigmoid(out["logits"]).cpu())
        soft.append(batch["soft"].cpu())
    m = concept_metrics(torch.cat(probs).numpy(), torch.cat(soft).numpy(), model.hierarchy.scored_names,
                        binary_threshold)
    p_np, s_np = torch.cat(probs).numpy(), torch.cat(soft).numpy()
    m.update(consistency_metrics(p_np, model.hierarchy))
    m.update(model.tree_metrics())
    m.update(group_metrics(model, p_np, s_np, binary_threshold))
    out = {f"{stage}/loss": sums.get("total", 0.0) / max(n, 1)}
    out.update({f"{stage}/loss_{k}": v / max(n, 1) for k, v in sums.items() if k != "total"})
    out.update({f"{stage}/{k}": (v if math.isfinite(v) else 0.0) for k, v in m.items()})
    return out


@torch.no_grad()
def predict(model: ConceptSpaceModel, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    rows, probs, lesion, depth = [], [], [], []
    for batch in loader:
        batch = _to(batch, device)
        out = model(batch["x"])
        rows.append(batch["row"].cpu())
        probs.append(torch.sigmoid(out["logits"]).cpu())
        lesion.append(out["lesion"].cpu())
        depth.append(model.geometry.depth(out["lesion"]).cpu())
    return {"row": torch.cat(rows).numpy(), "probs": torch.cat(probs).numpy(),
            "lesion": torch.cat(lesion).numpy(), "lesion_depth": torch.cat(depth).numpy()}


# --------------------------------------------------------------- checkpoints
def _rng_state() -> dict[str, Any]:
    # The training loader shuffles from the global torch RNG, so this also restores the batch order.
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s: dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def _save(path: Path, **state: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a job killed mid-write never leaves a corrupt checkpoint


def load_weights(model: ConceptSpaceModel, path: str | Path, device: torch.device) -> None:
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])


# ----------------------------------------------------------------------- fit
@dataclass
class FitResult:
    best_path: Path
    best_epoch: int
    best_val_loss: float  # best value of the selection metric (AUROC for AUROC selection, despite the name)
    epochs_run: int
    stopped_early: bool
    select_metric: str = "val/loss"


def selection_sign(metric: str) -> float:
    """+1 for metrics to minimise (``.../loss...``), -1 for metrics to maximise (AUROC, AUPRC, ...)."""
    return 1.0 if metric.split("/")[-1].startswith("loss") or "brier" in metric or "violation" in metric else -1.0


def fit(model: ConceptSpaceModel, train_loader: DataLoader, val_loader: DataLoader,
        optim: dict, trainer: dict, run_dir: Path, device: torch.device,
        binary_threshold: float = 0.5, train_eval_loader: DataLoader | None = None) -> FitResult:
    """``train_eval_loader``: the training split in fixed order, scored every epoch (train/... metrics)."""
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
    max_epochs, patience = int(trainer["max_epochs"]), int(trainer["early_stopping_patience"])
    metric = trainer.get("select_metric", "val/loss")
    if not metric.startswith("val/"):
        raise ValueError(f"trainer.select_metric must be a validation metric (val/...), got {metric!r}")
    sign = selection_sign(metric)
    # no selection before this epoch: concept loss can bottom out before the tree terms have arranged the
    # concept points (an early best.pt would hold an untrained hierarchy). Clamped so a short run still
    # selects from its last epoch.
    select_from = min(int(trainer.get("select_from_epoch", 0)), max_epochs - 1)
    opt, sched = make_optimizer(model, optim, total_steps=max_epochs * len(train_loader))

    start_epoch, best, best_epoch, wait, stopped = 0, math.inf, -1, 0, False
    if last_path.exists():
        s = torch.load(last_path, map_location=device, weights_only=False)
        if s.get("select_metric", "val/loss") != metric:
            raise RuntimeError(f"{last_path} was trained selecting on {s.get('select_metric', 'val/loss')!r}, "
                               f"this config selects on {metric!r}. Use another run folder (tag=...).")
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        sched.load_state_dict(s["scheduler"])
        _set_rng_state(s["rng"])
        start_epoch, best, best_epoch, wait, stopped = (s["epoch"] + 1, s["best"], s["best_epoch"], s["wait"],
                                                         s["stopped"])
        log.info("Resumed from %s at epoch %d (best %s %.4f @ %d)", last_path, start_epoch, metric, sign * best,
                 best_epoch)

    log_path = run_dir / "metrics_per_epoch.csv"
    t0 = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, max_epochs):
        if stopped:
            break
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"]}
        row.update(train_one_epoch(model, train_loader, opt, sched, device, trainer.get("grad_clip_norm")))
        row.update(model.curvature_log())
        if train_eval_loader is not None:  # fixed order, eval mode (no dropout): comparable with val
            ev = evaluate(model, train_eval_loader, device, "train", binary_threshold)
            row.update({k: v for k, v in ev.items() if not k.startswith("train/loss")})
        row.update(evaluate(model, val_loader, device, "val", binary_threshold))
        _append_csv(log_path, row)

        if metric not in row:
            raise KeyError(f"trainer.select_metric {metric!r} is not computed; available: "
                           f"{sorted(k for k in row if k.startswith('val/'))[:40]}")
        current = sign * row[metric]
        if epoch < select_from:
            pass  # warm-up: neither selection nor early stopping yet
        elif current < best:  # strict improvement, as in the original setup
            best, best_epoch, wait = current, epoch, 0
            _save(best_path, model=model.state_dict(), epoch=epoch, val_loss=row["val/loss"],
                  select_metric=metric, select_value=row[metric])
        else:
            wait += 1
            stopped = wait >= patience
        _save(last_path, model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
              rng=_rng_state(), epoch=epoch, best=best, best_epoch=best_epoch, wait=wait, stopped=stopped,
              select_metric=metric)
        if epoch % 10 == 0 or stopped:
            tr = f"  train AUROC {row['train/auroc_macro']:.3f}" if "train/auroc_macro" in row else ""
            log.info("epoch %3d  train loss %.4f%s  val AUROC %.3f  %s %.4f  (best %.4f @ %d)%s",
                     epoch, row["train/loss_total"], tr, row["val/auroc_macro"], metric, row[metric],
                     sign * best, best_epoch, "  -> early stop" if stopped else "")
    log.info("Training finished after %d epochs in %.0f s", epoch + 1, time.time() - t0)
    return FitResult(best_path, best_epoch, sign * best, epoch + 1, stopped, metric)


def _append_csv(path: Path, row: dict[str, float]) -> None:
    """Append one epoch; rewrites the header if a later epoch adds columns (keeps earlier rows)."""
    rows: list[dict] = []
    if path.exists():
        with path.open() as f:
            rows = list(csv.DictReader(f))
    rows.append({k: v for k, v in row.items()})
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
