"""Plain-YAML configuration (replaces Hydra).

A config is one YAML file. Three small conveniences, nothing else:

* ``base: other.yaml``  inherit from another file (path relative to this file). Mappings are
  merged recursively; lists and scalars are replaced. Used by ``configs/experiments/*.yaml``.
* ``${VAR}`` / ``${VAR:-default}``  environment variables inside strings, so cluster paths
  live in ``scripts/env.sh`` rather than in committed configs. An unset variable without a
  default is an error, not an empty string.
* ``--set a.b=value`` command-line overrides. Values are parsed as YAML (numbers, booleans,
  lists). Unknown keys are rejected, so a typo can't silently do nothing.

Keys named in ``PATH_KEYS`` hold file paths relative to the YAML they appear in and are
made absolute at load time. Every run saves its fully resolved config next to its results.
"""

from __future__ import annotations

import copy
import datetime as _dt
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(str(files("hypertrace") / "configs"))
PATH_KEYS = ("concepts_file",)
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class ConfigError(ValueError):
    pass


def default_config(name: str) -> Path:
    """Path of a packaged config, e.g. ``default_config("train")``."""
    return CONFIG_DIR / f"{name}.yaml"


def _deep_merge(base: dict, over: Mapping) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_paths(node: Any, here: Path) -> Any:
    if isinstance(node, dict):
        return {k: (str((here / v).resolve()) if k in PATH_KEYS and isinstance(v, str) and not v.startswith("$")
                    and not Path(v).is_absolute() else _resolve_paths(v, here)) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_paths(v, here) for v in node]
    return node


def _load_file(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    path = Path(path).expanduser().resolve()
    if path in _seen:
        raise ConfigError(f"Circular 'base:' chain: {' -> '.join(map(str, (*_seen, path)))}")
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    raw = _resolve_paths(raw, path.parent)
    base = raw.pop("base", None)
    if base is None:
        return raw
    return _deep_merge(_load_file(path.parent / base, (*_seen, path)), raw)


def _expand_env(node: Any, where: str = "") -> Any:
    if isinstance(node, dict):
        return {k: _expand_env(v, f"{where}.{k}" if where else k) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v, f"{where}[{i}]") for i, v in enumerate(node)]
    if not isinstance(node, str):
        return node

    def sub(m: re.Match) -> str:
        var, default = m.group(1), m.group(2)
        if var in os.environ:
            return os.environ[var]
        if default is not None:
            return default
        raise ConfigError(f"{where}: environment variable ${var} is not set (and has no :-default)")

    return _ENV.sub(sub, node)


def parse_value(text: str) -> Any:
    """YAML-parse an override value; '1e-3' becomes a float (YAML 1.1 would keep it a string)."""
    v = yaml.safe_load(text)
    if isinstance(v, str) and _NUM.match(v.strip()):
        return float(v) if any(c in v for c in ".eE") else int(v)
    return v


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for ov in overrides:
        if "=" not in ov:
            raise ConfigError(f"Override '{ov}' must look like key.sub=value")
        key, text = ov.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for i, p in enumerate(parts[:-1]):
            if not isinstance(node.get(p), dict):
                raise ConfigError(f"Override '{key}': '{'.'.join(parts[:i + 1])}' is not a section of the config")
            node = node[p]
        if parts[-1] not in node:
            raise ConfigError(f"Override '{key}': unknown key. Existing keys here: {sorted(node)}")
        node[parts[-1]] = parse_value(text)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    return _expand_env(apply_overrides(_load_file(Path(path)), overrides))


def get(cfg: Mapping, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, Mapping) or p not in node:
            return default
        node = node[p]
    return node


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def run_metadata(argv: list[str] | None = None) -> dict[str, Any]:
    """What a reviewer needs to reproduce a run: command, code version, library versions, time."""
    import numpy
    import torch

    from hypertrace import __version__

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True,
                                  timeout=5, check=True).stdout.strip() or None
        except Exception:
            return None

    return {
        "command": " ".join(argv if argv is not None else sys.argv),
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "hypertrace": __version__,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(numpy.__version__),
        "cuda": str(torch.version.cuda) if torch.version.cuda else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }


def save_config(cfg: Mapping, path: str | Path, meta: Mapping | None = None) -> None:
    doc = {"_meta": dict(meta)} if meta else {}
    doc.update(cfg)
    Path(path).write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=False))
