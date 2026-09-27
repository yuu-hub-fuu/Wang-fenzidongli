"""YAML configuration: ``configs/benchmark.yaml`` + ``configs/baselines/<key>.yaml``."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import yaml

from . import atlas_data
from .baselines import RunContext

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BENCHMARK_CONFIG = os.path.join(ROOT, "configs", "benchmark.yaml")
BASELINE_CONFIG_DIR = os.path.join(ROOT, "configs", "baselines")

_PATH_KEYS = {"atlas_dir", "split", "runs_dir", "third_party_dir", "repo_dir"}


def load_yaml(path: Optional[str]) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _expand(value, variables: Dict[str, str]):
    if isinstance(value, str):
        for k, v in variables.items():
            value = value.replace("${" + k + "}", str(v))
        return os.path.expanduser(value)
    if isinstance(value, list):
        return [_expand(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, variables) for k, v in value.items()}
    return value


def _abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def load_benchmark_config(path: Optional[str] = None) -> Dict:
    cfg = {
        "atlas_dir": "data/atlas",
        "split": os.path.relpath(atlas_data.DEFAULT_SPLIT, ROOT),
        "runs_dir": "runs",
        "third_party_dir": "third_party",
        "n_samples": 250,
        "seed": 42,
        "gpus": ["0"],
        "eval_workers": 8,
        "baselines": [],
    }
    cfg.update(load_yaml(path or DEFAULT_BENCHMARK_CONFIG))
    cfg = _expand(cfg, {"root": ROOT})
    for k in _PATH_KEYS & set(cfg):
        cfg[k] = _abs(cfg[k])
    cfg["gpus"] = [str(g) for g in cfg["gpus"]]
    return cfg


def load_baseline_config(key: str, bench: Dict, path: Optional[str] = None) -> Dict:
    path = path or os.path.join(BASELINE_CONFIG_DIR, f"{key}.yaml")
    cfg = _expand(load_yaml(path), {"root": ROOT, "third_party": bench["third_party_dir"],
                                     "atlas_dir": bench["atlas_dir"], "runs": bench["runs_dir"]})
    if cfg.get("repo_dir"):
        cfg["repo_dir"] = _abs(cfg["repo_dir"])
    return cfg


def make_context(
    key: str,
    bench: Dict,
    base_cfg: Dict,
    targets: Optional[Sequence[str]] = None,
    gpus: Optional[List[str]] = None,
    dry_run: bool = False,
    run_dir: Optional[str] = None,
) -> RunContext:
    options = dict(base_cfg.get("options", {}) or {})
    names = list(targets) if targets else None
    tlist = atlas_data.read_split(bench["split"], names)
    exclude = set(options.get("exclude_targets", []) or [])
    tlist = [t for t in tlist if t.name not in exclude]
    return RunContext(
        atlas_dir=bench["atlas_dir"],
        run_dir=run_dir or os.path.join(bench["runs_dir"], base_cfg.get("run_name", key)),
        targets=tlist,
        repo_dir=base_cfg.get("repo_dir"),
        python=base_cfg.get("python", "python"),
        gpus=[str(g) for g in (gpus or base_cfg.get("gpus") or bench["gpus"])],
        n_samples=int(base_cfg.get("n_samples", bench["n_samples"])),
        seed=int(base_cfg.get("seed", bench["seed"])),
        dry_run=dry_run,
        options=options,
    )
