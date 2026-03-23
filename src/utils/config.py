"""
YAML-based config system.

Loading order (later overrides earlier):
  1. configs/base.yaml
  2. experiment YAML (via `base: base.yaml` inheritance)
  3. CLI key=value overrides

Usage:
    cfg = load_config("configs/mnist_baseline.yaml", overrides=["lr=0.0005"])
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


# ---------------------------------------------------------------------------
# Sub-configs (nested dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureConfig:
    hidden_layers: List[int] = field(default_factory=lambda: [512])


@dataclass
class TopologyConfig:
    mode: str = "learned"          # learned | full | random_sparse | transfer
    target_sparsity: float = 0.5   # used when mode == "random_sparse"
    transfer_from: str = ""        # checkpoint path when mode == "transfer"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # experiment identity
    experiment_name: str = "experiment"
    dataset: str = "mnist"

    # model
    n_input: int = 784
    n_output: int = 10
    T: int = 25
    beta: float = 0.9

    # architecture
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)

    # topology
    topology: TopologyConfig = field(default_factory=TopologyConfig)

    # annealing
    tau_start: float = 1.0
    tau_end: float = 0.05
    tau_anneal_epochs: int = 25

    # training
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    lr_min: float = 1e-5   # cosine scheduler의 최솟값
    lambda_sparse: float = 0.005
    lambda_commit: float = 0.08
    edge_threshold: float = 0.5
    seed: int = 42

    # paths
    data_dir: str = "./data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (in-place on a copy)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_config(d: dict) -> Config:
    arch_d = d.pop("architecture", {})
    topo_d = d.pop("topology", {})
    cfg = Config(**d)
    cfg.architecture = ArchitectureConfig(**arch_d)
    cfg.topology = TopologyConfig(**topo_d)
    return cfg


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_inheritance(yaml_path: str | Path) -> dict:
    """Load a YAML file, resolving a `base:` key by merging parent first."""
    configs_dir = Path(yaml_path).parent
    data = _load_yaml(yaml_path)
    base_name = data.pop("base", None)

    if base_name:
        base_path = configs_dir / base_name
        base_data = _resolve_inheritance(base_path)
        data = _deep_merge(base_data, data)

    return data


def _apply_cli_overrides(d: dict, overrides: List[str]) -> dict:
    """
    Apply overrides of the form "key=value" or "section.key=value".
    Tries to parse value as YAML scalar (int, float, bool, list, etc.).
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"CLI override must be key=value, got: {item!r}")
        key_path, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        keys = key_path.split(".")
        target = d
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path | None = None,
    overrides: List[str] | None = None,
) -> Config:
    """
    Load config with optional YAML file and CLI overrides.

    If config_path is None, returns Config() with defaults.
    """
    if config_path is not None:
        data = _resolve_inheritance(config_path)
    else:
        data = {}

    if overrides:
        data = _apply_cli_overrides(data, overrides)

    # extract nested sections before passing to Config()
    arch_d   = data.pop("architecture", {})
    topo_d   = data.pop("topology", {})

    cfg = Config(**data)
    cfg.architecture = ArchitectureConfig(**arch_d)
    cfg.topology     = TopologyConfig(**topo_d)
    return cfg
