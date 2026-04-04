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


@dataclass
class LiquidConfig:
    n_liquid: int = 200              # 리퀴드 뉴런 수
    exc_ratio: float = 0.8           # 흥분성 뉴런 비율
    p_input: float = 0.1             # 입력→리퀴드 연결 확률
    recurrent_mode: str = "learned"  # learned | random_sparse | fixed | grad_r
    recurrent_sparsity: float = 0.2  # random_sparse 모드 시 연결 확률
    self_connection: bool = False    # 자기 연결 허용 여부
    theta_init_mean: float = 0.0     # theta 초기화 평균 (음수→희소 초기 연결, e.g. -2.0→12%)
    theta_init_std: float = 0.01     # theta 초기화 표준편차
    grad_clip_max_norm_w: float = 100.0   # w_raw/readout gradient clipping (순환 BPTT: param 44k × T steps → norm 수백~수천이 정상)
    grad_clip_max_norm_theta: float = 10.0 # theta gradient clipping (w보다 작게 유지해 시간 스케일 분리 보장)
    input_weight_scale: float = 0.1  # 입력 가중치 스케일
    w_raw_max: float = -3.0          # w_raw 상한 clamp (softplus(-3.0)≈0.049, spectral radius < 1 for N≤500)
    bptt_truncate: int = 0           # truncated BPTT: 마지막 K 타임스텝만 gradient 흘림 (0 = full BPTT)
    beta_min: float = 0.7            # 뉴런별 LIF leak 범위 (하한)
    beta_max: float = 0.95           # 뉴런별 LIF leak 범위 (상한)
    threshold_min: float = 0.8       # 뉴런별 발화 임계값 범위 (하한)
    threshold_max: float = 1.5       # 뉴런별 발화 임계값 범위 (상한)
    theta_warmup_epochs: int = 0     # Phase 1 길이: theta 고정, w_raw/readout만 학습 (0=비활성화)
    theta_lr_scale: float = 0.1      # theta LR = base_lr × theta_lr_scale
    noise_scale: float = 0.1         # 에폭 단위 Gumbel noise 크기 (0=결정적, 1=표준 Gumbel std≈1.81)
                                     # 작은 값 → 경계(theta≈0) 엣지만 뒤집힘, 확실한 ON/OFF는 유지


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

    # liquid (LSM 전용)
    liquid: LiquidConfig = field(default_factory=LiquidConfig)

    # annealing
    tau_start: float = 1.0
    tau_end: float = 0.05
    tau_anneal_epochs: int = 25
    tau_hold_epochs: int = 0         # Phase 2 시작 후 tau=tau_start를 유지하는 epoch 수 (이후 annealing 시작)

    # training
    epochs: int = 100
    patience: int = 20   # early stopping patience (0 = 비활성화)
    batch_size: int = 128
    lr: float = 1e-3
    lr_min: float = 1e-5   # cosine scheduler의 최솟값
    lambda_sparse: float = 0.005
    lambda_commit: float = 0.08
    weight_decay: float = 0.0
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
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
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
    liq_d    = data.pop("liquid", {})

    cfg = Config(**data)
    cfg.architecture = ArchitectureConfig(**arch_d)
    cfg.topology     = TopologyConfig(**topo_d)
    cfg.liquid       = LiquidConfig(**liq_d)
    return cfg
