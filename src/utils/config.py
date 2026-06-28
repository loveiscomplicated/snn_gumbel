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
    mode: str = "learned"  # learned | full | random_sparse | transfer
    target_sparsity: float = 0.5  # used when mode == "random_sparse"
    transfer_from: str = ""  # checkpoint path when mode == "transfer"


@dataclass
class DiagnosticsConfig:
    enabled: bool = False
    log_every_epoch: bool = True
    topology_log_interval: int = 5
    save_raw_jsonl: bool = True
    save_summary_json: bool = True
    save_red_flags_json: bool = True
    save_markdown_report: bool = True
    save_trend_plots: bool = True
    save_full_topology_snapshots: bool = False
    full_snapshot_epochs: List[str] = field(
        default_factory=lambda: ["best", "freeze", "final"]
    )

    # Activity helper thresholds for scalar summaries.
    silent_firing_rate_threshold: float = 0.001
    overactive_firing_rate_threshold: float = 0.20

    # Conservative red-flag heuristic thresholds.
    missing_required_fraction_threshold: float = 0.35
    dead_mean_firing_rate_threshold: float = 0.005
    dead_fraction_epochs: float = 0.70
    adaptation_near_zero_threshold: float = 0.01
    val_improvement_min_delta: float = 0.01
    high_max_firing_rate_threshold: float = 0.80
    rec_input_high_threshold: float = 2.0
    topology_entropy_drop_threshold: float = 0.25
    degree_gini_rise_threshold: float = 0.20
    top_edge_prob_rise_threshold: float = 0.02
    theta_grad_spike_abs_threshold: float = 50.0
    theta_grad_spike_multiplier: float = 3.0
    adaptation_saturation_threshold: float = 1.0
    adaptation_saturation_flat_delta: float = 0.05


@dataclass
class LiquidConfig:
    n_liquid: int = 200  # 리퀴드 뉴런 수
    exc_ratio: float = 0.8  # 흥분성 뉴런 비율
    neuron_type: str = "lif"  # lif | alif
    p_input: float = 0.1  # 입력→리퀴드 연결 확률
    recurrent_mode: str = "learned"  # learned | learned_lowrank | learned_lowrank_grad_r | random_sparse | fixed | grad_r | ablation modes
    recurrent_sparsity: float = 0.2  # random_sparse 모드 시 연결 확률
    self_connection: bool = False  # 자기 연결 허용 여부
    theta_init_mean: float = (
        0.0  # theta 초기화 평균 (음수→희소 초기 연결, e.g. -2.0→12%)
    )
    theta_init_std: float = 0.01  # theta 초기화 표준편차
    theta_rank: int = 16  # learned_lowrank 모드에서 source/target 임베딩 차원
    theta_lowrank_init_std: float = (
        0.30  # learned_lowrank 임베딩 초기화 표준편차
    )
    grad_clip_max_norm_w: float = (
        100.0  # w_raw/readout gradient clipping (순환 BPTT: param 44k × T steps → norm 수백~수천이 정상)
    )
    grad_clip_max_norm_theta: float = (
        10.0  # theta gradient clipping (w보다 작게 유지해 시간 스케일 분리 보장)
    )
    input_weight_scale: float = 0.1  # 입력 가중치 스케일
    input_projection_mode: str = "fixed_sparse"  # fixed_sparse | learned_sparse
    train_input_projection: bool = False  # learned_sparse 입력 projection 학습 여부
    input_proj_lr_scale: float = 1.0  # input projection LR = base LR × scale
    input_proj_grad_clip: float | None = None  # input projection 전용 grad clip
    w_raw_init_mean: float = -4.0  # recurrent raw weight 초기 평균; softplus(-4.0)≈0.018
    w_raw_init_std: float = 0.01  # recurrent raw weight 초기 표준편차
    train_w_raw: bool = True  # recurrent raw weight 학습 여부
    w_raw_max: float = (
        -3.0
    )  # w_raw 상한 clamp (softplus(-3.0)≈0.049, spectral radius < 1 for N≤500)
    recurrent_weight_scale: float = 1.0  # smooth conductance recurrent gain scale
    match_initial_w_eff_scale: bool = False  # smooth lowrank 초기 W_eff norm matching
    frozen_w_mode: str = "initialized_w"  # initialized_w | constant_g
    frozen_w_constant_g: float | None = None  # constant_g mode conductance; None=derive
    temp_init: float = 1.0  # soft_gate sigmoid temperature at phase-2 start
    temp_final: float = 0.2  # soft_gate final sigmoid temperature; must stay > 0
    target_density_init: float = 0.3  # soft_gate initial gate-density target
    target_density_final: float = 0.05  # soft_gate final gate-density target
    target_anneal_epochs: int = 40  # soft_gate phase-2 anneal length
    density_penalty_lambda: float = 0.0  # soft_gate gate-only density penalty
    mag_from_separate_param: bool = False  # soft_gate optional separate mag tensor
    bptt_truncate: int = (
        0  # truncated BPTT: 마지막 K 타임스텝만 gradient 흘림 (0 = full BPTT)
    )
    beta_min: float = 0.7  # 뉴런별 LIF leak 범위 (하한)
    beta_max: float = 0.95  # 뉴런별 LIF leak 범위 (상한)
    threshold_min: float = 0.8  # 뉴런별 발화 임계값 범위 (하한)
    threshold_max: float = 1.5  # 뉴런별 발화 임계값 범위 (상한)
    alif_rho_init: float = 0.9  # ALIF adaptation decay 초기값
    alif_beta_init: float = 0.4  # ALIF adaptation strength 초기값
    alif_adapt_increment: float = 1.0  # ALIF spike-to-adaptation increment scale
    alif_learn_rho: bool = False  # ALIF rho 학습 여부
    alif_learn_beta: bool = False  # ALIF beta 학습 여부
    init_mode: str = "manual"  # manual | fdi_calibrated
    fdi_probe_batches: int = 8
    fdi_target_rate_hz: float = 10.0
    fdi_target_rate_hz_min: float = 5.0
    fdi_target_rate_hz_max: float = 20.0
    fdi_max_silent_frac: float = 0.35
    fdi_silent_rate_hz: float = 1.0
    fdi_max_overactive_frac: float = 0.05
    fdi_overactive_rate_hz: float = 50.0
    fdi_target_xi_min: float = 1.0
    fdi_target_xi_max: float = 3.0
    fdi_max_adapt_ratio: float = 0.35
    fdi_recurrent_to_input_ratio_min: float = 0.3
    fdi_recurrent_to_input_ratio_max: float = 1.5
    fdi_candidate_input_scales: List[float] = field(
        default_factory=lambda: [0.75, 1.0, 1.25, 1.5]
    )
    fdi_candidate_recurrent_scales: List[float] = field(
        default_factory=lambda: [0.75, 1.0, 1.25, 1.5]
    )
    fdi_candidate_threshold_scales: List[float] = field(
        default_factory=lambda: [0.75, 1.0, 1.25, 1.5]
    )
    fdi_strict_mode: bool = False
    theta_warmup_epochs: int = (
        0  # Phase 1 길이: theta 고정, w_raw/readout만 학습 (0=비활성화)
    )
    theta_warmup_dynamic: bool = False  # P1 metric plateau 시 P2로 조기 전환
    theta_warmup_strategy: str = "slope"  # slope | best
    theta_warmup_window: int = 3  # slope 전략에서 볼 최근 P1 epoch 수
    theta_warmup_min_epochs: int = 5  # dynamic warmup에서 최소 P1 epoch 수
    theta_warmup_patience: int = 3  # dynamic warmup plateau 허용 epoch 수
    theta_warmup_min_delta: float = 0.002  # P1 metric 개선으로 인정할 최소 변화량
    theta_warmup_metric: str = "test_acc"  # val_acc | test_acc | train_acc | train_loss
    theta_lr_scale: float = 0.1  # theta LR = base_lr × theta_lr_scale
    theta_lr_ramp_epochs: int = 1  # P2 topology LR ramp length (1 = no ramp)
    theta_bias_lr_scale: float = 1.0  # learned_lowrank bias LR relative to topology LR
    theta_freeze_epoch: int = (
        0  # learned mode에서 해당 epoch 시작 시 theta freeze (0 = 비활성화)
    )
    theta_adaptive_freeze: bool = False  # gradient-triggered adaptive theta freeze 활성화
    theta_freeze_min_epoch: int = 20  # adaptive freeze를 고려하기 시작하는 최소 epoch
    theta_freeze_grad_threshold: float = 30.0  # theta grad norm 임계값
    theta_freeze_patience: int = 2  # 연속으로 임계값 초과해야 하는 epoch 수
    topology_adaptive_freeze: bool = False  # validation-based adaptive topology freeze
    topology_freeze_metric: str = "val_acc"  # currently only val_acc is supported
    topology_freeze_min_epoch: int = 40  # validation-based freeze earliest epoch
    topology_freeze_patience: int = 8  # validation metric plateau patience
    topology_freeze_min_delta: float = 0.0  # minimum val_acc improvement for snapshot
    topology_freeze_rollback_best: bool = True  # rollback best topology before freezing
    topology_freeze_verbose: bool = True  # print one-time validation freeze event
    topology_runaway_guard_enabled: bool = False
    topology_runaway_grad_threshold: float = 50.0
    topology_runaway_firing_threshold: float = 0.9
    topology_runaway_patience: int = 2
    topology_runaway_freeze_epochs: int = 3
    noise_scale: float = (
        0.1  # 에폭 단위 Gumbel noise 크기 (0=결정적, 1=표준 Gumbel std≈1.81)
    )
    # 작은 값 → 경계(theta≈0) 엣지만 뒤집힘, 확실한 ON/OFF는 유지
    readout_mode: str = (
        "spike_count"  # spike_count | membrane_trace | spike_adaptation_concat | non_spiking_lif_final_mem
        # | motor_lif | motor_lif_count_membrane
    )
    readout_lif_beta: float = 0.95  # non-spiking LIF readout membrane decay
    readout_lif_learn_beta: bool = False  # learn non-spiking LIF readout decay
    readout_lif_normalize: bool = True  # divide final membrane by decay window
    readout_lif_bias_once: bool = True  # add class bias after temporal accumulation
    motor_beta: float = 0.9  # motor LIF readout membrane decay
    motor_threshold: float = 1.0  # motor LIF readout firing threshold
    motor_mem_clamp: float = 5.0  # motor LIF membrane clamp magnitude
    motor_logit_scale: float = 1.0  # scale raw motor spike counts before CE
    motor_membrane_logit_scale: float = 1.0  # scale motor membrane trace logits
    motor_final_bias: bool = True  # add class bias once to final motor logits
    pred_aux_enabled: bool = False  # next-state prediction auxiliary loss 활성화
    pred_trace_decay: float = 0.9  # filtered trace EMA decay (spike → trace)


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

    # diagnostics
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    # annealing
    tau_start: float = 1.0
    tau_end: float = 0.05
    tau_anneal_epochs: int = 25
    tau_hold_epochs: int = (
        0  # Phase 2 시작 후 tau=tau_start를 유지하는 epoch 수 (이후 annealing 시작)
    )

    # training
    epochs: int = 100
    patience: int = 20  # early stopping patience (0 = 비활성화)
    batch_size: int = 128
    val_fraction: float = 0.1
    val_seed: int = 42
    use_validation: bool = True
    selection_val_loss_tie_break: bool = False
    selection_tie_epsilon: float = 0.0
    selection_tie_break_later_if_loss_missing: bool = False
    checkpoint_top_k_val: int = 5
    lr: float = 1e-3
    lr_min: float = 1e-5  # cosine scheduler의 최솟값
    lambda_sparse: float = 0.005
    lambda_commit: float = 0.08
    lambda_pred: float = 0.0
    weight_decay: float = 0.0
    edge_threshold: float = 0.5
    seed: int = 42
    device: str = "auto"  # auto | cuda | mps | cpu

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


def _validate_config(cfg: Config) -> None:
    liq = cfg.liquid
    mode = str(liq.recurrent_mode)
    valid_modes = {
        "learned",
        "learned_lowrank",
        "learned_lowrank_grad_r",
        "learned_lowrank_frozen_w",
        "softplus_w_only",
        "edgewise_soft_conductance",
        "smooth_lowrank_conductance",
        "soft_gate_lowrank",
        "soft_gate_edgewise",
        "random_sparse",
        "fixed",
        "grad_r",
    }
    soft_gate_modes = {"soft_gate_lowrank", "soft_gate_edgewise"}
    if mode not in valid_modes:
        raise ValueError(
            "liquid.recurrent_mode must be one of "
            f"{sorted(valid_modes)}, got {mode!r}"
        )
    if liq.frozen_w_mode not in {"initialized_w", "constant_g"}:
        raise ValueError(
            "liquid.frozen_w_mode must be one of: initialized_w, constant_g; "
            f"got {liq.frozen_w_mode!r}"
        )
    if mode != "learned_lowrank_frozen_w":
        if liq.frozen_w_mode != "initialized_w":
            raise ValueError(
                "liquid.frozen_w_mode is only valid for "
                "recurrent_mode=learned_lowrank_frozen_w."
            )
        if liq.frozen_w_constant_g is not None:
            raise ValueError(
                "liquid.frozen_w_constant_g is only valid for "
                "recurrent_mode=learned_lowrank_frozen_w."
            )
    if liq.match_initial_w_eff_scale and mode != "smooth_lowrank_conductance":
        raise ValueError(
            "liquid.match_initial_w_eff_scale=true is only valid for "
            "recurrent_mode=smooth_lowrank_conductance."
        )
    if mode == "softplus_w_only" and not liq.train_w_raw:
        raise ValueError(
            "softplus_w_only uses softplus(w_raw) as its conductance. "
            "Set liquid.train_w_raw=true."
        )
    if (
        mode in {"edgewise_soft_conductance", "smooth_lowrank_conductance"}
        and liq.train_w_raw
    ):
        raise ValueError(
            f"{mode} does not use w_raw. Set liquid.train_w_raw=false."
        )
    if mode in soft_gate_modes:
        if liq.train_w_raw:
            raise ValueError(
                f"{mode} uses gate*mag conductance and does not use w_raw. "
                "Set liquid.train_w_raw=false."
            )
        if float(liq.noise_scale) != 0.0:
            raise ValueError(
                f"{mode} is deterministic soft-gate topology. "
                "Set liquid.noise_scale=0.0 to disable Gumbel/sampling noise."
            )
    elif float(liq.density_penalty_lambda) != 0.0:
        raise ValueError(
            "liquid.density_penalty_lambda is only valid for "
            "soft_gate_lowrank or soft_gate_edgewise."
        )
    if mode not in soft_gate_modes and bool(liq.mag_from_separate_param):
        raise ValueError(
            "liquid.mag_from_separate_param is only valid for soft_gate modes."
        )
    if mode == "learned_lowrank_frozen_w" and liq.train_w_raw:
        raise ValueError(
            "learned_lowrank_frozen_w requires liquid.train_w_raw=false."
        )
    if mode == "learned_lowrank_grad_r" and float(liq.noise_scale) != 0.0:
        raise ValueError(
            "learned_lowrank_grad_r is deterministic. Set liquid.noise_scale=0.0."
        )
    if liq.frozen_w_constant_g is not None and float(liq.frozen_w_constant_g) < 0.0:
        raise ValueError("liquid.frozen_w_constant_g must be non-negative.")
    if float(liq.temp_init) <= 0.0:
        raise ValueError("liquid.temp_init must be positive.")
    if float(liq.temp_final) <= 0.0:
        raise ValueError("liquid.temp_final must be positive.")
    for key in ("target_density_init", "target_density_final"):
        value = float(getattr(liq, key))
        if not 0.0 < value < 1.0:
            raise ValueError(f"liquid.{key} must be in (0, 1), got {value}")
    if int(liq.target_anneal_epochs) < 0:
        raise ValueError("liquid.target_anneal_epochs must be non-negative.")
    if float(liq.density_penalty_lambda) < 0.0:
        raise ValueError("liquid.density_penalty_lambda must be non-negative.")


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
    arch_d = data.pop("architecture", {})
    topo_d = data.pop("topology", {})
    liq_d = data.pop("liquid", {})
    diag_d = data.pop("diagnostics", {})

    cfg = Config(**data)
    cfg.architecture = ArchitectureConfig(**arch_d)
    cfg.topology = TopologyConfig(**topo_d)
    cfg.liquid = LiquidConfig(**liq_d)
    cfg.diagnostics = DiagnosticsConfig(**diag_d)
    _validate_config(cfg)
    return cfg
