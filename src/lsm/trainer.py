"""
LSM training loop.

Differences from feedforward trainer (src/training/trainer.py):
  - Gradient clipping (clip_grad_norm_) for BPTT stability
  - Extended logging: grad_norm, firing rates, theta stats
  - Early warning: grad explosion, neuron runaway
"""

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path
from functools import cmp_to_key
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.loaders import get_train_val_test_dataloaders
from src.lsm.diagnostics import (
    DiagnosticsLogger,
    collect_epoch_diagnostics,
    collect_topology_metrics,
)
from src.lsm.initialization.fdi_calibration import calibrate_fdi_style_initial_regime
from src.lsm.model import LSMModel
from src.utils.config import Config


ce_loss = nn.CrossEntropyLoss()


def _tqdm_disabled() -> bool:
    value = os.environ.get("DISABLE_TQDM") or os.environ.get("TQDM_DISABLE")
    return str(value).lower() in {"1", "true", "yes", "on"}


def get_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested device='cuda' but CUDA is not available. "
                "Check the NVIDIA driver and install a CUDA-enabled PyTorch wheel. "
                f"torch.version.cuda={torch.version.cuda!r}"
            )
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device='mps' but MPS is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, mps, cpu")


def _device_summary(device: torch.device) -> str:
    parts = [f"device={device}", f"torch={torch.__version__}"]
    if device.type == "cuda":
        parts.append(f"torch_cuda={torch.version.cuda}")
        parts.append(f"gpu={torch.cuda.get_device_name(device)}")
    return "  ".join(parts)


def get_tau(epoch: int, cfg: Config, warmup_epochs: int | None = None) -> float:
    # Tau annealing starts from Phase 2 (after warmup), with an optional hold period.
    # During Phase 1, tau is computed but unused (hard mask ignores it).
    warmup = cfg.liquid.theta_warmup_epochs if warmup_epochs is None else warmup_epochs
    hold = cfg.tau_hold_epochs
    phase2_epoch = max(epoch - warmup, 0)  # epochs elapsed since Phase 2 start
    anneal_epoch = max(phase2_epoch - hold, 0)  # epochs elapsed since annealing start
    if anneal_epoch >= cfg.tau_anneal_epochs:
        return cfg.tau_end
    progress = anneal_epoch / cfg.tau_anneal_epochs
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.tau_end + (cfg.tau_start - cfg.tau_end) * cosine


def get_soft_gate_schedule(
    epoch: int,
    cfg: Config,
    warmup_epochs: int | None = None,
) -> tuple[float, float]:
    """Return (temperature, target_density) for deterministic soft-gate modes."""

    liq = cfg.liquid
    warmup = liq.theta_warmup_epochs if warmup_epochs is None else warmup_epochs
    phase2_epoch = max(epoch - warmup, 0)
    anneal_epochs = max(int(liq.target_anneal_epochs), 0)
    if anneal_epochs == 0:
        progress = 1.0
    else:
        progress = min(1.0, phase2_epoch / anneal_epochs)
    temp = (
        float(liq.temp_init)
        + (float(liq.temp_final) - float(liq.temp_init)) * progress
    )
    target = float(liq.target_density_init) + (
        float(liq.target_density_final) - float(liq.target_density_init)
    ) * progress
    return temp, target


def _make_experiment_dir(cfg: Config) -> Path:
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    exp_dir = Path("experiments") / f"{cfg.experiment_name}_{timestamp}"
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)

    import yaml, dataclasses

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(dataclasses.asdict(cfg), f, default_flow_style=False, sort_keys=False)

    return exp_dir


def build_model(cfg: Config, device: torch.device) -> LSMModel:
    liq = cfg.liquid
    return LSMModel(
        n_input=cfg.n_input,
        n_liquid=liq.n_liquid,
        n_output=cfg.n_output,
        T=cfg.T,
        exc_ratio=liq.exc_ratio,
        neuron_type=liq.neuron_type,
        beta_min=liq.beta_min,
        beta_max=liq.beta_max,
        threshold_min=liq.threshold_min,
        threshold_max=liq.threshold_max,
        alif_rho_init=liq.alif_rho_init,
        alif_beta_init=liq.alif_beta_init,
        alif_adapt_increment=liq.alif_adapt_increment,
        alif_learn_rho=liq.alif_learn_rho,
        alif_learn_beta=liq.alif_learn_beta,
        p_input=liq.p_input,
        input_weight_scale=liq.input_weight_scale,
        input_projection_mode=liq.input_projection_mode,
        train_input_projection=liq.train_input_projection,
        recurrent_mode=liq.recurrent_mode,
        recurrent_sparsity=liq.recurrent_sparsity,
        self_connection=liq.self_connection,
        theta_init_mean=liq.theta_init_mean,
        theta_init_std=liq.theta_init_std,
        theta_rank=liq.theta_rank,
        theta_lowrank_init_std=liq.theta_lowrank_init_std,
        w_raw_init_mean=liq.w_raw_init_mean,
        w_raw_init_std=liq.w_raw_init_std,
        train_w_raw=liq.train_w_raw,
        w_raw_max=liq.w_raw_max,
        recurrent_weight_scale=liq.recurrent_weight_scale,
        match_initial_w_eff_scale=liq.match_initial_w_eff_scale,
        frozen_w_mode=liq.frozen_w_mode,
        frozen_w_constant_g=liq.frozen_w_constant_g,
        soft_gate_temp_init=liq.temp_init,
        soft_gate_target_density_init=liq.target_density_init,
        mag_from_separate_param=liq.mag_from_separate_param,
        bptt_truncate=liq.bptt_truncate,
        noise_scale=liq.noise_scale,
        readout_mode=liq.readout_mode,
        motor_beta=liq.motor_beta,
        motor_threshold=liq.motor_threshold,
        motor_mem_clamp=liq.motor_mem_clamp,
        motor_logit_scale=liq.motor_logit_scale,
        motor_membrane_logit_scale=liq.motor_membrane_logit_scale,
        motor_final_bias=liq.motor_final_bias,
        pred_aux_enabled=liq.pred_aux_enabled,
        pred_trace_decay=liq.pred_trace_decay,
        readout_lif_beta=liq.readout_lif_beta,
        readout_lif_learn_beta=liq.readout_lif_learn_beta,
        readout_lif_normalize=liq.readout_lif_normalize,
        readout_lif_bias_once=liq.readout_lif_bias_once,
    ).to(device)


def _compute_loss(rates, labels, model, cfg):
    loss = ce_loss(rates, labels)
    sp = model.sparsity_loss()
    cm = model.commitment_loss()
    pd = model.prediction_loss()
    return loss + cfg.lambda_sparse * sp + cfg.lambda_commit * cm + cfg.lambda_pred * pd


def _evaluate_metrics(
    model: LSMModel, loader, device: torch.device, tau: float
) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            rates = model(x, tau=tau)
            loss = ce_loss(rates, y)
            pred = rates.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)
    return correct / total, loss_sum / total


def _evaluate(model: LSMModel, loader, device: torch.device, tau: float) -> float:
    acc, _ = _evaluate_metrics(model, loader, device, tau)
    return acc


def _metric_improved(
    metric_name: str, value: float, best_value: float | None, min_delta: float
) -> bool:
    if best_value is None:
        return True
    if metric_name == "train_loss":
        return value < best_value - min_delta
    return value > best_value + min_delta


def _select_warmup_metric(metric_name: str, row: dict) -> float:
    if metric_name not in {"val_acc", "test_acc", "train_acc", "train_loss"}:
        raise ValueError(
            "liquid.theta_warmup_metric must be one of: "
            "val_acc, test_acc, train_acc, train_loss"
        )
    value = row.get(metric_name)
    if value is None:
        raise ValueError(
            f"liquid.theta_warmup_metric={metric_name!r} requires that metric to be available."
        )
    return value


def _warmup_score(metric_name: str, value: float) -> float:
    return -value if metric_name == "train_loss" else value


def _warmup_slope(scores: list[float], window: int) -> float | None:
    if len(scores) < window:
        return None
    recent = scores[-window:]
    return (recent[-1] - recent[0]) / max(window - 1, 1)


def _grad_norm(params: list[torch.nn.Parameter]) -> float:
    grad_sq = 0.0
    for param in params:
        if param.grad is not None:
            grad_sq += param.grad.norm().item() ** 2
    return grad_sq**0.5 if grad_sq > 0.0 else 0.0


def _count_trainable(params) -> int:
    return sum(param.numel() for param in params if param.requires_grad)


def _tensor_mean_std(values: torch.Tensor) -> tuple[float, float]:
    values = values.detach().float().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 0.0, 0.0
    std = values.std(unbiased=False).item() if values.numel() > 1 else 0.0
    return values.mean().item(), std


def _collect_recurrent_ablation_metrics(
    model: LSMModel,
    topology_params: list[nn.Parameter],
) -> dict[str, Any]:
    liquid = model.liquid
    out: dict[str, Any] = {
        "recurrent_mode": liquid.mode,
        "train_w_raw": bool(liquid.w_raw.requires_grad),
        "num_trainable_params_total": _count_trainable(model.parameters()),
        "num_trainable_params_recurrent": _count_trainable(liquid.parameters()),
        "num_trainable_params_topology": _count_trainable(topology_params),
        "num_trainable_params_w_raw": (
            liquid.w_raw.numel() if liquid.w_raw.requires_grad else 0
        ),
        "recurrent_weight_scale": _safe_float(
            getattr(liquid, "recurrent_weight_scale", None)
        ),
        "frozen_w_constant_g": _safe_float(
            getattr(liquid, "frozen_w_constant_g", None)
        ),
        "frozen_w_mode": getattr(liquid, "frozen_w_mode", ""),
        "mag_from_separate_param": bool(
            getattr(liquid, "mag_from_separate_param", False)
        ),
    }
    with torch.no_grad():
        w_eff = liquid.get_effective_weight().detach().float()
        valid = liquid.self_conn_mask.detach().bool()
        valid_w = w_eff[valid]
        out.update(
            {
                "w_eff_mean": _safe_float(valid_w.mean()) if valid_w.numel() else 0.0,
                "w_eff_std": (
                    _safe_float(valid_w.std(unbiased=False))
                    if valid_w.numel() > 1
                    else 0.0
                ),
                "w_eff_abs_mean": (
                    _safe_float(valid_w.abs().mean()) if valid_w.numel() else 0.0
                ),
                "w_eff_abs_max": (
                    _safe_float(valid_w.abs().max()) if valid_w.numel() else 0.0
                ),
                "w_eff_fro_norm": _safe_float(w_eff.norm()),
                "effective_density": (
                    _safe_float((valid_w.abs() > 1e-12).float().mean())
                    if valid_w.numel()
                    else 0.0
                ),
            }
        )
        if hasattr(liquid, "src_embed") and hasattr(liquid, "dst_embed"):
            theta = liquid.get_theta().detach().float()
            conductance = torch.nn.functional.softplus(theta)
            theta_valid = theta[valid]
            conductance_valid = conductance[valid]
            out.update(
                {
                    "role_src_norm_mean": _safe_float(
                        liquid.src_embed.detach().float().norm(dim=1).mean()
                    ),
                    "role_dst_norm_mean": _safe_float(
                        liquid.dst_embed.detach().float().norm(dim=1).mean()
                    ),
                    "lowrank_logit_mean": (
                        _safe_float(theta_valid.mean()) if theta_valid.numel() else 0.0
                    ),
                    "lowrank_logit_std": (
                        _safe_float(theta_valid.std(unbiased=False))
                        if theta_valid.numel() > 1
                        else 0.0
                    ),
                    "lowrank_conductance_mean": (
                        _safe_float(conductance_valid.mean())
                        if conductance_valid.numel()
                        else 0.0
                    ),
                    "lowrank_conductance_std": (
                        _safe_float(conductance_valid.std(unbiased=False))
                        if conductance_valid.numel() > 1
                        else 0.0
                    ),
                }
            )
        if liquid.mode in {"learned", "learned_lowrank", "learned_lowrank_frozen_w", "grad_r"}:
            if liquid.mode == "grad_r":
                theta = liquid.theta.detach().float()
                probs = torch.sigmoid(theta)
                mask = (theta > 0).float()
            else:
                theta = liquid.get_theta().detach().float()
                probs = torch.sigmoid(theta)
                mask = (probs >= 0.5).float()
            probs_valid = probs[valid]
            mask_valid = (mask * liquid.self_conn_mask)[valid]
            eps = 1e-6
            entropy = -(
                probs_valid * (probs_valid + eps).log()
                + (1.0 - probs_valid)
                * (1.0 - probs_valid + eps).log()
            )
            out.update(
                {
                    "mask_mean": _safe_float(mask_valid.mean()),
                    "mask_std": (
                        _safe_float(mask_valid.std(unbiased=False))
                        if mask_valid.numel() > 1
                        else 0.0
                    ),
                    "mask_entropy": _safe_float(entropy.mean())
                    if entropy.numel()
                    else 0.0,
                }
            )
        if liquid.mode == "edgewise_soft_conductance":
            theta = liquid.theta.detach().float()
            theta_valid = theta[valid]
            conductance = torch.nn.functional.softplus(theta)[valid]
            out.update(
                {
                    "edgewise_logit_mean": (
                        _safe_float(theta_valid.mean()) if theta_valid.numel() else 0.0
                    ),
                    "edgewise_logit_std": (
                        _safe_float(theta_valid.std(unbiased=False))
                        if theta_valid.numel() > 1
                        else 0.0
                    ),
                    "edgewise_conductance_mean": (
                        _safe_float(conductance.mean()) if conductance.numel() else 0.0
                    ),
                    "edgewise_conductance_std": (
                        _safe_float(conductance.std(unbiased=False))
                        if conductance.numel() > 1
                        else 0.0
                    ),
                }
            )
    return out


def _param_group_lr(optimizer: optim.Optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    return 0.0


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _row_epoch(row: dict[str, Any] | None) -> int | None:
    if row is None:
        return None
    value = row.get("epoch")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _validation_row_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    cfg: Config,
) -> tuple[bool, str]:
    """Return whether candidate wins under the shared validation comparator."""

    candidate_val = _safe_float(candidate.get("val_acc"))
    if candidate_val is None:
        return False, "missing_val_acc"
    if incumbent is None:
        return True, "first_validation_candidate"

    incumbent_val = _safe_float(incumbent.get("val_acc"))
    if incumbent_val is None:
        return True, "incumbent_missing_val_acc"

    eps = max(float(cfg.selection_tie_epsilon), 0.0)
    if candidate_val > incumbent_val + eps:
        return True, "higher_val_acc"
    if candidate_val < incumbent_val - eps:
        return False, "lower_val_acc"

    candidate_loss = _safe_float(candidate.get("val_loss"))
    incumbent_loss = _safe_float(incumbent.get("val_loss"))
    if (
        cfg.selection_val_loss_tie_break
        and candidate_loss is not None
        and incumbent_loss is not None
    ):
        if candidate_loss < incumbent_loss:
            return True, "val_acc_tie_lower_val_loss"
        if candidate_loss > incumbent_loss:
            return False, "val_acc_tie_higher_val_loss"
        reason = "val_acc_tie_equal_val_loss"
    else:
        reason = "val_acc_tie_missing_val_loss"

    if cfg.selection_tie_break_later_if_loss_missing:
        candidate_epoch = _row_epoch(candidate)
        incumbent_epoch = _row_epoch(incumbent)
        if (
            candidate_epoch is not None
            and incumbent_epoch is not None
            and candidate_epoch > incumbent_epoch
        ):
            return True, f"{reason}_later_epoch"

    return False, f"{reason}_keep_existing"


def _selection_row_is_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    cfg: Config,
    selection_metric_name: str,
) -> tuple[bool, str]:
    if selection_metric_name == "val_acc":
        return _validation_row_is_better(candidate, incumbent, cfg)
    candidate_value = _safe_float(candidate.get(selection_metric_name))
    if candidate_value is None:
        return False, f"missing_{selection_metric_name}"
    if incumbent is None:
        return True, f"first_{selection_metric_name}_candidate"
    incumbent_value = _safe_float(incumbent.get(selection_metric_name))
    if incumbent_value is None:
        return True, f"incumbent_missing_{selection_metric_name}"
    if candidate_value > incumbent_value:
        return True, f"higher_{selection_metric_name}"
    return False, f"{selection_metric_name}_not_improved"


def _rank_validation_rows(
    rows: list[dict[str, Any]],
    cfg: Config,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if _safe_float(row.get("val_acc")) is not None]

    def _compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        left_better, _ = _validation_row_is_better(left, right, cfg)
        right_better, _ = _validation_row_is_better(right, left, cfg)
        if left_better and not right_better:
            return -1
        if right_better and not left_better:
            return 1
        left_epoch = _row_epoch(left) or 10**12
        right_epoch = _row_epoch(right) or 10**12
        return (left_epoch > right_epoch) - (left_epoch < right_epoch)

    return sorted(candidates, key=cmp_to_key(_compare))


def _clone_optimizer_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_optimizer_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_value(item) for item in value)
    return value


def _topology_optimizer_param_groups(
    optimizer: optim.Optimizer,
) -> list[dict[str, Any]]:
    return [
        group
        for group in optimizer.param_groups
        if group.get("name") in {"topology", "topology_bias"}
    ]


def _snapshot_topology_optimizer_state(
    optimizer: optim.Optimizer,
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for group in _topology_optimizer_param_groups(optimizer):
        params = list(group.get("params", []))
        groups.append(
            {
                "name": group.get("name"),
                "lr": float(group.get("lr", 0.0)),
                "params": params,
                "state": {
                    param: _clone_optimizer_value(optimizer.state.get(param, {}))
                    for param in params
                    if param in optimizer.state
                },
            }
        )
    return {"groups": groups}


def _restore_topology_optimizer_state(
    optimizer: optim.Optimizer,
    snapshot: dict[str, Any] | None,
) -> None:
    if not snapshot:
        return
    for saved_group in snapshot.get("groups", []):
        group_name = saved_group.get("name")
        for group in optimizer.param_groups:
            if group.get("name") == group_name:
                group["lr"] = float(saved_group.get("lr", group["lr"]))
                break
        for param, state in saved_group.get("state", {}).items():
            optimizer.state[param] = _clone_optimizer_value(state)


def _snapshot_topology_bundle(
    model: LSMModel,
    optimizer: optim.Optimizer,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": _row_epoch(row),
        "row": {
            "epoch": row.get("epoch"),
            "val_acc": row.get("val_acc"),
            "val_loss": row.get("val_loss"),
            "train_acc": row.get("train_acc"),
            "train_loss": row.get("train_loss"),
            "test_acc": row.get("test_acc"),
            "test_loss": row.get("test_loss"),
            "max_firing_rate": row.get("max_firing_rate"),
            "topology_grad_pre_clip": row.get("topology_grad_pre_clip"),
        },
        "topology_state": model.liquid.topology_state_dict(),
        "optimizer_state": _snapshot_topology_optimizer_state(optimizer),
    }


def _restore_topology_bundle(
    model: LSMModel,
    optimizer: optim.Optimizer,
    snapshot: dict[str, Any] | None,
) -> int | None:
    if snapshot is None:
        return None
    model.liquid.load_topology_state_dict(snapshot["topology_state"])
    _restore_topology_optimizer_state(optimizer, snapshot.get("optimizer_state"))
    return _row_epoch(snapshot.get("row"))


def _top_val_checkpoint_path(exp_dir: Path, epoch: int) -> Path:
    return exp_dir / "checkpoints" / f"top_val_epoch_{epoch:03d}.pt"


def _checkpoint_metadata(
    row: dict[str, Any],
    *,
    topo_frozen: bool,
    topology_rollback_target_epoch: int | None,
    checkpoint_kind: str,
    selection_tie_break_reason: str,
    checkpoint_in_top_k_val: bool,
) -> dict[str, Any]:
    return {
        "kind": checkpoint_kind,
        "epoch": row.get("epoch"),
        "train_loss": row.get("train_loss"),
        "train_acc": row.get("train_acc"),
        "val_loss": row.get("val_loss"),
        "val_acc": row.get("val_acc"),
        "test_loss": row.get("test_loss"),
        "test_acc": row.get("test_acc"),
        "topo_frozen": topo_frozen,
        "topology_rollback_target_epoch": topology_rollback_target_epoch,
        "theta_grad_norm_pre_clip": row.get("theta_grad_norm_pre_clip"),
        "theta_grad_norm_post_clip": row.get("theta_grad_norm_post_clip"),
        "topology_grad_norm_pre_clip": row.get("topology_grad_norm_pre_clip"),
        "topology_grad_norm_post_clip": row.get("topology_grad_norm_post_clip"),
        "max_firing_rate": row.get("max_firing_rate"),
        "selection_tie_break_reason": selection_tie_break_reason,
        "checkpoint_in_top_k_val": checkpoint_in_top_k_val,
        "topology_runaway_guard_active": row.get("topology_runaway_guard_active"),
        "topology_runaway_guard_triggered": row.get(
            "topology_runaway_guard_triggered"
        ),
        "topology_runaway_freeze_remaining": row.get(
            "topology_runaway_freeze_remaining"
        ),
        "topology_runaway_rollback_epoch": row.get("topology_runaway_rollback_epoch"),
    }


def _build_checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    row: dict[str, Any],
    history: list[dict[str, Any]],
    best_row: dict[str, Any] | None,
    best_metric_name: str,
    topology_freeze_enabled: bool,
    topology_freeze_reason: str | None,
    topology_frozen_epoch: int | None,
    topology_best_epoch: int | None,
    topology_best_metric_name: str,
    topology_best_metric_value: float | None,
    topology_rollback_applied_any: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    best_metric_value = (
        _safe_float(best_row.get(best_metric_name)) if best_row is not None else None
    )
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": row.get("epoch"),
        "best_acc": best_metric_value,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "best_val_acc": best_row.get("val_acc") if best_row is not None else None,
        "best_val_loss": best_row.get("val_loss") if best_row is not None else None,
        "best_test_loss": best_row.get("test_loss") if best_row is not None else None,
        "best_test_acc_at_best_val": (
            best_row.get("test_acc") if best_row is not None else None
        ),
        "best_epoch": _row_epoch(best_row),
        "topology_freeze_enabled": topology_freeze_enabled,
        "topology_freeze_reason": topology_freeze_reason,
        "topology_frozen_epoch": topology_frozen_epoch,
        "topology_best_epoch": topology_best_epoch,
        "topology_best_metric_name": topology_best_metric_name,
        "topology_best_metric_value": topology_best_metric_value,
        "topology_rollback_applied": topology_rollback_applied_any,
        "checkpoint_metadata": metadata,
        "history": history,
    }


def _save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    row: dict[str, Any],
    history: list[dict[str, Any]],
    best_row: dict[str, Any] | None,
    best_metric_name: str,
    topology_freeze_enabled: bool,
    topology_freeze_reason: str | None,
    topology_frozen_epoch: int | None,
    topology_best_epoch: int | None,
    topology_best_metric_name: str,
    topology_best_metric_value: float | None,
    topology_rollback_applied_any: bool,
    metadata: dict[str, Any],
) -> None:
    torch.save(
        _build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            row=row,
            history=history,
            best_row=best_row,
            best_metric_name=best_metric_name,
            topology_freeze_enabled=topology_freeze_enabled,
            topology_freeze_reason=topology_freeze_reason,
            topology_frozen_epoch=topology_frozen_epoch,
            topology_best_epoch=topology_best_epoch,
            topology_best_metric_name=topology_best_metric_name,
            topology_best_metric_value=topology_best_metric_value,
            topology_rollback_applied_any=topology_rollback_applied_any,
            metadata=metadata,
        ),
        path,
    )


def _prune_top_val_checkpoints(exp_dir: Path, keep_epochs: set[int]) -> None:
    pattern = re.compile(r"top_val_epoch_(\d+)\.pt$")
    for path in (exp_dir / "checkpoints").glob("top_val_epoch_*.pt"):
        match = pattern.match(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch not in keep_epochs:
            path.unlink()


def _topology_param_group(model: LSMModel, cfg: Config) -> list[nn.Parameter]:
    if cfg.liquid.recurrent_mode == "grad_r":
        return [model.liquid.theta]
    return list(model.liquid.topology_parameters())


def _build_optimizer_param_groups(
    model: LSMModel, cfg: Config
) -> tuple[list[dict], dict[str, list[nn.Parameter]]]:
    has_theta = cfg.liquid.recurrent_mode in {
        "learned",
        "learned_lowrank",
        "learned_lowrank_frozen_w",
        "edgewise_soft_conductance",
        "smooth_lowrank_conductance",
        "soft_gate_lowrank",
        "soft_gate_edgewise",
        "grad_r",
    }
    topology_params = _topology_param_group(model, cfg) if has_theta else []
    theta_bias_params: list[nn.Parameter] = []
    theta_main_params = topology_params
    if cfg.liquid.recurrent_mode in {
        "learned_lowrank",
        "learned_lowrank_frozen_w",
        "smooth_lowrank_conductance",
        "soft_gate_lowrank",
    }:
        theta_bias_params = [model.liquid.theta_bias]
        theta_main_params = [model.liquid.src_embed, model.liquid.dst_embed]

    input_projection_params = list(model.input_proj.parameters())
    trainable_input_projection_params = [
        param for param in input_projection_params if param.requires_grad
    ]
    excluded_param_ids = {id(param) for param in topology_params}
    excluded_param_ids.update(id(param) for param in input_projection_params)
    other_params = [
        param for param in model.parameters() if id(param) not in excluded_param_ids
    ]

    param_groups = [
        {
            "params": other_params,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "name": "other",
        }
    ]
    if theta_main_params:
        param_groups.append(
            {
                "params": theta_main_params,
                "lr": cfg.lr * cfg.liquid.theta_lr_scale,
                "weight_decay": 0.0,
                "name": "topology",
            }
        )
    if theta_bias_params:
        param_groups.append(
            {
                "params": theta_bias_params,
                "lr": (
                    cfg.lr
                    * cfg.liquid.theta_lr_scale
                    * cfg.liquid.theta_bias_lr_scale
                ),
                "weight_decay": 0.0,
                "name": "topology_bias",
            }
        )
    if trainable_input_projection_params:
        param_groups.append(
            {
                "params": trainable_input_projection_params,
                "lr": cfg.lr * cfg.liquid.input_proj_lr_scale,
                "weight_decay": 0.0,
                "name": "input_projection",
            }
        )

    metadata = {
        "other_params": other_params,
        "theta_params": topology_params,
        "theta_main_params": theta_main_params,
        "theta_bias_params": theta_bias_params,
        "input_projection_params": trainable_input_projection_params,
    }
    return param_groups, metadata


def _selection_state(val_loader) -> str:
    return "val_acc" if val_loader is not None else "test_acc"


def train(cfg: Config) -> tuple:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    tqdm.write(_device_summary(device))

    if cfg.liquid.init_mode not in {"manual", "fdi_calibrated"}:
        raise ValueError(
            "liquid.init_mode must be one of: manual, fdi_calibrated; "
            f"got {cfg.liquid.init_mode!r}"
        )

    exp_dir = _make_experiment_dir(cfg)
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    last_checkpoint_path = exp_dir / "checkpoints" / "last.pt"
    log_path = exp_dir / "logs" / "train.jsonl"
    diagnostics_logger = (
        DiagnosticsLogger(exp_dir, cfg) if cfg.diagnostics.enabled else None
    )

    train_loader, val_loader, test_loader = get_train_val_test_dataloaders(cfg)
    model = build_model(cfg, device)
    selection_metric_name = _selection_state(val_loader)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset) if val_loader is not None else 0
    test_size = len(test_loader.dataset)
    tqdm.write(
        "Data split: "
        f"train={train_size}  "
        + (
            f"val={val_size}  " if val_loader is not None else "val=disabled  "
        )
        + f"test={test_size}"
    )
    fdi_report = None
    if cfg.liquid.init_mode == "fdi_calibrated":
        fdi_report = calibrate_fdi_style_initial_regime(
            model=model,
            train_loader=train_loader,
            config=cfg,
            device=device,
            output_dir=exp_dir,
        )

    def _set_topology_requires_grad(requires_grad: bool) -> None:
        if cfg.liquid.recurrent_mode == "grad_r":
            model.liquid.theta.requires_grad_(requires_grad)
        else:
            model.liquid.set_topology_requires_grad(requires_grad)

    # Phase 1 warmup is only for hard Gumbel topology modes. Soft conductance
    # modes learn from epoch 1 because there is no sampled mask to stabilize.
    warmup = cfg.liquid.theta_warmup_epochs
    hard_topology_modes = {"learned", "learned_lowrank", "learned_lowrank_frozen_w"}
    soft_gate_modes = {"soft_gate_lowrank", "soft_gate_edgewise"}
    lowrank_param_modes = {
        "learned_lowrank",
        "learned_lowrank_frozen_w",
        "smooth_lowrank_conductance",
        "soft_gate_lowrank",
    }
    topology_param_modes = {
        "learned",
        "learned_lowrank",
        "learned_lowrank_frozen_w",
        "edgewise_soft_conductance",
        "smooth_lowrank_conductance",
        "soft_gate_lowrank",
        "soft_gate_edgewise",
        "grad_r",
    }
    is_learned = cfg.liquid.recurrent_mode in hard_topology_modes
    is_soft_gate = cfg.liquid.recurrent_mode in soft_gate_modes
    uses_topology_warmup = is_learned or is_soft_gate
    has_theta = cfg.liquid.recurrent_mode in topology_param_modes
    topology_trainable_modes = topology_param_modes
    topology_freeze_enabled = has_theta and cfg.liquid.topology_adaptive_freeze
    dynamic_warmup = uses_topology_warmup and cfg.liquid.theta_warmup_dynamic and warmup > 0
    warmup_min_epochs = min(max(cfg.liquid.theta_warmup_min_epochs, 1), warmup)
    warmup_patience = max(cfg.liquid.theta_warmup_patience, 1)
    warmup_min_delta = max(cfg.liquid.theta_warmup_min_delta, 0.0)
    warmup_metric = cfg.liquid.theta_warmup_metric
    warmup_strategy = cfg.liquid.theta_warmup_strategy
    warmup_window = max(cfg.liquid.theta_warmup_window, 2)
    theta_freeze_epoch = max(cfg.liquid.theta_freeze_epoch, 0)
    warmup_metric_best: float | None = None
    warmup_slow_count = 0
    warmup_scores: list[float] = []
    if warmup_strategy not in {"slope", "best"}:
        raise ValueError("liquid.theta_warmup_strategy must be one of: slope, best")
    if topology_freeze_enabled and cfg.liquid.topology_freeze_metric != "val_acc":
        raise ValueError(
            "liquid.topology_freeze_metric must be 'val_acc'. "
            "Test accuracy is reporting-only and cannot drive topology freeze."
        )
    if topology_freeze_enabled and val_loader is None:
        raise ValueError(
            "liquid.topology_adaptive_freeze=true requires an internal validation split."
        )
    if (
        cfg.liquid.topology_runaway_guard_enabled
        and cfg.liquid.recurrent_mode in lowrank_param_modes
        and val_loader is None
    ):
        raise ValueError(
            "liquid.topology_runaway_guard_enabled=true requires an internal validation split."
        )
    if warmup_metric == "val_acc" and val_loader is None:
        warmup_metric = "test_acc"
        tqdm.write(
            "  Validation disabled; falling back to test_acc for theta warmup metric"
        )
    if uses_topology_warmup and warmup > 0:
        _set_topology_requires_grad(False)
        tqdm.write(f"  Phase 1 warmup: theta frozen for {warmup} epochs")
        if dynamic_warmup:
            tqdm.write(
                "  Dynamic warmup enabled: "
                f"min={warmup_min_epochs}, max={warmup}, "
                f"strategy={warmup_strategy}, window={warmup_window}, "
                f"patience={warmup_patience}, metric={warmup_metric}, "
                f"min_delta={warmup_min_delta:g}"
            )

    def _current_topology_lr_scale(epoch_idx: int) -> float:
        if not has_theta:
            return 0.0
        if uses_topology_warmup and epoch_idx < warmup:
            return 0.0
        if not uses_topology_warmup:
            return cfg.liquid.theta_lr_scale
        ramp_epochs = max(cfg.liquid.theta_lr_ramp_epochs, 1)
        p2_epoch = epoch_idx - warmup
        ramp = min(1.0, (p2_epoch + 1) / ramp_epochs)
        return cfg.liquid.theta_lr_scale * ramp

    # Separate optimizer groups so topology and input projection gradients cannot
    # suppress w_raw/readout updates. Independent clipping is applied before step().
    param_groups, optimizer_metadata = _build_optimizer_param_groups(model, cfg)
    other_params = optimizer_metadata["other_params"]
    theta_params = optimizer_metadata["theta_params"]
    input_projection_params = optimizer_metadata["input_projection_params"]
    optimizer = optim.Adam(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    best_metric_name = selection_metric_name
    best_val_acc: float | None = None
    best_val_loss: float | None = None
    best_test_acc_at_best_val: float | None = None
    best_epoch = 0
    best_checkpoint_row: dict[str, Any] | None = None
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []
    clip_norm_w = cfg.liquid.grad_clip_max_norm_w
    clip_norm_theta = cfg.liquid.grad_clip_max_norm_theta
    clip_norm_input_projection = cfg.liquid.input_proj_grad_clip

    adaptive_freeze_bad_epochs = 0
    theta_adaptive_frozen = False
    topology_frozen_by_validation = False
    topology_freeze_reason: str | None = None
    topology_frozen_epoch: int | None = None
    best_topology_metric = float("-inf")
    best_topology_epoch: int | None = None
    best_topology_snapshot: dict[str, Any] | None = None
    latest_stable_topology_snapshot: dict[str, Any] | None = None
    topology_bad_count = 0
    topology_best_metric_name = cfg.liquid.topology_freeze_metric
    topology_rollback_applied_any = False
    topology_rollback_target_epoch: int | None = None
    topology_runaway_bad_epochs = 0
    topology_runaway_freeze_remaining = 0
    previous_topology_logit: torch.Tensor | None = None
    previous_topology_sigmoid: torch.Tensor | None = None
    previous_topology_logit_mean: float | None = None
    previous_topology_logit_std: float | None = None
    previous_sparsity: float | None = None

    def _topology_is_frozen(epoch_idx: int) -> bool:
        fixed_epoch_frozen = topology_freeze_reason == "fixed_epoch" or (
            has_theta and theta_freeze_epoch > 0 and epoch_idx + 1 > theta_freeze_epoch
        )
        emergency_frozen = topology_runaway_freeze_remaining > 0
        return (
            fixed_epoch_frozen
            or theta_adaptive_frozen
            or topology_frozen_by_validation
            or emergency_frozen
        )

    def _freeze_topology(reason: str, epoch_idx: int) -> None:
        nonlocal theta_adaptive_frozen
        nonlocal topology_frozen_by_validation
        nonlocal topology_freeze_reason
        nonlocal topology_frozen_epoch

        model.liquid.freeze_topology()
        if reason == "gradient_threshold":
            theta_adaptive_frozen = True
        elif reason == "validation_adaptive":
            topology_frozen_by_validation = True
        topology_freeze_reason = reason
        topology_frozen_epoch = epoch_idx + 1

    disable_tqdm = _tqdm_disabled()
    epoch_bar = tqdm(range(cfg.epochs), desc="Epochs", unit="ep", disable=disable_tqdm)
    alif_batch_debug_logged = False
    last_epoch_completed = 0

    with open(log_path, "a") as log_f:
        for epoch in epoch_bar:
            current_lr = optimizer.param_groups[0]["lr"]
            topology_runaway_frozen_at_epoch_start = (
                topology_runaway_freeze_remaining > 0
            )
            topology_lr_scale = (
                0.0
                if topology_runaway_frozen_at_epoch_start
                else _current_topology_lr_scale(epoch)
            )
            for group in optimizer.param_groups:
                if group.get("name") == "topology":
                    group["lr"] = current_lr * topology_lr_scale
                elif group.get("name") == "topology_bias":
                    group["lr"] = (
                        current_lr
                        * topology_lr_scale
                        * cfg.liquid.theta_bias_lr_scale
                    )
                elif group.get("name") == "input_projection":
                    group["lr"] = current_lr * cfg.liquid.input_proj_lr_scale

            # Phase transition: unfreeze theta at warmup boundary
            if (
                uses_topology_warmup
                and warmup > 0
                and epoch == warmup
                and not _topology_is_frozen(epoch)
            ):
                _set_topology_requires_grad(True)
                tqdm.write(
                    f"  Phase 2: theta unfrozen at epoch {epoch+1}, topology learning begins"
                )
            if topology_runaway_frozen_at_epoch_start and not (
                theta_adaptive_frozen or topology_frozen_by_validation
            ):
                _set_topology_requires_grad(False)
            if (
                has_theta
                and theta_freeze_epoch > 0
                and epoch + 1 == theta_freeze_epoch
                and not _topology_is_frozen(epoch)
            ):
                _freeze_topology("fixed_epoch", epoch)
                tqdm.write(
                    f"  Theta frozen at epoch {epoch+1}; topology fixed deterministically"
                )
            topology_is_frozen = _topology_is_frozen(epoch)

            tau = get_tau(epoch, cfg, warmup_epochs=warmup) if is_learned else 1.0
            soft_gate_temp, soft_gate_target_density = (
                get_soft_gate_schedule(epoch, cfg, warmup_epochs=warmup)
                if is_soft_gate
                else (None, None)
            )
            if is_soft_gate:
                model.liquid.set_soft_gate_temperature(float(soft_gate_temp))
            phase_label = "P1" if (uses_topology_warmup and epoch < warmup) else "P2"
            model.train()

            # Phase 2: sample Gumbel noise ONCE per epoch, lock mask for all batches.
            # This keeps topology stable within an epoch (BPTT safe) while allowing
            # exploration across epochs (OFF edges get a chance to be ON → w_raw learns).
            if is_learned and epoch >= warmup and not topology_is_frozen:
                eps = torch.rand_like(model.liquid.get_theta()).clamp(1e-6, 1 - 1e-6)
                epoch_noise = (torch.log(eps) - torch.log(1.0 - eps)).to(device)
                model.liquid.sample_epoch_mask(tau=tau, epoch_noise=epoch_noise)

            total_l = correct = n = 0
            epoch_grad_norm = 0.0
            n_batches = 0

            epoch_topology_grad_pre = 0.0
            epoch_topology_grad_post = 0.0
            epoch_w_raw_grad_pre = 0.0
            epoch_src_grad_pre = 0.0
            epoch_dst_grad_pre = 0.0
            epoch_bias_grad_pre = 0.0
            epoch_src_grad_post = 0.0
            epoch_dst_grad_post = 0.0
            epoch_bias_grad_post = 0.0
            epoch_input_proj_grad_pre = 0.0
            epoch_density_penalty = 0.0
            topology_clip_violations = 0
            diagnostic_mean_keys = (
                "mean_spike_rate",
                "adaptation_mean",
                "membrane_mean",
                "input_current_abs_mean",
                "recurrent_current_abs_mean",
            )
            diagnostic_max_keys = (
                "max_spike_rate",
                "adaptation_max",
                "membrane_max",
                "input_current_abs_max",
                "recurrent_current_abs_max",
            )
            epoch_diagnostic_weight = 0
            epoch_diagnostic_sums = {key: 0.0 for key in diagnostic_mean_keys}
            epoch_diagnostic_max: dict[str, float | None] = {
                key: None for key in diagnostic_max_keys
            }

            batch_bar = tqdm(
                train_loader,
                desc="  Train",
                leave=False,
                unit="batch",
                disable=disable_tqdm,
            )
            for x, y in batch_bar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                rates, batch_diagnostics = model(x, tau=tau, return_diagnostics=True)
                loss = _compute_loss(rates, y, model, cfg)
                if is_soft_gate:
                    density_penalty = model.liquid.soft_gate_density_penalty(
                        float(soft_gate_target_density)
                    )
                    loss = (
                        loss
                        + float(cfg.liquid.density_penalty_lambda) * density_penalty
                    )
                    epoch_density_penalty += (
                        float(density_penalty.detach().cpu().item()) * y.size(0)
                    )
                # NaN detection
                if torch.isnan(loss):
                    tqdm.write(
                        f"  ✖ NaN loss detected at epoch {epoch+1}, batch {n_batches+1}. Stopping."
                    )
                    return history, exp_dir

                loss.backward()

                if cfg.liquid.neuron_type == "alif" and not alif_batch_debug_logged:
                    adapt_info = model.adaptation_info()
                    tqdm.write(
                        "  [ALIF] first batch adaptation: "
                        f"mean={adapt_info['mean']:.4f}  max={adapt_info['max']:.4f}"
                    )
                    alif_batch_debug_logged = True

                # per-component grad norms (before clipping)
                w_raw_g = model.liquid.w_raw.grad
                topology_grad_pre = _grad_norm(theta_params)
                if topology_grad_pre > 0.0:
                    epoch_topology_grad_pre += topology_grad_pre
                if w_raw_g is not None:
                    epoch_w_raw_grad_pre += w_raw_g.norm().item()
                if cfg.liquid.recurrent_mode in lowrank_param_modes:
                    epoch_src_grad_pre += _grad_norm([model.liquid.src_embed])
                    epoch_dst_grad_pre += _grad_norm([model.liquid.dst_embed])
                    epoch_bias_grad_pre += _grad_norm([model.liquid.theta_bias])
                input_proj_grad_pre = _grad_norm(input_projection_params)
                if input_proj_grad_pre > 0.0:
                    epoch_input_proj_grad_pre += input_proj_grad_pre

                # Independent per-group clipping:
                #   topology: clip_norm_theta (moderate — prevents runaway while allowing
                #          Adam to normalize; smaller than clip_norm_w to enforce time-
                #          scale separation: topology changes slowly, weights adapt fast)
                #   other: clip_norm_w (large enough for recurrent BPTT norms ~10^2–10^4)
                #   input_projection: optional dedicated clip, isolated from other groups
                if has_theta and theta_params:
                    torch.nn.utils.clip_grad_norm_(
                        theta_params, max_norm=clip_norm_theta
                    )
                    topology_grad_post = _grad_norm(theta_params)
                    if topology_grad_post > 0.0:
                        epoch_topology_grad_post += topology_grad_post
                    if topology_grad_post > clip_norm_theta + 1e-3:
                        topology_clip_violations += 1
                    if cfg.liquid.recurrent_mode in lowrank_param_modes:
                        epoch_src_grad_post += _grad_norm([model.liquid.src_embed])
                        epoch_dst_grad_post += _grad_norm([model.liquid.dst_embed])
                        epoch_bias_grad_post += _grad_norm([model.liquid.theta_bias])
                    other_norm = torch.nn.utils.clip_grad_norm_(
                        other_params, max_norm=clip_norm_w
                    )
                    grad_norm = other_norm
                else:
                    topology_grad_post = 0.0
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        other_params, max_norm=clip_norm_w
                    )
                if input_projection_params and clip_norm_input_projection is not None:
                    torch.nn.utils.clip_grad_norm_(
                        input_projection_params,
                        max_norm=float(clip_norm_input_projection),
                    )
                optimizer.step()

                total_l += loss.item() * y.size(0)
                correct += (rates.argmax(1) == y).sum().item()
                n += y.size(0)
                epoch_grad_norm += grad_norm.item()
                n_batches += 1
                batch_weight = int(y.size(0))
                epoch_diagnostic_weight += batch_weight
                for key in diagnostic_mean_keys:
                    value = _safe_float(batch_diagnostics.get(key))
                    if value is not None:
                        epoch_diagnostic_sums[key] += value * batch_weight
                for key in diagnostic_max_keys:
                    value = _safe_float(batch_diagnostics.get(key))
                    if value is not None:
                        current = epoch_diagnostic_max[key]
                        epoch_diagnostic_max[key] = (
                            value if current is None else max(current, value)
                        )

                batch_bar.set_postfix(loss=f"{total_l/n:.4f}", acc=f"{correct/n:.4f}")

            train_acc = correct / n
            train_loss = total_l / n
            avg_grad_norm = epoch_grad_norm / max(n_batches, 1)
            avg_topology_grad_pre = epoch_topology_grad_pre / max(n_batches, 1)
            avg_topology_grad_post = epoch_topology_grad_post / max(n_batches, 1)
            avg_w_raw_grad = epoch_w_raw_grad_pre / max(n_batches, 1)
            avg_src_grad = epoch_src_grad_pre / max(n_batches, 1)
            avg_dst_grad = epoch_dst_grad_pre / max(n_batches, 1)
            avg_bias_grad = epoch_bias_grad_pre / max(n_batches, 1)
            avg_src_grad_post = epoch_src_grad_post / max(n_batches, 1)
            avg_dst_grad_post = epoch_dst_grad_post / max(n_batches, 1)
            avg_bias_grad_post = epoch_bias_grad_post / max(n_batches, 1)
            avg_input_proj_grad = epoch_input_proj_grad_pre / max(n_batches, 1)
            avg_density_penalty = epoch_density_penalty / max(n, 1)
            topology_clip_violation = topology_clip_violations > 0
            diagnostic_weight = max(epoch_diagnostic_weight, 1)

            def _diagnostic_mean(key: str) -> float:
                return epoch_diagnostic_sums.get(key, 0.0) / diagnostic_weight

            def _diagnostic_max(key: str) -> float:
                value = epoch_diagnostic_max.get(key)
                return 0.0 if value is None else value

            train_input_abs_mean = _diagnostic_mean("input_current_abs_mean")
            train_recurrent_abs_mean = _diagnostic_mean("recurrent_current_abs_mean")
            train_activity_info = {
                "mean_firing_rate": _diagnostic_mean("mean_spike_rate"),
                "max_firing_rate": _diagnostic_max("max_spike_rate"),
                "mean_adaptation": _diagnostic_mean("adaptation_mean"),
                "max_adaptation": _diagnostic_max("adaptation_max"),
                "adaptation_mean": _diagnostic_mean("adaptation_mean"),
                "adaptation_max": _diagnostic_max("adaptation_max"),
                "membrane_mean": _diagnostic_mean("membrane_mean"),
                "membrane_max": _diagnostic_max("membrane_max"),
                "input_current_abs_mean": train_input_abs_mean,
                "input_current_abs_max": _diagnostic_max("input_current_abs_max"),
                "recurrent_current_abs_mean": train_recurrent_abs_mean,
                "recurrent_current_abs_max": _diagnostic_max(
                    "recurrent_current_abs_max"
                ),
                "rec_input_abs_ratio": train_recurrent_abs_mean
                / max(train_input_abs_mean, 1e-12),
            }

            # Unlock epoch mask before eval so eval uses fresh deterministic mask
            model.liquid.unlock_epoch_mask()
            if val_loader is not None:
                val_acc, val_loss = _evaluate_metrics(model, val_loader, device, tau)
            else:
                val_acc = None
                val_loss = None
            test_acc, test_loss = _evaluate_metrics(model, test_loader, device, tau)
            selection_acc = val_acc if val_loader is not None else test_acc
            sparsity = model.sparsity_info()
            fr_info = {
                "mean": train_activity_info["mean_firing_rate"],
                "max": train_activity_info["max_firing_rate"],
            }
            adapt_info = {
                "mean": train_activity_info["mean_adaptation"],
                "max": train_activity_info["max_adaptation"],
            }
            motor_info = model.motor_info()
            readout_lif_info = model.readout_lif_info()
            topology_rollback_applied_epoch = False
            topology_rollback_reason = ""
            topology_runaway_guard_triggered = False
            topology_runaway_rollback_epoch = None
            topology_grad_exceeded_threshold = bool(
                has_theta
                and avg_topology_grad_pre
                > float(cfg.liquid.topology_runaway_grad_threshold)
            )
            max_firing_rate_exceeded_threshold = bool(
                fr_info["max"] > float(cfg.liquid.topology_runaway_firing_threshold)
            )
            topology_runaway_condition = (
                topology_grad_exceeded_threshold
                and max_firing_rate_exceeded_threshold
            )
            guard_enabled = (
                cfg.liquid.topology_runaway_guard_enabled
                and cfg.liquid.recurrent_mode in lowrank_param_modes
                and has_theta
            )

            validation_candidate_row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "topology_grad_pre_clip": avg_topology_grad_pre,
                "max_firing_rate": fr_info["max"],
            }

            if (
                (topology_freeze_enabled or guard_enabled)
                and cfg.liquid.recurrent_mode in topology_trainable_modes
                and not topology_is_frozen
            ):
                if val_acc is None:
                    raise RuntimeError(
                        "Validation accuracy is required for topology_adaptive_freeze."
                    )

                improved_snapshot = False
                if not topology_runaway_condition:
                    snapshot = _snapshot_topology_bundle(
                        model, optimizer, validation_candidate_row
                    )
                    latest_stable_topology_snapshot = snapshot
                    incumbent_row = (
                        best_topology_snapshot.get("row")
                        if best_topology_snapshot is not None
                        else None
                    )
                    improved_snapshot, _ = _validation_row_is_better(
                        validation_candidate_row, incumbent_row, cfg
                    )
                    if improved_snapshot:
                        best_topology_snapshot = snapshot
                        best_topology_metric = float(val_acc)
                        best_topology_epoch = epoch + 1
                        topology_rollback_target_epoch = best_topology_epoch
                        topology_bad_count = 0
                if not improved_snapshot:
                    topology_bad_count += 1

            permanent_topology_frozen = theta_adaptive_frozen or topology_frozen_by_validation
            if (
                guard_enabled
                and not permanent_topology_frozen
                and not topology_runaway_frozen_at_epoch_start
                and not topology_is_frozen
            ):
                if topology_runaway_condition:
                    topology_runaway_bad_epochs += 1
                else:
                    topology_runaway_bad_epochs = 0

                if topology_runaway_bad_epochs >= max(
                    int(cfg.liquid.topology_runaway_patience), 1
                ):
                    rollback_snapshot = (
                        best_topology_snapshot or latest_stable_topology_snapshot
                    )
                    topology_runaway_rollback_epoch = _restore_topology_bundle(
                        model, optimizer, rollback_snapshot
                    )
                    topology_rollback_target_epoch = topology_runaway_rollback_epoch
                    topology_rollback_applied_epoch = rollback_snapshot is not None
                    topology_rollback_applied_any = (
                        topology_rollback_applied_any
                        or topology_rollback_applied_epoch
                    )
                    topology_rollback_reason = "runaway_guard"
                    topology_runaway_guard_triggered = True
                    topology_runaway_bad_epochs = 0
                    topology_runaway_freeze_remaining = max(
                        int(cfg.liquid.topology_runaway_freeze_epochs), 1
                    )
                    _set_topology_requires_grad(False)
                    topology_is_frozen = True
                    tqdm.write(
                        "[TopologyRunawayGuard] triggered at epoch "
                        f"{epoch + 1}: topo_grad={avg_topology_grad_pre:.3g}, "
                        f"max_firing_rate={fr_info['max']:.3f}. "
                        + (
                            f"Rolled back to epoch {topology_runaway_rollback_epoch}. "
                            if topology_runaway_rollback_epoch is not None
                            else "No stable rollback snapshot was available. "
                        )
                        + f"Temporarily freezing topology for {topology_runaway_freeze_remaining} epochs."
                    )
            elif not topology_runaway_frozen_at_epoch_start:
                topology_runaway_bad_epochs = 0

            # Adaptive theta freeze runs after the emergency guard. A guard event
            # consumes this epoch and prevents immediate permanent shutdown.
            if (
                has_theta
                and cfg.liquid.theta_adaptive_freeze
                and not theta_adaptive_frozen
                and not topology_is_frozen
                and not topology_runaway_guard_triggered
                and not topology_runaway_frozen_at_epoch_start
                and epoch + 1 >= cfg.liquid.theta_freeze_min_epoch
            ):
                if avg_topology_grad_pre > cfg.liquid.theta_freeze_grad_threshold:
                    adaptive_freeze_bad_epochs += 1
                else:
                    adaptive_freeze_bad_epochs = 0

                if adaptive_freeze_bad_epochs >= cfg.liquid.theta_freeze_patience:
                    _freeze_topology("gradient_threshold", epoch)
                    topology_is_frozen = True
                    tqdm.write(
                        f"  Freezing theta at epoch {epoch + 1}: "
                        f"adaptive_grad>{cfg.liquid.theta_freeze_grad_threshold} "
                        f"for {cfg.liquid.theta_freeze_patience} epochs"
                    )

            if (
                topology_freeze_enabled
                and cfg.liquid.recurrent_mode in topology_trainable_modes
                and not topology_is_frozen
            ):
                can_freeze_topology = (
                    epoch + 1 >= cfg.liquid.topology_freeze_min_epoch
                    and topology_bad_count >= cfg.liquid.topology_freeze_patience
                    and not _topology_is_frozen(epoch)
                )
                if can_freeze_topology:
                    if cfg.liquid.topology_freeze_rollback_best:
                        if (
                            best_topology_snapshot is None
                            or best_topology_epoch is None
                        ):
                            raise RuntimeError(
                                "No topology snapshot available for rollback."
                            )
                        _restore_topology_bundle(
                            model, optimizer, best_topology_snapshot
                        )
                        topology_rollback_target_epoch = best_topology_epoch
                        topology_rollback_applied_epoch = True
                        topology_rollback_applied_any = True
                        topology_rollback_reason = "validation_adaptive"

                    _freeze_topology("validation_adaptive", epoch)
                    topology_is_frozen = True
                    if cfg.liquid.topology_freeze_verbose:
                        rollback_msg = ""
                        if topology_rollback_applied_epoch:
                            rollback_msg = (
                                f" Rolling back topology to epoch {best_topology_epoch} "
                                f"with val_acc={best_topology_metric:.4f}."
                            )
                        tqdm.write(
                            f"[TopologyFreeze] validation_adaptive triggered at epoch {epoch + 1}."
                            f"{rollback_msg} Freezing topology parameters."
                        )

            scheduler.step()

            # effective topology logit stats
            with torch.no_grad():
                if has_theta:
                    topology_logit = (
                        model.liquid.theta
                        if cfg.liquid.recurrent_mode == "grad_r"
                        else model.liquid.get_theta()
                    )
                    topology_logit_mean = topology_logit.mean().item()
                    topology_logit_std = topology_logit.std().item()
                    topology_sigmoid = torch.sigmoid(topology_logit)
                    topology_sigmoid_mean = topology_sigmoid.mean().item()
                    topology_sigmoid_std = topology_sigmoid.std().item()
                    topology_logit_p05 = torch.quantile(
                        topology_logit.reshape(-1), 0.05
                    ).item()
                    topology_logit_p95 = torch.quantile(
                        topology_logit.reshape(-1), 0.95
                    ).item()
                    topology_logit_flat = topology_logit.detach().reshape(-1).cpu()
                    topology_sigmoid_flat = topology_sigmoid.detach().reshape(-1).cpu()
                    eps_entropy = 1e-6
                    topology_entropy = (
                        -(
                            topology_sigmoid_flat
                            * (topology_sigmoid_flat + eps_entropy).log()
                            + (1 - topology_sigmoid_flat)
                            * (1 - topology_sigmoid_flat + eps_entropy).log()
                        )
                        .mean()
                        .item()
                    )
                    topology_edge_prob_low_frac = (
                        (topology_sigmoid_flat < 0.1).float().mean().item()
                    )
                    topology_edge_prob_high_frac = (
                        (topology_sigmoid_flat > 0.9).float().mean().item()
                    )
                    if previous_topology_logit is not None:
                        logit_delta = topology_logit_flat - previous_topology_logit
                        sigmoid_delta = (
                            topology_sigmoid_flat - previous_topology_sigmoid
                        )
                        topology_logit_delta_l2 = logit_delta.norm().item()
                        topology_logit_delta_mean_abs = logit_delta.abs().mean().item()
                        topology_sigmoid_delta_mean_abs = (
                            sigmoid_delta.abs().mean().item()
                        )
                    else:
                        topology_logit_delta_l2 = None
                        topology_logit_delta_mean_abs = None
                        topology_sigmoid_delta_mean_abs = None
                    topology_logit_mean_delta = (
                        topology_logit_mean - previous_topology_logit_mean
                        if previous_topology_logit_mean is not None
                        else None
                    )
                    topology_logit_std_delta = (
                        topology_logit_std - previous_topology_logit_std
                        if previous_topology_logit_std is not None
                        else None
                    )
                else:
                    topology_logit_mean = 0.0
                    topology_logit_std = 0.0
                    topology_sigmoid_mean = 0.0
                    topology_sigmoid_std = 0.0
                    topology_logit_p05 = 0.0
                    topology_logit_p95 = 0.0
                    topology_entropy = 0.0
                    topology_edge_prob_low_frac = 0.0
                    topology_edge_prob_high_frac = 0.0
                    topology_logit_flat = None
                    topology_sigmoid_flat = None
                    topology_logit_delta_l2 = None
                    topology_logit_delta_mean_abs = None
                    topology_sigmoid_delta_mean_abs = None
                    topology_logit_mean_delta = None
                    topology_logit_std_delta = None

            topology_metrics = collect_topology_metrics(model)
            recurrent_metrics = _collect_recurrent_ablation_metrics(
                model,
                theta_params,
            )
            soft_gate_metrics = (
                model.liquid.soft_gate_stats(
                    target_density=float(soft_gate_target_density)
                )
                if is_soft_gate
                else {}
            )
            if is_soft_gate:
                soft_gate_metrics["density_penalty_train_mean"] = avg_density_penalty

            topology_lr = _param_group_lr(optimizer, "topology")
            theta_bias_lr = _param_group_lr(optimizer, "topology_bias")
            input_proj_lr = _param_group_lr(optimizer, "input_projection")
            fdi_selected_candidate = (
                fdi_report.get("selected_candidate") if fdi_report else {}
            )

            row = dict(
                epoch=epoch + 1,
                phase=phase_label,
                lr=current_lr,
                base_lr=current_lr,
                tau=tau if is_learned else None,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                selection_metric=selection_metric_name,
                selection_acc=selection_acc,
                neuron_type=cfg.liquid.neuron_type,
                **recurrent_metrics,
                **soft_gate_metrics,
                readout_mode=cfg.liquid.readout_mode,
                readout_lif_beta=readout_lif_info["beta"],
                readout_lif_normalize=cfg.liquid.readout_lif_normalize,
                readout_lif_bias_once=cfg.liquid.readout_lif_bias_once,
                readout_lif_mem_norm=readout_lif_info["mem_norm"],
                readout_lif_final_logit_norm=readout_lif_info["final_logit_norm"],
                sparsity=sparsity,
                hard_density=sparsity,
                hard_density_delta=(
                    sparsity - previous_sparsity
                    if previous_sparsity is not None
                    else None
                ),
                topology_logit_mean=topology_logit_mean,
                topology_logit_std=topology_logit_std,
                topology_logit_mean_delta=topology_logit_mean_delta,
                topology_logit_std_delta=topology_logit_std_delta,
                topology_logit_p05=topology_logit_p05,
                topology_logit_p95=topology_logit_p95,
                topology_logit_delta_l2=topology_logit_delta_l2,
                topology_logit_delta_mean_abs=topology_logit_delta_mean_abs,
                topology_sigmoid_mean=topology_sigmoid_mean,
                topology_sigmoid_std=topology_sigmoid_std,
                topology_sigmoid_delta_mean_abs=topology_sigmoid_delta_mean_abs,
                topology_entropy=topology_entropy,
                topology_probability_source=topology_metrics.get(
                    "topology_probability_source"
                ),
                edge_prob_entropy=topology_metrics.get("edge_prob_entropy"),
                edge_prob_mean=topology_metrics.get("edge_prob_mean"),
                edge_prob_std=topology_metrics.get("edge_prob_std"),
                top_edge_prob_mean=topology_metrics.get("top_edge_prob_mean"),
                in_degree_gini=topology_metrics.get("in_degree_gini"),
                out_degree_gini=topology_metrics.get("out_degree_gini"),
                max_in_degree=topology_metrics.get("max_in_degree"),
                max_out_degree=topology_metrics.get("max_out_degree"),
                theta_bias=topology_metrics.get("theta_bias"),
                topology_edge_prob_low_frac=topology_edge_prob_low_frac,
                topology_edge_prob_high_frac=topology_edge_prob_high_frac,
                theta_mean=topology_logit_mean,
                theta_std=topology_logit_std,
                grad_norm=avg_grad_norm,
                topology_grad_pre_clip=avg_topology_grad_pre,
                topology_grad_post_clip=avg_topology_grad_post,
                topology_grad_norm_pre_clip=avg_topology_grad_pre,
                topology_grad_norm_post_clip=avg_topology_grad_post,
                topology_clip_violation=topology_clip_violation,
                theta_grad_norm=avg_topology_grad_pre,
                theta_grad_norm_pre_clip=avg_topology_grad_pre,
                theta_grad_norm_post_clip=avg_topology_grad_post,
                w_raw_grad_norm=avg_w_raw_grad,
                w_raw_grad_pre_clip=avg_w_raw_grad,
                src_grad_pre_clip=avg_src_grad,
                dst_grad_pre_clip=avg_dst_grad,
                bias_grad_pre_clip=avg_bias_grad,
                src_grad_post_clip=avg_src_grad_post,
                dst_grad_post_clip=avg_dst_grad_post,
                bias_grad_post_clip=avg_bias_grad_post,
                topology_lr=topology_lr,
                theta_bias_lr=theta_bias_lr,
                topology_lr_scale=topology_lr_scale,
                topology_lr_scale_effective=topology_lr_scale,
                fdi_selected_input_scale=fdi_selected_candidate.get("input_scale"),
                fdi_selected_recurrent_scale=fdi_selected_candidate.get(
                    "recurrent_scale"
                ),
                fdi_selected_threshold_scale=fdi_selected_candidate.get(
                    "threshold_scale"
                ),
                input_projection_mode=cfg.liquid.input_projection_mode,
                input_proj_trainable=model.input_proj.trainable,
                device=str(device),
                input_proj_lr=input_proj_lr,
                input_proj_grad_norm=avg_input_proj_grad,
                input_proj_weight_norm=model.input_proj.effective_weight_norm(),
                input_proj_effective_weight_norm=model.input_proj.effective_weight_norm(),
                input_proj_effective_density=model.input_proj.effective_density(),
                mean_firing_rate=fr_info["mean"],
                max_firing_rate=fr_info["max"],
                mean_adaptation=adapt_info["mean"],
                max_adaptation=adapt_info["max"],
                firing_rate_mean=train_activity_info["mean_firing_rate"],
                firing_rate_max=train_activity_info["max_firing_rate"],
                adaptation_mean=train_activity_info["adaptation_mean"],
                adaptation_max=train_activity_info["adaptation_max"],
                membrane_mean=train_activity_info["membrane_mean"],
                membrane_max=train_activity_info["membrane_max"],
                input_current_abs_mean=train_activity_info["input_current_abs_mean"],
                input_current_abs_max=train_activity_info["input_current_abs_max"],
                recurrent_current_abs_mean=train_activity_info[
                    "recurrent_current_abs_mean"
                ],
                recurrent_current_abs_max=train_activity_info[
                    "recurrent_current_abs_max"
                ],
                rec_input_abs_ratio=train_activity_info["rec_input_abs_ratio"],
                mean_motor_firing_rate=motor_info["mean_rate"],
                max_motor_firing_rate=motor_info["max_rate"],
                mean_motor_spike_count=motor_info["mean_count"],
                max_motor_spike_count=motor_info["max_count"],
                mean_motor_membrane_trace=motor_info["mean_membrane"],
                max_motor_membrane_trace=motor_info["max_membrane"],
                pred_loss=model.prediction_info(),
                warmup_epoch=warmup if uses_topology_warmup else 0,
                warmup_dynamic=dynamic_warmup,
                warmup_strategy=warmup_strategy if dynamic_warmup else "",
                theta_freeze_epoch=theta_freeze_epoch if has_theta else 0,
                theta_frozen=topology_is_frozen,
                theta_adaptive_freeze=cfg.liquid.theta_adaptive_freeze,
                theta_freeze_reason=topology_freeze_reason or "",
                adaptive_freeze_bad_epochs=adaptive_freeze_bad_epochs,
                topology_adaptive_freeze=cfg.liquid.topology_adaptive_freeze,
                topology_frozen=topology_is_frozen,
                topology_frozen_epoch=topology_frozen_epoch,
                topology_freeze_reason=topology_freeze_reason,
                topology_best_metric_name=topology_best_metric_name,
                topology_best_metric_value=(
                    best_topology_metric if best_topology_epoch is not None else None
                ),
                topology_best_epoch=best_topology_epoch,
                topology_bad_count=topology_bad_count,
                topology_rollback_applied=topology_rollback_applied_epoch,
                topology_rollback_reason=topology_rollback_reason,
                topology_rollback_target_epoch=topology_rollback_target_epoch,
                topology_grad_exceeded_threshold=topology_grad_exceeded_threshold,
                max_firing_rate_exceeded_threshold=max_firing_rate_exceeded_threshold,
                topology_runaway_guard_enabled=guard_enabled,
                topology_runaway_guard_active=topology_runaway_frozen_at_epoch_start,
                topology_runaway_guard_triggered=topology_runaway_guard_triggered,
                topology_runaway_bad_epochs=topology_runaway_bad_epochs,
                topology_runaway_freeze_remaining=topology_runaway_freeze_remaining,
                topology_runaway_rollback_epoch=topology_runaway_rollback_epoch,
                warmup_metric_value=None,
                warmup_slope=None,
                warmup_slow_count=0,
            )

            postfix: dict = dict(
                loss=f"{train_loss:.4f}",
                train=f"{train_acc:.4f}",
                test=f"{test_acc:.4f}",
                select=f"{selection_metric_name}:{selection_acc:.4f}",
                sp=f"{sparsity:.3f}",
                topo_bad=str(topology_bad_count),
                topo_frozen=str(topology_is_frozen).lower(),
            )
            if val_acc is not None:
                postfix["val"] = f"{val_acc:.4f}"
            if best_topology_epoch is not None:
                postfix["topo_best"] = (
                    f"{topology_best_metric_name}:{best_topology_metric:.4f}@{best_topology_epoch}"
                )
            if is_learned:
                postfix["tau"] = f"{tau:.3f}"
            if is_soft_gate:
                postfix["soft_den"] = f"{soft_gate_metrics.get('soft_density', 0.0):.3f}"
                postfix["target"] = f"{float(soft_gate_target_density):.3f}"
                postfix["temp"] = f"{float(soft_gate_temp):.3f}"
            epoch_bar.set_postfix(postfix)
            grad_detail = (
                f"  topo_grad={avg_topology_grad_pre:.2e}/{avg_topology_grad_post:.2e}  "
                f"w_grad={avg_w_raw_grad:.2e}"
                if has_theta
                else ""
            )
            lr_detail = (
                f"  topo_lr={topology_lr:.2e}"
                + (
                    f"  bias_lr={theta_bias_lr:.2e}"
                    if cfg.liquid.recurrent_mode in lowrank_param_modes
                    else ""
                )
                if has_theta
                else ""
            )
            motor_detail = (
                f"  motor_fr={motor_info['mean_rate']:.3f}/{motor_info['max_rate']:.3f}"
                f"  motor_count={motor_info['mean_count']:.2f}/{motor_info['max_count']:.2f}"
                f"  motor_mem={motor_info['mean_membrane']:.2f}/{motor_info['max_membrane']:.2f}"
                if cfg.liquid.readout_mode
                in {"motor_lif", "motor_lif_count_membrane"}
                else ""
            )
            readout_lif_detail = (
                f"  readout_lif_beta={readout_lif_info['beta']:.3f}"
                f"  readout_lif_norm={readout_lif_info['final_logit_norm']:.2f}"
                if cfg.liquid.readout_mode == "non_spiking_lif_final_mem"
                else ""
            )
            tau_str = f"  tau={tau:.3f}" if is_learned else ""
            soft_gate_detail = (
                f"  soft_den={soft_gate_metrics.get('soft_density', 0.0):.3f}"
                f"/{float(soft_gate_target_density):.3f}"
                f"  temp={float(soft_gate_temp):.3f}"
                f"  dens_pen={soft_gate_metrics.get('density_penalty', 0.0):.3g}"
                f"  hard_act={soft_gate_metrics.get('hard_active_fraction', 0.0):.3f}"
                if is_soft_gate
                else ""
            )
            tqdm.write(
                f"[{epoch+1:03d}/{cfg.epochs}|{phase_label}] "
                f"lr={current_lr:.2e}{tau_str}  loss={train_loss:.4f}  "
                f"train={train_acc:.4f}  "
                + (f"val={val_acc:.4f}  " if val_acc is not None else "")
                + f"test={test_acc:.4f}  "
                f"select={selection_metric_name}:{selection_acc:.4f}  "
                + (
                    f"topo_best={topology_best_metric_name}:{best_topology_metric:.4f}@{best_topology_epoch}  "
                    if best_topology_epoch is not None
                    else ""
                )
                + f"topo_bad={topology_bad_count}  topo_frozen={str(topology_is_frozen).lower()}  "
                f"sp={sparsity:.3f}  grad={avg_grad_norm:.1f}  "
                f"fr={fr_info['mean']:.3f}/{fr_info['max']:.3f}  "
                f"adapt={adapt_info['mean']:.3f}/{adapt_info['max']:.3f}  "
                f"logit={topology_logit_mean:.3f}±{topology_logit_std:.3f}  "
                f"sig={topology_sigmoid_mean:.3f}±{topology_sigmoid_std:.3f}"
                + soft_gate_detail
                + motor_detail
                + readout_lif_detail
                + grad_detail
                + lr_detail
            )

            # early warnings
            if avg_grad_norm > 100:
                tqdm.write(
                    f"  ⚠ grad_norm={avg_grad_norm:.1f} — consider reducing lr or clip_max_norm"
                )
            if has_theta and avg_topology_grad_pre > 50:
                schedule_label = (
                    f"tau={tau:.3f}"
                    if is_learned
                    else (
                        f"temp={float(soft_gate_temp):.3f}"
                        if is_soft_gate
                        else "deterministic"
                    )
                )
                tqdm.write(
                    f"  ⚠ topo_grad={avg_topology_grad_pre:.1f} — topology gradient exploding ({schedule_label})"
                )
            if has_theta and topology_clip_violation:
                tqdm.write(
                    f"  ⚠ topo_grad_post={avg_topology_grad_post:.2f} exceeded clip max {clip_norm_theta:.2f}"
                )
            if is_learned and avg_w_raw_grad > 50:
                tqdm.write(
                    f"  ⚠ w_raw_grad={avg_w_raw_grad:.1f} — weight gradient exploding"
                )
            if fr_info["max"] > 0.9:
                tqdm.write(
                    f"  ⚠ max_firing_rate={fr_info['max']:.3f} — possible excitatory loop runaway"
                )
            if epoch > 20 and topology_logit_std < 0.01:
                tqdm.write(
                    f"  ⚠ logit_std={topology_logit_std:.4f} — topology logit stagnating, consider increasing lambda_commit"
                )

            if dynamic_warmup and phase_label == "P1":
                metric_value = _select_warmup_metric(warmup_metric, row)
                warmup_scores.append(_warmup_score(warmup_metric, metric_value))
                warmup_slope = _warmup_slope(warmup_scores, warmup_window)

                if warmup_strategy == "best":
                    if _metric_improved(
                        warmup_metric,
                        metric_value,
                        warmup_metric_best,
                        warmup_min_delta,
                    ):
                        warmup_metric_best = metric_value
                        warmup_slow_count = 0
                    else:
                        warmup_slow_count += 1
                else:
                    if warmup_slope is None or warmup_slope >= warmup_min_delta:
                        warmup_slow_count = 0
                    else:
                        warmup_slow_count += 1

                row["warmup_metric_value"] = metric_value
                row["warmup_slope"] = warmup_slope
                row["warmup_slow_count"] = warmup_slow_count

                p1_epochs_done = epoch + 1
                can_switch = (
                    p1_epochs_done >= warmup_min_epochs
                    and warmup_slow_count >= warmup_patience
                    and p1_epochs_done < warmup
                )
                if can_switch:
                    warmup = p1_epochs_done
                    row["warmup_epoch"] = warmup
                    tqdm.write(
                        "  Dynamic warmup: "
                        f"{warmup_metric} slowed for {warmup_patience} checks; "
                        f"switching to P2 at epoch {warmup + 1}"
                    )

            candidate_history = history + [row]
            best_improved, selection_tie_break_reason = _selection_row_is_better(
                row, best_checkpoint_row, cfg, selection_metric_name
            )
            top_k_val = max(int(cfg.checkpoint_top_k_val), 0)
            top_val_rows = (
                _rank_validation_rows(candidate_history, cfg)[:top_k_val]
                if top_k_val > 0 and val_loader is not None
                else []
            )
            top_val_epochs = {
                item_epoch
                for item_epoch in (_row_epoch(item) for item in top_val_rows)
                if item_epoch is not None
            }

            row["checkpoint_is_best"] = best_improved
            row["selection_tie_break_reason"] = selection_tie_break_reason
            row["checkpoint_in_top_k_val"] = (epoch + 1) in top_val_epochs
            if selection_tie_break_reason == "val_acc_tie_lower_val_loss":
                tqdm.write(
                    "  val_acc tie within "
                    f"{max(float(cfg.selection_tie_epsilon), 0.0):g}; "
                    "selecting lower val_loss checkpoint"
                )
            elif selection_tie_break_reason.startswith("val_acc_tie_missing_val_loss"):
                tqdm.write(
                    "  val_acc tie without comparable val_loss; keeping existing "
                    "checkpoint by default"
                )

            history.append(row)
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

            if best_improved:
                best_checkpoint_row = dict(row)
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_test_acc_at_best_val = test_acc
                best_epoch = epoch + 1
                epochs_no_improve = 0
                _save_training_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    row=row,
                    history=history,
                    best_row=best_checkpoint_row,
                    best_metric_name=best_metric_name,
                    topology_freeze_enabled=topology_freeze_enabled,
                    topology_freeze_reason=topology_freeze_reason,
                    topology_frozen_epoch=topology_frozen_epoch,
                    topology_best_epoch=best_topology_epoch,
                    topology_best_metric_name=topology_best_metric_name,
                    topology_best_metric_value=(
                        best_topology_metric
                        if best_topology_epoch is not None
                        else None
                    ),
                    topology_rollback_applied_any=topology_rollback_applied_any,
                    metadata=_checkpoint_metadata(
                        row,
                        topo_frozen=topology_is_frozen,
                        topology_rollback_target_epoch=topology_rollback_target_epoch,
                        checkpoint_kind="best",
                        selection_tie_break_reason=selection_tie_break_reason,
                        checkpoint_in_top_k_val=row["checkpoint_in_top_k_val"],
                    ),
                )
            else:
                epochs_no_improve += 1

            _save_training_checkpoint(
                last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                row=row,
                history=history,
                best_row=best_checkpoint_row,
                best_metric_name=best_metric_name,
                topology_freeze_enabled=topology_freeze_enabled,
                topology_freeze_reason=topology_freeze_reason,
                topology_frozen_epoch=topology_frozen_epoch,
                topology_best_epoch=best_topology_epoch,
                topology_best_metric_name=topology_best_metric_name,
                topology_best_metric_value=(
                    best_topology_metric if best_topology_epoch is not None else None
                ),
                topology_rollback_applied_any=topology_rollback_applied_any,
                metadata=_checkpoint_metadata(
                    row,
                    topo_frozen=topology_is_frozen,
                    topology_rollback_target_epoch=topology_rollback_target_epoch,
                    checkpoint_kind="last",
                    selection_tie_break_reason=selection_tie_break_reason,
                    checkpoint_in_top_k_val=row["checkpoint_in_top_k_val"],
                ),
            )

            if row["checkpoint_in_top_k_val"]:
                _save_training_checkpoint(
                    _top_val_checkpoint_path(exp_dir, epoch + 1),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    row=row,
                    history=history,
                    best_row=best_checkpoint_row,
                    best_metric_name=best_metric_name,
                    topology_freeze_enabled=topology_freeze_enabled,
                    topology_freeze_reason=topology_freeze_reason,
                    topology_frozen_epoch=topology_frozen_epoch,
                    topology_best_epoch=best_topology_epoch,
                    topology_best_metric_name=topology_best_metric_name,
                    topology_best_metric_value=(
                        best_topology_metric
                        if best_topology_epoch is not None
                        else None
                    ),
                    topology_rollback_applied_any=topology_rollback_applied_any,
                    metadata=_checkpoint_metadata(
                        row,
                        topo_frozen=topology_is_frozen,
                        topology_rollback_target_epoch=topology_rollback_target_epoch,
                        checkpoint_kind="top_k_val",
                        selection_tie_break_reason=selection_tie_break_reason,
                        checkpoint_in_top_k_val=True,
                    ),
                )
            if top_k_val > 0:
                _prune_top_val_checkpoints(exp_dir, top_val_epochs)

            last_epoch_completed = epoch + 1
            will_stop = cfg.patience > 0 and epochs_no_improve >= cfg.patience
            is_final_epoch = epoch + 1 >= cfg.epochs or will_stop
            if diagnostics_logger is not None:
                if cfg.diagnostics.log_every_epoch or is_final_epoch:
                    diagnostic_input = dict(row)
                    diagnostic_input.update(
                        best_val_acc_so_far=best_val_acc,
                        test_at_best_val=(
                            best_test_acc_at_best_val
                            if val_loader is not None
                            else None
                        ),
                        test_at_best_val_expected=(
                            val_loader is not None and test_acc is not None
                        ),
                    )
                    force_topology = bool(
                        is_final_epoch
                        or topology_rollback_applied_epoch
                        or topology_frozen_epoch == epoch + 1
                    )
                    diagnostics_logger.log_epoch(
                        epoch + 1,
                        collect_epoch_diagnostics(
                            model,
                            diagnostic_input,
                            cfg,
                            force_topology=force_topology,
                            final_epoch=is_final_epoch,
                        ),
                    )
                if best_improved:
                    diagnostics_logger.save_topology_snapshot(
                        model, "best", epoch + 1
                    )
                if topology_rollback_applied_epoch or topology_frozen_epoch == epoch + 1:
                    diagnostics_logger.save_topology_snapshot(
                        model, "freeze", epoch + 1
                    )

            if has_theta and topology_logit_flat is not None:
                previous_topology_logit = topology_logit_flat.clone()
                previous_topology_sigmoid = topology_sigmoid_flat.clone()
                previous_topology_logit_mean = topology_logit_mean
                previous_topology_logit_std = topology_logit_std
                previous_sparsity = sparsity

            if topology_runaway_frozen_at_epoch_start:
                topology_runaway_freeze_remaining = max(
                    topology_runaway_freeze_remaining - 1, 0
                )
                if (
                    topology_runaway_freeze_remaining == 0
                    and not theta_adaptive_frozen
                    and not topology_frozen_by_validation
                    and not (
                        has_theta
                        and theta_freeze_epoch > 0
                        and epoch + 1 >= theta_freeze_epoch
                    )
                    and uses_topology_warmup
                    and epoch + 1 >= warmup
                ):
                    _set_topology_requires_grad(True)
                    tqdm.write(
                        f"  [TopologyRunawayGuard] temporary freeze expired after epoch {epoch + 1}; topology learning resumes."
                    )

            if will_stop:
                tqdm.write(f"Early stopping: no improvement for {cfg.patience} epochs.")
                break

    if diagnostics_logger is not None:
        if last_epoch_completed > 0:
            diagnostics_logger.save_topology_snapshot(
                model, "final", last_epoch_completed
            )
        diagnostics_logger.summarize_run()

    best_metric_value = (
        _safe_float(best_checkpoint_row.get(best_metric_name))
        if best_checkpoint_row is not None
        else float("nan")
    )
    if best_metric_value is None:
        best_metric_value = float("nan")
    print(f"\nBest {best_metric_name}: {best_metric_value:.4f} at epoch {best_epoch}")
    if best_test_acc_at_best_val is not None and best_metric_name != "test_acc":
        print(f"Test accuracy at best {best_metric_name}: {best_test_acc_at_best_val:.4f}")
    print(f"Experiment saved to: {exp_dir}")
    return history, exp_dir
