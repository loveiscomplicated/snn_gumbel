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
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.loaders import get_train_val_test_dataloaders
from src.lsm.diagnostics import DiagnosticsLogger, collect_epoch_diagnostics
from src.lsm.initialization.fdi_calibration import calibrate_fdi_style_initial_regime
from src.lsm.model import LSMModel
from src.utils.config import Config


ce_loss = nn.CrossEntropyLoss()


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


def _param_group_lr(optimizer: optim.Optimizer, name: str) -> float:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return float(group["lr"])
    return 0.0


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
        "grad_r",
    }
    topology_params = _topology_param_group(model, cfg) if has_theta else []
    theta_bias_params: list[nn.Parameter] = []
    theta_main_params = topology_params
    if cfg.liquid.recurrent_mode == "learned_lowrank":
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

    # Phase 1 warmup: freeze theta so w_raw/readout learn on stable random topology.
    # Phase 2: unfreeze theta to learn topology via Gumbel-STE.
    warmup = cfg.liquid.theta_warmup_epochs
    learned_modes = {"learned", "learned_lowrank"}
    is_learned = cfg.liquid.recurrent_mode in learned_modes
    # grad_r also has a trainable theta but skips Gumbel noise / warmup logic
    has_theta = is_learned or cfg.liquid.recurrent_mode == "grad_r"
    topology_trainable_modes = {"learned", "learned_lowrank", "grad_r"}
    topology_freeze_enabled = has_theta and cfg.liquid.topology_adaptive_freeze
    dynamic_warmup = is_learned and cfg.liquid.theta_warmup_dynamic and warmup > 0
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
    if warmup_metric == "val_acc" and val_loader is None:
        warmup_metric = "test_acc"
        tqdm.write(
            "  Validation disabled; falling back to test_acc for theta warmup metric"
        )
    if is_learned and warmup > 0:
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
        if not is_learned or epoch_idx < warmup:
            return 0.0
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

    best_acc = float("-inf")
    best_metric_name = selection_metric_name
    best_metric_value: float | None = None
    best_val_acc: float | None = None
    best_test_acc_at_best_val: float | None = None
    best_epoch = 0
    epochs_no_improve = 0
    history: list[dict] = []
    clip_norm_w = cfg.liquid.grad_clip_max_norm_w
    clip_norm_theta = cfg.liquid.grad_clip_max_norm_theta
    clip_norm_input_projection = cfg.liquid.input_proj_grad_clip

    adaptive_freeze_bad_epochs = 0
    theta_adaptive_frozen = False
    topology_frozen_by_validation = False
    topology_freeze_reason: str | None = None
    topology_frozen_epoch: int | None = None
    best_topology_state: dict[str, torch.Tensor] | None = None
    best_topology_metric = float("-inf")
    best_topology_epoch: int | None = None
    topology_bad_count = 0
    topology_best_metric_name = cfg.liquid.topology_freeze_metric
    topology_rollback_applied_any = False

    def _topology_is_frozen(epoch_idx: int) -> bool:
        fixed_epoch_frozen = topology_freeze_reason == "fixed_epoch" or (
            has_theta and theta_freeze_epoch > 0 and epoch_idx + 1 > theta_freeze_epoch
        )
        return fixed_epoch_frozen or theta_adaptive_frozen or topology_frozen_by_validation

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

    epoch_bar = tqdm(range(cfg.epochs), desc="Epochs", unit="ep")
    alif_batch_debug_logged = False
    last_epoch_completed = 0

    with open(log_path, "a") as log_f:
        for epoch in epoch_bar:
            current_lr = optimizer.param_groups[0]["lr"]
            topology_lr_scale = _current_topology_lr_scale(epoch)
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
            if is_learned and warmup > 0 and epoch == warmup and not _topology_is_frozen(epoch):
                _set_topology_requires_grad(True)
                tqdm.write(
                    f"  Phase 2: theta unfrozen at epoch {epoch+1}, topology learning begins"
                )
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
            phase_label = "P1" if (is_learned and epoch < warmup) else "P2"
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
            topology_clip_violations = 0

            batch_bar = tqdm(train_loader, desc="  Train", leave=False, unit="batch")
            for x, y in batch_bar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                rates = model(x, tau=tau)
                loss = _compute_loss(rates, y, model, cfg)
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
                if cfg.liquid.recurrent_mode == "learned_lowrank":
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
                    if cfg.liquid.recurrent_mode == "learned_lowrank":
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
            topology_clip_violation = topology_clip_violations > 0

            # Adaptive theta freeze: check after batch loop once avg_theta_grad is known
            if (
                has_theta
                and cfg.liquid.theta_adaptive_freeze
                and not theta_adaptive_frozen
                and not topology_is_frozen
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
            fr_info = model.firing_rate_info()
            adapt_info = model.adaptation_info()
            motor_info = model.motor_info()
            readout_lif_info = model.readout_lif_info()
            topology_rollback_applied_epoch = False

            if (
                topology_freeze_enabled
                and cfg.liquid.recurrent_mode in topology_trainable_modes
                and not topology_is_frozen
            ):
                current_metric = val_acc
                if current_metric is None:
                    raise RuntimeError(
                        "Validation accuracy is required for topology_adaptive_freeze."
                    )

                improved = (
                    current_metric
                    > best_topology_metric + cfg.liquid.topology_freeze_min_delta
                )
                if improved:
                    best_topology_metric = current_metric
                    best_topology_epoch = epoch + 1
                    best_topology_state = model.liquid.topology_state_dict()
                    topology_bad_count = 0
                else:
                    topology_bad_count += 1

                can_freeze_topology = (
                    epoch + 1 >= cfg.liquid.topology_freeze_min_epoch
                    and topology_bad_count >= cfg.liquid.topology_freeze_patience
                    and not _topology_is_frozen(epoch)
                )
                if can_freeze_topology:
                    if cfg.liquid.topology_freeze_rollback_best:
                        if best_topology_state is None or best_topology_epoch is None:
                            raise RuntimeError(
                                "No topology snapshot available for rollback."
                            )
                        model.liquid.load_topology_state_dict(best_topology_state)
                        topology_rollback_applied_epoch = True
                        topology_rollback_applied_any = True

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
                        model.liquid.get_theta()
                        if cfg.liquid.recurrent_mode in learned_modes
                        else model.liquid.theta
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
                else:
                    topology_logit_mean = 0.0
                    topology_logit_std = 0.0
                    topology_sigmoid_mean = 0.0
                    topology_sigmoid_std = 0.0
                    topology_logit_p05 = 0.0
                    topology_logit_p95 = 0.0

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
                readout_mode=cfg.liquid.readout_mode,
                readout_lif_beta=readout_lif_info["beta"],
                readout_lif_normalize=cfg.liquid.readout_lif_normalize,
                readout_lif_bias_once=cfg.liquid.readout_lif_bias_once,
                readout_lif_mem_norm=readout_lif_info["mem_norm"],
                readout_lif_final_logit_norm=readout_lif_info["final_logit_norm"],
                sparsity=sparsity,
                hard_density=sparsity,
                topology_logit_mean=topology_logit_mean,
                topology_logit_std=topology_logit_std,
                topology_logit_p05=topology_logit_p05,
                topology_logit_p95=topology_logit_p95,
                topology_sigmoid_mean=topology_sigmoid_mean,
                topology_sigmoid_std=topology_sigmoid_std,
                theta_mean=topology_logit_mean,
                theta_std=topology_logit_std,
                grad_norm=avg_grad_norm,
                topology_grad_pre_clip=avg_topology_grad_pre,
                topology_grad_post_clip=avg_topology_grad_post,
                topology_clip_violation=topology_clip_violation,
                theta_grad_norm=avg_topology_grad_pre,
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
                mean_motor_firing_rate=motor_info["mean_rate"],
                max_motor_firing_rate=motor_info["max_rate"],
                mean_motor_spike_count=motor_info["mean_count"],
                max_motor_spike_count=motor_info["max_count"],
                mean_motor_membrane_trace=motor_info["mean_membrane"],
                max_motor_membrane_trace=motor_info["max_membrane"],
                pred_loss=model.prediction_info(),
                warmup_epoch=warmup if is_learned else 0,
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
                    if cfg.liquid.recurrent_mode == "learned_lowrank"
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
                tqdm.write(
                    f"  ⚠ topo_grad={avg_topology_grad_pre:.1f} — topology gradient exploding (tau={tau:.3f})"
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

            history.append(row)
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

            # checkpoint best
            best_improved = selection_acc > best_acc
            if best_improved:
                best_acc = selection_acc
                best_metric_value = selection_acc
                best_val_acc = val_acc
                best_test_acc_at_best_val = test_acc
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "best_acc": best_acc,
                        "best_metric_name": best_metric_name,
                        "best_metric_value": best_metric_value,
                        "best_val_acc": best_val_acc,
                        "best_val_loss": val_loss,
                        "best_test_loss": test_loss,
                        "best_test_acc_at_best_val": best_test_acc_at_best_val,
                        "best_epoch": best_epoch,
                        "topology_freeze_enabled": topology_freeze_enabled,
                        "topology_freeze_reason": topology_freeze_reason,
                        "topology_frozen_epoch": topology_frozen_epoch,
                        "topology_best_epoch": best_topology_epoch,
                        "topology_best_metric_name": topology_best_metric_name,
                        "topology_best_metric_value": (
                            best_topology_metric
                            if best_topology_epoch is not None
                            else None
                        ),
                        "topology_rollback_applied": topology_rollback_applied_any,
                        "history": history,
                    },
                    checkpoint_path,
                )
            else:
                epochs_no_improve += 1

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

            if will_stop:
                tqdm.write(f"Early stopping: no improvement for {cfg.patience} epochs.")
                break

    if diagnostics_logger is not None:
        if last_epoch_completed > 0:
            diagnostics_logger.save_topology_snapshot(
                model, "final", last_epoch_completed
            )
        diagnostics_logger.summarize_run()

    if best_metric_value is None:
        best_metric_value = float("nan")
    print(f"\nBest {best_metric_name}: {best_metric_value:.4f} at epoch {best_epoch}")
    if best_test_acc_at_best_val is not None and best_metric_name != "test_acc":
        print(f"Test accuracy at best {best_metric_name}: {best_test_acc_at_best_val:.4f}")
    print(f"Experiment saved to: {exp_dir}")
    return history, exp_dir
