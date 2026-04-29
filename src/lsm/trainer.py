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

from src.data.loaders import get_dataloaders
from src.lsm.model import LSMModel
from src.utils.config import Config


ce_loss = nn.CrossEntropyLoss()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
        beta_min=liq.beta_min,
        beta_max=liq.beta_max,
        threshold_min=liq.threshold_min,
        threshold_max=liq.threshold_max,
        p_input=liq.p_input,
        input_weight_scale=liq.input_weight_scale,
        recurrent_mode=liq.recurrent_mode,
        recurrent_sparsity=liq.recurrent_sparsity,
        self_connection=liq.self_connection,
        theta_init_mean=liq.theta_init_mean,
        theta_init_std=liq.theta_init_std,
        w_raw_init_mean=liq.w_raw_init_mean,
        w_raw_init_std=liq.w_raw_init_std,
        train_w_raw=liq.train_w_raw,
        w_raw_max=liq.w_raw_max,
        bptt_truncate=liq.bptt_truncate,
        noise_scale=liq.noise_scale,
    ).to(device)


def _compute_loss(rates, labels, model, cfg):
    loss = ce_loss(rates, labels)
    sp = model.sparsity_loss()
    cm = model.commitment_loss()
    return loss + cfg.lambda_sparse * sp + cfg.lambda_commit * cm


def _evaluate(model: LSMModel, loader, device: torch.device, tau: float) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            rates = model(x, tau=tau)
            pred = rates.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def _metric_improved(
    metric_name: str, value: float, best_value: float | None, min_delta: float
) -> bool:
    if best_value is None:
        return True
    if metric_name == "train_loss":
        return value < best_value - min_delta
    return value > best_value + min_delta


def _select_warmup_metric(metric_name: str, row: dict) -> float:
    if metric_name not in {"test_acc", "train_acc", "train_loss"}:
        raise ValueError(
            "liquid.theta_warmup_metric must be one of: "
            "test_acc, train_acc, train_loss"
        )
    return row[metric_name]


def _warmup_score(metric_name: str, value: float) -> float:
    return -value if metric_name == "train_loss" else value


def _warmup_slope(scores: list[float], window: int) -> float | None:
    if len(scores) < window:
        return None
    recent = scores[-window:]
    return (recent[-1] - recent[0]) / max(window - 1, 1)


def train(cfg: Config) -> tuple:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    torch.manual_seed(cfg.seed)
    device = get_device()

    exp_dir = _make_experiment_dir(cfg)
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    log_path = exp_dir / "logs" / "train.jsonl"

    train_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg, device)

    # Phase 1 warmup: freeze theta so w_raw/readout learn on stable random topology.
    # Phase 2: unfreeze theta to learn topology via Gumbel-STE.
    warmup = cfg.liquid.theta_warmup_epochs
    is_learned = cfg.liquid.recurrent_mode == "learned"
    dynamic_warmup = is_learned and cfg.liquid.theta_warmup_dynamic and warmup > 0
    warmup_min_epochs = min(max(cfg.liquid.theta_warmup_min_epochs, 1), warmup)
    warmup_patience = max(cfg.liquid.theta_warmup_patience, 1)
    warmup_min_delta = max(cfg.liquid.theta_warmup_min_delta, 0.0)
    warmup_metric = cfg.liquid.theta_warmup_metric
    warmup_strategy = cfg.liquid.theta_warmup_strategy
    warmup_window = max(cfg.liquid.theta_warmup_window, 2)
    warmup_metric_best: float | None = None
    warmup_slow_count = 0
    warmup_scores: list[float] = []
    if warmup_strategy not in {"slope", "best"}:
        raise ValueError("liquid.theta_warmup_strategy must be one of: slope, best")
    if is_learned and warmup > 0:
        model.liquid.theta.requires_grad_(False)
        tqdm.write(f"  Phase 1 warmup: theta frozen for {warmup} epochs")
        if dynamic_warmup:
            tqdm.write(
                "  Dynamic warmup enabled: "
                f"min={warmup_min_epochs}, max={warmup}, "
                f"strategy={warmup_strategy}, window={warmup_window}, "
                f"patience={warmup_patience}, metric={warmup_metric}, "
                f"min_delta={warmup_min_delta:g}"
            )

    # Separate optimizer groups so theta gradient cannot suppress w_raw updates.
    # Independent per-group clipping is applied before optimizer.step().
    if is_learned:
        theta_params = [model.liquid.theta]
        other_params = [p for n, p in model.named_parameters() if n != "liquid.theta"]
        param_groups = [
            {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay},
            {
                "params": theta_params,
                "lr": cfg.lr * cfg.liquid.theta_lr_scale,
                "weight_decay": 0.0,
            },
        ]
    else:
        other_params = list(model.parameters())
        theta_params = []
        param_groups = [
            {"params": other_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay}
        ]
    optimizer = optim.Adam(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    best_acc = 0.0
    epochs_no_improve = 0
    history: list[dict] = []
    clip_norm_w = cfg.liquid.grad_clip_max_norm_w
    clip_norm_theta = cfg.liquid.grad_clip_max_norm_theta

    epoch_bar = tqdm(range(cfg.epochs), desc="Epochs", unit="ep")

    with open(log_path, "a") as log_f:
        for epoch in epoch_bar:
            # Phase transition: unfreeze theta at warmup boundary
            if is_learned and warmup > 0 and epoch == warmup:
                model.liquid.theta.requires_grad_(True)
                tqdm.write(
                    f"  Phase 2: theta unfrozen at epoch {epoch+1}, topology learning begins"
                )

            tau = get_tau(epoch, cfg, warmup_epochs=warmup)
            phase_label = "P1" if (is_learned and epoch < warmup) else "P2"
            model.train()

            # Phase 2: sample Gumbel noise ONCE per epoch, lock mask for all batches.
            # This keeps topology stable within an epoch (BPTT safe) while allowing
            # exploration across epochs (OFF edges get a chance to be ON → w_raw learns).
            if is_learned and epoch >= warmup:
                eps = torch.rand_like(model.liquid.theta).clamp(1e-6, 1 - 1e-6)
                epoch_noise = (torch.log(eps) - torch.log(1.0 - eps)).to(device)
                model.liquid.sample_epoch_mask(tau=tau, epoch_noise=epoch_noise)

            total_l = correct = n = 0
            epoch_grad_norm = 0.0
            n_batches = 0

            epoch_theta_grad_norm = 0.0
            epoch_w_raw_grad_norm = 0.0

            batch_bar = tqdm(train_loader, desc="  Train", leave=False, unit="batch")
            for x, y in batch_bar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                rates = model(x, tau=tau)
                loss = _compute_loss(rates, y, model, cfg)
                # NaN detection
                if torch.isnan(loss):
                    tqdm.write(
                        f"  ✖ NaN loss detected at epoch {epoch+1}, batch {n_batches+1}. Stopping."
                    )
                    return history, exp_dir

                loss.backward()

                # per-component grad norms (before clipping)
                theta_g = model.liquid.theta.grad
                w_raw_g = model.liquid.w_raw.grad
                if theta_g is not None:
                    epoch_theta_grad_norm += theta_g.norm().item()
                if w_raw_g is not None:
                    epoch_w_raw_grad_norm += w_raw_g.norm().item()

                # Independent per-group clipping:
                #   theta: clip_norm_theta (moderate — prevents runaway while allowing
                #          Adam to normalize; smaller than clip_norm_w to enforce time-
                #          scale separation: topology changes slowly, weights adapt fast)
                #   other: clip_norm_w (large enough for recurrent BPTT norms ~10^2–10^4)
                if is_learned and theta_params:
                    torch.nn.utils.clip_grad_norm_(
                        theta_params, max_norm=clip_norm_theta
                    )
                    other_norm = torch.nn.utils.clip_grad_norm_(
                        other_params, max_norm=clip_norm_w
                    )
                    grad_norm = other_norm
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        other_params, max_norm=clip_norm_w
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
            avg_theta_grad = epoch_theta_grad_norm / max(n_batches, 1)
            avg_w_raw_grad = epoch_w_raw_grad_norm / max(n_batches, 1)

            # Unlock epoch mask before eval so eval uses fresh deterministic mask
            model.liquid.unlock_epoch_mask()
            test_acc = _evaluate(model, test_loader, device, tau)
            sparsity = model.sparsity_info()
            fr_info = model.firing_rate_info()
            current_lr = scheduler.get_last_lr()[0]

            scheduler.step()

            # theta stats
            with torch.no_grad():
                theta = model.liquid.theta
                theta_mean = theta.mean().item()
                theta_std = theta.std().item()

            row = dict(
                epoch=epoch + 1,
                phase=phase_label,
                lr=current_lr,
                tau=tau,
                train_loss=train_loss,
                train_acc=train_acc,
                test_acc=test_acc,
                sparsity=sparsity,
                theta_mean=theta_mean,
                theta_std=theta_std,
                grad_norm=avg_grad_norm,
                theta_grad_norm=avg_theta_grad,
                w_raw_grad_norm=avg_w_raw_grad,
                mean_firing_rate=fr_info["mean"],
                max_firing_rate=fr_info["max"],
                warmup_epoch=warmup if is_learned else 0,
                warmup_dynamic=dynamic_warmup,
                warmup_strategy=warmup_strategy if dynamic_warmup else "",
                warmup_metric_value=None,
                warmup_slope=None,
                warmup_slow_count=0,
            )

            epoch_bar.set_postfix(
                tau=f"{tau:.3f}",
                loss=f"{train_loss:.4f}",
                train=f"{train_acc:.4f}",
                test=f"{test_acc:.4f}",
                sp=f"{sparsity:.3f}",
            )
            grad_detail = (
                f"  θ_grad={avg_theta_grad:.2e}  w_grad={avg_w_raw_grad:.2e}"
                if is_learned
                else ""
            )
            tqdm.write(
                f"[{epoch+1:03d}/{cfg.epochs}|{phase_label}] "
                f"lr={current_lr:.2e}  tau={tau:.3f}  loss={train_loss:.4f}  "
                f"train={train_acc:.4f}  test={test_acc:.4f}  "
                f"sp={sparsity:.3f}  grad={avg_grad_norm:.1f}  "
                f"fr={fr_info['mean']:.3f}/{fr_info['max']:.3f}  "
                f"θ={theta_mean:.3f}±{theta_std:.3f}" + grad_detail
            )

            # early warnings
            if avg_grad_norm > 100:
                tqdm.write(
                    f"  ⚠ grad_norm={avg_grad_norm:.1f} — consider reducing lr or clip_max_norm"
                )
            if is_learned and avg_theta_grad > 50:
                tqdm.write(
                    f"  ⚠ theta_grad={avg_theta_grad:.1f} — topology gradient exploding (tau={tau:.3f})"
                )
            if is_learned and avg_w_raw_grad > 50:
                tqdm.write(
                    f"  ⚠ w_raw_grad={avg_w_raw_grad:.1f} — weight gradient exploding"
                )
            if fr_info["max"] > 0.9:
                tqdm.write(
                    f"  ⚠ max_firing_rate={fr_info['max']:.3f} — possible excitatory loop runaway"
                )
            if epoch > 20 and theta_std < 0.01:
                tqdm.write(
                    f"  ⚠ theta_std={theta_std:.4f} — theta stagnating, consider increasing lambda_commit"
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
            if test_acc > best_acc:
                best_acc = test_acc
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch + 1,
                        "best_acc": best_acc,
                        "history": history,
                    },
                    checkpoint_path,
                )
            else:
                epochs_no_improve += 1

            if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
                tqdm.write(f"Early stopping: no improvement for {cfg.patience} epochs.")
                break

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Experiment saved to: {exp_dir}")
    return history, exp_dir
