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


def get_tau(epoch: int, cfg: Config) -> float:
    if epoch >= cfg.tau_anneal_epochs:
        return cfg.tau_end
    progress = epoch / cfg.tau_anneal_epochs
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.tau_end + (cfg.tau_start - cfg.tau_end) * cosine


def _make_experiment_dir(cfg: Config) -> Path:
    timestamp = datetime.now().strftime("%y%m%d%H%M")
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
        theta_init_std=liq.theta_init_std,
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


def train(cfg: Config) -> tuple:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    torch.manual_seed(cfg.seed)
    device = get_device()

    exp_dir = _make_experiment_dir(cfg)
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    log_path = exp_dir / "logs" / "train.jsonl"

    train_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    best_acc = 0.0
    epochs_no_improve = 0
    history: list[dict] = []
    clip_max_norm = cfg.liquid.grad_clip_max_norm

    epoch_bar = tqdm(range(cfg.epochs), desc="Epochs", unit="ep")

    with open(log_path, "a") as log_f:
        for epoch in epoch_bar:
            tau = get_tau(epoch, cfg)
            model.train()
            total_l = correct = n = 0
            epoch_grad_norm = 0.0
            n_batches = 0

            batch_bar = tqdm(train_loader, desc="  Train", leave=False, unit="batch")
            for x, y in batch_bar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                rates = model(x, tau=tau)
                loss = _compute_loss(rates, y, model, cfg)
                loss.backward()

                # NaN detection
                if torch.isnan(loss):
                    tqdm.write(f"  ✖ NaN loss detected at epoch {epoch+1}, batch {n_batches+1}. Stopping.")
                    return history, exp_dir

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=clip_max_norm
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
                lr=current_lr,
                tau=tau,
                train_loss=train_loss,
                train_acc=train_acc,
                test_acc=test_acc,
                sparsity=sparsity,
                theta_mean=theta_mean,
                theta_std=theta_std,
                grad_norm=avg_grad_norm,
                mean_firing_rate=fr_info["mean"],
                max_firing_rate=fr_info["max"],
            )
            history.append(row)
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

            epoch_bar.set_postfix(
                tau=f"{tau:.3f}",
                loss=f"{train_loss:.4f}",
                train=f"{train_acc:.4f}",
                test=f"{test_acc:.4f}",
                sp=f"{sparsity:.3f}",
            )
            tqdm.write(
                f"[{epoch+1:03d}/{cfg.epochs}] "
                f"lr={current_lr:.2e}  tau={tau:.3f}  loss={train_loss:.4f}  "
                f"train={train_acc:.4f}  test={test_acc:.4f}  "
                f"sp={sparsity:.3f}  grad={avg_grad_norm:.1f}  "
                f"fr={fr_info['mean']:.3f}/{fr_info['max']:.3f}  "
                f"θ={theta_mean:.3f}±{theta_std:.3f}"
            )

            # early warnings
            if avg_grad_norm > 100:
                tqdm.write(f"  ⚠ grad_norm={avg_grad_norm:.1f} — consider reducing lr or clip_max_norm")
            if fr_info["max"] > 0.9:
                tqdm.write(f"  ⚠ max_firing_rate={fr_info['max']:.3f} — possible excitatory loop runaway")
            if epoch > 20 and theta_std < 0.01:
                tqdm.write(f"  ⚠ theta_std={theta_std:.4f} — theta stagnating, consider increasing lambda_commit")

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
