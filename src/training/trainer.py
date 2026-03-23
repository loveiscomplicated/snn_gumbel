"""
Training loop.

Responsibilities:
  - Create experiment directory: experiments/{name}_{YYMMDDHHmm}/
  - Save config snapshot to that directory
  - Train for cfg.epochs, checkpointing best model
  - Stream metrics to logs/train.jsonl
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.loaders import get_dataloaders
from src.models.snn import SNNModel
from src.training.losses import total_loss
from src.utils.config import Config


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

    # snapshot: always dump the fully resolved config (after inheritance + CLI overrides)
    import yaml
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(_cfg_to_dict(cfg), f, default_flow_style=False, sort_keys=False)

    return exp_dir


def _cfg_to_dict(cfg: Config) -> dict:
    import dataclasses
    return dataclasses.asdict(cfg)


def _evaluate(model: SNNModel, loader, device: torch.device, tau: float) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            rates = model(x, tau=tau, hard=True)
            pred = rates.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def build_model(cfg: Config, device: torch.device) -> SNNModel:
    model = SNNModel(
        n_input=cfg.n_input,
        hidden_layers=cfg.architecture.hidden_layers,
        n_output=cfg.n_output,
        T=cfg.T,
        beta=cfg.beta,
        topology_mode=cfg.topology.mode,
        target_sparsity=cfg.topology.target_sparsity,
    ).to(device)

    if cfg.topology.mode == "transfer" and cfg.topology.transfer_from:
        model.load_topology_from_checkpoint(cfg.topology.transfer_from, device)

    return model


def train(cfg: Config, resume: bool = False) -> list:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    torch.manual_seed(cfg.seed)
    device = get_device()

    exp_dir = _make_experiment_dir(cfg)
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    log_path = exp_dir / "logs" / "train.jsonl"

    train_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    start_epoch = 0
    best_acc = 0.0
    history: list[dict] = []

    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_acc"]
        history = ckpt.get("history", [])

    epoch_bar = tqdm(range(start_epoch, cfg.epochs), desc="Epochs", unit="ep")

    with open(log_path, "a") as log_f:
        for epoch in epoch_bar:
            tau = get_tau(epoch, cfg)
            model.train()
            total_l = correct = n = 0

            batch_bar = tqdm(train_loader, desc=f"  Train", leave=False, unit="batch")
            for x, y in batch_bar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                rates = model(x, tau=tau, hard=False)
                loss = total_loss(rates, y, model, cfg.lambda_sparse, cfg.lambda_commit)
                loss.backward()
                optimizer.step()
                total_l += loss.item() * y.size(0)
                correct += (rates.argmax(1) == y).sum().item()
                n += y.size(0)
                batch_bar.set_postfix(loss=f"{total_l/n:.4f}", acc=f"{correct/n:.4f}")

            train_acc = correct / n
            train_loss = total_l / n
            test_acc = _evaluate(model, test_loader, device, tau)
            sparsities = model.sparsity_info()
            current_lr = scheduler.get_last_lr()[0]

            scheduler.step()

            row = dict(
                epoch=epoch + 1,
                lr=current_lr,
                tau=tau,
                train_loss=train_loss,
                train_acc=train_acc,
                test_acc=test_acc,
                **{f"sparsity_l{i+1}": s for i, s in enumerate(sparsities)},
            )
            history.append(row)

            # JSON lines log
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()

            sp_str = "  ".join(f"sp{i+1}={s:.3f}" for i, s in enumerate(sparsities))
            epoch_bar.set_postfix(
                lr=f"{current_lr:.2e}",
                tau=f"{tau:.3f}",
                loss=f"{train_loss:.4f}",
                train=f"{train_acc:.4f}",
                test=f"{test_acc:.4f}",
            )
            tqdm.write(
                f"[{epoch+1:03d}/{cfg.epochs}] "
                f"lr={current_lr:.2e}  tau={tau:.3f}  loss={train_loss:.4f}  "
                f"train={train_acc:.4f}  test={test_acc:.4f}  {sp_str}"
            )

            if test_acc > best_acc:
                best_acc = test_acc
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

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Experiment saved to: {exp_dir}")
    return history, exp_dir
