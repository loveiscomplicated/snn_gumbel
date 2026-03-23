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
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
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


def _make_experiment_dir(cfg: Config, config_path: str | None) -> Path:
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    exp_dir = Path("experiments") / f"{cfg.experiment_name}_{timestamp}"
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "figures").mkdir(exist_ok=True)

    # snapshot the config file actually used
    if config_path and os.path.exists(config_path):
        shutil.copy(config_path, exp_dir / "config.yaml")
    else:
        # write a minimal snapshot from the dataclass
        import yaml
        with open(exp_dir / "config.yaml", "w") as f:
            yaml.dump(_cfg_to_dict(cfg), f, default_flow_style=False)

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


def train(cfg: Config, config_path: str | None = None, resume: bool = False) -> list:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    torch.manual_seed(cfg.seed)
    device = get_device()

    exp_dir = _make_experiment_dir(cfg, config_path)
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    log_path = exp_dir / "logs" / "train.jsonl"

    train_loader, test_loader = get_dataloaders(cfg)
    model = build_model(cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    start_epoch = 0
    best_acc = 0.0
    history: list[dict] = []

    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
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

            row = dict(
                epoch=epoch + 1,
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
                tau=f"{tau:.3f}",
                loss=f"{train_loss:.4f}",
                train=f"{train_acc:.4f}",
                test=f"{test_acc:.4f}",
            )
            tqdm.write(
                f"[{epoch+1:03d}/{cfg.epochs}] "
                f"tau={tau:.3f}  loss={train_loss:.4f}  "
                f"train={train_acc:.4f}  test={test_acc:.4f}  {sp_str}"
            )

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch + 1,
                        "best_acc": best_acc,
                        "history": history,
                    },
                    checkpoint_path,
                )

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Experiment saved to: {exp_dir}")
    return history, exp_dir
