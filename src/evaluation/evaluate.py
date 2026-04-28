"""
Evaluation utilities.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils.config import Config


def _is_lsm(cfg: Config) -> bool:
    """Detect if config describes an LSM model."""
    return cfg.liquid.n_liquid > 0 and not cfg.architecture.hidden_layers


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, cfg: Config, device: torch.device):
    if _is_lsm(cfg):
        from src.lsm.trainer import build_model
    else:
        from src.training.trainer import build_model
    model = build_model(cfg, device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    return model, ckpt.get("history", [])


def run_evaluation(checkpoint_path: str, cfg: Config) -> tuple:
    device = get_device()
    model, history = load_model(checkpoint_path, cfg, device)
    model.eval()

    from src.data.loaders import get_dataloaders

    _, test_loader = get_dataloaders(cfg)

    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            rates = model(x, tau=cfg.tau_end, hard=True)
            pred = rates.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    sparsities = model.sparsity_info()

    print(f"Test accuracy: {acc:.4f}")
    print(
        "  " + "  ".join(f"sparsity_l{i+1}={s:.3f}" for i, s in enumerate(sparsities))
    )
    return acc, model, history
