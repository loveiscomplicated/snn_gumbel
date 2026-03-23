"""Inference and evaluation using the trained binary topology."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import cfg
from model import SNNModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device):
    model = SNNModel(
        n_input=cfg.n_input,
        n_hidden=cfg.n_hidden,
        n_output=cfg.n_output,
        T=cfg.T,
        beta=cfg.beta,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt.get("history", [])


def run_evaluation(checkpoint_path: str = None):
    if checkpoint_path is None:
        checkpoint_path = cfg.checkpoint_path

    device = get_device()
    model, history = load_model(checkpoint_path, device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),
        ]
    )
    test_ds = datasets.MNIST(
        cfg.data_dir, train=False, download=True, transform=transform
    )
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    # Use hard binary mask at inference (tau doesn't matter since hard=True)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            rates = model(x, tau=cfg.tau_end, hard=True)
            pred = rates.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    sp1, sp2 = model.sparsity_info()
    total_l1 = cfg.n_input * cfg.n_hidden
    total_l2 = cfg.n_hidden * cfg.n_output
    active_l1 = int(sp1 * total_l1)
    active_l2 = int(sp2 * total_l2)

    print(f"Test accuracy (binary topology): {acc:.4f}")
    print(f"Layer 1 active connections: {active_l1}/{total_l1}  ({sp1*100:.1f}%)")
    print(f"Layer 2 active connections: {active_l2}/{total_l2}  ({sp2*100:.1f}%)")
    print(
        f"Total active connections:   {active_l1+active_l2}/{total_l1+total_l2}  "
        f"({(active_l1+active_l2)/(total_l1+total_l2)*100:.1f}%)"
    )

    return acc, model, history


if __name__ == "__main__":
    run_evaluation()
