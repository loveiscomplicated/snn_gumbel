"""Training loop for SNN with Gumbel-Sigmoid learnable topology."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from model import SNNModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_tau(epoch: int) -> float:
    """Cosine anneal tau from tau_start to tau_end over tau_anneal_epochs."""
    if epoch >= cfg.tau_anneal_epochs:
        return cfg.tau_end
    progress = epoch / cfg.tau_anneal_epochs
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.tau_end + (cfg.tau_start - cfg.tau_end) * cosine


def build_dataloaders():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten to [784]
            transforms.Lambda(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
            ),  # to [0,1]
        ]
    )
    train_ds = datasets.MNIST(
        cfg.data_dir, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        cfg.data_dir, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, test_loader


def evaluate(model, loader, device, tau):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="  Eval", leave=False, unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            rates = model(x, tau=tau, hard=True)
            pred = rates.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(acc=f"{correct/total:.4f}")
    return correct / total


def train(resume: bool = False):
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device}")

    train_loader, test_loader = build_dataloaders()

    model = SNNModel(
        n_input=cfg.n_input,
        n_hidden=cfg.n_hidden,
        n_output=cfg.n_output,
        T=cfg.T,
        beta=cfg.beta,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    ce_loss = nn.CrossEntropyLoss()

    start_epoch = 0
    best_acc = 0.0
    history = []

    if resume:
        if os.path.exists(cfg.checkpoint_path):
            ckpt = torch.load(cfg.checkpoint_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"]
            best_acc = ckpt["best_acc"]
            history = ckpt.get("history", [])
            print(f"Resumed from epoch {start_epoch}/{cfg.epochs}, best_acc={best_acc:.4f}")
        else:
            print(f"Checkpoint not found at {cfg.checkpoint_path}, starting from scratch.")

    epoch_pbar = tqdm(range(start_epoch, cfg.epochs), desc="Training", unit="epoch",
                      initial=start_epoch, total=cfg.epochs)
    for epoch in epoch_pbar:
        tau = get_tau(epoch)
        model.train()

        total_loss = correct = total = 0
        batch_pbar = tqdm(train_loader, desc=f"  Train", leave=False, unit="batch")
        for x, y in batch_pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            rates = model(x, tau=tau, hard=False)

            loss = (ce_loss(rates, y)
                    + cfg.lambda_sparse * model.sparsity_loss()
                    + cfg.lambda_commit * model.commitment_loss())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (rates.argmax(1) == y).sum().item()
            total += y.size(0)
            batch_pbar.set_postfix(
                loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}"
            )

        train_acc = correct / total
        train_loss = total_loss / total
        test_acc = evaluate(model, test_loader, device, tau)
        sp1, sp2 = model.sparsity_info()

        epoch_pbar.set_postfix(
            tau=f"{tau:.3f}",
            loss=f"{train_loss:.4f}",
            train=f"{train_acc:.4f}",
            test=f"{test_acc:.4f}",
            sp1=f"{sp1:.3f}",
            sp2=f"{sp2:.3f}",
        )
        tqdm.write(
            f"Epoch {epoch+1:3d}/{cfg.epochs} | tau={tau:.3f} | "
            f"loss={train_loss:.4f} | train={train_acc:.4f} | test={test_acc:.4f} | "
            f"sparsity L1={sp1:.3f} L2={sp2:.3f}"
        )

        history.append(
            dict(
                epoch=epoch + 1,
                tau=tau,
                train_loss=train_loss,
                train_acc=train_acc,
                test_acc=test_acc,
                sparsity_l1=sp1,
                sparsity_l2=sp2,
            )
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
                cfg.checkpoint_path,
            )

    print(f"\nBest test accuracy: {best_acc:.4f}")
    print(f"Checkpoint saved to {cfg.checkpoint_path}")
    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    train(resume=args.resume)
