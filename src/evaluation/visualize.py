"""
Visualisation utilities.
All plot functions accept an explicit save_path.
run_all() saves figures into the experiment figures/ directory.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.snn import SNNModel


def plot_training_curves(history: list, save_path: str):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_acc"] for h in history], label="Train")
    axes[0].plot(epochs, [h["test_acc"]  for h in history], label="Test")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, [h["train_loss"] for h in history], color="red")
    ax2 = axes[1].twinx()
    ax2.plot(epochs, [h["tau"] for h in history], color="gray", linestyle="--")
    axes[1].set_title("Loss & Temperature"); axes[1].grid(True)

    # collect all sparsity keys in order
    sp_keys = sorted(k for k in history[0] if k.startswith("sparsity_l"))
    for key in sp_keys:
        label = key.replace("sparsity_", "Layer ")
        axes[2].plot(epochs, [h[key] * 100 for h in history], label=label)
    axes[2].set_title("Edge Sparsity (%)"); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_topology(model: SNNModel, save_path: str):
    n = len(model.layers)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    cmaps = ["Blues", "Oranges", "Greens", "Purples"]
    for i, layer in enumerate(model.layers):
        with torch.no_grad():
            mask = layer.get_binary_mask().cpu().numpy()
        axes[i].imshow(mask[:64].T, aspect="auto", cmap=cmaps[i % len(cmaps)])
        axes[i].set_title(f"Layer {i+1} (first 64 inputs)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_theta_distribution(model: SNNModel, save_path: str):
    n = len(model.layers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    colors = ["steelblue", "darkorange", "mediumseagreen", "mediumpurple"]
    for i, layer in enumerate(model.layers):
        with torch.no_grad():
            probs = torch.sigmoid(layer.theta).cpu().numpy().ravel()
        axes[i].hist(probs, bins=50, color=colors[i % len(colors)])
        axes[i].axvline(0.5, color="red", linestyle="--")
        axes[i].set_title(f"Layer {i+1} σ(θ)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_threshold_distribution(model: SNNModel, save_path: str):
    n = len(model.layers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for i, layer in enumerate(model.layers):
        with torch.no_grad():
            thr = layer.threshold.cpu().numpy()
        if i < n - 1:
            axes[i].hist(thr, bins=30, color="mediumseagreen")
            axes[i].set_title(f"Layer {i+1} thresholds (hidden)")
        else:
            axes[i].bar(range(len(thr)), thr, color="salmon")
            axes[i].set_title(f"Layer {i+1} thresholds (output)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_input_connectivity(model: SNNModel, save_path: str):
    """Visualise first layer's input→hidden connectivity as a 28×28 heatmap."""
    with torch.no_grad():
        mask1 = model.layers[0].get_binary_mask().cpu().numpy()

    input_degree = mask1.sum(axis=1).reshape(28, 28)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(input_degree, cmap="hot")
    axes[0].set_title("Input pixel connectivity")
    axes[1].hist(mask1.sum(axis=0), bins=30, color="coral")
    axes[1].set_title("Hidden neuron in-degree")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def run_all(checkpoint_path: str, cfg, figures_dir: str | None = None):
    from src.evaluation.evaluate import load_model
    from src.training.trainer import get_device

    device = get_device()
    model, history = load_model(checkpoint_path, cfg, device)
    model.eval()

    save_dir = Path(figures_dir) if figures_dir else Path("figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    if history:
        plot_training_curves(history, str(save_dir / "training_curves.png"))
    plot_topology(model,              str(save_dir / "topology.png"))
    plot_theta_distribution(model,    str(save_dir / "theta_distribution.png"))
    plot_threshold_distribution(model, str(save_dir / "threshold_distribution.png"))
    plot_input_connectivity(model,    str(save_dir / "input_receptive_field.png"))
