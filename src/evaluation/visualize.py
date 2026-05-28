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


# ===========================================================================
# Feedforward SNN plots
# ===========================================================================


def plot_training_curves(history: list, save_path: str):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_acc"] for h in history], label="Train")
    axes[0].plot(epochs, [h["test_acc"] for h in history], label="Test")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, [h["train_loss"] for h in history], color="red")
    ax2 = axes[1].twinx()
    ax2.plot(epochs, [h["tau"] for h in history], color="gray", linestyle="--")
    axes[1].set_title("Loss & Temperature")
    axes[1].grid(True)

    # collect all sparsity keys in order
    sp_keys = sorted(k for k in history[0] if k.startswith("sparsity_l"))
    for key in sp_keys:
        label = key.replace("sparsity_", "Layer ")
        axes[2].plot(epochs, [h[key] * 100 for h in history], label=label)
    axes[2].set_title("Edge Sparsity (%)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_topology(model, save_path: str):
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


def plot_theta_distribution(model, save_path: str):
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


def plot_threshold_distribution(model, save_path: str):
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


def plot_input_connectivity(model, save_path: str):
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


# ===========================================================================
# LSM-specific plots
# ===========================================================================


def _history_metric(row: dict, new_key: str, old_key: str, default=0):
    if new_key in row:
        return row.get(new_key, default)
    return row.get(old_key, default)


def lsm_plot_training_curves(history: list, save_path: str):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    has_val = any(h.get("val_acc") is not None for h in history)

    # Row 1: accuracy, loss+tau, sparsity
    axes[0, 0].plot(epochs, [h["train_acc"] for h in history], label="Train")
    if has_val:
        axes[0, 0].plot(
            epochs,
            [h.get("val_acc", np.nan) if h.get("val_acc") is not None else np.nan for h in history],
            label="Val",
        )
    axes[0, 0].plot(epochs, [h["test_acc"] for h in history], label="Test")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, [h["train_loss"] for h in history], color="red")
    ax_tau = axes[0, 1].twinx()
    ax_tau.plot(
        epochs, [h["tau"] for h in history], color="gray", linestyle="--", label="tau"
    )
    axes[0, 1].set_title("Loss & Temperature")
    axes[0, 1].grid(True)

    axes[0, 2].plot(epochs, [h["sparsity"] * 100 for h in history], color="teal")
    axes[0, 2].set_title("Liquid Sparsity (%)")
    axes[0, 2].grid(True)

    # Row 2: grad_norm, firing rates, topology logit stats
    axes[1, 0].plot(epochs, [h.get("grad_norm", 0) for h in history], color="purple")
    axes[1, 0].set_title("Grad Norm")
    axes[1, 0].set_yscale("symlog")
    axes[1, 0].grid(True)

    axes[1, 1].plot(
        epochs, [h.get("mean_firing_rate", 0) for h in history], label="Mean"
    )
    axes[1, 1].plot(epochs, [h.get("max_firing_rate", 0) for h in history], label="Max")
    axes[1, 1].set_title("Firing Rates")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    logit_mean = [
        _history_metric(h, "topology_logit_mean", "theta_mean", 0) for h in history
    ]
    logit_std = [
        _history_metric(h, "topology_logit_std", "theta_std", 0) for h in history
    ]
    axes[1, 2].plot(epochs, logit_mean, label="mean")
    axes[1, 2].fill_between(
        epochs,
        [mean - std for mean, std in zip(logit_mean, logit_std)],
        [mean + std for mean, std in zip(logit_mean, logit_std)],
        alpha=0.3,
    )
    axes[1, 2].set_title("Topology Logit (mean +/- std)")
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def lsm_plot_topology(model, save_path: str):
    """Visualise liquid recurrent connectivity mask."""
    with torch.no_grad():
        mask = model.liquid.get_binary_mask().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(mask, aspect="auto", cmap="Blues")
    axes[0].set_title(f"Liquid Mask ({mask.shape[0]}x{mask.shape[1]})")
    axes[0].set_xlabel("Post-synaptic")
    axes[0].set_ylabel("Pre-synaptic")

    # degree distributions
    in_degree = mask.sum(axis=0)
    out_degree = mask.sum(axis=1)
    axes[1].hist(
        in_degree, bins=30, alpha=0.6, label=f"In-degree (mean={in_degree.mean():.1f})"
    )
    axes[1].hist(
        out_degree,
        bins=30,
        alpha=0.6,
        label=f"Out-degree (mean={out_degree.mean():.1f})",
    )
    axes[1].set_title("Degree Distribution")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def lsm_plot_theta_distribution(model, save_path: str):
    """Visualise sigma(theta) distribution for the liquid layer."""
    with torch.no_grad():
        probs = torch.sigmoid(model.liquid.get_theta()).cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
    ax.set_title(f"Liquid σ(θ) distribution (N={len(probs)})")
    ax.set_xlabel("σ(θ)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def lsm_plot_threshold_distribution(model, save_path: str):
    """Visualise learned neuron thresholds and beta (membrane decay)."""
    with torch.no_grad():
        thr = model.liquid.threshold.cpu().numpy()
        beta = model.liquid.beta.cpu().numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(thr, bins=30, color="mediumseagreen", edgecolor="white")
    axes[0].set_title(f"Threshold (mean={thr.mean():.3f})")
    axes[0].grid(True)

    axes[1].hist(beta, bins=30, color="coral", edgecolor="white")
    axes[1].set_title(f"Beta / membrane decay (mean={beta.mean():.3f})")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def lsm_plot_weight_distribution(model, save_path: str):
    """Visualise effective weight magnitude distribution."""
    import torch.nn.functional as F

    with torch.no_grad():
        w_mag = F.softplus(model.liquid.w_raw).cpu().numpy().ravel()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(w_mag, bins=80, color="darkorange", edgecolor="white", linewidth=0.3)
    ax.set_title(f"softplus(w_raw) distribution (mean={w_mag.mean():.4f})")
    ax.set_xlabel("Weight magnitude")
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# ===========================================================================
# Dispatcher
# ===========================================================================


def run_all(checkpoint_path: str, cfg, figures_dir: str | None = None):
    from src.evaluation.evaluate import load_model, get_device

    device = get_device()
    model, history = load_model(checkpoint_path, cfg, device)
    model.eval()

    save_dir = Path(figures_dir) if figures_dir else Path("figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    is_lsm = hasattr(model, "liquid")

    if is_lsm:
        if history:
            lsm_plot_training_curves(history, str(save_dir / "training_curves.png"))
        lsm_plot_topology(model, str(save_dir / "topology.png"))
        lsm_plot_theta_distribution(model, str(save_dir / "theta_distribution.png"))
        lsm_plot_threshold_distribution(
            model, str(save_dir / "threshold_distribution.png")
        )
        lsm_plot_weight_distribution(model, str(save_dir / "weight_distribution.png"))
    else:
        if history:
            plot_training_curves(history, str(save_dir / "training_curves.png"))
        plot_topology(model, str(save_dir / "topology.png"))
        plot_theta_distribution(model, str(save_dir / "theta_distribution.png"))
        plot_threshold_distribution(model, str(save_dir / "threshold_distribution.png"))
        plot_input_connectivity(model, str(save_dir / "input_receptive_field.png"))
