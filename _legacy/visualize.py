"""Visualisation: topology, weight distributions, training curves, threshold distributions."""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import cfg
from evaluate import load_model, get_device


def plot_training_curves(history: list, save_path: str = "training_curves.png"):
    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    test_acc = [h["test_acc"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    sp1 = [h["sparsity_l1"] for h in history]
    sp2 = [h["sparsity_l2"] for h in history]
    tau = [h["tau"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_acc, label="Train")
    axes[0].plot(epochs, test_acc, label="Test")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_loss, color="red")
    ax2 = axes[1].twinx()
    ax2.plot(epochs, tau, color="gray", linestyle="--", label="tau")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    ax2.set_ylabel("tau")
    axes[1].set_title("Loss & Temperature")
    axes[1].grid(True)

    axes[2].plot(epochs, [s * 100 for s in sp1], label="Layer 1")
    axes[2].plot(epochs, [s * 100 for s in sp2], label="Layer 2")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Active edges (%)")
    axes[2].set_title("Edge Sparsity")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def plot_topology(model, save_path: str = "topology.png"):
    """
    Visualise binary edge masks as heatmaps.
    Layer 1: [784, 256] — too large to show all, so we sample 64 pre neurons.
    Layer 2: [256, 10]  — shown in full.
    """
    with torch.no_grad():
        mask1 = (torch.sigmoid(model.layer1.theta.cpu()) > 0.5).float().numpy()
        mask2 = (torch.sigmoid(model.layer2.theta.cpu()) > 0.5).float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Layer 1: show first 64 input neurons (e.g. top-left 8x8 patch)
    idx = np.arange(0, min(64, mask1.shape[0]))
    im1 = axes[0].imshow(
        mask1[idx].T, aspect="auto", cmap="Blues", interpolation="nearest"
    )
    axes[0].set_title(f"Layer 1 topology\n(first {len(idx)} inputs × all hidden)")
    axes[0].set_xlabel("Input neuron index")
    axes[0].set_ylabel("Hidden neuron index")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        mask2.T, aspect="auto", cmap="Oranges", interpolation="nearest"
    )
    axes[1].set_title("Layer 2 topology\n(all hidden × output)")
    axes[1].set_xlabel("Hidden neuron index")
    axes[1].set_ylabel("Output neuron (digit)")
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels(list(range(10)))
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def plot_theta_distribution(model, save_path: str = "theta_distribution.png"):
    """Histogram of sigmoid(theta) — distribution of connection probabilities."""
    with torch.no_grad():
        probs1 = torch.sigmoid(model.layer1.theta).cpu().numpy().ravel()
        probs2 = torch.sigmoid(model.layer2.theta).cpu().numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(probs1, bins=50, color="steelblue", edgecolor="k", linewidth=0.3)
    axes[0].axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
    axes[0].set_title("Layer 1 — connection probability σ(θ)")
    axes[0].set_xlabel("p(edge exists)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(probs2, bins=50, color="darkorange", edgecolor="k", linewidth=0.3)
    axes[1].axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
    axes[1].set_title("Layer 2 — connection probability σ(θ)")
    axes[1].set_xlabel("p(edge exists)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def plot_threshold_distribution(model, save_path: str = "threshold_distribution.png"):
    """Distribution of learned per-neuron firing thresholds."""
    with torch.no_grad():
        thr1 = model.layer1.threshold.cpu().numpy()
        thr2 = model.layer2.threshold.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(thr1, bins=30, color="mediumseagreen", edgecolor="k", linewidth=0.3)
    axes[0].set_title("Hidden layer — firing thresholds")
    axes[0].set_xlabel("Threshold value")
    axes[0].set_ylabel("Count")

    axes[1].bar(range(10), thr2, color="salmon", edgecolor="k")
    axes[1].set_title("Output layer — firing thresholds (per digit)")
    axes[1].set_xlabel("Digit")
    axes[1].set_ylabel("Threshold value")
    axes[1].set_xticks(range(10))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def plot_input_connectivity(model, save_path: str = "input_receptive_field.png"):
    """
    For each hidden neuron, count how many input pixels it receives.
    Reshape the input-side mask to 28x28 aggregated map.
    """
    with torch.no_grad():
        mask1 = (torch.sigmoid(model.layer1.theta.cpu()) > 0.5).float().numpy()

    # [784, n_hidden] -> sum over hidden -> [784] -> [28, 28]
    input_degree = mask1.sum(axis=1).reshape(28, 28)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(input_degree, cmap="hot")
    axes[0].set_title(
        "Input pixel connectivity\n(# hidden neurons each pixel connects to)"
    )
    plt.colorbar(im, ax=axes[0])

    # Hidden neuron degree distribution
    hidden_degree = mask1.sum(axis=0)
    axes[1].hist(hidden_degree, bins=30, color="coral", edgecolor="k", linewidth=0.3)
    axes[1].set_title("Hidden neuron in-degree distribution")
    axes[1].set_xlabel("# input connections")
    axes[1].set_ylabel("# hidden neurons")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.show()


def run_all(checkpoint_path: str = None):
    if checkpoint_path is None:
        checkpoint_path = cfg.checkpoint_path

    device = get_device()
    model, history = load_model(checkpoint_path, device)
    model.eval()

    cur_dir = os.path.dirname(__file__)
    save_dir = os.path.join(cur_dir, "resources")
    plot_training_curves_path = os.path.join(save_dir, "training_curves.png")
    plot_topology_path = os.path.join(save_dir, "topology.png")
    plot_theta_distribution_path = os.path.join(save_dir, "theta_distribution.png")
    plot_threshold_path = os.path.join(save_dir, "threshold_distribution.png")
    plot_input_connectivity_path = os.path.join(save_dir, "input_receptive_field.png")

    if history:
        plot_training_curves(history, plot_training_curves_path)
    plot_topology(model, plot_topology_path)
    plot_theta_distribution(model, plot_theta_distribution_path)
    plot_threshold_distribution(model, plot_threshold_path)
    plot_input_connectivity(model, plot_input_connectivity_path)


if __name__ == "__main__":
    run_all()
