"""
Liquid layer diagnostic script.

Checks:
  1. Beta / threshold heterogeneity (are per-neuron values actually different?)
  2. Separation property (do different-class inputs produce different spike vectors?)
  3. Broadcasting sanity (beta shape vs liquid_mem shape)

Usage:
    python scripts/diagnose_liquid.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=fixed
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.lsm.trainer import build_model, get_device
from src.data.loaders import get_dataloaders


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/lsm_shd_baseline.yaml"
    overrides = sys.argv[2:] if len(sys.argv) > 2 else []
    cfg = load_config(config_path, overrides)

    device = get_device()
    model = build_model(cfg, device)
    model.eval()

    # ------------------------------------------------------------------
    # 1. Parameter heterogeneity
    # ------------------------------------------------------------------
    beta = model.liquid.beta.detach().cpu()
    threshold = model.liquid.threshold.detach().cpu()

    print("=" * 60)
    print("[1] Parameter heterogeneity")
    print(
        f"  beta      : min={beta.min():.4f}  max={beta.max():.4f}  std={beta.std():.4f}"
    )
    print(f"              first 5: {beta[:5].tolist()}")
    print(
        f"  threshold : min={threshold.min():.4f}  max={threshold.max():.4f}  std={threshold.std():.4f}"
    )
    print(f"              first 5: {threshold[:5].tolist()}")
    print(f"  beta shape: {beta.shape}  (should be ({cfg.liquid.n_liquid},))")

    # ------------------------------------------------------------------
    # 2. Separation property: spike vectors from different classes
    # ------------------------------------------------------------------
    print("\n[2] Separation property")
    _, test_loader = get_dataloaders(cfg)

    # collect one batch of samples, group by label
    samples_by_class: dict[int, list] = {}
    with torch.no_grad():
        for x, y in test_loader:
            for xi, yi in zip(x, y):
                label = yi.item()
                if label not in samples_by_class:
                    samples_by_class[label] = []
                if len(samples_by_class[label]) < 8:
                    samples_by_class[label].append(xi)
            if all(
                len(v) >= 8
                for v in samples_by_class.values()
                if len(samples_by_class) >= 5
            ):
                break

    # pick 5 classes with enough samples
    chosen = sorted(samples_by_class.keys())[:5]
    tau = cfg.tau_end

    # compute mean spike-rate vector per class
    class_vecs = {}
    model.liquid.sample_mask(tau=tau, hard=True)
    with torch.no_grad():
        for cls in chosen:
            batch = torch.stack(samples_by_class[cls]).to(device)  # (8, T, n_input)
            # run liquid manually to get spike_sum
            n_liquid = cfg.liquid.n_liquid
            T = cfg.T
            liq_mem = torch.zeros(len(batch), n_liquid, device=device)
            liq_spk = torch.zeros(len(batch), n_liquid, device=device)
            spike_sum = torch.zeros(len(batch), n_liquid, device=device)

            from src.models.layers import spike_fn

            for t in range(T):
                inp = model.input_proj(batch[:, t])
                rec = model.liquid(liq_spk)
                liq_mem = model.liquid.beta * liq_mem + inp + rec
                liq_mem = torch.clamp(liq_mem, -3.0, 3.0)
                liq_spk = spike_fn(liq_mem - model.liquid.threshold.clamp(min=0.01))
                liq_mem = liq_mem * (1.0 - liq_spk)
                spike_sum += liq_spk

            mean_rate = (spike_sum / T).mean(dim=0).cpu()  # (n_liquid,)
            class_vecs[cls] = mean_rate

    # intra-class cosine similarity (8 samples → mean vec; compare mean vecs across classes)
    print(f"  Classes analysed: {chosen}")
    print(f"  Mean firing rate per class (mean over neurons):")
    for cls in chosen:
        v = class_vecs[cls]
        print(
            f"    class {cls:2d}: mean={v.mean():.4f}  std={v.std():.4f}  "
            f"active_neurons(>0.05)={( v > 0.05).sum().item()}"
        )

    print(f"\n  Pairwise cosine similarity between class mean-rate vectors:")
    keys = list(class_vecs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = class_vecs[keys[i]]
            b = class_vecs[keys[j]]
            sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            l2 = (a - b).norm().item()
            print(f"    class {keys[i]} vs {keys[j]}: cosine={sim:.4f}  L2={l2:.4f}")

    # ------------------------------------------------------------------
    # 3. Per-neuron firing rate distribution
    # ------------------------------------------------------------------
    print("\n[3] Per-neuron firing rate distribution (class 0 vs class 1)")
    if len(chosen) >= 2:
        v0 = class_vecs[chosen[0]]
        v1 = class_vecs[chosen[1]]
        diff = (v0 - v1).abs()
        print(
            f"  |rate_diff| per neuron: mean={diff.mean():.4f}  max={diff.max():.4f}  "
            f"neurons_with_diff>0.1: {(diff > 0.1).sum().item()}"
        )
        # top 5 most discriminative neurons
        top5 = diff.topk(5)
        print(f"  Top-5 discriminative neuron indices & |diff|:")
        for idx, val in zip(top5.indices.tolist(), top5.values.tolist()):
            print(
                f"    neuron {idx:4d}: |diff|={val:.4f}  "
                f"class{chosen[0]}={v0[idx]:.4f}  class{chosen[1]}={v1[idx]:.4f}"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()
