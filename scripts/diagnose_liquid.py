"""
Liquid layer diagnostic script.

Checks:
  1. Config/model sanity
  2. Recurrent sparsity
  3. SHD input spike statistics
  4. Input/recurrent current scale
  5. Firing rates
  6. Class separation

Usage:
    python scripts/diagnose_liquid.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=fixed
    python scripts/diagnose_liquid.py experiments/<exp>/config.yaml --checkpoint experiments/<exp>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import get_dataloaders
from src.lsm.trainer import build_model, get_device
from src.models.layers import spike_fn
from src.utils.config import load_config


class RunningStats:
    """Streaming scalar stats for tensor values."""

    def __init__(self):
        self.n = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.abs_sum = 0.0
        self.max_abs = 0.0

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().cpu()
        flat = x.reshape(-1)
        if flat.numel() == 0:
            return
        self.n += flat.numel()
        self.sum += flat.sum().item()
        self.sum_sq += (flat * flat).sum().item()
        abs_flat = flat.abs()
        self.abs_sum += abs_flat.sum().item()
        self.max_abs = max(self.max_abs, abs_flat.max().item())

    def as_dict(self) -> dict[str, float]:
        if self.n == 0:
            return {"mean": 0.0, "std": 0.0, "abs_mean": 0.0, "max_abs": 0.0}
        mean = self.sum / self.n
        var = max(self.sum_sq / self.n - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": var**0.5,
            "abs_mean": self.abs_sum / self.n,
            "max_abs": self.max_abs,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose LSM liquid dynamics")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/lsm_shd_baseline.yaml",
        help="Config YAML path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path to load before diagnosis",
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=5,
        help="Number of classes to include in separation diagnostics",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=8,
        help="Samples per class for separation diagnostics",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=4,
        help="Number of test batches for input/current/firing diagnostics",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = [item for item in overrides if item != "--"]
    return args


def load_checkpoint_if_requested(model, checkpoint_path: str | None, device) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if missing:
        print(f"  missing keys: {len(missing)}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")


def fmt_stats(stats: dict[str, float]) -> str:
    return (
        f"mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
        f"abs_mean={stats['abs_mean']:.4f}  max_abs={stats['max_abs']:.4f}"
    )


def print_header(title: str) -> None:
    print(f"\n[{title}]")


def print_config_summary(cfg, model, checkpoint_path: str | None) -> None:
    print("=" * 72)
    print("[0] Config / model summary")
    print(f"  dataset       : {cfg.dataset}")
    print(f"  n_input/output: {cfg.n_input} -> {cfg.n_output}")
    print(f"  T             : {cfg.T}")
    print(f"  n_liquid      : {cfg.liquid.n_liquid}")
    print(f"  recurrent mode: {cfg.liquid.recurrent_mode}")
    print(f"  recurrent p   : {cfg.liquid.recurrent_sparsity}")
    print(f"  p_input       : {cfg.liquid.p_input}")
    print(f"  input scale   : {cfg.liquid.input_weight_scale}")
    print(
        f"  w_raw init    : mean={cfg.liquid.w_raw_init_mean} std={cfg.liquid.w_raw_init_std}"
    )
    print(f"  train w_raw   : {cfg.liquid.train_w_raw}")
    print(f"  w_raw_max     : {cfg.liquid.w_raw_max}")
    print(f"  seed          : {cfg.seed}")
    print(f"  checkpoint    : {checkpoint_path or '(none; initialized model)'}")
    print(f"  model class   : {model.__class__.__name__}")


def print_parameter_sanity(cfg, model) -> None:
    print_header("1. Parameter sanity")
    beta = model.liquid.beta.detach().cpu()
    threshold = model.liquid.threshold.detach().cpu()
    print(
        f"  beta      : min={beta.min():.4f}  max={beta.max():.4f}  "
        f"mean={beta.mean():.4f}  std={beta.std():.4f}"
    )
    print(
        f"  threshold : min={threshold.min():.4f}  max={threshold.max():.4f}  "
        f"mean={threshold.mean():.4f}  std={threshold.std():.4f}"
    )
    print(
        f"  beta shape       : {tuple(beta.shape)}  expected=({cfg.liquid.n_liquid},)"
    )
    print(
        f"  dale_sign shape  : {tuple(model.liquid.dale_sign.shape)}  "
        f"expected=({cfg.liquid.n_liquid}, 1)"
    )
    diag_sum = model.liquid.self_conn_mask.diag().sum().item()
    print(f"  self diag sum    : {diag_sum:.1f}")
    print(f"  theta trainable  : {model.liquid.theta.requires_grad}")
    print(f"  w_raw trainable  : {model.liquid.w_raw.requires_grad}")
    print(f"  beta trainable   : {model.liquid.logit_beta.requires_grad}")
    print(f"  threshold trainable: {model.liquid.threshold.requires_grad}")


def print_recurrent_sparsity(model, tau: float) -> None:
    print_header("2. Recurrent sparsity")
    model.liquid.sample_mask(tau=tau)
    mask = model.liquid.get_binary_mask().detach().cpu()
    n_total = mask.numel()
    n_active = int(mask.sum().item())
    diag_active = int(mask.diag().sum().item())
    print(f"  density     : {n_active / n_total:.4f}")
    print(f"  active edges: {n_active} / {n_total}")
    print(f"  self edges  : {diag_active}")
    if model.liquid.mode == "learned":
        probs = torch.sigmoid(model.liquid.theta.detach().cpu())
        print(
            f"  sigma(theta): mean={probs.mean():.4f}  std={probs.std():.4f}  "
            f"min={probs.min():.4f}  max={probs.max():.4f}"
        )
    print_recurrent_weight_stats(model, mask)


def tensor_summary(x: torch.Tensor) -> str:
    x = x.detach().float().cpu().reshape(-1)
    if x.numel() == 0:
        return "mean=0.0000  std=0.0000  min=0.0000  max=0.0000"
    std = x.std().item() if x.numel() > 1 else 0.0
    return (
        f"mean={x.mean().item():.4f}  std={std:.4f}  "
        f"min={x.min().item():.4f}  max={x.max().item():.4f}"
    )


def print_recurrent_weight_stats(model, binary_mask: torch.Tensor) -> None:
    print("\n  recurrent weight magnitude:")
    print(
        "  note: w_raw_max is an upper clamp; it does not raise weights below the cap."
    )
    with torch.no_grad():
        w_raw = model.liquid.w_raw.detach().cpu()
        w_clamped = torch.clamp(w_raw, max=model.liquid.w_raw_max)
        w_mag = F.softplus(w_raw)
        w_clamped_mag = F.softplus(w_clamped)
        active = (binary_mask * model.liquid.self_conn_mask.detach().cpu()).bool()
        dale = model.liquid.dale_sign.detach().cpu().reshape(-1)
        exc_pre = dale > 0
        inh_pre = dale < 0
        exc_active = active & exc_pre[:, None]
        inh_active = active & inh_pre[:, None]
        active_mag = w_clamped_mag[active]
        exc_mag = w_clamped_mag[exc_active]
        inh_mag = w_clamped_mag[inh_active]
        clamped_fraction = (w_raw > model.liquid.w_raw_max).float().mean().item()
        exc_edges = int(exc_active.sum().item())
        inh_edges = int(inh_active.sum().item())
        active_edges = max(exc_edges + inh_edges, 1)
        exc_in_degree = exc_active.float().sum(dim=0)
        inh_in_degree = inh_active.float().sum(dim=0)

    print(f"  w_raw                    : {tensor_summary(w_raw)}")
    print(f"  w_raw_clamped            : {tensor_summary(w_clamped)}")
    print(f"  softplus(w_raw)          : {tensor_summary(w_mag)}")
    print(f"  softplus(w_raw_clamped)  : {tensor_summary(w_clamped_mag)}")
    print(f"  effective nonzero |w|    : {tensor_summary(active_mag)}")
    print(f"  clamped fraction         : {clamped_fraction:.4f}")

    print("\n  recurrent E/I balance:")
    print(
        f"  active E/I edges         : exc={exc_edges} ({exc_edges / active_edges:.3f})  "
        f"inh={inh_edges} ({inh_edges / active_edges:.3f})"
    )
    print(f"  exc |w| on active edges  : {tensor_summary(exc_mag)}")
    print(f"  inh |w| on active edges  : {tensor_summary(inh_mag)}")
    print(f"  incoming exc degree/post : {tensor_summary(exc_in_degree)}")
    print(f"  incoming inh degree/post : {tensor_summary(inh_in_degree)}")


def collect_batches(loader, n_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        batches.append((x, y))
    return batches


def print_input_spike_stats(batches: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    print_header("3. SHD input spike statistics")
    if not batches:
        print("  no batches collected")
        return
    x_all = torch.cat([x for x, _ in batches], dim=0).float()
    sample_spikes = x_all.sum(dim=(1, 2))
    timestep_spikes = x_all.sum(dim=2)
    channel_spikes = x_all.sum(dim=(0, 1))
    nonzero_timestep_ratio = (timestep_spikes > 0).float().mean(dim=1)
    print(f"  samples analysed       : {x_all.shape[0]}")
    print(
        f"  spikes/sample          : mean={sample_spikes.mean():.1f}  "
        f"std={sample_spikes.std():.1f}  min={sample_spikes.min():.0f}  max={sample_spikes.max():.0f}"
    )
    print(
        f"  spikes/timestep/sample : mean={timestep_spikes.mean():.3f}  "
        f"std={timestep_spikes.std():.3f}  max={timestep_spikes.max():.0f}"
    )
    print(
        f"  active channels        : {(channel_spikes > 0).sum().item()} / {x_all.shape[2]}"
    )
    print(
        f"  nonzero timestep ratio : mean={nonzero_timestep_ratio.mean():.3f}  "
        f"min={nonzero_timestep_ratio.min():.3f}  max={nonzero_timestep_ratio.max():.3f}"
    )


def run_liquid_diagnostics(model, batches, device, tau: float) -> dict:
    input_stats = RunningStats()
    recurrent_stats = RunningStats()
    exc_recurrent_stats = RunningStats()
    inh_recurrent_stats = RunningStats()
    spike_rates = []

    model.liquid.sample_mask(tau=tau)
    with torch.no_grad():
        w_eff = model.liquid.get_effective_weight()
        dale = model.liquid.dale_sign.reshape(-1)
        exc_w_eff = w_eff * (dale > 0).float().view(-1, 1)
        inh_w_eff = w_eff * (dale < 0).float().view(-1, 1)

        for x, _ in batches:
            x = x.to(device)
            batch_size = x.shape[0]
            liquid_mem = torch.zeros(batch_size, model.n_liquid, device=device)
            liquid_spike = torch.zeros(batch_size, model.n_liquid, device=device)
            spike_sum = torch.zeros(batch_size, model.n_liquid, device=device)

            for t in range(model.T):
                input_current = model.input_proj(x[:, t])
                exc_recurrent_current = liquid_spike @ exc_w_eff
                inh_recurrent_current = liquid_spike @ inh_w_eff
                recurrent_current = exc_recurrent_current + inh_recurrent_current
                input_stats.update(input_current)
                recurrent_stats.update(recurrent_current)
                exc_recurrent_stats.update(exc_recurrent_current)
                inh_recurrent_stats.update(inh_recurrent_current)

                liquid_mem = (
                    model.liquid.beta * liquid_mem + input_current + recurrent_current
                )
                liquid_mem = torch.clamp(liquid_mem, -3.0, 3.0)
                liquid_spike = spike_fn(
                    liquid_mem - model.liquid.threshold.clamp(min=0.01)
                )
                liquid_mem = liquid_mem * (1.0 - liquid_spike)
                spike_sum += liquid_spike

            spike_rates.append((spike_sum / model.T).detach().cpu())

    rates = torch.cat(spike_rates, dim=0) if spike_rates else torch.empty(0)
    return {
        "input_current": input_stats.as_dict(),
        "recurrent_current": recurrent_stats.as_dict(),
        "exc_recurrent_current": exc_recurrent_stats.as_dict(),
        "inh_recurrent_current": inh_recurrent_stats.as_dict(),
        "rates": rates,
    }


def print_current_and_firing_stats(diag: dict) -> None:
    print_header("4. Current scale")
    input_stats = diag["input_current"]
    recurrent_stats = diag["recurrent_current"]
    exc_recurrent_stats = diag["exc_recurrent_current"]
    inh_recurrent_stats = diag["inh_recurrent_current"]
    ratio = recurrent_stats["abs_mean"] / max(input_stats["abs_mean"], 1e-12)
    exc_ratio = exc_recurrent_stats["abs_mean"] / max(input_stats["abs_mean"], 1e-12)
    inh_ratio = inh_recurrent_stats["abs_mean"] / max(input_stats["abs_mean"], 1e-12)
    print(f"  input_current    : {fmt_stats(input_stats)}")
    print(f"  recurrent_current: {fmt_stats(recurrent_stats)}")
    print(f"  |recurrent| / |input|: {ratio:.4f}")
    print(f"  exc_recurrent_current: {fmt_stats(exc_recurrent_stats)}")
    print(f"  inh_recurrent_current: {fmt_stats(inh_recurrent_stats)}")
    print(f"  |exc recurrent| / |input|: {exc_ratio:.4f}")
    print(f"  |inh recurrent| / |input|: {inh_ratio:.4f}")

    print_header("5. Firing rate")
    rates = diag["rates"]
    if rates.numel() == 0:
        print("  no spike rates collected")
        return
    sample_mean = rates.mean(dim=1)
    neuron_mean = rates.mean(dim=0)
    print(f"  overall mean       : {rates.mean():.4f}")
    print(f"  overall max        : {rates.max():.4f}")
    print(
        f"  sample mean rate   : mean={sample_mean.mean():.4f}  "
        f"std={sample_mean.std():.4f}  min={sample_mean.min():.4f}  max={sample_mean.max():.4f}"
    )
    print(
        f"  neuron mean rate   : mean={neuron_mean.mean():.4f}  "
        f"std={neuron_mean.std():.4f}  max={neuron_mean.max():.4f}"
    )
    print(
        f"  active neurons >0.01: {(neuron_mean > 0.01).sum().item()} / {rates.shape[1]}"
    )
    print(
        f"  active neurons >0.05: {(neuron_mean > 0.05).sum().item()} / {rates.shape[1]}"
    )


def collect_samples_by_class(
    loader, n_classes: int, samples_per_class: int
) -> dict[int, list]:
    samples_by_class: dict[int, list] = {}
    for x, y in loader:
        for xi, yi in zip(x, y):
            label = int(yi.item())
            bucket = samples_by_class.setdefault(label, [])
            if len(bucket) < samples_per_class:
                bucket.append(xi)
        ready = [
            cls
            for cls, samples in samples_by_class.items()
            if len(samples) >= samples_per_class
        ]
        if len(ready) >= n_classes:
            break
    return samples_by_class


def liquid_mean_rate(model, batch: torch.Tensor, device, tau: float) -> torch.Tensor:
    model.liquid.sample_mask(tau=tau)
    batch = batch.to(device)
    batch_size = batch.shape[0]
    liquid_mem = torch.zeros(batch_size, model.n_liquid, device=device)
    liquid_spike = torch.zeros(batch_size, model.n_liquid, device=device)
    spike_sum = torch.zeros(batch_size, model.n_liquid, device=device)
    with torch.no_grad():
        for t in range(model.T):
            input_current = model.input_proj(batch[:, t])
            recurrent_current = model.liquid(liquid_spike)
            liquid_mem = (
                model.liquid.beta * liquid_mem + input_current + recurrent_current
            )
            liquid_mem = torch.clamp(liquid_mem, -3.0, 3.0)
            liquid_spike = spike_fn(liquid_mem - model.liquid.threshold.clamp(min=0.01))
            liquid_mem = liquid_mem * (1.0 - liquid_spike)
            spike_sum += liquid_spike
    return (spike_sum / model.T).mean(dim=0).detach().cpu()


def print_class_separation(
    model, loader, device, tau: float, n_classes: int, samples_per_class: int
) -> None:
    print_header("6. Class separation")
    samples_by_class = collect_samples_by_class(loader, n_classes, samples_per_class)
    counts = {cls: len(samples) for cls, samples in sorted(samples_by_class.items())}
    eligible = [cls for cls, count in counts.items() if count >= samples_per_class]
    chosen = eligible[:n_classes]
    if len(chosen) < n_classes:
        print(
            f"  warning: requested {n_classes} classes with {samples_per_class} samples each, "
            f"but found {len(chosen)} eligible classes"
        )
        print(f"  collected sample counts: {counts}")
    if len(chosen) < 2:
        print("  not enough classes collected")
        return

    class_vecs = {}
    for cls in chosen:
        batch = torch.stack(samples_by_class[cls])
        class_vecs[cls] = liquid_mean_rate(model, batch, device, tau)

    print(f"  classes analysed: {chosen}")
    print("  mean firing rate per class:")
    for cls in chosen:
        v = class_vecs[cls]
        print(
            f"    class {cls:2d}: samples={len(samples_by_class[cls])}  "
            f"mean={v.mean():.4f}  std={v.std():.4f}  "
            f"active_neurons(>0.05)={(v > 0.05).sum().item()}"
        )

    print("\n  pairwise cosine similarity between class mean-rate vectors:")
    sims = []
    keys = list(class_vecs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = class_vecs[keys[i]]
            b = class_vecs[keys[j]]
            sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            l2 = (a - b).norm().item()
            sims.append(sim)
            print(f"    class {keys[i]} vs {keys[j]}: cosine={sim:.4f}  L2={l2:.4f}")
    if sims:
        sims_t = torch.tensor(sims)
        print(
            f"  cosine summary: mean={sims_t.mean():.4f}  "
            f"min={sims_t.min():.4f}  max={sims_t.max():.4f}"
        )

    print("\n  top discriminative neurons for first class pair:")
    v0 = class_vecs[keys[0]]
    v1 = class_vecs[keys[1]]
    diff = (v0 - v1).abs()
    topk = diff.topk(min(5, diff.numel()))
    for idx, val in zip(topk.indices.tolist(), topk.values.tolist()):
        print(
            f"    neuron {idx:4d}: |diff|={val:.4f}  "
            f"class{keys[0]}={v0[idx]:.4f}  class{keys[1]}={v1[idx]:.4f}"
        )


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    device = get_device()
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, device)
    load_checkpoint_if_requested(model, args.checkpoint, device)
    model.eval()
    model.liquid.unlock_epoch_mask()

    print_config_summary(cfg, model, args.checkpoint)
    print_parameter_sanity(cfg, model)
    print_recurrent_sparsity(model, cfg.tau_end)

    _, test_loader = get_dataloaders(cfg)
    batches = collect_batches(test_loader, args.batches)
    print_input_spike_stats(batches)

    diag = run_liquid_diagnostics(model, batches, device, cfg.tau_end)
    print_current_and_firing_stats(diag)

    print_class_separation(
        model,
        test_loader,
        device,
        cfg.tau_end,
        args.classes,
        args.samples_per_class,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
