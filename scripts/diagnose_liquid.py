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
import csv
import json
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
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Save full diagnostics as JSON",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Append summary row to CSV (creates file if missing)",
    )
    parser.add_argument(
        "--save-embeddings",
        type=str,
        default=None,
        help="Save class mean-rate vectors to CSV",
    )
    parser.add_argument(
        "--skip-cycle-metrics",
        action="store_true",
        help="Skip directed 3-cycle counting for faster topology diagnostics",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Skip average undirected clustering for faster topology diagnostics",
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
    if model.liquid.mode == "grad_r":
        topology_trainable = model.liquid.theta.requires_grad
    else:
        topology_trainable = any(
            param.requires_grad for param in model.liquid.topology_parameters()
        )
    print(f"  self diag sum    : {diag_sum:.1f}")
    print(f"  topology trainable: {topology_trainable}")
    print(f"  w_raw trainable  : {model.liquid.w_raw.requires_grad}")
    print(f"  beta trainable   : {model.liquid.logit_beta.requires_grad}")
    print(f"  threshold trainable: {model.liquid.threshold.requires_grad}")
    if model.liquid.mode == "learned_lowrank":
        theta_named = [
            name for name, _ in model.liquid.named_parameters() if "theta" in name
        ]
        embed_named = [
            name
            for name, _ in model.liquid.named_parameters()
            if "embed" in name or "bias" in name
        ]
        print(f"  theta rank       : {model.liquid.src_embed.shape[1]}")
        print(f"  has dense theta  : {hasattr(model.liquid, 'theta')}")
        print(f"  theta params     : {theta_named}")
        print(f"  topology params  : {embed_named}")


def print_recurrent_sparsity(
    model, tau: float, skip_cycle_metrics: bool = False, skip_clustering: bool = False
) -> dict:
    print_header("2. Recurrent sparsity")
    model.liquid.sample_mask(tau=tau)
    mask = model.liquid.get_binary_mask().detach().cpu()
    mask = mask * model.liquid.self_conn_mask.detach().cpu()
    n_total = mask.numel()
    n_active = int(mask.sum().item())
    diag_active = int(mask.diag().sum().item())
    print(f"  density     : {n_active / n_total:.4f}")
    print(f"  active edges: {n_active} / {n_total}")
    print(f"  self edges  : {diag_active}")
    if model.liquid.mode in ("learned", "learned_lowrank"):
        topology_logit = model.liquid.get_theta().detach().cpu()
        probs = torch.sigmoid(topology_logit)
        print(
            f"  sigma(logit): mean={probs.mean():.4f}  std={probs.std():.4f}  "
            f"min={probs.min():.4f}  max={probs.max():.4f}"
        )
        if model.liquid.mode == "learned_lowrank":
            src = model.liquid.src_embed.detach().cpu()
            dst = model.liquid.dst_embed.detach().cpu()
            bias = model.liquid.theta_bias.detach().cpu().item()
            print(
                f"  src_embed   : {tensor_summary(src)}  norm={src.norm(dim=1).mean().item():.4f}"
            )
            print(
                f"  dst_embed   : {tensor_summary(dst)}  norm={dst.norm(dim=1).mean().item():.4f}"
            )
            print(f"  theta_bias  : {bias:.4f}")
    print_recurrent_weight_stats(model, mask)
    graph_stats = print_graph_structure_stats(mask)
    graph_metrics = compute_graph_metrics(
        mask,
        model.liquid.dale_sign.detach().cpu(),
        skip_cycle_metrics=skip_cycle_metrics,
        skip_clustering=skip_clustering,
    )
    print_graph_topology_metrics(graph_metrics)
    return {
        "density": n_active / n_total,
        "active_edges": n_active,
        "self_edges": diag_active,
        **graph_stats,
        **graph_metrics,
    }


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


def connected_component_sizes(mask: torch.Tensor) -> list[int]:
    """Weakly connected component sizes on the undirected version of the graph."""
    undirected = (mask.bool() | mask.bool().T)
    n = undirected.shape[0]
    visited = torch.zeros(n, dtype=torch.bool)
    sizes: list[int] = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = torch.nonzero(undirected[node], as_tuple=False).flatten().tolist()
            for nxt in neighbors:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def gini(x: torch.Tensor) -> float:
    """Gini coefficient for a non-negative vector; all-zero vectors return 0."""
    values = x.detach().float().cpu().reshape(-1)
    if values.numel() == 0:
        return 0.0
    values = values.clamp(min=0)
    total = values.sum()
    if total.item() == 0.0:
        return 0.0
    sorted_values = values.sort().values
    n = sorted_values.numel()
    index = torch.arange(1, n + 1, dtype=sorted_values.dtype)
    coeff = (2.0 * torch.dot(index, sorted_values) / (n * total)) - ((n + 1.0) / n)
    return float(coeff.clamp(min=0.0, max=1.0).item())


def _adjacency_without_self_loops(mask: torch.Tensor) -> torch.Tensor:
    active = mask.detach().cpu().bool().clone()
    if active.ndim != 2 or active.shape[0] != active.shape[1]:
        raise ValueError(f"Expected square adjacency mask, got shape {tuple(active.shape)}")
    active.fill_diagonal_(False)
    return active


def ei_block_counts(mask: torch.Tensor, dale_sign: torch.Tensor) -> dict[str, float]:
    active = _adjacency_without_self_loops(mask)
    dale = dale_sign.detach().cpu().reshape(-1)
    if dale.numel() != active.shape[0]:
        raise ValueError(
            f"dale_sign length {dale.numel()} does not match mask size {active.shape[0]}"
        )
    exc = dale > 0
    inh = dale < 0
    active_edges = int(active.sum().item())

    ee_count = int((active & exc[:, None] & exc[None, :]).sum().item())
    ei_count = int((active & exc[:, None] & inh[None, :]).sum().item())
    ie_count = int((active & inh[:, None] & exc[None, :]).sum().item())
    ii_count = int((active & inh[:, None] & inh[None, :]).sum().item())
    return {
        "ee_count": ee_count,
        "ei_count": ei_count,
        "ie_count": ie_count,
        "ii_count": ii_count,
        "ee_frac": ee_count / active_edges if active_edges else 0.0,
        "ei_frac": ei_count / active_edges if active_edges else 0.0,
        "ie_frac": ie_count / active_edges if active_edges else 0.0,
        "ii_frac": ii_count / active_edges if active_edges else 0.0,
    }


def reciprocity_metrics(mask: torch.Tensor) -> dict[str, float]:
    active = _adjacency_without_self_loops(mask)
    active_edges = int(active.sum().item())
    reciprocal_directed_edges = int((active & active.T).sum().item())
    return {
        "reciprocity": reciprocal_directed_edges / active_edges if active_edges else 0.0,
        "reciprocal_pair_count": reciprocal_directed_edges // 2,
    }


def directed_3cycle_count(mask: torch.Tensor) -> int:
    active = _adjacency_without_self_loops(mask)
    if active.sum().item() == 0:
        return 0
    a = active.float()
    tri = torch.trace(a @ a @ a).item()
    return int(round(tri / 3.0))


def average_undirected_clustering(mask: torch.Tensor) -> float:
    active = _adjacency_without_self_loops(mask)
    undirected = active | active.T
    undirected.fill_diagonal_(False)
    local_values: list[float] = []

    for node in range(undirected.shape[0]):
        neighbors = torch.nonzero(undirected[node], as_tuple=False).flatten()
        k = int(neighbors.numel())
        if k < 2:
            continue
        subgraph = undirected[neighbors][:, neighbors]
        actual = float(subgraph.sum().item()) / 2.0
        possible = k * (k - 1) / 2.0
        local_values.append(actual / possible)

    if not local_values:
        return 0.0
    return float(sum(local_values) / len(local_values))


def compute_graph_metrics(
    mask: torch.Tensor,
    dale_sign: torch.Tensor,
    skip_cycle_metrics: bool = False,
    skip_clustering: bool = False,
) -> dict[str, float]:
    active = _adjacency_without_self_loops(mask)
    in_degree = active.float().sum(dim=0)
    out_degree = active.float().sum(dim=1)

    metrics: dict[str, float] = {
        "in_degree_gini": gini(in_degree),
        "out_degree_gini": gini(out_degree),
        **ei_block_counts(active, dale_sign),
        **reciprocity_metrics(active),
        "directed_3cycle_count": float("nan") if skip_cycle_metrics else directed_3cycle_count(active),
        "clustering": float("nan") if skip_clustering else average_undirected_clustering(active),
    }
    return metrics


def _sanity_check_graph_metrics() -> None:
    """Small checks documenting expected graph metric behavior."""
    dale = torch.tensor([1.0, -1.0, 1.0])

    empty = torch.zeros(3, 3, dtype=torch.bool)
    empty_metrics = compute_graph_metrics(empty, dale)
    assert empty_metrics["in_degree_gini"] == 0.0
    assert empty_metrics["out_degree_gini"] == 0.0
    assert empty_metrics["reciprocity"] == 0.0
    assert empty_metrics["directed_3cycle_count"] == 0
    assert empty_metrics["clustering"] == 0.0

    complete = torch.ones(3, 3, dtype=torch.bool)
    complete.fill_diagonal_(False)
    complete_metrics = compute_graph_metrics(complete, dale)
    assert complete_metrics["reciprocity"] == 1.0

    cycle = torch.zeros(3, 3, dtype=torch.bool)
    cycle[0, 1] = True
    cycle[1, 2] = True
    cycle[2, 0] = True
    cycle_metrics = compute_graph_metrics(cycle, dale)
    assert cycle_metrics["directed_3cycle_count"] == 1


def print_graph_topology_metrics(metrics: dict[str, float]) -> None:
    print_header("Graph topology metrics")
    print(f"  in-degree Gini         : {metrics['in_degree_gini']:.4f}")
    print(f"  out-degree Gini        : {metrics['out_degree_gini']:.4f}")
    print(
        "  E/I block counts       : "
        f"EE={metrics['ee_count']}  EI={metrics['ei_count']}  "
        f"IE={metrics['ie_count']}  II={metrics['ii_count']}"
    )
    print(
        "  E/I block fractions    : "
        f"EE={metrics['ee_frac']:.4f}  EI={metrics['ei_frac']:.4f}  "
        f"IE={metrics['ie_frac']:.4f}  II={metrics['ii_frac']:.4f}"
    )
    print(
        f"  reciprocity            : {metrics['reciprocity']:.4f}  "
        f"pairs={metrics['reciprocal_pair_count']}"
    )
    cycle_count = metrics["directed_3cycle_count"]
    if isinstance(cycle_count, float) and torch.isnan(torch.tensor(cycle_count)):
        print("  directed 3-cycles      : skipped")
    else:
        print(f"  directed 3-cycles      : {int(cycle_count)}")
    clustering = metrics["clustering"]
    if isinstance(clustering, float) and torch.isnan(torch.tensor(clustering)):
        print("  clustering             : skipped")
    else:
        print(f"  clustering             : {clustering:.4f}")


def print_graph_structure_stats(mask: torch.Tensor) -> dict:
    print("\n  graph structure:")
    active = mask.bool()
    in_degree = active.float().sum(dim=0)
    out_degree = active.float().sum(dim=1)
    total_degree = in_degree + out_degree
    isolated = total_degree == 0
    component_sizes = connected_component_sizes(active)
    giant_size = component_sizes[0] if component_sizes else 0

    print(f"  in-degree              : {tensor_summary(in_degree)}")
    print(f"  out-degree             : {tensor_summary(out_degree)}")
    print(
        f"  isolated neurons       : {isolated.sum().item()} / {active.shape[0]} "
        f"({isolated.float().mean().item():.4f})"
    )
    print(
        f"  weak components        : count={len(component_sizes)}  "
        f"giant={giant_size} ({giant_size / max(active.shape[0], 1):.4f})"
    )
    top_in = in_degree.topk(min(5, in_degree.numel()))
    top_out = out_degree.topk(min(5, out_degree.numel()))
    print("  top in-degree hubs     : " + ", ".join(
        f"{idx}:{val:.0f}" for idx, val in zip(top_in.indices.tolist(), top_in.values.tolist())
    ))
    print("  top out-degree hubs    : " + ", ".join(
        f"{idx}:{val:.0f}" for idx, val in zip(top_out.indices.tolist(), top_out.values.tolist())
    ))
    return {
        "isolated_neurons": int(isolated.sum().item()),
        "weak_components": len(component_sizes),
        "giant_component": giant_size,
        "in_degree_mean": in_degree.mean().item(),
        "in_degree_std": in_degree.std().item(),
        "out_degree_mean": out_degree.mean().item(),
        "out_degree_std": out_degree.std().item(),
    }


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


def print_current_and_firing_stats(diag: dict) -> dict:
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
    print(f"  dead neurons ==0.0 : {(neuron_mean == 0.0).sum().item()} / {rates.shape[1]}")
    print(
        f"  active neurons >0.01: {(neuron_mean > 0.01).sum().item()} / {rates.shape[1]}"
    )
    print(
        f"  active neurons >0.05: {(neuron_mean > 0.05).sum().item()} / {rates.shape[1]}"
    )
    print(
        f"  overactive >0.20    : {(neuron_mean > 0.20).sum().item()} / {rates.shape[1]}"
    )
    input_stats = diag["input_current"]
    recurrent_stats = diag["recurrent_current"]
    ratio = recurrent_stats["abs_mean"] / max(input_stats["abs_mean"], 1e-12)
    return {
        "firing_rate_mean": rates.mean().item(),
        "firing_rate_max": rates.max().item(),
        "dead_neurons": int((neuron_mean == 0.0).sum().item()),
        "active_gt001": int((neuron_mean > 0.01).sum().item()),
        "active_gt005": int((neuron_mean > 0.05).sum().item()),
        "overactive_gt020": int((neuron_mean > 0.20).sum().item()),
        "recurrent_input_ratio": ratio,
        "input_current": input_stats,
        "recurrent_current": recurrent_stats,
    }


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
) -> tuple[dict[int, list], dict]:
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
        return samples_by_class, {}

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

    separation_stats = {}
    if sims:
        sims_t = torch.tensor(sims)
        separation_stats = {
            "cosine_mean": sims_t.mean().item(),
            "cosine_min": sims_t.min().item(),
            "cosine_max": sims_t.max().item(),
        }

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
    return samples_by_class, {**separation_stats, "class_vecs": class_vecs}


def readout_logits_mean(model, batch: torch.Tensor, device, tau: float) -> torch.Tensor:
    with torch.no_grad():
        logits = model(batch.to(device), tau=tau)
    return logits.mean(dim=0).detach().cpu()


def print_readout_separation(
    model,
    samples_by_class: dict[int, list],
    device,
    tau: float,
    n_classes: int,
    samples_per_class: int,
) -> dict:
    print_header("7. Readout separation")
    eligible = [
        cls for cls, samples in sorted(samples_by_class.items()) if len(samples) >= samples_per_class
    ]
    chosen = eligible[:n_classes]
    if len(chosen) < 2:
        print("  not enough classes collected")
        return {}

    class_logits = {}
    for cls in chosen:
        batch = torch.stack(samples_by_class[cls])
        class_logits[cls] = readout_logits_mean(model, batch, device, tau)

    print(f"  classes analysed: {chosen}")
    print("  mean logits per class:")
    margins = []
    class_correct = []
    for cls in chosen:
        v = class_logits[cls]
        top2 = v.topk(k=min(2, v.numel()))
        margin = float("-inf")
        if v.numel() > 1:
            others = torch.cat([v[:cls], v[cls + 1 :]])
            if others.numel() > 0:
                margin = (v[cls] - others.max()).item()
        margins.append(margin)
        class_correct.append(float(top2.indices[0].item() == cls))
        print(
            f"    class {cls:2d}: true_logit={v[cls]:.4f}  "
            f"margin_vs_best_other={margin:.4f}  "
            f"top_pred={top2.indices[0].item()} ({top2.values[0].item():.4f})"
        )

    print("\n  pairwise cosine similarity between class mean-logit vectors:")
    sims = []
    keys = list(class_logits.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = class_logits[keys[i]]
            b = class_logits[keys[j]]
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
    margins_t = torch.tensor(margins)
    return {
        "readout_margin_mean": margins_t.mean().item(),
        "readout_margin_min": margins_t.min().item(),
        "readout_margin_max": margins_t.max().item(),
        "readout_class_mean_accuracy": (
            sum(class_correct) / len(class_correct) if class_correct else 0.0
        ),
    }


def diagnostic_batch_readout_stats(model, batches, device, tau: float) -> dict:
    """Compute readout accuracy and margins on the diagnostic batches."""
    correct_margins = []
    incorrect_margins = []
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for x, y in batches:
            logits = model(x.to(device), tau=tau).detach().cpu()
            labels = y.detach().cpu().long()
            preds = logits.argmax(dim=1)
            true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
            masked = logits.clone()
            masked[torch.arange(labels.numel()), labels] = float("-inf")
            margins = true_logits - masked.max(dim=1).values
            correct = preds == labels
            n_correct += int(correct.sum().item())
            n_total += int(labels.numel())
            correct_margins.extend(margins[correct].tolist())
            incorrect_margins.extend(margins[~correct].tolist())

    def _mean(values: list[float]) -> float:
        if not values:
            return float("nan")
        return float(torch.tensor(values).mean().item())

    all_margins = correct_margins + incorrect_margins
    return {
        "readout_accuracy": n_correct / n_total if n_total else float("nan"),
        "readout_sample_margin_mean": _mean(all_margins),
        "readout_correct_margin_mean": _mean(correct_margins),
        "readout_incorrect_margin_mean": _mean(incorrect_margins),
        "readout_num_samples": n_total,
        "readout_num_correct": n_correct,
    }


def _save_json(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        raise TypeError(type(obj))

    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=_convert)
    print(f"  saved JSON → {path}")


def _append_csv(path: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    if p.exists():
        with open(p, newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        expected_header = list(row.keys())
        if existing_header != expected_header:
            missing = [name for name in expected_header if name not in (existing_header or [])]
            extra = [name for name in (existing_header or []) if name not in expected_header]
            raise ValueError(
                f"Existing CSV schema does not match current diagnostics: {p}\n"
                f"  missing columns in existing file: {missing}\n"
                f"  extra columns in existing file: {extra}\n"
                "Remove the old CSV or choose a new --out-csv path before regenerating."
            )
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  appended CSV row → {path}")


def _save_embeddings(path: str, class_vecs: dict) -> None:
    if not class_vecs:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_neurons = next(iter(class_vecs.values())).shape[0]
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + [f"n{i}" for i in range(n_neurons)])
        for cls, vec in sorted(class_vecs.items()):
            writer.writerow([cls] + vec.tolist())
    print(f"  saved embeddings → {path}")


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
    sparsity_stats = print_recurrent_sparsity(
        model,
        cfg.tau_end,
        skip_cycle_metrics=args.skip_cycle_metrics,
        skip_clustering=args.skip_clustering,
    )

    _, test_loader = get_dataloaders(cfg)
    batches = collect_batches(test_loader, args.batches)
    print_input_spike_stats(batches)

    diag = run_liquid_diagnostics(model, batches, device, cfg.tau_end)
    firing_stats = print_current_and_firing_stats(diag)

    samples_by_class, sep_stats = print_class_separation(
        model,
        test_loader,
        device,
        cfg.tau_end,
        args.classes,
        args.samples_per_class,
    )
    readout_class_stats = print_readout_separation(
        model,
        samples_by_class,
        device,
        cfg.tau_end,
        args.classes,
        args.samples_per_class,
    )
    readout_batch_stats = diagnostic_batch_readout_stats(
        model, batches, device, cfg.tau_end
    )
    print("=" * 72)

    class_vecs = sep_stats.pop("class_vecs", {})

    if args.out_json:
        graph_metric_keys = [
            "in_degree_gini",
            "out_degree_gini",
            "ee_count",
            "ei_count",
            "ie_count",
            "ii_count",
            "ee_frac",
            "ei_frac",
            "ie_frac",
            "ii_frac",
            "reciprocity",
            "reciprocal_pair_count",
            "directed_3cycle_count",
            "clustering",
        ]
        _save_json(args.out_json, {
            "checkpoint": args.checkpoint,
            "config": args.config,
            "sparsity": sparsity_stats,
            "graph_metrics": {k: sparsity_stats[k] for k in graph_metric_keys},
            "firing": firing_stats,
            "separation": sep_stats,
            "readout_class_separation": readout_class_stats,
            "readout_batch_diagnostics": readout_batch_stats,
        })

    if args.out_csv:
        exp_name = Path(args.config).parent.name
        row = {
            "exp": exp_name,
            "recurrent_mode": cfg.liquid.recurrent_mode,
            "seed": cfg.seed,
            **sparsity_stats,
            **{k: v for k, v in firing_stats.items()
               if k not in ("input_current", "recurrent_current")},
            **sep_stats,
            **readout_class_stats,
            **readout_batch_stats,
        }
        _append_csv(args.out_csv, row)

    if args.save_embeddings:
        _save_embeddings(args.save_embeddings, class_vecs)


if __name__ == "__main__":
    main()
