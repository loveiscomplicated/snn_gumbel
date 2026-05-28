"""
Canonical batch topology diagnostics for saved LSM experiments.

Phase 1 scope:
  - deterministic recurrent mask extraction
  - graph/topology diagnostics
  - standardized CSV and JSON outputs
  - placeholder activity outputs only

Examples:
    python scripts/topology_diagnostics.py \
      --experiments experiments/run_a experiments/run_b \
      --output-dir runs/diagnostics/topology_v1 \
      --device auto \
      --skip-path-metrics

    python scripts/topology_diagnostics.py \
      --config experiments/<run>/config.yaml \
      --checkpoint experiments/<run>/checkpoints/best.pt \
      --output-dir runs/diagnostics/<run> \
      --device auto \
      --skip-path-metrics
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime with a clear error
    pd = None

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional dependency
    nx = None

from scripts.diagnose_liquid import (
    average_undirected_clustering,
    connected_component_sizes,
    directed_3cycle_count,
    gini,
)
from src.lsm.trainer import build_model
from src.utils.config import load_config


GRAPH_COLUMNS = [
    "experiment_name",
    "experiment_dir",
    "config_path",
    "checkpoint_path",
    "recurrent_mode",
    "seed",
    "method_label",
    "n_nodes",
    "n_possible_edges",
    "n_active_edges",
    "density",
    "in_degree_mean",
    "in_degree_std",
    "in_degree_min",
    "in_degree_max",
    "out_degree_mean",
    "out_degree_std",
    "out_degree_min",
    "out_degree_max",
    "in_gini",
    "out_gini",
    "edge_EE_count",
    "edge_EI_count",
    "edge_IE_count",
    "edge_II_count",
    "edge_EE_frac",
    "edge_EI_frac",
    "edge_IE_frac",
    "edge_II_frac",
    "reciprocal_pair_count",
    "reciprocity",
    "three_cycle_count",
    "three_cycle_density",
    "clustering_undirected",
    "weak_components",
    "largest_weak_component",
    "strong_components",
    "largest_strong_component",
    "reachability_ratio",
    "avg_shortest_path_reachable",
    "effective_diameter_p90",
    "src_embed_norm_mean",
    "src_embed_norm_std",
    "src_embed_norm_min",
    "src_embed_norm_max",
    "dst_embed_norm_mean",
    "dst_embed_norm_std",
    "dst_embed_norm_min",
    "dst_embed_norm_max",
    "src_dst_norm_corr",
    "theta_bias_value",
    "topology_logit_mean",
    "topology_logit_std",
    "topology_logit_p05",
    "topology_logit_p50",
    "topology_logit_p95",
    "topology_prob_mean",
    "topology_prob_std",
    "topology_prob_p05",
    "topology_prob_p50",
    "topology_prob_p95",
    "src_norm_out_degree_corr",
    "dst_norm_in_degree_corr",
    "readout_in_degree_corr",
    "readout_out_degree_corr",
    "readout_total_degree_corr",
    "top_degree_top_readout_overlap",
]

ACTIVITY_COLUMNS = [
    "experiment_name",
    "method_label",
    "recurrent_mode",
    "seed",
    "activity_diagnostics_available",
    "num_batches",
    "num_samples",
    "mean_firing_rate",
    "max_firing_rate",
    "active_neurons_gt_005",
    "rec_input_abs_ratio",
    "class_mean_cosine_mean",
    "mean_logit_margin",
    "accuracy_on_diagnostic_batches",
    "skip_reason",
]

SUMMARY_COLUMNS = [
    "experiment_name",
    "method_label",
    "recurrent_mode",
    "seed",
    "density",
    "n_active_edges",
    "in_gini",
    "out_gini",
    "edge_EE_frac",
    "edge_EI_frac",
    "edge_IE_frac",
    "edge_II_frac",
    "reciprocity",
    "three_cycle_density",
    "clustering_undirected",
    "largest_weak_component",
    "largest_strong_component",
    "reachability_ratio",
    "avg_shortest_path_reachable",
    "effective_diameter_p90",
    "src_norm_out_degree_corr",
    "dst_norm_in_degree_corr",
    "readout_in_degree_corr",
    "readout_out_degree_corr",
    "readout_total_degree_corr",
    "topology_logit_mean",
    "topology_logit_std",
    "topology_prob_mean",
    "topology_prob_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch topology diagnostics for LSM runs")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Experiment directories containing config.yaml and checkpoints/best.pt",
    )
    parser.add_argument("--config", type=str, default=None, help="Explicit config path")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Explicit checkpoint path"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for CSV/JSON files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device",
    )
    parser.add_argument(
        "--skip-path-metrics",
        action="store_true",
        help="Skip reachability and shortest-path graph metrics",
    )
    parser.add_argument(
        "--activity-placeholder-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Phase 1 keeps activity metrics as placeholders only",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    """Select device with explicit auto semantics: CUDA > MPS > CPU."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps but MPS is not available.")
        return torch.device("mps")
    return torch.device("cpu")


def resolve_experiment_specs(args: argparse.Namespace) -> list[dict[str, str | None]]:
    """Resolve CLI inputs into normalized experiment specs."""
    has_experiments = bool(args.experiments)
    has_explicit = bool(args.config or args.checkpoint)

    if has_experiments and has_explicit:
        raise ValueError(
            "Use either --experiments or --config/--checkpoint, not both."
        )
    if bool(args.config) != bool(args.checkpoint):
        raise ValueError(
            "--config and --checkpoint must be provided together in explicit mode."
        )
    if not has_experiments and not has_explicit:
        raise ValueError(
            "Provide either --experiments <dir...> or --config <path> --checkpoint <path>."
        )

    specs: list[dict[str, str | None]] = []
    if has_experiments:
        for exp in args.experiments:
            exp_dir = Path(exp).resolve()
            config_path = exp_dir / "config.yaml"
            checkpoint_path = exp_dir / "checkpoints" / "best.pt"
            specs.append(
                {
                    "experiment_dir": str(exp_dir),
                    "config_path": str(config_path),
                    "checkpoint_path": str(checkpoint_path),
                }
            )
    else:
        config_path = Path(args.config).resolve()
        checkpoint_path = Path(args.checkpoint).resolve()
        exp_dir = config_path.parent.resolve()
        specs.append(
            {
                "experiment_dir": str(exp_dir),
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint_path),
            }
        )
    return specs


def load_config_and_model(
    config_path: str, checkpoint_path: str, device: torch.device
) -> tuple[Any, torch.nn.Module]:
    """Load config, instantiate the model, and restore checkpoint state."""
    config_file = Path(config_path)
    checkpoint_file = Path(checkpoint_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Missing config file: {config_file}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_file}")

    cfg = load_config(str(config_file))
    torch.manual_seed(int(getattr(cfg, "seed", 0)))
    model = build_model(cfg, device)
    ckpt = torch.load(str(checkpoint_file), map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    if hasattr(model, "liquid") and hasattr(model.liquid, "unlock_epoch_mask"):
        model.liquid.unlock_epoch_mask()
    if missing or unexpected:
        # Keep checkpoint loading permissive but make the discrepancy visible.
        print(
            f"[warn] load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}"
        )
    return cfg, model


def infer_method_label(exp_dir: str | None, cfg: Any) -> str:
    """Infer a paper-facing method label from experiment name and config."""
    name_parts = [
        str(exp_dir or "").lower(),
        str(getattr(cfg, "experiment_name", "")).lower(),
    ]
    text = " ".join(name_parts)
    recurrent_mode = str(getattr(cfg.liquid, "recurrent_mode", "")).lower()

    if recurrent_mode == "random_sparse":
        if "density_control" in text or "rs_density" in text:
            return "random_sparse_density_control"
        return "unknown"
    if recurrent_mode == "grad_r":
        if getattr(cfg.liquid, "theta_adaptive_freeze", False) or "gfreeze" in text:
            return "grad_r_adaptive_freeze"
        return "unknown"
    if recurrent_mode == "learned_lowrank":
        if "m50p10" in text:
            return "learned_lowrank_m50p10"
        return "learned_lowrank_other"
    if recurrent_mode == "learned":
        return "learned"
    if recurrent_mode == "fixed":
        return "fixed"
    if recurrent_mode in {"none", "no_recurrence"}:
        return "no_recurrence"
    return "unknown"


def extract_recurrent_mask(model: torch.nn.Module) -> torch.Tensor:
    """Extract the deterministic recurrent adjacency as a CPU bool tensor."""
    model.eval()
    liquid = model.liquid
    if hasattr(liquid, "unlock_epoch_mask"):
        liquid.unlock_epoch_mask()

    if hasattr(liquid, "get_binary_mask"):
        mask = liquid.get_binary_mask()
    else:
        mode = str(getattr(liquid, "mode", "")).lower()
        if mode in {"random_sparse", "fixed"}:
            mask = liquid.fixed_mask
        elif mode == "learned":
            mask = torch.sigmoid(liquid.theta) >= 0.5
        elif mode == "grad_r":
            mask = liquid.theta > 0
        elif mode == "learned_lowrank":
            theta_like = liquid.src_embed @ liquid.dst_embed.T + liquid.theta_bias
            mask = torch.sigmoid(theta_like) >= 0.5
        else:
            raise RuntimeError(f"Unsupported liquid mode for diagnostics: {mode!r}")

    mask = mask.detach().bool()
    if hasattr(liquid, "self_conn_mask") and liquid.self_conn_mask is not None:
        mask = mask & liquid.self_conn_mask.detach().bool()
    return mask.cpu()


def safe_float(x: Any) -> float:
    """Convert scalar-like values to a Python float, preserving NaN."""
    if x is None:
        return float("nan")
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, np.generic):
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().cpu().item())
    return float(x)


def safe_pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation with NaN on size mismatch or zero variance."""
    a = x.detach().float().cpu().reshape(-1)
    b = y.detach().float().cpu().reshape(-1)
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom.item() == 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / denom.item())


def _adjacency_without_self_loops(mask: torch.Tensor) -> torch.Tensor:
    active = mask.detach().cpu().bool().clone()
    active.fill_diagonal_(False)
    return active


def _possible_edges(mask: torch.Tensor, model: torch.nn.Module) -> int:
    liquid = model.liquid
    if hasattr(liquid, "self_conn_mask") and liquid.self_conn_mask is not None:
        return int(liquid.self_conn_mask.detach().bool().sum().item())
    return int(mask.numel())


def _dale_sign_vector(model: torch.nn.Module, cfg: Any, n_nodes: int) -> torch.Tensor:
    liquid = model.liquid
    if hasattr(liquid, "dale_sign") and liquid.dale_sign is not None:
        dale = liquid.dale_sign.detach().cpu().reshape(-1)
        if dale.numel() == n_nodes:
            return dale
    exc_ratio = float(getattr(cfg.liquid, "exc_ratio", 0.8))
    n_exc = int(exc_ratio * n_nodes)
    dale = torch.ones(n_nodes)
    dale[n_exc:] = -1.0
    return dale


def _edge_type_metrics(mask: torch.Tensor, dale_sign: torch.Tensor) -> dict[str, float]:
    active = _adjacency_without_self_loops(mask)
    dale = dale_sign.detach().cpu().reshape(-1)
    exc = dale > 0
    inh = dale < 0
    n_active = int(active.sum().item())

    ee_count = int((active & exc[:, None] & exc[None, :]).sum().item())
    ei_count = int((active & exc[:, None] & inh[None, :]).sum().item())
    ie_count = int((active & inh[:, None] & exc[None, :]).sum().item())
    ii_count = int((active & inh[:, None] & inh[None, :]).sum().item())
    return {
        "edge_EE_count": ee_count,
        "edge_EI_count": ei_count,
        "edge_IE_count": ie_count,
        "edge_II_count": ii_count,
        "edge_EE_frac": ee_count / n_active if n_active else 0.0,
        "edge_EI_frac": ei_count / n_active if n_active else 0.0,
        "edge_IE_frac": ie_count / n_active if n_active else 0.0,
        "edge_II_frac": ii_count / n_active if n_active else 0.0,
    }


def _reciprocity_metrics(mask: torch.Tensor) -> dict[str, float]:
    active = _adjacency_without_self_loops(mask)
    symmetric_support = (active | active.T).triu(diagonal=1)
    reciprocal_pairs = int((active & active.T).triu(diagonal=1).sum().item())
    possible_pairs = int(symmetric_support.sum().item())
    return {
        "reciprocal_pair_count": reciprocal_pairs,
        "reciprocity": reciprocal_pairs / possible_pairs if possible_pairs else 0.0,
    }


def _strong_component_sizes(mask: torch.Tensor) -> list[int]:
    active = _adjacency_without_self_loops(mask)
    n_nodes = active.shape[0]
    out_neighbors = [
        torch.nonzero(active[i], as_tuple=False).flatten().tolist() for i in range(n_nodes)
    ]
    rev_neighbors = [
        torch.nonzero(active[:, i], as_tuple=False).flatten().tolist() for i in range(n_nodes)
    ]

    visited = [False] * n_nodes
    order: list[int] = []
    for start in range(n_nodes):
        if visited[start]:
            continue
        stack: list[tuple[int, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))
            for nxt in out_neighbors[node]:
                if not visited[nxt]:
                    stack.append((nxt, False))

    assigned = [False] * n_nodes
    sizes: list[int] = []
    for start in reversed(order):
        if assigned[start]:
            continue
        size = 0
        stack = [start]
        assigned[start] = True
        while stack:
            node = stack.pop()
            size += 1
            for nxt in rev_neighbors[node]:
                if not assigned[nxt]:
                    assigned[nxt] = True
                    stack.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _path_metrics(mask: torch.Tensor) -> dict[str, float]:
    active = _adjacency_without_self_loops(mask)
    n_nodes = active.shape[0]
    if n_nodes < 2:
        return {
            "reachability_ratio": 0.0,
            "avg_shortest_path_reachable": float("nan"),
            "effective_diameter_p90": float("nan"),
        }

    out_neighbors = [
        torch.nonzero(active[i], as_tuple=False).flatten().tolist() for i in range(n_nodes)
    ]
    total_pairs = n_nodes * (n_nodes - 1)
    reachable_pairs = 0
    distances: list[int] = []

    for src in range(n_nodes):
        dist = [-1] * n_nodes
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            next_dist = dist[node] + 1
            for nxt in out_neighbors[node]:
                if dist[nxt] != -1:
                    continue
                dist[nxt] = next_dist
                queue.append(nxt)

        for dst, d in enumerate(dist):
            if dst == src or d <= 0:
                continue
            reachable_pairs += 1
            distances.append(d)

    if not distances:
        return {
            "reachability_ratio": 0.0,
            "avg_shortest_path_reachable": float("nan"),
            "effective_diameter_p90": float("nan"),
        }
    dist_arr = np.asarray(distances, dtype=float)
    return {
        "reachability_ratio": reachable_pairs / total_pairs,
        "avg_shortest_path_reachable": float(dist_arr.mean()),
        "effective_diameter_p90": float(np.quantile(dist_arr, 0.90)),
    }


def compute_core_graph_metrics(
    mask: torch.Tensor,
    model: torch.nn.Module,
    cfg: Any,
    *,
    skip_path_metrics: bool,
    use_networkx: bool,
) -> dict[str, float]:
    """Compute core graph metrics from a deterministic recurrent mask."""
    active = _adjacency_without_self_loops(mask)
    n_nodes = int(active.shape[0])
    n_possible_edges = _possible_edges(mask, model)
    n_active_edges = int(active.sum().item())
    in_degree = active.float().sum(dim=0)
    out_degree = active.float().sum(dim=1)
    dale = _dale_sign_vector(model, cfg, n_nodes)

    if use_networkx and nx is not None:
        graph = nx.from_numpy_array(active.numpy(), create_using=nx.DiGraph)
        weak_sizes = sorted(
            (len(component) for component in nx.weakly_connected_components(graph)),
            reverse=True,
        )
        strong_sizes = sorted(
            (len(component) for component in nx.strongly_connected_components(graph)),
            reverse=True,
        )
        clustering = (
            float(nx.average_clustering(graph.to_undirected()))
            if n_nodes > 0
            else 0.0
        )
    else:
        weak_sizes = connected_component_sizes(active)
        strong_sizes = _strong_component_sizes(active)
        clustering = average_undirected_clustering(active)

    metrics: dict[str, float] = {
        "n_nodes": n_nodes,
        "n_possible_edges": n_possible_edges,
        "n_active_edges": n_active_edges,
        "density": n_active_edges / n_possible_edges if n_possible_edges else 0.0,
        "in_degree_mean": safe_float(in_degree.mean()),
        "in_degree_std": safe_float(in_degree.std()),
        "in_degree_min": safe_float(in_degree.min()),
        "in_degree_max": safe_float(in_degree.max()),
        "out_degree_mean": safe_float(out_degree.mean()),
        "out_degree_std": safe_float(out_degree.std()),
        "out_degree_min": safe_float(out_degree.min()),
        "out_degree_max": safe_float(out_degree.max()),
        "in_gini": gini(in_degree),
        "out_gini": gini(out_degree),
        "three_cycle_count": safe_float(directed_3cycle_count(active)),
        "three_cycle_density": (
            safe_float(directed_3cycle_count(active)) / (n_nodes * (n_nodes - 1) * (n_nodes - 2))
            if n_nodes >= 3
            else float("nan")
        ),
        "clustering_undirected": safe_float(clustering),
        "weak_components": len(weak_sizes),
        "largest_weak_component": weak_sizes[0] if weak_sizes else 0,
        "strong_components": len(strong_sizes),
        "largest_strong_component": strong_sizes[0] if strong_sizes else 0,
        **_edge_type_metrics(active, dale),
        **_reciprocity_metrics(active),
    }
    if skip_path_metrics:
        metrics.update(
            {
                "reachability_ratio": float("nan"),
                "avg_shortest_path_reachable": float("nan"),
                "effective_diameter_p90": float("nan"),
            }
        )
    else:
        metrics.update(_path_metrics(active))
    return metrics


def compute_lowrank_metrics(model: torch.nn.Module, mask: torch.Tensor) -> dict[str, float]:
    """Compute learned-lowrank-specific embedding/logit diagnostics."""
    nan_metrics = {
        "src_embed_norm_mean": float("nan"),
        "src_embed_norm_std": float("nan"),
        "src_embed_norm_min": float("nan"),
        "src_embed_norm_max": float("nan"),
        "dst_embed_norm_mean": float("nan"),
        "dst_embed_norm_std": float("nan"),
        "dst_embed_norm_min": float("nan"),
        "dst_embed_norm_max": float("nan"),
        "src_dst_norm_corr": float("nan"),
        "theta_bias_value": float("nan"),
        "topology_logit_mean": float("nan"),
        "topology_logit_std": float("nan"),
        "topology_logit_p05": float("nan"),
        "topology_logit_p50": float("nan"),
        "topology_logit_p95": float("nan"),
        "topology_prob_mean": float("nan"),
        "topology_prob_std": float("nan"),
        "topology_prob_p05": float("nan"),
        "topology_prob_p50": float("nan"),
        "topology_prob_p95": float("nan"),
        "src_norm_out_degree_corr": float("nan"),
        "dst_norm_in_degree_corr": float("nan"),
    }

    if str(getattr(model.liquid, "mode", "")).lower() != "learned_lowrank":
        return nan_metrics

    src = model.liquid.src_embed.detach().cpu()
    dst = model.liquid.dst_embed.detach().cpu()
    theta = model.liquid.get_theta().detach().cpu()
    probs = torch.sigmoid(theta)
    src_norm = src.norm(dim=1)
    dst_norm = dst.norm(dim=1)
    active = _adjacency_without_self_loops(mask)
    out_degree = active.float().sum(dim=1)
    in_degree = active.float().sum(dim=0)

    return {
        "src_embed_norm_mean": safe_float(src_norm.mean()),
        "src_embed_norm_std": safe_float(src_norm.std()),
        "src_embed_norm_min": safe_float(src_norm.min()),
        "src_embed_norm_max": safe_float(src_norm.max()),
        "dst_embed_norm_mean": safe_float(dst_norm.mean()),
        "dst_embed_norm_std": safe_float(dst_norm.std()),
        "dst_embed_norm_min": safe_float(dst_norm.min()),
        "dst_embed_norm_max": safe_float(dst_norm.max()),
        "src_dst_norm_corr": safe_pearson_corr(src_norm, dst_norm),
        "theta_bias_value": safe_float(model.liquid.theta_bias.detach().cpu()),
        "topology_logit_mean": safe_float(theta.mean()),
        "topology_logit_std": safe_float(theta.std()),
        "topology_logit_p05": safe_float(torch.quantile(theta.reshape(-1), 0.05)),
        "topology_logit_p50": safe_float(torch.quantile(theta.reshape(-1), 0.50)),
        "topology_logit_p95": safe_float(torch.quantile(theta.reshape(-1), 0.95)),
        "topology_prob_mean": safe_float(probs.mean()),
        "topology_prob_std": safe_float(probs.std()),
        "topology_prob_p05": safe_float(torch.quantile(probs.reshape(-1), 0.05)),
        "topology_prob_p50": safe_float(torch.quantile(probs.reshape(-1), 0.50)),
        "topology_prob_p95": safe_float(torch.quantile(probs.reshape(-1), 0.95)),
        "src_norm_out_degree_corr": safe_pearson_corr(src_norm, out_degree),
        "dst_norm_in_degree_corr": safe_pearson_corr(dst_norm, in_degree),
    }


def compute_readout_topology_metrics(
    model: torch.nn.Module, mask: torch.Tensor
) -> dict[str, float]:
    """Correlate readout importance with topology centrality."""
    active = _adjacency_without_self_loops(mask)
    in_degree = active.float().sum(dim=0)
    out_degree = active.float().sum(dim=1)
    total_degree = in_degree + out_degree

    readout_weight = model.readout.weight.detach().cpu()
    readout_importance = readout_weight.norm(dim=0)
    k = min(10, total_degree.numel())
    top_degree = set(total_degree.topk(k).indices.tolist()) if k > 0 else set()
    top_readout = set(readout_importance.topk(k).indices.tolist()) if k > 0 else set()
    overlap = len(top_degree & top_readout) / k if k > 0 else float("nan")

    return {
        "readout_in_degree_corr": safe_pearson_corr(readout_importance, in_degree),
        "readout_out_degree_corr": safe_pearson_corr(readout_importance, out_degree),
        "readout_total_degree_corr": safe_pearson_corr(readout_importance, total_degree),
        "top_degree_top_readout_overlap": overlap,
    }


def make_activity_placeholder(row_meta: dict[str, Any]) -> dict[str, Any]:
    """Create the Phase 1 placeholder row for activity metrics."""
    return {
        "experiment_name": row_meta["experiment_name"],
        "method_label": row_meta["method_label"],
        "recurrent_mode": row_meta["recurrent_mode"],
        "seed": row_meta["seed"],
        "activity_diagnostics_available": False,
        "num_batches": float("nan"),
        "num_samples": float("nan"),
        "mean_firing_rate": float("nan"),
        "max_firing_rate": float("nan"),
        "active_neurons_gt_005": float("nan"),
        "rec_input_abs_ratio": float("nan"),
        "class_mean_cosine_mean": float("nan"),
        "mean_logit_margin": float("nan"),
        "accuracy_on_diagnostic_batches": float("nan"),
        "skip_reason": (
            "Phase 1 graph-only pipeline; existing diagnose_liquid.py can still be "
            "used for single-run activity diagnostics"
        ),
    }


def build_summary_row(graph_row: dict[str, Any], activity_row: dict[str, Any]) -> dict[str, Any]:
    """Merge the report rows into the compact summary schema."""
    merged = {**graph_row, **activity_row}
    return {column: merged.get(column, float("nan")) for column in SUMMARY_COLUMNS}


def write_outputs(
    graph_rows: list[dict[str, Any]],
    activity_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    """Write standardized CSV and JSON outputs."""
    if pd is None:
        raise RuntimeError(
            "pandas is required for topology_diagnostics.py. Install it before running this script."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(graph_rows, columns=GRAPH_COLUMNS).to_csv(
        output_dir / "graph_metrics.csv", index=False
    )
    pd.DataFrame(activity_rows, columns=ACTIVITY_COLUMNS).to_csv(
        output_dir / "activity_metrics.csv", index=False
    )
    pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).to_csv(
        output_dir / "summary_metrics.csv", index=False
    )
    with open(output_dir / "diagnostics_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _selection_metadata(spec: dict[str, str | None], cfg: Any) -> dict[str, Any]:
    return {
        "experiment_name": getattr(cfg, "experiment_name", Path(spec["experiment_dir"]).name),
        "experiment_dir": spec["experiment_dir"],
        "config_path": spec["config_path"],
        "checkpoint_path": spec["checkpoint_path"],
        "use_validation": getattr(cfg, "use_validation", None),
        "val_fraction": getattr(cfg, "val_fraction", None),
        "val_seed": getattr(cfg, "val_seed", None),
        "topology_freeze_metric": getattr(cfg.liquid, "topology_freeze_metric", None),
        "topology_freeze_min_epoch": getattr(cfg.liquid, "topology_freeze_min_epoch", None),
        "topology_freeze_patience": getattr(cfg.liquid, "topology_freeze_patience", None),
        "topology_freeze_rollback_best": getattr(
            cfg.liquid, "topology_freeze_rollback_best", None
        ),
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    specs = resolve_experiment_specs(args)

    warnings: list[str] = []
    skipped_metrics: list[str] = []
    if args.skip_path_metrics:
        skipped_metrics.extend(
            [
                "reachability_ratio",
                "avg_shortest_path_reachable",
                "effective_diameter_p90",
            ]
        )
    if args.activity_placeholder_only:
        skipped_metrics.extend(
            [
                "mean_firing_rate",
                "max_firing_rate",
                "active_neurons_gt_005",
                "rec_input_abs_ratio",
                "class_mean_cosine_mean",
                "mean_logit_margin",
                "accuracy_on_diagnostic_batches",
            ]
        )

    graph_rows: list[dict[str, Any]] = []
    activity_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    processed_experiments: list[dict[str, Any]] = []

    for spec in specs:
        try:
            cfg, model = load_config_and_model(
                str(spec["config_path"]), str(spec["checkpoint_path"]), device
            )
            mask = extract_recurrent_mask(model)
            experiment_name = str(
                getattr(cfg, "experiment_name", Path(spec["experiment_dir"]).name)
            )
            row_meta = {
                "experiment_name": experiment_name,
                "experiment_dir": str(spec["experiment_dir"]),
                "config_path": str(spec["config_path"]),
                "checkpoint_path": str(spec["checkpoint_path"]),
                "recurrent_mode": str(getattr(cfg.liquid, "recurrent_mode", "unknown")),
                "seed": int(getattr(cfg, "seed", -1)),
                "method_label": infer_method_label(str(spec["experiment_dir"]), cfg),
            }

            graph_row = {
                **row_meta,
                **compute_core_graph_metrics(
                    mask,
                    model,
                    cfg,
                    skip_path_metrics=args.skip_path_metrics,
                    use_networkx=nx is not None,
                ),
                **compute_lowrank_metrics(model, mask),
                **compute_readout_topology_metrics(model, mask),
            }
            activity_row = make_activity_placeholder(row_meta)
            summary_row = build_summary_row(graph_row, activity_row)

            selection_meta = _selection_metadata(spec, cfg)
            processed_experiments.append(selection_meta)
            if selection_meta["use_validation"] in (None, False):
                warnings.append(
                    f"{experiment_name}: This run may not be validation-selected; "
                    "diagnostics do not create leakage, but selection protocol should be checked."
                )

            graph_rows.append(graph_row)
            activity_rows.append(activity_row)
            summary_rows.append(summary_row)
        except Exception as exc:
            warnings.append(
                f"Failed to process {spec['experiment_dir']}: {type(exc).__name__}: {exc}"
            )

    if not graph_rows:
        raise RuntimeError("No experiments were processed successfully.")

    output_dir = Path(args.output_dir).resolve()
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "output_dir": str(output_dir),
        "num_experiments": len(specs),
        "processed_experiments": processed_experiments,
        "warnings": warnings,
        "skipped_metrics": skipped_metrics,
        "device": str(device),
        "path_metrics_skipped": bool(args.skip_path_metrics),
        "activity_placeholder_only": bool(args.activity_placeholder_only),
        "pandas_available": pd is not None,
        "networkx_available": nx is not None,
    }
    write_outputs(graph_rows, activity_rows, summary_rows, metadata, output_dir)

    print(f"[OK] processed {len(graph_rows)} experiments")
    print("[OK] wrote graph_metrics.csv")
    print("[OK] wrote activity_metrics.csv")
    print("[OK] wrote summary_metrics.csv")


if __name__ == "__main__":
    main()
