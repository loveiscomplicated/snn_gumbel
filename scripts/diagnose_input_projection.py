"""
Diagnose input projection information loss and drive scale.

Checks:
  1. Fixed input projection matrix sparsity / fan-in / fan-out / rank
  2. Geometry preservation on sampled nonzero input timesteps
  3. Projected current scale relative to liquid thresholds

Usage:
    python scripts/diagnose_input_projection.py configs/lsm_shd_baseline.yaml
    python scripts/diagnose_input_projection.py experiments/<exp>/config.yaml \
        --checkpoint experiments/<exp>/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnose_liquid import (
    RunningStats,
    collect_batches,
    fmt_stats,
    get_dataloaders,
    get_device,
    load_checkpoint_if_requested,
    load_config,
    print_config_summary,
    print_header,
    print_input_spike_stats,
)
from src.lsm.trainer import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose LSM input projection information loss"
    )
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
        "--batches",
        type=int,
        default=4,
        help="Number of test batches to sample for geometry/current diagnostics",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=256,
        help="Max nonzero timestep inputs used for geometry preservation checks",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional JSON path for the full diagnostic output",
    )
    args, overrides = parser.parse_known_args()
    args.overrides = [item for item in overrides if item != "--"]
    return args


def _pairwise_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = x_centered.norm() * y_centered.norm()
    if denom <= 0:
        return float("nan")
    return float((x_centered * y_centered).sum().item() / denom.item())


def compute_projection_matrix_stats(model) -> dict:
    weight = model.input_proj.weight.detach().float().cpu()
    active = weight != 0
    fan_out = active.float().sum(dim=1)
    fan_in = active.float().sum(dim=0)
    row_norm = weight.norm(dim=1)
    col_norm = weight.norm(dim=0)

    max_rank = min(weight.shape)
    rank = int(torch.linalg.matrix_rank(weight).item())
    singular_values = torch.linalg.svdvals(weight)
    spectral_norm = singular_values.max().item() if singular_values.numel() else 0.0
    fro_norm_sq = weight.pow(2).sum().item()
    stable_rank = fro_norm_sq / max(spectral_norm * spectral_norm, 1e-12)

    return {
        "shape": list(weight.shape),
        "density": active.float().mean().item(),
        "active_edges": int(active.sum().item()),
        "rank": rank,
        "max_rank": max_rank,
        "rank_ratio": rank / max_rank if max_rank else float("nan"),
        "stable_rank": stable_rank,
        "zero_fan_out_inputs": int((fan_out == 0).sum().item()),
        "zero_fan_in_liquid": int((fan_in == 0).sum().item()),
        "fan_out_summary": {
            "mean": fan_out.mean().item(),
            "std": fan_out.std().item(),
            "min": fan_out.min().item(),
            "max": fan_out.max().item(),
        },
        "fan_in_summary": {
            "mean": fan_in.mean().item(),
            "std": fan_in.std().item(),
            "min": fan_in.min().item(),
            "max": fan_in.max().item(),
        },
        "row_norm_summary": {
            "mean": row_norm.mean().item(),
            "std": row_norm.std().item(),
            "min": row_norm.min().item(),
            "max": row_norm.max().item(),
        },
        "col_norm_summary": {
            "mean": col_norm.mean().item(),
            "std": col_norm.std().item(),
            "min": col_norm.min().item(),
            "max": col_norm.max().item(),
        },
        "abs_weight_summary": {
            "mean": weight.abs().mean().item(),
            "std": weight.abs().std().item(),
            "min": weight.abs().min().item(),
            "max": weight.abs().max().item(),
        },
    }


def collect_nonzero_timestep_inputs(
    batches: list[tuple[torch.Tensor, torch.Tensor]], max_samples: int
) -> torch.Tensor:
    if not batches:
        return torch.empty(0)
    inputs = torch.cat([x.float().reshape(-1, x.shape[-1]) for x, _ in batches], dim=0)
    nonzero = inputs.abs().sum(dim=1) > 0
    inputs = inputs[nonzero]
    if inputs.shape[0] > max_samples:
        idx = torch.linspace(0, inputs.shape[0] - 1, steps=max_samples).round().long()
        inputs = inputs[idx]
    return inputs


def compute_projection_geometry_stats(
    model,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    max_samples: int,
) -> dict:
    inputs = collect_nonzero_timestep_inputs(batches, max_samples=max_samples)
    if inputs.numel() == 0:
        return {
            "n_samples": 0,
            "norm_ratio_summary": None,
            "pairwise_l2_corr": float("nan"),
            "distance_scale_summary": None,
            "distance_shape_error_mean": float("nan"),
            "distance_shape_error_max": float("nan"),
        }

    weight = model.input_proj.weight.detach().float().cpu()
    projected = inputs @ weight

    input_norm = inputs.norm(dim=1)
    projected_norm = projected.norm(dim=1)
    valid_norm = input_norm > 0
    norm_ratio = projected_norm[valid_norm] / input_norm[valid_norm]

    raw_dist = torch.pdist(inputs, p=2)
    proj_dist = torch.pdist(projected, p=2)
    valid_dist = raw_dist > 1e-8
    if valid_dist.any():
        raw_dist = raw_dist[valid_dist]
        proj_dist = proj_dist[valid_dist]
        distance_scale = proj_dist / raw_dist
        scale_mean = distance_scale.mean().item()
        normalized_scale = distance_scale / max(scale_mean, 1e-12)
        distance_shape_error = (normalized_scale - 1.0).abs()
        pairwise_l2_corr = _pairwise_corr(raw_dist, proj_dist)
        distance_scale_summary = {
            "mean": distance_scale.mean().item(),
            "std": distance_scale.std().item(),
            "min": distance_scale.min().item(),
            "max": distance_scale.max().item(),
        }
        distance_shape_error_mean = distance_shape_error.mean().item()
        distance_shape_error_max = distance_shape_error.max().item()
    else:
        pairwise_l2_corr = float("nan")
        distance_scale_summary = None
        distance_shape_error_mean = float("nan")
        distance_shape_error_max = float("nan")

    return {
        "n_samples": int(inputs.shape[0]),
        "norm_ratio_summary": {
            "mean": norm_ratio.mean().item(),
            "std": norm_ratio.std().item(),
            "min": norm_ratio.min().item(),
            "max": norm_ratio.max().item(),
        },
        "pairwise_l2_corr": pairwise_l2_corr,
        "distance_scale_summary": distance_scale_summary,
        "distance_shape_error_mean": distance_shape_error_mean,
        "distance_shape_error_max": distance_shape_error_max,
    }


def compute_projected_current_stats(model, batches, device) -> dict:
    input_stats = RunningStats()
    with torch.no_grad():
        for x, _ in batches:
            x = x.to(device)
            for t in range(model.T):
                input_stats.update(model.input_proj(x[:, t]))

    threshold = model.liquid.threshold.detach().float().cpu()
    stats = input_stats.as_dict()
    threshold_mean = threshold.mean().item()
    threshold_min = threshold.min().item()
    return {
        "current_summary": stats,
        "threshold_summary": {
            "mean": threshold_mean,
            "std": threshold.std().item(),
            "min": threshold_min,
            "max": threshold.max().item(),
        },
        "abs_mean_over_threshold_mean": stats["abs_mean"] / max(threshold_mean, 1e-12),
        "max_abs_over_threshold_min": stats["max_abs"] / max(threshold_min, 1e-12),
    }


def print_projection_matrix_stats(stats: dict) -> None:
    print_header("4. Input projection matrix")
    print(f"  shape                 : {tuple(stats['shape'])}")
    print(f"  density               : {stats['density']:.4f}")
    print(
        f"  active edges          : {stats['active_edges']} / "
        f"{stats['shape'][0] * stats['shape'][1]}"
    )
    print(
        f"  rank                  : {stats['rank']} / {stats['max_rank']} "
        f"({stats['rank_ratio']:.4f})"
    )
    print(f"  stable rank           : {stats['stable_rank']:.2f}")
    print(
        f"  zero fan-out inputs   : {stats['zero_fan_out_inputs']} / {stats['shape'][0]}"
    )
    print(
        f"  zero fan-in liquids   : {stats['zero_fan_in_liquid']} / {stats['shape'][1]}"
    )
    print(
        f"  input fan-out         : mean={stats['fan_out_summary']['mean']:.2f}  "
        f"std={stats['fan_out_summary']['std']:.2f}  "
        f"min={stats['fan_out_summary']['min']:.0f}  "
        f"max={stats['fan_out_summary']['max']:.0f}"
    )
    print(
        f"  liquid fan-in         : mean={stats['fan_in_summary']['mean']:.2f}  "
        f"std={stats['fan_in_summary']['std']:.2f}  "
        f"min={stats['fan_in_summary']['min']:.0f}  "
        f"max={stats['fan_in_summary']['max']:.0f}"
    )
    print(
        f"  row ||W||             : mean={stats['row_norm_summary']['mean']:.4f}  "
        f"std={stats['row_norm_summary']['std']:.4f}  "
        f"min={stats['row_norm_summary']['min']:.4f}  "
        f"max={stats['row_norm_summary']['max']:.4f}"
    )
    print(
        f"  col ||W||             : mean={stats['col_norm_summary']['mean']:.4f}  "
        f"std={stats['col_norm_summary']['std']:.4f}  "
        f"min={stats['col_norm_summary']['min']:.4f}  "
        f"max={stats['col_norm_summary']['max']:.4f}"
    )
    abs_stats = stats["abs_weight_summary"]
    print(
        f"  |W|                   : mean={abs_stats['mean']:.4f}  "
        f"std={abs_stats['std']:.4f}  min={abs_stats['min']:.4f}  "
        f"max={abs_stats['max']:.4f}"
    )


def print_projection_geometry_stats(stats: dict) -> None:
    print("\n  sampled geometry preservation:")
    print(f"  nonzero timestep inputs: {stats['n_samples']}")
    if stats["n_samples"] < 2 or stats["norm_ratio_summary"] is None:
        print("  not enough nonzero timestep samples for geometry analysis")
        return

    norm_ratio = stats["norm_ratio_summary"]
    print(
        f"  ||xW|| / ||x||        : mean={norm_ratio['mean']:.4f}  "
        f"std={norm_ratio['std']:.4f}  min={norm_ratio['min']:.4f}  "
        f"max={norm_ratio['max']:.4f}"
    )
    print(f"  pairwise L2 corr      : {stats['pairwise_l2_corr']:.4f}")
    distance_scale = stats["distance_scale_summary"]
    if distance_scale is None:
        print("  pairwise distance check unavailable")
        return
    print(
        f"  distance scale        : mean={distance_scale['mean']:.4f}  "
        f"std={distance_scale['std']:.4f}  min={distance_scale['min']:.4f}  "
        f"max={distance_scale['max']:.4f}"
    )
    print(
        f"  scale-free shape err  : mean={stats['distance_shape_error_mean']:.4f}  "
        f"max={stats['distance_shape_error_max']:.4f}"
    )


def print_projected_current_stats(stats: dict) -> None:
    print("\n  projected current vs threshold:")
    print(f"  projected current     : {fmt_stats(stats['current_summary'])}")
    thr = stats["threshold_summary"]
    print(
        f"  threshold             : mean={thr['mean']:.4f}  std={thr['std']:.4f}  "
        f"min={thr['min']:.4f}  max={thr['max']:.4f}"
    )
    print(
        f"  |input| / thr_mean    : {stats['abs_mean_over_threshold_mean']:.4f}"
    )
    print(
        f"  max|input| / thr_min  : {stats['max_abs_over_threshold_min']:.4f}"
    )


def _json_default(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(type(obj))


def save_json(path: str, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"\nSaved JSON -> {out_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.overrides)

    device = get_device()
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, device)
    load_checkpoint_if_requested(model, args.checkpoint, device)
    model.eval()
    model.liquid.unlock_epoch_mask()

    print("=" * 72)
    print("[Input projection diagnostic]")
    print_config_summary(cfg, model, args.checkpoint)

    _, test_loader = get_dataloaders(cfg)
    batches = collect_batches(test_loader, args.batches)
    print_input_spike_stats(batches)

    matrix_stats = compute_projection_matrix_stats(model)
    geometry_stats = compute_projection_geometry_stats(
        model, batches, max_samples=args.max_samples
    )
    current_stats = compute_projected_current_stats(model, batches, device)

    print_projection_matrix_stats(matrix_stats)
    print_projection_geometry_stats(geometry_stats)
    print_projected_current_stats(current_stats)
    print("\nInterpretation guide:")
    print("  higher pairwise L2 corr and lower scale-free shape error imply less geometric distortion")
    print(
        "  full rank only means no extra collapse beyond the "
        f"{matrix_stats['shape'][0]}->{matrix_stats['shape'][1]} bottleneck"
    )
    print("  low |input| / threshold suggests weak input drive even if geometry is preserved")

    if args.out_json:
        save_json(
            args.out_json,
            {
                "checkpoint": args.checkpoint,
                "config": args.config,
                "matrix": matrix_stats,
                "geometry": geometry_stats,
                "current": current_stats,
            },
        )


if __name__ == "__main__":
    main()
