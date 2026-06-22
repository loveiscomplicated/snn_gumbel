"""
Neuron-level topology/activity diagnostics for saved LSM checkpoints.

This script complements the existing graph-level diagnostics by producing one
row per liquid neuron, joining topology, input projection, firing/activity,
ALIF adaptation, and readout-weight importance.

Example:
    python scripts/neuron_diagnostics.py \
      --experiments experiments/run_a experiments/run_b \
      --output-dir runs/diagnostics/neuron_level \
      --batches 4 \
      --make-plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import get_dataloaders
from src.lsm.trainer import build_model
from src.utils.config import load_config


NEURON_COLUMNS = [
    "experiment_name",
    "experiment_dir",
    "checkpoint_path",
    "seed",
    "group_label",
    "checkpoint_epoch",
    "best_epoch",
    "best_val_acc",
    "best_test_acc_at_best_val",
    "oracle_best_test_epoch",
    "oracle_best_test_acc",
    "neuron_id",
    "ei_type",
    "input_fan_in",
    "input_density",
    "input_l1_norm",
    "input_l2_norm",
    "in_degree",
    "out_degree",
    "total_degree",
    "reciprocal_degree",
    "triangle_count",
    "firing_rate",
    "input_current_mean",
    "input_current_std",
    "input_current_abs_mean",
    "recurrent_current_mean",
    "recurrent_current_std",
    "recurrent_current_abs_mean",
    "rec_input_std_ratio",
    "rec_input_abs_ratio",
    "adaptation_mean",
    "adaptation_abs_mean",
    "adaptation_max",
    "readout_spike_weight_norm",
    "readout_adapt_weight_norm",
    "readout_total_weight_norm",
    "adapt_readout_contribution_proxy",
]

CORRELATION_PAIRS = [
    ("input_l2_norm", "firing_rate"),
    ("out_degree", "firing_rate"),
    ("in_degree", "adaptation_mean"),
    ("firing_rate", "readout_total_weight_norm"),
    ("adaptation_mean", "readout_adapt_weight_norm"),
    ("adaptation_abs_mean", "adapt_readout_contribution_proxy"),
    ("input_l2_norm", "out_degree"),
    ("input_l2_norm", "readout_total_weight_norm"),
    ("rec_input_abs_ratio", "firing_rate"),
]

HEATMAP_METRICS = [
    "input_l2_norm",
    "input_fan_in",
    "in_degree",
    "out_degree",
    "total_degree",
    "reciprocal_degree",
    "triangle_count",
    "firing_rate",
    "input_current_abs_mean",
    "recurrent_current_abs_mean",
    "rec_input_abs_ratio",
    "adaptation_mean",
    "adaptation_abs_mean",
    "readout_spike_weight_norm",
    "readout_adapt_weight_norm",
    "readout_total_weight_norm",
    "adapt_readout_contribution_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create neuron-level diagnostics for LSM checkpoints."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Experiment directories containing config.yaml and checkpoints/best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for neuron_table.csv, correlations.csv, summary JSON, and figures.",
    )
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for trace collection.",
    )
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate PNG diagnostic figures.",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but CUDA is not available.")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS but MPS is not available.")
    return torch.device(device_arg)


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value)
    return float(value)


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def pearson_corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation for Spearman correlation without scipy."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size < 2:
        return float("nan")
    return pearson_corr(rankdata(a), rankdata(b))


def gini(values: torch.Tensor) -> float:
    x = values.detach().float().cpu().reshape(-1).clamp(min=0)
    if x.numel() == 0 or x.sum().item() == 0.0:
        return 0.0
    sorted_x = x.sort().values
    n = sorted_x.numel()
    index = torch.arange(1, n + 1, dtype=sorted_x.dtype)
    coeff = (2.0 * torch.dot(index, sorted_x) / (n * sorted_x.sum())) - ((n + 1.0) / n)
    return float(coeff.clamp(0.0, 1.0).item())


def graph_node_metrics(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    active = mask.detach().cpu().bool().clone()
    if active.ndim != 2 or active.shape[0] != active.shape[1]:
        raise ValueError(f"Expected square recurrent mask, got {tuple(active.shape)}")
    active.fill_diagonal_(False)
    in_degree = active.float().sum(dim=0)
    out_degree = active.float().sum(dim=1)
    reciprocal_degree = (active & active.T).float().sum(dim=1)
    triangle_count = torch.diag(active.float() @ active.float() @ active.float())
    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": in_degree + out_degree,
        "reciprocal_degree": reciprocal_degree,
        "triangle_count": triangle_count,
    }


def readout_importance(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    weight = model.readout.weight.detach().float().cpu()
    n_liquid = int(model.n_liquid)
    if getattr(model, "readout_mode", "") == "spike_adaptation_concat":
        spike_weight = weight[:, :n_liquid]
        adapt_weight = weight[:, n_liquid : 2 * n_liquid]
        spike_norm = spike_weight.norm(dim=0)
        adapt_norm = adapt_weight.norm(dim=0)
        total_norm = torch.sqrt(spike_norm.pow(2) + adapt_norm.pow(2))
    else:
        spike_norm = weight[:, :n_liquid].norm(dim=0)
        adapt_norm = torch.full((n_liquid,), float("nan"))
        total_norm = spike_norm
    return {
        "readout_spike_weight_norm": spike_norm,
        "readout_adapt_weight_norm": adapt_norm,
        "readout_total_weight_norm": total_norm,
    }


def input_projection_metrics(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    weight = model.input_proj.effective_weight().detach().float().cpu()
    mask = getattr(model.input_proj, "mask", weight != 0).detach().float().cpu()
    fan_in = mask.sum(dim=0)
    return {
        "input_fan_in": fan_in,
        "input_density": fan_in / max(mask.shape[0], 1),
        "input_l1_norm": weight.abs().sum(dim=0),
        "input_l2_norm": weight.norm(dim=0),
    }


class PerNeuronAccumulator:
    def __init__(self, n_liquid: int, has_adaptation: bool):
        self.n_liquid = n_liquid
        self.has_adaptation = has_adaptation
        self.count = 0
        self.spike_sum = torch.zeros(n_liquid)
        self.input_sum = torch.zeros(n_liquid)
        self.input_sum_sq = torch.zeros(n_liquid)
        self.input_abs_sum = torch.zeros(n_liquid)
        self.recurrent_sum = torch.zeros(n_liquid)
        self.recurrent_sum_sq = torch.zeros(n_liquid)
        self.recurrent_abs_sum = torch.zeros(n_liquid)
        self.adaptation_sum = torch.zeros(n_liquid)
        self.adaptation_abs_sum = torch.zeros(n_liquid)
        self.adaptation_max = torch.full((n_liquid,), float("nan"))

    def update(self, traces: dict[str, torch.Tensor]) -> None:
        spikes = traces["spikes"].detach().float().cpu()
        input_current = traces["input_current"].detach().float().cpu()
        recurrent_current = traces["recurrent_current"].detach().float().cpu()
        n = spikes.shape[0] * spikes.shape[1]
        self.count += n
        self.spike_sum += spikes.sum(dim=(0, 1))
        self.input_sum += input_current.sum(dim=(0, 1))
        self.input_sum_sq += input_current.pow(2).sum(dim=(0, 1))
        self.input_abs_sum += input_current.abs().sum(dim=(0, 1))
        self.recurrent_sum += recurrent_current.sum(dim=(0, 1))
        self.recurrent_sum_sq += recurrent_current.pow(2).sum(dim=(0, 1))
        self.recurrent_abs_sum += recurrent_current.abs().sum(dim=(0, 1))
        if self.has_adaptation and "adaptation" in traces:
            adaptation = traces["adaptation"].detach().float().cpu()
            self.adaptation_sum += adaptation.sum(dim=(0, 1))
            self.adaptation_abs_sum += adaptation.abs().sum(dim=(0, 1))
            batch_max = adaptation.reshape(-1, self.n_liquid).max(dim=0).values
            if torch.isnan(self.adaptation_max).all():
                self.adaptation_max = batch_max
            else:
                self.adaptation_max = torch.maximum(self.adaptation_max, batch_max)

    def finalize(self) -> dict[str, torch.Tensor]:
        denom = max(self.count, 1)
        input_mean = self.input_sum / denom
        recurrent_mean = self.recurrent_sum / denom
        input_var = (self.input_sum_sq / denom - input_mean.pow(2)).clamp(min=0.0)
        recurrent_var = (
            self.recurrent_sum_sq / denom - recurrent_mean.pow(2)
        ).clamp(min=0.0)
        adaptation_mean = (
            self.adaptation_sum / denom
            if self.has_adaptation
            else torch.full((self.n_liquid,), float("nan"))
        )
        adaptation_abs_mean = (
            self.adaptation_abs_sum / denom
            if self.has_adaptation
            else torch.full((self.n_liquid,), float("nan"))
        )
        adaptation_max = (
            self.adaptation_max
            if self.has_adaptation
            else torch.full((self.n_liquid,), float("nan"))
        )
        return {
            "firing_rate": self.spike_sum / denom,
            "input_current_mean": input_mean,
            "input_current_std": input_var.sqrt(),
            "input_current_abs_mean": self.input_abs_sum / denom,
            "recurrent_current_mean": recurrent_mean,
            "recurrent_current_std": recurrent_var.sqrt(),
            "recurrent_current_abs_mean": self.recurrent_abs_sum / denom,
            "adaptation_mean": adaptation_mean,
            "adaptation_abs_mean": adaptation_abs_mean,
            "adaptation_max": adaptation_max,
        }


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"[warn] {checkpoint_path}: missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    model.eval()
    if hasattr(model.liquid, "unlock_epoch_mask"):
        model.liquid.unlock_epoch_mask()
    return checkpoint


def collect_batches(loader, n_batches: int):
    batches = []
    for idx, batch in enumerate(loader):
        if idx >= n_batches:
            break
        batches.append(batch)
    return batches


def infer_group_label(exp_dir: Path, cfg: Any) -> str:
    text = f"{exp_dir.name} {getattr(cfg, 'experiment_name', '')}".lower()
    if "b010_inc0125_biaslr05" in text:
        return "alif_b010_inc0125_biaslr05"
    if "alif_beta_init_01_incre_0125" in text:
        return "alif_b010_inc0125_biaslr1"
    if "learned_input_proj_fdi" in text and "alif" not in text:
        return "lif_learned_input_fdi"
    return str(getattr(cfg, "experiment_name", exp_dir.name))


def log_oracle_reference(exp_dir: Path) -> dict[str, float | int | None]:
    log_path = exp_dir / "logs" / "train.jsonl"
    if not log_path.exists():
        return {
            "oracle_best_test_epoch": None,
            "oracle_best_test_acc": None,
        }
    rows = []
    for line in log_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return {
            "oracle_best_test_epoch": None,
            "oracle_best_test_acc": None,
        }
    best = max(rows, key=lambda row: row.get("test_acc", float("-inf")))
    return {
        "oracle_best_test_epoch": int(best["epoch"]),
        "oracle_best_test_acc": float(best["test_acc"]),
    }


def build_neuron_rows(
    exp_dir: Path,
    cfg: Any,
    model: torch.nn.Module,
    checkpoint: dict,
    checkpoint_path: Path,
    batches,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    has_adaptation = str(getattr(cfg.liquid, "neuron_type", "")) == "alif"
    accumulator = PerNeuronAccumulator(model.n_liquid, has_adaptation=has_adaptation)
    with torch.no_grad():
        for x, _ in batches:
            logits, traces = model(x.to(device), tau=cfg.tau_end, return_traces=True)
            del logits
            accumulator.update(traces)

    graph = graph_node_metrics(model.liquid.get_binary_mask())
    input_metrics = input_projection_metrics(model)
    readout = readout_importance(model)
    activity = accumulator.finalize()
    rec_input_std_ratio = activity["recurrent_current_std"] / activity[
        "input_current_std"
    ].clamp(min=1e-12)
    rec_input_abs_ratio = activity["recurrent_current_abs_mean"] / activity[
        "input_current_abs_mean"
    ].clamp(min=1e-12)
    adapt_readout_contribution_proxy = activity["adaptation_abs_mean"] * readout[
        "readout_adapt_weight_norm"
    ]

    dale = model.liquid.dale_sign.detach().cpu().reshape(-1)
    oracle = log_oracle_reference(exp_dir)
    meta = {
        "experiment_name": getattr(cfg, "experiment_name", exp_dir.name),
        "experiment_dir": str(exp_dir),
        "checkpoint_path": str(checkpoint_path),
        "seed": int(getattr(cfg, "seed", -1)),
        "group_label": infer_group_label(exp_dir, cfg),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_acc": checkpoint.get("best_val_acc"),
        "best_test_acc_at_best_val": checkpoint.get("best_test_acc_at_best_val"),
        **oracle,
    }

    rows = []
    for neuron_id in range(model.n_liquid):
        row = {
            **meta,
            "neuron_id": neuron_id,
            "ei_type": "E" if dale[neuron_id].item() > 0 else "I",
        }
        for source in (input_metrics, graph, activity, readout):
            for key, values in source.items():
                row[key] = _safe_float(values[neuron_id])
        row["rec_input_std_ratio"] = _safe_float(rec_input_std_ratio[neuron_id])
        row["rec_input_abs_ratio"] = _safe_float(rec_input_abs_ratio[neuron_id])
        row["adapt_readout_contribution_proxy"] = _safe_float(
            adapt_readout_contribution_proxy[neuron_id]
        )
        rows.append(row)

    run_summary = {
        **meta,
        "num_neurons": model.n_liquid,
        "num_samples": int(sum(x.shape[0] for x, _ in batches)),
        "num_batches": len(batches),
        "in_degree_gini": gini(graph["in_degree"]),
        "out_degree_gini": gini(graph["out_degree"]),
        "input_l2_gini": gini(input_metrics["input_l2_norm"]),
        "firing_rate_gini": gini(activity["firing_rate"]),
        "readout_total_weight_gini": gini(readout["readout_total_weight_norm"]),
    }
    return rows, run_summary


def top_ids(rows: list[dict[str, Any]], key: str, topk: int) -> list[int]:
    valid = [
        row
        for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]
    valid.sort(key=lambda row: float(row[key]), reverse=True)
    return [int(row["neuron_id"]) for row in valid[:topk]]


def threshold_flags(rows: list[dict[str, Any]], key: str, frac: float = 0.10) -> set[int]:
    valid = [
        row
        for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]
    if not valid:
        return set()
    k = max(1, int(math.ceil(len(valid) * frac)))
    valid.sort(key=lambda row: float(row[key]), reverse=True)
    return {int(row["neuron_id"]) for row in valid[:k]}


def summarize_high_firing(rows: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    high_firing = {
        int(row["neuron_id"])
        for row in rows
        if math.isfinite(float(row.get("firing_rate", float("nan"))))
        and float(row["firing_rate"]) > 0.20
    }
    top_input = threshold_flags(rows, "input_l2_norm")
    top_recurrent = threshold_flags(rows, "rec_input_abs_ratio")
    top_readout = threshold_flags(rows, "readout_total_weight_norm")
    top_adaptation = threshold_flags(rows, "adaptation_mean")
    return {
        "top_in_degree": top_ids(rows, "in_degree", topk),
        "top_out_degree": top_ids(rows, "out_degree", topk),
        "top_total_degree": top_ids(rows, "total_degree", topk),
        "top_input_norm": top_ids(rows, "input_l2_norm", topk),
        "top_firing_rate": top_ids(rows, "firing_rate", topk),
        "top_recurrent_ratio": top_ids(rows, "rec_input_abs_ratio", topk),
        "top_readout_weight": top_ids(rows, "readout_total_weight_norm", topk),
        "top_adaptation": top_ids(rows, "adaptation_mean", topk),
        "high_firing_gt020_count": len(high_firing),
        "high_firing_and_top_input_count": len(high_firing & top_input),
        "high_firing_and_top_recurrent_count": len(high_firing & top_recurrent),
        "high_firing_and_top_readout_count": len(high_firing & top_readout),
        "high_firing_and_top_adaptation_count": len(high_firing & top_adaptation),
    }


def correlation_rows(
    rows_by_run: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for experiment_name, rows in rows_by_run.items():
        meta = rows[0]
        for x_key, y_key in CORRELATION_PAIRS:
            x = [float(row.get(x_key, float("nan"))) for row in rows]
            y = [float(row.get(y_key, float("nan"))) for row in rows]
            output.append(
                {
                    "experiment_name": experiment_name,
                    "experiment_dir": meta["experiment_dir"],
                    "seed": meta["seed"],
                    "group_label": meta["group_label"],
                    "x": x_key,
                    "y": y_key,
                    "pearson": pearson_corr(x, y),
                    "spearman": spearman_corr(x, y),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = columns or list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(type(obj))


def values_for(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=float)


def plot_outputs(rows_by_run: dict[str, list[dict[str, Any]]], figures_dir: Path) -> None:
    cache_dir = figures_dir.parent / ".plot_cache"
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_dir / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    scatter_pairs = [
        ("input_l2_norm", "firing_rate", "input_norm_vs_firing"),
        ("out_degree", "firing_rate", "out_degree_vs_firing"),
        ("rec_input_abs_ratio", "firing_rate", "rec_input_ratio_vs_firing"),
        ("firing_rate", "readout_total_weight_norm", "firing_vs_readout_weight"),
        (
            "adaptation_mean",
            "readout_adapt_weight_norm",
            "adaptation_vs_readout_adapt_weight",
        ),
        (
            "adaptation_mean",
            "adapt_readout_contribution_proxy",
            "adaptation_mean_vs_adapt_readout_contribution",
        ),
        (
            "adaptation_abs_mean",
            "adapt_readout_contribution_proxy",
            "adaptation_abs_mean_vs_adapt_readout_contribution",
        ),
    ]

    for experiment_name, rows in rows_by_run.items():
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in experiment_name)

        corr = np.full((len(HEATMAP_METRICS), len(HEATMAP_METRICS)), np.nan)
        for i, x_key in enumerate(HEATMAP_METRICS):
            for j, y_key in enumerate(HEATMAP_METRICS):
                corr[i, j] = pearson_corr(values_for(rows, x_key), values_for(rows, y_key))
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(len(HEATMAP_METRICS)))
        ax.set_yticks(range(len(HEATMAP_METRICS)))
        ax.set_xticklabels(HEATMAP_METRICS, rotation=90, fontsize=7)
        ax.set_yticklabels(HEATMAP_METRICS, fontsize=7)
        ax.set_title(f"Neuron metric correlations\n{experiment_name}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures_dir / f"correlation_heatmap_{safe_name}.png", dpi=160)
        plt.close(fig)

        for x_key, y_key, stem in scatter_pairs:
            x = values_for(rows, x_key)
            y = values_for(rows, y_key)
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 2:
                continue
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            ax.scatter(x[valid], y[valid], s=12, alpha=0.75)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(f"{stem}\n{experiment_name}", fontsize=9)
            fig.tight_layout()
            fig.savefig(figures_dir / f"{stem}_{safe_name}.png", dpi=160)
            plt.close(fig)

        hist_specs = [
            ("in_degree", "degree_hist"),
            ("firing_rate", "firing_hist"),
            ("input_l2_norm", "input_norm_hist"),
            ("readout_total_weight_norm", "readout_weight_hist"),
        ]
        for key, stem in hist_specs:
            vals = values_for(rows, key)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            ax.hist(vals, bins=30)
            ax.set_xlabel(key)
            ax.set_ylabel("count")
            ax.set_title(f"{key}\n{experiment_name}", fontsize=9)
            fig.tight_layout()
            fig.savefig(figures_dir / f"{stem}_{safe_name}.png", dpi=160)
            plt.close(fig)


def process_experiment(
    exp_dir: Path,
    device: torch.device,
    n_batches: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_path = exp_dir / "config.yaml"
    checkpoint_path = exp_dir / "checkpoints" / "best.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    cfg = load_config(str(config_path))
    torch.manual_seed(int(getattr(cfg, "seed", 0)))
    model = build_model(cfg, device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    _, test_loader = get_dataloaders(cfg)
    batches = collect_batches(test_loader, n_batches)
    if not batches:
        raise RuntimeError(f"No diagnostic batches collected for {exp_dir}")
    return build_neuron_rows(exp_dir, cfg, model, checkpoint, checkpoint_path, batches, device)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = select_device(args.device)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    rows_by_run: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []

    for exp in args.experiments:
        exp_dir = Path(exp)
        try:
            print(f"[diagnose] {exp_dir}", flush=True)
            rows, summary = process_experiment(exp_dir, device, args.batches)
            summary.update(summarize_high_firing(rows, args.topk))
            all_rows.extend(rows)
            summaries.append(summary)
            rows_by_run[str(summary["experiment_name"])] = rows
        except Exception as exc:
            warnings.append(f"{exp_dir}: {type(exc).__name__}: {exc}")

    if not all_rows:
        raise RuntimeError("No experiments were processed successfully.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "neuron_table.csv", all_rows, NEURON_COLUMNS)
    corr_rows = correlation_rows(rows_by_run)
    write_csv(
        output_dir / "correlations.csv",
        corr_rows,
        [
            "experiment_name",
            "experiment_dir",
            "seed",
            "group_label",
            "x",
            "y",
            "pearson",
            "spearman",
        ],
    )

    group_summary: dict[str, dict[str, Any]] = {}
    for group in sorted({summary["group_label"] for summary in summaries}):
        group_runs = [summary for summary in summaries if summary["group_label"] == group]
        group_summary[group] = {
            "num_runs": len(group_runs),
            "mean_best_test_acc_at_best_val": _mean(
                [summary.get("best_test_acc_at_best_val", float("nan")) for summary in group_runs]
            ),
            "mean_in_degree_gini": _mean([summary["in_degree_gini"] for summary in group_runs]),
            "mean_out_degree_gini": _mean([summary["out_degree_gini"] for summary in group_runs]),
            "mean_input_l2_gini": _mean([summary["input_l2_gini"] for summary in group_runs]),
            "mean_firing_rate_gini": _mean([summary["firing_rate_gini"] for summary in group_runs]),
            "mean_readout_total_weight_gini": _mean(
                [summary["readout_total_weight_gini"] for summary in group_runs]
            ),
        }

    payload = {
        "device": str(device),
        "num_runs": len(summaries),
        "num_neuron_rows": len(all_rows),
        "warnings": warnings,
        "runs": summaries,
        "groups": group_summary,
        "correlation_pairs": CORRELATION_PAIRS,
        "oracle_policy": (
            "Oracle best-test checkpoints are not reconstructed; train.jsonl oracle "
            "epochs are recorded as metadata only."
        ),
    }
    with (output_dir / "run_summary.json").open("w") as f:
        json.dump(payload, f, indent=2, default=json_default)

    if args.make_plots:
        plot_outputs(rows_by_run, output_dir / "figures")

    print(f"[OK] wrote {output_dir / 'neuron_table.csv'}")
    print(f"[OK] wrote {output_dir / 'correlations.csv'}")
    print(f"[OK] wrote {output_dir / 'run_summary.json'}")
    if args.make_plots:
        print(f"[OK] wrote figures under {output_dir / 'figures'}")
    if warnings:
        print("[WARN] some experiments failed:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
