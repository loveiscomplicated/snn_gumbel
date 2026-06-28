"""Post-hoc diagnostics for learned-lowrank recurrent runaway hypotheses.

This module is intentionally read-only with respect to training artifacts.  It
loads existing runs, centers checkpoint-level analysis on validation-selected
``best.pt``, and records missing inputs as insufficient evidence instead of
failing the whole diagnostic job.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-12
PERCENTILE_EVENT_Q = 0.95
DENSITY_TOL = 1e-6

EPOCH_ALIASES: dict[str, list[str]] = {
    "epoch": ["epoch"],
    "val_acc": ["val_acc", "validation_acc", "valid_acc"],
    "test_acc": ["test_acc", "test_acc_at_epoch"],
    "mean_firing_rate": ["mean_firing_rate", "firing_rate_mean"],
    "max_firing_rate": ["max_firing_rate", "firing_rate_max"],
    "mean_adaptation": ["mean_adaptation", "adaptation_mean"],
    "max_adaptation": ["max_adaptation", "adaptation_max"],
    "theta_grad_norm": [
        "theta_grad_norm_pre_clip",
        "topology_grad_pre_clip",
        "theta_grad_pre_clip",
        "theta_grad_norm",
        "topology_grad_norm",
        "topology_grad_post_clip",
        "theta_grad_post_clip",
    ],
    "w_raw_grad_norm": ["w_raw_grad_norm", "w_raw_grad_pre_clip"],
    "grad_norm": ["grad_norm"],
    "hard_density": ["hard_density", "sparsity", "recurrent_density"],
    "sparsity": ["sparsity", "hard_density"],
    "theta_bias": ["theta_bias", "theta_bias_value"],
    "rec_input_abs_ratio": [
        "rec_input_abs_ratio",
        "recurrent_input_abs_ratio",
        "recurrent_to_input_abs_ratio",
    ],
}

CORRELATION_PAIRS = [
    ("expected_in_prob", "in_degree"),
    ("expected_out_prob", "out_degree"),
    ("expected_degree_score", "total_degree"),
    ("src_norm", "out_degree"),
    ("dst_norm", "in_degree"),
    ("row_prob_entropy", "out_degree"),
    ("col_prob_entropy", "in_degree"),
    ("in_degree", "recurrent_current_abs_mean"),
    ("out_degree", "firing_rate"),
    ("weighted_in_abs_strength", "recurrent_current_abs_mean"),
    ("weighted_out_abs_strength", "firing_rate"),
    ("incoming_exc_abs_strength", "firing_rate"),
    ("incoming_inh_abs_strength", "firing_rate"),
    ("incoming_ei_abs_balance", "firing_rate"),
    ("recurrent_current_abs_mean", "firing_rate"),
    ("rec_input_abs_ratio", "firing_rate"),
    ("input_l2_norm", "firing_rate"),
    ("firing_rate", "readout_total_weight_norm"),
    ("adaptation_abs_mean", "adapt_readout_contribution_proxy"),
]

TOPK_METRICS = [
    "expected_degree_score",
    "expected_in_prob",
    "expected_out_prob",
    "src_norm",
    "dst_norm",
    "in_degree",
    "out_degree",
    "weighted_in_abs_strength",
    "weighted_out_abs_strength",
    "incoming_exc_abs_strength",
    "incoming_inh_abs_strength",
    "recurrent_current_abs_mean",
    "rec_input_abs_ratio",
    "firing_rate",
    "input_l2_norm",
    "readout_total_weight_norm",
    "adaptation_mean",
    "adapt_readout_contribution_proxy",
]

EMPHASIZED_OVERLAPS = [
    ("firing_rate", "recurrent_current_abs_mean"),
    ("firing_rate", "rec_input_abs_ratio"),
    ("firing_rate", "input_l2_norm"),
    ("firing_rate", "expected_degree_score"),
    ("adapt_readout_contribution_proxy", "recurrent_current_abs_mean"),
    ("adapt_readout_contribution_proxy", "expected_degree_score"),
    ("adapt_readout_contribution_proxy", "incoming_exc_abs_strength"),
    ("adapt_readout_contribution_proxy", "incoming_inh_abs_strength"),
]

FUTURE_LOGGING_KEYS = [
    "theta_bias",
    "hard_density",
    "edge_prob_entropy",
    "in_degree_gini",
    "out_degree_gini",
    "max_in_degree",
    "max_out_degree",
    "role_src_norm_mean/max",
    "role_dst_norm_mean/max",
    "top_edge_prob_mean",
    "rec_input_abs_ratio",
    "theta_grad_norm_pre_clip",
    "max_firing_rate",
    "mean_firing_rate",
]


@dataclass
class DiagnosticOptions:
    run_dirs: list[Path]
    output_dir: Path
    num_batches: int = 4
    batch_size: int = 64
    top_k: int = 50
    firing_threshold: float = 0.9
    theta_grad_threshold: float = 50.0
    device: str = "auto"


@dataclass
class RunArtifacts:
    run_dir: Path
    config_path: Path | None
    train_log_path: Path | None
    best_checkpoint_path: Path | None
    final_checkpoint_path: Path | None
    topology_snapshot_paths: list[Path]
    fdi_report_path: Path | None


def safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().cpu().reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_values(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _mean(values: list[float]) -> float:
    arr = finite_values(values)
    return float(arr.mean()) if arr.size else float("nan")


def _max(values: list[float]) -> float:
    arr = finite_values(values)
    return float(arr.max()) if arr.size else float("nan")


def _std(values: list[float]) -> float:
    arr = finite_values(values)
    return float(arr.std()) if arr.size else float("nan")


def tensor_stats(prefix: str, tensor: torch.Tensor, *, quantiles: bool = False) -> dict[str, float]:
    x = tensor.detach().float().cpu().reshape(-1)
    valid = x[torch.isfinite(x)]
    if valid.numel() == 0:
        out = {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
        if quantiles:
            out.update(
                {
                    f"{prefix}_p05": float("nan"),
                    f"{prefix}_p50": float("nan"),
                    f"{prefix}_p95": float("nan"),
                }
            )
        return out
    out = {
        f"{prefix}_mean": safe_float(valid.mean()),
        f"{prefix}_std": safe_float(valid.std()) if valid.numel() > 1 else 0.0,
        f"{prefix}_min": safe_float(valid.min()),
        f"{prefix}_max": safe_float(valid.max()),
    }
    if quantiles:
        out.update(
            {
                f"{prefix}_p05": safe_float(torch.quantile(valid, 0.05)),
                f"{prefix}_p50": safe_float(torch.quantile(valid, 0.50)),
                f"{prefix}_p95": safe_float(torch.quantile(valid, 0.95)),
            }
        )
    return out


def gini(values: torch.Tensor | list[float] | np.ndarray) -> float:
    x = torch.as_tensor(values, dtype=torch.float32).detach().cpu().reshape(-1)
    x = x[torch.isfinite(x)].clamp(min=0.0)
    if x.numel() == 0:
        return 0.0
    total = x.sum()
    if total.item() == 0.0:
        return 0.0
    sorted_x = x.sort().values
    n = sorted_x.numel()
    index = torch.arange(1, n + 1, dtype=sorted_x.dtype)
    coeff = (2.0 * torch.dot(index, sorted_x) / (n * total)) - ((n + 1.0) / n)
    return float(coeff.clamp(0.0, 1.0).item())


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
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
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


def _corr_row(
    run_name: str,
    seed: int | None,
    group_label: str,
    x_name: str,
    y_name: str,
    x_values: list[float],
    y_values: list[float],
    *,
    lag_label: str | None = None,
) -> dict[str, Any]:
    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    return {
        "run_name": run_name,
        "seed": seed,
        "group_label": group_label,
        "lag": lag_label or "same_epoch",
        "x": x_name,
        "y": y_name,
        "n": int(valid.sum()),
        "pearson": pearson_corr(x_arr, y_arr),
        "spearman": spearman_corr(x_arr, y_arr),
    }


def lagged_correlations(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    *,
    run_name: str = "",
    seed: int | None = None,
    group_label: str = "",
    x_name: str = "max_firing_rate",
    y_name: str = "theta_grad_norm",
) -> list[dict[str, Any]]:
    a = list(np.asarray(x, dtype=float))
    b = list(np.asarray(y, dtype=float))
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    rows = [
        _corr_row(run_name, seed, group_label, x_name, y_name, a, b, lag_label="x_t_vs_y_t")
    ]
    if n >= 2:
        rows.append(
            _corr_row(
                run_name,
                seed,
                group_label,
                f"{x_name}[t-1]",
                f"{y_name}[t]",
                a[:-1],
                b[1:],
                lag_label="x_t_minus_1_vs_y_t",
            )
        )
        rows.append(
            _corr_row(
                run_name,
                seed,
                group_label,
                f"{y_name}[t-1]",
                f"{x_name}[t]",
                b[:-1],
                a[1:],
                lag_label="y_t_minus_1_vs_x_t",
            )
        )
    else:
        rows.append(
            _corr_row(
                run_name,
                seed,
                group_label,
                f"{x_name}[t-1]",
                f"{y_name}[t]",
                [],
                [],
                lag_label="x_t_minus_1_vs_y_t",
            )
        )
        rows.append(
            _corr_row(
                run_name,
                seed,
                group_label,
                f"{y_name}[t-1]",
                f"{x_name}[t]",
                [],
                [],
                lag_label="y_t_minus_1_vs_x_t",
            )
        )
    return rows


def topk_ids(values: dict[int, float] | list[float] | np.ndarray | set[int], top_k: int) -> set[int]:
    if isinstance(values, set):
        return set(values)
    if isinstance(values, dict):
        pairs = [(int(k), safe_float(v)) for k, v in values.items()]
    else:
        arr = np.asarray(values, dtype=float)
        pairs = [(idx, safe_float(v)) for idx, v in enumerate(arr)]
    valid = [(idx, val) for idx, val in pairs if math.isfinite(val)]
    valid.sort(key=lambda item: item[1], reverse=True)
    return {idx for idx, _ in valid[: max(0, int(top_k))]}


def topk_overlap(
    values_a: dict[int, float] | list[float] | np.ndarray | set[int],
    values_b: dict[int, float] | list[float] | np.ndarray | set[int],
    top_k: int,
) -> dict[str, Any]:
    a_ids = topk_ids(values_a, top_k)
    b_ids = topk_ids(values_b, top_k)
    overlap = a_ids & b_ids
    union = a_ids | b_ids
    denom = min(max(int(top_k), 0), len(a_ids), len(b_ids))
    return {
        "top_k": int(top_k),
        "a_count": len(a_ids),
        "b_count": len(b_ids),
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / denom if denom else float("nan"),
        "jaccard": len(overlap) / len(union) if union else float("nan"),
        "overlap_ids": sorted(overlap),
    }


def _select_device(device_arg: str) -> torch.device:
    value = str(device_arg).lower()
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but CUDA is not available.")
    if value == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS but MPS is not available.")
    return torch.device(value)


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def read_train_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def discover_run_artifacts(run_dir: Path) -> RunArtifacts:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        config_path = None

    log_candidates = [run_dir / "logs" / "train.jsonl", run_dir / "train.jsonl"]
    train_log_path = next((path for path in log_candidates if path.exists()), None)

    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not best_checkpoint_path.exists():
        best_checkpoint_path = None

    final_checkpoint_path = run_dir / "checkpoints" / "final.pt"
    if not final_checkpoint_path.exists():
        final_checkpoint_path = None

    snapshot_paths = sorted(
        path
        for path in (run_dir / "checkpoints").glob("*.pt")
        if any(token in path.name.lower() for token in ("topology", "rollback", "snapshot"))
    )
    fdi_report_path = run_dir / "init_fdi_calibration_report.json"
    if not fdi_report_path.exists():
        fdi_report_path = None

    return RunArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        train_log_path=train_log_path,
        best_checkpoint_path=best_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
        topology_snapshot_paths=snapshot_paths,
        fdi_report_path=fdi_report_path,
    )


def _first_value(row: dict[str, Any], aliases: list[str]) -> tuple[Any, str | None]:
    for key in aliases:
        if key in row and row[key] is not None:
            return row[key], key
    return None, None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def infer_group_label(run_dir: Path, cfg: Any | None) -> str:
    if cfg is None:
        return run_dir.name
    return str(getattr(cfg, "experiment_name", run_dir.name))


def _percentile_threshold(values: list[float], q: float) -> float:
    arr = finite_values(values)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def normalize_epoch_timeseries(
    run_name: str,
    seed: int | None,
    group_label: str,
    rows: list[dict[str, Any]],
    firing_threshold: float,
    theta_grad_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    normalized: list[dict[str, Any]] = []
    missing_log_keys: set[str] = set()

    for idx, raw in enumerate(rows):
        row: dict[str, Any] = {
            "run_name": run_name,
            "seed": seed,
            "group_label": group_label,
            "row_index": idx,
        }
        source_keys: dict[str, str | None] = {}
        for canonical, aliases in EPOCH_ALIASES.items():
            value, source = _first_value(raw, aliases)
            if source is None:
                missing_log_keys.add(canonical)
            row[canonical] = safe_float(value)
            source_keys[f"{canonical}_source_key"] = source or ""
        row.update(source_keys)
        row["epoch"] = int(row["epoch"]) if math.isfinite(row["epoch"]) else idx + 1
        row["topology_frozen"] = _safe_bool(
            raw.get("topology_frozen", raw.get("theta_frozen", False))
        )
        row["topology_frozen_epoch"] = safe_float(
            raw.get("topology_frozen_epoch", raw.get("theta_freeze_epoch"))
        )
        row["topology_rollback_applied"] = _safe_bool(
            raw.get("topology_rollback_applied", False)
        )
        row["topology_best_epoch"] = safe_float(raw.get("topology_best_epoch"))
        row["topology_best_metric_value"] = safe_float(raw.get("topology_best_metric_value"))
        normalized.append(row)

    firing_values = [row["max_firing_rate"] for row in normalized]
    theta_values = [row["theta_grad_norm"] for row in normalized]
    firing_p95 = _percentile_threshold(firing_values, PERCENTILE_EVENT_Q)
    theta_p95 = _percentile_threshold(theta_values, PERCENTILE_EVENT_Q)

    for row in normalized:
        firing = row["max_firing_rate"]
        theta_grad = row["theta_grad_norm"]
        row["firing_fixed_threshold"] = firing_threshold
        row["theta_grad_fixed_threshold"] = theta_grad_threshold
        row["firing_percentile_threshold"] = firing_p95
        row["theta_grad_percentile_threshold"] = theta_p95
        row["firing_fixed_event"] = math.isfinite(firing) and firing > firing_threshold
        row["theta_grad_fixed_event"] = (
            math.isfinite(theta_grad) and theta_grad > theta_grad_threshold
        )
        row["firing_percentile_event"] = math.isfinite(firing) and math.isfinite(firing_p95) and firing >= firing_p95
        row["theta_grad_percentile_event"] = math.isfinite(theta_grad) and math.isfinite(theta_p95) and theta_grad >= theta_p95
        row["firing_event"] = row["firing_fixed_event"] or row["firing_percentile_event"]
        row["theta_grad_event"] = row["theta_grad_fixed_event"] or row["theta_grad_percentile_event"]
        row["same_epoch_event"] = row["firing_event"] and row["theta_grad_event"]
        row["firing_then_grad_next"] = False
        row["grad_then_firing_next"] = False

    for idx in range(1, len(normalized)):
        prev = normalized[idx - 1]
        cur = normalized[idx]
        cur["firing_then_grad_next"] = bool(prev["firing_event"] and cur["theta_grad_event"])
        cur["grad_then_firing_next"] = bool(prev["theta_grad_event"] and cur["firing_event"])

    event_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(normalized):
        event_flags = [
            key
            for key in (
                "firing_fixed_event",
                "firing_percentile_event",
                "theta_grad_fixed_event",
                "theta_grad_percentile_event",
                "same_epoch_event",
                "firing_then_grad_next",
                "grad_then_firing_next",
            )
            if row.get(key)
        ]
        if not event_flags:
            continue
        prev = normalized[idx - 1] if idx > 0 else None
        nxt = normalized[idx + 1] if idx + 1 < len(normalized) else None
        event = dict(row)
        event["event_flags"] = ";".join(event_flags)
        for metric in ("val_acc", "test_acc"):
            event[f"{metric}_delta_from_prev"] = (
                row[metric] - prev[metric]
                if prev is not None
                and math.isfinite(row[metric])
                and math.isfinite(prev[metric])
                else float("nan")
            )
            event[f"{metric}_delta_to_next"] = (
                nxt[metric] - row[metric]
                if nxt is not None
                and math.isfinite(row[metric])
                and math.isfinite(nxt[metric])
                else float("nan")
            )
        event_rows.append(event)

    corr_rows = lagged_correlations(
        [row["max_firing_rate"] for row in normalized],
        [row["theta_grad_norm"] for row in normalized],
        run_name=run_name,
        seed=seed,
        group_label=group_label,
    )

    for key in sorted(missing_log_keys):
        warnings.append(f"{run_name}: train log missing canonical key {key!r}")
    if not rows:
        warnings.append(f"{run_name}: missing / insufficient evidence: train.jsonl unavailable")
    return normalized, event_rows, corr_rows, warnings


def _adjacency_without_self_loops(mask: torch.Tensor) -> torch.Tensor:
    active = mask.detach().cpu().bool().clone()
    if active.ndim != 2 or active.shape[0] != active.shape[1]:
        raise ValueError(f"Expected square adjacency mask, got {tuple(active.shape)}")
    active.fill_diagonal_(False)
    return active


def directed_3cycle_count(mask: torch.Tensor) -> int:
    active = _adjacency_without_self_loops(mask)
    if active.sum().item() == 0:
        return 0
    a = active.float()
    return int(round(float(torch.trace(a @ a @ a).item()) / 3.0))


def average_undirected_clustering(mask: torch.Tensor) -> float:
    active = _adjacency_without_self_loops(mask)
    undirected = active | active.T
    undirected.fill_diagonal_(False)
    vals: list[float] = []
    for node in range(undirected.shape[0]):
        neighbors = torch.nonzero(undirected[node], as_tuple=False).flatten()
        k = int(neighbors.numel())
        if k < 2:
            continue
        subgraph = undirected[neighbors][:, neighbors]
        actual = float(subgraph.sum().item()) / 2.0
        possible = k * (k - 1) / 2.0
        vals.append(actual / possible)
    return float(sum(vals) / len(vals)) if vals else 0.0


def strongly_connected_component_sizes(mask: torch.Tensor) -> list[int]:
    active = _adjacency_without_self_loops(mask)
    n = active.shape[0]
    out_neighbors = [
        torch.nonzero(active[i], as_tuple=False).flatten().tolist() for i in range(n)
    ]
    in_neighbors = [
        torch.nonzero(active[:, i], as_tuple=False).flatten().tolist() for i in range(n)
    ]
    visited = [False] * n
    order: list[int] = []
    for start in range(n):
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

    assigned = [False] * n
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
            for nxt in in_neighbors[node]:
                if not assigned[nxt]:
                    assigned[nxt] = True
                    stack.append(nxt)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def materialize_lowrank_topology(
    src_embed: torch.Tensor,
    dst_embed: torch.Tensor,
    theta_bias: torch.Tensor | float,
    self_conn_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    src = src_embed.detach().float().cpu()
    dst = dst_embed.detach().float().cpu()
    bias = torch.as_tensor(theta_bias, dtype=src.dtype).detach().cpu()
    if bias.numel() == 1:
        logit = src @ dst.T + bias.reshape(())
    else:
        logit = src @ dst.T + bias
    if self_conn_mask is None:
        valid_mask = torch.ones_like(logit, dtype=torch.bool)
    else:
        valid_mask = self_conn_mask.detach().bool().cpu()
    edge_prob = torch.sigmoid(logit)
    hard_mask = (edge_prob >= 0.5) & valid_mask
    return {
        "topology_logit": logit,
        "edge_prob": edge_prob,
        "valid_mask": valid_mask,
        "hard_mask": hard_mask,
    }


def _masked_row_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = values * valid_mask.float()
    denom = valid_mask.float().sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


def _masked_col_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked = values * valid_mask.float()
    denom = valid_mask.float().sum(dim=0).clamp(min=1.0)
    return masked.sum(dim=0) / denom


def _row_top_concentration(values: torch.Tensor, valid_mask: torch.Tensor, frac: float = 0.10) -> torch.Tensor:
    out = []
    for row, mask in zip(values, valid_mask):
        valid = row[mask.bool()]
        if valid.numel() == 0:
            out.append(float("nan"))
            continue
        k = max(1, int(math.ceil(valid.numel() * frac)))
        top_mean = valid.topk(k).values.mean()
        out.append(safe_float(top_mean / valid.mean().clamp(min=EPS)))
    return torch.tensor(out, dtype=torch.float32)


def _col_top_concentration(values: torch.Tensor, valid_mask: torch.Tensor, frac: float = 0.10) -> torch.Tensor:
    return _row_top_concentration(values.T, valid_mask.T, frac=frac)


def lowrank_node_probability_metrics(
    src_embed: torch.Tensor,
    dst_embed: torch.Tensor,
    edge_prob: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    eps = 1e-8
    entropy = -(
        edge_prob * (edge_prob + eps).log()
        + (1.0 - edge_prob) * (1.0 - edge_prob + eps).log()
    )
    expected_out_prob = _masked_row_mean(edge_prob, valid_mask)
    expected_in_prob = _masked_col_mean(edge_prob, valid_mask)
    row_prob_entropy = _masked_row_mean(entropy, valid_mask)
    col_prob_entropy = _masked_col_mean(entropy, valid_mask)
    out_conc = _row_top_concentration(edge_prob, valid_mask)
    in_conc = _col_top_concentration(edge_prob, valid_mask)
    return {
        "expected_in_prob": expected_in_prob,
        "expected_out_prob": expected_out_prob,
        "expected_degree_score": expected_in_prob + expected_out_prob,
        "src_norm": src_embed.detach().float().cpu().norm(dim=1),
        "dst_norm": dst_embed.detach().float().cpu().norm(dim=1),
        "row_prob_entropy": row_prob_entropy,
        "col_prob_entropy": col_prob_entropy,
        "out_top_edge_prob_concentration": out_conc,
        "in_top_edge_prob_concentration": in_conc,
        "top_edge_prob_concentration": torch.maximum(
            torch.nan_to_num(out_conc, nan=0.0),
            torch.nan_to_num(in_conc, nan=0.0),
        ),
    }


def graph_node_metrics(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    active = _adjacency_without_self_loops(mask)
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


def effective_recurrent_weight(model: torch.nn.Module, hard_mask: torch.Tensor | None = None) -> torch.Tensor:
    liquid = model.liquid
    if hard_mask is None:
        hard_mask = liquid.get_binary_mask().detach().cpu().bool()
    mask = hard_mask.detach().float().cpu()
    if hasattr(liquid, "self_conn_mask") and liquid.self_conn_mask is not None:
        mask = mask * liquid.self_conn_mask.detach().float().cpu()
    w_raw = liquid.w_raw.detach().float().cpu()
    w_clamped = torch.clamp(w_raw, max=float(getattr(liquid, "w_raw_max", 0.0)))
    dale = liquid.dale_sign.detach().float().cpu()
    signed_w = dale * F.softplus(w_clamped)
    return mask * signed_w


def sign_aware_strength_metrics(model: torch.nn.Module, hard_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    eff = effective_recurrent_weight(model, hard_mask)
    abs_eff = eff.abs()
    dale = model.liquid.dale_sign.detach().cpu().reshape(-1)
    exc = dale > 0
    inh = dale < 0
    incoming_exc = abs_eff[exc, :].sum(dim=0)
    incoming_inh = abs_eff[inh, :].sum(dim=0)
    outgoing_abs = abs_eff.sum(dim=1)
    outgoing_exc = torch.where(exc, outgoing_abs, torch.zeros_like(outgoing_abs))
    outgoing_inh = torch.where(inh, outgoing_abs, torch.zeros_like(outgoing_abs))
    return {
        "weighted_in_abs_strength": abs_eff.sum(dim=0),
        "weighted_out_abs_strength": abs_eff.sum(dim=1),
        "weighted_in_signed_strength": eff.sum(dim=0),
        "weighted_out_signed_strength": eff.sum(dim=1),
        "incoming_exc_abs_strength": incoming_exc,
        "incoming_inh_abs_strength": incoming_inh,
        "outgoing_exc_abs_strength": outgoing_exc,
        "outgoing_inh_abs_strength": outgoing_inh,
        "incoming_exc_inh_abs_ratio": incoming_exc / incoming_inh.clamp(min=EPS),
        "incoming_ei_abs_balance": (incoming_exc - incoming_inh)
        / (incoming_exc + incoming_inh).clamp(min=EPS),
    }


def compute_lowrank_role_summary(
    model: torch.nn.Module,
    run_name: str,
    seed: int | None,
    group_label: str,
    warnings: list[str],
    tau: float,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    liquid = model.liquid
    base = {
        "run_name": run_name,
        "seed": seed,
        "group_label": group_label,
        "recurrent_mode": str(getattr(liquid, "mode", "")),
        "lowrank_available": False,
        "evidence_status": "missing",
    }
    if str(getattr(liquid, "mode", "")).lower() not in {
        "learned_lowrank",
        "learned_lowrank_grad_r",
    }:
        return {**base, "evidence_status": "not_lowrank"}, {}

    mats = materialize_lowrank_topology(
        liquid.src_embed,
        liquid.dst_embed,
        liquid.theta_bias,
        getattr(liquid, "self_conn_mask", None),
    )
    logit = mats["topology_logit"]
    edge_prob = mats["edge_prob"]
    valid_mask = mats["valid_mask"]
    hard_mask = mats["hard_mask"]
    valid_logits = logit[valid_mask]
    valid_probs = edge_prob[valid_mask]
    entropy = -(
        edge_prob * (edge_prob + 1e-8).log()
        + (1.0 - edge_prob) * (1.0 - edge_prob + 1e-8).log()
    )

    model_eval_mask = liquid.get_binary_mask().detach().cpu().bool()
    if hasattr(liquid, "self_conn_mask") and liquid.self_conn_mask is not None:
        model_eval_mask = model_eval_mask & liquid.self_conn_mask.detach().cpu().bool()
    sampled_mask = liquid.sample_mask(tau=tau).detach().cpu().bool()
    if hasattr(liquid, "self_conn_mask") and liquid.self_conn_mask is not None:
        sampled_mask = sampled_mask & liquid.self_conn_mask.detach().cpu().bool()
    reconstructed_density = safe_float(hard_mask.float().mean())
    model_eval_density = safe_float(model_eval_mask.float().mean())
    current_mask_density = safe_float(sampled_mask.float().mean())
    density_match = (
        abs(reconstructed_density - model_eval_density) <= DENSITY_TOL
        and abs(reconstructed_density - current_mask_density) <= DENSITY_TOL
    )
    if not density_match:
        warnings.append(
            f"{run_name}: reconstructed hard density mismatch "
            f"(reconstructed={reconstructed_density:.8f}, "
            f"model_eval={model_eval_density:.8f}, current_mask={current_mask_density:.8f})"
        )

    active = _adjacency_without_self_loops(hard_mask)
    graph = graph_node_metrics(active)
    scc_sizes = strongly_connected_component_sizes(active)
    top_probs = valid_probs.sort(descending=True).values

    def top_frac_mean(frac: float) -> float:
        if top_probs.numel() == 0:
            return float("nan")
        k = max(1, int(math.ceil(top_probs.numel() * frac)))
        return safe_float(top_probs[:k].mean())

    summary = {
        **base,
        "lowrank_available": True,
        "evidence_status": "available",
        "n_nodes": int(hard_mask.shape[0]),
        "n_valid_edges": int(valid_mask.sum().item()),
        "n_active_edges": int(hard_mask.sum().item()),
        "hard_density_valid_edges": safe_float(hard_mask.float().sum() / valid_mask.float().sum().clamp(min=1.0)),
        "reconstructed_hard_density": reconstructed_density,
        "model_eval_hard_density": model_eval_density,
        "current_mask_density": current_mask_density,
        "hard_density_match": density_match,
        "edge_entropy_mean": safe_float(entropy[valid_mask].mean()) if valid_mask.any() else float("nan"),
        "edge_prob_top_1pct_mean": top_frac_mean(0.01),
        "edge_prob_top_5pct_mean": top_frac_mean(0.05),
        "edge_prob_top_10pct_mean": top_frac_mean(0.10),
        "in_degree_gini": gini(graph["in_degree"]),
        "out_degree_gini": gini(graph["out_degree"]),
        "reciprocal_edge_count": int((active & active.T).sum().item()),
        "reciprocal_pair_count": int((active & active.T).triu(diagonal=1).sum().item()),
        "reciprocal_edge_rate": safe_float((active & active.T).float().sum() / active.float().sum().clamp(min=1.0)),
        "directed_3cycle_count": directed_3cycle_count(active),
        "largest_scc": scc_sizes[0] if scc_sizes else 0,
        "num_scc": len(scc_sizes),
        "clustering_undirected": average_undirected_clustering(active),
        **tensor_stats("logit", valid_logits, quantiles=True),
        **tensor_stats("edge_prob", valid_probs, quantiles=True),
        **tensor_stats("src_norm", liquid.src_embed.detach().cpu().norm(dim=1), quantiles=True),
        **tensor_stats("dst_norm", liquid.dst_embed.detach().cpu().norm(dim=1), quantiles=True),
        "theta_bias": safe_float(liquid.theta_bias.detach().cpu()),
    }
    node_prob = lowrank_node_probability_metrics(
        liquid.src_embed.detach().cpu(),
        liquid.dst_embed.detach().cpu(),
        edge_prob,
        valid_mask,
    )
    return summary, {**mats, **graph, **node_prob}


def input_projection_metrics(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    weight = model.input_proj.effective_weight().detach().float().cpu()
    mask = getattr(model.input_proj, "mask", weight != 0).detach().float().cpu()
    fan_in = mask.sum(dim=0)
    return {
        "input_fan_in": fan_in,
        "input_l1_norm": weight.abs().sum(dim=0),
        "input_l2_norm": weight.norm(dim=0),
        "input_density": fan_in / max(mask.shape[0], 1),
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
        if self.has_adaptation:
            adaptation_mean = self.adaptation_sum / denom
            adaptation_abs_mean = self.adaptation_abs_sum / denom
            adaptation_max = self.adaptation_max
        else:
            adaptation_mean = torch.full((self.n_liquid,), float("nan"))
            adaptation_abs_mean = torch.full((self.n_liquid,), float("nan"))
            adaptation_max = torch.full((self.n_liquid,), float("nan"))
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


def _empty_activity_metrics(n_liquid: int, has_adaptation: bool) -> dict[str, torch.Tensor]:
    nan_vec = torch.full((n_liquid,), float("nan"))
    out = {
        "firing_rate": nan_vec.clone(),
        "input_current_mean": nan_vec.clone(),
        "input_current_std": nan_vec.clone(),
        "input_current_abs_mean": nan_vec.clone(),
        "recurrent_current_mean": nan_vec.clone(),
        "recurrent_current_std": nan_vec.clone(),
        "recurrent_current_abs_mean": nan_vec.clone(),
        "adaptation_mean": nan_vec.clone(),
        "adaptation_abs_mean": nan_vec.clone(),
        "adaptation_max": nan_vec.clone(),
    }
    if not has_adaptation:
        return out
    return out


def collect_diagnostic_batches(cfg: Any, batch_size: int, num_batches: int):
    from src.data.loaders import get_dataloaders, get_train_val_test_dataloaders

    old_batch_size = getattr(cfg, "batch_size", None)
    cfg.batch_size = int(batch_size)
    try:
        if str(getattr(cfg, "dataset", "")).lower() == "shd":
            _, _, test_loader = get_train_val_test_dataloaders(cfg)
        else:
            _, test_loader = get_dataloaders(cfg)
        batches = []
        for idx, batch in enumerate(test_loader):
            if idx >= num_batches:
                break
            batches.append(batch)
        return batches
    finally:
        if old_batch_size is not None:
            cfg.batch_size = old_batch_size


def build_neuron_rows(
    run_name: str,
    run_dir: Path,
    cfg: Any,
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    role_tensors: dict[str, torch.Tensor],
    batches: list[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
    warnings: list[str],
) -> list[dict[str, Any]]:
    has_adaptation = str(getattr(cfg.liquid, "neuron_type", "")) == "alif"
    n_liquid = int(model.n_liquid)
    if batches:
        accumulator = PerNeuronAccumulator(n_liquid, has_adaptation=has_adaptation)
        model.eval()
        if hasattr(model.liquid, "unlock_epoch_mask"):
            model.liquid.unlock_epoch_mask()
        with torch.no_grad():
            for x, _ in batches:
                _, traces = model(x.to(device), tau=getattr(cfg, "tau_end", 0.05), return_traces=True)
                accumulator.update(traces)
        activity = accumulator.finalize()
    else:
        warnings.append(f"{run_name}: missing / insufficient evidence: diagnostic batches unavailable")
        activity = _empty_activity_metrics(n_liquid, has_adaptation)

    hard_mask = role_tensors.get("hard_mask")
    if hard_mask is None:
        hard_mask = model.liquid.get_binary_mask().detach().cpu().bool()
    graph = graph_node_metrics(hard_mask)
    strengths = sign_aware_strength_metrics(model, hard_mask)
    input_metrics = input_projection_metrics(model)
    readout = readout_importance(model)
    rec_input_std_ratio = activity["recurrent_current_std"] / activity[
        "input_current_std"
    ].clamp(min=EPS)
    rec_input_abs_ratio = activity["recurrent_current_abs_mean"] / activity[
        "input_current_abs_mean"
    ].clamp(min=EPS)
    adapt_readout_contribution_proxy = activity["adaptation_abs_mean"] * readout[
        "readout_adapt_weight_norm"
    ]

    dale = model.liquid.dale_sign.detach().cpu().reshape(-1)
    seed = int(getattr(cfg, "seed", -1))
    group_label = infer_group_label(run_dir, cfg)
    rows: list[dict[str, Any]] = []
    per_neuron_sources = [
        input_metrics,
        graph,
        strengths,
        activity,
        readout,
        {
            "rec_input_std_ratio": rec_input_std_ratio,
            "rec_input_abs_ratio": rec_input_abs_ratio,
            "adapt_readout_contribution_proxy": adapt_readout_contribution_proxy,
        },
    ]
    for key in (
        "expected_in_prob",
        "expected_out_prob",
        "expected_degree_score",
        "src_norm",
        "dst_norm",
        "row_prob_entropy",
        "col_prob_entropy",
        "out_top_edge_prob_concentration",
        "in_top_edge_prob_concentration",
        "top_edge_prob_concentration",
    ):
        if key in role_tensors:
            per_neuron_sources.append({key: role_tensors[key]})
    for neuron_id in range(n_liquid):
        row: dict[str, Any] = {
            "run_name": run_name,
            "experiment_dir": str(run_dir),
            "seed": seed,
            "group_label": group_label,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_acc": checkpoint.get("best_val_acc"),
            "best_test_acc_at_best_val": checkpoint.get("best_test_acc_at_best_val"),
            "neuron_id": neuron_id,
            "ei_type": "E" if dale[neuron_id].item() > 0 else "I",
        }
        for source in per_neuron_sources:
            for key, values in source.items():
                if isinstance(values, torch.Tensor) and values.numel() == n_liquid:
                    row[key] = safe_float(values[neuron_id])
        rows.append(row)
    return rows


def _row_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [safe_float(row.get(key)) for row in rows]


def compute_correlation_rows(rows_by_run: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run_name, rows in rows_by_run.items():
        if not rows:
            continue
        meta = rows[0]
        for x_key, y_key in CORRELATION_PAIRS:
            out.append(
                _corr_row(
                    run_name,
                    int(meta.get("seed", -1)),
                    str(meta.get("group_label", "")),
                    x_key,
                    y_key,
                    _row_values(rows, x_key),
                    _row_values(rows, y_key),
                    lag_label="neuron_level",
                )
            )
    return out


def compute_topk_overlap_rows(
    rows_by_run: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run_name, rows in rows_by_run.items():
        if not rows:
            continue
        meta = rows[0]
        metric_values = {metric: _row_values(rows, metric) for metric in TOPK_METRICS}
        for a_key, b_key in EMPHASIZED_OVERLAPS:
            overlap = topk_overlap(metric_values.get(a_key, []), metric_values.get(b_key, []), top_k)
            out.append(
                {
                    "run_name": run_name,
                    "seed": meta.get("seed"),
                    "group_label": meta.get("group_label"),
                    "comparison": f"top_{a_key}__top_{b_key}",
                    "metric_a": a_key,
                    "metric_b": b_key,
                    "emphasized": True,
                    **{k: v for k, v in overlap.items() if k != "overlap_ids"},
                    "overlap_ids": ";".join(str(x) for x in overlap["overlap_ids"]),
                }
            )
        for metric in TOPK_METRICS:
            if metric == "firing_rate":
                continue
            overlap = topk_overlap(metric_values.get("firing_rate", []), metric_values.get(metric, []), top_k)
            out.append(
                {
                    "run_name": run_name,
                    "seed": meta.get("seed"),
                    "group_label": meta.get("group_label"),
                    "comparison": f"top_firing_rate__top_{metric}",
                    "metric_a": "firing_rate",
                    "metric_b": metric,
                    "emphasized": False,
                    **{k: v for k, v in overlap.items() if k != "overlap_ids"},
                    "overlap_ids": ";".join(str(x) for x in overlap["overlap_ids"]),
                }
            )
    return out


def _oracle_from_log(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if math.isfinite(safe_float(row.get("test_acc")))]
    if not valid:
        return {"oracle_best_test_epoch": None, "oracle_best_test_acc": None}
    best = max(valid, key=lambda row: safe_float(row.get("test_acc")))
    return {
        "oracle_best_test_epoch": int(safe_float(best.get("epoch"))),
        "oracle_best_test_acc": safe_float(best.get("test_acc")),
    }


def load_model_and_checkpoint(
    cfg: Any,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], list[str]]:
    from src.lsm.trainer import build_model

    warnings: list[str] = []
    torch.manual_seed(int(getattr(cfg, "seed", 0)))
    model = build_model(cfg, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        warnings.append(
            f"{checkpoint_path}: load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.eval()
    if hasattr(model.liquid, "unlock_epoch_mask"):
        model.liquid.unlock_epoch_mask()
    return model, checkpoint, warnings


def _load_config(path: Path):
    from src.utils.config import load_config

    return load_config(str(path))


def process_run(
    run_dir: Path,
    options: DiagnosticOptions,
    device: torch.device,
) -> dict[str, Any]:
    artifacts = discover_run_artifacts(run_dir)
    run_name = artifacts.run_dir.name
    warnings: list[str] = []
    cfg = None
    checkpoint: dict[str, Any] = {}
    model = None
    role_summary: dict[str, Any] = {
        "run_name": run_name,
        "seed": None,
        "group_label": run_name,
        "evidence_status": "missing",
    }
    role_tensors: dict[str, torch.Tensor] = {}

    if artifacts.config_path is None:
        warnings.append(f"{run_name}: missing config.yaml")
    else:
        try:
            cfg = _load_config(artifacts.config_path)
        except Exception as exc:  # pragma: no cover - runtime resilience
            warnings.append(f"{run_name}: failed to load config.yaml: {type(exc).__name__}: {exc}")

    seed = int(getattr(cfg, "seed", -1)) if cfg is not None else None
    group_label = infer_group_label(artifacts.run_dir, cfg)

    raw_log_rows = read_train_jsonl(artifacts.train_log_path)
    epoch_rows, event_rows, lagged_rows, epoch_warnings = normalize_epoch_timeseries(
        run_name,
        seed,
        group_label,
        raw_log_rows,
        options.firing_threshold,
        options.theta_grad_threshold,
    )
    warnings.extend(epoch_warnings)

    if artifacts.best_checkpoint_path is None:
        warnings.append(f"{run_name}: missing checkpoints/best.pt")
    elif cfg is not None:
        try:
            model, checkpoint, load_warnings = load_model_and_checkpoint(
                cfg, artifacts.best_checkpoint_path, device
            )
            warnings.extend(f"{run_name}: {msg}" for msg in load_warnings)
            role_summary, role_tensors = compute_lowrank_role_summary(
                model,
                run_name,
                seed,
                group_label,
                warnings,
                tau=float(getattr(cfg, "tau_end", 0.05)),
            )
        except Exception as exc:  # pragma: no cover - runtime resilience
            warnings.append(
                f"{run_name}: checkpoint-level analysis failed: {type(exc).__name__}: {exc}"
            )

    batches = None
    if cfg is not None and model is not None and options.num_batches <= 0:
        warnings.append(f"{run_name}: diagnostic forward skipped because --num-batches <= 0")
    elif cfg is not None and model is not None:
        try:
            batches = collect_diagnostic_batches(
                cfg, options.batch_size, options.num_batches
            )
            if not batches:
                warnings.append(f"{run_name}: diagnostic dataloader returned no batches")
        except Exception as exc:
            warnings.append(
                f"{run_name}: missing / insufficient evidence: diagnostic batch collection failed: "
                f"{type(exc).__name__}: {exc}"
            )

    neuron_rows: list[dict[str, Any]] = []
    if cfg is not None and model is not None:
        try:
            neuron_rows = build_neuron_rows(
                run_name,
                artifacts.run_dir,
                cfg,
                model,
                checkpoint,
                role_tensors,
                batches,
                device,
                warnings,
            )
        except Exception as exc:  # pragma: no cover - runtime resilience
            warnings.append(
                f"{run_name}: neuron-level analysis failed: {type(exc).__name__}: {exc}"
            )

    oracle = _oracle_from_log(raw_log_rows)
    fdi_report = _read_json(artifacts.fdi_report_path)
    max_firing = _max([row.get("max_firing_rate", float("nan")) for row in epoch_rows])
    max_theta_grad = _max([row.get("theta_grad_norm", float("nan")) for row in epoch_rows])
    functional_alif = detect_functional_high_recurrent_alif_regime(
        cfg, epoch_rows, neuron_rows, options
    )
    run_summary = {
        "run_name": run_name,
        "experiment_dir": str(artifacts.run_dir),
        "seed": seed,
        "group_label": group_label,
        "config_available": artifacts.config_path is not None,
        "train_log_available": artifacts.train_log_path is not None,
        "best_checkpoint_available": artifacts.best_checkpoint_path is not None,
        "final_checkpoint_available": artifacts.final_checkpoint_path is not None,
        "topology_snapshot_count": len(artifacts.topology_snapshot_paths),
        "fdi_report_available": artifacts.fdi_report_path is not None,
        "num_epoch_rows": len(epoch_rows),
        "num_event_rows": len(event_rows),
        "num_neuron_rows": len(neuron_rows),
        "best_checkpoint_epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_acc": checkpoint.get("best_val_acc"),
        "best_test_acc_at_best_val": checkpoint.get("best_test_acc_at_best_val"),
        **oracle,
        "max_firing_rate_observed": max_firing,
        "max_theta_grad_observed": max_theta_grad,
        "functional_high_recurrent_alif_feature_regime": functional_alif,
        "fdi_selected_candidate": json.dumps(fdi_report.get("selected_candidate", {}))
        if fdi_report
        else "",
        "warnings": " | ".join(warnings),
    }
    return {
        "run_summary": run_summary,
        "epoch_rows": epoch_rows,
        "event_rows": event_rows,
        "lagged_rows": lagged_rows,
        "role_summary": role_summary,
        "neuron_rows": neuron_rows,
        "warnings": warnings,
    }


def detect_functional_high_recurrent_alif_regime(
    cfg: Any | None,
    epoch_rows: list[dict[str, Any]],
    neuron_rows: list[dict[str, Any]],
    options: DiagnosticOptions,
) -> bool:
    if cfg is None or str(getattr(cfg.liquid, "neuron_type", "")).lower() != "alif":
        return False
    max_firing = _max([safe_float(row.get("max_firing_rate")) for row in epoch_rows])
    max_theta_grad = _max([safe_float(row.get("theta_grad_norm")) for row in epoch_rows])
    if not math.isfinite(max_firing) or max_firing <= options.firing_threshold:
        return False
    if math.isfinite(max_theta_grad) and max_theta_grad > options.theta_grad_threshold:
        return False
    if not neuron_rows:
        return False
    adapt_proxy = finite_values(_row_values(neuron_rows, "adapt_readout_contribution_proxy"))
    recurrent = finite_values(_row_values(neuron_rows, "recurrent_current_abs_mean"))
    if adapt_proxy.size == 0 or recurrent.size == 0:
        return False
    return bool(np.nanquantile(adapt_proxy, 0.90) > 0.0 and np.nanquantile(recurrent, 0.90) > 0.0)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        columns = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([safe_float(row.get(key)) for row in rows], dtype=float)


def _placeholder_figure(path: Path, title: str, message: str = "insufficient evidence") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter_figure(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = _values(rows, x_key)
    y = _values(rows, y_key)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        _placeholder_figure(path, title)
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x[valid], y[valid], s=14, alpha=0.7)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _hist_figure(path: Path, values: list[float], title: str, xlabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = finite_values(values)
    if arr.size == 0:
        _placeholder_figure(path, title)
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(arr, bins=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(
    output_dir: Path,
    epoch_rows: list[dict[str, Any]],
    neuron_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
) -> None:
    cache_dir = output_dir / ".plot_cache"
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _scatter_figure(
        figures_dir / "max_firing_vs_theta_grad_scatter.png",
        epoch_rows,
        "max_firing_rate",
        "theta_grad_norm",
        "Max firing vs theta/topology grad",
    )
    write_epoch_line_figure(figures_dir / "epoch_max_firing_theta_grad_line.png", epoch_rows)
    _scatter_figure(
        figures_dir / "role_alignment_vs_degree.png",
        neuron_rows,
        "expected_degree_score",
        "total_degree",
        "Expected degree proxy vs hard degree",
    )
    _scatter_figure(
        figures_dir / "recurrent_current_vs_firing.png",
        neuron_rows,
        "recurrent_current_abs_mean",
        "firing_rate",
        "Recurrent current vs firing",
    )
    _scatter_figure(
        figures_dir / "rec_input_ratio_vs_firing.png",
        neuron_rows,
        "rec_input_abs_ratio",
        "firing_rate",
        "Recurrent/input ratio vs firing",
    )
    _scatter_figure(
        figures_dir / "input_norm_vs_firing.png",
        neuron_rows,
        "input_l2_norm",
        "firing_rate",
        "Input norm vs firing",
    )
    if finite_values(_row_values(neuron_rows, "adapt_readout_contribution_proxy")).size:
        _scatter_figure(
            figures_dir / "adaptation_contribution_vs_recurrent_current.png",
            neuron_rows,
            "recurrent_current_abs_mean",
            "adapt_readout_contribution_proxy",
            "Adaptation contribution proxy vs recurrent current",
        )
    edge_values: list[float] = []
    for row in role_rows:
        for key in ("edge_prob_mean", "edge_prob_p05", "edge_prob_p50", "edge_prob_p95"):
            value = safe_float(row.get(key))
            if math.isfinite(value):
                edge_values.append(value)
    _hist_figure(figures_dir / "edge_prob_hist.png", edge_values, "Edge probability summary histogram", "edge probability summary")
    degree_values = _row_values(neuron_rows, "total_degree")
    _hist_figure(figures_dir / "degree_hist.png", degree_values, "Total degree histogram", "total_degree")


def write_epoch_line_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_rows = [
        row
        for row in rows
        if math.isfinite(safe_float(row.get("epoch")))
        and (
            math.isfinite(safe_float(row.get("max_firing_rate")))
            or math.isfinite(safe_float(row.get("theta_grad_norm")))
        )
    ]
    if not valid_rows:
        _placeholder_figure(path, "Epoch max firing and theta/topology grad")
        return
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    for run_name in sorted({str(row.get("run_name")) for row in valid_rows}):
        run_rows = sorted(
            [row for row in valid_rows if str(row.get("run_name")) == run_name],
            key=lambda row: safe_float(row.get("epoch")),
        )
        epochs = [safe_float(row.get("epoch")) for row in run_rows]
        ax1.plot(
            epochs,
            [safe_float(row.get("max_firing_rate")) for row in run_rows],
            alpha=0.75,
            label=f"{run_name} firing",
        )
        ax2.plot(
            epochs,
            [safe_float(row.get("theta_grad_norm")) for row in run_rows],
            alpha=0.45,
            linestyle="--",
            label=f"{run_name} theta_grad",
        )
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("max_firing_rate")
    ax2.set_ylabel("theta/topology grad norm")
    ax1.set_title("Epoch max firing and theta/topology grad")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format_float(value: Any, digits: int = 4) -> str:
    val = safe_float(value)
    if not math.isfinite(val):
        return "missing"
    return f"{val:.{digits}f}"


def _available_key_count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(math.isfinite(safe_float(row.get(key))) for row in rows)


def classify_verdict(
    run_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    lagged_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    support: list[str] = []
    weaken: list[str] = []
    if not run_rows or _available_key_count(run_rows, "theta_grad_norm") < 3:
        return (
            "insufficient_temporal_evidence",
            support,
            ["theta/topology gradient time series is missing or too short"],
        )

    same = next((row for row in lagged_rows if row.get("lag") == "x_t_vs_y_t"), None)
    corr = safe_float(same.get("pearson") if same else None)
    if math.isfinite(corr) and corr > 0.25:
        support.append("max_firing_rate and theta/topology grad show positive correlation")
    elif math.isfinite(corr) and abs(corr) < 0.10:
        weaken.append("max_firing_rate and theta/topology grad are weakly related")

    if any(row.get("same_epoch_event") or row.get("firing_then_grad_next") or row.get("grad_then_firing_next") for row in event_rows):
        support.append("firing and topology-gradient events co-occur or are adjacent in logs")
    else:
        weaken.append("no same/adjacent firing-gradient event coupling was detected")

    firing_rec = [
        row
        for row in overlap_rows
        if row.get("comparison") == "top_firing_rate__top_recurrent_current_abs_mean"
    ]
    firing_input = [
        row for row in overlap_rows if row.get("comparison") == "top_firing_rate__top_input_l2_norm"
    ]
    if firing_rec and firing_input:
        rec_overlap = _mean([safe_float(row.get("overlap_fraction")) for row in firing_rec])
        input_overlap = _mean([safe_float(row.get("overlap_fraction")) for row in firing_input])
        if math.isfinite(rec_overlap) and math.isfinite(input_overlap):
            if rec_overlap > input_overlap:
                support.append("top firing overlaps recurrent-current metrics more than input norm")
            elif input_overlap > rec_overlap:
                weaken.append("top firing overlaps input norm more than recurrent-current metrics")

    if support and weaken:
        verdict = "mixed_correlational_evidence"
    elif support:
        verdict = "supportive_correlational_evidence"
    elif weaken:
        verdict = "not_supported_by_available_logs"
    else:
        verdict = "insufficient_temporal_evidence"
    return verdict, support, weaken


def generate_report(
    output_dir: Path,
    run_summaries: list[dict[str, Any]],
    epoch_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    lagged_rows: list[dict[str, Any]],
    role_rows: list[dict[str, Any]],
    neuron_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    verdict, support, weaken = classify_verdict(
        epoch_rows, event_rows, lagged_rows, overlap_rows
    )
    functional_runs = [
        row["run_name"]
        for row in run_summaries
        if row.get("functional_high_recurrent_alif_feature_regime")
    ]
    missing_hooks = sorted(
        {
            key
            for key in FUTURE_LOGGING_KEYS
            if key not in {"hard_density", "max_firing_rate", "mean_firing_rate"}
            and not any(key in row for row in epoch_rows)
        }
    )
    density_warnings = [
        row
        for row in role_rows
        if row.get("lowrank_available") and not bool(row.get("hard_density_match", True))
    ]

    lines: list[str] = []
    lines.append("# Lowrank Recurrent Runaway Diagnostics")
    lines.append("")
    lines.append("## 1. 분석 목적")
    lines.append(
        "Existing run logs and validation-selected `best.pt` checkpoints are inspected for correlational evidence related to learned-lowrank recurrent hub/runaway hypotheses. This report does not claim causal proof."
    )
    lines.append("")
    lines.append("## 2. 입력 run 목록")
    if run_summaries:
        for row in run_summaries:
            lines.append(
                f"- `{row.get('run_name')}` seed={row.get('seed')} best_val={_format_float(row.get('best_val_acc'))} test_at_best_val={_format_float(row.get('best_test_acc_at_best_val'))}"
            )
    else:
        lines.append("- missing / insufficient evidence: no runs processed")
    lines.append("")
    lines.append("## 3. 사용 가능한 로그/체크포인트 요약")
    for row in run_summaries:
        lines.append(
            f"- `{row.get('run_name')}` log={row.get('train_log_available')} best.pt={row.get('best_checkpoint_available')} final.pt={row.get('final_checkpoint_available')} topology_snapshots={row.get('topology_snapshot_count')} neuron_rows={row.get('num_neuron_rows')}"
        )
    if warnings:
        lines.append("")
        lines.append("Warnings / missing evidence:")
        for warning in warnings[:40]:
            lines.append(f"- {warning}")
        if len(warnings) > 40:
            lines.append(f"- ... {len(warnings) - 40} more warnings")
    if density_warnings:
        lines.append("")
        lines.append("Hard-mask reconstruction warnings:")
        for row in density_warnings:
            lines.append(
                f"- `{row.get('run_name')}` reconstructed={_format_float(row.get('reconstructed_hard_density'), 8)} model_eval={_format_float(row.get('model_eval_hard_density'), 8)} current_mask={_format_float(row.get('current_mask_density'), 8)}"
            )
    lines.append("")
    lines.append("## 4. epoch-level evidence")
    lines.append(
        f"- epoch rows={len(epoch_rows)}, event rows={len(event_rows)}, lagged correlation rows={len(lagged_rows)}"
    )
    same_corr = [row for row in lagged_rows if row.get("lag") == "x_t_vs_y_t"]
    if same_corr:
        lines.append(
            f"- mean same-epoch Pearson(max_firing, theta_grad)={_format_float(_mean([safe_float(row.get('pearson')) for row in same_corr]))}"
        )
    if event_rows:
        lines.append("- event detection uses fixed thresholds and within-run top-5% percentile thresholds.")
    else:
        lines.append("- missing / insufficient evidence: no fixed-threshold or relative event spikes detected.")
    lines.append("")
    lines.append("## 5. lowrank role alignment evidence")
    lines.append(
        "Terminology note: row/column mean edge probabilities are reported as expected-degree proxies, not independent proof of role alignment."
    )
    for row in role_rows:
        lines.append(
            f"- `{row.get('run_name')}` status={row.get('evidence_status')} hard_density={_format_float(row.get('reconstructed_hard_density'))} in_gini={_format_float(row.get('in_degree_gini'))} out_gini={_format_float(row.get('out_degree_gini'))} largest_scc={row.get('largest_scc', 'missing')}"
        )
    lines.append("")
    lines.append("## 6. neuron-level recurrent/input/readout evidence")
    if neuron_rows:
        lines.append(
            f"- neuron rows={len(neuron_rows)}; sign-aware weighted recurrent strengths are included in `neuron_table.csv`."
        )
        lines.append(
            f"- mean recurrent_current_abs_mean={_format_float(_mean(_row_values(neuron_rows, 'recurrent_current_abs_mean')))}; mean input_current_abs_mean={_format_float(_mean(_row_values(neuron_rows, 'input_current_abs_mean')))}"
        )
    else:
        lines.append("- missing / insufficient evidence: neuron-level forward diagnostics unavailable.")
    if functional_runs:
        lines.append(
            "- ALIF functional high-recurrent feature regime candidates: "
            + ", ".join(f"`{name}`" for name in functional_runs)
        )
    lines.append("")
    lines.append("## 7. top-k overlap evidence")
    emphasized = [row for row in overlap_rows if row.get("emphasized")]
    for row in emphasized[:20]:
        lines.append(
            f"- `{row.get('run_name')}` {row.get('comparison')}: overlap={row.get('overlap_count')}/{row.get('top_k')} fraction={_format_float(row.get('overlap_fraction'))} jaccard={_format_float(row.get('jaccard'))}"
        )
    if not emphasized:
        lines.append("- missing / insufficient evidence: top-k overlaps unavailable.")
    lines.append("")
    lines.append("## 8. hypothesis verdict")
    lines.append(f"- Verdict: `{verdict}`")
    if support:
        lines.append("- Supporting correlational patterns:")
        for item in support:
            lines.append(f"- {item}")
    if weaken:
        lines.append("- Weakening or alternative patterns:")
        for item in weaken:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("## 9. limitations")
    lines.append("- A single `best.pt` cannot prove temporal growth in lowrank role structure.")
    lines.append("- Checkpoint-level expected-probability evidence is a mechanism candidate only.")
    lines.append("- Temporal causality is limited to `train.jsonl` event evidence unless epoch topology snapshots exist.")
    lines.append("- If `theta_grad_norm_pre_clip` is missing, available clipped or alias gradient keys are used.")
    lines.append("- High ALIF firing is not automatically failure; stable gradients plus high adaptation/readout contribution are labeled separately.")
    lines.append("- Causal proof requires hub removal, role-cluster edge removal, and density-preserving shuffle diagnostics.")
    lines.append("")
    lines.append("## 10. next recommended diagnostics")
    lines.append("- Add optional logging hooks only in a separate PR if the current logs are insufficient.")
    if missing_hooks:
        lines.append("- Suggested future logging keys: " + ", ".join(f"`{key}`" for key in missing_hooks))
    else:
        lines.append("- Current logs contain the primary requested firing/density keys; optional hooks may still improve causal resolution.")
    lines.append("- Run targeted interventions: hub removal, role-cluster edge removal, density-preserving topology shuffle.")
    report = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(report)
    return report


def run_diagnostics(options: DiagnosticOptions) -> dict[str, Any]:
    device = _select_device(options.device)
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    lagged_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    neuron_rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    rows_by_run: dict[str, list[dict[str, Any]]] = {}

    for run_dir in options.run_dirs:
        result = process_run(Path(run_dir), options, device)
        run_summaries.append(result["run_summary"])
        epoch_rows.extend(result["epoch_rows"])
        event_rows.extend(result["event_rows"])
        lagged_rows.extend(result["lagged_rows"])
        role_rows.append(result["role_summary"])
        neuron_rows.extend(result["neuron_rows"])
        all_warnings.extend(result["warnings"])
        if result["neuron_rows"]:
            rows_by_run[result["run_summary"]["run_name"]] = result["neuron_rows"]

    correlation_rows = compute_correlation_rows(rows_by_run)
    overlap_rows = compute_topk_overlap_rows(rows_by_run, options.top_k)

    write_csv(output_dir / "run_summary.csv", run_summaries)
    write_csv(output_dir / "epoch_timeseries.csv", epoch_rows)
    write_csv(output_dir / "epoch_event_table.csv", event_rows)
    write_csv(output_dir / "epoch_lagged_correlations.csv", lagged_rows)
    write_csv(output_dir / "lowrank_role_summary.csv", role_rows)
    write_csv(output_dir / "neuron_table.csv", neuron_rows)
    write_csv(output_dir / "correlation_summary.csv", correlation_rows)
    write_csv(output_dir / "topk_overlap_summary.csv", overlap_rows)
    generate_report(
        output_dir,
        run_summaries,
        epoch_rows,
        event_rows,
        lagged_rows,
        role_rows,
        neuron_rows,
        correlation_rows,
        overlap_rows,
        all_warnings,
    )
    write_figures(output_dir, epoch_rows, neuron_rows, role_rows)
    return {
        "output_dir": output_dir,
        "device": str(device),
        "run_summaries": run_summaries,
        "warnings": all_warnings,
        "artifact_paths": [
            output_dir / "run_summary.csv",
            output_dir / "epoch_timeseries.csv",
            output_dir / "epoch_event_table.csv",
            output_dir / "epoch_lagged_correlations.csv",
            output_dir / "lowrank_role_summary.csv",
            output_dir / "neuron_table.csv",
            output_dir / "correlation_summary.csv",
            output_dir / "topk_overlap_summary.csv",
            output_dir / "report.md",
            output_dir / "figures",
        ],
    }
