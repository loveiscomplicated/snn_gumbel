"""Lightweight runtime diagnostics for SHD LSM experiments.

The logger records scalar summaries only. It intentionally avoids extra forward
passes, sampled topology state, and any mutation of training behavior.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


EPS = 1e-12

PERFORMANCE_METRICS = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "best_val_acc_so_far",
    "test_at_best_val",
)

ACTIVITY_METRICS = (
    "mean_firing_rate",
    "max_firing_rate",
    "silent_fraction",
    "overactive_fraction",
    "adaptation_mean",
    "adaptation_max",
    "membrane_mean",
    "membrane_max",
)

TOPOLOGY_METRICS = (
    "theta_grad_norm_pre_clip",
    "theta_grad_norm_post_clip",
    "theta_bias",
    "edge_prob_entropy",
    "edge_prob_mean",
    "edge_prob_std",
    "top_edge_prob_mean",
    "in_degree_gini",
    "out_degree_gini",
    "max_in_degree",
    "max_out_degree",
    "rec_input_abs_ratio",
)

REQUIRED_EPOCH_METRICS = PERFORMANCE_METRICS + ACTIVITY_METRICS + TOPOLOGY_METRICS
INTERVAL_TOPOLOGY_METRICS = (
    "theta_bias",
    "edge_prob_entropy",
    "edge_prob_mean",
    "edge_prob_std",
    "top_edge_prob_mean",
    "in_degree_gini",
    "out_degree_gini",
    "max_in_degree",
    "max_out_degree",
)
TRACE_ONLY_METRICS: tuple[str, ...] = ()


def safe_float(value: Any) -> float | None:
    """Convert scalar-like values to finite Python floats, otherwise None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        item = value.detach().float().reshape(-1)[0]
        out = float(item.cpu().item())
        return out if math.isfinite(out) else None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_list(values: list[Any] | tuple[Any, ...]) -> list[float]:
    out: list[float] = []
    for value in values:
        item = safe_float(value)
        if item is not None:
            out.append(item)
    return out


def safe_mean(values: list[Any] | tuple[Any, ...]) -> float | None:
    finite = _finite_list(values)
    return sum(finite) / len(finite) if finite else None


def safe_max(values: list[Any] | tuple[Any, ...]) -> float | None:
    finite = _finite_list(values)
    return max(finite) if finite else None


def safe_min(values: list[Any] | tuple[Any, ...]) -> float | None:
    finite = _finite_list(values)
    return min(finite) if finite else None


def safe_argmax(values: list[Any] | tuple[Any, ...]) -> int | None:
    best_idx: int | None = None
    best_value: float | None = None
    for idx, value in enumerate(values):
        item = safe_float(value)
        if item is None:
            continue
        if best_value is None or item > best_value:
            best_value = item
            best_idx = idx
    return best_idx


def relative_change(start: Any, final: Any) -> float | None:
    a = safe_float(start)
    b = safe_float(final)
    if a is None or b is None:
        return None
    denom = abs(a)
    if denom < EPS:
        return None
    return (b - a) / denom


def gini(values: Any) -> float | None:
    """Gini coefficient over finite non-negative values."""
    x = torch.as_tensor(values, dtype=torch.float32).detach().cpu().reshape(-1)
    x = x[torch.isfinite(x)].clamp(min=0.0)
    if x.numel() == 0:
        return None
    total = x.sum()
    if float(total.item()) <= 0.0:
        return 0.0
    sorted_x = x.sort().values
    n = sorted_x.numel()
    index = torch.arange(1, n + 1, dtype=sorted_x.dtype)
    coeff = (2.0 * torch.dot(index, sorted_x) / (n * total)) - ((n + 1.0) / n)
    return float(coeff.clamp(0.0, 1.0).item())


def entropy(probabilities: Any) -> float | None:
    """Mean Bernoulli entropy for deterministic edge probabilities."""
    p = torch.as_tensor(probabilities, dtype=torch.float32).detach().cpu().reshape(-1)
    p = p[torch.isfinite(p)]
    if p.numel() == 0:
        return None
    eps = max(EPS, float(torch.finfo(p.dtype).eps))
    p = p.clamp(eps, 1.0 - eps)
    ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return safe_float(ent.mean())


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_ready(safe_float(value))
        return [_json_ready(v) for v in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = safe_float(value)
    return item


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n")


def _row_value(row: dict[str, Any], key: str) -> float | None:
    return safe_float(row.get(key))


def _series(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    return [_row_value(row, key) for row in rows]


def _finite_pairs(rows: list[dict[str, Any]], key: str) -> list[tuple[int, float]]:
    pairs: list[tuple[int, float]] = []
    for row in rows:
        epoch = row.get("epoch")
        value = safe_float(row.get(key))
        if epoch is None or value is None:
            continue
        pairs.append((int(epoch), value))
    return pairs


def _peak_epoch(rows: list[dict[str, Any]], key: str) -> int | None:
    pairs = _finite_pairs(rows, key)
    if not pairs:
        return None
    return max(pairs, key=lambda item: item[1])[0]


def _largest_rise_epoch(rows: list[dict[str, Any]], key: str) -> int | None:
    pairs = _finite_pairs(rows, key)
    if len(pairs) < 2:
        return None
    best_epoch: int | None = None
    best_delta: float | None = None
    for (prev_epoch, prev), (epoch, value) in zip(pairs, pairs[1:]):
        delta = value - prev
        if best_delta is None or delta > best_delta:
            best_delta = delta
            best_epoch = epoch
    if best_delta is None or best_delta <= 0.0:
        return None
    return best_epoch


def _largest_drop_epoch(rows: list[dict[str, Any]], key: str) -> int | None:
    pairs = _finite_pairs(rows, key)
    if len(pairs) < 2:
        return None
    best_epoch: int | None = None
    best_delta: float | None = None
    for (prev_epoch, prev), (epoch, value) in zip(pairs, pairs[1:]):
        delta = value - prev
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_epoch = epoch
    if best_delta is None or best_delta >= 0.0:
        return None
    return best_epoch


def _infer_probability_source(raw: torch.Tensor, recurrent_mode: str) -> str:
    mode = str(recurrent_mode).lower()
    if mode in {
        "learned",
        "learned_lowrank",
        "learned_lowrank_grad_r",
        "learned_lowrank_frozen_w",
        "grad_r",
    }:
        return "logits_sigmoid"
    finite = raw.detach().float().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() and float(finite.min().item()) >= 0.0 and float(finite.max().item()) <= 1.0:
        return "probabilities_direct"
    return "logits_sigmoid"


def deterministic_edge_probabilities(
    model: torch.nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str]:
    """Return deterministic edge probabilities, valid mask, and source label."""
    liquid = getattr(model, "liquid", None)
    if liquid is None or not hasattr(liquid, "get_theta"):
        return None, None, "unavailable"
    mode = str(getattr(liquid, "mode", "")).lower()
    self_mask = getattr(liquid, "self_conn_mask", None)
    if self_mask is None:
        valid_mask = None
    else:
        valid_mask = self_mask.detach().bool().cpu()
    if mode in {"random_sparse", "fixed"}:
        fixed_mask = getattr(liquid, "fixed_mask", None)
        if fixed_mask is None:
            return None, None, f"unsupported_mode:{mode or 'unknown'}"
        probs = fixed_mask.detach().float().cpu().clamp(0.0, 1.0)
        if valid_mask is None:
            valid_mask = torch.ones_like(probs, dtype=torch.bool)
        return probs, valid_mask, f"fixed_mask:{mode}"
    if mode in {
        "none",
        "no_recurrence",
        "softplus_w_only",
        "edgewise_soft_conductance",
        "smooth_lowrank_conductance",
    }:
        return None, None, f"unsupported_mode:{mode or 'unknown'}"
    if mode in {"soft_gate_lowrank", "soft_gate_edgewise"}:
        if not hasattr(liquid, "_soft_gate_gate"):
            return None, None, f"unsupported_mode:{mode or 'unknown'}"
        with torch.no_grad():
            probs = liquid._soft_gate_gate().detach().float().cpu().clamp(0.0, 1.0)
        density_mask = getattr(liquid, "density_mask", None)
        if density_mask is not None:
            valid_mask = density_mask.detach().bool().cpu()
        elif valid_mask is None:
            valid_mask = torch.ones_like(probs, dtype=torch.bool)
        return probs, valid_mask, "soft_gate"
    with torch.no_grad():
        if mode == "grad_r" and hasattr(liquid, "theta"):
            raw = liquid.theta.detach().float().cpu()
        else:
            try:
                raw = liquid.get_theta().detach().float().cpu()
            except RuntimeError:
                return None, None, f"unsupported_mode:{mode or 'unknown'}"
    source = _infer_probability_source(raw, mode)
    if source == "probabilities_direct":
        probs = raw.clamp(0.0, 1.0)
    else:
        probs = torch.sigmoid(raw)
    if valid_mask is None:
        valid_mask = torch.ones_like(probs, dtype=torch.bool)
    return probs, valid_mask, source


def collect_topology_metrics(model: torch.nn.Module) -> dict[str, Any]:
    """Compute deterministic scalar topology metrics from edge probabilities."""
    probs, valid_mask, source = deterministic_edge_probabilities(model)
    out: dict[str, Any] = {
        "topology_probability_source": source,
        "theta_bias": None,
        "edge_prob_entropy": None,
        "edge_prob_mean": None,
        "edge_prob_std": None,
        "top_edge_prob_mean": None,
        "in_degree_gini": None,
        "out_degree_gini": None,
        "max_in_degree": None,
        "max_out_degree": None,
    }
    liquid = getattr(model, "liquid", None)
    if liquid is not None and hasattr(liquid, "theta_bias"):
        out["theta_bias"] = safe_float(liquid.theta_bias.detach().cpu())
    if probs is None or valid_mask is None:
        return out

    valid_probs = probs[valid_mask]
    if valid_probs.numel() == 0:
        return out

    expected = probs * valid_mask.float()
    expected_out_degree = expected.sum(dim=1)
    expected_in_degree = expected.sum(dim=0)
    k = min(50, int(valid_probs.numel()))
    top_probs = valid_probs.reshape(-1).sort(descending=True).values[:k]

    out.update(
        {
            "edge_prob_entropy": entropy(valid_probs),
            "edge_prob_mean": safe_float(valid_probs.mean()),
            "edge_prob_std": (
                safe_float(valid_probs.std(unbiased=False))
                if valid_probs.numel() > 1
                else 0.0
            ),
            "top_edge_prob_mean": safe_float(top_probs.mean()) if k > 0 else None,
            "in_degree_gini": gini(expected_in_degree),
            "out_degree_gini": gini(expected_out_degree),
            "max_in_degree": safe_float(expected_in_degree.max()),
            "max_out_degree": safe_float(expected_out_degree.max()),
        }
    )
    return out


def _activity_fractions(model: torch.nn.Module, cfg: Any) -> dict[str, float | None]:
    rates = getattr(model, "_last_spike_rates", None)
    if rates is None:
        return {"silent_fraction": None, "overactive_fraction": None}
    with torch.no_grad():
        per_neuron = rates.detach().float().mean(dim=0)
        silent_threshold = float(getattr(cfg, "silent_firing_rate_threshold", 0.001))
        overactive_threshold = float(
            getattr(cfg, "overactive_firing_rate_threshold", 0.20)
        )
        return {
            "silent_fraction": safe_float(
                (per_neuron <= silent_threshold).float().mean()
            ),
            "overactive_fraction": safe_float(
                (per_neuron >= overactive_threshold).float().mean()
            ),
        }


def collect_epoch_diagnostics(
    model: torch.nn.Module,
    raw_metrics: dict[str, Any],
    config: Any,
    *,
    force_topology: bool = False,
    final_epoch: bool = False,
) -> dict[str, Any]:
    """Build one canonical diagnostics row from trainer scalars and model state."""
    diag_cfg = getattr(config, "diagnostics", config)
    epoch = int(raw_metrics.get("epoch", 0))
    interval = max(1, int(getattr(diag_cfg, "topology_log_interval", 1)))
    recurrent_mode = str(getattr(getattr(config, "liquid", None), "recurrent_mode", ""))
    topology_due = bool(
        force_topology
        or final_epoch
        or epoch == 1
        or (interval > 0 and epoch % interval == 0)
        or raw_metrics.get("topology_rollback_applied")
        or raw_metrics.get("topology_frozen_epoch") == epoch
    )

    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": safe_float(raw_metrics.get("train_loss")),
        "val_loss": safe_float(raw_metrics.get("val_loss")),
        "train_acc": safe_float(raw_metrics.get("train_acc")),
        "val_acc": safe_float(raw_metrics.get("val_acc")),
        "best_val_acc_so_far": safe_float(raw_metrics.get("best_val_acc_so_far")),
        "test_at_best_val": safe_float(raw_metrics.get("test_at_best_val")),
        "test_at_best_val_expected": bool(
            raw_metrics.get("test_at_best_val_expected", False)
        ),
        "mean_firing_rate": safe_float(raw_metrics.get("mean_firing_rate")),
        "max_firing_rate": safe_float(raw_metrics.get("max_firing_rate")),
        "adaptation_mean": safe_float(
            raw_metrics.get("adaptation_mean", raw_metrics.get("mean_adaptation"))
        ),
        "adaptation_max": safe_float(
            raw_metrics.get("adaptation_max", raw_metrics.get("max_adaptation"))
        ),
        "membrane_mean": safe_float(raw_metrics.get("membrane_mean")),
        "membrane_max": safe_float(raw_metrics.get("membrane_max")),
        "input_current_abs_mean": safe_float(
            raw_metrics.get("input_current_abs_mean")
        ),
        "input_current_abs_max": safe_float(raw_metrics.get("input_current_abs_max")),
        "recurrent_current_abs_mean": safe_float(
            raw_metrics.get("recurrent_current_abs_mean")
        ),
        "recurrent_current_abs_max": safe_float(
            raw_metrics.get("recurrent_current_abs_max")
        ),
        "rec_input_abs_ratio": safe_float(raw_metrics.get("rec_input_abs_ratio")),
        "theta_grad_norm_pre_clip": safe_float(
            raw_metrics.get(
                "theta_grad_norm_pre_clip",
                raw_metrics.get("topology_grad_pre_clip"),
            )
        ),
        "theta_grad_norm_post_clip": safe_float(
            raw_metrics.get(
                "theta_grad_norm_post_clip",
                raw_metrics.get("topology_grad_post_clip"),
            )
        ),
        "recurrent_mode": recurrent_mode,
        "neuron_type": raw_metrics.get("neuron_type"),
        "readout_mode": raw_metrics.get("readout_mode"),
        "topology_metrics_logged": topology_due,
        "topology_probability_source": raw_metrics.get("topology_probability_source"),
        "theta_bias": safe_float(raw_metrics.get("theta_bias")),
        "edge_prob_entropy": safe_float(
            raw_metrics.get(
                "edge_prob_entropy",
                raw_metrics.get("topology_entropy"),
            )
        ),
        "edge_prob_mean": safe_float(raw_metrics.get("edge_prob_mean")),
        "edge_prob_std": safe_float(raw_metrics.get("edge_prob_std")),
        "top_edge_prob_mean": safe_float(raw_metrics.get("top_edge_prob_mean")),
        "in_degree_gini": safe_float(raw_metrics.get("in_degree_gini")),
        "out_degree_gini": safe_float(raw_metrics.get("out_degree_gini")),
        "max_in_degree": safe_float(raw_metrics.get("max_in_degree")),
        "max_out_degree": safe_float(raw_metrics.get("max_out_degree")),
        "interval_skipped_metrics": [],
        "unsupported_without_extra_forward_trace_metrics": list(TRACE_ONLY_METRICS),
    }
    row.update(_activity_fractions(model, diag_cfg))

    topology_missing = any(row.get(key) is None for key in INTERVAL_TOPOLOGY_METRICS)
    if topology_missing and topology_due:
        row.update(collect_topology_metrics(model))
    elif topology_missing:
        row["interval_skipped_metrics"] = [
            key for key in INTERVAL_TOPOLOGY_METRICS if row.get(key) is None
        ]
    else:
        row["topology_metrics_logged"] = True

    return {key: row.get(key) for key in REQUIRED_EPOCH_METRICS} | {
        "recurrent_mode": row.get("recurrent_mode"),
        "neuron_type": row.get("neuron_type"),
        "readout_mode": row.get("readout_mode"),
        "topology_metrics_logged": row.get("topology_metrics_logged"),
        "topology_probability_source": row.get("topology_probability_source"),
        "interval_skipped_metrics": row.get("interval_skipped_metrics", []),
        "unsupported_without_extra_forward_trace_metrics": row.get(
            "unsupported_without_extra_forward_trace_metrics", []
        ),
        "test_at_best_val_expected": row.get("test_at_best_val_expected"),
        "input_current_abs_mean": row.get("input_current_abs_mean"),
        "input_current_abs_max": row.get("input_current_abs_max"),
        "recurrent_current_abs_mean": row.get("recurrent_current_abs_mean"),
        "recurrent_current_abs_max": row.get("recurrent_current_abs_max"),
    }


class DiagnosticsLogger:
    def __init__(self, run_dir, config):
        self.run_dir = Path(run_dir)
        self.config = config
        self.cfg = getattr(config, "diagnostics", config)
        self.diagnostics_dir = self.run_dir / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = self.diagnostics_dir / "epoch_metrics.jsonl"
        self.summary_path = self.diagnostics_dir / "run_summary.json"
        self.red_flags_path = self.diagnostics_dir / "red_flags.json"
        self.report_path = self.diagnostics_dir / "diagnostic_report.md"
        self.plot_path = self.diagnostics_dir / "metric_trends.png"
        self.rows: list[dict[str, Any]] = []
        if bool(getattr(self.cfg, "save_raw_jsonl", True)) and not self.raw_path.exists():
            self.raw_path.write_text("")

    def log_epoch(self, epoch: int, metrics: dict):
        row = dict(metrics)
        row["epoch"] = int(epoch)
        row = _json_ready(row)
        self.rows.append(row)
        if bool(getattr(self.cfg, "save_raw_jsonl", True)):
            with self.raw_path.open("a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    def save_topology_snapshot(
        self, model: torch.nn.Module, label: str, epoch: int
    ) -> Path | None:
        if not bool(getattr(self.cfg, "save_full_topology_snapshots", False)):
            return None
        enabled_labels = {
            str(item).lower()
            for item in getattr(self.cfg, "full_snapshot_epochs", [])
        }
        label = str(label).lower()
        if label not in enabled_labels:
            return None
        probs, valid_mask, source = deterministic_edge_probabilities(model)
        if probs is None or valid_mask is None:
            return None
        path = self.diagnostics_dir / f"topology_snapshot_{label}.json"
        _write_json(
            path,
            {
                "epoch": int(epoch),
                "label": label,
                "topology_probability_source": source,
                "edge_probabilities": probs.tolist(),
                "valid_mask": valid_mask.int().tolist(),
            },
        )
        return path

    def summarize_run(self):
        summary = self._build_summary()
        red_flags = self.detect_red_flags(summary)
        summary["status"] = red_flags["status"]
        summary["primary_status"] = red_flags["status"]
        summary["red_flags"] = red_flags["triggered_flags"]
        if bool(getattr(self.cfg, "save_summary_json", True)):
            _write_json(self.summary_path, summary)
        if bool(getattr(self.cfg, "save_red_flags_json", True)):
            _write_json(self.red_flags_path, red_flags)
        if bool(getattr(self.cfg, "save_markdown_report", True)):
            self.write_markdown_report(summary, red_flags)
        if bool(getattr(self.cfg, "save_trend_plots", True)):
            self.save_trend_plots()
        return summary

    def _build_summary(self) -> dict[str, Any]:
        rows = list(self.rows)
        missing = self._missing_metric_categories(rows)
        metric_summaries = {}
        for key in REQUIRED_EPOCH_METRICS:
            values = _series(rows, key)
            metric_summaries[key] = {
                "start": next((v for v in values if v is not None), None),
                "final": next((v for v in reversed(values) if v is not None), None),
                "min": safe_min(values),
                "max": safe_max(values),
                "mean": safe_mean(values),
            }

        val_values = _series(rows, "val_acc")
        test_at_best_values = _series(rows, "test_at_best_val")
        final_row = rows[-1] if rows else {}
        probability_sources = sorted(
            {
                str(row.get("topology_probability_source"))
                for row in rows
                if row.get("topology_probability_source")
            }
        )
        summary = {
            "num_epochs": len(rows),
            "status": None,
            "topology_probability_source": (
                probability_sources[0]
                if len(probability_sources) == 1
                else probability_sources
            ),
            "performance": {
                "best_val_acc": safe_max(val_values),
                "test_at_best_val": next(
                    (v for v in reversed(test_at_best_values) if v is not None),
                    None,
                ),
                "final_val_acc": safe_float(final_row.get("val_acc")),
                "final_train_acc": safe_float(final_row.get("train_acc")),
            },
            "activity": {
                "mean_firing_rate": metric_summaries["mean_firing_rate"],
                "max_firing_rate": metric_summaries["max_firing_rate"],
                "adaptation_mean": metric_summaries["adaptation_mean"],
                "adaptation_max": metric_summaries["adaptation_max"],
            },
            "topology": {
                "edge_prob_entropy": metric_summaries["edge_prob_entropy"],
                "in_degree_gini": metric_summaries["in_degree_gini"],
                "out_degree_gini": metric_summaries["out_degree_gini"],
                "top_edge_prob_mean": metric_summaries["top_edge_prob_mean"],
                "max_in_degree": metric_summaries["max_in_degree"],
                "max_out_degree": metric_summaries["max_out_degree"],
            },
            "gradient": {
                "theta_grad_norm_pre_clip": metric_summaries[
                    "theta_grad_norm_pre_clip"
                ],
                "theta_grad_norm_post_clip": metric_summaries[
                    "theta_grad_norm_post_clip"
                ],
            },
            "metric_summaries": metric_summaries,
            "temporal_order": self._temporal_order(rows),
            "missing_metrics": missing,
            "interval_skipped_metrics": missing["interval_skipped_metrics"],
            "unsupported_without_extra_forward_trace_metrics": missing[
                "unsupported_without_extra_forward_trace_metrics"
            ],
            "unexpectedly_missing_metrics": missing["unexpectedly_missing_metrics"],
        }
        return summary

    def _missing_metric_categories(self, rows: list[dict[str, Any]]) -> dict[str, list[str]]:
        interval_skipped = sorted(
            {
                metric
                for row in rows
                for metric in row.get("interval_skipped_metrics", [])
            }
        )
        trace_only = sorted(
            {
                metric
                for row in rows
                for metric in row.get(
                    "unsupported_without_extra_forward_trace_metrics", []
                )
            }
            or set(TRACE_ONLY_METRICS)
        )
        unexpected: list[str] = []
        for metric in REQUIRED_EPOCH_METRICS:
            if metric in trace_only:
                continue
            if metric == "test_at_best_val" and not any(
                bool(row.get("test_at_best_val_expected")) for row in rows
            ):
                continue
            finite_count = sum(row.get(metric) is not None for row in rows)
            if finite_count > 0:
                continue
            if metric in interval_skipped:
                continue
            unexpected.append(metric)
        return {
            "interval_skipped_metrics": interval_skipped,
            "unsupported_without_extra_forward_trace_metrics": trace_only,
            "unexpectedly_missing_metrics": sorted(unexpected),
        }

    def _temporal_order(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        in_gini_rise = _largest_rise_epoch(rows, "in_degree_gini")
        out_gini_rise = _largest_rise_epoch(rows, "out_degree_gini")
        degree_gini_rise_epoch = min(
            [v for v in (in_gini_rise, out_gini_rise) if v is not None],
            default=None,
        )
        entropy_drop = _largest_drop_epoch(rows, "edge_prob_entropy")
        firing_peak = _peak_epoch(rows, "max_firing_rate")
        adaptation_peak = _peak_epoch(rows, "adaptation_max")
        grad_spike = self._theta_grad_spike_epoch(rows)
        rec_rise = _largest_rise_epoch(rows, "rec_input_abs_ratio")

        clauses: list[str] = []
        if entropy_drop is not None and firing_peak is not None:
            if entropy_drop < firing_peak:
                clauses.append(
                    "edge entropy decreased before recurrent firing peaked; this is consistent with topology concentration preceding recurrent-heavy activity, but not causal proof"
                )
            elif entropy_drop > firing_peak:
                clauses.append(
                    "recurrent firing peaked before the largest observed edge entropy drop; topology concentration was not clearly earlier"
                )
        if degree_gini_rise_epoch is not None and grad_spike is not None:
            if degree_gini_rise_epoch <= grad_spike:
                clauses.append(
                    "degree concentration rose before or at the topology-gradient spike"
                )
            else:
                clauses.append(
                    "the topology-gradient spike appeared before the largest expected-degree gini rise"
                )
        interpretation = (
            "; ".join(clauses) + "."
            if clauses
            else "available scalar diagnostics do not establish a clear temporal ordering."
        )
        return {
            "edge_entropy_drop_epoch": entropy_drop,
            "degree_gini_rise_epoch": degree_gini_rise_epoch,
            "rec_input_ratio_rise_epoch": rec_rise,
            "max_firing_peak_epoch": firing_peak,
            "adaptation_peak_epoch": adaptation_peak,
            "theta_grad_spike_epoch": grad_spike,
            "interpretation": interpretation,
        }

    def _theta_grad_spike_epoch(self, rows: list[dict[str, Any]]) -> int | None:
        pairs = _finite_pairs(rows, "theta_grad_norm_pre_clip")
        if not pairs:
            return None
        values = [value for _, value in pairs]
        max_epoch, max_value = max(pairs, key=lambda item: item[1])
        median = sorted(values)[len(values) // 2]
        abs_threshold = float(getattr(self.cfg, "theta_grad_spike_abs_threshold", 50.0))
        multiplier = float(getattr(self.cfg, "theta_grad_spike_multiplier", 3.0))
        if max_value >= abs_threshold:
            return max_epoch
        if median > 0.0 and max_value >= median * multiplier:
            return max_epoch
        return None

    def detect_red_flags(self, summary: dict):
        rows = self.rows
        thresholds = self.cfg
        missing = summary.get("unexpectedly_missing_metrics", [])
        required_interpretable = [
            metric
            for metric in REQUIRED_EPOCH_METRICS
            if metric not in TRACE_ONLY_METRICS and metric != "test_at_best_val"
        ]
        missing_fraction = len(missing) / max(len(required_interpretable), 1)
        insufficient = missing_fraction >= float(
            getattr(thresholds, "missing_required_fraction_threshold", 0.35)
        )

        mean_firing_values = _finite_list(_series(rows, "mean_firing_rate"))
        max_firing_values = _finite_list(_series(rows, "max_firing_rate"))
        adaptation_mean_values = _finite_list(_series(rows, "adaptation_mean"))
        adaptation_max_values = _finite_list(_series(rows, "adaptation_max"))
        val_values = _finite_list(_series(rows, "val_acc"))
        rec_values = _finite_list(_series(rows, "rec_input_abs_ratio"))
        entropy_values = _finite_list(_series(rows, "edge_prob_entropy"))
        in_gini_values = _finite_list(_series(rows, "in_degree_gini"))
        out_gini_values = _finite_list(_series(rows, "out_degree_gini"))
        top_prob_values = _finite_list(_series(rows, "top_edge_prob_mean"))
        grad_values = _finite_list(_series(rows, "theta_grad_norm_pre_clip"))

        low_firing_threshold = float(
            getattr(thresholds, "dead_mean_firing_rate_threshold", 0.005)
        )
        dead_fraction_threshold = float(
            getattr(thresholds, "dead_fraction_epochs", 0.70)
        )
        low_firing_fraction = (
            sum(v <= low_firing_threshold for v in mean_firing_values)
            / len(mean_firing_values)
            if mean_firing_values
            else 0.0
        )
        val_improvement = (
            max(val_values) - val_values[0] if len(val_values) >= 2 else None
        )
        dead = (
            bool(mean_firing_values)
            and low_firing_fraction >= dead_fraction_threshold
            and (
                not adaptation_mean_values
                or safe_max(adaptation_mean_values) is None
                or safe_max(adaptation_mean_values)
                <= float(getattr(thresholds, "adaptation_near_zero_threshold", 0.01))
            )
            and (
                val_improvement is None
                or val_improvement
                <= float(getattr(thresholds, "val_improvement_min_delta", 0.01))
            )
        )

        high_firing = (
            safe_max(max_firing_values) is not None
            and safe_max(max_firing_values)
            >= float(getattr(thresholds, "high_max_firing_rate_threshold", 0.80))
        )
        rec_high = (
            safe_max(rec_values) is not None
            and safe_max(rec_values)
            >= float(getattr(thresholds, "rec_input_high_threshold", 2.0))
        )
        val_volatility = self._volatile_or_degrading(val_values)
        unstable = bool(high_firing and (rec_high or val_volatility))

        entropy_drop = self._relative_drop(entropy_values)
        in_gini_rise = self._relative_rise(in_gini_values)
        out_gini_rise = self._relative_rise(out_gini_values)
        top_prob_rise = self._relative_rise(top_prob_values)
        topology_collapse = (
            entropy_drop
            >= float(getattr(thresholds, "topology_entropy_drop_threshold", 0.25))
            and max(in_gini_rise, out_gini_rise)
            >= float(getattr(thresholds, "degree_gini_rise_threshold", 0.20))
            and top_prob_rise
            >= float(getattr(thresholds, "top_edge_prob_rise_threshold", 0.02))
        )

        grad_spike_epoch = summary.get("temporal_order", {}).get(
            "theta_grad_spike_epoch"
        )
        gradient_instability = grad_spike_epoch is not None or (
            safe_max(grad_values) is not None
            and safe_max(grad_values)
            >= float(getattr(thresholds, "theta_grad_spike_abs_threshold", 50.0))
        )

        adaptation_saturation = self._adaptation_saturation(
            adaptation_max_values,
            max_firing_values,
        )

        triggered = []
        support = {
            "missing_fraction": missing_fraction,
            "low_firing_fraction": low_firing_fraction,
            "val_improvement": val_improvement,
            "max_firing_rate": safe_max(max_firing_values),
            "max_rec_input_abs_ratio": safe_max(rec_values),
            "entropy_relative_drop": entropy_drop,
            "in_degree_gini_relative_rise": in_gini_rise,
            "out_degree_gini_relative_rise": out_gini_rise,
            "top_edge_prob_relative_rise": top_prob_rise,
            "theta_grad_spike_epoch": grad_spike_epoch,
            "adaptation_max": safe_max(adaptation_max_values),
        }
        for name, value in (
            ("insufficient_diagnostics", insufficient),
            ("gradient_instability", gradient_instability),
            ("topology_collapse", topology_collapse),
            ("unstable_recurrent_regime", unstable),
            ("adaptation_saturation", adaptation_saturation),
            ("dead_reservoir", dead),
        ):
            if value:
                triggered.append(name)
        status = triggered[0] if triggered else "healthy"
        return {
            "status": status,
            "triggered_flags": triggered,
            "support": support,
            "missing_metrics": summary.get("missing_metrics", {}),
        }

    def _relative_drop(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        start = values[0]
        min_value = min(values)
        if abs(start) < EPS:
            return 0.0
        return max(0.0, (start - min_value) / abs(start))

    def _relative_rise(self, values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        start = values[0]
        max_value = max(values)
        denom = max(abs(start), EPS)
        return max(0.0, (max_value - start) / denom)

    def _volatile_or_degrading(self, values: list[float]) -> bool:
        if len(values) < 4:
            return False
        final = values[-1]
        best = max(values)
        degradation = best - final
        mean_value = sum(values) / len(values)
        variance = sum((v - mean_value) ** 2 for v in values) / len(values)
        return degradation >= 0.05 or math.sqrt(variance) >= 0.04

    def _adaptation_saturation(
        self, adaptation_max_values: list[float], max_firing_values: list[float]
    ) -> bool:
        if len(adaptation_max_values) < 4 or not max_firing_values:
            return False
        high_adapt = max(adaptation_max_values) >= float(
            getattr(self.cfg, "adaptation_saturation_threshold", 1.0)
        )
        tail = adaptation_max_values[len(adaptation_max_values) // 2 :]
        flat_tail = (
            max(tail) - min(tail)
            <= float(getattr(self.cfg, "adaptation_saturation_flat_delta", 0.05))
            if len(tail) >= 2
            else False
        )
        firing_high = max(max_firing_values) >= float(
            getattr(self.cfg, "high_max_firing_rate_threshold", 0.80)
        )
        return high_adapt and flat_tail and firing_high

    def write_markdown_report(self, summary: dict, red_flags: dict):
        def fmt(value: Any) -> str:
            item = safe_float(value)
            return "null" if item is None else f"{item:.6g}"

        perf = summary.get("performance", {})
        activity = summary.get("activity", {})
        topology = summary.get("topology", {})
        gradient = summary.get("gradient", {})
        temporal = summary.get("temporal_order", {})
        missing = summary.get("missing_metrics", {})
        lines = [
            "# Diagnostic Report",
            "",
            "## Status",
            f"status: {red_flags.get('status', 'unknown')}",
            "",
            "## Performance",
            f"- best_val_acc: {fmt(perf.get('best_val_acc'))}",
            f"- test@best_val: {fmt(perf.get('test_at_best_val'))}",
            f"- final_val_acc: {fmt(perf.get('final_val_acc'))}",
            "",
            "## Activity",
            f"- mean_firing_rate: {fmt(activity.get('mean_firing_rate', {}).get('final'))}",
            f"- max_firing_rate: {fmt(activity.get('max_firing_rate', {}).get('max'))}",
            f"- adaptation_mean: {fmt(activity.get('adaptation_mean', {}).get('final'))}",
            f"- adaptation_max: {fmt(activity.get('adaptation_max', {}).get('max'))}",
            "",
            "## Topology",
            "- edge_prob_entropy start/final/min: "
            f"{fmt(topology.get('edge_prob_entropy', {}).get('start'))}/"
            f"{fmt(topology.get('edge_prob_entropy', {}).get('final'))}/"
            f"{fmt(topology.get('edge_prob_entropy', {}).get('min'))}",
            "- in_degree_gini start/final/max: "
            f"{fmt(topology.get('in_degree_gini', {}).get('start'))}/"
            f"{fmt(topology.get('in_degree_gini', {}).get('final'))}/"
            f"{fmt(topology.get('in_degree_gini', {}).get('max'))}",
            "- out_degree_gini start/final/max: "
            f"{fmt(topology.get('out_degree_gini', {}).get('start'))}/"
            f"{fmt(topology.get('out_degree_gini', {}).get('final'))}/"
            f"{fmt(topology.get('out_degree_gini', {}).get('max'))}",
            "- top_edge_prob_mean start/final/max: "
            f"{fmt(topology.get('top_edge_prob_mean', {}).get('start'))}/"
            f"{fmt(topology.get('top_edge_prob_mean', {}).get('final'))}/"
            f"{fmt(topology.get('top_edge_prob_mean', {}).get('max'))}",
            "",
            "## Gradient",
            "- theta_grad_norm_pre_clip max: "
            f"{fmt(gradient.get('theta_grad_norm_pre_clip', {}).get('max'))}",
            f"- theta_grad_spike_epoch: {temporal.get('theta_grad_spike_epoch')}",
            "",
            "## Temporal Order",
            str(temporal.get("interpretation", "")),
            "",
            "## Red Flags",
        ]
        triggered = red_flags.get("triggered_flags", [])
        lines.extend([f"- {flag}" for flag in triggered] or ["- none"])
        lines.extend(
            [
                "",
                "## Missing Metrics",
                "- interval_skipped_metrics: "
                + ", ".join(missing.get("interval_skipped_metrics", []) or ["none"]),
                "- unsupported_without_extra_forward_trace_metrics: "
                + ", ".join(
                    missing.get(
                        "unsupported_without_extra_forward_trace_metrics", []
                    )
                    or ["none"]
                ),
                "- unexpectedly_missing_metrics: "
                + ", ".join(
                    missing.get("unexpectedly_missing_metrics", []) or ["none"]
                ),
                "",
                "## Interpretation",
                self._diagnosis(summary, red_flags),
                "",
            ]
        )
        self.report_path.write_text("\n".join(lines))

    def _diagnosis(self, summary: dict, red_flags: dict) -> str:
        status = red_flags.get("status", "unknown")
        if status == "healthy":
            return "No conservative red-flag heuristic was triggered by the available scalar diagnostics."
        if status == "insufficient_diagnostics":
            return "Required scalar diagnostics were unexpectedly missing, so run health cannot be classified reliably."
        if status == "topology_collapse":
            return "Expected-degree concentration and edge-probability concentration increased while entropy fell; this is consistent with topology collapse."
        if status == "gradient_instability":
            return "Topology-gradient spikes were detected. The temporal-order section describes whether concentration or firing appeared earlier."
        if status == "unstable_recurrent_regime":
            return "High firing and volatile or recurrent-heavy activity were detected; this suggests an unstable recurrent regime."
        if status == "dead_reservoir":
            return "Firing stayed very low with little validation improvement, consistent with a dead reservoir."
        if status == "adaptation_saturation":
            return "Adaptation was high and flat while firing remained high, consistent with adaptation reacting but not fully controlling dynamics."
        return "Diagnostics are conservative and do not establish causal mechanism."

    def save_trend_plots(self):
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None
        if not self.rows:
            return None

        epochs = [row.get("epoch") for row in self.rows]
        series = [
            ("val_acc", "val_acc"),
            ("max_firing_rate", "max_firing"),
            ("edge_prob_entropy", "edge_entropy"),
            ("in_degree_gini", "in_gini"),
            ("out_degree_gini", "out_gini"),
            ("theta_grad_norm_pre_clip", "theta_grad"),
        ]
        fig, axes = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)
        for ax, (key, title) in zip(axes.reshape(-1), series):
            values = [row.get(key) for row in self.rows]
            ax.plot(epochs, [float("nan") if v is None else v for v in values])
            ax.set_title(title)
            ax.set_xlabel("epoch")
        fig.savefig(self.plot_path, dpi=140)
        plt.close(fig)
        return self.plot_path
