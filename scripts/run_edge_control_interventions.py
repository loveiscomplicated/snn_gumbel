"""Run matched edge-control interventions for learned-lowrank SHD LSM checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.lowrank_interventions import (  # noqa: E402
    InterventionSpec,
    evaluate_on_batches,
    stable_seed,
    temporary_recurrent_mask,
)
from src.analysis.lowrank_runaway import (  # noqa: E402
    _select_device,
    load_model_and_checkpoint,
)
from src.data.loaders import get_train_val_test_dataloaders  # noqa: E402
from src.utils.config import load_config  # noqa: E402


BASELINE_RULE_NATIVE = "native_eval"
BASELINE_RULE_TOPK = "topk_configured"
NATIVE_ACTIVE_EDGE_RULE = "native_eval_recurrent_mask"
TOPK_ACTIVE_EDGE_RULE = "topk_by_configured_recurrent_sparsity"
ACTIVE_EDGE_RULE = TOPK_ACTIVE_EDGE_RULE
BASELINE_MATCH_TOL = 1e-6
MISSING_RECURRENT_CURRENT = "missing_recurrent_current_diagnostics"


@dataclass(frozen=True)
class TopologyInfo:
    edge_prob: torch.Tensor
    theta_min: float
    theta_max: float
    topology_probability_source: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DegreeMatchInfo:
    mean_target_expected_out_degree: float | None = None
    mean_control_expected_out_degree: float | None = None
    mean_abs_degree_gap: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint-level recurrent edge-control interventions."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-random-repeats", type=int, default=20)
    parser.add_argument("--top-k-edges", type=int, default=50)
    parser.add_argument("--top-k-neurons", type=int, default=10)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--baseline-mask-rule",
        choices=[BASELINE_RULE_NATIVE, BASELINE_RULE_TOPK],
        default=BASELINE_RULE_NATIVE,
        help=(
            "Baseline recurrent mask for edge interventions. native_eval uses the "
            "checkpoint's deterministic eval mask; topk_configured keeps the "
            "configured-sparsity deterministic top-k mask."
        ),
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=None,
        help="Optional lowrank runaway diagnostic directory containing neuron_table.csv.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().float().cpu().reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(finite) if finite else None


def _std(values: Iterable[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    if len(finite) == 1:
        return 0.0
    return statistics.pstdev(finite)


def _format_ids(values: Iterable[int]) -> str:
    return ";".join(str(int(v)) for v in values)


def valid_nonself_mask(
    shape: tuple[int, int],
    self_conn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = torch.ones(shape, dtype=torch.bool)
    if shape[0] == shape[1]:
        valid.fill_diagonal_(False)
    if self_conn_mask is not None:
        valid &= self_conn_mask.detach().cpu().bool()
    if shape[0] == shape[1]:
        valid.fill_diagonal_(False)
    return valid


def extract_topology_probabilities(model: torch.nn.Module) -> TopologyInfo:
    liquid = model.liquid
    theta = liquid.get_theta().detach().float().cpu()
    theta_min = _safe_float(theta.min())
    theta_max = _safe_float(theta.max())
    warnings: list[str] = []

    if str(getattr(liquid, "mode", "")).lower() == "learned_lowrank":
        source = "logits_sigmoid"
        if theta_min >= 0.0 and theta_max <= 1.0:
            warnings.append(
                "theta_values_bounded_0_1_for_learned_lowrank_logits_path"
            )
        edge_prob = torch.sigmoid(theta)
    elif theta_min >= 0.0 and theta_max <= 1.0:
        source = "probabilities_direct"
        edge_prob = theta
    else:
        source = "logits_sigmoid"
        edge_prob = torch.sigmoid(theta)

    return TopologyInfo(
        edge_prob=edge_prob,
        theta_min=theta_min,
        theta_max=theta_max,
        topology_probability_source=source,
        warnings=tuple(warnings),
    )


def ranked_edges_by_score(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> list[tuple[int, int]]:
    scores = scores.detach().float().cpu()
    candidate_mask = candidate_mask.detach().cpu().bool()
    positions = candidate_mask.nonzero(as_tuple=False)
    rows: list[tuple[float, int, int]] = []
    for src, dst in positions.tolist():
        rows.append((float(scores[src, dst].item()), int(src), int(dst)))
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(src, dst) for _, src, dst in rows]


def build_topk_active_mask(
    edge_prob: torch.Tensor,
    valid_mask: torch.Tensor,
    recurrent_sparsity: float,
) -> tuple[torch.Tensor, int]:
    valid_count = int(valid_mask.sum().item())
    num_active = int(round(float(recurrent_sparsity) * valid_count))
    num_active = max(0, min(valid_count, num_active))
    active = torch.zeros_like(valid_mask, dtype=torch.bool)
    for src, dst in ranked_edges_by_score(edge_prob, valid_mask)[:num_active]:
        active[src, dst] = True
    return active, num_active


def build_native_active_mask(
    model: torch.nn.Module,
    valid_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    model.eval()
    if hasattr(model.liquid, "unlock_epoch_mask"):
        model.liquid.unlock_epoch_mask()
    with torch.no_grad(), temporary_recurrent_mask(model):
        mask = model.liquid.sample_mask(tau=tau).detach().cpu().bool()
    return mask & valid_mask.detach().cpu().bool()


def active_density(active_mask: torch.Tensor, valid_mask: torch.Tensor) -> float:
    valid_count = int(valid_mask.detach().cpu().bool().sum().item())
    if valid_count <= 0:
        return 0.0
    return float(active_mask.detach().cpu().bool().sum().item() / valid_count)


def top_probability_active_edges(
    active_mask: torch.Tensor,
    edge_prob: torch.Tensor,
    top_k: int,
) -> list[tuple[int, int]]:
    count = max(0, min(int(top_k), int(active_mask.sum().item())))
    return ranked_edges_by_score(edge_prob, active_mask)[:count]


def active_edge_list(active_mask: torch.Tensor) -> list[tuple[int, int]]:
    return [(int(src), int(dst)) for src, dst in active_mask.nonzero(as_tuple=False).tolist()]


def sample_random_active_edges(
    active_mask: torch.Tensor,
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    edges = active_edge_list(active_mask)
    rng = random.Random(int(seed))
    if count >= len(edges):
        return sorted(edges)
    return sorted(rng.sample(edges, int(count)))


def apply_edge_removal(
    active_mask: torch.Tensor,
    edges: Iterable[tuple[int, int]],
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    keep_mask = active_mask.detach().cpu().bool().clone()
    removed: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for src, dst in edges:
        edge = (int(src), int(dst))
        if edge in seen:
            continue
        seen.add(edge)
        if bool(keep_mask[edge[0], edge[1]].item()):
            keep_mask[edge[0], edge[1]] = False
            removed.append(edge)
    return keep_mask, removed


def expected_degrees(
    edge_prob: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weighted = edge_prob.detach().float().cpu() * valid_mask.detach().float().cpu()
    expected_out = weighted.sum(dim=1)
    expected_in = weighted.sum(dim=0)
    return expected_in, expected_out


def select_top_neurons_by_score(score: torch.Tensor, top_k: int) -> list[int]:
    rows = [(float(value), int(idx)) for idx, value in enumerate(score.detach().cpu().tolist())]
    rows.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in rows[: max(0, min(int(top_k), len(rows)))]]


def hub_edges(
    active_mask: torch.Tensor,
    neurons: Iterable[int],
    direction: str,
) -> list[tuple[int, int]]:
    active = active_mask.detach().cpu().bool()
    edges: list[tuple[int, int]] = []
    for neuron in neurons:
        idx = int(neuron)
        if direction == "incoming":
            sources = active[:, idx].nonzero(as_tuple=False).reshape(-1).tolist()
            edges.extend((int(src), idx) for src in sources)
        elif direction == "outgoing":
            targets = active[idx].nonzero(as_tuple=False).reshape(-1).tolist()
            edges.extend((idx, int(dst)) for dst in targets)
        else:
            raise ValueError(f"unknown hub edge direction: {direction}")
    return edges


def ei_types_from_model(model: torch.nn.Module) -> list[str] | None:
    dale = getattr(model.liquid, "dale_sign", None)
    if not torch.is_tensor(dale):
        return None
    values = dale.detach().cpu().reshape(-1)
    if values.numel() == 0:
        return None
    return ["E" if float(value.item()) > 0.0 else "I" for value in values]


def sample_ei_matched_neurons(
    target_neurons: Iterable[int],
    ei_types: list[str],
    seed: int,
) -> tuple[list[int] | None, str]:
    targets = [int(idx) for idx in target_neurons]
    if not targets:
        return [], ""
    rng = random.Random(int(seed))
    target_set = set(targets)
    counts: dict[str, int] = {}
    for idx in targets:
        if idx < 0 or idx >= len(ei_types):
            return None, "missing_ei_metadata"
        counts[ei_types[idx]] = counts.get(ei_types[idx], 0) + 1
    selected: list[int] = []
    for ei_type, count in counts.items():
        pool = [
            idx
            for idx, value in enumerate(ei_types)
            if value == ei_type and idx not in target_set
        ]
        if len(pool) < count:
            return None, "insufficient_ei_matched_pool"
        selected.extend(rng.sample(pool, count))
    return sorted(selected), ""


def sample_degree_matched_neurons(
    target_neurons: Iterable[int],
    expected_out_degree: torch.Tensor,
    seed: int,
    nearest_pool_size: int = 20,
) -> tuple[list[int] | None, str, DegreeMatchInfo]:
    targets = [int(idx) for idx in target_neurons]
    if not targets:
        return [], "", DegreeMatchInfo(0.0, 0.0, 0.0)
    values = expected_out_degree.detach().float().cpu()
    n = int(values.numel())
    if any(idx < 0 or idx >= n for idx in targets):
        return None, "missing_degree_metadata", DegreeMatchInfo()

    rng = random.Random(int(seed))
    target_set = set(targets)
    selected: list[int] = []
    selected_set: set[int] = set()
    gaps: list[float] = []
    for target in targets:
        target_value = float(values[target].item())
        candidates: list[tuple[float, int]] = []
        for idx in range(n):
            if idx in target_set or idx in selected_set:
                continue
            candidates.append((abs(float(values[idx].item()) - target_value), idx))
        if not candidates:
            return None, "insufficient_degree_matched_pool", DegreeMatchInfo()
        candidates.sort(key=lambda item: (item[0], item[1]))
        adaptive_pool_size = max(1, int(math.ceil(0.02 * n)))
        pool_size = max(1, min(int(nearest_pool_size), adaptive_pool_size, len(candidates)))
        nearest = candidates[:pool_size]
        gap, control = rng.choice(nearest)
        selected.append(control)
        selected_set.add(control)
        gaps.append(float(gap))

    target_mean = float(values[targets].mean().item())
    control_mean = float(values[selected].mean().item()) if selected else 0.0
    return (
        sorted(selected),
        "",
        DegreeMatchInfo(
            mean_target_expected_out_degree=target_mean,
            mean_control_expected_out_degree=control_mean,
            mean_abs_degree_gap=statistics.fmean(gaps) if gaps else 0.0,
        ),
    )


def load_recurrent_current_targets(
    diagnostic_dir: Path | None,
    run_name: str,
    top_k: int,
) -> tuple[list[int] | None, str]:
    if diagnostic_dir is None:
        return None, MISSING_RECURRENT_CURRENT
    table_path = diagnostic_dir / "neuron_table.csv"
    if not table_path.exists():
        return None, MISSING_RECURRENT_CURRENT
    with table_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "recurrent_current_abs_mean" not in rows[0] or "neuron_id" not in rows[0]:
        return None, MISSING_RECURRENT_CURRENT

    matching = [row for row in rows if str(row.get("run_name", "")) == run_name]
    if matching:
        rows = matching
    elif len({str(row.get("run_name", "")) for row in rows}) > 1:
        return None, MISSING_RECURRENT_CURRENT

    candidates: list[tuple[float, int]] = []
    for row in rows:
        value = _safe_float(row.get("recurrent_current_abs_mean"))
        neuron_id = _safe_float(row.get("neuron_id"))
        if math.isfinite(value) and math.isfinite(neuron_id):
            candidates.append((value, int(neuron_id)))
    if not candidates:
        return None, MISSING_RECURRENT_CURRENT
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [idx for _, idx in candidates[: int(top_k)]], ""


def materialize_split_batches(
    cfg: Any,
    split: str,
    max_batches: int | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    train_loader, val_loader, test_loader = get_train_val_test_dataloaders(cfg)
    del train_loader
    if split == "val":
        if val_loader is None:
            raise ValueError("Validation split is unavailable for this config.")
        loader = val_loader
    else:
        loader = test_loader

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for idx, (x, y) in enumerate(loader):
        if max_batches is not None and idx >= int(max_batches):
            break
        batches.append((x.detach().cpu().clone(), y.detach().cpu().clone()))
    if not batches:
        raise RuntimeError(f"No batches were loaded for split={split!r}.")
    return batches


def evaluate_native_model(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    tau: float,
) -> float:
    model.eval()
    if hasattr(model.liquid, "unlock_epoch_mask"):
        model.liquid.unlock_epoch_mask()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in batches:
            x = x.to(device)
            y = y.to(device)
            logits = model(x, tau=tau)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def evaluate_mask_acc(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    mask: torch.Tensor,
    tau: float,
    intervention_id: str,
) -> float:
    spec = InterventionSpec(
        intervention_id=intervention_id,
        family="edge_control",
        intervention_type=intervention_id,
        mask_override=mask.detach().cpu().bool(),
    )
    metrics, _ = evaluate_on_batches(model, batches, spec, device, tau=tau)
    return float(metrics["accuracy"])


def verify_mask_override_effect(
    model: torch.nn.Module,
    active_mask: torch.Tensor,
) -> dict[str, Any]:
    zero_mask = torch.zeros_like(active_mask, dtype=torch.bool)
    with temporary_recurrent_mask(model, active_mask):
        active_effective_nonzero = int((model.liquid.get_effective_weight().detach().cpu() != 0).sum().item())
    with temporary_recurrent_mask(model, zero_mask):
        zero_effective_nonzero = int((model.liquid.get_effective_weight().detach().cpu() != 0).sum().item())

    verified = zero_effective_nonzero == 0 and active_effective_nonzero != zero_effective_nonzero
    if not verified:
        raise RuntimeError(
            "Mask override verification failed: recurrent mask override does not appear to affect effective weights."
        )
    return {
        "mask_override_verified": True,
        "active_effective_nonzero": active_effective_nonzero,
        "zero_effective_nonzero": zero_effective_nonzero,
    }


def result_row(
    *,
    intervention: str,
    repeat_id: int | str,
    baseline_acc: float,
    intervention_acc: float | None,
    removed_edges: int,
    target_neurons: Iterable[int] = (),
    control_type: str = "",
    available: bool = True,
    unavailable_reason: str = "",
    seed: int,
    split: str,
    checkpoint: Path,
    degree_info: DegreeMatchInfo | None = None,
) -> dict[str, Any]:
    intervention_acc_value = intervention_acc if intervention_acc is not None else float("nan")
    delta = (
        intervention_acc_value - baseline_acc
        if available and math.isfinite(intervention_acc_value)
        else float("nan")
    )
    info = degree_info or DegreeMatchInfo()
    return {
        "intervention": intervention,
        "repeat_id": repeat_id,
        "baseline_acc": baseline_acc,
        "intervention_acc": intervention_acc_value,
        "delta_acc": delta,
        "removed_edges": int(removed_edges),
        "target_neurons": _format_ids(target_neurons),
        "control_type": control_type,
        "available": bool(available),
        "unavailable_reason": unavailable_reason,
        "seed": int(seed),
        "split": split,
        "checkpoint": str(checkpoint),
        "mean_target_expected_out_degree": info.mean_target_expected_out_degree,
        "mean_control_expected_out_degree": info.mean_control_expected_out_degree,
        "mean_abs_degree_gap": info.mean_abs_degree_gap,
    }


def unavailable_row(
    *,
    intervention: str,
    baseline_acc: float,
    reason: str,
    seed: int,
    split: str,
    checkpoint: Path,
    control_type: str = "",
) -> dict[str, Any]:
    return result_row(
        intervention=intervention,
        repeat_id="",
        baseline_acc=baseline_acc,
        intervention_acc=None,
        removed_edges=0,
        control_type=control_type,
        available=False,
        unavailable_reason=reason,
        seed=seed,
        split=split,
        checkpoint=checkpoint,
    )


def evaluate_edge_removal(
    *,
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    active_mask: torch.Tensor,
    edges: Iterable[tuple[int, int]],
    baseline_acc: float,
    tau: float,
    intervention: str,
    repeat_id: int | str,
    target_neurons: Iterable[int],
    control_type: str,
    seed: int,
    split: str,
    checkpoint: Path,
    degree_info: DegreeMatchInfo | None = None,
) -> dict[str, Any]:
    keep_mask, removed = apply_edge_removal(active_mask, edges)
    acc = evaluate_mask_acc(
        model,
        batches,
        device,
        keep_mask,
        tau,
        intervention_id=f"{intervention}_{repeat_id}",
    )
    return result_row(
        intervention=intervention,
        repeat_id=repeat_id,
        baseline_acc=baseline_acc,
        intervention_acc=acc,
        removed_edges=len(removed),
        target_neurons=target_neurons,
        control_type=control_type,
        available=True,
        seed=seed,
        split=split,
        checkpoint=checkpoint,
        degree_info=degree_info,
    )


def summarize_results(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["intervention"] == "baseline":
            continue
        by_name.setdefault(str(row["intervention"]), []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for name, group in sorted(by_name.items()):
        available = [row for row in group if row.get("available")]
        deltas = [_safe_float(row.get("delta_acc")) for row in available]
        removed = [_safe_float(row.get("removed_edges")) for row in available]
        unavailable = [row for row in group if not row.get("available")]
        out[name] = {
            "mean_delta_acc": _mean(deltas),
            "std_delta_acc": _std(deltas),
            "removed_edges": _mean(removed),
            "n_rows": len(group),
            "n_available": len(available),
            "available": bool(available) and not unavailable,
            "unavailable_reasons": sorted(
                {
                    str(row.get("unavailable_reason"))
                    for row in unavailable
                    if row.get("unavailable_reason")
                }
            ),
        }
        for key in (
            "mean_target_expected_out_degree",
            "mean_control_expected_out_degree",
            "mean_abs_degree_gap",
        ):
            vals = [_safe_float(row.get(key)) for row in available]
            mean_val = _mean(vals)
            if mean_val is not None:
                out[name][key] = mean_val
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "intervention",
        "repeat_id",
        "baseline_acc",
        "intervention_acc",
        "delta_acc",
        "removed_edges",
        "target_neurons",
        "control_type",
        "available",
        "unavailable_reason",
        "seed",
        "split",
        "checkpoint",
        "mean_target_expected_out_degree",
        "mean_control_expected_out_degree",
        "mean_abs_degree_gap",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def interpretation_for(
    summaries: dict[str, dict[str, Any]],
    target: str,
    control: str,
    label: str,
) -> str:
    target_summary = summaries.get(target, {})
    control_summary = summaries.get(control, {})
    target_delta = target_summary.get("mean_delta_acc")
    control_delta = control_summary.get("mean_delta_acc")
    if target_delta is None or control_delta is None:
        return f"{label}: comparison unavailable because one side has no available rows."
    if target_delta < control_delta:
        return (
            f"{label}: target removal caused a larger accuracy drop than its control "
            f"({target_delta:.4f} vs {control_delta:.4f}), consistent with checkpoint-level "
            "decision sensitivity for that edge bundle."
        )
    return (
        f"{label}: target removal did not exceed the matched control drop "
        f"({target_delta:.4f} vs {control_delta:.4f}); this weakens a targeted-specificity interpretation."
    )


def write_report(
    path: Path,
    *,
    config: Path,
    checkpoint: Path,
    split: str,
    seed: int,
    original_model_eval_acc: float,
    native_active_baseline_acc: float,
    topk_active_baseline_acc: float,
    baseline_mask_rule: str,
    native_active_density: float,
    topk_active_density: float,
    baseline_matches_original_eval: bool,
    active_edge_rule: str,
    topology_probability_source: str,
    summaries: dict[str, dict[str, Any]],
    unavailable_rows: list[dict[str, Any]],
) -> str:
    baseline_phrase = (
        "native eval active-mask baseline"
        if baseline_mask_rule == BASELINE_RULE_NATIVE
        else "deterministic top-k configured-sparsity active-mask baseline"
    )
    lines = [
        "# Edge-Control Intervention Report",
        "",
        "## Setup",
        f"- config: `{config}`",
        f"- checkpoint: `{checkpoint}`",
        f"- split: `{split}`",
        f"- seed: `{seed}`",
        f"- original_model_eval_acc: `{original_model_eval_acc:.6f}`",
        f"- native_active_baseline_acc: `{native_active_baseline_acc:.6f}`",
        f"- topk_active_baseline_acc: `{topk_active_baseline_acc:.6f}`",
        f"- baseline_mask_rule: `{baseline_mask_rule}`",
        f"- native_active_density: `{native_active_density:.6f}`",
        f"- topk_active_density: `{topk_active_density:.6f}`",
        f"- baseline_matches_original_eval: `{str(baseline_matches_original_eval).lower()}`",
        f"- active_edge_rule: `{active_edge_rule}`",
        f"- topology_probability_source: `{topology_probability_source}`",
        "",
        f"Delta accuracy is measured against the {baseline_phrase}.",
        "The top-k configured active-mask baseline is retained as secondary sensitivity analysis and may differ from the original checkpoint eval path.",
        "",
        "## Results Summary",
        "",
        "| Intervention | Removed edges | Mean Δacc | Std Δacc | Available | Interpretation |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, summary in sorted(summaries.items()):
        mean_delta = summary.get("mean_delta_acc")
        std_delta = summary.get("std_delta_acc")
        removed = summary.get("removed_edges")
        available = summary.get("available")
        if summary.get("n_available", 0):
            if mean_delta is None:
                interpretation = "available; mean delta unavailable"
            elif mean_delta < 0:
                interpretation = f"accuracy lower than the {baseline_mask_rule} baseline"
            elif mean_delta > 0:
                interpretation = f"accuracy above the {baseline_mask_rule} baseline"
            else:
                interpretation = f"no mean accuracy change from the {baseline_mask_rule} baseline"
        else:
            interpretation = "; ".join(summary.get("unavailable_reasons", []))
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                name,
                "" if removed is None else f"{removed:.1f}",
                "" if mean_delta is None else f"{mean_delta:.4f}",
                "" if std_delta is None else f"{std_delta:.4f}",
                "yes" if available else "partial/no",
                interpretation or "",
            )
        )
    lines.extend(
        [
            "",
            "## Key Comparisons",
            "",
            "### Hub incoming vs random same-count",
            interpretation_for(
                summaries,
                "hub_incoming_remove",
                "random_edges_same_count_as_hub_incoming",
                "Hub incoming vs random same-count",
            ),
            "",
            "### Hub outgoing vs random same-count",
            interpretation_for(
                summaries,
                "hub_outgoing_remove",
                "random_edges_same_count_as_hub_outgoing",
                "Hub outgoing vs random same-count",
            ),
            "",
            "### Hub outgoing vs E/I-matched",
            interpretation_for(
                summaries,
                "hub_outgoing_remove",
                "ei_matched_outgoing_control",
                "Hub outgoing vs E/I-matched",
            ),
            "",
            "### Hub outgoing vs degree-matched",
            interpretation_for(
                summaries,
                "hub_outgoing_remove",
                "degree_matched_outgoing_control",
                "Hub outgoing vs degree-matched",
            ),
            "",
            "## Unavailable Interventions",
        ]
    )
    if unavailable_rows:
        seen_unavailable: set[tuple[str, str]] = set()
        for row in unavailable_rows:
            key = (str(row["intervention"]), str(row["unavailable_reason"]))
            if key in seen_unavailable:
                continue
            seen_unavailable.add(key)
            lines.append(f"- `{row['intervention']}`: {row['unavailable_reason']}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "This is fixed-checkpoint decision sensitivity, not retraining recovery and not proof of training-time causality.",
            "",
        ]
    )
    text = "\n".join(lines)
    path.write_text(text)
    return text


def maybe_write_plot(path: Path, summaries: dict[str, dict[str, Any]]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names = []
    values = []
    for name, summary in sorted(summaries.items()):
        value = summary.get("mean_delta_acc")
        if value is not None:
            names.append(name)
            values.append(value)
    if not values:
        return False
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(values)), values)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean delta accuracy")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def build_and_run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(str(args.config))
    device = _select_device(args.device)
    model, checkpoint_obj, load_warnings = load_model_and_checkpoint(cfg, args.checkpoint, device)
    del checkpoint_obj
    batches = materialize_split_batches(cfg, args.split, args.max_batches)
    tau = float(getattr(cfg, "tau_end", 1.0))

    original_acc = evaluate_native_model(model, batches, device, tau)
    topology = extract_topology_probabilities(model)
    self_conn_mask = getattr(model.liquid, "self_conn_mask", None)
    valid_mask = valid_nonself_mask(tuple(topology.edge_prob.shape), self_conn_mask)
    native_active_mask = build_native_active_mask(model, valid_mask, tau)
    topk_active_mask, topk_active_count = build_topk_active_mask(
        topology.edge_prob,
        valid_mask,
        recurrent_sparsity=float(cfg.liquid.recurrent_sparsity),
    )
    native_active_count = int(native_active_mask.sum().item())
    native_density = active_density(native_active_mask, valid_mask)
    topk_density = active_density(topk_active_mask, valid_mask)

    native_baseline_acc = evaluate_mask_acc(
        model,
        batches,
        device,
        native_active_mask,
        tau,
        intervention_id="baseline_native_active_mask",
    )
    topk_baseline_acc = evaluate_mask_acc(
        model,
        batches,
        device,
        topk_active_mask,
        tau,
        intervention_id="baseline_topk_active_mask",
    )
    baseline_diff = abs(native_baseline_acc - original_acc)
    baseline_matches_original_eval = baseline_diff <= BASELINE_MATCH_TOL
    active_mask = (
        native_active_mask
        if args.baseline_mask_rule == BASELINE_RULE_NATIVE
        else topk_active_mask
    )
    baseline_acc = (
        native_baseline_acc
        if args.baseline_mask_rule == BASELINE_RULE_NATIVE
        else topk_baseline_acc
    )
    active_edge_rule = (
        NATIVE_ACTIVE_EDGE_RULE
        if args.baseline_mask_rule == BASELINE_RULE_NATIVE
        else TOPK_ACTIVE_EDGE_RULE
    )
    num_active = int(active_mask.sum().item())
    override_info = verify_mask_override_effect(model, active_mask)

    expected_in, expected_out = expected_degrees(topology.edge_prob, valid_mask)
    hub_in_neurons = select_top_neurons_by_score(expected_in, args.top_k_neurons)
    hub_out_neurons = select_top_neurons_by_score(expected_out, args.top_k_neurons)
    run_name = args.checkpoint.parent.parent.name if args.checkpoint.parent.name == "checkpoints" else args.checkpoint.stem

    rows: list[dict[str, Any]] = [
        result_row(
            intervention="baseline",
            repeat_id="",
            baseline_acc=baseline_acc,
            intervention_acc=baseline_acc,
            removed_edges=0,
            control_type="baseline",
            seed=args.seed,
            split=args.split,
            checkpoint=args.checkpoint,
        )
    ]

    top_edges = top_probability_active_edges(active_mask, topology.edge_prob, args.top_k_edges)
    rows.append(
        evaluate_edge_removal(
            model=model,
            batches=batches,
            device=device,
            active_mask=active_mask,
            edges=top_edges,
            baseline_acc=baseline_acc,
            tau=tau,
            intervention="top_prob_edges_remove",
            repeat_id="",
            target_neurons=(),
            control_type="target",
            seed=args.seed,
            split=args.split,
            checkpoint=args.checkpoint,
        )
    )
    top_edge_count = int(rows[-1]["removed_edges"])
    for repeat in range(args.num_random_repeats):
        seed = stable_seed(args.seed, "random_edges_same_as_top_prob", repeat)
        rows.append(
            evaluate_edge_removal(
                model=model,
                batches=batches,
                device=device,
                active_mask=active_mask,
                edges=sample_random_active_edges(active_mask, top_edge_count, seed),
                baseline_acc=baseline_acc,
                tau=tau,
                intervention="random_edges_same_as_top_prob",
                repeat_id=repeat,
                target_neurons=(),
                control_type="random_same_count",
                seed=seed,
                split=args.split,
                checkpoint=args.checkpoint,
            )
        )

    hub_in_edges = hub_edges(active_mask, hub_in_neurons, "incoming")
    rows.append(
        evaluate_edge_removal(
            model=model,
            batches=batches,
            device=device,
            active_mask=active_mask,
            edges=hub_in_edges,
            baseline_acc=baseline_acc,
            tau=tau,
            intervention="hub_incoming_remove",
            repeat_id="",
            target_neurons=hub_in_neurons,
            control_type="target",
            seed=args.seed,
            split=args.split,
            checkpoint=args.checkpoint,
        )
    )
    hub_in_count = int(rows[-1]["removed_edges"])
    for repeat in range(args.num_random_repeats):
        seed = stable_seed(args.seed, "random_edges_same_count_as_hub_incoming", repeat)
        rows.append(
            evaluate_edge_removal(
                model=model,
                batches=batches,
                device=device,
                active_mask=active_mask,
                edges=sample_random_active_edges(active_mask, hub_in_count, seed),
                baseline_acc=baseline_acc,
                tau=tau,
                intervention="random_edges_same_count_as_hub_incoming",
                repeat_id=repeat,
                target_neurons=(),
                control_type="random_same_count",
                seed=seed,
                split=args.split,
                checkpoint=args.checkpoint,
            )
        )

    hub_out_edges = hub_edges(active_mask, hub_out_neurons, "outgoing")
    rows.append(
        evaluate_edge_removal(
            model=model,
            batches=batches,
            device=device,
            active_mask=active_mask,
            edges=hub_out_edges,
            baseline_acc=baseline_acc,
            tau=tau,
            intervention="hub_outgoing_remove",
            repeat_id="",
            target_neurons=hub_out_neurons,
            control_type="target",
            seed=args.seed,
            split=args.split,
            checkpoint=args.checkpoint,
        )
    )
    hub_out_count = int(rows[-1]["removed_edges"])
    for repeat in range(args.num_random_repeats):
        seed = stable_seed(args.seed, "random_edges_same_count_as_hub_outgoing", repeat)
        rows.append(
            evaluate_edge_removal(
                model=model,
                batches=batches,
                device=device,
                active_mask=active_mask,
                edges=sample_random_active_edges(active_mask, hub_out_count, seed),
                baseline_acc=baseline_acc,
                tau=tau,
                intervention="random_edges_same_count_as_hub_outgoing",
                repeat_id=repeat,
                target_neurons=(),
                control_type="random_same_count",
                seed=seed,
                split=args.split,
                checkpoint=args.checkpoint,
            )
        )

    ei_types = ei_types_from_model(model)
    if ei_types is None:
        rows.append(
            unavailable_row(
                intervention="ei_matched_outgoing_control",
                baseline_acc=baseline_acc,
                reason="missing_ei_metadata",
                seed=args.seed,
                split=args.split,
                checkpoint=args.checkpoint,
                control_type="ei_matched",
            )
        )
    else:
        for repeat in range(args.num_random_repeats):
            seed = stable_seed(args.seed, "ei_matched_outgoing_control", repeat)
            controls, reason = sample_ei_matched_neurons(hub_out_neurons, ei_types, seed)
            if controls is None:
                rows.append(
                    unavailable_row(
                        intervention="ei_matched_outgoing_control",
                        baseline_acc=baseline_acc,
                        reason=reason,
                        seed=seed,
                        split=args.split,
                        checkpoint=args.checkpoint,
                        control_type="ei_matched",
                    )
                )
                continue
            rows.append(
                evaluate_edge_removal(
                    model=model,
                    batches=batches,
                    device=device,
                    active_mask=active_mask,
                    edges=hub_edges(active_mask, controls, "outgoing"),
                    baseline_acc=baseline_acc,
                    tau=tau,
                    intervention="ei_matched_outgoing_control",
                    repeat_id=repeat,
                    target_neurons=controls,
                    control_type="ei_matched",
                    seed=seed,
                    split=args.split,
                    checkpoint=args.checkpoint,
                )
            )

    for repeat in range(args.num_random_repeats):
        seed = stable_seed(args.seed, "degree_matched_outgoing_control", repeat)
        controls, reason, degree_info = sample_degree_matched_neurons(
            hub_out_neurons,
            expected_out,
            seed,
        )
        if controls is None:
            rows.append(
                unavailable_row(
                    intervention="degree_matched_outgoing_control",
                    baseline_acc=baseline_acc,
                    reason=reason,
                    seed=seed,
                    split=args.split,
                    checkpoint=args.checkpoint,
                    control_type="degree_matched",
                )
            )
            continue
        rows.append(
            evaluate_edge_removal(
                model=model,
                batches=batches,
                device=device,
                active_mask=active_mask,
                edges=hub_edges(active_mask, controls, "outgoing"),
                baseline_acc=baseline_acc,
                tau=tau,
                intervention="degree_matched_outgoing_control",
                repeat_id=repeat,
                target_neurons=controls,
                control_type="degree_matched",
                seed=seed,
                split=args.split,
                checkpoint=args.checkpoint,
                degree_info=degree_info,
            )
        )

    recurrent_targets, recurrent_reason = load_recurrent_current_targets(
        args.diagnostic_dir,
        run_name,
        args.top_k_neurons,
    )
    if recurrent_targets is None:
        rows.append(
            unavailable_row(
                intervention="recurrent_current_top_neuron_outgoing_remove",
                baseline_acc=baseline_acc,
                reason=recurrent_reason,
                seed=args.seed,
                split=args.split,
                checkpoint=args.checkpoint,
                control_type="target",
            )
        )
    else:
        rows.append(
            evaluate_edge_removal(
                model=model,
                batches=batches,
                device=device,
                active_mask=active_mask,
                edges=hub_edges(active_mask, recurrent_targets, "outgoing"),
                baseline_acc=baseline_acc,
                tau=tau,
                intervention="recurrent_current_top_neuron_outgoing_remove",
                repeat_id="",
                target_neurons=recurrent_targets,
                control_type="target",
                seed=args.seed,
                split=args.split,
                checkpoint=args.checkpoint,
            )
        )

    summaries = summarize_results(rows)
    warnings = [*load_warnings, *topology.warnings]
    if not baseline_matches_original_eval:
        warnings.append(
            "native_active_baseline_acc differs from original_model_eval_acc "
            f"by {baseline_diff:.8f}; verify the diagnostic forward path before "
            "interpreting native-mask deltas."
        )
    summary = {
        "original_model_eval_acc": original_acc,
        "native_active_baseline_acc": native_baseline_acc,
        "topk_active_baseline_acc": topk_baseline_acc,
        "baseline_acc": baseline_acc,
        "baseline_mask_rule": args.baseline_mask_rule,
        "baseline_matches_original_eval": baseline_matches_original_eval,
        "baseline_original_eval_abs_diff": baseline_diff,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "num_random_repeats": int(args.num_random_repeats),
        "topology_probability_source": topology.topology_probability_source,
        "theta_value_min": topology.theta_min,
        "theta_value_max": topology.theta_max,
        "active_edge_rule": active_edge_rule,
        "num_active_edges": int(num_active),
        "num_native_active_edges": native_active_count,
        "num_topk_active_edges": topk_active_count,
        "num_valid_nonself_edges": int(valid_mask.sum().item()),
        "native_active_density": native_density,
        "topk_active_density": topk_density,
        "self_connections_excluded": True,
        "mask_override_verification": override_info,
        "warnings": warnings,
        "interventions": summaries,
    }

    write_csv(output_dir / "edge_control_results.csv", rows)
    (output_dir / "edge_control_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True)
    )
    unavailable = [row for row in rows if not row.get("available")]
    write_report(
        output_dir / "edge_control_report.md",
        config=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        seed=args.seed,
        original_model_eval_acc=original_acc,
        native_active_baseline_acc=native_baseline_acc,
        topk_active_baseline_acc=topk_baseline_acc,
        baseline_mask_rule=args.baseline_mask_rule,
        native_active_density=native_density,
        topk_active_density=topk_density,
        baseline_matches_original_eval=baseline_matches_original_eval,
        active_edge_rule=active_edge_rule,
        topology_probability_source=topology.topology_probability_source,
        summaries=summaries,
        unavailable_rows=unavailable,
    )
    plot_written = maybe_write_plot(output_dir / "edge_control_delta_plot.png", summaries)
    return {
        "output_dir": output_dir,
        "rows": rows,
        "summary": summary,
        "plot_written": plot_written,
    }


def main() -> None:
    result = build_and_run(parse_args())
    for warning in result["summary"].get("warnings", []):
        print(f"[WARN] {warning}", file=sys.stderr)
    print(f"[OK] wrote edge-control interventions to {result['output_dir']}")
    print("[OK] edge_control_results.csv")
    print("[OK] edge_control_summary.json")
    print("[OK] edge_control_report.md")
    if result["plot_written"]:
        print("[OK] edge_control_delta_plot.png")


if __name__ == "__main__":
    main()
