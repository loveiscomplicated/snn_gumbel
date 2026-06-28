"""Checkpoint-level intervention sensitivity diagnostics for learned lowrank LSMs.

This module is read-only with respect to experiment artifacts.  It evaluates a
fixed ``best.pt`` checkpoint on a fixed subset of diagnostic batches, then
re-evaluates temporary forward-time interventions against that same subset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.lowrank_runaway import (
    collect_diagnostic_batches,
    discover_run_artifacts,
    finite_values,
    graph_node_metrics,
    load_model_and_checkpoint,
    materialize_lowrank_topology,
    read_train_jsonl,
    safe_float,
    write_csv,
    _select_device,
)
from src.models.layers import spike_fn
from src.utils.config import load_config


EPS = 1e-12
ACTIVE_THRESHOLD = 0.05
OVERACTIVE_THRESHOLD = 0.20

NEURON_SELECTION_COLUMNS = {
    "top_recurrent_current_abs_mean": "recurrent_current_abs_mean",
    "top_rec_input_abs_ratio": "rec_input_abs_ratio",
    "top_expected_degree_score": "expected_degree_score",
    "top_in_degree": "in_degree",
    "top_out_degree": "out_degree",
    "top_firing_rate": "firing_rate",
    "top_readout_total_weight_norm": "readout_total_weight_norm",
    "top_readout_spike_weight_norm": "readout_spike_weight_norm",
    "top_readout_adapt_weight_norm": "readout_adapt_weight_norm",
    "top_adapt_readout_contribution_proxy": "adapt_readout_contribution_proxy",
}

METRIC_KEYS = [
    "accuracy",
    "loss",
    "mean_logit_margin",
    "correct_margin",
    "predicted_class_entropy",
    "logit_norm",
    "mean_spike_feature_norm",
    "mean_adaptation_feature_norm",
    "mean_firing_rate",
    "max_firing_rate",
    "mean_communication_firing_rate",
    "active_neuron_count",
    "overactive_neuron_count",
    "recurrent_current_abs_mean",
    "input_current_abs_mean",
    "rec_input_abs_ratio",
    "adaptation_mean",
    "adaptation_max",
]

ACTIVITY_KEYS = [
    "mean_firing_rate",
    "max_firing_rate",
    "mean_communication_firing_rate",
    "active_neuron_count",
    "overactive_neuron_count",
    "recurrent_current_abs_mean",
    "input_current_abs_mean",
    "rec_input_abs_ratio",
    "adaptation_mean",
    "adaptation_max",
]


@dataclass
class InterventionOptions:
    run_dirs: list[Path]
    diagnostic_dir: Path
    output_dir: Path
    num_batches: int = 4
    batch_size: int = 64
    top_k: list[int] | tuple[int, ...] = (50,)
    top_frac: list[float] | tuple[float, ...] = ()
    random_repeats: int = 10
    device: str = "auto"
    intervention_set: str = "core"
    neuron_intervention_mode: str = "communication_knockout"
    degree_bin_controls: bool = False
    seed: int = 0


@dataclass
class InterventionSpec:
    intervention_id: str
    family: str
    intervention_type: str
    selection_name: str = ""
    selected_neurons: tuple[int, ...] = ()
    selected_edges: tuple[tuple[int, int], ...] = ()
    mask_override: torch.Tensor | None = None
    neuron_mode: str = "communication_knockout"
    readout_mask_spike: bool = False
    readout_mask_adapt: bool = False
    adaptation_mode: str = ""
    top_k: int | None = None
    top_frac: float | None = None
    repeat_index: int | None = None
    random_control_matching: str = ""
    control_for: str = ""
    seed: int | None = None
    evidence_status: str = "available"
    insufficient_reason: str = ""


def stable_seed(base_seed: int, *parts: Any) -> int:
    text = "|".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_diagnostic_tables(diagnostic_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {
        "neuron_table": read_csv_rows(diagnostic_dir / "neuron_table.csv"),
        "lowrank_role_summary": read_csv_rows(diagnostic_dir / "lowrank_role_summary.csv"),
        "topk_overlap_summary": read_csv_rows(diagnostic_dir / "topk_overlap_summary.csv"),
        "correlation_summary": read_csv_rows(diagnostic_dir / "correlation_summary.csv"),
    }


def rows_for_run(
    rows: list[dict[str, Any]],
    run_name: str,
    run_dir: Path | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        row_run = str(row.get("run_name", ""))
        row_dir = str(row.get("experiment_dir", ""))
        if row_run == run_name:
            out.append(row)
        elif run_dir is not None and row_dir and Path(row_dir).name == run_dir.name:
            out.append(row)
    out.sort(key=lambda row: int(safe_float(row.get("neuron_id"))))
    return out


def _neuron_id(row: dict[str, Any]) -> int:
    return int(safe_float(row.get("neuron_id")))


def _selection_count(n_items: int, *, top_k: int | None, top_frac: float | None) -> int:
    if top_frac is not None:
        return max(1, min(n_items, int(math.ceil(n_items * float(top_frac)))))
    return max(0, min(n_items, int(top_k or 0)))


def select_top_neurons(
    neuron_rows: list[dict[str, Any]],
    selection_name: str,
    *,
    top_k: int | None = None,
    top_frac: float | None = None,
) -> tuple[list[int], str]:
    metric = NEURON_SELECTION_COLUMNS.get(selection_name)
    if metric is None:
        return [], f"unsupported neuron selection: {selection_name}"
    candidates: list[tuple[int, float]] = []
    for row in neuron_rows:
        value = safe_float(row.get(metric))
        if math.isfinite(value):
            candidates.append((_neuron_id(row), value))
    if not candidates:
        return [], f"missing or non-finite diagnostic column: {metric}"
    candidates.sort(key=lambda item: item[1], reverse=True)
    count = _selection_count(len(neuron_rows), top_k=top_k, top_frac=top_frac)
    return [idx for idx, _ in candidates[:count]], ""


def sample_random_neurons(
    neuron_rows: list[dict[str, Any]],
    k: int,
    seed: int,
) -> list[int]:
    ids = [_neuron_id(row) for row in neuron_rows]
    rng = random.Random(int(seed))
    if k >= len(ids):
        return sorted(ids)
    return sorted(rng.sample(ids, k))


def sample_ei_matched_random_neurons(
    neuron_rows: list[dict[str, Any]],
    selected_ids: Iterable[int],
    seed: int,
) -> tuple[list[int], str]:
    selected = set(int(idx) for idx in selected_ids)
    if not selected:
        return [], "same_k"
    row_by_id = {_neuron_id(row): row for row in neuron_rows}
    if any(not str(row_by_id.get(idx, {}).get("ei_type", "")).strip() for idx in selected):
        return sample_random_neurons(neuron_rows, len(selected), seed), "same_k"

    rng = random.Random(int(seed))
    by_type: dict[str, list[int]] = {}
    selected_by_type: dict[str, int] = {}
    for row in neuron_rows:
        ei_type = str(row.get("ei_type", "")).strip()
        if not ei_type:
            return sample_random_neurons(neuron_rows, len(selected), seed), "same_k"
        by_type.setdefault(ei_type, []).append(_neuron_id(row))
    for idx in selected:
        ei_type = str(row_by_id[idx].get("ei_type", "")).strip()
        selected_by_type[ei_type] = selected_by_type.get(ei_type, 0) + 1

    sampled: list[int] = []
    for ei_type, count in selected_by_type.items():
        pool = by_type.get(ei_type, [])
        if len(pool) < count:
            return sample_random_neurons(neuron_rows, len(selected), seed), "same_k"
        sampled.extend(rng.sample(pool, count))
    return sorted(sampled), "ei_matched"


def _degree_bins(neuron_rows: list[dict[str, Any]], n_bins: int = 4) -> dict[int, int] | None:
    pairs = [
        (_neuron_id(row), safe_float(row.get("total_degree")))
        for row in neuron_rows
        if math.isfinite(safe_float(row.get("total_degree")))
    ]
    if len(pairs) != len(neuron_rows) or not pairs:
        return None
    values = np.asarray([v for _, v in pairs], dtype=float)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return {idx: int(np.searchsorted(quantiles, val, side="right")) for idx, val in pairs}


def sample_degree_bin_matched_random_neurons(
    neuron_rows: list[dict[str, Any]],
    selected_ids: Iterable[int],
    seed: int,
) -> tuple[list[int], str]:
    selected = set(int(idx) for idx in selected_ids)
    bins = _degree_bins(neuron_rows)
    if bins is None or not selected:
        return sample_ei_matched_random_neurons(neuron_rows, selected, seed)
    rng = random.Random(int(seed))
    row_by_id = {_neuron_id(row): row for row in neuron_rows}
    pools: dict[tuple[str, int], list[int]] = {}
    wanted: dict[tuple[str, int], int] = {}
    for row in neuron_rows:
        idx = _neuron_id(row)
        key = (str(row.get("ei_type", "")).strip(), bins[idx])
        pools.setdefault(key, []).append(idx)
    for idx in selected:
        if idx not in row_by_id:
            return sample_ei_matched_random_neurons(neuron_rows, selected, seed)
        key = (str(row_by_id[idx].get("ei_type", "")).strip(), bins[idx])
        wanted[key] = wanted.get(key, 0) + 1
    sampled: list[int] = []
    for key, count in wanted.items():
        pool = pools.get(key, [])
        if len(pool) < count:
            return sample_ei_matched_random_neurons(neuron_rows, selected, seed)
        sampled.extend(rng.sample(pool, count))
    return sorted(sampled), "degree_bin_matched"


def shuffle_adaptation_features(
    adaptation: torch.Tensor,
    mode: str,
    *,
    selected: Iterable[int] = (),
    seed: int = 0,
) -> torch.Tensor:
    out = adaptation.clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if mode == "adaptation_all_zero":
        out.zero_()
    elif mode == "adaptation_selected_zero":
        ids = [int(idx) for idx in selected]
        if ids:
            out[:, ids] = 0.0
    elif mode == "adaptation_neuron_shuffle":
        perm = torch.randperm(out.shape[1], generator=generator, device="cpu").to(out.device)
        out = out[:, perm]
    elif mode == "adaptation_batch_shuffle":
        perm = torch.randperm(out.shape[0], generator=generator, device="cpu").to(out.device)
        out = out[perm]
    else:
        raise ValueError(f"unknown adaptation shuffle mode: {mode}")
    return out


def _valid_edge_mask(mask: torch.Tensor) -> torch.Tensor:
    valid = torch.ones_like(mask, dtype=torch.bool)
    if mask.dim() == 2 and mask.shape[0] == mask.shape[1]:
        valid.fill_(True)
        valid.fill_diagonal_(False)
    return valid


def density_preserving_random_shuffle(mask: torch.Tensor, seed: int) -> torch.Tensor:
    mask = mask.detach().cpu().bool()
    valid = _valid_edge_mask(mask)
    valid_pos = valid.nonzero(as_tuple=False)
    edge_count = int((mask & valid).sum().item())
    out = torch.zeros_like(mask, dtype=torch.bool)
    if edge_count <= 0 or valid_pos.numel() == 0:
        return out
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    choice = torch.randperm(valid_pos.shape[0], generator=generator)[:edge_count]
    selected = valid_pos[choice]
    out[selected[:, 0], selected[:, 1]] = True
    return out


def degree_preserving_directed_swap(
    mask: torch.Tensor,
    seed: int,
    *,
    max_attempts: int | None = None,
) -> tuple[torch.Tensor, str, str]:
    base = mask.detach().cpu().bool().clone()
    valid = _valid_edge_mask(base)
    edges = {(int(i), int(j)) for i, j in (base & valid).nonzero(as_tuple=False).tolist()}
    if len(edges) < 2:
        return base, "insufficient_evidence", "fewer than two active directed edges"
    rng = random.Random(int(seed))
    attempts = max_attempts or max(1000, len(edges) * 20)
    target_successes = max(10, min(len(edges), len(edges) // 2))
    successes = 0
    edge_list = list(edges)
    for _ in range(attempts):
        (a, b), (c, d) = rng.sample(edge_list, 2)
        if a == c or b == d or a == d or c == b:
            continue
        e1 = (a, d)
        e2 = (c, b)
        if e1 in edges or e2 in edges:
            continue
        edges.remove((a, b))
        edges.remove((c, d))
        edges.add(e1)
        edges.add(e2)
        edge_list = list(edges)
        successes += 1
        if successes >= target_successes:
            break
    out = torch.zeros_like(base, dtype=torch.bool)
    for i, j in edges:
        out[i, j] = True
    if successes == 0:
        fallback = density_preserving_random_shuffle(base, seed)
        return fallback, "fallback_density_preserving_random_shuffle", "directed double-edge swap made zero swaps"
    if successes < max(1, target_successes // 4):
        return out, "partial_degree_preserving_directed_swap", f"only {successes}/{target_successes} swaps succeeded"
    return out, "available", f"{successes} directed double-edge swaps"


@contextmanager
def temporary_recurrent_mask(model: torch.nn.Module, mask: torch.Tensor | None = None):
    old_mask = getattr(model.liquid, "current_mask", None)
    old_mask_clone = old_mask.detach().clone() if isinstance(old_mask, torch.Tensor) else old_mask
    try:
        if mask is not None:
            model.liquid.current_mask = mask.to(
                device=next(model.parameters()).device,
                dtype=torch.float32,
            )
        yield
    finally:
        model.liquid.current_mask = old_mask_clone


def _mask_feature_block(features: torch.Tensor, selected: Iterable[int]) -> torch.Tensor:
    ids = [int(idx) for idx in selected]
    if not ids:
        return features
    out = features.clone()
    out[:, ids] = 0.0
    return out


def diagnostic_forward(
    model: torch.nn.Module,
    spikes: torch.Tensor,
    spec: InterventionSpec | None = None,
    *,
    tau: float = 1.0,
    batch_index: int = 0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    spec = spec or InterventionSpec(
        intervention_id="original_subset",
        family="baseline",
        intervention_type="original_subset",
    )
    batch_size = spikes.shape[0]
    device = spikes.device
    selected = torch.as_tensor(spec.selected_neurons, dtype=torch.long, device=device)
    has_selected = selected.numel() > 0

    with temporary_recurrent_mask(model):
        if spec.mask_override is None:
            model.liquid.sample_mask(tau=tau)
        else:
            model.liquid.current_mask = spec.mask_override.to(device=device, dtype=torch.float32)

        liquid_mem = torch.zeros(batch_size, model.n_liquid, device=device)
        state_spike = torch.zeros(batch_size, model.n_liquid, device=device)
        communication_spike = torch.zeros(batch_size, model.n_liquid, device=device)
        liquid_a = torch.zeros(batch_size, model.n_liquid, device=device) if model.neuron_type == "alif" else None

        readout_mem = torch.zeros(batch_size, model.n_output, device=device)
        motor_mem = None
        motor_spike_count = None
        motor_membrane_sum = None
        if getattr(model, "is_motor_readout", False):
            motor_mem = torch.zeros(batch_size, model.n_output, device=device)
            motor_spike_count = torch.zeros(batch_size, model.n_output, device=device)
            motor_membrane_sum = torch.zeros(batch_size, model.n_output, device=device)

        spike_sum = torch.zeros(batch_size, model.n_liquid, device=device)
        raw_spike_sum = torch.zeros(batch_size, model.n_liquid, device=device)
        communication_spike_sum = torch.zeros(batch_size, model.n_liquid, device=device)
        membrane_sum = (
            torch.zeros(batch_size, model.n_liquid, device=device)
            if model.readout_mode == "membrane_trace"
            else None
        )
        adaptation_sum = (
            torch.zeros(batch_size, model.n_liquid, device=device)
            if model.readout_mode == "spike_adaptation_concat"
            else None
        )

        trace_raw_spikes: list[torch.Tensor] = []
        trace_comm_spikes: list[torch.Tensor] = []
        trace_input_current: list[torch.Tensor] = []
        trace_recurrent_current: list[torch.Tensor] = []
        trace_adaptation: list[torch.Tensor] = []
        neuron_intervention = spec.family == "neuron"

        grad_start = (model.T - model.bptt_truncate) if getattr(model, "bptt_truncate", 0) > 0 else 0
        for t in range(model.T):
            if t == grad_start and t > 0:
                liquid_mem = liquid_mem.detach()
                state_spike = state_spike.detach()
                communication_spike = communication_spike.detach()
                if liquid_a is not None:
                    liquid_a = liquid_a.detach()
                if motor_mem is not None:
                    motor_mem = motor_mem.detach()

            input_current = model.input_proj(spikes[:, t])
            recurrent_current = model.liquid(communication_spike)
            trace_input_current.append(input_current.detach())
            trace_recurrent_current.append(recurrent_current.detach())

            liquid_mem = model.liquid.beta * liquid_mem + input_current + recurrent_current
            liquid_mem = torch.clamp(liquid_mem, -3.0, 3.0)
            if membrane_sum is not None:
                membrane_sum = membrane_sum + liquid_mem

            if model.neuron_type == "alif":
                assert liquid_a is not None
                liquid_a = model.liquid.alif_rho * liquid_a + model.liquid.alif_adapt_increment * state_spike
                if adaptation_sum is not None:
                    adaptation_sum = adaptation_sum + liquid_a
                theta_eff = model.liquid.threshold + model.liquid.alif_beta * liquid_a
                raw_spike = spike_fn(liquid_mem - theta_eff.clamp(min=0.01))
                trace_adaptation.append(liquid_a.detach())
            else:
                raw_spike = spike_fn(liquid_mem - model.liquid.threshold.clamp(min=0.01))

            if neuron_intervention and spec.neuron_mode == "full_neuron_silence" and has_selected:
                raw_spike = raw_spike.clone()
                raw_spike[:, selected] = 0.0

            state_spike = raw_spike
            communication_spike = raw_spike
            if neuron_intervention and spec.neuron_mode == "communication_knockout" and has_selected:
                communication_spike = communication_spike.clone()
                communication_spike[:, selected] = 0.0

            liquid_mem = liquid_mem * (1.0 - state_spike)
            raw_spike_sum = raw_spike_sum + state_spike
            communication_spike_sum = communication_spike_sum + communication_spike
            spike_sum = spike_sum + communication_spike
            trace_raw_spikes.append(state_spike.detach())
            trace_comm_spikes.append(communication_spike.detach())

            if model.readout_mode == "spike_count":
                pass
            elif getattr(model, "is_non_spiking_lif_readout", False):
                readout_mem = model.readout.step(communication_spike, readout_mem)
            elif getattr(model, "is_motor_readout", False):
                assert motor_mem is not None
                assert motor_spike_count is not None
                assert motor_membrane_sum is not None
                motor_current = model.readout(communication_spike)
                motor_mem = model.motor_beta * motor_mem + motor_current
                motor_mem = torch.clamp(motor_mem, -model.motor_mem_clamp, model.motor_mem_clamp)
                motor_mem_pre_spike = motor_mem
                motor_membrane_sum = motor_membrane_sum + motor_mem_pre_spike
                motor_spike = spike_fn(motor_mem - model.motor_threshold)
                motor_mem = motor_mem * (1.0 - motor_spike)
                motor_spike_count = motor_spike_count + motor_spike

        spike_feature = spike_sum / model.T
        raw_spike_feature = raw_spike_sum / model.T
        communication_spike_feature = communication_spike_sum / model.T
        adaptation_feature = torch.empty(batch_size, 0, device=device)
        readout_input = spike_feature

        if model.readout_mode == "spike_count":
            readout_input = spike_feature
            if spec.readout_mask_spike and has_selected:
                readout_input = _mask_feature_block(readout_input, spec.selected_neurons)
            logits = model.readout(readout_input)
        elif model.readout_mode == "membrane_trace":
            assert membrane_sum is not None
            readout_input = membrane_sum / model.T
            if spec.readout_mask_spike and has_selected:
                readout_input = _mask_feature_block(readout_input, spec.selected_neurons)
            logits = model.readout(readout_input)
        elif model.readout_mode == "spike_adaptation_concat":
            assert adaptation_sum is not None
            spike_block = spike_feature
            adaptation_feature = adaptation_sum / model.T
            if spec.adaptation_mode:
                adaptation_feature = shuffle_adaptation_features(
                    adaptation_feature,
                    spec.adaptation_mode,
                    selected=spec.selected_neurons,
                    seed=stable_seed(spec.seed or 0, spec.intervention_id, batch_index),
                )
            if spec.readout_mask_spike and has_selected:
                spike_block = _mask_feature_block(spike_block, spec.selected_neurons)
            if spec.readout_mask_adapt and has_selected:
                adaptation_feature = _mask_feature_block(adaptation_feature, spec.selected_neurons)
            if neuron_intervention and has_selected:
                adaptation_feature = _mask_feature_block(adaptation_feature, spec.selected_neurons)
            readout_input = torch.cat([spike_block, adaptation_feature], dim=1)
            logits = model.readout(readout_input)
        elif model.readout_mode == "motor_lif":
            assert motor_spike_count is not None
            logits = motor_spike_count * model.motor_logit_scale
            if model.motor_output_bias is not None:
                logits = logits + model.motor_output_bias
        elif model.readout_mode == "motor_lif_count_membrane":
            assert motor_spike_count is not None
            assert motor_membrane_sum is not None
            motor_membrane_trace = motor_membrane_sum / model.T
            logits = (
                motor_spike_count * model.motor_logit_scale
                + motor_membrane_trace * model.motor_membrane_logit_scale
            )
            if model.motor_output_bias is not None:
                logits = logits + model.motor_output_bias
        elif getattr(model, "is_non_spiking_lif_readout", False):
            logits = model.readout.finalize(readout_mem, model.T)
        else:
            logits = model.readout(spike_feature)

        traces = {
            "raw_spikes": torch.stack(trace_raw_spikes, dim=1),
            "communication_spikes": torch.stack(trace_comm_spikes, dim=1),
            "spikes": torch.stack(trace_raw_spikes, dim=1),
            "input_current": torch.stack(trace_input_current, dim=1),
            "recurrent_current": torch.stack(trace_recurrent_current, dim=1),
        }
        if trace_adaptation:
            traces["adaptation"] = torch.stack(trace_adaptation, dim=1)
        features = {
            "spike_feature": spike_feature.detach(),
            "raw_spike_feature": raw_spike_feature.detach(),
            "communication_spike_feature": communication_spike_feature.detach(),
            "adaptation_feature": adaptation_feature.detach(),
            "readout_input": readout_input.detach(),
        }
        return logits, traces, features


class EvaluationAccumulator:
    def __init__(self, n_classes: int, n_liquid: int, has_adaptation: bool):
        self.n_classes = n_classes
        self.n_liquid = n_liquid
        self.has_adaptation = has_adaptation
        self.total = 0
        self.correct = 0
        self.loss_sum = 0.0
        self.margin_sum = 0.0
        self.correct_margin_sum = 0.0
        self.logit_norm_sum = 0.0
        self.spike_feature_norm_sum = 0.0
        self.adaptation_feature_norm_sum = 0.0
        self.adaptation_feature_count = 0
        self.pred_counts = torch.zeros(n_classes)
        self.class_total = torch.zeros(n_classes)
        self.class_correct = torch.zeros(n_classes)
        self.raw_spike_sum = torch.zeros(n_liquid)
        self.comm_spike_sum = torch.zeros(n_liquid)
        self.spike_time_count = 0
        self.input_abs_sum = 0.0
        self.recurrent_abs_sum = 0.0
        self.current_count = 0
        self.adaptation_sum = 0.0
        self.adaptation_max = float("nan")

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        traces: dict[str, torch.Tensor],
        features: dict[str, torch.Tensor],
    ) -> None:
        batch = labels.numel()
        loss = F.cross_entropy(logits, labels, reduction="sum")
        pred = logits.argmax(dim=1)
        self.total += batch
        self.correct += int((pred == labels).sum().item())
        self.loss_sum += safe_float(loss)

        top2 = torch.topk(logits, k=min(2, logits.shape[1]), dim=1).values
        if top2.shape[1] == 1:
            margin = top2[:, 0]
        else:
            margin = top2[:, 0] - top2[:, 1]
        true_logits = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        other_logits = logits.masked_fill(
            F.one_hot(labels, num_classes=logits.shape[1]).bool(),
            float("-inf"),
        ).max(dim=1).values
        self.margin_sum += safe_float(margin.sum())
        self.correct_margin_sum += safe_float((true_logits - other_logits).sum())
        self.logit_norm_sum += safe_float(logits.norm(dim=1).sum())
        self.spike_feature_norm_sum += safe_float(features["spike_feature"].norm(dim=1).sum())
        adapt_feature = features.get("adaptation_feature")
        if adapt_feature is not None and adapt_feature.numel() > 0:
            self.adaptation_feature_norm_sum += safe_float(adapt_feature.norm(dim=1).sum())
            self.adaptation_feature_count += adapt_feature.shape[0]

        for cls in range(self.n_classes):
            mask = labels.detach().cpu() == cls
            self.class_total[cls] += int(mask.sum().item())
            self.class_correct[cls] += int(((pred.detach().cpu() == cls) & mask).sum().item())
            self.pred_counts[cls] += int((pred.detach().cpu() == cls).sum().item())

        raw_spikes = traces["raw_spikes"].detach().float().cpu()
        comm_spikes = traces["communication_spikes"].detach().float().cpu()
        self.raw_spike_sum += raw_spikes.sum(dim=(0, 1))
        self.comm_spike_sum += comm_spikes.sum(dim=(0, 1))
        self.spike_time_count += raw_spikes.shape[0] * raw_spikes.shape[1]
        input_current = traces["input_current"].detach().float().cpu()
        recurrent_current = traces["recurrent_current"].detach().float().cpu()
        self.input_abs_sum += safe_float(input_current.abs().sum())
        self.recurrent_abs_sum += safe_float(recurrent_current.abs().sum())
        self.current_count += input_current.numel()
        if self.has_adaptation and "adaptation" in traces:
            adaptation = traces["adaptation"].detach().float().cpu()
            self.adaptation_sum += safe_float(adaptation.sum())
            batch_max = safe_float(adaptation.max()) if adaptation.numel() else float("nan")
            if math.isfinite(batch_max):
                self.adaptation_max = batch_max if not math.isfinite(self.adaptation_max) else max(self.adaptation_max, batch_max)

    def finalize(self) -> tuple[dict[str, float], list[dict[str, Any]]]:
        denom = max(self.total, 1)
        temporal_count = max(self.current_count, 1)
        per_neuron_firing = self.raw_spike_sum / max(self.spike_time_count, 1)
        per_neuron_comm = self.comm_spike_sum / max(self.spike_time_count, 1)
        pred_probs = self.pred_counts / self.pred_counts.sum().clamp(min=1.0)
        pred_entropy = -safe_float((pred_probs * (pred_probs + EPS).log()).sum())
        metrics = {
            "accuracy": self.correct / denom,
            "loss": self.loss_sum / denom,
            "mean_logit_margin": self.margin_sum / denom,
            "correct_margin": self.correct_margin_sum / denom,
            "predicted_class_entropy": pred_entropy,
            "logit_norm": self.logit_norm_sum / denom,
            "mean_spike_feature_norm": self.spike_feature_norm_sum / denom,
            "mean_adaptation_feature_norm": (
                self.adaptation_feature_norm_sum / max(self.adaptation_feature_count, 1)
                if self.adaptation_feature_count
                else float("nan")
            ),
            "mean_firing_rate": safe_float(per_neuron_firing.mean()),
            "max_firing_rate": safe_float(per_neuron_firing.max()) if per_neuron_firing.numel() else float("nan"),
            "mean_communication_firing_rate": safe_float(per_neuron_comm.mean()),
            "active_neuron_count": int((per_neuron_firing > ACTIVE_THRESHOLD).sum().item()),
            "overactive_neuron_count": int((per_neuron_firing > OVERACTIVE_THRESHOLD).sum().item()),
            "recurrent_current_abs_mean": self.recurrent_abs_sum / temporal_count,
            "input_current_abs_mean": self.input_abs_sum / temporal_count,
            "rec_input_abs_ratio": self.recurrent_abs_sum / max(self.input_abs_sum, EPS),
            "adaptation_mean": self.adaptation_sum / temporal_count if self.has_adaptation else float("nan"),
            "adaptation_max": self.adaptation_max if self.has_adaptation else float("nan"),
        }
        classwise = []
        for cls in range(self.n_classes):
            total = int(self.class_total[cls].item())
            correct = int(self.class_correct[cls].item())
            classwise.append(
                {
                    "class_id": cls,
                    "class_total": total,
                    "class_correct": correct,
                    "class_accuracy": correct / total if total else float("nan"),
                }
            )
        return metrics, classwise


def evaluate_on_batches(
    model: torch.nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    spec: InterventionSpec,
    device: torch.device,
    *,
    tau: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    has_adaptation = str(getattr(model, "neuron_type", "")).lower() == "alif"
    accumulator = EvaluationAccumulator(
        n_classes=int(model.n_output),
        n_liquid=int(model.n_liquid),
        has_adaptation=has_adaptation,
    )
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(batches):
            x = x.to(device)
            y = y.to(device)
            logits, traces, features = diagnostic_forward(
                model,
                x,
                spec,
                tau=tau,
                batch_index=batch_idx,
            )
            accumulator.update(logits, y, traces, features)
    return accumulator.finalize()


def compute_metric_deltas(
    intervention_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    metric_keys: Iterable[str] = METRIC_KEYS,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in metric_keys:
        value = safe_float(intervention_metrics.get(key))
        baseline = safe_float(baseline_metrics.get(key))
        out[f"delta_{key}"] = value - baseline if math.isfinite(value) and math.isfinite(baseline) else float("nan")
    return out


def materialize_fixed_batches(cfg: Any, batch_size: int, num_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = collect_diagnostic_batches(cfg, batch_size=batch_size, num_batches=num_batches)
    return [(x.detach().cpu().clone(), y.detach().cpu().clone()) for x, y in batches]


def _base_metadata(
    run_name: str,
    run_dir: Path,
    cfg: Any | None,
    checkpoint: dict[str, Any],
    spec: InterventionSpec,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "experiment_dir": str(run_dir),
        "seed": getattr(cfg, "seed", None) if cfg is not None else None,
        "group_label": getattr(cfg, "experiment_name", run_name) if cfg is not None else run_name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_acc": checkpoint.get("best_val_acc"),
        "best_test_acc_at_best_val": checkpoint.get("best_test_acc_at_best_val"),
        "intervention_id": spec.intervention_id,
        "family": spec.family,
        "intervention_type": spec.intervention_type,
        "selection_name": spec.selection_name,
        "neuron_mode": spec.neuron_mode,
        "top_k": spec.top_k,
        "top_frac": spec.top_frac,
        "repeat_index": spec.repeat_index,
        "random_control_matching": spec.random_control_matching,
        "control_for": spec.control_for,
        "random_seed": spec.seed,
        "n_selected_neurons": len(spec.selected_neurons),
        "n_selected_edges": len(spec.selected_edges),
        "evidence_status": spec.evidence_status,
        "insufficient_reason": spec.insufficient_reason,
    }


def _metric_row(
    run_name: str,
    run_dir: Path,
    cfg: Any | None,
    checkpoint: dict[str, Any],
    spec: InterventionSpec,
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    row = _base_metadata(run_name, run_dir, cfg, checkpoint, spec)
    row.update(metrics)
    if baseline_metrics is not None:
        row.update(compute_metric_deltas(metrics, baseline_metrics))
    else:
        for key in METRIC_KEYS:
            row[f"delta_{key}"] = 0.0 if spec.intervention_type == "original_subset" else float("nan")
    return row


def _classwise_rows(
    run_name: str,
    spec: InterventionSpec,
    classwise: list[dict[str, Any]],
    baseline_classwise: dict[int, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows = []
    for row in classwise:
        cls = int(row["class_id"])
        out = {
            "run_name": run_name,
            "intervention_id": spec.intervention_id,
            "family": spec.family,
            "intervention_type": spec.intervention_type,
            "selection_name": spec.selection_name,
            **row,
        }
        if baseline_classwise is not None and cls in baseline_classwise:
            out["delta_class_accuracy"] = safe_float(row.get("class_accuracy")) - safe_float(
                baseline_classwise[cls].get("class_accuracy")
            )
        else:
            out["delta_class_accuracy"] = 0.0 if spec.intervention_type == "original_subset" else float("nan")
        rows.append(out)
    return rows


def _hard_mask_and_edge_prob(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    liquid = model.liquid
    if str(getattr(liquid, "mode", "")).lower() in {
        "learned_lowrank",
        "learned_lowrank_grad_r",
    }:
        mats = materialize_lowrank_topology(
            liquid.src_embed,
            liquid.dst_embed,
            liquid.theta_bias,
            getattr(liquid, "self_conn_mask", None),
        )
        return mats["hard_mask"].bool(), mats["edge_prob"].float()
    return liquid.get_binary_mask().detach().cpu().bool(), None


def _edge_records(
    run_name: str,
    spec: InterventionSpec,
    edge_prob: torch.Tensor | None,
    hard_mask: torch.Tensor | None,
) -> list[dict[str, Any]]:
    rows = []
    for rank, (src, dst) in enumerate(spec.selected_edges, start=1):
        rows.append(
            {
                "run_name": run_name,
                "intervention_id": spec.intervention_id,
                "family": spec.family,
                "intervention_type": spec.intervention_type,
                "selection_name": spec.selection_name,
                "edge_rank": rank,
                "src_neuron": src,
                "dst_neuron": dst,
                "edge_prob": safe_float(edge_prob[src, dst]) if edge_prob is not None else float("nan"),
                "original_active": bool(hard_mask[src, dst].item()) if hard_mask is not None else "",
                "evidence_status": spec.evidence_status,
                "insufficient_reason": spec.insufficient_reason,
            }
        )
    return rows


def _neuron_records(
    run_name: str,
    spec: InterventionSpec,
    neuron_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {_neuron_id(row): row for row in neuron_rows}
    rows = []
    degree_bins = _degree_bins(neuron_rows) or {}
    for rank, idx in enumerate(spec.selected_neurons, start=1):
        src = by_id.get(idx, {})
        rows.append(
            {
                "run_name": run_name,
                "intervention_id": spec.intervention_id,
                "family": spec.family,
                "intervention_type": spec.intervention_type,
                "selection_name": spec.selection_name,
                "selection_rank": rank,
                "neuron_id": idx,
                "ei_type": src.get("ei_type", ""),
                "degree_bin": degree_bins.get(idx, ""),
                "metric_value": safe_float(src.get(NEURON_SELECTION_COLUMNS.get(spec.selection_name, ""))),
                "random_control_matching": spec.random_control_matching,
                "control_for": spec.control_for,
                "evidence_status": spec.evidence_status,
                "insufficient_reason": spec.insufficient_reason,
            }
        )
    return rows


def _make_insufficient_spec(
    intervention_id: str,
    family: str,
    intervention_type: str,
    reason: str,
    *,
    selection_name: str = "",
    top_k: int | None = None,
    top_frac: float | None = None,
) -> InterventionSpec:
    return InterventionSpec(
        intervention_id=intervention_id,
        family=family,
        intervention_type=intervention_type,
        selection_name=selection_name,
        top_k=top_k,
        top_frac=top_frac,
        evidence_status="insufficient_evidence",
        insufficient_reason=reason,
    )


def _size_specs(options: InterventionOptions, n_items: int) -> list[tuple[int | None, float | None, str]]:
    specs = [(int(k), None, f"k{int(k)}") for k in options.top_k]
    specs.extend((None, float(frac), f"frac{float(frac):g}") for frac in options.top_frac)
    if not specs:
        specs = [(min(50, n_items), None, "k50")]
    return specs


def _neuron_modes(options: InterventionOptions) -> list[str]:
    if options.neuron_intervention_mode == "both":
        return ["communication_knockout", "full_neuron_silence"]
    return [options.neuron_intervention_mode]


def _add_random_controls(
    specs: list[InterventionSpec],
    target: InterventionSpec,
    neuron_rows: list[dict[str, Any]],
    options: InterventionOptions,
) -> None:
    if not target.selected_neurons:
        return
    for repeat in range(options.random_repeats):
        seed = stable_seed(options.seed, target.intervention_id, "same_k", repeat)
        selected = sample_random_neurons(neuron_rows, len(target.selected_neurons), seed)
        specs.append(
            InterventionSpec(
                intervention_id=f"{target.intervention_id}__random_same_k_r{repeat}",
                family=target.family,
                intervention_type=target.intervention_type,
                selection_name="random_neurons_matched_k",
                selected_neurons=tuple(selected),
                neuron_mode=target.neuron_mode,
                readout_mask_spike=target.readout_mask_spike,
                readout_mask_adapt=target.readout_mask_adapt,
                top_k=target.top_k,
                top_frac=target.top_frac,
                repeat_index=repeat,
                random_control_matching="same_k",
                control_for=target.intervention_id,
                seed=seed,
            )
        )
        seed = stable_seed(options.seed, target.intervention_id, "ei_or_degree", repeat)
        if options.degree_bin_controls:
            matched, matching = sample_degree_bin_matched_random_neurons(neuron_rows, target.selected_neurons, seed)
        else:
            matched, matching = sample_ei_matched_random_neurons(neuron_rows, target.selected_neurons, seed)
        if matching != "same_k":
            specs.append(
                InterventionSpec(
                    intervention_id=f"{target.intervention_id}__random_{matching}_r{repeat}",
                    family=target.family,
                    intervention_type=target.intervention_type,
                    selection_name="random_neurons_matched_k",
                    selected_neurons=tuple(matched),
                    neuron_mode=target.neuron_mode,
                    readout_mask_spike=target.readout_mask_spike,
                    readout_mask_adapt=target.readout_mask_adapt,
                    top_k=target.top_k,
                    top_frac=target.top_frac,
                    repeat_index=repeat,
                    random_control_matching=matching,
                    control_for=target.intervention_id,
                    seed=seed,
                )
            )


def _top_edges(
    hard_mask: torch.Tensor,
    edge_prob: torch.Tensor | None,
    *,
    top_k: int | None,
    top_frac: float | None,
) -> list[tuple[int, int]]:
    active = hard_mask.detach().cpu().bool() & _valid_edge_mask(hard_mask)
    positions = active.nonzero(as_tuple=False)
    if positions.numel() == 0:
        return []
    count = _selection_count(int(positions.shape[0]), top_k=top_k, top_frac=top_frac)
    if edge_prob is None:
        scores = torch.ones(positions.shape[0])
    else:
        scores = edge_prob.detach().cpu()[positions[:, 0], positions[:, 1]]
    order = torch.argsort(scores, descending=True)[:count]
    return [(int(positions[i, 0]), int(positions[i, 1])) for i in order.tolist()]


def _remove_edges_mask(hard_mask: torch.Tensor, edges: Iterable[tuple[int, int]]) -> torch.Tensor:
    out = hard_mask.detach().cpu().bool().clone()
    for src, dst in edges:
        out[int(src), int(dst)] = False
    return out


def _hub_edges(
    hard_mask: torch.Tensor,
    hubs: Iterable[int],
    direction: str,
) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    active = hard_mask.detach().cpu().bool()
    for hub in hubs:
        idx = int(hub)
        if direction == "outgoing":
            targets = active[idx].nonzero(as_tuple=False).reshape(-1).tolist()
            edges.extend((idx, int(dst)) for dst in targets)
        else:
            sources = active[:, idx].nonzero(as_tuple=False).reshape(-1).tolist()
            edges.extend((int(src), idx) for src in sources)
    return edges


def build_intervention_specs(
    run_name: str,
    neuron_rows: list[dict[str, Any]],
    hard_mask: torch.Tensor | None,
    edge_prob: torch.Tensor | None,
    options: InterventionOptions,
) -> list[InterventionSpec]:
    specs: list[InterventionSpec] = []
    selected_set = options.intervention_set
    include_core = selected_set in {"core", "all"}
    include_adaptation = selected_set in {"adaptation", "all"}
    include_topology = selected_set in {"topology", "all"}
    include_edge = selected_set in {"edge", "all"}

    if include_core:
        selectors = [
            "top_recurrent_current_abs_mean",
            "top_rec_input_abs_ratio",
            "top_expected_degree_score",
            "top_in_degree",
            "top_out_degree",
            "top_firing_rate",
        ]
        for top_k, top_frac, size_label in _size_specs(options, len(neuron_rows)):
            for selector in selectors:
                selected, reason = select_top_neurons(neuron_rows, selector, top_k=top_k, top_frac=top_frac)
                for mode in _neuron_modes(options):
                    intervention_id = f"{run_name}__{mode}__{selector}__{size_label}"
                    if reason:
                        specs.append(
                            _make_insufficient_spec(
                                intervention_id,
                                "neuron",
                                mode,
                                reason,
                                selection_name=selector,
                                top_k=top_k,
                                top_frac=top_frac,
                            )
                        )
                        continue
                    spec = InterventionSpec(
                        intervention_id=intervention_id,
                        family="neuron",
                        intervention_type=mode,
                        selection_name=selector,
                        selected_neurons=tuple(selected),
                        neuron_mode=mode,
                        readout_mask_spike=True,
                        readout_mask_adapt=True,
                        top_k=top_k,
                        top_frac=top_frac,
                    )
                    specs.append(spec)
                    _add_random_controls(specs, spec, neuron_rows, options)

        readout_selectors = [
            ("top_readout_total_weight_norm", True, False),
            ("top_readout_spike_weight_norm", True, False),
            ("top_recurrent_current_abs_mean", True, False),
        ]
        for top_k, top_frac, size_label in _size_specs(options, len(neuron_rows)):
            for selector, mask_spike, mask_adapt in readout_selectors:
                selected, reason = select_top_neurons(neuron_rows, selector, top_k=top_k, top_frac=top_frac)
                intervention_id = f"{run_name}__readout_mask__{selector}__{size_label}"
                if reason:
                    specs.append(
                        _make_insufficient_spec(
                            intervention_id,
                            "readout",
                            "readout_only_feature_mask",
                            reason,
                            selection_name=selector,
                            top_k=top_k,
                            top_frac=top_frac,
                        )
                    )
                    continue
                spec = InterventionSpec(
                    intervention_id=intervention_id,
                    family="readout",
                    intervention_type="readout_only_feature_mask",
                    selection_name=selector,
                    selected_neurons=tuple(selected),
                    readout_mask_spike=mask_spike,
                    readout_mask_adapt=mask_adapt,
                    top_k=top_k,
                    top_frac=top_frac,
                )
                specs.append(spec)
                _add_random_controls(specs, spec, neuron_rows, options)

    if include_adaptation:
        for top_k, top_frac, size_label in _size_specs(options, len(neuron_rows)):
            selected, reason = select_top_neurons(
                neuron_rows,
                "top_adapt_readout_contribution_proxy",
                top_k=top_k,
                top_frac=top_frac,
            )
            for selector, mask_spike, mask_adapt in [
                ("top_readout_adapt_weight_norm", False, True),
                ("top_adapt_readout_contribution_proxy", False, True),
                ("top_adapt_readout_contribution_proxy", True, True),
            ]:
                selected_mask, reason_mask = select_top_neurons(neuron_rows, selector, top_k=top_k, top_frac=top_frac)
                intervention_id = f"{run_name}__adapt_readout_mask__{selector}__{'spike_adapt' if mask_spike else 'adapt'}__{size_label}"
                if reason_mask:
                    specs.append(
                        _make_insufficient_spec(
                            intervention_id,
                            "adaptation",
                            "readout_only_feature_mask",
                            reason_mask,
                            selection_name=selector,
                            top_k=top_k,
                            top_frac=top_frac,
                        )
                    )
                else:
                    specs.append(
                        InterventionSpec(
                            intervention_id=intervention_id,
                            family="adaptation",
                            intervention_type="readout_only_feature_mask",
                            selection_name=selector,
                            selected_neurons=tuple(selected_mask),
                            readout_mask_spike=mask_spike,
                            readout_mask_adapt=mask_adapt,
                            top_k=top_k,
                            top_frac=top_frac,
                        )
                    )
            for mode in [
                "adaptation_all_zero",
                "adaptation_selected_zero",
                "adaptation_neuron_shuffle",
                "adaptation_batch_shuffle",
            ]:
                intervention_id = f"{run_name}__{mode}__{size_label}"
                if mode == "adaptation_selected_zero" and reason:
                    specs.append(
                        _make_insufficient_spec(
                            intervention_id,
                            "adaptation",
                            mode,
                            reason,
                            selection_name="top_adapt_readout_contribution_proxy",
                            top_k=top_k,
                            top_frac=top_frac,
                        )
                    )
                else:
                    specs.append(
                        InterventionSpec(
                            intervention_id=intervention_id,
                            family="adaptation",
                            intervention_type=mode,
                            selection_name="top_adapt_readout_contribution_proxy" if mode == "adaptation_selected_zero" else "all_adaptation_features",
                            selected_neurons=tuple(selected if mode == "adaptation_selected_zero" else ()),
                            adaptation_mode=mode,
                            top_k=top_k if mode == "adaptation_selected_zero" else None,
                            top_frac=top_frac if mode == "adaptation_selected_zero" else None,
                            seed=stable_seed(options.seed, intervention_id),
                        )
                    )

    if include_topology:
        if hard_mask is None:
            specs.append(
                _make_insufficient_spec(
                    f"{run_name}__topology_missing",
                    "topology",
                    "density_preserving_random_shuffle",
                    "missing hard recurrent mask",
                )
            )
        else:
            for repeat in range(options.random_repeats):
                seed = stable_seed(options.seed, run_name, "topology_density", repeat)
                specs.append(
                    InterventionSpec(
                        intervention_id=f"{run_name}__density_preserving_random_shuffle_r{repeat}",
                        family="topology",
                        intervention_type="density_preserving_random_shuffle",
                        mask_override=density_preserving_random_shuffle(hard_mask, seed),
                        repeat_index=repeat,
                        seed=seed,
                    )
                )
                seed = stable_seed(options.seed, run_name, "topology_degree", repeat)
                swapped, status, reason = degree_preserving_directed_swap(hard_mask, seed)
                specs.append(
                    InterventionSpec(
                        intervention_id=f"{run_name}__degree_preserving_directed_swap_r{repeat}",
                        family="topology",
                        intervention_type="degree_preserving_directed_swap",
                        mask_override=swapped,
                        repeat_index=repeat,
                        seed=seed,
                        evidence_status=status,
                        insufficient_reason="" if status == "available" else reason,
                    )
                )
                seed = stable_seed(options.seed, run_name, "topology_random_sparse", repeat)
                specs.append(
                    InterventionSpec(
                        intervention_id=f"{run_name}__random_sparse_like_same_density_r{repeat}",
                        family="topology",
                        intervention_type="random_sparse_like_same_density",
                        mask_override=density_preserving_random_shuffle(hard_mask, seed),
                        repeat_index=repeat,
                        seed=seed,
                    )
                )

    if include_edge:
        if hard_mask is None:
            specs.append(_make_insufficient_spec(f"{run_name}__edge_missing", "edge", "edge_removal", "missing hard recurrent mask"))
        else:
            for top_k, top_frac, size_label in _size_specs(options, int(hard_mask.sum().item())):
                edges = _top_edges(hard_mask, edge_prob, top_k=top_k, top_frac=top_frac)
                specs.append(
                    InterventionSpec(
                        intervention_id=f"{run_name}__top_expected_prob_edges_remove__{size_label}",
                        family="edge",
                        intervention_type="top_expected_prob_edges_remove",
                        selection_name="top_expected_prob_edges",
                        selected_edges=tuple(edges),
                        mask_override=_remove_edges_mask(hard_mask, edges),
                        top_k=top_k,
                        top_frac=top_frac,
                    )
                )
                hub_specs = [
                    ("hub_outgoing_edges_remove", "top_out_degree", "outgoing"),
                    ("hub_incoming_edges_remove", "top_in_degree", "incoming"),
                    ("recurrent_current_top_neuron_outgoing_edges_remove", "top_recurrent_current_abs_mean", "outgoing"),
                ]
                for intervention_type, selector, direction in hub_specs:
                    hubs, reason = select_top_neurons(neuron_rows, selector, top_k=top_k, top_frac=top_frac)
                    intervention_id = f"{run_name}__{intervention_type}__{size_label}"
                    if reason:
                        specs.append(
                            _make_insufficient_spec(
                                intervention_id,
                                "edge",
                                intervention_type,
                                reason,
                                selection_name=selector,
                                top_k=top_k,
                                top_frac=top_frac,
                            )
                        )
                    else:
                        edges = _hub_edges(hard_mask, hubs, direction)
                        specs.append(
                            InterventionSpec(
                                intervention_id=intervention_id,
                                family="edge",
                                intervention_type=intervention_type,
                                selection_name=selector,
                                selected_neurons=tuple(hubs),
                                selected_edges=tuple(edges),
                                mask_override=_remove_edges_mask(hard_mask, edges),
                                top_k=top_k,
                                top_frac=top_frac,
                            )
                        )
    return specs


def _empty_metrics() -> dict[str, float]:
    return {key: float("nan") for key in METRIC_KEYS}


def _aggregate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("intervention_type") == "original_subset":
            continue
        key = (
            row.get("family"),
            row.get("intervention_type"),
            row.get("selection_name"),
            row.get("neuron_mode"),
            row.get("top_k"),
            row.get("top_frac"),
            row.get("random_control_matching"),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        available = [row for row in group if row.get("evidence_status") in {"available", "partial_degree_preserving_directed_swap", "fallback_density_preserving_random_shuffle"}]
        summary = {
            "family": key[0],
            "intervention_type": key[1],
            "selection_name": key[2],
            "neuron_mode": key[3],
            "top_k": key[4],
            "top_frac": key[5],
            "random_control_matching": key[6],
            "n_rows": len(group),
            "n_available": len(available),
        }
        for metric in ["accuracy", "loss", "mean_logit_margin", "correct_margin", *ACTIVITY_KEYS]:
            vals = finite_values([safe_float(row.get(metric)) for row in available])
            deltas = finite_values([safe_float(row.get(f"delta_{metric}")) for row in available])
            summary[f"mean_{metric}"] = float(vals.mean()) if vals.size else float("nan")
            summary[f"mean_delta_{metric}"] = float(deltas.mean()) if deltas.size else float("nan")
        out.append(summary)
    return out


def _verdict_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("intervention_id")): row for row in rows}
    random_by_target: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        control_for = str(row.get("control_for") or "")
        if control_for:
            random_by_target.setdefault(control_for, []).append(row)
    verdicts = []
    for intervention_id, row in by_id.items():
        if row.get("intervention_type") == "original_subset" or row.get("control_for"):
            continue
        delta_acc = safe_float(row.get("delta_accuracy"))
        delta_margin = safe_float(row.get("delta_mean_logit_margin"))
        controls = random_by_target.get(intervention_id, [])
        control_acc = finite_values([safe_float(ctrl.get("delta_accuracy")) for ctrl in controls])
        control_margin = finite_values([safe_float(ctrl.get("delta_mean_logit_margin")) for ctrl in controls])
        mean_ctrl_acc = float(control_acc.mean()) if control_acc.size else float("nan")
        mean_ctrl_margin = float(control_margin.mean()) if control_margin.size else float("nan")
        sensitivity = "insufficient_evidence"
        if math.isfinite(delta_acc):
            if control_acc.size and delta_acc < mean_ctrl_acc - 0.02:
                sensitivity = "target_more_sensitive_than_random"
            elif delta_acc < -0.02:
                sensitivity = "checkpoint_sensitive"
            else:
                sensitivity = "weak_or_no_subset_sensitivity"
        verdicts.append(
            {
                "run_name": row.get("run_name"),
                "intervention_id": intervention_id,
                "family": row.get("family"),
                "intervention_type": row.get("intervention_type"),
                "selection_name": row.get("selection_name"),
                "evidence_status": row.get("evidence_status"),
                "delta_accuracy": delta_acc,
                "delta_mean_logit_margin": delta_margin,
                "random_control_count": len(controls),
                "random_control_matching": ";".join(sorted({str(ctrl.get("random_control_matching")) for ctrl in controls if ctrl.get("random_control_matching")})),
                "mean_random_delta_accuracy": mean_ctrl_acc,
                "mean_random_delta_margin": mean_ctrl_margin,
                "verdict": sensitivity,
            }
        )
    return verdicts


def _placeholder_figure(path: Path, title: str, message: str = "insufficient evidence") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bar_figure(path: Path, rows: list[dict[str, Any]], metric: str, title: str, *, family: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filtered = [
        row
        for row in rows
        if row.get("intervention_type") != "original_subset"
        and (family is None or row.get("family") == family)
        and math.isfinite(safe_float(row.get(metric)))
        and not row.get("control_for")
    ]
    if not filtered:
        _placeholder_figure(path, title)
        return
    grouped: dict[str, list[float]] = {}
    for row in filtered:
        label = str(row.get("intervention_type"))
        if row.get("selection_name"):
            label += "\n" + str(row.get("selection_name")).replace("top_", "")
        grouped.setdefault(label, []).append(safe_float(row.get(metric)))
    labels = list(grouped)[:16]
    values = [float(np.mean(grouped[label])) for label in labels]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.6), 4.5))
    ax.bar(range(len(labels)), values)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _hist_figure(path: Path, rows: list[dict[str, Any]], metric: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = finite_values([safe_float(row.get(metric)) for row in rows if row.get("control_for")])
    if values.size == 0:
        _placeholder_figure(path, title)
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=30)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel(metric)
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    cache_dir = output_dir / ".plot_cache"
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _bar_figure(figures / "intervention_accuracy_delta_bar.png", rows, "delta_accuracy", "Accuracy delta vs original_subset")
    _bar_figure(figures / "intervention_margin_delta_bar.png", rows, "delta_mean_logit_margin", "Logit margin delta vs original_subset")
    _bar_figure(figures / "intervention_activity_delta_bar.png", rows, "delta_mean_firing_rate", "Mean firing-rate delta vs original_subset")
    _hist_figure(figures / "random_control_delta_distribution.png", rows, "delta_accuracy", "Random-control accuracy deltas")
    _bar_figure(figures / "topology_shuffle_accuracy_delta.png", rows, "delta_accuracy", "Topology shuffle accuracy delta", family="topology")
    _bar_figure(figures / "adaptation_zero_shuffle_delta.png", rows, "delta_accuracy", "Adaptation intervention accuracy delta", family="adaptation")


def generate_report(
    output_dir: Path,
    run_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    verdict_rows: list[dict[str, Any]],
    warnings: list[str],
    metadata: dict[str, Any],
) -> str:
    baseline_rows = [row for row in run_rows if row.get("intervention_type") == "original_subset"]
    available = [row for row in run_rows if row.get("evidence_status") in {"available", "partial_degree_preserving_directed_swap", "fallback_density_preserving_random_shuffle"}]
    lines = [
        "# Lowrank Checkpoint-Level Intervention Sensitivity",
        "",
        "## 1. Purpose",
        "This report evaluates checkpoint-level intervention sensitivity for fixed validation-selected `best.pt` checkpoints. It does not establish a full mechanism claim.",
        "",
        "## 2. Runs and checkpoints",
    ]
    if baseline_rows:
        for row in baseline_rows:
            lines.append(
                f"- `{row.get('run_name')}` original_subset acc={safe_float(row.get('accuracy')):.4f} loss={safe_float(row.get('loss')):.4f} batches={metadata.get('num_batches')}"
            )
    else:
        lines.append("- insufficient evidence: no original_subset baseline rows were produced")
    lines.extend(
        [
            "",
            "## 3. Baseline reproduction",
            "All deltas use the fixed `original_subset` baseline evaluated on the exact same materialized diagnostic batches as every intervention.",
            "",
            "## 4. Neuron output knockout results",
        ]
    )
    _append_summary(lines, summary_rows, family="neuron")
    lines.append("")
    lines.append("## 5. Readout-only masking results")
    _append_summary(lines, summary_rows, family="readout")
    lines.append("")
    lines.append("## 6. Adaptation zero/shuffle results")
    _append_summary(lines, summary_rows, family="adaptation")
    lines.append("")
    lines.append("## 7. Topology shuffle results")
    _append_summary(lines, summary_rows, family="topology")
    lines.append("")
    lines.append("## 8. Edge/hub removal results")
    _append_summary(lines, summary_rows, family="edge")
    lines.append("")
    lines.append("## 9. Random matched controls")
    matching = sorted({str(row.get("random_control_matching")) for row in run_rows if row.get("random_control_matching")})
    lines.append("- Random control matching available: " + (", ".join(matching) if matching else "none"))
    lines.append(f"- Random control rows: {sum(1 for row in run_rows if row.get('control_for'))}")
    lines.append("")
    lines.append("## 10. Interpretation")
    if verdict_rows:
        for row in verdict_rows[:30]:
            lines.append(
                f"- `{row.get('run_name')}` {row.get('intervention_type')} {row.get('selection_name')}: {row.get('verdict')} delta_acc={safe_float(row.get('delta_accuracy')):.4f} random_mean={safe_float(row.get('mean_random_delta_accuracy')):.4f}"
            )
    else:
        lines.append("- insufficient evidence: no verdict rows")
    lines.append("")
    lines.append("## 11. Limitations")
    lines.append("- This is checkpoint-level intervention evidence, not a retraining or full-mechanism result.")
    lines.append("- Retraining recovery is not tested.")
    lines.append("- Checkpoints are fixed and interventions are temporary diagnostic forward overrides.")
    lines.append("- Conclusions are limited to whether this checkpoint's decisions/features are sensitive to the intervention.")
    lines.append("")
    lines.append("## 12. Next recommended actions")
    lines.append("- Start with `--intervention-set core`; then run `adaptation`, `topology`, or `edge` only for runs with clear subset sensitivity.")
    lines.append("- If topology shuffles degrade accuracy, follow up with retraining-with-constraint experiments outside this diagnostic.")
    if warnings:
        lines.append("")
        lines.append("Warnings / insufficient evidence:")
        for warning in warnings[:60]:
            lines.append(f"- {warning}")
        if len(warnings) > 60:
            lines.append(f"- ... {len(warnings) - 60} more warnings")
    report = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(report)
    return report


def _append_summary(lines: list[str], summary_rows: list[dict[str, Any]], *, family: str) -> None:
    rows = [row for row in summary_rows if row.get("family") == family and not row.get("random_control_matching")]
    if not rows:
        lines.append("- insufficient evidence or not run in this intervention set")
        return
    for row in rows[:12]:
        lines.append(
            f"- {row.get('intervention_type')} {row.get('selection_name')}: n={row.get('n_available')}/{row.get('n_rows')} mean_delta_acc={safe_float(row.get('mean_delta_accuracy')):.4f} mean_delta_margin={safe_float(row.get('mean_delta_mean_logit_margin')):.4f}"
        )


def _metadata_for_options(options: InterventionOptions, device: torch.device) -> dict[str, Any]:
    return {
        "run_dirs": [str(path) for path in options.run_dirs],
        "diagnostic_dir": str(options.diagnostic_dir),
        "output_dir": str(options.output_dir),
        "num_batches": options.num_batches,
        "batch_size": options.batch_size,
        "top_k": [int(k) for k in options.top_k],
        "top_frac": [float(x) for x in options.top_frac],
        "random_repeats": options.random_repeats,
        "device": str(device),
        "intervention_set": options.intervention_set,
        "neuron_intervention_mode": options.neuron_intervention_mode,
        "degree_bin_controls": options.degree_bin_controls,
        "baseline_name": "original_subset",
        "delta_reference": "original_subset",
        "seed": options.seed,
    }


def _validate_spec_for_model(spec: InterventionSpec, model: torch.nn.Module) -> InterventionSpec:
    if spec.evidence_status == "insufficient_evidence":
        return spec
    readout_mode = str(getattr(model, "readout_mode", ""))
    neuron_type = str(getattr(model, "neuron_type", "")).lower()
    if spec.family == "adaptation":
        if neuron_type != "alif" or readout_mode != "spike_adaptation_concat":
            spec.evidence_status = "insufficient_evidence"
            spec.insufficient_reason = "adaptation intervention requires ALIF spike_adaptation_concat readout"
    if spec.family == "readout" and readout_mode not in {"spike_count", "spike_adaptation_concat", "membrane_trace"}:
        spec.evidence_status = "insufficient_evidence"
        spec.insufficient_reason = f"readout-only masking unsupported for readout_mode={readout_mode}"
    return spec


def _run_insufficient_row(
    run_dir: Path,
    reason: str,
) -> dict[str, Any]:
    spec = _make_insufficient_spec(
        f"{run_dir.name}__original_subset",
        "baseline",
        "original_subset",
        reason,
    )
    row = _base_metadata(run_dir.name, run_dir, None, {}, spec)
    row.update(_empty_metrics())
    for key in METRIC_KEYS:
        row[f"delta_{key}"] = float("nan")
    return row


def process_run(
    run_dir: Path,
    tables: dict[str, list[dict[str, str]]],
    options: InterventionOptions,
    device: torch.device,
) -> dict[str, Any]:
    artifacts = discover_run_artifacts(run_dir)
    run_name = artifacts.run_dir.name
    warnings: list[str] = []
    run_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    selected_neuron_rows: list[dict[str, Any]] = []
    selected_edge_rows: list[dict[str, Any]] = []

    if artifacts.config_path is None:
        reason = "missing config.yaml"
        warnings.append(f"{run_name}: {reason}")
        run_rows.append(_run_insufficient_row(run_dir, reason))
        return {
            "run_rows": run_rows,
            "classwise_rows": classwise_rows,
            "selected_neuron_rows": selected_neuron_rows,
            "selected_edge_rows": selected_edge_rows,
            "warnings": warnings,
        }
    if artifacts.best_checkpoint_path is None:
        reason = "missing checkpoints/best.pt"
        warnings.append(f"{run_name}: {reason}")
        run_rows.append(_run_insufficient_row(run_dir, reason))
        return {
            "run_rows": run_rows,
            "classwise_rows": classwise_rows,
            "selected_neuron_rows": selected_neuron_rows,
            "selected_edge_rows": selected_edge_rows,
            "warnings": warnings,
        }

    try:
        cfg = load_config(artifacts.config_path)
        model, checkpoint, load_warnings = load_model_and_checkpoint(cfg, artifacts.best_checkpoint_path, device)
        warnings.extend(f"{run_name}: {warning}" for warning in load_warnings)
        batches = materialize_fixed_batches(cfg, options.batch_size, options.num_batches)
        if not batches:
            raise RuntimeError("diagnostic dataloader returned no batches")
    except Exception as exc:
        reason = f"run setup failed: {type(exc).__name__}: {exc}"
        warnings.append(f"{run_name}: {reason}")
        run_rows.append(_run_insufficient_row(run_dir, reason))
        return {
            "run_rows": run_rows,
            "classwise_rows": classwise_rows,
            "selected_neuron_rows": selected_neuron_rows,
            "selected_edge_rows": selected_edge_rows,
            "warnings": warnings,
        }

    raw_log_rows = read_train_jsonl(artifacts.train_log_path)
    if not raw_log_rows:
        warnings.append(f"{run_name}: missing or empty logs/train.jsonl")

    baseline_spec = InterventionSpec(
        intervention_id=f"{run_name}__original_subset",
        family="baseline",
        intervention_type="original_subset",
    )
    tau = float(getattr(cfg, "tau_end", 0.05))
    baseline_metrics, baseline_classwise = evaluate_on_batches(model, batches, baseline_spec, device, tau=tau)
    baseline_by_class = {int(row["class_id"]): row for row in baseline_classwise}
    run_rows.append(_metric_row(run_name, run_dir, cfg, checkpoint, baseline_spec, baseline_metrics, None))
    classwise_rows.extend(_classwise_rows(run_name, baseline_spec, baseline_classwise, None))

    neuron_rows = rows_for_run(tables.get("neuron_table", []), run_name, run_dir)
    if not neuron_rows:
        warnings.append(f"{run_name}: diagnostic neuron_table rows missing; neuron/readout selections will be insufficient")
    hard_mask, edge_prob = _hard_mask_and_edge_prob(model)
    specs = build_intervention_specs(run_name, neuron_rows, hard_mask, edge_prob, options)

    for spec in specs:
        spec = _validate_spec_for_model(spec, model)
        selected_neuron_rows.extend(_neuron_records(run_name, spec, neuron_rows))
        selected_edge_rows.extend(_edge_records(run_name, spec, edge_prob, hard_mask))
        if spec.evidence_status == "insufficient_evidence":
            metrics = _empty_metrics()
            run_rows.append(_metric_row(run_name, run_dir, cfg, checkpoint, spec, metrics, baseline_metrics))
            continue
        try:
            metrics, classwise = evaluate_on_batches(model, batches, spec, device, tau=tau)
            row = _metric_row(run_name, run_dir, cfg, checkpoint, spec, metrics, baseline_metrics)
            run_rows.append(row)
            classwise_rows.extend(_classwise_rows(run_name, spec, classwise, baseline_by_class))
        except Exception as exc:
            spec.evidence_status = "insufficient_evidence"
            spec.insufficient_reason = f"evaluation failed: {type(exc).__name__}: {exc}"
            warnings.append(f"{run_name}: {spec.intervention_id}: {spec.insufficient_reason}")
            metrics = _empty_metrics()
            run_rows.append(_metric_row(run_name, run_dir, cfg, checkpoint, spec, metrics, baseline_metrics))

    return {
        "run_rows": run_rows,
        "classwise_rows": classwise_rows,
        "selected_neuron_rows": selected_neuron_rows,
        "selected_edge_rows": selected_edge_rows,
        "warnings": warnings,
    }


def run_intervention_diagnostics(options: InterventionOptions) -> dict[str, Any]:
    device = _select_device(options.device)
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = load_diagnostic_tables(Path(options.diagnostic_dir))
    all_run_rows: list[dict[str, Any]] = []
    all_classwise_rows: list[dict[str, Any]] = []
    all_selected_neurons: list[dict[str, Any]] = []
    all_selected_edges: list[dict[str, Any]] = []
    warnings: list[str] = []

    for run_dir in options.run_dirs:
        result = process_run(Path(run_dir), tables, options, device)
        all_run_rows.extend(result["run_rows"])
        all_classwise_rows.extend(result["classwise_rows"])
        all_selected_neurons.extend(result["selected_neuron_rows"])
        all_selected_edges.extend(result["selected_edge_rows"])
        warnings.extend(result["warnings"])

    summary_rows = _aggregate_summary(all_run_rows)
    verdict_rows = _verdict_rows(all_run_rows)
    activity_rows = [
        {
            **{
                key: row.get(key)
                for key in [
                    "run_name",
                    "intervention_id",
                    "family",
                    "intervention_type",
                    "selection_name",
                    "evidence_status",
                ]
            },
            **{key: row.get(key) for key in ACTIVITY_KEYS},
            **{f"delta_{key}": row.get(f"delta_{key}") for key in ACTIVITY_KEYS},
        }
        for row in all_run_rows
    ]
    random_rows = [row for row in all_run_rows if row.get("control_for")]
    metadata = _metadata_for_options(options, device)
    metadata["diagnostic_input_files"] = {
        name: str(Path(options.diagnostic_dir) / filename)
        for name, filename in {
            "neuron_table": "neuron_table.csv",
            "lowrank_role_summary": "lowrank_role_summary.csv",
            "topk_overlap_summary": "topk_overlap_summary.csv",
            "correlation_summary": "correlation_summary.csv",
        }.items()
    }
    metadata["warnings"] = warnings

    write_csv(output_dir / "intervention_run_level.csv", all_run_rows)
    write_csv(output_dir / "intervention_summary.csv", summary_rows)
    write_csv(output_dir / "intervention_classwise.csv", all_classwise_rows)
    write_csv(output_dir / "intervention_activity_summary.csv", activity_rows)
    write_csv(output_dir / "intervention_random_controls.csv", random_rows)
    write_csv(output_dir / "intervention_verdicts.csv", verdict_rows)
    write_csv(output_dir / "selected_neurons.csv", all_selected_neurons)
    write_csv(output_dir / "selected_edges.csv", all_selected_edges)
    (output_dir / "intervention_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    generate_report(output_dir, all_run_rows, summary_rows, verdict_rows, warnings, metadata)
    write_figures(output_dir, all_run_rows)

    artifact_paths = [
        output_dir / "intervention_summary.csv",
        output_dir / "intervention_run_level.csv",
        output_dir / "intervention_classwise.csv",
        output_dir / "intervention_activity_summary.csv",
        output_dir / "intervention_random_controls.csv",
        output_dir / "intervention_verdicts.csv",
        output_dir / "selected_neurons.csv",
        output_dir / "selected_edges.csv",
        output_dir / "intervention_metadata.json",
        output_dir / "report.md",
        output_dir / "figures",
    ]
    return {
        "output_dir": output_dir,
        "device": str(device),
        "warnings": warnings,
        "artifact_paths": artifact_paths,
        "run_rows": all_run_rows,
        "summary_rows": summary_rows,
        "verdict_rows": verdict_rows,
    }
