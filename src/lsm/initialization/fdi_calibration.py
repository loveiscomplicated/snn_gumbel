"""Probe-batch FDI-style firing regime calibration for SHD LSM models."""

from __future__ import annotations

import itertools
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPORT_FILENAME = "init_fdi_calibration_report.json"
EPS = 1e-8


class _MomentStats:
    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.max_value: float | None = None

    def update(self, x: torch.Tensor) -> None:
        values = x.detach().float()
        count = values.numel()
        if count == 0:
            return
        self.n += count
        self.total += float(values.sum().cpu().item())
        self.total_sq += float(values.square().sum().cpu().item())
        value_max = float(values.max().cpu().item())
        if self.max_value is None or value_max > self.max_value:
            self.max_value = value_max

    @property
    def mean(self) -> float | None:
        if self.n == 0:
            return None
        return self.total / self.n

    @property
    def std(self) -> float | None:
        if self.n == 0:
            return None
        mean = self.total / self.n
        var = max(self.total_sq / self.n - mean * mean, 0.0)
        return math.sqrt(var)

    @property
    def max(self) -> float | None:
        return self.max_value


def _liquid_cfg(config) -> Any:
    return getattr(config, "liquid", config)


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _infer_dt_sec(config) -> float:
    liq = _liquid_cfg(config)
    for owner in (liq, config):
        for name in ("fdi_dt_sec", "dt_sec", "timestep_sec"):
            value = getattr(owner, name, None)
            if value is not None:
                value = float(value)
                if value <= 0.0:
                    raise ValueError(f"{name} must be positive, got {value}")
                return value

    dataset = str(getattr(config, "dataset", "")).lower()
    if dataset == "shd":
        return 0.01
    return 1.0


@contextmanager
def _preserve_model_probe_state(model):
    was_training = model.training
    liquid = getattr(model, "liquid", None)
    epoch_noise = getattr(liquid, "_epoch_noise", None)
    epoch_tau = getattr(liquid, "_epoch_tau", None)

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    mps_rng_state = None
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "get_rng_state"):
        try:
            mps_rng_state = mps.get_rng_state()
        except RuntimeError:
            mps_rng_state = None

    try:
        model.eval()
        if liquid is not None and hasattr(liquid, "unlock_epoch_mask"):
            liquid.unlock_epoch_mask()
        yield
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if (
            mps_rng_state is not None
            and mps is not None
            and hasattr(mps, "set_rng_state")
        ):
            try:
                mps.set_rng_state(mps_rng_state)
            except RuntimeError:
                pass
        if liquid is not None:
            if hasattr(liquid, "_epoch_noise"):
                liquid._epoch_noise = epoch_noise
            if hasattr(liquid, "_epoch_tau"):
                liquid._epoch_tau = epoch_tau
        if was_training:
            model.train()
        else:
            model.eval()


def _batch_x(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _materialize_probe_batches(train_loader, n_batches: int) -> list:
    if n_batches <= 0:
        raise ValueError("liquid.fdi_probe_batches must be positive")

    probe_batches = []
    for batch in train_loader:
        if isinstance(batch, (tuple, list)):
            materialized = tuple(
                item.detach().cpu() if torch.is_tensor(item) else item for item in batch
            )
        elif torch.is_tensor(batch):
            materialized = batch.detach().cpu()
        else:
            materialized = batch
        probe_batches.append(materialized)
        if len(probe_batches) >= n_batches:
            break
    if not probe_batches:
        raise ValueError("FDI calibration requires at least one probe batch")
    return probe_batches


@torch.no_grad()
def collect_initial_regime_stats(model, probe_batches, config, device) -> dict:
    """Collect initial firing/current/adaptation stats from materialized probe batches."""

    dt_sec = _infer_dt_sec(config)
    liq = _liquid_cfg(config)

    membrane = _MomentStats()
    input_current = _MomentStats()
    recurrent_current = _MomentStats()
    theta_eff = _MomentStats()
    adaptation = _MomentStats()
    adaptation_threshold_contrib = _MomentStats()

    spike_counts: torch.Tensor | None = None
    total_duration_sec = 0.0
    total_samples = 0
    total_timesteps = 0

    threshold = getattr(getattr(model, "liquid", None), "threshold", None)
    threshold_mean = None
    if torch.is_tensor(threshold):
        threshold_mean = float(threshold.detach().float().mean().cpu().item())

    with _preserve_model_probe_state(model):
        for batch in probe_batches:
            x = _batch_x(batch).to(device)
            _, traces = model(x, tau=1.0, return_traces=True)
            spikes = traces["spikes"].detach().float()
            batch_size, timesteps, n_neurons = spikes.shape
            if spike_counts is None:
                spike_counts = torch.zeros(n_neurons, dtype=torch.float64)
            spike_counts += spikes.sum(dim=(0, 1)).detach().cpu().double()
            total_duration_sec += batch_size * timesteps * dt_sec
            total_samples += batch_size
            total_timesteps += timesteps

            membrane.update(traces["membrane"])
            input_current.update(traces["input_current"])
            recurrent_current.update(traces["recurrent_current"])

            trace_theta_eff = traces.get("theta_eff")
            if trace_theta_eff is not None:
                theta_eff.update(trace_theta_eff)

            trace_adaptation = traces.get("adaptation")
            if trace_adaptation is not None:
                adaptation.update(trace_adaptation)
                liquid = model.liquid
                if hasattr(liquid, "alif_beta"):
                    beta = liquid.alif_beta.detach()
                    contrib = trace_adaptation.detach().float() * beta.view(1, 1, -1)
                    adaptation_threshold_contrib.update(contrib)

    if spike_counts is None or total_duration_sec <= 0.0:
        raise ValueError("FDI calibration collected no probe timesteps")

    per_neuron_rate_hz = spike_counts / total_duration_sec
    mean_rate_hz = float(per_neuron_rate_hz.mean().item())
    median_rate_hz = float(per_neuron_rate_hz.median().item())
    max_rate_hz = float(per_neuron_rate_hz.max().item())
    silent_neuron_frac = float(
        (per_neuron_rate_hz < float(liq.fdi_silent_rate_hz)).double().mean().item()
    )
    overactive_neuron_frac = float(
        (per_neuron_rate_hz > float(liq.fdi_overactive_rate_hz)).double().mean().item()
    )

    membrane_mean = membrane.mean
    membrane_std = membrane.std
    input_current_std = input_current.std
    recurrent_current_std = recurrent_current.std
    recurrent_to_input_std_ratio = None
    if input_current_std is not None and recurrent_current_std is not None:
        recurrent_to_input_std_ratio = recurrent_current_std / max(
            input_current_std, EPS
        )

    theta_eff_mean = theta_eff.mean
    threshold_ref = theta_eff_mean if theta_eff_mean is not None else threshold_mean
    xi_mean = None
    if (
        threshold_ref is not None
        and membrane_mean is not None
        and membrane_std is not None
    ):
        xi_mean = (threshold_ref - membrane_mean) / (membrane_std + EPS)

    stats = {
        "dt_sec": float(dt_sec),
        "probe_batches": int(len(probe_batches)),
        "probe_samples": int(total_samples),
        "probe_timesteps": int(total_timesteps),
        "mean_rate_hz": mean_rate_hz,
        "median_rate_hz": median_rate_hz,
        "max_rate_hz": max_rate_hz,
        "silent_neuron_frac": silent_neuron_frac,
        "overactive_neuron_frac": overactive_neuron_frac,
        "membrane_mean": _finite_or_none(membrane_mean),
        "membrane_std": _finite_or_none(membrane_std),
        "threshold_mean": _finite_or_none(threshold_mean),
        "xi_mean": _finite_or_none(xi_mean),
        "input_current_std": _finite_or_none(input_current_std),
        "recurrent_current_std": _finite_or_none(recurrent_current_std),
        "recurrent_to_input_std_ratio": _finite_or_none(recurrent_to_input_std_ratio),
    }

    if theta_eff_mean is not None:
        stats["theta_eff_mean"] = _finite_or_none(theta_eff_mean)
    if adaptation.n > 0:
        adapt_ratio = None
        if threshold_mean is not None:
            adapt_ratio = (adaptation_threshold_contrib.mean or 0.0) / (
                abs(threshold_mean) + EPS
            )
        stats.update(
            {
                "adaptation_mean": _finite_or_none(adaptation.mean),
                "adaptation_max": _finite_or_none(adaptation.max),
                "adaptation_to_threshold_ratio": _finite_or_none(adapt_ratio),
            }
        )

    return stats


@torch.no_grad()
def score_initial_regime(stats: dict, config) -> float:
    liq = _liquid_cfg(config)
    score = 0.0

    mean_rate_hz = float(stats.get("mean_rate_hz", 0.0))
    target_rate = max(float(liq.fdi_target_rate_hz), EPS)
    score += abs(mean_rate_hz - target_rate) / target_rate
    score += (
        5.0
        * max(0.0, float(liq.fdi_target_rate_hz_min) - mean_rate_hz)
        / max(float(liq.fdi_target_rate_hz_min), EPS)
    )
    score += (
        5.0
        * max(0.0, mean_rate_hz - float(liq.fdi_target_rate_hz_max))
        / max(float(liq.fdi_target_rate_hz_max), EPS)
    )

    score += 10.0 * max(
        0.0,
        float(stats.get("silent_neuron_frac", 0.0)) - float(liq.fdi_max_silent_frac),
    )
    score += 10.0 * max(
        0.0,
        float(stats.get("overactive_neuron_frac", 0.0))
        - float(liq.fdi_max_overactive_frac),
    )

    xi_mean = stats.get("xi_mean")
    if xi_mean is not None and math.isfinite(float(xi_mean)):
        xi_mean = float(xi_mean)
        score += 2.0 * max(0.0, float(liq.fdi_target_xi_min) - xi_mean)
        score += 2.0 * max(0.0, xi_mean - float(liq.fdi_target_xi_max))

    adapt_ratio = stats.get("adaptation_to_threshold_ratio")
    if adapt_ratio is not None and math.isfinite(float(adapt_ratio)):
        score += 5.0 * max(0.0, float(adapt_ratio) - float(liq.fdi_max_adapt_ratio))

    current_ratio = stats.get("recurrent_to_input_std_ratio")
    if current_ratio is not None and math.isfinite(float(current_ratio)):
        current_ratio = float(current_ratio)
        ratio_min = max(float(liq.fdi_recurrent_to_input_ratio_min), EPS)
        ratio_max = max(float(liq.fdi_recurrent_to_input_ratio_max), EPS)
        score += max(0.0, ratio_min - current_ratio) / ratio_min
        score += max(0.0, current_ratio - ratio_max) / ratio_max

    return float(score)


def _constraint_warnings(stats: dict, config) -> list[str]:
    liq = _liquid_cfg(config)
    warnings: list[str] = []
    mean_rate_hz = float(stats.get("mean_rate_hz", 0.0))
    if mean_rate_hz < float(liq.fdi_target_rate_hz_min):
        warnings.append(
            "selected candidate mean_rate_hz="
            f"{mean_rate_hz:.6g} is below fdi_target_rate_hz_min="
            f"{float(liq.fdi_target_rate_hz_min):.6g}"
        )
    if mean_rate_hz > float(liq.fdi_target_rate_hz_max):
        warnings.append(
            "selected candidate mean_rate_hz="
            f"{mean_rate_hz:.6g} is above fdi_target_rate_hz_max="
            f"{float(liq.fdi_target_rate_hz_max):.6g}"
        )
    silent_frac = float(stats.get("silent_neuron_frac", 0.0))
    if silent_frac > float(liq.fdi_max_silent_frac):
        warnings.append(
            "selected candidate silent_neuron_frac="
            f"{silent_frac:.6g} exceeds fdi_max_silent_frac="
            f"{float(liq.fdi_max_silent_frac):.6g}"
        )
    overactive_frac = float(stats.get("overactive_neuron_frac", 0.0))
    if overactive_frac > float(liq.fdi_max_overactive_frac):
        warnings.append(
            "selected candidate overactive_neuron_frac="
            f"{overactive_frac:.6g} exceeds fdi_max_overactive_frac="
            f"{float(liq.fdi_max_overactive_frac):.6g}"
        )

    xi_mean = stats.get("xi_mean")
    if xi_mean is not None and math.isfinite(float(xi_mean)):
        xi_mean = float(xi_mean)
        if xi_mean < float(liq.fdi_target_xi_min):
            warnings.append(
                f"selected candidate xi_mean={xi_mean:.6g} is below "
                f"fdi_target_xi_min={float(liq.fdi_target_xi_min):.6g}"
            )
        if xi_mean > float(liq.fdi_target_xi_max):
            warnings.append(
                f"selected candidate xi_mean={xi_mean:.6g} is above "
                f"fdi_target_xi_max={float(liq.fdi_target_xi_max):.6g}"
            )

    adapt_ratio = stats.get("adaptation_to_threshold_ratio")
    if adapt_ratio is not None and math.isfinite(float(adapt_ratio)):
        adapt_ratio = float(adapt_ratio)
        if adapt_ratio > float(liq.fdi_max_adapt_ratio):
            warnings.append(
                "selected candidate adaptation_to_threshold_ratio="
                f"{adapt_ratio:.6g} exceeds fdi_max_adapt_ratio="
                f"{float(liq.fdi_max_adapt_ratio):.6g}"
            )

    current_ratio = stats.get("recurrent_to_input_std_ratio")
    if current_ratio is not None and math.isfinite(float(current_ratio)):
        current_ratio = float(current_ratio)
        ratio_min = float(liq.fdi_recurrent_to_input_ratio_min)
        ratio_max = float(liq.fdi_recurrent_to_input_ratio_max)
        if current_ratio < ratio_min or current_ratio > ratio_max:
            warnings.append(
                "selected candidate recurrent_to_input_std_ratio="
                f"{current_ratio:.6g} is outside target range "
                f"[{ratio_min:.6g}, {ratio_max:.6g}]"
            )
    return warnings


def _state_dict_clone(model) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_state(model, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(EPS)
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))


def _scale_input_projection(model, scale: float) -> tuple[bool, str | None]:
    if abs(scale - 1.0) <= EPS:
        return True, None
    input_proj = getattr(model, "input_proj", None)
    weight = getattr(input_proj, "weight", None)
    if not torch.is_tensor(weight) or isinstance(weight, nn.Parameter):
        return (
            False,
            "input_scale skipped: model.input_proj.weight is not a fixed tensor buffer",
        )
    weight.mul_(float(scale))
    return True, None


def _scale_threshold(model, scale: float) -> tuple[bool, str | None]:
    if abs(scale - 1.0) <= EPS:
        return True, None
    liquid = getattr(model, "liquid", None)
    threshold = getattr(liquid, "threshold", None)
    n_liquid = getattr(model, "n_liquid", None)
    if not torch.is_tensor(threshold):
        return False, "threshold_scale skipped: model.liquid.threshold is unavailable"
    if n_liquid is not None and threshold.numel() != int(n_liquid):
        return (
            False,
            "threshold_scale skipped: model.liquid.threshold shape is not the liquid threshold vector",
        )
    threshold.mul_(float(scale))
    return True, None


def _scale_recurrent_weight(model, scale: float) -> tuple[bool, str | None]:
    if abs(scale - 1.0) <= EPS:
        return True, None
    liquid = getattr(model, "liquid", None)
    if liquid is None:
        return False, "recurrent_scale skipped: model has no liquid layer"
    if not hasattr(liquid, "w_raw") or not torch.is_tensor(liquid.w_raw):
        return False, "recurrent_scale skipped: liquid has no w_raw tensor"
    if not hasattr(liquid, "w_raw_max"):
        return False, "recurrent_scale skipped: liquid has no w_raw_max clamp"
    if not hasattr(liquid, "dale_sign") or not hasattr(liquid, "self_conn_mask"):
        return (
            False,
            "recurrent_scale skipped: recurrent sign/mask parameterization is not explicit",
        )
    if not hasattr(liquid, "get_effective_weight"):
        return (
            False,
            "recurrent_scale skipped: liquid exposes no effective-weight helper",
        )

    w_raw = liquid.w_raw
    w_raw_max = float(liquid.w_raw_max)
    current_mag = F.softplus(torch.clamp(w_raw.detach(), max=w_raw_max))
    target_mag = current_mag * float(scale)
    target_raw = _inverse_softplus(target_mag)
    if bool((target_raw > w_raw_max + 1e-6).any().item()):
        return (
            False,
            "recurrent_scale skipped for this candidate: inverse-softplus target exceeds w_raw_max",
        )
    w_raw.copy_(target_raw.to(device=w_raw.device, dtype=w_raw.dtype))
    return True, None


def _supports_input_scale(model) -> tuple[bool, str | None]:
    input_proj = getattr(model, "input_proj", None)
    weight = getattr(input_proj, "weight", None)
    if torch.is_tensor(weight) and not isinstance(weight, nn.Parameter):
        return True, None
    return (
        False,
        "input_scale skipped: model.input_proj.weight is not a fixed tensor buffer",
    )


def _supports_threshold_scale(model) -> tuple[bool, str | None]:
    threshold = getattr(getattr(model, "liquid", None), "threshold", None)
    n_liquid = getattr(model, "n_liquid", None)
    if not torch.is_tensor(threshold):
        return False, "threshold_scale skipped: model.liquid.threshold is unavailable"
    if n_liquid is not None and threshold.numel() != int(n_liquid):
        return (
            False,
            "threshold_scale skipped: model.liquid.threshold shape is not the liquid threshold vector",
        )
    return True, None


def _supports_recurrent_scale(model) -> tuple[bool, str | None]:
    liquid = getattr(model, "liquid", None)
    if liquid is None:
        return False, "recurrent_scale skipped: model has no liquid layer"
    if not hasattr(liquid, "w_raw") or not torch.is_tensor(liquid.w_raw):
        return False, "recurrent_scale skipped: liquid has no w_raw tensor"
    if not hasattr(liquid, "w_raw_max"):
        return False, "recurrent_scale skipped: liquid has no w_raw_max clamp"
    if not hasattr(liquid, "dale_sign") or not hasattr(liquid, "self_conn_mask"):
        return (
            False,
            "recurrent_scale skipped: recurrent sign/mask parameterization is not explicit",
        )
    if not hasattr(liquid, "get_effective_weight"):
        return (
            False,
            "recurrent_scale skipped: liquid exposes no effective-weight helper",
        )
    return True, None


def _candidate_scales(
    model,
    config,
    warnings: list[str],
) -> tuple[list[float], list[float], list[float], list[dict]]:
    liq = _liquid_cfg(config)
    skipped: list[dict] = []

    input_ok, input_reason = _supports_input_scale(model)
    recurrent_ok, recurrent_reason = _supports_recurrent_scale(model)
    threshold_ok, threshold_reason = _supports_threshold_scale(model)

    if input_ok:
        input_scales = [float(v) for v in liq.fdi_candidate_input_scales]
    else:
        input_scales = [1.0]
        skipped.append({"dimension": "input_scale", "reason": input_reason})
        warnings.append(input_reason)

    if recurrent_ok:
        recurrent_scales = [float(v) for v in liq.fdi_candidate_recurrent_scales]
    else:
        recurrent_scales = [1.0]
        skipped.append({"dimension": "recurrent_scale", "reason": recurrent_reason})
        warnings.append(recurrent_reason)

    if threshold_ok:
        threshold_scales = [float(v) for v in liq.fdi_candidate_threshold_scales]
    else:
        threshold_scales = [1.0]
        skipped.append({"dimension": "threshold_scale", "reason": threshold_reason})
        warnings.append(threshold_reason)

    return input_scales, recurrent_scales, threshold_scales, skipped


def _apply_candidate(model, candidate: dict) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    ok, reason = _scale_input_projection(model, float(candidate["input_scale"]))
    if not ok and reason is not None:
        warnings.append(reason)
        return False, warnings
    ok, reason = _scale_recurrent_weight(model, float(candidate["recurrent_scale"]))
    if not ok and reason is not None:
        warnings.append(reason)
        return False, warnings
    ok, reason = _scale_threshold(model, float(candidate["threshold_scale"]))
    if not ok and reason is not None:
        warnings.append(reason)
        return False, warnings
    return True, warnings


def _write_report(report: dict, output_dir) -> None:
    if output_dir is None:
        return
    path = Path(output_dir) / REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def _print_summary(report: dict) -> None:
    candidate = report["selected_candidate"]
    stats = report["selected_stats"]
    print("[FDI calibration]")
    print(
        "selected "
        f"input_scale={candidate['input_scale']} "
        f"recurrent_scale={candidate['recurrent_scale']} "
        f"threshold_scale={candidate['threshold_scale']}"
    )
    print(
        f"mean_rate_hz={stats.get('mean_rate_hz')} "
        f"silent_frac={stats.get('silent_neuron_frac')} "
        f"overactive_frac={stats.get('overactive_neuron_frac')} "
        f"xi_mean={stats.get('xi_mean')} "
        f"adaptation_to_threshold_ratio={stats.get('adaptation_to_threshold_ratio')} "
        f"recurrent/input std ratio={stats.get('recurrent_to_input_std_ratio')}"
    )


@torch.no_grad()
def calibrate_fdi_style_initial_regime(
    model,
    train_loader,
    config,
    device,
    output_dir=None,
) -> dict:
    """Select and permanently apply the best safe FDI-style initialization scale."""

    liq = _liquid_cfg(config)
    warnings: list[str] = []
    probe_batches = _materialize_probe_batches(train_loader, int(liq.fdi_probe_batches))
    if len(probe_batches) < int(liq.fdi_probe_batches):
        warnings.append(
            "FDI calibration received fewer probe batches than requested: "
            f"{len(probe_batches)} < {int(liq.fdi_probe_batches)}"
        )

    original_state = _state_dict_clone(model)
    was_training = model.training
    input_scales, recurrent_scales, threshold_scales, skipped = _candidate_scales(
        model, config, warnings
    )
    all_candidates: list[dict] = []

    try:
        for input_scale, recurrent_scale, threshold_scale in itertools.product(
            input_scales,
            recurrent_scales,
            threshold_scales,
        ):
            candidate = {
                "input_scale": float(input_scale),
                "recurrent_scale": float(recurrent_scale),
                "threshold_scale": float(threshold_scale),
            }
            _restore_state(model, original_state)
            applied, candidate_warnings = _apply_candidate(model, candidate)
            if not applied:
                warnings.extend(candidate_warnings)
                all_candidates.append(
                    {
                        "candidate": candidate,
                        "score": 1.0e30,
                        "stats": {},
                        "warnings": candidate_warnings,
                        "skipped": True,
                    }
                )
                continue

            stats = collect_initial_regime_stats(model, probe_batches, config, device)
            score = score_initial_regime(stats, config)
            all_candidates.append(
                {
                    "candidate": candidate,
                    "score": score,
                    "stats": stats,
                    "warnings": candidate_warnings,
                }
            )

        valid_candidates = [
            item
            for item in all_candidates
            if item.get("stats") and math.isfinite(float(item["score"]))
        ]
        if not valid_candidates:
            _restore_state(model, original_state)
            raise RuntimeError("FDI calibration found no valid candidate to apply")

        selected = min(valid_candidates, key=lambda item: float(item["score"]))
        selected_candidate = selected["candidate"]
        selected_stats = selected["stats"]
        selected_score = float(selected["score"])
        selected_constraint_warnings = _constraint_warnings(selected_stats, config)
        if selected_constraint_warnings:
            warnings.extend(selected_constraint_warnings)

        report = {
            "selected_candidate": selected_candidate,
            "selected_score": selected_score,
            "selected_stats": selected_stats,
            "all_candidates": all_candidates,
            "skipped_scale_dimensions": skipped,
            "warnings": warnings,
        }

        if selected_constraint_warnings and bool(
            getattr(liq, "fdi_strict_mode", False)
        ):
            _restore_state(model, original_state)
            _write_report(report, output_dir)
            raise RuntimeError(
                "FDI calibration strict mode failed: "
                + "; ".join(selected_constraint_warnings)
            )

        _restore_state(model, original_state)
        applied, selected_apply_warnings = _apply_candidate(model, selected_candidate)
        if not applied:
            _restore_state(model, original_state)
            warnings.extend(selected_apply_warnings)
            report["warnings"] = warnings
            _write_report(report, output_dir)
            raise RuntimeError(
                "FDI calibration selected candidate could not be applied: "
                + "; ".join(selected_apply_warnings)
            )

        _write_report(report, output_dir)
        _print_summary(report)
        return report
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
