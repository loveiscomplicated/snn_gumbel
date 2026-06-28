"""
Smoke check for the ALIFReservoirBlock refactor.

This script intentionally uses a dummy batch and does not run training.
It validates reference config construction, forward contracts, trace semantics,
diagnostics, and state_dict key compatibility.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lsm.model import ALIFReservoirBlock
from src.lsm.trainer import build_model
from src.utils.config import load_config


REFERENCE_CONFIG = (
    "configs/"
    "lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_"
    "spike_adaptation_b010_inc0125_biaslr05.yaml"
)

TRACE_KEYS = (
    "spikes",
    "membrane",
    "input_current",
    "recurrent_current",
    "adaptation",
    "theta_eff",
)

DIAGNOSTIC_KEYS = (
    "mean_spike_rate",
    "max_spike_rate",
    "adaptation_mean",
    "adaptation_max",
    "membrane_mean",
    "membrane_max",
    "recurrent_current_abs_mean",
    "recurrent_current_abs_max",
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_finite_tensor(name: str, value: torch.Tensor) -> None:
    _check(torch.isfinite(value).all().item(), f"{name} contains NaN or Inf")


def _check_finite_diagnostics(diagnostics: dict) -> None:
    missing = sorted(set(DIAGNOSTIC_KEYS).difference(diagnostics))
    _check(not missing, f"missing diagnostics keys: {missing}")
    for key in DIAGNOSTIC_KEYS:
        value = diagnostics[key]
        _check(
            isinstance(value, (float, int)) and math.isfinite(float(value)),
            f"diagnostic {key} is not a finite scalar: {value!r}",
        )


def main() -> int:
    torch.manual_seed(0)
    cfg = load_config(REFERENCE_CONFIG)
    device = torch.device("cpu")
    model = build_model(cfg, device)
    model.eval()

    _check(
        cfg.liquid.neuron_type == "alif",
        f"reference config neuron_type changed: {cfg.liquid.neuron_type!r}",
    )
    _check(
        cfg.liquid.readout_mode == "spike_adaptation_concat",
        f"reference config readout_mode changed: {cfg.liquid.readout_mode!r}",
    )
    _check(
        isinstance(model.alif_reservoir, ALIFReservoirBlock),
        "model.alif_reservoir is not an ALIFReservoirBlock",
    )

    batch_size = 2
    x = torch.zeros(batch_size, cfg.T, cfg.n_input, device=device)

    with torch.no_grad():
        logits = model(x, tau=cfg.tau_end)
        trace_logits, traces = model(x, tau=cfg.tau_end, return_traces=True)
        diag_logits, diagnostics = model(
            x,
            tau=cfg.tau_end,
            return_diagnostics=True,
        )
        combined = model(
            x,
            tau=cfg.tau_end,
            return_traces=True,
            return_diagnostics=True,
        )

    _check(tuple(logits.shape) == (batch_size, cfg.n_output), "bad logits shape")
    _check_finite_tensor("logits", logits)
    _check_finite_tensor("trace_logits", trace_logits)
    _check_finite_tensor("diag_logits", diag_logits)
    _check(torch.allclose(logits, trace_logits), "return_traces changed logits")
    _check(torch.allclose(logits, diag_logits), "return_diagnostics changed logits")

    _check(isinstance(combined, tuple) and len(combined) == 3, "bad combined return")
    combined_logits, combined_traces, combined_diagnostics = combined
    _check_finite_tensor("combined_logits", combined_logits)
    _check(torch.allclose(logits, combined_logits), "combined return changed logits")

    expected_trace_shape = (batch_size, cfg.T, cfg.liquid.n_liquid)
    missing_trace_keys = sorted(set(TRACE_KEYS).difference(traces))
    _check(not missing_trace_keys, f"missing trace keys: {missing_trace_keys}")
    for key in TRACE_KEYS:
        _check(
            tuple(traces[key].shape) == expected_trace_shape,
            f"trace {key} has shape {tuple(traces[key].shape)}, expected {expected_trace_shape}",
        )
        _check_finite_tensor(f"trace {key}", traces[key])
        _check(
            key in combined_traces
            and tuple(combined_traces[key].shape) == expected_trace_shape,
            f"combined trace {key} shape/key mismatch",
        )

    _check_finite_diagnostics(diagnostics)
    _check_finite_diagnostics(combined_diagnostics)

    state_keys = list(model.state_dict())
    duplicated_reservoir_keys = [
        key
        for key in state_keys
        if key.startswith("alif_reservoir.") or ".alif_reservoir." in key
    ]
    has_duplicated_reservoir_keys = bool(duplicated_reservoir_keys)
    _check(
        not has_duplicated_reservoir_keys,
        f"state_dict contains duplicated reservoir keys: {duplicated_reservoir_keys}",
    )

    print(f"config: {REFERENCE_CONFIG}")
    print(f"logits shape: {tuple(logits.shape)} finite=True")
    print("trace keys and shapes:")
    for key in TRACE_KEYS:
        print(f"  {key}: {tuple(traces[key].shape)}")
    print("diagnostics:")
    for key in DIAGNOSTIC_KEYS:
        print(f"  {key}: {diagnostics[key]:.6g}")
    print(f"state_dict keys: {len(state_keys)}")
    print(f"duplicated reservoir parameter keys: {has_duplicated_reservoir_keys}")
    print("ALIF reservoir refactor smoke check: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ALIF reservoir refactor smoke check: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
