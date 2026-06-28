"""Sanity-check recurrent ablation modes without running training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lsm.trainer import build_model
from src.utils.config import Config


MODE_SPECS = {
    "A_current_learned_lowrank": {
        "mode": "learned_lowrank",
        "train_w_raw": False,
        "expect_topology_grad": True,
        "expect_w_raw_grad": False,
    },
    "B_random_floor": {
        "mode": "random_sparse",
        "train_w_raw": False,
        "expect_topology_grad": False,
        "expect_w_raw_grad": False,
    },
    "C_softplus_w_only": {
        "mode": "softplus_w_only",
        "train_w_raw": True,
        "expect_topology_grad": False,
        "expect_w_raw_grad": True,
    },
    "D_smooth_lowrank": {
        "mode": "smooth_lowrank_conductance",
        "train_w_raw": False,
        "match_initial_w_eff_scale": True,
        "expect_topology_grad": True,
        "expect_w_raw_grad": False,
    },
    "E_edgewise_soft": {
        "mode": "edgewise_soft_conductance",
        "train_w_raw": False,
        "theta_init_mean": -2.25,
        "theta_init_std": 0.01,
        "expect_topology_grad": True,
        "expect_w_raw_grad": False,
    },
    "F_fixed_random_learned_w": {
        "mode": "random_sparse",
        "train_w_raw": True,
        "expect_topology_grad": False,
        "expect_w_raw_grad": True,
    },
    "G_lowrank_frozen_constant_g": {
        "mode": "learned_lowrank_frozen_w",
        "train_w_raw": False,
        "frozen_w_mode": "constant_g",
        "expect_topology_grad": True,
        "expect_w_raw_grad": False,
    },
    "H_lowrank_frozen_initialized_w": {
        "mode": "learned_lowrank_frozen_w",
        "train_w_raw": False,
        "frozen_w_mode": "initialized_w",
        "expect_topology_grad": True,
        "expect_w_raw_grad": False,
    },
    "SG_lowrank": {
        "mode": "soft_gate_lowrank",
        "train_w_raw": False,
        "expect_topology_grad": True,
        "expect_w_raw_grad": False,
    },
    "SG_edgewise": {
        "mode": "soft_gate_edgewise",
        "train_w_raw": False,
        "expect_topology_grad": True,
        "expect_w_raw_grad": False,
    },
}


def _make_cfg(spec: dict) -> Config:
    cfg = Config(dataset="shd", n_input=8, n_output=3, T=5, batch_size=2)
    cfg.liquid.n_liquid = 7
    cfg.liquid.neuron_type = "alif"
    cfg.liquid.readout_mode = "spike_adaptation_concat"
    cfg.liquid.recurrent_mode = spec["mode"]
    cfg.liquid.recurrent_sparsity = 0.1
    cfg.liquid.theta_rank = 3
    cfg.liquid.theta_init_mean = float(spec.get("theta_init_mean", -1.0))
    cfg.liquid.theta_init_std = float(spec.get("theta_init_std", 0.2))
    cfg.liquid.theta_lowrank_init_std = 0.2
    cfg.liquid.w_raw_init_mean = -2.25
    cfg.liquid.w_raw_init_std = 0.01
    cfg.liquid.w_raw_max = -2.0
    cfg.liquid.train_w_raw = bool(spec["train_w_raw"])
    cfg.liquid.noise_scale = 0.0 if str(spec["mode"]).startswith("soft_gate") else cfg.liquid.noise_scale
    cfg.liquid.temp_init = 1.0
    cfg.liquid.temp_final = 0.2
    cfg.liquid.target_density_init = 0.3
    cfg.liquid.target_density_final = 0.05
    cfg.liquid.target_anneal_epochs = 40
    cfg.liquid.density_penalty_lambda = 1.0 if str(spec["mode"]).startswith("soft_gate") else 0.0
    cfg.liquid.match_initial_w_eff_scale = bool(
        spec.get("match_initial_w_eff_scale", False)
    )
    cfg.liquid.frozen_w_mode = str(spec.get("frozen_w_mode", "initialized_w"))
    cfg.liquid.self_connection = False
    cfg.liquid.input_projection_mode = "learned_sparse"
    cfg.liquid.train_input_projection = True
    cfg.liquid.alif_rho_init = 0.85
    cfg.liquid.alif_beta_init = 0.10
    cfg.liquid.alif_adapt_increment = 0.125
    cfg.liquid.threshold_min = 0.1
    cfg.liquid.threshold_max = 0.2
    return cfg


def _grad_exists(params: list[torch.nn.Parameter]) -> bool:
    return any(param.grad is not None and torch.isfinite(param.grad).all() for param in params)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _run_one(name: str, spec: dict) -> None:
    torch.manual_seed(123)
    cfg = _make_cfg(spec)
    model = build_model(cfg, torch.device("cpu"))
    liquid = model.liquid
    model.train()
    if liquid.mode.startswith("soft_gate"):
        init_density = liquid.soft_gate_density().item()
        _check(abs(init_density - cfg.liquid.target_density_init) < 2e-3, "soft-gate init density missed target")
        deterministic_gate = liquid._soft_gate_gate().detach().clone()
        liquid.sample_epoch_mask(tau=0.01, epoch_noise=torch.randn_like(liquid.get_theta()) * 100.0)
        _check(torch.allclose(liquid.sample_mask().detach(), deterministic_gate), "soft-gate used epoch noise")
        liquid.unlock_epoch_mask()
    liquid.sample_mask()
    w_eff = liquid.get_effective_weight()
    _check(tuple(w_eff.shape) == (cfg.liquid.n_liquid, cfg.liquid.n_liquid), "bad W_eff shape")
    _check(torch.isfinite(w_eff).all().item(), "W_eff contains NaN/Inf")
    _check(torch.allclose(torch.diag(w_eff), torch.zeros(cfg.liquid.n_liquid)), "W_eff diagonal is not zero")

    x = (torch.rand(cfg.batch_size, cfg.T, cfg.n_input) < 0.7).float()
    logits = model(x)
    _check(tuple(logits.shape) == (cfg.batch_size, cfg.n_output), "bad forward shape")
    _check(torch.isfinite(logits).all().item(), "forward produced NaN/Inf")
    model.zero_grad(set_to_none=True)
    logits.square().mean().backward()

    model.zero_grad(set_to_none=True)
    liquid.sample_mask()
    recurrent_loss = liquid(torch.ones(cfg.batch_size, cfg.liquid.n_liquid)).square().mean()
    if recurrent_loss.requires_grad:
        recurrent_loss.backward()

    topology_params = liquid.topology_parameters()
    topology_grad = _grad_exists(topology_params)
    w_raw_grad = liquid.w_raw.grad is not None and torch.isfinite(liquid.w_raw.grad).all().item()
    _check(topology_grad == bool(spec["expect_topology_grad"]), f"unexpected topology grad: {topology_grad}")
    _check(w_raw_grad == bool(spec["expect_w_raw_grad"]), f"unexpected w_raw grad: {w_raw_grad}")

    if liquid.mode.startswith("soft_gate"):
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            if liquid.mode == "soft_gate_lowrank":
                liquid.theta_bias.add_(-20.0)
            else:
                liquid.theta_offset.add_(-20.0)
        liquid.sample_mask()
        _check(float(liquid.current_mask.max().item()) < 1e-2, "negative score did not close gate")
        _check(float(liquid.get_effective_weight().abs().max().item()) < 1e-3, "negative score did not suppress W_eff")
        liquid.get_effective_weight().abs().sum().backward()
        neg_grad = sum(
            0.0 if param.grad is None else float(param.grad.detach().abs().sum().item())
            for param in topology_params
        )
        _check(neg_grad > 0.0, "negative soft-gate score produced zero topology gradient")

    trainable = [param_name for param_name, param in model.named_parameters() if param.requires_grad]
    print(f"\n[{name}]")
    print(f"recurrent_mode={liquid.mode}")
    print(f"trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"w_raw_trainable={liquid.w_raw.requires_grad}")
    print(f"topology_trainable={any(p.requires_grad for p in topology_params)}")
    print(f"W_eff_shape={tuple(w_eff.shape)} finite=True diag_zero=True")
    print(f"forward_success=True backward_success=True")
    print("requires_grad names:")
    for param_name in trainable:
        print(f"  {param_name}")


def main() -> int:
    for name, spec in MODE_SPECS.items():
        _run_one(name, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
