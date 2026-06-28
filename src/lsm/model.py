"""
LSM model: InputProjection → LiquidLayer (recurrent) → Readout.

Liquid topology modes:
  - "learned"       : Gumbel-Sigmoid mask, trained end-to-end
  - "learned_lowrank": Gumbel-Sigmoid mask with directed low-rank theta
  - "learned_lowrank_frozen_w": learned_lowrank mask with frozen conductance
  - "softplus_w_only": dense softplus(w_raw) conductance, no topology
  - "edgewise_soft_conductance": independent softplus(theta_ij) conductance
  - "smooth_lowrank_conductance": softplus(lowrank logit) conductance
  - "soft_gate_lowrank": differentiable gate*conductance from low-rank score
  - "soft_gate_edgewise": differentiable gate*conductance from edge score
  - "random_sparse"  : fixed random binary mask at init
  - "fixed"          : random sparse + weights frozen (traditional LSM)
  - "grad_r"         : hard threshold (theta > 0) mask

Current learned_lowrank baseline:
  W_eff = mask_lowrank * self_conn_mask * dale_sign * softplus(clamp(w_raw)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import sigmoid_ste, spike_fn


LOWRANK_MODES = {
    "learned_lowrank",
    "learned_lowrank_frozen_w",
    "smooth_lowrank_conductance",
    "soft_gate_lowrank",
}
HARD_TOPOLOGY_MODES = {"learned", "learned_lowrank", "learned_lowrank_frozen_w"}
SOFT_GATE_MODES = {"soft_gate_lowrank", "soft_gate_edgewise"}
SOFT_CONDUCTANCE_MODES = {
    "softplus_w_only",
    "edgewise_soft_conductance",
    "smooth_lowrank_conductance",
}
NO_W_RAW_CONDUCTANCE_MODES = {
    "edgewise_soft_conductance",
    "smooth_lowrank_conductance",
    "soft_gate_lowrank",
    "soft_gate_edgewise",
}
VALID_RECURRENT_MODES = {
    "learned",
    "learned_lowrank",
    "learned_lowrank_frozen_w",
    "softplus_w_only",
    "edgewise_soft_conductance",
    "smooth_lowrank_conductance",
    "soft_gate_lowrank",
    "soft_gate_edgewise",
    "random_sparse",
    "fixed",
    "grad_r",
}


# ---------------------------------------------------------------------------
# InputProjection: sparse random mixed-sign input → liquid
# ---------------------------------------------------------------------------


class InputProjection(nn.Module):
    """Sparse mixed-sign connections from input to liquid."""

    def __init__(
        self,
        n_input: int,
        n_liquid: int,
        p_input: float = 0.1,
        weight_scale: float = 0.1,
        mode: str = "fixed_sparse",
        trainable: bool = False,
    ):
        super().__init__()
        if mode not in {"fixed_sparse", "learned_sparse"}:
            raise ValueError(
                "input projection mode must be one of: fixed_sparse, learned_sparse; "
                f"got {mode!r}"
            )
        self.mode = mode
        mask = (torch.rand(n_input, n_liquid) < p_input).float()
        weight = torch.randn(n_input, n_liquid) * weight_scale * mask
        self.register_buffer("mask", mask)
        if mode == "fixed_sparse":
            self.register_buffer("weight", weight)
        else:
            self.weight = nn.Parameter(weight, requires_grad=bool(trainable))

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.mask

    def effective_density(self) -> float:
        with torch.no_grad():
            return float((self.mask != 0).float().mean().item())

    def effective_weight_norm(self) -> float:
        with torch.no_grad():
            return float(self.effective_weight().norm().item())

    def grad_norm(self) -> float:
        grad = getattr(self.weight, "grad", None)
        if grad is None:
            return 0.0
        return float(grad.norm().item())

    @property
    def trainable(self) -> bool:
        return isinstance(self.weight, nn.Parameter) and self.weight.requires_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.effective_weight()


# ---------------------------------------------------------------------------
# LiquidLayer: recurrent connections with Gumbel mask + Dale's Law
# ---------------------------------------------------------------------------


class LiquidLayer(nn.Module):
    """
    Recurrent liquid layer with topology learning.

    Parameters learned: topology logits, w_raw (N,N), threshold (N,), log_beta
    Buffers (fixed): dale_sign (N,1), self_conn_mask (N,N)
    """

    def __init__(
        self,
        n_liquid: int,
        exc_ratio: float = 0.8,
        neuron_type: str = "lif",
        mode: str = "learned",
        target_sparsity: float = 0.2,
        self_connection: bool = False,
        theta_init_mean: float = 0.0,
        theta_init_std: float = 0.01,
        theta_rank: int = 16,
        theta_lowrank_init_std: float = 0.30,
        w_raw_init_mean: float = -4.0,
        w_raw_init_std: float = 0.01,
        train_w_raw: bool = True,
        w_raw_max: float = -1.0,
        recurrent_weight_scale: float = 1.0,
        match_initial_w_eff_scale: bool = False,
        frozen_w_mode: str = "initialized_w",
        frozen_w_constant_g: float | None = None,
        soft_gate_temp_init: float = 1.0,
        soft_gate_target_density_init: float = 0.3,
        mag_from_separate_param: bool = False,
        beta_min: float = 0.7,
        beta_max: float = 0.95,
        threshold_min: float = 0.8,
        threshold_max: float = 1.5,
        alif_rho_init: float = 0.9,
        alif_beta_init: float = 0.4,
        alif_adapt_increment: float = 1.0,
        alif_learn_rho: bool = False,
        alif_learn_beta: bool = False,
        noise_scale: float = 0.1,
    ):
        super().__init__()
        self.n_liquid = n_liquid
        self.neuron_type = neuron_type
        self.mode = mode
        self.w_raw_max = w_raw_max
        self.noise_scale = noise_scale
        self.frozen_w_mode = frozen_w_mode
        self.mag_from_separate_param = bool(mag_from_separate_param)

        if mode not in VALID_RECURRENT_MODES:
            raise ValueError(
                "Unknown liquid mode: "
                f"{mode!r}; expected one of {sorted(VALID_RECURRENT_MODES)}"
            )
        if neuron_type not in {"lif", "alif"}:
            raise ValueError(f"Unknown neuron_type: {neuron_type}")
        if frozen_w_mode not in {"initialized_w", "constant_g"}:
            raise ValueError(
                "frozen_w_mode must be one of: initialized_w, constant_g; "
                f"got {frozen_w_mode!r}"
            )
        if mode != "learned_lowrank_frozen_w" and frozen_w_mode != "initialized_w":
            raise ValueError(
                "frozen_w_mode is only supported for learned_lowrank_frozen_w."
            )
        if float(soft_gate_temp_init) <= 0.0:
            raise ValueError(
                f"soft_gate_temp_init must be positive, got {soft_gate_temp_init}"
            )
        if not 0.0 < float(soft_gate_target_density_init) < 1.0:
            raise ValueError(
                "soft_gate_target_density_init must be in (0, 1), "
                f"got {soft_gate_target_density_init}"
            )
        if not 0.0 <= alif_rho_init < 1.0:
            raise ValueError(f"alif_rho_init must be in [0, 1), got {alif_rho_init}")
        if alif_adapt_increment < 0.0:
            raise ValueError(
                "alif_adapt_increment must be non-negative, "
                f"got {alif_adapt_increment}"
            )

        # --- learnable parameters ---
        weight_trainable = mode != "fixed"
        if mode == "learned":
            self.theta = nn.Parameter(
                torch.randn(n_liquid, n_liquid) * theta_init_std + theta_init_mean,
                requires_grad=True,
            )
        elif mode in LOWRANK_MODES:
            self.src_embed = nn.Parameter(
                torch.randn(n_liquid, theta_rank) * theta_lowrank_init_std
            )
            self.dst_embed = nn.Parameter(
                torch.randn(n_liquid, theta_rank) * theta_lowrank_init_std
            )
            self.theta_bias = nn.Parameter(torch.tensor(float(theta_init_mean)))
        elif mode == "edgewise_soft_conductance":
            self.theta = nn.Parameter(
                torch.randn(n_liquid, n_liquid) * theta_init_std + theta_init_mean,
                requires_grad=True,
            )
        elif mode == "soft_gate_edgewise":
            self.theta = nn.Parameter(
                torch.randn(n_liquid, n_liquid) * theta_init_std,
                requires_grad=True,
            )
            self.theta_offset = nn.Parameter(torch.tensor(float(theta_init_mean)))
        else:
            self.theta = nn.Parameter(
                torch.randn(n_liquid, n_liquid) * theta_init_std + theta_init_mean,
                requires_grad=(mode == "grad_r"),
            )
        # softplus(w_raw) is the weight magnitude.
        # softplus(0)=0.693 is way too large for recurrent nets.
        # With N=200, p=0.2: ~40 inputs/neuron, 80% exc.
        # softplus(-4.0)≈0.018 → recurrent current ≈ 0.58 (sub-threshold)
        w_raw_requires_grad = (
            weight_trainable
            and train_w_raw
            and mode != "learned_lowrank_frozen_w"
            and mode not in NO_W_RAW_CONDUCTANCE_MODES
        )
        self.w_raw = nn.Parameter(
            torch.randn(n_liquid, n_liquid) * w_raw_init_std + w_raw_init_mean,
            requires_grad=w_raw_requires_grad,
        )
        if self.mode in SOFT_GATE_MODES and self.mag_from_separate_param:
            self.w_core = nn.Parameter(
                torch.randn(n_liquid, n_liquid) * w_raw_init_std + w_raw_init_mean,
                requires_grad=True,
            )
        else:
            self.w_core = None
        # shuffle so beta/threshold are not correlated with E/I neuron ordering
        beta_vals = torch.linspace(beta_min, beta_max, n_liquid)
        beta_vals = beta_vals[torch.randperm(n_liquid)]
        self.logit_beta = nn.Parameter(
            torch.log(beta_vals / (1.0 - beta_vals)), requires_grad=weight_trainable
        )
        thr_vals = torch.linspace(threshold_min, threshold_max, n_liquid)
        thr_vals = thr_vals[torch.randperm(n_liquid)]
        self.threshold = nn.Parameter(thr_vals, requires_grad=weight_trainable)

        if neuron_type == "alif":
            self.register_buffer(
                "alif_adapt_increment", torch.tensor(float(alif_adapt_increment))
            )
            if alif_learn_rho:
                rho_init = torch.full((n_liquid,), float(alif_rho_init))
                rho_init = torch.clamp(rho_init, 1e-6, 1.0 - 1e-6)
                self.alif_rho_param = nn.Parameter(torch.logit(rho_init))
            else:
                self.register_buffer(
                    "alif_rho_buffer", torch.tensor(float(alif_rho_init))
                )
                self.alif_rho_param = None
            if alif_learn_beta:
                beta_init = torch.full((n_liquid,), float(alif_beta_init))
                self.alif_beta_param = nn.Parameter(beta_init)
            else:
                self.register_buffer(
                    "alif_beta_buffer", torch.tensor(float(alif_beta_init))
                )
                self.alif_beta_param = None
        else:
            self.alif_rho_param = None
            self.alif_beta_param = None

        # --- Dale's Law: exc (+1) / inh (-1) sign buffer ---
        n_exc = int(exc_ratio * n_liquid)
        dale_sign = torch.ones(n_liquid, 1)
        dale_sign[n_exc:] = -1.0
        self.register_buffer("dale_sign", dale_sign)

        # --- self-connection mask: diagonal = 0 ---
        if self_connection:
            self_conn_mask = torch.ones(n_liquid, n_liquid)
        else:
            self_conn_mask = 1.0 - torch.eye(n_liquid)
        self.register_buffer("self_conn_mask", self_conn_mask)
        self.register_buffer("density_mask", 1.0 - torch.eye(n_liquid))
        self.register_buffer(
            "recurrent_weight_scale",
            torch.tensor(float(recurrent_weight_scale)),
        )
        self.register_buffer(
            "soft_gate_temp",
            torch.tensor(float(soft_gate_temp_init)),
        )

        # --- fixed mask for random_sparse / fixed modes ---
        if mode in ("random_sparse", "fixed"):
            # Generate a sparse binary mask with a density of target_sparsity
            mask = (torch.rand(n_liquid, n_liquid) < target_sparsity).float()
            mask = (
                mask * self_conn_mask
            )  # respect self-connection setting & torch tensor * means element wise mul.
            self.register_buffer("fixed_mask", mask)
        else:
            self.register_buffer("fixed_mask", None)

        # cached mask for current simulation
        self.current_mask: torch.Tensor | None = None
        self.register_buffer(
            "frozen_w_constant_g",
            torch.tensor(0.0 if frozen_w_constant_g is None else float(frozen_w_constant_g)),
        )
        if mode == "learned_lowrank_frozen_w" and frozen_w_mode == "constant_g":
            if frozen_w_constant_g is None:
                self._initialize_frozen_w_constant_g()
        if mode == "smooth_lowrank_conductance" and match_initial_w_eff_scale:
            self._match_initial_lowrank_conductance_scale()
        if mode in SOFT_GATE_MODES:
            self._initialize_soft_gate_density(
                target_density=float(soft_gate_target_density_init),
                temp=float(soft_gate_temp_init),
            )
        # epoch-level Gumbel noise (Phase 2): stored here, STE recomputed each batch
        self._epoch_noise: torch.Tensor | None = None
        self._epoch_tau: float = 1.0

    @property
    def beta(self):
        return torch.sigmoid(self.logit_beta)

    @property
    def alif_rho(self):
        if self.neuron_type != "alif":
            raise RuntimeError("ALIF rho requested for non-ALIF liquid layer")
        if self.alif_rho_param is not None:
            return torch.sigmoid(self.alif_rho_param)
        return self.alif_rho_buffer

    @property
    def alif_beta(self):
        if self.neuron_type != "alif":
            raise RuntimeError("ALIF beta requested for non-ALIF liquid layer")
        if self.alif_beta_param is not None:
            return self.alif_beta_param.clamp(min=0.0)
        return self.alif_beta_buffer.clamp(min=0.0)

    def get_theta(self) -> torch.Tensor:
        if self.mode == "learned":
            return self.theta
        if self.mode in LOWRANK_MODES:
            return self.src_embed @ self.dst_embed.T + self.theta_bias
        if self.mode == "edgewise_soft_conductance":
            return self.theta
        if self.mode == "soft_gate_edgewise":
            return self.theta + self.theta_offset
        raise RuntimeError(f"Topology logits are not defined for mode: {self.mode}")

    def topology_parameters(self) -> list[nn.Parameter]:
        if self.mode in ("learned", "grad_r", "edgewise_soft_conductance"):
            return [self.theta]
        if self.mode in LOWRANK_MODES:
            return [self.src_embed, self.dst_embed, self.theta_bias]
        if self.mode == "soft_gate_edgewise":
            return [self.theta, self.theta_offset]
        return []

    def set_topology_requires_grad(self, requires_grad: bool) -> None:
        for param in self.topology_parameters():
            param.requires_grad_(requires_grad)

    def topology_state_dict(self) -> dict[str, torch.Tensor]:
        if self.mode in LOWRANK_MODES:
            return {
                "src_embed": self.src_embed.detach().clone(),
                "dst_embed": self.dst_embed.detach().clone(),
                "theta_bias": self.theta_bias.detach().clone(),
            }
        if self.mode in ("learned", "grad_r", "edgewise_soft_conductance"):
            return {"theta": self.theta.detach().clone()}
        if self.mode == "soft_gate_edgewise":
            return {
                "theta": self.theta.detach().clone(),
                "theta_offset": self.theta_offset.detach().clone(),
            }
        return {}

    def load_topology_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if self.mode in LOWRANK_MODES:
            required_keys = {"src_embed", "dst_embed", "theta_bias"}
        elif self.mode in ("learned", "grad_r", "edgewise_soft_conductance"):
            required_keys = {"theta"}
        elif self.mode == "soft_gate_edgewise":
            required_keys = {"theta", "theta_offset"}
        else:
            if state:
                raise ValueError(
                    f"Topology state is not supported for mode {self.mode!r}, got keys {sorted(state)}."
                )
            return

        missing_keys = required_keys.difference(state)
        if missing_keys:
            raise ValueError(
                f"Missing topology state keys for mode {self.mode!r}: {sorted(missing_keys)}"
            )

        with torch.no_grad():
            if self.mode in LOWRANK_MODES:
                self.src_embed.copy_(state["src_embed"])
                self.dst_embed.copy_(state["dst_embed"])
                self.theta_bias.copy_(state["theta_bias"])
            elif self.mode == "soft_gate_edgewise":
                self.theta.copy_(state["theta"])
                self.theta_offset.copy_(state["theta_offset"])
            else:
                self.theta.copy_(state["theta"])

    def _w_raw_conductance(self) -> torch.Tensor:
        return F.softplus(torch.clamp(self.w_raw, max=self.w_raw_max))

    def _lowrank_conductance(self) -> torch.Tensor:
        return F.softplus(self.get_theta())

    def _edgewise_conductance(self) -> torch.Tensor:
        return F.softplus(self.theta)

    def _hard_lowrank_mask(self) -> torch.Tensor:
        theta = self.get_theta()
        return (torch.sigmoid(theta) >= 0.5).float()

    def set_soft_gate_temperature(self, temp: float) -> None:
        if self.mode not in SOFT_GATE_MODES:
            return
        if float(temp) <= 0.0:
            raise ValueError(f"soft-gate temperature must be positive, got {temp}")
        self.soft_gate_temp.copy_(
            torch.as_tensor(float(temp), device=self.soft_gate_temp.device)
        )

    def _soft_gate_gate(
        self,
        score: torch.Tensor | None = None,
        temp: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if score is None:
            score = self.get_theta()
        if temp is None:
            temp_t = self.soft_gate_temp.to(device=score.device, dtype=score.dtype)
        else:
            temp_t = torch.as_tensor(temp, device=score.device, dtype=score.dtype)
        return torch.sigmoid(score / temp_t.clamp_min(torch.finfo(score.dtype).eps))

    def _soft_gate_mag(self, score: torch.Tensor | None = None) -> torch.Tensor:
        if self.mag_from_separate_param:
            if self.w_core is None:
                raise RuntimeError("mag_from_separate_param=true but w_core is missing")
            return F.softplus(self.w_core)
        if score is None:
            score = self.get_theta()
        return F.softplus(score)

    def soft_gate_components(
        self,
        *,
        temp: float | torch.Tensor | None = None,
        use_current_gate: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode not in SOFT_GATE_MODES:
            raise RuntimeError(f"soft-gate components are not defined for {self.mode!r}")
        score = self.get_theta()
        if use_current_gate and self.current_mask is not None:
            gate = self.current_mask
        else:
            gate = self._soft_gate_gate(score, temp=temp)
        mag = self._soft_gate_mag(score)
        return score, gate, mag

    def soft_gate_density(self, gate: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode not in SOFT_GATE_MODES:
            raise RuntimeError(f"soft density is not defined for mode {self.mode!r}")
        if gate is None:
            gate = (
                self.current_mask
                if self.current_mask is not None
                else self._soft_gate_gate()
            )
        mask = self.density_mask.to(device=gate.device, dtype=gate.dtype)
        return (gate * mask).sum() / mask.sum().clamp_min(1.0)

    def soft_gate_density_penalty(
        self, target_density: float | torch.Tensor
    ) -> torch.Tensor:
        density = self.soft_gate_density()
        target = torch.as_tensor(
            target_density, device=density.device, dtype=density.dtype
        )
        return (density - target) ** 2

    def soft_gate_stats(
        self,
        *,
        target_density: float | None = None,
        hard_eps: float = 1e-12,
    ) -> dict[str, float]:
        if self.mode not in SOFT_GATE_MODES:
            return {}
        with torch.no_grad():
            score, gate, mag = self.soft_gate_components()
            w_eff = (
                self.recurrent_weight_scale
                * self.self_conn_mask
                * self.dale_sign
                * gate
                * mag
            ).detach().float()
            mask = self.density_mask.detach().bool()
            score_v = score.detach().float()[mask]
            gate_v = gate.detach().float()[mask]
            mag_v = mag.detach().float()[mask]
            w_v = w_eff[mask]
            out = {
                "soft_density": float(gate_v.mean().item()) if gate_v.numel() else 0.0,
                "hard_active_fraction": (
                    float((w_v.abs() > hard_eps).float().mean().item())
                    if w_v.numel()
                    else 0.0
                ),
                "soft_gate_temp": float(self.soft_gate_temp.detach().cpu().item()),
                "score_mean": float(score_v.mean().item()) if score_v.numel() else 0.0,
                "score_std": (
                    float(score_v.std(unbiased=False).item())
                    if score_v.numel() > 1
                    else 0.0
                ),
                "gate_mean": float(gate_v.mean().item()) if gate_v.numel() else 0.0,
                "gate_p50": (
                    float(torch.quantile(gate_v, 0.50).item())
                    if gate_v.numel()
                    else 0.0
                ),
                "gate_p95": (
                    float(torch.quantile(gate_v, 0.95).item())
                    if gate_v.numel()
                    else 0.0
                ),
                "mag_mean": float(mag_v.mean().item()) if mag_v.numel() else 0.0,
                "mag_max": float(mag_v.max().item()) if mag_v.numel() else 0.0,
                "soft_gate_w_eff_mean": (
                    float(w_v.mean().item()) if w_v.numel() else 0.0
                ),
                "soft_gate_w_eff_abs_max": (
                    float(w_v.abs().max().item()) if w_v.numel() else 0.0
                ),
                "soft_gate_w_eff_fro_norm": float(w_eff.norm().item()),
            }
            if target_density is not None:
                target = float(target_density)
                out["target_density"] = target
                out["density_penalty"] = (out["soft_density"] - target) ** 2
            return out

    def _shift_soft_gate_score(self, delta: torch.Tensor) -> None:
        if self.mode == "soft_gate_lowrank":
            self.theta_bias.add_(
                delta.to(device=self.theta_bias.device, dtype=self.theta_bias.dtype)
            )
        elif self.mode == "soft_gate_edgewise":
            self.theta_offset.add_(
                delta.to(device=self.theta_offset.device, dtype=self.theta_offset.dtype)
            )
        else:
            raise RuntimeError(f"soft-gate score shift is not defined for {self.mode!r}")

    def _initialize_soft_gate_density(self, target_density: float, temp: float) -> None:
        if self.mode not in SOFT_GATE_MODES:
            return
        with torch.no_grad():
            valid = self.density_mask.bool()
            score0 = self.get_theta().detach()
            score0_valid = score0[valid]
            if score0_valid.numel() == 0:
                return
            q = torch.quantile(score0_valid.float(), 1.0 - float(target_density)).to(
                score0.device
            )
            self._shift_soft_gate_score(-q)

            # The quantile step makes score>0 match the desired hard fraction.
            # The density penalty, however, sees mean(sigmoid(score/temp)), so
            # finish with a scalar bisection shift that matches that actual tensor.
            lo = score0_valid.new_tensor(-80.0)
            hi = score0_valid.new_tensor(80.0)
            target = score0_valid.new_tensor(float(target_density))
            valid = valid.to(device=score0.device)
            for _ in range(80):
                mid = (lo + hi) * 0.5
                shifted_score = self.get_theta() + mid
                gate = self._soft_gate_gate(shifted_score, temp=float(temp))[valid]
                if gate.mean() < target:
                    lo = mid
                else:
                    hi = mid
            self._shift_soft_gate_score((lo + hi) * 0.5)

    def _initialize_frozen_w_constant_g(self) -> None:
        with torch.no_grad():
            valid = self.self_conn_mask.bool()
            active = (self._hard_lowrank_mask() * self.self_conn_mask).bool()
            conductance = self._w_raw_conductance()
            selected = conductance[active]
            if selected.numel() == 0:
                selected = conductance[valid]
            if selected.numel() == 0:
                value = conductance.new_tensor(0.0)
            else:
                value = selected.square().mean().sqrt()
            self.frozen_w_constant_g.copy_(value.detach())

    def _match_initial_lowrank_conductance_scale(self) -> None:
        with torch.no_grad():
            target = (
                self._hard_lowrank_mask()
                * self.self_conn_mask
                * self.dale_sign
                * self._w_raw_conductance()
            )
            raw = self.self_conn_mask * self.dale_sign * self._lowrank_conductance()
            raw_norm = raw.norm()
            if bool(torch.isfinite(raw_norm).item()) and raw_norm.item() > 1e-12:
                self.recurrent_weight_scale.copy_((target.norm() / raw_norm).detach())

    def freeze_topology(self) -> None:
        self.set_topology_requires_grad(False)
        for param in self.topology_parameters():
            param.grad = None
        self.unlock_epoch_mask()

    def sample_epoch_mask(self, tau: float, epoch_noise: torch.Tensor) -> None:
        """Store epoch-level Gumbel noise for Phase 2 training.

        Noise is fixed for the entire epoch so all batches share the same hard topology
        → BPTT gradients accumulate consistently → no explosion.
        Across epochs the noise changes → OFF edges occasionally flip ON → OFF edges
        get w_raw gradient → can be permanently promoted.

        Critically, the STE tensor is NOT stored here. sample_mask() recomputes it
        freshly each batch using this stored noise, so each batch gets its own graph
        that is safely freed after backward().

        Actual mask generation is done by sample_mask
        """
        self._epoch_noise = epoch_noise
        self._epoch_tau = tau

    def unlock_epoch_mask(self):
        """Clear epoch noise. Called before eval so eval uses deterministic mask."""
        self._epoch_noise = None

    def sample_mask(self, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """Compute mask for one forward pass.

        Phase 2 (epoch noise set): STE with fixed noise → same hard topology as all
            other batches this epoch, but a fresh computation graph each call.
        Phase 1 / eval: deterministic hard mask, no gradient.
        """
        if self.mode in HARD_TOPOLOGY_MODES:
            theta = self.get_theta()
            if self._epoch_noise is not None:
                # Phase 2: recompute STE with the epoch noise every batch.
                # Same noise → same hard{0,1} topology. New graph each call → backward safe.
                # noise_scale controls exploration radius:
                #   0.1 → only edges with |theta| < 0.18 can flip (~0.3% of all edges)
                #   1.0 → standard Gumbel, ~33% flip regardless of theta magnitude
                noisy_logits = (
                    theta / self._epoch_tau + self.noise_scale * self._epoch_noise
                )
                soft = torch.sigmoid(noisy_logits)
                hard_mask = (soft >= 0.5).float()
                self.current_mask = hard_mask - soft.detach() + soft
            elif self.training and any(
                param.requires_grad for param in self.topology_parameters()
            ):
                # Phase 2 fallback without noise (shouldn't be reached in normal flow)
                self.current_mask = sigmoid_ste(theta)
            else:
                # Phase 1 or eval: pure deterministic
                self.current_mask = (torch.sigmoid(theta) >= 0.5).float()
        elif self.mode in ("random_sparse", "fixed"):
            self.current_mask = self.fixed_mask
        elif self.mode == "grad_r":
            if self.training and self.theta.requires_grad:
                # STE: forward=hard threshold, backward=sigmoid gradient
                self.current_mask = sigmoid_ste(self.theta)
            else:
                self.current_mask = (self.theta > 0).float()
        elif self.mode in SOFT_GATE_MODES:
            self.current_mask = self._soft_gate_gate()
        elif self.mode in SOFT_CONDUCTANCE_MODES:
            self.current_mask = torch.ones_like(self.self_conn_mask)
        else:
            raise ValueError(f"Unknown liquid mode: {self.mode}")
        return self.current_mask

    def get_effective_weight(self) -> torch.Tensor:
        """Compute the effective recurrent matrix.

        Existing learned_lowrank baseline:
            W_eff = current_mask * self_conn_mask * dale_sign * softplus(clamp(w_raw)).
        New soft-conductance ablations replace only the conductance generator while
        keeping Dale sign and self-connection masking fixed.
        """
        if self.mode == "softplus_w_only":
            return self.self_conn_mask * self.dale_sign * self._w_raw_conductance()
        if self.mode == "edgewise_soft_conductance":
            return self.self_conn_mask * self.dale_sign * self._edgewise_conductance()
        if self.mode == "smooth_lowrank_conductance":
            return (
                self.recurrent_weight_scale
                * self.self_conn_mask
                * self.dale_sign
                * self._lowrank_conductance()
            )
        if self.mode in SOFT_GATE_MODES:
            if self.current_mask is None:
                self.sample_mask()
            score = self.get_theta()
            mag = self._soft_gate_mag(score)
            return (
                self.recurrent_weight_scale
                * self.self_conn_mask
                * self.dale_sign
                * self.current_mask
                * mag
            )
        if self.current_mask is None and self.mode not in SOFT_CONDUCTANCE_MODES:
            self.sample_mask()
        if self.mode == "learned_lowrank_frozen_w" and self.frozen_w_mode == "constant_g":
            return (
                self.current_mask
                * self.self_conn_mask
                * self.dale_sign
                * self.frozen_w_constant_g
            )
        return self.current_mask * self.self_conn_mask * self.dale_sign * self._w_raw_conductance()

    def forward(self, spike: torch.Tensor) -> torch.Tensor:
        """
        Compute recurrent current from liquid spikes.
        spike: (batch, N)
        returns: (batch, N)
        """
        w_eff = self.get_effective_weight()  # (N, N)
        # w_eff[i, j] = pre_i → post_j weight
        return spike @ w_eff

    def sparsity(self) -> float:
        with torch.no_grad():
            mask = self.get_binary_mask()
            return mask.mean().item()

    def get_binary_mask(self) -> torch.Tensor:
        if self.mode in ("random_sparse", "fixed"):
            return self.fixed_mask
        if self.mode == "grad_r":
            return (self.theta > 0).float() * self.self_conn_mask
        if self.mode in HARD_TOPOLOGY_MODES:
            theta = self.get_theta()
            return ((torch.sigmoid(theta) >= 0.5).float()) * self.self_conn_mask
        if self.mode == "softplus_w_only":
            return self.self_conn_mask
        if self.mode in ("edgewise_soft_conductance", "smooth_lowrank_conductance"):
            return (self.get_theta() > 0).float() * self.self_conn_mask
        if self.mode in SOFT_GATE_MODES:
            gate = self._soft_gate_gate()
            return (gate >= 0.5).float() * self.density_mask
        raise RuntimeError(f"Unknown liquid mode: {self.mode}")


@dataclass
class ALIFReservoirState:
    spike: torch.Tensor
    membrane: torch.Tensor
    adaptation: torch.Tensor
    recurrent_current: torch.Tensor
    membrane_pre_reset: torch.Tensor
    theta_eff: torch.Tensor


class ALIFReservoirBlock:
    """Plain wrapper for ALIF recurrent dynamics.

    This intentionally is not an nn.Module. LiquidLayer remains the sole owner of
    all parameters and buffers, which preserves existing state_dict key names.
    """

    def __init__(self, liquid: LiquidLayer):
        if liquid.neuron_type != "alif":
            raise ValueError(
                "ALIFReservoirBlock requires a LiquidLayer with neuron_type='alif'"
            )
        self.liquid = liquid

    @property
    def n_liquid(self) -> int:
        return self.liquid.n_liquid

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ALIFReservoirState:
        if device is None:
            device = self.liquid.threshold.device
        if dtype is None:
            dtype = self.liquid.threshold.dtype
        zeros = torch.zeros(
            batch_size,
            self.n_liquid,
            device=device,
            dtype=dtype,
        )
        return ALIFReservoirState(
            spike=zeros,
            membrane=zeros.clone(),
            adaptation=zeros.clone(),
            recurrent_current=zeros.clone(),
            membrane_pre_reset=zeros.clone(),
            theta_eff=zeros.clone(),
        )

    def detach_state(self, state: ALIFReservoirState) -> ALIFReservoirState:
        return ALIFReservoirState(
            spike=state.spike.detach(),
            membrane=state.membrane.detach(),
            adaptation=state.adaptation.detach(),
            recurrent_current=state.recurrent_current.detach(),
            membrane_pre_reset=state.membrane_pre_reset.detach(),
            theta_eff=state.theta_eff.detach(),
        )

    def __call__(
        self,
        input_current: torch.Tensor,
        state: ALIFReservoirState,
    ) -> tuple[torch.Tensor, ALIFReservoirState]:
        return self.forward(input_current, state)

    def forward(
        self,
        input_current: torch.Tensor,
        state: ALIFReservoirState,
    ) -> tuple[torch.Tensor, ALIFReservoirState]:
        recurrent_current = self.liquid(state.spike)
        membrane_pre_reset = (
            self.liquid.beta * state.membrane + input_current + recurrent_current
        )
        membrane_pre_reset = torch.clamp(membrane_pre_reset, -3.0, 3.0)
        adaptation = (
            self.liquid.alif_rho * state.adaptation
            + self.liquid.alif_adapt_increment * state.spike
        )
        theta_eff = self.liquid.threshold + self.liquid.alif_beta * adaptation
        spike = spike_fn(membrane_pre_reset - theta_eff.clamp(min=0.01))
        membrane = membrane_pre_reset * (1.0 - spike)
        next_state = ALIFReservoirState(
            spike=spike,
            membrane=membrane,
            adaptation=adaptation,
            recurrent_current=recurrent_current,
            membrane_pre_reset=membrane_pre_reset,
            theta_eff=theta_eff,
        )
        return spike, next_state


# ---------------------------------------------------------------------------
# LSMModel: full model combining input, liquid, readout
# ---------------------------------------------------------------------------


class NonSpikingLIFReadout(nn.Module):
    """Differentiable LIF-style readout that returns the final membrane."""

    def __init__(
        self,
        n_liquid: int,
        n_output: int,
        beta: float = 0.95,
        learn_beta: bool = False,
        normalize: bool = True,
        bias_once: bool = True,
    ):
        super().__init__()
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError(f"readout LIF beta must be in [0, 1), got {beta}")
        self.linear = nn.Linear(n_liquid, n_output)
        self.learn_beta = bool(learn_beta)
        self.normalize = bool(normalize)
        self.bias_once = bool(bias_once)
        if self.learn_beta:
            beta_init = torch.tensor(float(beta)).clamp(1e-6, 1.0 - 1e-6)
            self.logit_beta = nn.Parameter(torch.logit(beta_init))
            self.register_buffer("beta_buffer", None)
        else:
            self.logit_beta = None
            self.register_buffer("beta_buffer", torch.tensor(float(beta)))

    @property
    def beta(self) -> torch.Tensor:
        if self.logit_beta is not None:
            return torch.sigmoid(self.logit_beta)
        return self.beta_buffer

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def step(
        self,
        liquid_spike_t: torch.Tensor,
        readout_mem: torch.Tensor,
    ) -> torch.Tensor:
        bias = None if self.bias_once else self.linear.bias
        return self.beta * readout_mem + F.linear(
            liquid_spike_t,
            self.linear.weight,
            bias,
        )

    def _normalizer(
        self,
        lengths: int | torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.normalize:
            return torch.ones((), device=device, dtype=dtype)
        beta = self.beta.to(device=device, dtype=dtype)
        lengths_t = torch.as_tensor(lengths, device=device, dtype=dtype)
        denom = (1.0 - beta.pow(lengths_t)) / (1.0 - beta)
        return denom.clamp_min(torch.finfo(dtype).eps)

    def finalize(
        self,
        readout_mem: torch.Tensor,
        lengths: int | torch.Tensor,
    ) -> torch.Tensor:
        logits = readout_mem / self._normalizer(
            lengths,
            device=readout_mem.device,
            dtype=readout_mem.dtype,
        )
        if self.bias_once and self.linear.bias is not None:
            logits = logits + self.linear.bias
        return logits

    def forward(
        self,
        liquid_spikes: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if liquid_spikes.dim() != 3:
            raise ValueError(
                "liquid_spikes must have shape (batch, time, n_liquid); "
                f"got {tuple(liquid_spikes.shape)}"
            )
        batch_size, timesteps, _ = liquid_spikes.shape
        readout_mem = liquid_spikes.new_zeros(batch_size, self.linear.out_features)
        if valid_lengths is not None:
            if valid_lengths.shape != (batch_size,):
                raise ValueError(
                    "valid_lengths must have shape (batch,), "
                    f"got {tuple(valid_lengths.shape)}"
                )
            valid_lengths = valid_lengths.to(device=liquid_spikes.device)
        for t in range(timesteps):
            next_mem = self.step(liquid_spikes[:, t], readout_mem)
            if valid_lengths is None:
                readout_mem = next_mem
            else:
                active = (t < valid_lengths).unsqueeze(1)
                readout_mem = torch.where(active, next_mem, readout_mem)
        if valid_lengths is None:
            return self.finalize(readout_mem, timesteps)
        return self.finalize(readout_mem, valid_lengths.unsqueeze(1))


class LSMModel(nn.Module):
    def __init__(
        self,
        n_input: int = 700,
        n_liquid: int = 200,
        n_output: int = 20,
        T: int = 100,
        exc_ratio: float = 0.8,
        neuron_type: str = "lif",
        beta_min: float = 0.7,
        beta_max: float = 0.95,
        threshold_min: float = 0.8,
        threshold_max: float = 1.5,
        p_input: float = 0.1,
        input_weight_scale: float = 0.1,
        input_projection_mode: str = "fixed_sparse",
        train_input_projection: bool = False,
        recurrent_mode: str = "learned",
        recurrent_sparsity: float = 0.2,
        self_connection: bool = False,
        theta_init_mean: float = 0.0,
        theta_init_std: float = 0.01,
        theta_rank: int = 16,
        theta_lowrank_init_std: float = 0.30,
        w_raw_init_mean: float = -4.0,
        w_raw_init_std: float = 0.01,
        train_w_raw: bool = True,
        w_raw_max: float = -1.0,
        recurrent_weight_scale: float = 1.0,
        match_initial_w_eff_scale: bool = False,
        frozen_w_mode: str = "initialized_w",
        frozen_w_constant_g: float | None = None,
        soft_gate_temp_init: float = 1.0,
        soft_gate_target_density_init: float = 0.3,
        mag_from_separate_param: bool = False,
        bptt_truncate: int = 0,
        alif_rho_init: float = 0.9,
        alif_beta_init: float = 0.4,
        alif_adapt_increment: float = 1.0,
        alif_learn_rho: bool = False,
        alif_learn_beta: bool = False,
        noise_scale: float = 0.1,
        readout_mode: str = "spike_count",
        motor_beta: float = 0.9,
        motor_threshold: float = 1.0,
        motor_mem_clamp: float = 5.0,
        motor_logit_scale: float = 1.0,
        motor_membrane_logit_scale: float = 1.0,
        motor_final_bias: bool = True,
        pred_aux_enabled: bool = False,
        pred_trace_decay: float = 0.9,
        readout_lif_beta: float = 0.95,
        readout_lif_learn_beta: bool = False,
        readout_lif_normalize: bool = True,
        readout_lif_bias_once: bool = True,
    ):
        super().__init__()
        self.T = T  # time stamp
        self.bptt_truncate = bptt_truncate
        self.n_liquid = n_liquid
        self.n_output = n_output
        self.neuron_type = neuron_type
        if readout_mode not in {
            "spike_count",
            "membrane_trace",
            "spike_adaptation_concat",
            "non_spiking_lif_final_mem",
            "motor_lif",
            "motor_lif_count_membrane",
        }:
            raise ValueError(f"Unknown readout_mode: {readout_mode}")
        if readout_mode == "spike_adaptation_concat" and neuron_type != "alif":
            raise ValueError("spike_adaptation_concat readout requires ALIF neurons")
        self.readout_mode = readout_mode
        self.is_motor_readout = readout_mode in {
            "motor_lif",
            "motor_lif_count_membrane",
        }
        self.motor_beta = motor_beta
        self.motor_threshold = motor_threshold
        self.motor_mem_clamp = motor_mem_clamp
        self.motor_logit_scale = motor_logit_scale
        self.motor_membrane_logit_scale = motor_membrane_logit_scale
        self.motor_final_bias_enabled = motor_final_bias
        self.is_non_spiking_lif_readout = readout_mode == "non_spiking_lif_final_mem"

        self.input_proj = InputProjection(
            n_input,
            n_liquid,
            p_input=p_input,
            weight_scale=input_weight_scale,
            mode=input_projection_mode,
            trainable=train_input_projection,
        )
        self.liquid = LiquidLayer(
            n_liquid,
            exc_ratio=exc_ratio,
            neuron_type=neuron_type,
            mode=recurrent_mode,
            target_sparsity=recurrent_sparsity,
            self_connection=self_connection,
            theta_init_mean=theta_init_mean,
            theta_init_std=theta_init_std,
            theta_rank=theta_rank,
            theta_lowrank_init_std=theta_lowrank_init_std,
            w_raw_init_mean=w_raw_init_mean,
            w_raw_init_std=w_raw_init_std,
            train_w_raw=train_w_raw,
            w_raw_max=w_raw_max,
            recurrent_weight_scale=recurrent_weight_scale,
            match_initial_w_eff_scale=match_initial_w_eff_scale,
            frozen_w_mode=frozen_w_mode,
            frozen_w_constant_g=frozen_w_constant_g,
            soft_gate_temp_init=soft_gate_temp_init,
            soft_gate_target_density_init=soft_gate_target_density_init,
            mag_from_separate_param=mag_from_separate_param,
            beta_min=beta_min,
            beta_max=beta_max,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            alif_rho_init=alif_rho_init,
            alif_beta_init=alif_beta_init,
            alif_adapt_increment=alif_adapt_increment,
            alif_learn_rho=alif_learn_rho,
            alif_learn_beta=alif_learn_beta,
            noise_scale=noise_scale,
        )
        self.alif_reservoir = (
            ALIFReservoirBlock(self.liquid) if neuron_type == "alif" else None
        )
        readout_dim = (
            n_liquid * 2 if readout_mode == "spike_adaptation_concat" else n_liquid
        )
        if self.is_non_spiking_lif_readout:
            self.readout = NonSpikingLIFReadout(
                n_liquid,
                n_output,
                beta=readout_lif_beta,
                learn_beta=readout_lif_learn_beta,
                normalize=readout_lif_normalize,
                bias_once=readout_lif_bias_once,
            )
        else:
            self.readout = nn.Linear(
                readout_dim,
                n_output,
                bias=not self.is_motor_readout,
            )
        self.motor_output_bias = (
            nn.Parameter(torch.zeros(n_output))
            if self.is_motor_readout and motor_final_bias
            else None
        )

        self.pred_aux_enabled = pred_aux_enabled
        self.pred_trace_decay = pred_trace_decay
        self.pred_aux = nn.Linear(n_liquid, n_liquid) if pred_aux_enabled else None
        self._last_pred_loss: torch.Tensor | None = None

    def forward(
        self,
        spikes: torch.Tensor,
        tau: float = 1.0,
        return_traces: bool = False,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            spikes: (batch, T, n_input) spike train
            tau: Gumbel temperature
        Returns:
            (batch, n_output) average readout membrane over time, optionally
            paired with traces and/or reservoir diagnostics.
        """
        batch_size = spikes.shape[0]
        device = spikes.device

        # 1. sample recurrent mask once
        # For "learned" mode, sample_mask internally uses STE during training
        # and hard binary during eval — the hard flag here only matters for
        # non-learned modes (random_sparse, fixed) where it has no effect.
        self.liquid.sample_mask(tau=tau)

        # 2. init states
        liquid_mem = torch.zeros(batch_size, self.n_liquid, device=device)
        liquid_spike = torch.zeros(batch_size, self.n_liquid, device=device)
        liquid_a = None
        alif_state = None
        if self.neuron_type == "alif":
            if self.alif_reservoir is None:
                raise RuntimeError("ALIF reservoir block is not initialized")
            alif_state = self.alif_reservoir.init_state(
                batch_size,
                device=device,
                dtype=spikes.dtype,
            )
            liquid_spike = alif_state.spike
            liquid_mem = alif_state.membrane
            liquid_a = alif_state.adaptation
        readout_mem = torch.zeros(batch_size, self.n_output, device=device)
        motor_mem = None
        motor_spike_count = None
        motor_membrane_sum = None
        if self.is_motor_readout:
            motor_mem = torch.zeros(batch_size, self.n_output, device=device)
            motor_spike_count = torch.zeros(batch_size, self.n_output, device=device)
            motor_membrane_sum = torch.zeros(
                batch_size,
                self.n_output,
                device=device,
            )

        # track firing rates for monitoring
        spike_sum = torch.zeros(batch_size, self.n_liquid, device=device)
        membrane_sum = None
        adaptation_sum = None
        if self.readout_mode == "membrane_trace":
            membrane_sum = torch.zeros(batch_size, self.n_liquid, device=device)
        elif self.readout_mode == "spike_adaptation_concat":
            adaptation_sum = torch.zeros(batch_size, self.n_liquid, device=device)

        # filtered trace and accumulator for prediction auxiliary loss
        if self.pred_aux_enabled:
            liquid_trace = torch.zeros(batch_size, self.n_liquid, device=device)
            pred_loss_acc = torch.zeros((), device=device)
            pred_count = 0

        if return_traces:
            trace_spikes = []
            trace_membrane = []
            trace_input_current = []
            trace_recurrent_current = []
            trace_adaptation = [] if self.neuron_type == "alif" else None
            trace_theta_eff = [] if self.neuron_type == "alif" else None

        if return_diagnostics:
            diag_count = batch_size * self.T * self.n_liquid
            diag_membrane_sum = torch.zeros((), device=device)
            diag_membrane_max = None
            diag_input_abs_sum = torch.zeros((), device=device)
            diag_input_abs_max = None
            diag_recurrent_abs_sum = torch.zeros((), device=device)
            diag_recurrent_abs_max = None
            diag_adaptation_sum = torch.zeros((), device=device)
            diag_adaptation_max = None

        # 3. timestep loop
        # truncated BPTT: detach hidden state before the gradient window
        # self.bptt_truncate: window
        grad_start = (self.T - self.bptt_truncate) if self.bptt_truncate > 0 else 0

        for t in range(self.T):
            # Truncated BPTT:
            # keep the current liquid state values, but cut their history.
            # This makes the remaining timesteps start a fresh graph,
            # so gradients do not flow back before grad_start.
            if t == grad_start and t > 0:
                if alif_state is not None:
                    alif_state = self.alif_reservoir.detach_state(alif_state)
                    liquid_mem = alif_state.membrane
                    liquid_spike = alif_state.spike
                    liquid_a = alif_state.adaptation
                else:
                    liquid_mem = liquid_mem.detach()
                    liquid_spike = liquid_spike.detach()
                if motor_mem is not None:
                    motor_mem = motor_mem.detach()
                if self.pred_aux_enabled:
                    liquid_trace = liquid_trace.detach()

            # pick up the current timepoint
            input_current = self.input_proj(spikes[:, t])  # (batch, N)
            if alif_state is not None:
                liquid_spike, alif_state = self.alif_reservoir(
                    input_current,
                    alif_state,
                )
                recurrent_current = alif_state.recurrent_current
                liquid_mem = alif_state.membrane_pre_reset
                liquid_a = alif_state.adaptation
                theta_eff = alif_state.theta_eff
            else:
                recurrent_current = self.liquid(liquid_spike)  # (batch, N)
                liquid_mem = (
                    self.liquid.beta * liquid_mem + input_current + recurrent_current
                )
                liquid_mem = torch.clamp(liquid_mem, -3.0, 3.0)
            if return_traces:
                trace_input_current.append(input_current.detach())
                trace_recurrent_current.append(recurrent_current.detach())
            if return_traces:
                trace_membrane.append(liquid_mem.detach())
            if membrane_sum is not None:
                membrane_sum = membrane_sum + liquid_mem

            if return_diagnostics:
                liquid_mem_detached = liquid_mem.detach()
                input_abs = input_current.detach().abs()
                recurrent_abs = recurrent_current.detach().abs()
                diag_membrane_sum = diag_membrane_sum + liquid_mem_detached.sum()
                current_membrane_max = liquid_mem_detached.max()
                diag_membrane_max = (
                    current_membrane_max
                    if diag_membrane_max is None
                    else torch.maximum(diag_membrane_max, current_membrane_max)
                )
                diag_input_abs_sum = diag_input_abs_sum + input_abs.sum()
                current_input_abs_max = input_abs.max()
                diag_input_abs_max = (
                    current_input_abs_max
                    if diag_input_abs_max is None
                    else torch.maximum(diag_input_abs_max, current_input_abs_max)
                )
                diag_recurrent_abs_sum = (
                    diag_recurrent_abs_sum + recurrent_abs.sum()
                )
                current_recurrent_abs_max = recurrent_abs.max()
                diag_recurrent_abs_max = (
                    current_recurrent_abs_max
                    if diag_recurrent_abs_max is None
                    else torch.maximum(
                        diag_recurrent_abs_max,
                        current_recurrent_abs_max,
                    )
                )

            if alif_state is not None:
                if adaptation_sum is not None:
                    adaptation_sum = adaptation_sum + liquid_a
                if return_traces:
                    trace_adaptation.append(liquid_a.detach())
                    trace_theta_eff.append(theta_eff.detach())
                if return_diagnostics:
                    adaptation_detached = liquid_a.detach()
                    diag_adaptation_sum = (
                        diag_adaptation_sum + adaptation_detached.sum()
                    )
                    current_adaptation_max = adaptation_detached.max()
                    diag_adaptation_max = (
                        current_adaptation_max
                        if diag_adaptation_max is None
                        else torch.maximum(diag_adaptation_max, current_adaptation_max)
                    )
            else:
                liquid_spike = spike_fn(
                    liquid_mem - self.liquid.threshold.clamp(min=0.01)
                )
                liquid_mem = liquid_mem * (1.0 - liquid_spike)  # reset fired neurons
            if return_traces:
                trace_spikes.append(liquid_spike.detach())

            if self.pred_aux_enabled:
                prev_trace = liquid_trace
                liquid_trace = (
                    self.pred_trace_decay * liquid_trace
                    + (1.0 - self.pred_trace_decay) * liquid_spike
                )
                if t > 0:
                    pred_out = torch.sigmoid(self.pred_aux(prev_trace))
                    pred_loss_acc = pred_loss_acc + F.mse_loss(
                        pred_out, liquid_trace.detach(), reduction="mean"
                    )
                    pred_count += 1

            if self.readout_mode == "spike_count":
                readout_mem = readout_mem + self.readout(liquid_spike)
            elif self.is_non_spiking_lif_readout:
                # Fixed-length SHD uses T timesteps with no padding mask here,
                # so the final state after timestep T-1 is the sample logit.
                readout_mem = self.readout.step(liquid_spike, readout_mem)
            elif self.is_motor_readout:
                motor_current = self.readout(liquid_spike)
                motor_mem = self.motor_beta * motor_mem + motor_current
                motor_mem = torch.clamp(
                    motor_mem,
                    -self.motor_mem_clamp,
                    self.motor_mem_clamp,
                )
                motor_mem_pre_spike = motor_mem
                if self.readout_mode == "motor_lif_count_membrane":
                    motor_membrane_sum = motor_membrane_sum + motor_mem_pre_spike
                else:
                    motor_membrane_sum = (
                        motor_membrane_sum + motor_mem_pre_spike.detach()
                    )
                motor_spike = spike_fn(motor_mem - self.motor_threshold)
                motor_mem = motor_mem * (1.0 - motor_spike)
                motor_spike_count = motor_spike_count + motor_spike
            spike_sum = spike_sum + liquid_spike

            if alif_state is not None:
                liquid_mem = alif_state.membrane

        if self.readout_mode == "membrane_trace":
            readout_input = membrane_sum / self.T
            logits = self.readout(readout_input)
        elif self.readout_mode == "spike_adaptation_concat":
            readout_input = torch.cat(
                [spike_sum / self.T, adaptation_sum / self.T], dim=1
            )
            logits = self.readout(readout_input)
        elif self.readout_mode == "motor_lif":
            logits = motor_spike_count * self.motor_logit_scale
            if self.motor_output_bias is not None:
                logits = logits + self.motor_output_bias
        elif self.readout_mode == "motor_lif_count_membrane":
            motor_membrane_trace = motor_membrane_sum / self.T
            logits = (
                motor_spike_count * self.motor_logit_scale
                + motor_membrane_trace * self.motor_membrane_logit_scale
            )
            if self.motor_output_bias is not None:
                logits = logits + self.motor_output_bias
        elif self.is_non_spiking_lif_readout:
            logits = self.readout.finalize(readout_mem, self.T)
        else:
            logits = readout_mem / self.T

        # store for monitoring (detached)
        self._last_spike_rates = (spike_sum / self.T).detach()
        if motor_spike_count is not None:
            self._last_motor_spike_count = motor_spike_count.detach()
            self._last_motor_spike_rates = (motor_spike_count / self.T).detach()
            self._last_motor_membrane_trace = (motor_membrane_sum / self.T).detach()
        else:
            self._last_motor_spike_count = None
            self._last_motor_spike_rates = None
            self._last_motor_membrane_trace = None
        if self.is_non_spiking_lif_readout:
            self._last_readout_lif_mem = readout_mem.detach()
            self._last_readout_lif_logits = logits.detach()
        else:
            self._last_readout_lif_mem = None
            self._last_readout_lif_logits = None
        if liquid_a is not None:
            self._last_liquid_adaptation = liquid_a.detach()
        else:
            self._last_liquid_adaptation = None

        if self.pred_aux_enabled and pred_count > 0:
            self._last_pred_loss = pred_loss_acc / pred_count
        else:
            self._last_pred_loss = torch.zeros((), device=device)

        diagnostics = None
        if return_diagnostics:
            denom = max(diag_count, 1)
            input_abs_mean = diag_input_abs_sum / denom
            recurrent_abs_mean = diag_recurrent_abs_sum / denom
            diagnostics = {
                "mean_spike_rate": float(
                    (spike_sum / self.T).mean().detach().cpu().item()
                ),
                "max_spike_rate": float(
                    (spike_sum / self.T).mean(dim=0).max().detach().cpu().item()
                ),
                "adaptation_mean": float(
                    (diag_adaptation_sum / denom).detach().cpu().item()
                ),
                "adaptation_max": float(
                    (
                        torch.zeros((), device=device)
                        if diag_adaptation_max is None
                        else diag_adaptation_max
                    )
                    .detach()
                    .cpu()
                    .item()
                ),
                "membrane_mean": float(
                    (diag_membrane_sum / denom).detach().cpu().item()
                ),
                "membrane_max": float(
                    (
                        torch.zeros((), device=device)
                        if diag_membrane_max is None
                        else diag_membrane_max
                    )
                    .detach()
                    .cpu()
                    .item()
                ),
                "input_current_abs_mean": float(
                    input_abs_mean.detach().cpu().item()
                ),
                "input_current_abs_max": float(
                    (
                        torch.zeros((), device=device)
                        if diag_input_abs_max is None
                        else diag_input_abs_max
                    )
                    .detach()
                    .cpu()
                    .item()
                ),
                "recurrent_current_abs_mean": float(
                    recurrent_abs_mean.detach().cpu().item()
                ),
                "recurrent_current_abs_max": float(
                    (
                        torch.zeros((), device=device)
                        if diag_recurrent_abs_max is None
                        else diag_recurrent_abs_max
                    )
                    .detach()
                    .cpu()
                    .item()
                ),
                "rec_input_abs_ratio": float(
                    (recurrent_abs_mean / input_abs_mean.clamp(min=1e-12))
                    .detach()
                    .cpu()
                    .item()
                ),
            }

        if return_traces:
            traces = {
                "spikes": torch.stack(trace_spikes, dim=1),
                "membrane": torch.stack(trace_membrane, dim=1),
                "input_current": torch.stack(trace_input_current, dim=1),
                "recurrent_current": torch.stack(trace_recurrent_current, dim=1),
            }
            if trace_adaptation is not None and trace_theta_eff is not None:
                traces["adaptation"] = torch.stack(trace_adaptation, dim=1)
                traces["theta_eff"] = torch.stack(trace_theta_eff, dim=1)
            if return_diagnostics:
                return logits, traces, diagnostics
            return logits, traces

        if return_diagnostics:
            return logits, diagnostics

        return logits

    # ------------------------------------------------------------------
    # losses — scoped to liquid theta only
    # ------------------------------------------------------------------

    def sparsity_loss(self) -> torch.Tensor:
        # sparsity_loss는 theta 값을 조정하여 시그모이드 함수를 통과한에
        # 결과가 0에 더 가깝게 하여 분포가 희소하게 만드는 역할을 함.
        if self.liquid.mode not in HARD_TOPOLOGY_MODES:
            return torch.zeros((), device=self.readout.weight.device)
        theta = self.liquid.get_theta()
        return torch.sigmoid(theta).mean()

    def commitment_loss(self) -> torch.Tensor:
        # theta를 시그모이드를 통과한 것의 분포가 0 또는 1에 몰리게 하는 결과를 내도록 함
        # 만약 theta가 0.5 근처에 존재한다면 엣지의 존재유무가 확확 바뀌기 때문에 불안정해진다.
        if self.liquid.mode not in HARD_TOPOLOGY_MODES:
            return torch.zeros((), device=self.readout.weight.device)
        eps = 1e-6
        theta = self.liquid.get_theta()
        p = torch.sigmoid(theta)
        entropy = -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())
        return entropy.mean()

    def sparsity_info(self) -> float:
        return self.liquid.sparsity()

    def firing_rate_info(self) -> dict:
        """Return firing rate stats from last forward pass."""
        if not hasattr(self, "_last_spike_rates"):
            return {"mean": 0.0, "max": 0.0}
        rates = self._last_spike_rates
        return {
            "mean": rates.mean().item(),
            "max": rates.mean(dim=0).max().item(),
        }

    def adaptation_info(self) -> dict:
        """Return ALIF adaptation stats from the last forward pass."""
        if getattr(self, "_last_liquid_adaptation", None) is None:
            return {"mean": 0.0, "max": 0.0}
        adaptation = self._last_liquid_adaptation
        return {
            "mean": adaptation.mean().item(),
            "max": adaptation.max().item(),
        }

    def motor_info(self) -> dict:
        """Return motor readout spike stats from the last forward pass."""
        rates = getattr(self, "_last_motor_spike_rates", None)
        counts = getattr(self, "_last_motor_spike_count", None)
        membrane = getattr(self, "_last_motor_membrane_trace", None)
        if rates is None or counts is None:
            return {
                "mean_rate": 0.0,
                "max_rate": 0.0,
                "mean_count": 0.0,
                "max_count": 0.0,
                "mean_membrane": 0.0,
                "max_membrane": 0.0,
            }
        if membrane is None:
            mean_membrane = 0.0
            max_membrane = 0.0
        else:
            mean_membrane = membrane.mean().item()
            max_membrane = membrane.mean(dim=0).max().item()
        return {
            "mean_rate": rates.mean().item(),
            "max_rate": rates.mean(dim=0).max().item(),
            "mean_count": counts.mean().item(),
            "max_count": counts.mean(dim=0).max().item(),
            "mean_membrane": mean_membrane,
            "max_membrane": max_membrane,
        }

    def readout_lif_info(self) -> dict:
        """Return non-spiking LIF readout stats from the last forward pass."""
        if not self.is_non_spiking_lif_readout:
            return {"beta": 0.0, "mem_norm": 0.0, "final_logit_norm": 0.0}
        membrane = getattr(self, "_last_readout_lif_mem", None)
        logits = getattr(self, "_last_readout_lif_logits", None)
        beta = float(self.readout.beta.detach().cpu().item())
        mem_norm = (
            0.0 if membrane is None else float(membrane.norm(dim=1).mean().item())
        )
        logit_norm = 0.0 if logits is None else float(logits.norm(dim=1).mean().item())
        return {
            "beta": beta,
            "mem_norm": mem_norm,
            "final_logit_norm": logit_norm,
        }

    def prediction_loss(self) -> torch.Tensor:
        """Return next-state prediction auxiliary loss from the last forward pass."""
        if not self.pred_aux_enabled or self._last_pred_loss is None:
            return torch.zeros((), device=self.readout.weight.device)
        return self._last_pred_loss

    def prediction_info(self) -> float:
        """Return scalar prediction loss for logging."""
        if not self.pred_aux_enabled or self._last_pred_loss is None:
            return 0.0
        return float(self._last_pred_loss.detach().cpu().item())
