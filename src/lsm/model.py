"""
LSM model: InputProjection → LiquidLayer (recurrent) → Readout.

Liquid topology modes:
  - "learned"       : Gumbel-Sigmoid mask, trained end-to-end
  - "random_sparse" : fixed random binary mask at init
  - "fixed"         : random sparse + weights frozen (traditional LSM)
  - "grad_r"        : hard threshold (theta > 0) mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from src.models.layers import gumbel_sigmoid, gumbel_sigmoid_ste, sigmoid_ste, spike_fn


# ---------------------------------------------------------------------------
# InputProjection: fixed random excitatory input → liquid
# ---------------------------------------------------------------------------


class InputProjection(nn.Module):
    """Fixed random sparse connections from input to liquid. Mixed excitatory/inhibitory."""

    def __init__(
        self,
        n_input: int,
        n_liquid: int,
        p_input: float = 0.1,
        weight_scale: float = 0.1,
    ):
        super().__init__()
        mask = (torch.rand(n_input, n_liquid) < p_input).float()
        weight = torch.randn(n_input, n_liquid) * weight_scale * mask
        self.register_buffer("weight", weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


# ---------------------------------------------------------------------------
# LiquidLayer: recurrent connections with Gumbel mask + Dale's Law
# ---------------------------------------------------------------------------


class LiquidLayer(nn.Module):
    """
    Recurrent liquid layer with topology learning.

    Parameters learned: theta (N,N), w_raw (N,N), threshold (N,), log_beta
    Buffers (fixed): dale_sign (N,1), self_conn_mask (N,N)
    """

    def __init__(
        self,
        n_liquid: int,
        exc_ratio: float = 0.8,
        mode: str = "learned",
        target_sparsity: float = 0.2,
        self_connection: bool = False,
        theta_init_mean: float = 0.0,
        theta_init_std: float = 0.01,
        w_raw_init_mean: float = -4.0,
        w_raw_init_std: float = 0.01,
        w_raw_max: float = -1.0,
        beta_min: float = 0.7,
        beta_max: float = 0.95,
        threshold_min: float = 0.8,
        threshold_max: float = 1.5,
        noise_scale: float = 0.1,
    ):
        super().__init__()
        self.n_liquid = n_liquid
        self.mode = mode
        self.w_raw_max = w_raw_max
        self.noise_scale = noise_scale

        # --- learnable parameters ---
        weight_trainable = mode != "fixed"

        self.theta = nn.Parameter(
            torch.randn(n_liquid, n_liquid) * theta_init_std + theta_init_mean,
            requires_grad=(mode == "learned"),
        )
        # softplus(w_raw) is the weight magnitude.
        # softplus(0)=0.693 is way too large for recurrent nets.
        # With N=200, p=0.2: ~40 inputs/neuron, 80% exc.
        # softplus(-4.0)≈0.018 → recurrent current ≈ 0.58 (sub-threshold)
        self.w_raw = nn.Parameter(
            torch.randn(n_liquid, n_liquid) * w_raw_init_std + w_raw_init_mean,
            requires_grad=weight_trainable,
        )
        # shuffle so beta/threshold are not correlated with E/I neuron ordering
        beta_vals = torch.linspace(beta_min, beta_max, n_liquid)
        beta_vals = beta_vals[torch.randperm(n_liquid)]
        self.logit_beta = nn.Parameter(
            torch.log(beta_vals / (1.0 - beta_vals)), requires_grad=weight_trainable
        )
        thr_vals = torch.linspace(threshold_min, threshold_max, n_liquid)
        thr_vals = thr_vals[torch.randperm(n_liquid)]
        self.threshold = nn.Parameter(thr_vals, requires_grad=weight_trainable)

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

        # --- fixed mask for random_sparse / fixed modes ---
        if mode in ("random_sparse", "fixed"):
            mask = (torch.rand(n_liquid, n_liquid) < target_sparsity).float()
            mask = (
                mask * self_conn_mask
            )  # respect self-connection setting / torch tensor * means element wise mul.
            self.register_buffer("fixed_mask", mask)
        else:
            self.register_buffer("fixed_mask", None)

        # cached mask for current simulation
        self.current_mask: torch.Tensor | None = None
        # epoch-level Gumbel noise (Phase 2): stored here, STE recomputed each batch
        self._epoch_noise: torch.Tensor | None = None
        self._epoch_tau: float = 1.0

    @property
    def beta(self):
        return torch.sigmoid(self.logit_beta)

    def sample_epoch_mask(self, tau: float, epoch_noise: torch.Tensor) -> None:
        """Store epoch-level Gumbel noise for Phase 2 training.

        Noise is fixed for the entire epoch so all batches share the same hard topology
        → BPTT gradients accumulate consistently → no explosion.
        Across epochs the noise changes → OFF edges occasionally flip ON → OFF edges
        get w_raw gradient → can be permanently promoted.

        Critically, the STE tensor is NOT stored here. sample_mask() recomputes it
        freshly each batch using this stored noise, so each batch gets its own graph
        that is safely freed after backward().
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
        if self.mode == "learned":
            if self._epoch_noise is not None:
                # Phase 2: recompute STE with the epoch noise every batch.
                # Same noise → same hard{0,1} topology. New graph each call → backward safe.
                # noise_scale controls exploration radius:
                #   0.1 → only edges with |theta| < 0.18 can flip (~0.3% of all edges)
                #   1.0 → standard Gumbel, ~33% flip regardless of theta magnitude
                noisy_logits = (
                    self.theta / self._epoch_tau + self.noise_scale * self._epoch_noise
                )
                soft = torch.sigmoid(noisy_logits)
                hard_mask = (soft >= 0.5).float()
                self.current_mask = hard_mask - soft.detach() + soft
            elif self.training and self.theta.requires_grad:
                # Phase 2 fallback without noise (shouldn't be reached in normal flow)
                self.current_mask = sigmoid_ste(self.theta)
            else:
                # Phase 1 or eval: pure deterministic
                self.current_mask = (torch.sigmoid(self.theta) >= 0.5).float()
        elif self.mode in ("random_sparse", "fixed"):
            self.current_mask = self.fixed_mask
        elif self.mode == "grad_r":
            self.current_mask = (self.theta > 0).float()
        else:
            raise ValueError(f"Unknown liquid mode: {self.mode}")
        return self.current_mask

    def get_effective_weight(self) -> torch.Tensor:
        """Compute effective weight: mask * self_conn * (dale_sign * softplus(w_raw))."""
        w_clamped = torch.clamp(self.w_raw, max=self.w_raw_max)
        signed_w = self.dale_sign * F.softplus(w_clamped)  # (N, N)
        return self.current_mask * self.self_conn_mask * signed_w

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
        # learned
        return ((torch.sigmoid(self.theta) >= 0.5).float()) * self.self_conn_mask


# ---------------------------------------------------------------------------
# LSMModel: full model combining input, liquid, readout
# ---------------------------------------------------------------------------


class LSMModel(nn.Module):
    def __init__(
        self,
        n_input: int = 700,
        n_liquid: int = 200,
        n_output: int = 20,
        T: int = 100,
        exc_ratio: float = 0.8,
        beta_min: float = 0.7,
        beta_max: float = 0.95,
        threshold_min: float = 0.8,
        threshold_max: float = 1.5,
        p_input: float = 0.1,
        input_weight_scale: float = 0.1,
        recurrent_mode: str = "learned",
        recurrent_sparsity: float = 0.2,
        self_connection: bool = False,
        theta_init_mean: float = 0.0,
        theta_init_std: float = 0.01,
        w_raw_init_mean: float = -4.0,
        w_raw_init_std: float = 0.01,
        w_raw_max: float = -1.0,
        bptt_truncate: int = 0,
        noise_scale: float = 0.1,
    ):
        super().__init__()
        self.T = T  # time stamp
        self.bptt_truncate = bptt_truncate
        self.n_liquid = n_liquid
        self.n_output = n_output

        self.input_proj = InputProjection(
            n_input,
            n_liquid,
            p_input=p_input,
            weight_scale=input_weight_scale,
        )
        self.liquid = LiquidLayer(
            n_liquid,
            exc_ratio=exc_ratio,
            mode=recurrent_mode,
            target_sparsity=recurrent_sparsity,
            self_connection=self_connection,
            theta_init_mean=theta_init_mean,
            theta_init_std=theta_init_std,
            w_raw_init_mean=w_raw_init_mean,
            w_raw_init_std=w_raw_init_std,
            w_raw_max=w_raw_max,
            beta_min=beta_min,
            beta_max=beta_max,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            noise_scale=noise_scale,
        )
        self.readout = nn.Linear(n_liquid, n_output)

    def forward(self, spikes: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        Args:
            spikes: (batch, T, n_input) spike train
            tau: Gumbel temperature
        Returns:
            (batch, n_output) average readout membrane over time
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
        readout_mem = torch.zeros(batch_size, self.n_output, device=device)

        # track firing rates for monitoring
        spike_sum = torch.zeros(batch_size, self.n_liquid, device=device)

        # 3. timestep loop
        # truncated BPTT: detach hidden state before the gradient window
        # self.bptt_truncate: window
        grad_start = (self.T - self.bptt_truncate) if self.bptt_truncate > 0 else 0

        for t in range(self.T):
            if t == grad_start and t > 0:
                liquid_mem = liquid_mem.detach()
                liquid_spike = liquid_spike.detach()

            input_current = self.input_proj(spikes[:, t])  # (batch, N)
            recurrent_current = self.liquid(liquid_spike)  # (batch, N)

            liquid_mem = (
                self.liquid.beta * liquid_mem + input_current + recurrent_current
            )
            liquid_mem = torch.clamp(liquid_mem, -3.0, 3.0)
            liquid_spike = spike_fn(liquid_mem - self.liquid.threshold.clamp(min=0.01))
            liquid_mem = liquid_mem * (1.0 - liquid_spike)  # reset

            readout_mem = readout_mem + self.readout(liquid_spike)
            spike_sum = spike_sum + liquid_spike

        # store for monitoring (detached)
        self._last_spike_rates = (spike_sum / self.T).detach()

        return readout_mem / self.T

    # ------------------------------------------------------------------
    # losses — scoped to liquid theta only
    # ------------------------------------------------------------------

    def sparsity_loss(self) -> torch.Tensor:
        # sparsity_loss는 theta 값을 조정하여 시그모이드 함수를 통과한
        # 결과가 0에 더 가깝게 하여 분포가 희소하게 만드는 역할을 함.
        if self.liquid.mode != "learned":
            return torch.tensor(0.0, device=self.liquid.theta.device)
        return torch.sigmoid(self.liquid.theta).mean()

    def commitment_loss(self) -> torch.Tensor:
        # theta를 시그모이드를 통과한 것의 분포가 0 또는 1에 몰리게 하는 결과를 내도록 함
        if self.liquid.mode != "learned":
            return torch.tensor(0.0, device=self.liquid.theta.device)
        eps = 1e-6
        p = torch.sigmoid(self.liquid.theta)
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
