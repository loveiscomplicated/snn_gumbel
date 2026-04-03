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

from src.models.layers import gumbel_sigmoid, spike_fn


# ---------------------------------------------------------------------------
# InputProjection: fixed random excitatory input → liquid
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """Fixed random sparse connections from input to liquid. Excitatory only."""

    def __init__(self, n_input: int, n_liquid: int, p_input: float = 0.1,
                 weight_scale: float = 0.1):
        super().__init__()
        mask = (torch.rand(n_input, n_liquid) < p_input).float()
        weight = torch.rand(n_input, n_liquid) * weight_scale * mask
        self.register_buffer('weight', weight)

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
        theta_init_std: float = 0.01,
        beta: float = 0.9,
    ):
        super().__init__()
        self.n_liquid = n_liquid
        self.mode = mode

        # --- learnable parameters ---
        weight_trainable = mode != "fixed"

        self.theta = nn.Parameter(
            torch.randn(n_liquid, n_liquid) * theta_init_std,
            requires_grad=(mode == "learned"),
        )
        # softplus(w_raw) is the weight magnitude.
        # softplus(0)=0.693 is way too large for recurrent nets.
        # Target: spectral radius ≈ 0.9 → need softplus ≈ 0.15
        # softplus(-2.0) ≈ 0.127, softplus(-1.5) ≈ 0.201
        self.w_raw = nn.Parameter(
            torch.randn(n_liquid, n_liquid) * 0.01 - 2.0,
            requires_grad=weight_trainable,
        )
        self.threshold = nn.Parameter(torch.ones(n_liquid), requires_grad=weight_trainable)
        self.log_beta = nn.Parameter(torch.tensor(beta).log(), requires_grad=weight_trainable)

        # --- Dale's Law: exc (+1) / inh (-1) sign buffer ---
        n_exc = int(exc_ratio * n_liquid)
        dale_sign = torch.ones(n_liquid, 1)
        dale_sign[n_exc:] = -1.0
        self.register_buffer('dale_sign', dale_sign)

        # --- self-connection mask: diagonal = 0 ---
        if self_connection:
            self_conn_mask = torch.ones(n_liquid, n_liquid)
        else:
            self_conn_mask = 1.0 - torch.eye(n_liquid)
        self.register_buffer('self_conn_mask', self_conn_mask)

        # --- fixed mask for random_sparse / fixed modes ---
        if mode in ("random_sparse", "fixed"):
            mask = (torch.rand(n_liquid, n_liquid) < target_sparsity).float()
            mask = mask * self_conn_mask  # respect self-connection setting
            self.register_buffer('fixed_mask', mask)
        else:
            self.register_buffer('fixed_mask', None)

        # cached mask for current simulation
        self.current_mask: torch.Tensor | None = None

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)

    def sample_mask(self, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """Sample mask once before the simulation timestep loop."""
        if self.mode == "learned":
            self.current_mask = gumbel_sigmoid(self.theta, tau=tau, hard=hard)
        elif self.mode in ("random_sparse", "fixed"):
            self.current_mask = self.fixed_mask
        elif self.mode == "grad_r":
            self.current_mask = (self.theta > 0).float()
        else:
            raise ValueError(f"Unknown liquid mode: {self.mode}")
        return self.current_mask

    def get_effective_weight(self) -> torch.Tensor:
        """Compute effective weight: mask * self_conn * (dale_sign * softplus(w_raw))."""
        signed_w = self.dale_sign * F.softplus(self.w_raw)   # (N, N)
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
        beta: float = 0.9,
        exc_ratio: float = 0.8,
        p_input: float = 0.1,
        input_weight_scale: float = 0.1,
        recurrent_mode: str = "learned",
        recurrent_sparsity: float = 0.2,
        self_connection: bool = False,
        theta_init_std: float = 0.01,
    ):
        super().__init__()
        self.T = T
        self.n_liquid = n_liquid
        self.n_output = n_output

        self.input_proj = InputProjection(
            n_input, n_liquid, p_input=p_input, weight_scale=input_weight_scale,
        )
        self.liquid = LiquidLayer(
            n_liquid,
            exc_ratio=exc_ratio,
            mode=recurrent_mode,
            target_sparsity=recurrent_sparsity,
            self_connection=self_connection,
            theta_init_std=theta_init_std,
            beta=beta,
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
        hard = not self.training
        self.liquid.sample_mask(tau=tau, hard=hard)

        # 2. init states
        liquid_mem = torch.zeros(batch_size, self.n_liquid, device=device)
        liquid_spike = torch.zeros(batch_size, self.n_liquid, device=device)
        readout_mem = torch.zeros(batch_size, self.n_output, device=device)

        # track firing rates for monitoring
        spike_sum = torch.zeros(batch_size, self.n_liquid, device=device)

        # 3. timestep loop
        for t in range(self.T):
            input_current = self.input_proj(spikes[:, t])          # (batch, N)
            recurrent_current = self.liquid(liquid_spike)           # (batch, N)

            liquid_mem = self.liquid.beta * liquid_mem + input_current + recurrent_current
            liquid_spike = spike_fn(liquid_mem - self.liquid.threshold.clamp(min=0.01))
            liquid_mem = liquid_mem * (1.0 - liquid_spike)         # reset

            readout_mem = readout_mem + self.readout(liquid_spike)
            spike_sum = spike_sum + liquid_spike

        # store for monitoring (detached)
        self._last_spike_rates = (spike_sum / self.T).detach()

        return readout_mem / self.T

    # ------------------------------------------------------------------
    # losses — scoped to liquid theta only
    # ------------------------------------------------------------------

    def sparsity_loss(self) -> torch.Tensor:
        if self.liquid.mode != "learned":
            return torch.tensor(0.0, device=self.liquid.theta.device)
        return torch.sigmoid(self.liquid.theta).mean()

    def commitment_loss(self) -> torch.Tensor:
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
        if not hasattr(self, '_last_spike_rates'):
            return {"mean": 0.0, "max": 0.0}
        rates = self._last_spike_rates
        return {
            "mean": rates.mean().item(),
            "max": rates.mean(dim=0).max().item(),
        }
