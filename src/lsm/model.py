"""
LSM model: InputProjection → LiquidLayer (recurrent) → Readout.

Liquid topology modes:
  - "learned"       : Gumbel-Sigmoid mask, trained end-to-end
  - "learned_lowrank": Gumbel-Sigmoid mask with directed low-rank theta
  - "random_sparse"  : fixed random binary mask at init
  - "fixed"          : random sparse + weights frozen (traditional LSM)
  - "grad_r"         : hard threshold (theta > 0) mask
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import sigmoid_ste, spike_fn

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

        if neuron_type not in {"lif", "alif"}:
            raise ValueError(f"Unknown neuron_type: {neuron_type}")
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
        elif mode == "learned_lowrank":
            self.src_embed = nn.Parameter(
                torch.randn(n_liquid, theta_rank) * theta_lowrank_init_std
            )
            self.dst_embed = nn.Parameter(
                torch.randn(n_liquid, theta_rank) * theta_lowrank_init_std
            )
            self.theta_bias = nn.Parameter(torch.tensor(float(theta_init_mean)))
        else:
            self.theta = nn.Parameter(
                torch.randn(n_liquid, n_liquid) * theta_init_std + theta_init_mean,
                requires_grad=(mode == "grad_r"),
            )
        # softplus(w_raw) is the weight magnitude.
        # softplus(0)=0.693 is way too large for recurrent nets.
        # With N=200, p=0.2: ~40 inputs/neuron, 80% exc.
        # softplus(-4.0)≈0.018 → recurrent current ≈ 0.58 (sub-threshold)
        self.w_raw = nn.Parameter(
            torch.randn(n_liquid, n_liquid) * w_raw_init_std + w_raw_init_mean,
            requires_grad=(weight_trainable and train_w_raw),
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
        if self.mode == "learned_lowrank":
            return self.src_embed @ self.dst_embed.T + self.theta_bias
        raise RuntimeError(f"Topology logits are not defined for mode: {self.mode}")

    def topology_parameters(self) -> list[nn.Parameter]:
        if self.mode in ("learned", "grad_r"):
            return [self.theta]
        if self.mode == "learned_lowrank":
            return [self.src_embed, self.dst_embed, self.theta_bias]
        return []

    def set_topology_requires_grad(self, requires_grad: bool) -> None:
        for param in self.topology_parameters():
            param.requires_grad_(requires_grad)

    def topology_state_dict(self) -> dict[str, torch.Tensor]:
        if self.mode == "learned_lowrank":
            return {
                "src_embed": self.src_embed.detach().clone(),
                "dst_embed": self.dst_embed.detach().clone(),
                "theta_bias": self.theta_bias.detach().clone(),
            }
        if self.mode in ("learned", "grad_r"):
            return {"theta": self.theta.detach().clone()}
        return {}

    def load_topology_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if self.mode == "learned_lowrank":
            required_keys = {"src_embed", "dst_embed", "theta_bias"}
        elif self.mode in ("learned", "grad_r"):
            required_keys = {"theta"}
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
            if self.mode == "learned_lowrank":
                self.src_embed.copy_(state["src_embed"])
                self.dst_embed.copy_(state["dst_embed"])
                self.theta_bias.copy_(state["theta_bias"])
            else:
                self.theta.copy_(state["theta"])

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
        if self.mode in ("learned", "learned_lowrank"):
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
        if self.mode in ("learned", "learned_lowrank"):
            theta = self.get_theta()
            return ((torch.sigmoid(theta) >= 0.5).float()) * self.self_conn_mask
        raise RuntimeError(f"Unknown liquid mode: {self.mode}")


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
        neuron_type: str = "lif",
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
        theta_rank: int = 16,
        theta_lowrank_init_std: float = 0.30,
        w_raw_init_mean: float = -4.0,
        w_raw_init_std: float = 0.01,
        train_w_raw: bool = True,
        w_raw_max: float = -1.0,
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

        self.input_proj = InputProjection(
            n_input,
            n_liquid,
            p_input=p_input,
            weight_scale=input_weight_scale,
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
        readout_dim = (
            n_liquid * 2 if readout_mode == "spike_adaptation_concat" else n_liquid
        )
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
        liquid_a = None
        if self.neuron_type == "alif":
            liquid_a = torch.zeros(batch_size, self.n_liquid, device=device)
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
                liquid_mem = liquid_mem.detach()
                liquid_spike = liquid_spike.detach()
                if liquid_a is not None:
                    liquid_a = liquid_a.detach()
                if motor_mem is not None:
                    motor_mem = motor_mem.detach()
                if self.pred_aux_enabled:
                    liquid_trace = liquid_trace.detach()

            # pick up the current timepoint
            input_current = self.input_proj(spikes[:, t])  # (batch, N)
            recurrent_current = self.liquid(liquid_spike)  # (batch, N)

            liquid_mem = (
                self.liquid.beta * liquid_mem + input_current + recurrent_current
            )
            liquid_mem = torch.clamp(liquid_mem, -3.0, 3.0)
            if membrane_sum is not None:
                membrane_sum = membrane_sum + liquid_mem

            if self.neuron_type == "alif":
                liquid_a = (
                    self.liquid.alif_rho * liquid_a
                    + self.liquid.alif_adapt_increment * liquid_spike
                )
                if adaptation_sum is not None:
                    adaptation_sum = adaptation_sum + liquid_a
                theta_eff = self.liquid.threshold + self.liquid.alif_beta * liquid_a
                liquid_spike = spike_fn(liquid_mem - theta_eff.clamp(min=0.01))
            else:
                liquid_spike = spike_fn(
                    liquid_mem - self.liquid.threshold.clamp(min=0.01)
                )
            liquid_mem = liquid_mem * (1.0 - liquid_spike)  # reset fired neurons

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
        else:
            logits = readout_mem / self.T

        # store for monitoring (detached)
        self._last_spike_rates = (spike_sum / self.T).detach()
        if motor_spike_count is not None:
            self._last_motor_spike_count = motor_spike_count.detach()
            self._last_motor_spike_rates = (motor_spike_count / self.T).detach()
            self._last_motor_membrane_trace = (
                motor_membrane_sum / self.T
            ).detach()
        else:
            self._last_motor_spike_count = None
            self._last_motor_spike_rates = None
            self._last_motor_membrane_trace = None
        if liquid_a is not None:
            self._last_liquid_adaptation = liquid_a.detach()
        else:
            self._last_liquid_adaptation = None

        if self.pred_aux_enabled and pred_count > 0:
            self._last_pred_loss = pred_loss_acc / pred_count
        else:
            self._last_pred_loss = torch.zeros((), device=device)

        return logits

    # ------------------------------------------------------------------
    # losses — scoped to liquid theta only
    # ------------------------------------------------------------------

    def sparsity_loss(self) -> torch.Tensor:
        # sparsity_loss는 theta 값을 조정하여 시그모이드 함수를 통과한에
        # 결과가 0에 더 가깝게 하여 분포가 희소하게 만드는 역할을 함.
        if self.liquid.mode not in ("learned", "learned_lowrank"):
            return torch.zeros((), device=self.readout.weight.device)
        theta = self.liquid.get_theta()
        return torch.sigmoid(theta).mean()

    def commitment_loss(self) -> torch.Tensor:
        # theta를 시그모이드를 통과한 것의 분포가 0 또는 1에 몰리게 하는 결과를 내도록 함
        # 만약 theta가 0.5 근처에 존재한다면 엣지의 존재유무가 확확 바뀌기 때문에 불안정해진다.
        if self.liquid.mode not in ("learned", "learned_lowrank"):
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
