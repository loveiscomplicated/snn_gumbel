"""
GumbelLIFLayer and helpers.

topology.mode controls how the binary connectivity mask is produced:
  - "learned"      : Gumbel-Sigmoid during training, hard sigmoid at eval (default)
  - "full"         : mask is always 1 (theta not learned)
  - "random_sparse": random binary mask fixed at init with given sparsity
  - "transfer"     : theta loaded from external checkpoint, frozen
"""

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sg = torch.sigmoid(x)
        surrogate_grad = sg * (1.0 - sg)
        return grad_output * surrogate_grad


def spike_fn(x):
    return SurrogateSpike.apply(x)


def gumbel_sigmoid(logits, tau=1.0, hard=False):
    if hard:
        return (torch.sigmoid(logits) >= 0.5).float()
    eps = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    gumbel_noise = torch.log(eps) - torch.log(1.0 - eps)
    return torch.sigmoid((logits + gumbel_noise) / tau)


def sigmoid_ste(logits):
    """
    Deterministic Straight-Through Estimator for binary mask.

    Forward : hard binary threshold  (sigmoid(logits) >= 0.5)
    Backward: gradient of sigmoid(logits)

    Unlike gumbel_sigmoid_ste, no stochastic noise is added.
    This is critical for recurrent networks where per-batch topology
    fluctuation causes gradient explosion via BPTT.
    """
    soft = torch.sigmoid(logits)
    hard = (soft >= 0.5).float()
    return hard - soft.detach() + soft


def gumbel_sigmoid_ste(logits, tau=1.0):
    """
    Gumbel-Sigmoid with Straight-Through Estimator.

    Solves train/eval discrepancy while preserving Gumbel stochastic exploration:

      Forward : Gumbel noise → hard binary threshold  (train/eval see same binary mask)
      Backward: gradient of sigmoid(noisy_logits/tau)  (differentiable path to theta)

    Without Gumbel noise this collapses to Grad R (theta > 0 threshold), which is
    Baseline D. The noise is essential to maintain the paper's core differentiation.

    Temperature annealing recovers its meaning:
      high tau → noise dominates, wide exploration of edge uncertainty
      low tau  → noise shrinks, decisions converge to theta sign
    """
    eps = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    gumbel_noise = torch.log(eps) - torch.log(1.0 - eps)
    # Apply tau only to theta, not to noise.
    # (theta + noise) / tau gives P(connection) = sigmoid(theta), independent of tau —
    # temperature annealing has zero effect on stochasticity.
    # theta/tau + noise gives P(connection) = sigmoid(theta/tau), which properly
    # anneals: high tau → wide exploration; low tau → noise negligible, sign(theta) decides.
    noisy_logits = logits / tau + gumbel_noise

    soft = torch.sigmoid(noisy_logits)
    hard = (soft >= 0.5).float()
    # STE: forward value = hard binary, gradient = d(soft)/d(logits)
    return hard - soft.detach() + soft


class GumbelLIFLayer(nn.Module):
    """
    Single LIF layer with topology controlled by `mode`.

    Args:
        n_pre, n_post : layer dimensions
        beta          : initial membrane decay (overridden per-neuron via log_beta)
        learn_threshold: whether threshold is a learnable parameter
        mode          : "learned" | "full" | "random_sparse" | "transfer"
        target_sparsity: fraction of edges kept when mode=="random_sparse"
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        beta: float = 0.9,
        learn_threshold: bool = True,
        mode: str = "learned",
        target_sparsity: float = 0.5,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.mode = mode

        # theta is always created; for non-learned modes it is frozen or ignored
        self.theta = nn.Parameter(
            torch.randn(n_pre, n_post) * 0.01,
            requires_grad=(mode == "learned"),
        )

        self.weight = nn.Parameter(torch.empty(n_pre, n_post))
        nn.init.kaiming_uniform_(self.weight, a=0.1)

        self.threshold = nn.Parameter(
            torch.ones(n_post), requires_grad=learn_threshold
        )
        self.log_beta = nn.Parameter(torch.tensor(beta).log())

        # fixed mask for random_sparse
        if mode == "random_sparse":
            mask = (torch.rand(n_pre, n_post) < target_sparsity).float()
            self.register_buffer("fixed_mask", mask)
        else:
            self.fixed_mask = None

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)

    def forward(self, spikes_pre, tau=1.0, hard=False):
        if self.mode == "learned":
            mask = gumbel_sigmoid(self.theta, tau=tau, hard=hard)
        elif self.mode == "full":
            mask = torch.ones_like(self.weight)
        elif self.mode == "random_sparse":
            mask = self.fixed_mask
        elif self.mode == "transfer":
            # theta has been loaded and frozen; use hard mask
            mask = (torch.sigmoid(self.theta) >= 0.5).float()
        else:
            raise ValueError(f"Unknown topology mode: {self.mode}")

        eff_w = mask * self.weight
        current = spikes_pre @ eff_w
        return current

    def get_binary_mask(self) -> torch.Tensor:
        if self.mode == "full":
            return torch.ones_like(self.weight)
        if self.mode == "random_sparse":
            return self.fixed_mask
        return (torch.sigmoid(self.theta) > 0.5).float()

    def sparsity(self) -> float:
        with torch.no_grad():
            return self.get_binary_mask().mean().item()
