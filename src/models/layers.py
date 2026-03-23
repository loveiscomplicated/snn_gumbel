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
