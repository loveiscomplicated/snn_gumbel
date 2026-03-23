"""
SNNModel with arbitrary hidden layers and topology mode support.

hidden_layers: list[int]
    e.g. [512]      → 1 hidden layer of 512 neurons
    e.g. [512, 256] → 2 hidden layers

topology.mode propagates to every GumbelLIFLayer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List

from src.models.layers import GumbelLIFLayer, spike_fn


class SNNModel(nn.Module):
    def __init__(
        self,
        n_input: int,
        hidden_layers: List[int],
        n_output: int,
        T: int,
        beta: float = 0.9,
        topology_mode: str = "learned",
        target_sparsity: float = 0.5,
    ):
        super().__init__()
        self.T = T
        self.hidden_layers_sizes = hidden_layers
        self.n_output = n_output
        self.topology_mode = topology_mode

        # build layers: [n_input] + hidden_layers + [n_output]
        dims = [n_input] + hidden_layers + [n_output]
        layers = []
        for i, (n_pre, n_post) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = i == len(dims) - 2
            layers.append(
                GumbelLIFLayer(
                    n_pre,
                    n_post,
                    beta=beta,
                    learn_threshold=not is_last,
                    mode=topology_mode,
                    target_sparsity=target_sparsity,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, tau: float, hard: bool = False) -> torch.Tensor:
        batch = x.shape[0]
        device = x.device

        # initialise membrane potentials
        mems = [
            torch.zeros(batch, layer.n_post, device=device)
            for layer in self.layers
        ]
        spike_sum = torch.zeros(batch, self.n_output, device=device)

        for _ in range(self.T):
            spike = (torch.rand_like(x) < x).float()  # rate-coded input spikes

            for i, layer in enumerate(self.layers):
                current = layer(spike, tau=tau, hard=hard)
                mems[i] = layer.beta * mems[i] + current
                spike = spike_fn(mems[i] - layer.threshold.clamp(min=0.01))
                mems[i] = mems[i] * (1.0 - spike)

            spike_sum = spike_sum + spike  # last layer spikes

        return spike_sum / self.T

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------

    def sparsity_loss(self) -> torch.Tensor:
        return sum(
            torch.sigmoid(layer.theta).mean()
            for layer in self.layers
            if layer.mode == "learned"
        )

    def commitment_loss(self) -> torch.Tensor:
        eps = 1e-6
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        weights = [2.0] + [1.0] * (len(self.layers) - 1)
        for layer, w in zip(self.layers, weights):
            if layer.mode != "learned":
                continue
            p = torch.sigmoid(layer.theta)
            ent = -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())
            total = total + ent.mean() * w
        return total

    def sparsity_info(self) -> List[float]:
        return [layer.sparsity() for layer in self.layers]

    # ------------------------------------------------------------------
    # topology transfer helpers
    # ------------------------------------------------------------------

    def load_topology_from_checkpoint(self, checkpoint_path: str, device: torch.device):
        """Load theta values from another checkpoint and freeze them."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        for i, layer in enumerate(self.layers):
            key = f"layers.{i}.theta"
            if key in state:
                layer.theta.data.copy_(state[key])
                layer.theta.requires_grad_(False)
