"""
SNN with learnable topology via Gumbel-Sigmoid.

Learnable parameters per layer:
  - theta (edge logits): whether connection i->j exists
  - weight: synaptic strength
  - threshold: per-neuron firing threshold
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Surrogate gradient for Heaviside spike function
# ---------------------------------------------------------------------------


class SurrogateSpike(torch.autograd.Function):
    """
    Forward:  Heaviside(x) = 1 if x >= 0 else 0
    Backward: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    """

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


# ---------------------------------------------------------------------------
# Gumbel-Sigmoid (Bernoulli relaxation)
# ---------------------------------------------------------------------------


def gumbel_sigmoid(
    logits: torch.Tensor, tau: float = 1.0, hard: bool = False
) -> torch.Tensor:
    """
    Bernoulli Gumbel-Softmax relaxation (PGExplainer style).

    logits : theta_ij  (edge existence log-odds)
    tau    : temperature (high=soft/exploratory, low=hard/binary)
    hard   : if True, use straight-through estimator
    """
    if hard:
        return (torch.sigmoid(logits) >= 0.5).float()
    eps = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
    gumbel_noise = torch.log(eps) - torch.log(1.0 - eps)
    return torch.sigmoid((logits + gumbel_noise) / tau)


# ---------------------------------------------------------------------------
# Single LIF layer with learnable topology
# ---------------------------------------------------------------------------


class GumbelLIFLayer(nn.Module):
    """
    One feedforward layer of LIF neurons whose connections are learned
    via Gumbel-Sigmoid.

    Parameters
    ----------
    n_pre  : number of pre-synaptic neurons
    n_post : number of post-synaptic neurons
    beta   : initial membrane decay constant (learned per-layer scalar)
    """

    def __init__(self, n_pre: int, n_post: int, beta: float = 0.9, learn_threshold: bool = True):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post

        # Edge existence logits  theta_ij  [n_pre, n_post]
        self.theta = nn.Parameter(torch.zeros(n_pre, n_post))

        # Synaptic weights  w_ij  [n_pre, n_post]
        self.weight = nn.Parameter(torch.empty(n_pre, n_post))
        nn.init.kaiming_uniform_(self.weight, a=0.1)

        # Per-neuron firing threshold  v_i  [n_post]
        self.threshold = nn.Parameter(torch.ones(n_post), requires_grad=learn_threshold)

        # Membrane decay (one learnable scalar per layer)
        self.log_beta = nn.Parameter(torch.tensor(beta).log())

    @property
    def beta(self):
        return torch.sigmoid(self.log_beta)  # constrain to (0, 1)

    def forward(self, spikes_pre: torch.Tensor, tau: float, hard: bool = False):
        """
        Run one timestep.

        spikes_pre : [batch, n_pre]  binary spike tensor from previous layer
        Returns    : [batch, n_post] binary spikes and updated membrane [batch, n_post]

        NOTE: membrane state is managed externally (see SNNModel.forward).
        """
        # Edge mask  m_ij  [n_pre, n_post]
        mask = gumbel_sigmoid(self.theta, tau=tau, hard=hard)

        # Effective weight = mask * weight
        eff_w = mask * self.weight  # [n_pre, n_post]

        # Synaptic current: I[t] = spikes_pre @ eff_w
        current = spikes_pre @ eff_w  # [batch, n_post]
        return current

    def get_binary_mask(self) -> torch.Tensor:
        """Return hard binary mask using learned theta (for inference)."""
        return (torch.sigmoid(self.theta) > 0.5).float()

    def sparsity(self) -> float:
        """Fraction of connections that are active (sigmoid(theta) >= 0.5)."""
        with torch.no_grad():
            return (torch.sigmoid(self.theta) >= 0.5).float().mean().item()


# ---------------------------------------------------------------------------
# Full SNN Model
# ---------------------------------------------------------------------------


class SNNModel(nn.Module):
    """
    Two-layer feedforward SNN:
      input (784) -> hidden (n_hidden) -> output (n_output)

    Both layers use GumbelLIFLayer for learnable topology.
    """

    def __init__(
        self, n_input: int, n_hidden: int, n_output: int, T: int, beta: float = 0.9
    ):
        super().__init__()
        self.T = T
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.layer1 = GumbelLIFLayer(n_input, n_hidden, beta, learn_threshold=True)
        self.layer2 = GumbelLIFLayer(n_hidden, n_output, beta, learn_threshold=False)

    def forward(self, x: torch.Tensor, tau: float, hard: bool = False):
        """
        x    : [batch, 784]  normalised pixel values used as constant current
        tau  : Gumbel temperature
        hard : use straight-through estimator

        Returns
        -------
        firing_rates : [batch, n_output]  mean spike rate over T timesteps
        """
        batch = x.shape[0]
        device = x.device

        # Membrane potentials
        mem1 = torch.zeros(batch, self.n_hidden, device=device)
        mem2 = torch.zeros(batch, self.n_output, device=device)

        # Spike accumulators for output layer
        spike2_sum = torch.zeros(batch, self.n_output, device=device)

        # Input spikes: encode pixel values as Poisson rate over T steps
        # p(spike at t) = pixel value (already in [0,1])
        for _ in range(self.T):
            # --- rate-coded input spikes ---
            spike_in = (torch.rand_like(x) < x).float()

            # --- layer 1 ---
            current1 = self.layer1(spike_in, tau=tau, hard=hard)
            mem1 = self.layer1.beta * mem1 + current1
            spike1 = spike_fn(mem1 - self.layer1.threshold.clamp(min=0.01))
            mem1 = mem1 * (1.0 - spike1)  # soft reset

            # --- layer 2 ---
            current2 = self.layer2(spike1, tau=tau, hard=hard)
            mem2 = self.layer2.beta * mem2 + current2
            spike2 = spike_fn(mem2 - self.layer2.threshold.clamp(min=0.01))
            mem2 = mem2 * (1.0 - spike2)

            spike2_sum = spike2_sum + spike2

        firing_rates = spike2_sum / self.T
        return firing_rates

    def sparsity_loss(self) -> torch.Tensor:
        """L_sparse = mean of sigmoid(theta) across all layers."""
        l1 = torch.sigmoid(self.layer1.theta).mean()
        l2 = torch.sigmoid(self.layer2.theta).mean()
        return l1 + l2

    def commitment_loss(self) -> torch.Tensor:
        """Binary entropy of sigmoid(theta) — minimised when theta is 0 or 1."""
        eps = 1e-6
        p1 = torch.sigmoid(self.layer1.theta)
        ent1 = -(p1 * (p1 + eps).log() + (1 - p1) * (1 - p1 + eps).log())
        p2 = torch.sigmoid(self.layer2.theta)
        ent2 = -(p2 * (p2 + eps).log() + (1 - p2) * (1 - p2 + eps).log())
        return ent1.mean() * 2.0 + ent2.mean()

    def sparsity_info(self):
        """Returns (layer1_sparsity, layer2_sparsity) as fractions of active edges."""
        return self.layer1.sparsity(), self.layer2.sparsity()
