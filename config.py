from dataclasses import dataclass


@dataclass
class Config:
    # Network architecture
    n_input: int = 784
    n_hidden: int = 512  # 256
    n_output: int = 10

    # SNN simulation
    T: int = 25  # timesteps per sample
    beta: float = 0.9  # membrane decay factor (learnable optional)

    # Gumbel-Sigmoid temperature annealing
    tau_start: float = 1.0
    tau_end: float = 0.05
    tau_anneal_epochs: int = 25

    # Training
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    lambda_sparse: float = 0.005  # sparsity regularization weight
    lambda_commit: float = 0.08  # commitment (binary entropy) regularization weight

    # Inference
    edge_threshold: float = 0.5

    # Misc
    seed: int = 42
    data_dir: str = "./data"
    checkpoint_path: str = "./checkpoint.pt"


cfg = Config()
