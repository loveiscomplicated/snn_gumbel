# PROJECT_STRUCTURE

## .DS_Store `unknown`

## .claude/settings.local.json `JSON`

## .gitignore `unknown`

## LICENSE `unknown`

## _legacy/checkpoint.pt `PT`

## _legacy/config.py `Python`

## _legacy/evaluate.py `Python`

### 함수
- `get_device()` (L15)
- `load_model(checkpoint_path: str, device)` (L23)
- `run_evaluation(checkpoint_path: str = None)` (L36)

## _legacy/model.py `Python`

### 클래스 / 타입
- **SurrogateSpike** (L19)
  Forward:  Heaviside(x) = 1 if x >= 0 else 0
  - `forward`
  - `backward`
- **GumbelLIFLayer** (L69)
  One feedforward layer of LIF neurons whose connections are learned
  - `__init__`
  - `beta`
  - `forward`
  - `get_binary_mask`
  - `sparsity`
- **SNNModel** (L137)
  Two-layer feedforward SNN:
  - `__init__`
  - `forward`
  - `sparsity_loss`
  - `commitment_loss`
  - `sparsity_info`

### 함수
- `spike_fn(x)` (L38)
- `gumbel_sigmoid(
    logits: torch.Tensor, tau: float = 1.0, hard: bool = False
) torch.Tensor` (L47)
  Bernoulli Gumbel-Softmax relaxation (PGExplainer style).

## _legacy/resources/input_receptive_field.png `PNG`

## _legacy/resources/theta_distribution.png `PNG`

## _legacy/resources/threshold_distribution.png `PNG`

## _legacy/resources/topology.png `PNG`

## _legacy/resources/training_curves.png `PNG`

## _legacy/train.py `Python`

### 함수
- `get_device()` (L19)
- `get_tau(epoch: int) float` (L27)
  Cosine anneal tau from tau_start to tau_end over tau_anneal_epochs.
- `build_dataloaders()` (L36)
- `evaluate(model, loader, device, tau)` (L70)
- `train(resume: bool = False)` (L85)

## _legacy/visualize.py `Python`

### 함수
- `plot_training_curves(history: list, save_path: str = "training_curves.png")` (L16)
- `plot_topology(model, save_path: str = "topology.png")` (L58)
  Visualise binary edge masks as heatmaps.
- `plot_theta_distribution(model, save_path: str = "theta_distribution.png")` (L96)
  Histogram of sigmoid(theta) — distribution of connection probabilities.
- `plot_threshold_distribution(model, save_path: str = "threshold_distribution.png")` (L123)
  Distribution of learned per-neuron firing thresholds.
- `plot_input_connectivity(model, save_path: str = "input_receptive_field.png")` (L147)
  For each hidden neuron, count how many input pixels it receives.
- `run_all(checkpoint_path: str = None)` (L179)

## agent-data/sessions.db `DB`

## commands.txt `TXT`

## configs/ablation_full.yaml `YAML`

## configs/ablation_learned.yaml `YAML`

## configs/ablation_random_sparse.yaml `YAML`

## configs/ablation_transfer.yaml `YAML`

## configs/base.yaml `YAML`

## configs/fashion_mnist_baseline.yaml `YAML`

## configs/lsm_shd_baseline.yaml `YAML`

## configs/mnist_baseline.yaml `YAML`

## data/.DS_Store `unknown`

## data/DVSGesture/.DS_Store `unknown`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user24_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user25_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user26_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user27_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user28_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTest/user29_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user01_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user02_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user03_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user04_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user05_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user06_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user07_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user08_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user09_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user10_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user11_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user12_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user13_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user14_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user15_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user16_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user17_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user18_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user19_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user20_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user21_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user22_natural/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_fluorescent_led/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_lab/9.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/0.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/1.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/10.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/2.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/3.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/4.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/5.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/6.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/7.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/8.npy `NPY`

## data/DVSGesture/ibmGestureTrain/user23_led/9.npy `NPY`

## data/NMNIST/.DS_Store `unknown`

## data/SHD/shd_test.h5 `H5`

## data/SHD/shd_test.h5.zip `ZIP`

## data/SHD/shd_train.h5 `H5`

## data/SHD/shd_train.h5.zip `ZIP`

## docs/context.md `MD`

## docs/context_2.md `MD`

## docs/dev_guide.md `MD`

## docs/lsm_current_baseline.md `MD`

## docs/lsm_implementation_plan.md `MD`

## docs/text.txt `TXT`

## experiments/.DS_Store `unknown`

## experiments/ablation_full_2603231910/checkpoints/best.pt `PT`

## experiments/ablation_full_2603231910/config.yaml `YAML`

## experiments/ablation_full_2603231910/logs/best.txt `TXT`

## experiments/ablation_full_2603231910/logs/train.jsonl `JSONL`

## experiments/ablation_random_sparse_2603231910/checkpoints/best.pt `PT`

## experiments/ablation_random_sparse_2603231910/config.yaml `YAML`

## experiments/ablation_random_sparse_2603231910/logs/best.txt `TXT`

## experiments/ablation_random_sparse_2603231910/logs/train.jsonl `JSONL`

## experiments/ablation_transfer_2603232014/checkpoints/best.pt `PT`

## experiments/ablation_transfer_2603232014/config.yaml `YAML`

## experiments/ablation_transfer_2603232014/logs/train.jsonl `JSONL`

## experiments/dvs_gesture_baseline_2604031312/.DS_Store `unknown`

## experiments/dvs_gesture_baseline_2604031312/checkpoints/best.pt `PT`

## experiments/dvs_gesture_baseline_2604031312/config.yaml `YAML`

## experiments/dvs_gesture_baseline_2604031312/figures/input_receptive_field.png `PNG`

## experiments/dvs_gesture_baseline_2604031312/figures/theta_distribution.png `PNG`

## experiments/dvs_gesture_baseline_2604031312/figures/threshold_distribution.png `PNG`

## experiments/dvs_gesture_baseline_2604031312/figures/topology.png `PNG`

## experiments/dvs_gesture_baseline_2604031312/figures/training_curves.png `PNG`

## experiments/dvs_gesture_baseline_2604031312/logs/train.jsonl `JSONL`

## experiments/fashion_1024_mnist_baseline_2603231941/checkpoints/best.pt `PT`

## experiments/fashion_1024_mnist_baseline_2603231941/config.yaml `YAML`

## experiments/fashion_1024_mnist_baseline_2603231941/logs/best.txt `TXT`

## experiments/fashion_1024_mnist_baseline_2603231941/logs/train.jsonl `JSONL`

## experiments/fashion_512_mnist_baseline_2603231918/.DS_Store `unknown`

## experiments/fashion_512_mnist_baseline_2603231918/checkpoints/best.pt `PT`

## experiments/fashion_512_mnist_baseline_2603231918/config.yaml `YAML`

## experiments/fashion_512_mnist_baseline_2603231918/logs/best.txt `TXT`

## experiments/fashion_512_mnist_baseline_2603231918/logs/train.jsonl `JSONL`

## experiments/lsm_shd_260403221250_baseline_A/checkpoints/best.pt `PT`

## experiments/lsm_shd_260403221250_baseline_A/config.yaml `YAML`

## experiments/lsm_shd_260403221250_baseline_A/logs/train.jsonl `JSONL`

## experiments/lsm_shd_260403221253_baseline_B/checkpoints/best.pt `PT`

## experiments/lsm_shd_260403221253_baseline_B/config.yaml `YAML`

## experiments/lsm_shd_260403221253_baseline_B/logs/train.jsonl `JSONL`

## experiments/lsm_shd_B_p01_260405113256/checkpoints/best.pt `PT`

## experiments/lsm_shd_B_p01_260405113256/config.yaml `YAML`

## experiments/lsm_shd_B_p01_260405113256/logs/train.jsonl `JSONL`

## experiments/lsm_shd_B_p02_260405113305/checkpoints/best.pt `PT`

## experiments/lsm_shd_B_p02_260405113305/config.yaml `YAML`

## experiments/lsm_shd_B_p02_260405113305/logs/train.jsonl `JSONL`

## experiments/lsm_shd_B_p03_260405113343/checkpoints/best.pt `PT`

## experiments/lsm_shd_B_p03_260405113343/config.yaml `YAML`

## experiments/lsm_shd_B_p03_260405113343/logs/train.jsonl `JSONL`

## experiments/lsm_shd_B_p05_260405113400/checkpoints/best.pt `PT`

## experiments/lsm_shd_B_p05_260405113400/config.yaml `YAML`

## experiments/lsm_shd_B_p05_260405113400/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_260429113640/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_260429113640/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_260429113640/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_dynwarm_260429115051/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_dynwarm_260429115051/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_dynwarm_260429115051/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_dynwarm_slope_260429122127/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_dynwarm_slope_260429122127/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_dynwarm_slope_260429122127/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta080_std030_w225_260429160413/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta080_std030_w225_260429160413/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta080_std030_w225_260429160413/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_260429122610/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_260429122610/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_260429122610/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s43_260429141323/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s43_260429141323/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s43_260429141323/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s44_260429141328/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s44_260429141328/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s44_260429141328/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s45_260429155902/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s45_260429155902/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s45_260429155902/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s45_260429160420/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s45_260429160420/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_s45_260429160420/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_freeze60_s43_260429194121/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_freeze60_s43_260429194121/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_freeze60_s43_260429194121/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_s42_260429164913/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_s42_260429164913/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_s42_260429164959/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_s42_260429164959/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_s42_260429164959/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_s43_260429173225/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_s43_260429173225/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_s43_260429173225/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr005_s43_260429181501/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr005_s43_260429181501/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr005_s43_260429181501/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr0075_s43_260429190443/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr0075_s43_260429190443/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr0075_s43_260429190443/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s42_260429212915/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s42_260429212915/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s42_260429212915/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s43_260429201447/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s43_260429201447/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s43_260429201447/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s43_260429225102/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s43_260429225102/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s43_260429225102/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s44_260429234301/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s44_260429234301/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s44_260429234301/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta110_w225_260429155907/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta110_w225_260429155907/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta110_w225_260429155907/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta120_w225_260429150522/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta120_w225_260429150522/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta120_w225_260429150522/logs/train.jsonl `JSONL`

## experiments/lsm_shd_C_freeze_w_theta150_w225_260429122446/checkpoints/best.pt `PT`

## experiments/lsm_shd_C_freeze_w_theta150_w225_260429122446/config.yaml `YAML`

## experiments/lsm_shd_C_freeze_w_theta150_w225_260429122446/logs/train.jsonl `JSONL`

## experiments/lsm_shd_c_p01_260406131637/checkpoints/best.pt `PT`

## experiments/lsm_shd_c_p01_260406131637/config.yaml `YAML`

## experiments/lsm_shd_c_p01_260406131637/logs/train.jsonl `JSONL`

## experiments/lsm_shd_grad_r_w_theta100_w225_s44_260430154057/checkpoints/best.pt `PT`

## experiments/lsm_shd_grad_r_w_theta100_w225_s44_260430154057/config.yaml `YAML`

## experiments/lsm_shd_grad_r_w_theta100_w225_s44_260430154057/logs/train.jsonl `JSONL`

## experiments/lsm_shd_grad_r_w_theta100_w225_s44_260430165404/checkpoints/best.pt `PT`

## experiments/lsm_shd_grad_r_w_theta100_w225_s44_260430165404/config.yaml `YAML`

## experiments/lsm_shd_grad_r_w_theta100_w225_s44_260430165404/logs/train.jsonl `JSONL`

## experiments/lsm_shd_rs_p000_260428165253/checkpoints/best.pt `PT`

## experiments/lsm_shd_rs_p000_260428165253/config.yaml `YAML`

## experiments/lsm_shd_rs_p000_260428165253/logs/train.jsonl `JSONL`

## experiments/lsm_shd_rs_p002_w225_260428201309/checkpoints/best.pt `PT`

## experiments/lsm_shd_rs_p002_w225_260428201309/config.yaml `YAML`

## experiments/lsm_shd_rs_p002_w225_260428201309/logs/train.jsonl `JSONL`

## experiments/lsm_shd_rs_p002_w225_freeze_w_260428231428/checkpoints/best.pt `PT`

## experiments/lsm_shd_rs_p002_w225_freeze_w_260428231428/config.yaml `YAML`

## experiments/lsm_shd_rs_p002_w225_freeze_w_260428231428/logs/train.jsonl `JSONL`

## experiments/lsm_shd_rs_p003_w250_260428212005/checkpoints/best.pt `PT`

## experiments/lsm_shd_rs_p003_w250_260428212005/config.yaml `YAML`

## experiments/lsm_shd_rs_p003_w250_260428212005/logs/train.jsonl `JSONL`

## experiments/lsm_shd_rs_p003_w250_freeze_w_260429011258/checkpoints/best.pt `PT`

## experiments/lsm_shd_rs_p003_w250_freeze_w_260429011258/config.yaml `YAML`

## experiments/lsm_shd_rs_p003_w250_freeze_w_260429011258/logs/train.jsonl `JSONL`

## experiments/lsm_shd_rs_p0041_w225_freeze_w_260429150530/checkpoints/best.pt `PT`

## experiments/lsm_shd_rs_p0041_w225_freeze_w_260429150530/config.yaml `YAML`

## experiments/lsm_shd_rs_p0041_w225_freeze_w_260429150530/logs/train.jsonl `JSONL`

## experiments/lsm_shd_rs_p005_w350_cap300_260429011341/checkpoints/best.pt `PT`

## experiments/lsm_shd_rs_p005_w350_cap300_260429011341/config.yaml `YAML`

## experiments/lsm_shd_rs_p005_w350_cap300_260429011341/logs/train.jsonl `JSONL`

## experiments/lsm_shd_rs_p02_w225_260428163253/checkpoints/best.pt `PT`

## experiments/lsm_shd_rs_p02_w225_260428163253/config.yaml `YAML`

## experiments/lsm_shd_rs_p02_w225_260428163253/logs/train.jsonl `JSONL`

## experiments/lsm_shd_rs_p03_w250_260428163303/checkpoints/best.pt `PT`

## experiments/lsm_shd_rs_p03_w250_260428163303/config.yaml `YAML`

## experiments/lsm_shd_rs_p03_w250_260428163303/logs/train.jsonl `JSONL`

## experiments/mnist_snn_2603231917/checkpoints/best.pt `PT`

## experiments/mnist_snn_2603231917/config.yaml `YAML`

## experiments/mnist_snn_2603231917/figures/input_receptive_field.png `PNG`

## experiments/mnist_snn_2603231917/figures/theta_distribution.png `PNG`

## experiments/mnist_snn_2603231917/figures/threshold_distribution.png `PNG`

## experiments/mnist_snn_2603231917/figures/topology.png `PNG`

## experiments/mnist_snn_2603231917/figures/training_curves.png `PNG`

## experiments/mnist_snn_2603231917/logs/best.txt `TXT`

## experiments/mnist_snn_2603231917/logs/train.jsonl `JSONL`

## experiments/nmnist_baseline_2604030928/.DS_Store `unknown`

## experiments/nmnist_baseline_2604030928/checkpoints/best.pt `PT`

## experiments/nmnist_baseline_2604030928/config.yaml `YAML`

## experiments/nmnist_baseline_2604030928/figures/input_receptive_field.png `PNG`

## experiments/nmnist_baseline_2604030928/figures/theta_distribution.png `PNG`

## experiments/nmnist_baseline_2604030928/figures/threshold_distribution.png `PNG`

## experiments/nmnist_baseline_2604030928/figures/topology.png `PNG`

## experiments/nmnist_baseline_2604030928/figures/training_curves.png `PNG`

## experiments/nmnist_baseline_2604030928/logs/train.jsonl `JSONL`

## experiments/smoke_dynwarm_slope_260429120437/checkpoints/best.pt `PT`

## experiments/smoke_dynwarm_slope_260429120437/config.yaml `YAML`

## experiments/smoke_dynwarm_slope_260429120437/logs/train.jsonl `JSONL`

## experiments/smoke_freeze_w_260428231204/checkpoints/best.pt `PT`

## experiments/smoke_freeze_w_260428231204/config.yaml `YAML`

## experiments/smoke_freeze_w_260428231204/logs/train.jsonl `JSONL`

## requirements.txt `TXT`

## scripts/diagnose_liquid.py `Python`

### 클래스 / 타입
- **RunningStats** (L34)
  Streaming scalar stats for tensor values.
  - `__init__`
  - `update`
  - `as_dict`

### 함수
- `parse_args()` (L69)
- `load_checkpoint_if_requested(model, checkpoint_path: str | None, device) None` (L106)
- `fmt_stats(stats: dict[str, float]) str` (L119)
- `print_header(title: str) None` (L126)
- `print_config_summary(cfg, model, checkpoint_path: str | None) None` (L130)
- `print_parameter_sanity(cfg, model) None` (L151)
- `print_recurrent_sparsity(model, tau: float) None` (L178)
- `tensor_summary(x: torch.Tensor) str` (L197)
- `print_recurrent_weight_stats(model, binary_mask: torch.Tensor) None` (L208)
- `collect_batches(loader, n_batches: int) list[tuple[torch.Tensor, torch.Tensor]]` (L252)
- `print_input_spike_stats(batches: list[tuple[torch.Tensor, torch.Tensor]]) None` (L261)
- `run_liquid_diagnostics(model, batches, device, tau: float) dict` (L289)
- `print_current_and_firing_stats(diag: dict) None` (L342)
- `collect_samples_by_class(
    loader, n_classes: int, samples_per_class: int
) dict[int, list]` (L384)
- `liquid_mean_rate(model, batch: torch.Tensor, device, tau: float) torch.Tensor` (L404)
- `print_class_separation(
    model, loader, device, tau: float, n_classes: int, samples_per_class: int
) None` (L425)
- `main()` (L488)

## scripts/evaluate.py `Python`

### 함수
- `main()` (L23)

## scripts/train.py `Python`

### 함수
- `main()` (L26)

## scripts/train_lsm.py `Python`

### 함수
- `main()` (L13)

## scripts/upload_wandb.py `Python`

### 함수
- `load_config_yaml(exp_dir: Path) dict` (L27)
- `load_jsonl(exp_dir: Path) list[dict]` (L35)
- `upload(exp_dir: Path, project: str, entity: str | None)` (L49)
- `main()` (L95)

## scripts/visualize.py `Python`

### 함수
- `main()` (L32)

## src/.DS_Store `unknown`

## src/__init__.py `Python`

## src/data/__init__.py `Python`

## src/data/loaders.py `Python`

### 클래스 / 타입
- **_TonicDataset** (L22)
  Generic tonic dataset wrapper: events → [T, C*H*W] float spike tensor.
  - `__init__`
  - `__len__`
  - `__getitem__`
- **_SHDDataset** (L66)
  SHD dataset: spike events → (T, 700) binned tensor.
  - `__init__`
  - `__len__`
  - `__getitem__`

### 함수
- `_flat_normalized_transform(mean: float, std: float)` (L11)
- `_make_nmnist(root: str, train: bool, T: int)` (L39)
- `_make_dvs_gesture(root: str, train: bool, T: int)` (L50)
- `_make_shd(root: str, train: bool, T: int)` (L94)
- `get_dataloaders(cfg) tuple` (L100)
  Return (train_loader, test_loader) for the dataset specified in cfg.

## src/evaluation/__init__.py `Python`

## src/evaluation/evaluate.py `Python`

### 함수
- `_is_lsm(cfg: Config) bool` (L10)
  Detect if config describes an LSM model.
- `get_device() torch.device` (L15)
- `load_model(checkpoint_path: str, cfg: Config, device: torch.device)` (L23)
- `run_evaluation(checkpoint_path: str, cfg: Config) tuple` (L34)

## src/evaluation/visualize.py `Python`

### 함수
- `plot_training_curves(history: list, save_path: str)` (L23)
- `plot_topology(model, save_path: str)` (L53)
- `plot_theta_distribution(model, save_path: str)` (L71)
- `plot_threshold_distribution(model, save_path: str)` (L90)
- `plot_input_connectivity(model, save_path: str)` (L111)
  Visualise first layer's input→hidden connectivity as a 28×28 heatmap.
- `lsm_plot_training_curves(history: list, save_path: str)` (L132)
- `lsm_plot_topology(model, save_path: str)` (L184)
  Visualise liquid recurrent connectivity mask.
- `lsm_plot_theta_distribution(model, save_path: str)` (L217)
  Visualise sigma(theta) distribution for the liquid layer.
- `lsm_plot_threshold_distribution(model, save_path: str)` (L235)
  Visualise learned neuron thresholds and beta (membrane decay).
- `lsm_plot_weight_distribution(model, save_path: str)` (L255)
  Visualise effective weight magnitude distribution.
- `run_all(checkpoint_path: str, cfg, figures_dir: str | None = None)` (L278)

## src/lsm/__init__.py `Python`

## src/lsm/model.py `Python`

### 클래스 / 타입
- **InputProjection** (L26)
  Fixed random sparse connections from input to liquid. Mixed excitatory/inhibitory.
  - `__init__`
  - `forward`
- **LiquidLayer** (L50)
  Recurrent liquid layer with topology learning.
  - `__init__`
  - `beta`
  - `sample_epoch_mask`
  - `unlock_epoch_mask`
  - `sample_mask`
  - `get_effective_weight`
  - `forward`
  - `sparsity`
  - `get_binary_mask`
- **LSMModel** (L234)
  - `__init__`
  - `forward`
  - `sparsity_loss`
  - `commitment_loss`
  - `sparsity_info`
  - `firing_rate_info`

## src/lsm/trainer.py `Python`

### 함수
- `get_device() torch.device` (L30)
- `get_tau(epoch: int, cfg: Config, warmup_epochs: int | None = None) float` (L38)
- `_make_experiment_dir(cfg: Config) Path` (L52)
- `build_model(cfg: Config, device: torch.device) LSMModel` (L67)
- `_compute_loss(rates, labels, model, cfg)` (L95)
- `_evaluate(model: LSMModel, loader, device: torch.device, tau: float) float` (L102)
- `_metric_improved(
    metric_name: str, value: float, best_value: float | None, min_delta: float
) bool` (L115)
- `_select_warmup_metric(metric_name: str, row: dict) float` (L125)
- `_warmup_score(metric_name: str, value: float) float` (L134)
- `_warmup_slope(scores: list[float], window: int) float | None` (L138)
- `train(cfg: Config) tuple` (L145)

## src/models/__init__.py `Python`

## src/models/layers.py `Python`

### 클래스 / 타입
- **SurrogateSpike** (L15)
  - `forward`
  - `backward`
- **GumbelLIFLayer** (L88)
  Single LIF layer with topology controlled by `mode`.
  - `__init__`
  - `beta`
  - `forward`
  - `get_binary_mask`
  - `sparsity`

### 함수
- `spike_fn(x)` (L29)
- `gumbel_sigmoid(logits, tau=1.0, hard=False)` (L33)
- `sigmoid_ste(logits)` (L41)
  Deterministic Straight-Through Estimator for binary mask.
- `gumbel_sigmoid_ste(logits, tau=1.0)` (L57)
  Gumbel-Sigmoid with Straight-Through Estimator.

## src/models/snn.py `Python`

### 클래스 / 타입
- **SNNModel** (L20)
  - `__init__`
  - `forward`
  - `sparsity_loss`
  - `commitment_loss`
  - `sparsity_info`
  - `load_topology_from_checkpoint`

## src/training/__init__.py `Python`

## src/training/losses.py `Python`

### 함수
- `total_loss(
    rates, labels, model, lambda_sparse: float, lambda_commit: float
) torch.Tensor` (L12)

## src/training/trainer.py `Python`

### 함수
- `get_device() torch.device` (L28)
- `get_tau(epoch: int, cfg: Config) float` (L36)
- `_make_experiment_dir(cfg: Config) Path` (L44)
- `_cfg_to_dict(cfg: Config) dict` (L60)
- `_evaluate(model: SNNModel, loader, device: torch.device, tau: float) float` (L66)
- `build_model(cfg: Config, device: torch.device) SNNModel` (L79)
- `train(cfg: Config, resume: bool = False) list` (L96)

## src/utils/__init__.py `Python`

## src/utils/config.py `Python`

### 함수
- `_deep_merge(base: dict, override: dict) dict` (L147)
  Recursively merge override into base (in-place on a copy).
- `_load_yaml(path: str | Path) dict` (L158)
- `_resolve_inheritance(yaml_path: str | Path) dict` (L163)
  Load a YAML file, resolving a `base:` key by merging parent first.
- `_apply_cli_overrides(d: dict, overrides: List[str]) dict` (L177)
  Apply overrides of the form "key=value" or "section.key=value".
- `load_config(
    config_path: str | Path | None = None,
    overrides: List[str] | None = None,
) Config` (L205)
  Load config with optional YAML file and CLI overrides.
