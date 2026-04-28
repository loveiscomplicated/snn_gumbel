# LSM Current Baseline

## Purpose

This document fixes the current LSM baseline before further diagnosis or hyperparameter changes. Use this as the reference state for the next steps:

1. reproduce no-recurrence vs `random_sparse` behavior,
2. diagnose firing/current/separation,
3. find the minimum setting where recurrent dynamics help.

## Code Reference

Baseline-defining files:

| File | Role | Current status |
|------|------|----------------|
| `configs/lsm_shd_baseline.yaml` | SHD LSM experiment config | no local diff |
| `src/lsm/model.py` | LSM model, input projection, liquid, readout | no local diff |
| `src/lsm/trainer.py` | LSM training loop and logging | no local diff |
| `src/data/loaders.py` | SHD and other dataset loaders | no local diff |

Related recent fix:

| File | Role |
|------|------|
| `src/evaluation/evaluate.py` | Evaluation CLI now handles LSM forward without `hard=True` |

## Baseline Config

`configs/lsm_shd_baseline.yaml`:

| Field | Value |
|-------|-------|
| dataset | `shd` |
| input/output | `700 -> 20` |
| T | `100` |
| liquid size | `500` |
| excitatory ratio | `0.8` |
| input connection probability | `0.1` |
| recurrent mode | `learned` |
| recurrent sparsity | `0.1` |
| self connection | `false` |
| theta init | mean `-2.2`, std `0.5` |
| recurrent weight cap | `w_raw_max=-3.0` |
| BPTT | `bptt_truncate=25` |
| theta warmup | `10` epochs |
| theta LR scale | `0.3` |
| Gumbel noise scale | `0.1` |
| input weight scale | `0.3` |
| beta range | `0.7 -> 0.95` |
| threshold range | `0.8 -> 1.5` |
| tau schedule | start `1.0`, hold `15`, anneal `40`, end `0.05` |
| training | `100` epochs, patience `40`, batch size `64` |
| optimizer | Adam, lr `0.001`, lr_min `0.00001`, weight_decay `0.0001` |
| regularization | lambda_sparse `0.1`, lambda_commit `0.01` |
| seed | `42` |

## Model Behavior

Input projection:

- Fixed sparse random projection from input channels to liquid neurons.
- Shape: `(n_input, n_liquid)`.
- Mask: Bernoulli with `p_input`.
- Weight: `torch.randn(...) * input_weight_scale * mask`.
- Registered as a buffer, not trained.

Liquid:

- Recurrent mask modes: `learned`, `random_sparse`, `fixed`, `grad_r`.
- Dale's Law is applied by presynaptic row using `dale_sign` shape `(N, 1)`.
- Self-connections are disabled by default with a diagonal-zero mask.
- Effective recurrent weight:

```python
w_clamped = torch.clamp(w_raw, max=w_raw_max)
signed_w = dale_sign * F.softplus(w_clamped)
w_eff = current_mask * self_conn_mask * signed_w
```

Forward pass:

- `LSMModel.forward(spikes, tau=1.0)` accepts `(batch, T, n_input)`.
- The recurrent mask is sampled once at the start of forward and reused for all timesteps.
- Membrane potential is clamped to `[-3.0, 3.0]`.
- Threshold is clamped with `min=0.01`.
- Readout is a linear layer accumulated over timesteps and divided by `T`.
- Firing-rate stats from the last forward pass are stored for logging.

## Training Behavior

Learned topology uses two phases:

| Phase | Condition | Behavior |
|-------|-----------|----------|
| P1 | `epoch < theta_warmup_epochs` | theta frozen, deterministic hard mask, non-theta params train |
| P2 | `epoch >= theta_warmup_epochs` | theta unfrozen, one Gumbel noise tensor per epoch, STE mask recomputed per batch |

Gradient handling:

- Learned mode uses separate optimizer groups for theta and non-theta parameters.
- Non-theta gradient clipping: `grad_clip_max_norm_w`.
- Theta gradient clipping: `grad_clip_max_norm_theta`.
- Non-learned modes train all trainable parameters in one optimizer group.

Logged metrics:

- `phase`, `lr`, `tau`
- `train_loss`, `train_acc`, `test_acc`
- `sparsity`
- `theta_mean`, `theta_std`
- `grad_norm`, `theta_grad_norm`, `w_raw_grad_norm`
- `mean_firing_rate`, `max_firing_rate`

## Data Pipeline

SHD loader:

- Uses `tonic.datasets.SHD`.
- Converts events to a binary tensor of shape `(T, 700)`.
- Default bin width is `10_000us` (10ms).
- Multiple events in the same time/channel bin are clipped to `1.0`.
- DataLoader uses `num_workers=0` and `pin_memory=False`.

## Baseline Commands

Current learned baseline:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml
```

No/low recurrence comparison candidates:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.0

python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.1
```

Evaluation:

```bash
python scripts/evaluate.py \
  --config experiments/<exp>/config.yaml \
  --checkpoint experiments/<exp>/checkpoints/best.pt
```

## Known Issues

- Recurrent p-sweep results reported previously show recurrent density hurting accuracy rather than helping.
- Analysis tools for learned topology structure are not implemented yet.
- SHD loading depends on Tonic; direct HDF5 fallback is not implemented.
- `src/lsm/model.py` still has minor stale comments/imports that do not affect baseline behavior.
