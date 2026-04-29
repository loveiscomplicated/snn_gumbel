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
| recurrent raw weight init | `w_raw_init_mean=-4.0`, `w_raw_init_std=0.01` |
| recurrent raw weight training | `train_w_raw=true` |
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
- Recurrent raw weights initialize from `Normal(w_raw_init_mean, w_raw_init_std)`.
- `train_w_raw=false` freezes only recurrent raw weights; beta, threshold, and readout still train.
- `w_raw_max` is only an upper clamp. It does not raise weak initial weights.
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

Optional dynamic warmup can shorten P1 in learned mode:

- `liquid.theta_warmup_dynamic=false` by default, so existing runs use the fixed boundary.
- When enabled, P1 is held for at least `theta_warmup_min_epochs`.
- `theta_warmup_strategy=slope` switches when the recent window's average per-epoch score improvement stays below `theta_warmup_min_delta` for `theta_warmup_patience` checks.
- `theta_warmup_strategy=best` preserves the older best-metric plateau behavior.
- Supported metrics: `test_acc`, `train_acc`, `train_loss`.
- `theta_warmup_epochs` remains the maximum/default P1 boundary.

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
  liquid.recurrent_sparsity=0.0 \
  experiment_name=lsm_shd_rs_p000

python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.02 \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  experiment_name=lsm_shd_rs_p002_w225
```

Current recommended reproduction order:

1. Re-run the no-recurrence baseline with `recurrent_sparsity=0.0`.
2. Train the first recurrent candidate with `p=0.02`, `w_raw_init_mean=-2.25`, `w_raw_max=-2.0`.
3. If the first recurrent candidate is inconclusive, train `p=0.03`, `w_raw_init_mean=-2.5`.

Do not compare recurrent candidates against historical baseline numbers without first re-running the no-recurrence baseline in the current code state.

First fallback candidate:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.03 \
  liquid.w_raw_init_mean=-2.5 \
  liquid.w_raw_max=-2.0 \
  experiment_name=lsm_shd_rs_p003_w250
```

Evaluation:

```bash
python scripts/evaluate.py \
  --config experiments/<exp>/config.yaml \
  --checkpoint experiments/<exp>/checkpoints/best.pt
```

## Known Issues

- Recurrent p-sweep results reported previously show recurrent density hurting accuracy rather than helping.
- Initial diagnosis showed that the default `w_raw_init_mean=-4.0` makes recurrent current very weak. Increasing only `w_raw_max` does not change this, because the clamp is one-sided.
- With `n_liquid=500`, recurrent fan-in is much larger than in `n_liquid=50` diagnostics. Settings that looked reasonable at `N=50` can cause runaway at `N=500`.
- Analysis tools for learned topology structure are not implemented yet.
- SHD loading depends on Tonic; direct HDF5 fallback is not implemented.
- `src/lsm/model.py` still has minor stale comments/imports that do not affect baseline behavior.

## Reproduced No-Recurrence Baseline

The no-recurrence baseline has been re-run in the current code state.

Experiment:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.0 \
  experiment_name=lsm_shd_rs_p000
```

Experiment directory:

```text
experiments/lsm_shd_rs_p000_260428165253
```

Summary:

| Metric | Value |
|--------|------:|
| epochs | 100 completed |
| best epoch | 75 |
| best test accuracy | 0.5490 |
| final test accuracy | 0.5468 |
| best train accuracy | 0.6452 at epoch 75 |
| final train accuracy | 0.6471 |
| recurrent sparsity | 0.0 |
| max grad_norm | 0.8908 |
| max firing rate | 0.3054 |
| final firing rate | 0.0439 / 0.2263 |

Decision:

- The no-recurrence baseline is reproduced successfully.
- Use **54.90% best test accuracy** as the current-code baseline.
- Any random-sparse recurrent candidate must beat 54.90% to count as useful.
- The first recurrent candidate, `p=0.02`, `w_raw_init_mean=-2.25`, `w_raw_max=-2.0`, did not beat this baseline.
- The fallback recurrent candidate, `p=0.03`, `w_raw_init_mean=-2.5`, `w_raw_max=-2.0`, also did not beat this baseline.
- Current conclusion: simple random-sparse recurrence does not help in the current SHD LSM architecture.

## First Random-Sparse Candidate Result

Experiment:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.02 \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  experiment_name=lsm_shd_rs_p002_w225
```

Comparison:

| Run | Best test accuracy | Decision |
|-----|-------------------:|----------|
| `p=0.0` no recurrence | 0.5490 | baseline |
| `p=0.02`, `w_raw_init_mean=-2.25` | 0.5455 | reject |

Stability:

| Metric | Value |
|--------|------:|
| epoch 1 firing rate | 0.0746 / 0.3262 |
| best epoch firing rate | 0.0402 / 0.4142 |
| max firing rate | 0.4146 |
| max grad_norm | 0.8105 |

Interpretation:

- This run was stable and did not show runaway.
- The recurrent path still did not improve classification over the no-recurrence baseline.
- Because train accuracy was also lower than the no-recurrence baseline, this is not primarily an overfitting issue.
- Current interpretation: this random recurrent setting adds weak or misaligned dynamics that slightly interfere with the input-projection/readout baseline.
- Run the fallback candidate once before concluding that simple random recurrence is not useful in the current architecture.

Next fallback:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.03 \
  liquid.w_raw_init_mean=-2.5 \
  liquid.w_raw_max=-2.0 \
  experiment_name=lsm_shd_rs_p003_w250
```

## Fallback Random-Sparse Candidate Result

Experiment:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.03 \
  liquid.w_raw_init_mean=-2.5 \
  liquid.w_raw_max=-2.0 \
  experiment_name=lsm_shd_rs_p003_w250
```

Experiment directory:

```text
experiments/lsm_shd_rs_p003_w250_260428212005
```

Comparison:

| Run | Best test accuracy | Delta vs no recurrence | Decision |
|-----|-------------------:|-----------------------:|----------|
| `p=0.0` no recurrence | 0.5490 | 0.0000 | baseline |
| `p=0.02`, `w_raw_init_mean=-2.25` | 0.5455 | -0.0035 | reject |
| `p=0.03`, `w_raw_init_mean=-2.5` | 0.5269 | -0.0221 | reject |

Stability:

| Metric | Value |
|--------|------:|
| firing rate | about 0.041 / 0.38x |
| grad_norm | about 1.2 to 1.5 |
| recurrent sparsity | 0.030 |
| runaway | no |

Interpretation:

- The fallback was stable but substantially worse than the no-recurrence baseline.
- The issue is not recurrent runaway.
- Random recurrent loops appear to reduce useful input-driven representation quality in the current setup.
- Since train accuracy also drops relative to `p=0.0`, the problem is closer to representation/dynamics degradation than to generalization-only overfitting.
- Do not proceed directly to learned topology C before separating why random-sparse recurrent checkpoints degrade class separation.

Completed checkpoint diagnostics:

```bash
python scripts/diagnose_liquid.py \
  --checkpoint experiments/lsm_shd_rs_p000_260428165253/checkpoints/best.pt \
  --batches 1 \
  --classes 5 \
  --samples-per-class 8 \
  experiments/lsm_shd_rs_p000_260428165253/config.yaml \
  batch_size=8

python scripts/diagnose_liquid.py \
  --checkpoint experiments/lsm_shd_rs_p002_w225_260428201309/checkpoints/best.pt \
  --batches 1 \
  --classes 5 \
  --samples-per-class 8 \
  experiments/lsm_shd_rs_p002_w225_260428201309/config.yaml \
  batch_size=8

python scripts/diagnose_liquid.py \
  --checkpoint experiments/lsm_shd_rs_p003_w250_260428212005/checkpoints/best.pt \
  --batches 1 \
  --classes 5 \
  --samples-per-class 8 \
  experiments/lsm_shd_rs_p003_w250_260428212005/config.yaml \
  batch_size=8
```

Note: for `diagnose_liquid.py`, named argparse options must come before the config path if config overrides are also supplied. Put `batch_size=8` after the config path.

Trained checkpoint comparison:

| Metric | `p=0.0` no recurrence | `p=0.02`, `w=-2.25` | `p=0.03`, `w=-2.5` |
|--------|----------------------:|--------------------:|-------------------:|
| best test acc | 0.5490 | 0.5455 | 0.5269 |
| `\|recurrent\| / \|input\|` | 0.0000 | 0.0962 | 0.1317 |
| firing mean | 0.0511 | 0.0459 | 0.0460 |
| firing max | 0.4200 | 0.6700 | 0.6600 |
| active neurons `>0.01` | 327 / 500 | 269 / 500 | 240 / 500 |
| active neurons `>0.05` | 182 / 500 | 127 / 500 | 125 / 500 |
| cosine mean | 0.9676 | 0.9798 | 0.9790 |
| cosine min | 0.9234 | 0.9549 | 0.9545 |
| effective recurrent weight mean | 0.0000 | 0.1269 | 0.1268 |
| clamped fraction | 1.0000 | 1.0000 | 0.9999 |

Checkpoint diagnostic conclusion:

- Random recurrence increases recurrent current, but does not improve classification.
- Random recurrence lowers mean firing and sharply reduces the number of active liquid neurons.
- Class mean-rate vectors become more similar: cosine mean worsens from `0.9676` to about `0.979`.
- The random-sparse recurrent candidates are stable, so the failure is not a runaway issue.
- Trained recurrent weights saturate against `w_raw_max`: almost all active recurrent edges use nearly the same effective magnitude, `softplus(-2) ~= 0.1269`.
- Current conclusion: simple `random_sparse` recurrence acts like a random mask plus Dale sign plus near-uniform recurrent magnitude, and this degrades SHD liquid representations in the current architecture.

Next decision:

- Do not move directly to learned topology C.
- Move to cause separation first: check `w_raw` clamp saturation, recurrent E/I sign balance, and whether freezing or weakening recurrent weights avoids active-neuron collapse.

Implemented support for cause separation:

- `liquid.train_w_raw` config flag.
- `scripts/diagnose_liquid.py` E/I edge balance and E/I recurrent current output.
- `diagnose_liquid.py` accepts named options and `key=value` overrides more flexibly via `parse_known_args()`.

First `w_raw` freeze ablation:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.02 \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.train_w_raw=false \
  experiment_name=lsm_shd_rs_p002_w225_freeze_w
```

## Cause-Separation Ablation Results

The first cause-separation experiments tested whether random-sparse failure came from random recurrent topology itself or from trainable recurrent magnitude saturating against `w_raw_max`.

Training results:

| Run | Best test accuracy | Delta vs no recurrence | Decision |
|-----|-------------------:|-----------------------:|----------|
| `p=0.0` no recurrence | 0.5490 | 0.0000 | baseline |
| `p=0.02`, `w=-2.25`, `train_w_raw=true` | 0.5455 | -0.0035 | reject |
| `p=0.02`, `w=-2.25`, `train_w_raw=false` | 0.5499 | +0.0009 | best random recurrent so far |
| `p=0.03`, `w=-2.5`, `train_w_raw=true` | 0.5269 | -0.0221 | reject |
| `p=0.03`, `w=-2.5`, `train_w_raw=false` | 0.5367 | -0.0123 | reject |
| `p=0.05`, `w=-3.5`, `w_raw_max=-3.0` | 0.5477 | -0.0013 | near baseline, reject |

Checkpoint diagnostic comparison:

| Metric | `p=0.0` | `p=0.02,w=-2.25,train_w=true` | `p=0.02,w=-2.25,train_w=false` | `p=0.03,w=-2.5,train_w=false` |
|--------|--------:|-------------------------------:|--------------------------------:|--------------------------------:|
| best test acc | 0.5490 | 0.5455 | 0.5499 | 0.5367 |
| `\|rec\|/\|input\|` | 0.0000 | 0.0962 | 0.0736 | 0.0766 |
| `\|exc rec\|/\|input\|` | 0.0000 | not measured | 0.0456 | 0.0441 |
| `\|inh rec\|/\|input\|` | 0.0000 | not measured | 0.0576 | 0.0729 |
| firing mean | 0.0511 | 0.0459 | 0.0445 | 0.0434 |
| active neurons `>0.01` | 327 | 269 | 283 | 255 |
| active neurons `>0.05` | 182 | 127 | 124 | 118 |
| cosine mean | 0.9676 | 0.9798 | 0.9789 | 0.9798 |
| clamped fraction | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

Interpretation:

- `w_raw` training is a major failure mode. When enabled, recurrent weights saturate at the upper clamp and random recurrence hurts performance.
- Freezing `w_raw` removes clamp saturation and recovers baseline-level performance for `p=0.02`.
- Increasing density from `p=0.02` to `p=0.03` still hurts even with frozen `w_raw`.
- The `p=0.03` frozen run shows stronger inhibitory recurrent current, fewer active neurons, and worse accuracy.
- Weak-cap recurrence (`p=0.05`, `w=-3.5`, `w_raw_max=-3.0`) avoids catastrophic degradation but does not beat the baseline.
- Current conclusion: random recurrence must remain weak and sparse. The only useful random recurrent setting so far is effectively baseline-level, not a robust improvement.

Updated next decision:

- Do not train learned C with freely trainable `w_raw`.
- If learned topology C is attempted next, use `liquid.train_w_raw=false` first.
- Learned C should test whether learned topology can improve edge placement while avoiding recurrent magnitude saturation.

## Recent Diagnostic Summary

For `n_liquid=500`, the current useful random-sparse diagnostic window is:

| recurrent p | w_raw_init_mean | \|rec\|/\|input\| | mean firing rate | max firing rate | cosine mean | Status |
|-------------|-----------------|------------------|------------------|-----------------|-------------|--------|
| 0.01 | -2.50 | 0.0590 | 0.0827 | 0.5700 | 0.9807 | weak |
| 0.01 | -2.25 | 0.0759 | 0.0840 | 0.5700 | 0.9810 | weak |
| 0.02 | -2.50 | 0.1106 | 0.0882 | 0.5700 | 0.9818 | candidate |
| 0.02 | -2.25 | 0.1455 | 0.0919 | 0.5700 | 0.9825 | first candidate |
| 0.03 | -2.50 | 0.1621 | 0.0943 | 0.5900 | 0.9828 | fallback candidate |
| 0.03 | -2.25 | 0.2185 | 0.1007 | 0.5800 | 0.9838 | stronger candidate |
| 0.05 | -2.50 | 0.2969 | 0.1113 | 0.5800 | 0.9852 | upper-bound candidate |
| 0.05 | -2.25 | 0.4474 | 0.1317 | 0.6100 | 0.9873 | likely too strong |
| 0.05 | -2.00 | 2.3414 | 0.5562 | 0.9600 | 0.9967 | reject |

Training health check:

- A healthy first epoch should have mean firing roughly in the diagnostic range, not near `0.9`.
- If `max_firing_rate > 0.9` from epoch 1, stop the run and lower recurrent strength.
- Prefer sequential full runs. Parallel full runs make runaway detection slower and can overload GPU/MPS memory.
