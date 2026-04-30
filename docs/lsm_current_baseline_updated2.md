# LSM Current Baseline

## Purpose

This document fixes the current LSM baseline before further diagnosis or hyperparameter changes. Use this as the reference state for the next steps:

1. reproduce no-recurrence vs `random_sparse` behavior,
2. diagnose firing/current/separation,
3. find the minimum setting where recurrent dynamics help,
4. keep track of which learned-topology claims are already supported and which are still seed-sensitive.

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
| adaptive theta freeze | default off; `theta_adaptive_freeze=false`, `theta_freeze_grad_threshold=30.0`, `theta_freeze_patience=2` |

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

Adaptive theta freeze:

- Applies to trainable-topology modes (`learned`, `grad_r`) when `liquid.theta_adaptive_freeze=true`.
- Trigger condition: after `theta_freeze_min_epoch`, freeze theta once `theta_grad_norm > theta_freeze_grad_threshold` for `theta_freeze_patience` consecutive epochs.
- Current Grad R-STE setting: `theta_freeze_min_epoch=20`, `theta_freeze_grad_threshold=30.0`, `theta_freeze_patience=2`.
- If both fixed `theta_freeze_epoch` and adaptive freeze are enabled, theta should freeze when either condition triggers first.
- Freeze operation must set `theta.requires_grad_(False)`, clear learned-mode epoch noise if present, and lock a deterministic hard mask.

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
- Learned C should use `liquid.train_w_raw=false` and a less sparse hard-mask initialization.
- `theta_init_mean=-1.0`, `theta_init_std=0.5`, `w_raw_init_mean=-2.25` is the first successful learned C setting.

## Learned Topology C Result

The first learned C attempt with the default `theta_init_mean=-2.2` and frozen `w_raw` failed because the learned hard mask was almost off.

| Run | Best test accuracy | Decision |
|-----|-------------------:|----------|
| `p=0.0` no recurrence | 0.5490 | baseline |
| best random recurrent freeze | 0.5499 | random baseline |
| learned C, `theta_init_mean=-2.2`, `train_w_raw=false` | 0.5433 | too sparse, reject |
| learned C, `theta=-1.0,std=0.5,w=-2.25,freeze_w`, seed 42 | 0.5689 | success |
| learned C, same setting, seed 43 | 0.5751 | best tau=0.05 run |
| learned C, same setting, seed 44 | 0.5331 | unstable/reject |
| learned C, same setting, seed 45 | 0.5587 | success, but late theta-grad instability |
| learned C, same setting, `tau_end=0.2`, seed 42 | 0.5782 | tau-stabilized |
| learned C, same setting, `tau_end=0.2`, seed 43 | 0.5795 | current best; late theta-grad still spikes |
| learned C, `theta=-1.2,std=0.5,w=-2.25,freeze_w` | 0.5442 | reject |
| learned C, `theta=-0.8,std=0.3,w=-2.25,freeze_w` | 0.5389 | reject; not density-preserving |
| random sparse, `p=0.041,w=-2.25,freeze_w` | 0.5216 | same-density random control failed |

Successful command:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=learned \
  liquid.train_w_raw=false \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.theta_init_mean=-1.0 \
  liquid.theta_init_std=0.5 \
  experiment_name=lsm_shd_C_freeze_w_theta100_w225
```

Experiment directory:

```text
experiments/lsm_shd_C_freeze_w_theta100_w225_260429122610
```

Diagnostic comparison:

| Metric | no recurrence | learned C success |
|--------|--------------:|------------------:|
| best test acc | 0.5490 | 0.5689 |
| density | 0.0000 | 0.0578 |
| active edges | 0 | 14457 |
| `\|rec\|/\|input\|` | 0.0000 | 0.3261 |
| `\|exc rec\|/\|input\|` | 0.0000 | 0.2816 |
| `\|inh rec\|/\|input\|` | 0.0000 | 0.1936 |
| firing mean | 0.0511 | 0.0804 |
| firing max | 0.4200 | 0.7800 |
| active neurons `>0.01` | 327 / 500 | 398 / 500 |
| active neurons `>0.05` | 182 / 500 | 257 / 500 |
| cosine mean | 0.9676 | 0.9556 |
| cosine min | 0.9234 | 0.9142 |
| clamped fraction | 1.0000 | 0.0000 |

Interpretation:

- The default learned C failure was not evidence against learned topology; it was caused by near-zero hard density.
- `theta_init_mean=-1.0,std=0.5` has an initial hard density near `P(N(-1.0,0.5)>0) ~= 0.023`, not `0.05~0.06`.
- In the successful learned C checkpoints, topology learning raises the hard density into the `0.04~0.06` range, which is high enough for recurrent dynamics to matter.
- Freezing `w_raw` remains important: the successful run has `clamped fraction=0.0000` and stable effective recurrent magnitude around `0.1002`.
- Unlike random sparse recurrence, learned C expands active neurons and improves class separation.
- The current best hypothesis is that learned edge placement, not just recurrent density, produced the gain.

## Learned Topology C: Seed-44 Freeze64 Check

The seed-44 failure needed one more check: whether the problem was only late-stage instability or whether the topology itself was already unfavorable. A scheduled freeze at epoch 64 isolates that question.

Seed-44 freeze64 run:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=learned \
  liquid.train_w_raw=false \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.theta_init_mean=-1.0 \
  liquid.theta_init_std=0.5 \
  liquid.theta_lr_scale=0.3 \
  liquid.theta_freeze_epoch=64 \
  liquid.theta_warmup_epochs=10 \
  liquid.recurrent_sparsity=0.0 \
  schedule.tau_end=0.2 \
  seed=44 \
  experiment_name=lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s44
```

Result:

| Metric | Value |
|--------|------:|
| best test acc | 0.5477 |
| baseline best test acc | 0.5490 |
| delta | -0.0013 |
| density | 0.0587 |
| `|rec| / |input|` | 0.2809 |
| firing mean | 0.0719 / 0.8200 |
| active neurons `>0.05` | 220 / 500 |
| cosine mean / min | 0.9706 / 0.9426 |

Interpretation:

- Freeze64 improved seed 44 from the earlier 0.5331 run, so late-stage theta instability was part of the problem.
- However, it still did not beat the no-recurrence baseline, so freeze64 is a stabilization step, not a full robustness fix.
- Seed 44 now looks like a topology-quality or edge-placement failure, not just a late collapse failure.
- The current conclusion is that learned topology is promising but still seed-sensitive.

Current claim status:

- Confirmed: random-sparse recurrence does not help here.
- Confirmed: learned topology can beat the baseline on some seeds.
- Not yet confirmed: the current learned-C recipe is robustly better than no recurrence across seeds.
- Not yet resolved: whether the remaining seed-44 gap can be closed by topology selection, schedule changes, or a local auxiliary objective.

## Next Step From This Baseline

The next experiment should not be a broad hyperparameter sweep.

Priority order:

1. topology selection or checkpoint choice across several candidate freeze epochs,
2. theta schedule shaping,
3. dynamics-aware regularization,
4. then a local prediction auxiliary loss if topology quality remains seed-sensitive.

Recommended next conceptual move:

- store multiple candidate topology checkpoints,
- compare them with deterministic readout fine-tuning,
- then add a small next-state prediction auxiliary loss if selection alone is not enough.

Seed reproducibility check:

| Run | Seed | Best test accuracy | Density at best ckpt | `\|rec\|/\|input\|` | Firing mean/max | Active neurons `>0.05` | Cosine mean/min | Decision |
|-----|-----:|-------------------:|---------------------:|--------------------:|----------------:|------------------------:|----------------:|----------|
| learned C, `theta=-1.0,w=-2.25,freeze_w` | 42 | 0.5689 | 0.0578 | 0.3261 | 0.0804 / 0.7800 | 257 / 500 | 0.9556 / 0.9142 | success |
| learned C, `theta=-1.0,w=-2.25,freeze_w` | 43 | 0.5751 | 0.0409 | 0.1788 | 0.0625 / 0.6800 | 194 / 500 | 0.9724 / 0.9430 | best tau=0.05 run |
| learned C, `theta=-1.0,w=-2.25,freeze_w` | 44 | 0.5331 | 0.0581 | 0.2756 | 0.0712 / 0.8400 | 214 / 500 | 0.9707 / 0.9421 | unstable/reject |
| learned C, `theta=-1.0,w=-2.25,freeze_w` | 45 | 0.5587 | n/a | n/a | 0.091 / 0.563 late | n/a | n/a | success, unstable late theta-grad |
| learned C, `theta=-1.0,w=-2.25,freeze_w,tau_end=0.2` | 42 | 0.5782 | 0.0610 late | n/a | 0.083 / 0.678 final | n/a | n/a | stable theta-grad |
| learned C, `theta=-1.0,w=-2.25,freeze_w,tau_end=0.2` | 43 | 0.5795 | 0.0620 final | n/a | 0.100 / 0.660 final | n/a | n/a | current best, late theta-grad spikes |
| learned C, `theta=-1.0,w=-2.25,freeze_w,tau_end=0.2,theta_lr_scale=0.05` | 43 | 0.5530 | 0.023 final | n/a | 0.043 / 0.404 final | n/a | n/a | stable but topology under-opens |
| learned C, `theta=-1.0,w=-2.25,freeze_w,tau_end=0.2,theta_lr_scale=0.3,theta_freeze_epoch=64` | 42 | 0.5764 | 0.0579 | 0.3315 | 0.0815 / 0.7800 | 255 / 500 | 0.9594 / 0.9179 | peak mostly preserved |
| learned C, `theta=-1.0,w=-2.25,freeze_w,tau_end=0.2,theta_lr_scale=0.3,theta_freeze_epoch=64` | 43 | 0.5795 | 0.0597 | 0.3177 | 0.0776 / 0.8000 | 254 / 500 | 0.9573 / 0.9214 | current stable candidate |
| learned C, `theta=-1.0,w=-2.25,freeze_w,tau_end=0.2,theta_lr_scale=0.3,theta_freeze_epoch=64` | 44 | 0.5477 | 0.0587 | 0.2809 | 0.0719 / 0.8200 | 220 / 500 | 0.9706 / 0.9426 | freeze did not rescue seed 44 |

Seed reproducibility interpretation:

- Learned C has real upside: seeds 42, 43, and 45 beat the no-recurrence baseline by `+0.0199`, `+0.0261`, and `+0.0097`.
- Across seeds 42-45, sorted best accuracies are `0.5331, 0.5587, 0.5689, 0.5751`; the median is `0.5638`, which is `+0.0148` over the no-recurrence baseline.
- It is not yet robust: seed 44 falls below the no-recurrence baseline despite similar density to seed 42.
- Seed 44 is not weak; it has higher density, stronger recurrent current, and higher max firing than seed 43. The failure mode is therefore likely edge placement, recurrent current scale, or tau/topology instability, not insufficient recurrence.
- The simple mean-rate cosine diagnostic only partly explains accuracy. Seed 43 and seed 44 have similar cosine summaries, but very different test accuracy.
- Seed 43 reached its best accuracy while tau was still high (`tau ~= 0.773`, epoch 39). Later epochs show large topology gradients after tau anneals toward `0.05`, so tau/gradient stabilization is a likely next bottleneck.
- Current target window for learned hard density appears closer to `0.04~0.06`, but density alone is insufficient; edge placement and recurrent current scale both matter.
- Raising `tau_end` from `0.05` to `0.2` improved seed 42 from `0.5689` to `0.5782` and seed 43 from `0.5751` to `0.5795`, making `tau_end=0.2` the current default candidate.
- The stabilization is incomplete: seed 42 kept late `theta_grad_norm` around single/low-double digits, but seed 43 still produced repeated late spikes above `50` even with `tau=0.2`.
- Lowering `theta_lr_scale` to `0.05` removed the gradient spikes but kept sparsity near the initial `~0.023`, so the learned topology failed to open and best test fell to `0.5530`. A partial `theta_lr_scale=0.075` run showed the same under-opening pattern through the mid/late epochs.
- `theta_lr_scale=0.1` with scheduled freeze also under-opened: `theta_freeze_epoch=60` reached only `~0.025` sparsity and best test `0.5570`.
- Keeping `theta_lr_scale=0.3` and freezing later worked better for seeds 42 and 43. `theta_freeze_epoch=64` preserves the seed 43 best test `0.5795`, raises final test from the non-freeze `0.5252` to `0.5663`, and keeps post-freeze `theta_grad=0` with deterministic topology.
- Seed 42 and seed 43 freeze64 checkpoints converge to the same useful regime: density `~0.058~0.060`, `|rec|/|input| ~0.32`, mean firing `~0.08`, active neurons `>0.05` around `255/500`, and cosine mean/min around `0.96/0.92`.
- Seed 44 with the same freeze64 recipe reaches only `0.5477`, below the no-recurrence baseline `0.5490`. Scheduled theta freeze therefore stabilizes late training but does not by itself solve seed-sensitive topology quality.
- The seed 44 freeze64 checkpoint still opens the topology to density `0.0587`, keeps `w_raw` unclamped, and produces nontrivial recurrent current (`|rec|/|input|=0.2809`). Its failure is not caused by under-opening or recurrent-current absence.
- Compared with the successful seed 42/43 freeze64 checkpoints, seed 44 has fewer active neurons above `0.05` (`220/500`) and weaker class separation by cosine summary (`0.9706/0.9426`). The current failure hypothesis moves back toward edge placement and representation geometry, not merely training stability.

Same-density random control result:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.041 \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.train_w_raw=false \
  experiment_name=lsm_shd_rs_p0041_w225_freeze_w
```

This run reached best test accuracy `0.5216`, so matching learned C seed 43 density with random sparse topology is not enough. This strengthens the interpretation that learned edge placement matters.

The `theta=-0.8,std=0.3` run was also rejected:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=learned \
  liquid.train_w_raw=false \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.theta_init_mean=-0.8 \
  liquid.theta_init_std=0.3 \
  seed=42 \
  experiment_name=lsm_shd_C_freeze_w_theta080_std030_w225
```

It reached best test accuracy `0.5389`. This setting was not density-preserving: for `std=0.3`, preserving the initial hard density of `theta=-1.0,std=0.5` requires approximately `theta_init_mean=-0.6`, while targeting a more aggressive `4~6%` initial hard density requires approximately `theta_init_mean=-0.5`.

Current stabilization candidate: keep `theta=-1.0,std=0.5,w=-2.25,freeze_w`, `tau_end=0.2`, and `theta_lr_scale=0.3`, then set `liquid.theta_freeze_epoch=64`. This freezes theta and disables epoch-level Gumbel mask resampling from the start of epoch 64. On seeds 42 and 43, it preserves the best-test window (`0.5764` and `0.5795`) and prevents large late collapse. On seed 44, however, the same recipe reaches only `0.5477`, so freeze64 should be treated as a stabilization candidate, not as a robustness fix.

Seed 44 freeze64 command:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=learned \
  liquid.train_w_raw=false \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.theta_init_mean=-1.0 \
  liquid.theta_init_std=0.5 \
  tau_end=0.2 \
  liquid.theta_lr_scale=0.3 \
  liquid.theta_freeze_epoch=64 \
  seed=44 \
  experiment_name=lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64_s44
```

Seed 44 freeze64 diagnostic:

| Metric | Value |
|--------|------:|
| best test acc | 0.5477 |
| final test acc | 0.5287 |
| density | 0.0587 |
| active edges | 14680 / 250000 |
| `|rec|/|input|` | 0.2809 |
| `|exc rec|/|input|` | 0.2319 |
| `|inh rec|/|input|` | 0.1696 |
| firing mean / max | 0.0719 / 0.8200 |
| active neurons `>0.01` | 388 / 500 |
| active neurons `>0.05` | 220 / 500 |
| cosine mean / min / max | 0.9706 / 0.9426 / 0.9866 |
| clamped fraction | 0.0000 |

Next check: do not assume freeze64 solves seed instability. Either run seed 45 with freeze64 to complete the seed check, or analyze learned edge placement in seed 44 against seeds 42/43 before trying another broad hyperparameter sweep.

## Grad R-STE Adaptive Freeze Result

After fixed freeze experiments showed that `theta_freeze_epoch=64` and even earlier fixed epochs can be too late for Grad R-STE, adaptive theta freezing was added.

Current adaptive setting:

```bash
liquid.theta_adaptive_freeze=true
liquid.theta_freeze_min_epoch=20
liquid.theta_freeze_grad_threshold=30.0
liquid.theta_freeze_patience=2
```

Full command template:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=grad_r \
  liquid.train_w_raw=false \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.theta_init_mean=-1.0 \
  liquid.theta_init_std=0.5 \
  liquid.theta_adaptive_freeze=true \
  liquid.theta_freeze_min_epoch=20 \
  liquid.theta_freeze_grad_threshold=30.0 \
  liquid.theta_freeze_patience=2 \
  seed=<seed> \
  experiment_name=lsm_shd_grad_r_STE_theta100_w225_gfreeze30p2_s<seed>
```

Results:

| Run | Seed | Adaptive freeze epoch | Best test accuracy | Delta vs no recurrence | Decision |
|-----|-----:|-----------------------:|-------------------:|-----------------------:|----------|
| Grad R-STE + adaptive freeze | 42 | 39 | 0.6051 | +0.0561 | current best |
| Grad R-STE + adaptive freeze | 43 | 33 | 0.5808 | +0.0318 | strong success |
| Grad R-STE + adaptive freeze | 44 | 50 | 0.5866 | +0.0376 | rescues seed 44 |
| Grad R-STE + adaptive freeze | 45 | not triggered | 0.5486 | -0.0004 | near baseline; bad-or-stable topology |

Summary statistics over seeds 42/43/44/45:

| Metric | Value |
|--------|------:|
| mean best test accuracy | 0.5803 |
| median best test accuracy | 0.5837 |
| worst seed | 0.5486 |
| best seed | 0.6051 |

Comparison against earlier topology learners:

| Method | Seeds included | Mean | Median | Worst | Best |
|--------|----------------|-----:|-------:|------:|-----:|
| learned C, original tau=0.05 | 42/43/44/45 | 0.5590 | 0.5638 | 0.5331 | 0.5751 |
| Grad R-STE, non-freeze | 42/43/44/45 | 0.5706 | 0.5649 | 0.5486 | 0.6038 |
| Grad R-STE + adaptive freeze | 42/43/44/45 | 0.5803 | 0.5837 | 0.5486 | 0.6051 |

Interpretation:

- Grad R-STE + adaptive freeze is the current strongest LSM topology-learning recipe.
- The adaptive trigger improved seed 42 from `0.6038` to `0.6051`, seed 43 from `0.5711` to `0.5808`, and seed 44 from `0.5587` to `0.5866`.
- The seed-44 rescue is especially important: the same seed failed under Gumbel learned C (`0.5331`) and remained below baseline under learned C freeze64 (`0.5477`), but Grad R-STE + adaptive freeze reached `0.5866`.
- The result weakens the claim that Gumbel-Sigmoid is empirically superior to hard-threshold topology learning.
- The stronger claim is now: gradient-based recurrent topology learning matters, and timely topology stabilization is critical.
- Seed 45 did not trigger adaptive freeze and remained near baseline. This suggests a different failure mode: not gradient explosion, but possibly bad-but-stable topology formation.

Updated next step:

- Treat **Grad R-STE + adaptive freeze** as the current strongest baseline.
- Move to **Grad R-STE + adaptive freeze + prediction auxiliary loss**.
- First target seed 45, because it is the remaining bad-or-stable case that gradient-triggered freeze did not catch.
- Then confirm that prediction auxiliary loss does not degrade strong seeds 42/43/44.


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


Initial prediction auxiliary loss test:
Grad R-STE + adaptive freeze + trace prediction with lambda_pred=0.003 degraded performance.

seed 44:
  baseline Grad R-STE + adaptive freeze: 0.5866
  + pred aux: 0.5353

seed 45:
  baseline Grad R-STE + adaptive freeze: 0.5486
  + pred aux: 0.5464

Interpretation:
  The naive trace-prediction auxiliary objective likely conflicts with the supervised classification objective and can disrupt useful topology formation. Prediction auxiliary loss should not be treated as an automatic stabilizer. Next tests should either lower lambda_pred substantially or restrict the auxiliary gradient path.