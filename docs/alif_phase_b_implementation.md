# ALIF Phase B Implementation Notes

## Summary

This document records the current Phase B port that adds ALIF neurons to the LSM codebase without changing the default LIF path.

- Default behavior remains `neuron_type: lif`
- ALIF is opt-in through config
- Recurrent topology logic is unchanged
- Training loop structure is unchanged

## Implemented Changes

### 1. Config surface

Added the following `liquid` config fields in [src/utils/config.py](/Users/jeong-yunseong/Documents/programming/snn_gumbel/src/utils/config.py:41):

- `neuron_type`: `lif | alif`
- `alif_rho_init`
- `alif_beta_init`
- `alif_adapt_increment`
- `alif_learn_rho`
- `alif_learn_beta`

Default values preserve the old LIF behavior:

```yaml
liquid:
  neuron_type: lif
  alif_rho_init: 0.9
  alif_beta_init: 0.4
  alif_adapt_increment: 1.0
  alif_learn_rho: false
  alif_learn_beta: false
```

### 2. LiquidLayer ALIF support

Implemented in [src/lsm/model.py](/Users/jeong-yunseong/Documents/programming/snn_gumbel/src/lsm/model.py:57).

ALIF adds an adaptation state `a` on top of the existing LIF membrane dynamics:

```text
a[t] = rho * a[t-1] + adapt_increment * z[t-1]
theta_eff[t] = threshold_base + beta_adapt * a[t]
v[t] = beta_mem * v[t-1] + input_current + recurrent_current
z[t] = spike_fn(v[t] - clamp(theta_eff, min=0.01))
v[t] = v[t] * (1 - z[t])
```

Implementation details:

- `neuron_type='lif'` keeps the old code path
- `neuron_type='alif'` allocates and updates `liquid_a`
- full `theta_eff` is clamped with `min=0.01`
- membrane clamp `[-3.0, 3.0]` is preserved
- `rho_adapt` validation enforces `0 <= rho < 1`
- if `alif_learn_rho=true`, runtime `rho` uses `sigmoid(...)`
- if `alif_learn_beta=true`, runtime `beta_adapt` is clamped with `min=0.0`
- `alif_adapt_increment` scales spike-driven adaptation accumulation and defaults to `1.0`

### 3. Truncated BPTT behavior

Implemented in [src/lsm/model.py](/Users/jeong-yunseong/Documents/programming/snn_gumbel/src/lsm/model.py:489).

When `bptt_truncate > 0`, ALIF detaches:

- `liquid_mem`
- `liquid_spike`
- `liquid_a`

This keeps the adaptation state aligned with the existing truncated-BPTT cutoff.

### 4. Trainer integration

Integrated in [src/lsm/trainer.py](/Users/jeong-yunseong/Documents/programming/snn_gumbel/src/lsm/trainer.py:67).

The trainer now passes through:

- `neuron_type`
- `alif_rho_init`
- `alif_beta_init`
- `alif_adapt_increment`
- `alif_learn_rho`
- `alif_learn_beta`

Optimizer grouping remains unchanged:

- topology parameters come only from `topology_parameters()`
- ALIF parameters are not treated as topology parameters
- if ALIF parameters become learnable later, they stay in the non-topology optimizer group

## Logging

### JSONL fields

Per-epoch JSONL rows now include:

- `neuron_type`
- `mean_adaptation`
- `max_adaptation`

For LIF runs:

- `neuron_type = "lif"`
- adaptation fields are `0.0`

For ALIF runs:

- `neuron_type = "alif"`
- adaptation fields reflect the final forward pass of the epoch

### First-batch ALIF diagnostic

ALIF runs print a one-time first-batch diagnostic in training:

```text
[ALIF] first batch adaptation: mean=...  max=...
```

This is printed through `tqdm.write(...)` and is not a separate JSONL field.

## Added Configs

### Base ALIF config

[configs/lsm_shd_alif.yaml](/Users/jeong-yunseong/Documents/programming/snn_gumbel/configs/lsm_shd_alif.yaml:1)

- inherits from `lsm_shd_baseline.yaml`
- enables ALIF only

Command:

```bash
python scripts/train_lsm.py configs/lsm_shd_alif.yaml seed=42
```

### learned_lowrank + m50p10 ALIF config

[configs/lsm_shd_alif_learned_lowrank_m50p10.yaml](/Users/jeong-yunseong/Documents/programming/snn_gumbel/configs/lsm_shd_alif_learned_lowrank_m50p10.yaml:1)

- inherits from `lsm_shd_C_valrollback_m50p10.yaml`
- switches `recurrent_mode` to `learned_lowrank`
- enables ALIF

Command:

```bash
python scripts/train_lsm.py configs/lsm_shd_alif_learned_lowrank_m50p10.yaml seed=42
```

### Stabilized learned_lowrank + m50p10 ALIF config

[configs/lsm_shd_alif_learned_lowrank_m50p10_stable.yaml](/Users/jeong-yunseong/Documents/programming/snn_gumbel/configs/lsm_shd_alif_learned_lowrank_m50p10_stable.yaml:1)

Overrides:

- `theta_init_mean: -0.7`
- `theta_lowrank_init_std: 0.45`
- `theta_lr_scale: 0.05`
- `noise_scale: 0.05`
- `alif_beta_init: 0.25`

Command:

```bash
python scripts/train_lsm.py configs/lsm_shd_alif_learned_lowrank_m50p10_stable.yaml seed=42
```

### Fixed random-sparse ALIF ablation config

[configs/lsm_shd_alif_random_sparse_p045_fixed.yaml](/Users/jeong-yunseong/Documents/programming/snn_gumbel/configs/lsm_shd_alif_random_sparse_p045_fixed.yaml:1)

- inherits from `lsm_shd_C_valrollback_m50p10.yaml`
- switches to `recurrent_mode: random_sparse`
- uses `recurrent_sparsity: 0.045`
- keeps `train_w_raw: false`
- sets `alif_adapt_increment: 0.25`
- disables topology-learning warmup/freeze logic for cleaner ALIF ablation

Command:

```bash
python scripts/train_lsm.py configs/lsm_shd_alif_random_sparse_p045_fixed.yaml seed=42
```

## Observed Behavior So Far

### Baseline ALIF behavior

- ALIF forward path works
- adaptation state evolves without NaN in smoke tests
- LIF and ALIF are distinguishable in JSONL logs

### learned_lowrank ALIF behavior

Observed during SHD training:

- P1 density can remain very low if initialization is too negative
- P2 density rises after topology unfreeze
- topology gradients can explode under the original lowrank-style settings

The stabilized config was added specifically to reduce:

- overly sparse hard-mask initialization
- excessive topology updates in Phase 2
- excessive ALIF adaptation amplitude

## Constraints Preserved

The implementation does not change:

- default LIF path behavior
- recurrent topology modes and mask generation logic
- Dale's Law
- self-connection masking
- topology validation rollback
- topology freeze logic
- warmup / P1-P2 training structure
- input projection and readout structure

## Current Caveats

- `mean_adaptation` and `max_adaptation` are end-of-forward summaries, not full time-series diagnostics
- no dedicated ALIF-specific evaluation script has been added yet
- ALIF with `learned_lowrank` may need further tuning beyond the current stabilized config
- `alif_learn_rho` and `alif_learn_beta` are wired in, but the current recommended experiments keep them fixed
- `alif_adapt_increment` is now available and is the main knob for softening adaptation without changing `rho` or `beta_adapt`

## Recommended Starting Point

For SHD ALIF experiments under the current codebase, start with:

```bash
python scripts/train_lsm.py configs/lsm_shd_alif_learned_lowrank_m50p10_stable.yaml seed=42
```

Monitor these fields first:

- `sp`
- `topology_grad_pre_clip`
- `mean_adaptation`
- `max_adaptation`
- `val_acc`
