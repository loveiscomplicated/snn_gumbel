# Recurrent Mode Ablation Plan

Date: 2026-06-25

This ablation asks whether the current ALIF learned-lowrank recurrent operator
needs both a learned topology mask and `softplus(w_raw)`, or whether a simpler
conductance parameterization is enough.

All configs inherit from the current ALIF best config:

- ALIF `rho=0.85`, `beta=0.10`, `adapt_increment=0.125`
- learned sparse input projection
- FDI initialization
- `spike_adaptation_concat` readout
- `theta_rank=16`, `theta_lr_scale=0.3`, `theta_bias_lr_scale=0.5`
- seeds `42,43,44,45` for confirmation

Stage-1 configs disable validation rollback/freeze so the first comparison is
not decided by topology rollback machinery.

## Mode Matrix

| Config | Formula | Purpose |
| --- | --- | --- |
| A current | `mask_lowrank_gumbel * self_mask * dale_sign * softplus(w_raw)` | current baseline |
| B random floor | `fixed_random_mask * self_mask * dale_sign * softplus(w_raw_frozen)` | recurrent learning floor |
| C softplus W only | `self_mask * dale_sign * softplus(w_raw)` | dense W/conductance upper bound |
| D smooth lowrank | `scale * self_mask * dale_sign * softplus(src @ dst.T + bias)` | lowrank soft conductance, no Gumbel |
| E edgewise soft conductance | `self_mask * dale_sign * softplus(theta_edge)` | independent per-edge conductance baseline |
| F fixed random sparse + learned W | `fixed_random_mask * self_mask * dale_sign * softplus(w_raw)` | fixed placement, learned W |
| G lowrank frozen constant-g | `mask_lowrank_gumbel * self_mask * dale_sign * g` | placement-only mechanism check |
| H lowrank frozen initialized-W | `mask_lowrank_gumbel * self_mask * dale_sign * softplus(w_raw_frozen)` | placement + W heterogeneity |

`D` matches the initial `W_eff` Frobenius norm of the A-style hard lowrank
operator once at initialization. The norm is not constrained during training.
Report `W_eff` norm and effective density together; norm matching does not make
dense soft modes sparsity-matched.

## Staging

Stage 1: fast bracket.

```bash
python scripts/train.py \
  --config configs/ablation/lsm_shd_alif_A_current_learned_lowrank_no_rollback.yaml \
  --seed 42

python scripts/train.py \
  --config configs/ablation/lsm_shd_alif_B_random_floor_fixed_sparse_frozen_w.yaml \
  --seed 42

python scripts/train.py \
  --config configs/ablation/lsm_shd_alif_C_softplus_w_only_dense.yaml \
  --seed 42
```

Early exit:

- If B is close to A, recurrent topology/weight learning is probably not needed.
- If B is far below A but C is close to A, learned recurrent W matters but
  explicit topology learning is probably unnecessary.

Stage 2: parameterization row with axis-2 fixed to soft.

```bash
python scripts/train.py \
  --config configs/ablation/lsm_shd_alif_D_smooth_lowrank_conductance_matched_scale.yaml \
  --seed 42

python scripts/train.py \
  --config configs/ablation/lsm_shd_alif_E_edgewise_soft_conductance.yaml \
  --seed 42
```

Interpretation:

- C vs D vs E isolates dense W, lowrank shared conductance, and independent
  edgewise conductance under soft parameterization.
- A vs D isolates lowrank hard Gumbel discretization versus smooth lowrank
  conductance.

Stage 3: mechanism-only placement/weight split.

```bash
python scripts/train.py \
  --config configs/ablation/lsm_shd_alif_F_fixed_random_sparse_learned_w.yaml \
  --seed 42

python scripts/train.py \
  --config configs/ablation/lsm_shd_alif_G_lowrank_frozen_w_constant_g.yaml \
  --seed 42

python scripts/train.py \
  --config configs/ablation/lsm_shd_alif_H_lowrank_frozen_w_initialized_w.yaml \
  --seed 42
```

Run Stage 3 only if Stage 1/2 suggest topology or placement matters.

## Confirmation

For promising modes only:

```bash
for seed in 42 43 44 45; do
  python scripts/train.py --config <CONFIG> --seed "$seed"
done
```

Primary metric:

- test accuracy at best validation accuracy

Secondary metrics:

- best validation accuracy
- final test accuracy
- `theta_grad_norm_pre_clip`
- `w_raw_grad_pre_clip`
- max firing rate
- adaptation saturation
- `w_eff_fro_norm`
- effective density
- trainable recurrent/topology/W parameter counts

## Interpretation Rules

- B near A: recurrent learning is not a useful cognitive core for this setup.
- C near A and B below A: W learning is useful, but explicit topology is not.
- D near or above A: smooth lowrank conductance is a cleaner replacement for
  mask times W.
- E above D: independent edgewise conductance matters more than lowrank sharing.
- D near E: lowrank sharing is not the bottleneck.
- G good: lowrank placement alone has signal.
- H much better than G: placement plus initial W heterogeneity matters.
- A better than D/E: hard mask plus separate W has useful extra freedom, but
  check whether this depends on rollback/freeze in a follow-up run.
