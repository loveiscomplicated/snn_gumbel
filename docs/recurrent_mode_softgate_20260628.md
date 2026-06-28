# Soft-Gate Recurrent Conductance Mode

Date: 2026-06-28

## Mechanism

Soft-gate modes split recurrent topology into existence and magnitude while keeping
all operations differentiable:

```text
score_ij = src_i * dst_j + bias      # soft_gate_lowrank
score_ij = theta_edge_ij + offset    # soft_gate_edgewise
gate_ij  = sigmoid(score_ij / temp)
mag_ij   = softplus(score_ij)
W_eff_ij = scale * self_mask * dale_sign * gate_ij * mag_ij
```

By default, gate and magnitude share the same score. The optional
`mag_from_separate_param` flag replaces `mag_ij` with `softplus(w_core_ij)` but is
off for the first experiments.

No Gumbel noise, argmax, hard sampling, or threshold replacement is used in these
modes. `temp_final` stays positive so the gate remains soft.

## Density Penalty

The density penalty is gate-only:

```text
soft_density = (gate * density_mask).sum() / density_mask.sum()
density_penalty = (soft_density - target_density) ** 2
total_loss = task_loss + density_penalty_lambda * density_penalty
```

`density_mask` excludes the diagonal. Magnitude and Dale sign are not part of the
penalty, so surviving edges can still become strong.

## Initialization And Annealing

Initialization first applies the quantile recipe so `score > 0` has the requested
initial active fraction, then adjusts only the scalar bias/offset so the actual
penalty tensor `mean(sigmoid(score / temp_init))` matches `target_density_init`.

During phase 1, score topology parameters are frozen. Phase 2 starts the linear
anneal:

- `target_density`: `target_density_init -> target_density_final`
- `temp`: `temp_init -> temp_final`
- duration: `target_anneal_epochs`

## Experiment Staging

Initial configs use the ALIF + learned input projection + FDI + spike/adaptation
readout machine and disable topology rollback/guards:

- `configs/ablation/lsm_shd_alif_SG_lowrank.yaml`
- `configs/ablation/lsm_shd_alif_SG_edgewise.yaml`
- `configs/ablation/lsm_shd_alif_gradR.yaml`

Run `SG_lowrank` first. If it approaches the frozen lowrank constant-g baseline,
the project has a deterministic sparse topology-learning mode. If not, compare
`SG_edgewise` and `gradR` to separate lowrank parameterization, independent-edge
capacity, and stochastic hard-topology instability.

## Log Interpretation

Track these fields together:

- `soft_density`: gate density seen by the penalty.
- `target_density`: current annealed target.
- `hard_active_fraction`: fraction of non-diagonal `W_eff` entries above epsilon.
- `gate_mean`, `gate_p50`, `gate_p95`: whether gates are actually moving.
- `mag_mean`, `mag_max`: whether magnitude is absorbing recurrent strength.
- `score_mean`, `score_std`: whether topology scores collapse or spread.
- `max_firing_rate`, `topology_grad_pre_clip`: runaway or gradient instability.

If `soft_density` follows `target_density` but `hard_active_fraction` does not,
adjust `temp` or the active epsilon before interpreting density-matched results.
