# Learned Input Projection Summary

Date: 2026-06-15

This note summarizes the completed 4-seed comparison between:

- Baseline: `learned_lowrank r16 + validation rollback m50p10`
- Variant: the same setting with `learned_sparse` input projection

Metric policy:

- Primary metric: `test @ best validation`
- Seeds: `42, 43, 44, 45`

Reference runs:

- Baseline:
  - `experiments/lsm_shd_lowrank_r16_valrollback_m50p10_s42_260508112419`
  - `experiments/lsm_shd_lowrank_r16_valrollback_m50p10_s43_260508121237`
  - `experiments/lsm_shd_lowrank_r16_valrollback_m50p10_s44_260508121303`
  - `experiments/lsm_shd_lowrank_r16_valrollback_m50p10_s45_260508121340`
- Learned input projection:
  - `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_42_260615102323`
  - `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_43_260615102345`
  - `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_44_260615102401`
  - `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_45_260615102411`

## Headline

Under the same `learned_lowrank r16 + m50p10` validation-selection protocol, making the sparse input projection learnable produced a large positive shift in `test @ best val`.

- Baseline mean/std/worst: `0.5919 / 0.0126 / 0.5826`
- Learned-input-proj mean/std/worst: `0.6724 / 0.0283 / 0.6268`
- Mean gain: `+0.0805`
- Worst-seed gain: `+0.0442`
- All four seeds improved.

The effect size is large in absolute accuracy terms, but variance also increased relative to the baseline.

## Seed-Level Results

| Seed | Baseline test @ best val | Learned input proj test @ best val | Delta |
| --- | ---: | ---: | ---: |
| 42 | `0.5857` | `0.6873` | `+0.1016` |
| 43 | `0.6135` | `0.6731` | `+0.0596` |
| 44 | `0.5857` | `0.7023` | `+0.1166` |
| 45 | `0.5826` | `0.6268` | `+0.0442` |

## Aggregate Comparison

| Setting | Mean | Std | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline lowrank r16 + m50p10 | `0.5919` | `0.0126` | `0.5857` | `0.5826` | `0.6135` |
| + learned sparse input projection | `0.6724` | `0.0283` | `0.6802` | `0.6268` | `0.7023` |
| Delta | `+0.0805` | `+0.0157` | `+0.0945` | `+0.0442` | `+0.0888` |

## Interpretation

- The intervention is not a marginal tweak. On this 4-seed batch it shifts the entire distribution upward.
- The strongest evidence is the worst-seed improvement: `0.5826 -> 0.6268`. This suggests the gain is not only driven by one favorable seed.
- Seed variance increased, so the change currently looks like a higher-performing but less tightly clustered regime.
- Relative to the previous accepted baseline, the learned input projection should be treated as a materially stronger candidate on `test @ best val`, pending replication and diagnostics.

## Caveats

- This summary is limited to the 4 completed seeds `42-45`.
- The comparison isolates the accepted lowrank `m50p10` policy against the new learned-input-projection variant, but it does not explain mechanism.
- No claim is made here about why the gain occurs. That needs follow-up diagnostics on input projection weights, activity regime changes, and topology interaction.
