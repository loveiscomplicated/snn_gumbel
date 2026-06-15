# Learned Input Projection Summary

Date: 2026-06-15

This note summarizes two completed 4-seed comparisons around the accepted `learned_lowrank r16 + validation rollback m50p10` SHD setting:

- Baseline: `learned_lowrank r16 + validation rollback m50p10`
- Variant A: the same setting with `learned_sparse` input projection
- Variant B: Variant A plus FDI calibration

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
- Learned input projection + FDI:
  - `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_42_260615123903`
  - `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_43_260615123908`
  - `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_44_260615123913`
  - `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_45_260615123919`

## Headline

Under the same `learned_lowrank r16 + m50p10` validation-selection protocol, making the sparse input projection learnable produced a large positive shift in `test @ best val`, and adding FDI on top of that improved the mean again while tightening variance substantially.

- Baseline mean/std/worst: `0.5919 / 0.0126 / 0.5826`
- Learned-input-proj mean/std/worst: `0.6724 / 0.0283 / 0.6268`
- Learned-input-proj + FDI mean/std/worst: `0.7011 / 0.0093 / 0.6939`
- Mean gain: `+0.0805`
- Worst-seed gain: `+0.0442`
- All four seeds improved.
- FDI-on-top mean gain vs learned-input-only: `+0.0287`
- FDI-on-top worst-seed gain vs learned-input-only: `+0.0671`

The first intervention raised accuracy but increased variance relative to the baseline. The FDI variant then moved the mean up again and, on this 4-seed batch, reduced spread materially.

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

## FDI on Learned Input Projection

The relevant comparison baseline for FDI is not the old fixed-input model. It is the already-improved `learned_sparse input projection` regime above.

### Seed-Level Results

| Seed | Learned input proj test @ best val | Learned input proj + FDI test @ best val | Delta |
| --- | ---: | ---: | ---: |
| 42 | `0.6873` | `0.6948` | `+0.0075` |
| 43 | `0.6731` | `0.6939` | `+0.0208` |
| 44 | `0.7023` | `0.7169` | `+0.0146` |
| 45 | `0.6268` | `0.6988` | `+0.0720` |

### Aggregate Comparison

| Setting | Mean | Std | Median | Worst | Best |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned sparse input projection | `0.6724` | `0.0283` | `0.6802` | `0.6268` | `0.7023` |
| + FDI calibration | `0.7011` | `0.0093` | `0.6968` | `0.6939` | `0.7169` |
| Delta | `+0.0287` | `-0.0190` | `+0.0166` | `+0.0671` | `+0.0146` |

### FDI Selections

| Seed | Input scale | Recurrent scale | Threshold scale | Probe mean rate (Hz) | Probe silent frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | `1.25` | `1.0` | `0.75` | `9.3832` | `0.156` |
| 43 | `1.00` | `1.0` | `0.75` | `8.9983` | `0.216` |
| 44 | `1.25` | `1.0` | `0.75` | `9.4062` | `0.180` |
| 45 | `1.25` | `1.0` | `0.75` | `9.4921` | `0.172` |

### Interpretation

- On this batch, FDI helped in all four seeds relative to the already-strong learned-input-projection regime.
- The most important movement is the floor: worst-seed `test @ best val` improved from `0.6268` to `0.6939`.
- The variance reduction is also large: population std `0.0283 -> 0.0093`. This looks less like a lucky upside-only effect and more like a stabilization of initialization quality.
- The selected FDI pattern was consistent: all four runs chose `threshold_scale=0.75`, and three of four chose `input_scale=1.25`, while recurrent scale stayed at `1.0`.

## Interpretation

- The intervention is not a marginal tweak. On this 4-seed batch it shifts the entire distribution upward.
- The strongest evidence is the worst-seed improvement: `0.5826 -> 0.6268`. This suggests the gain is not only driven by one favorable seed.
- Seed variance increased, so the change currently looks like a higher-performing but less tightly clustered regime.
- Relative to the previous accepted baseline, the learned input projection should be treated as a materially stronger candidate on `test @ best val`.
- Relative to the learned-input-projection regime, the FDI variant currently looks stronger again and noticeably more stable on the same metric.

## Caveats

- This summary is limited to the 4 completed seeds `42-45`.
- The document mixes two comparisons: baseline vs learned input projection, and learned input projection vs learned input projection + FDI. The second is the comparison that matters for the new FDI result.
- The comparison isolates the accepted lowrank `m50p10` policy and its two extensions, but it does not explain mechanism.
- No claim is made here about why the gain occurs. That still needs follow-up diagnostics on input projection weights, activity regime changes, topology interaction, and whether the FDI-selected scales remain robust beyond this seed batch.
