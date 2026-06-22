# Lowrank Runaway Diagnostic Run Manifest

Generated: 2026-06-21

This manifest selects existing `experiments/` run directories for `scripts/diagnose_lowrank_runaway.py`. It does not request or imply new training.

## Selection Checks

Each selected run has:

- `config.yaml`
- `logs/train.jsonl`
- `checkpoints/best.pt`
- matching `seed`
- `liquid.recurrent_mode=learned_lowrank`
- `liquid.theta_rank=16`
- `liquid.input_projection_mode=learned_sparse`
- `liquid.train_input_projection=true`
- `liquid.init_mode=fdi_calibrated`
- matching `liquid.readout_mode`, also confirmed from the last `train.jsonl` row when present

The requested bad rerun timestamp was checked explicitly: no requested-prefix directory matching `2606152229` or `26061522xx` exists under `experiments/`.

## Group Summary

| Group | Found | Missing Seeds |
|---|---:|---|
| `lif_learned_input_fdi` | 4 | none |
| `alif_b010_inc0125_biaslr05` | 4 | none |
| `alif_b010_inc0125_biaslr10` | 4 | none |
| `optional_alif_b005_inc010_spike_adaptation_concat` | 4 | none |
| `optional_alif_spike_count` | 4 | none |

## Selected Runs

### lif_learned_input_fdi

- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_42_260615123903`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_43_260615123908`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_44_260615123913`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_45_260615123919`

### alif_b010_inc0125_biaslr05

- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_42_260616014632`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_43_260616083935`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_44_260616083940`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_45_260616083948`

### alif_b010_inc0125_biaslr10

- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_42_260616005645`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_43_260616014512`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_44_260616014523`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_45_260616014531`

### optional_alif_b005_inc010_spike_adaptation_concat

- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_42_260616000131`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_43_260616000140`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_44_260616000147`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_45_260616000153`

### optional_alif_spike_count

- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_42_260615182043`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_43_260615182112`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_44_260615182116`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_45_260615182120`

## Excluded Runs

- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_44_260616005518`
  - missing `checkpoints/best.pt`
  - missing `logs/train.jsonl`
  - `alif_beta_init` expected `0.05`, got `0.1`
  - `alif_adapt_increment` expected `0.1`, got `0.125`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_nonspiking_lif_readout*`
  - 11 broad-prefix candidates ignored because `readout_mode=non_spiking_lif_final_mem`, not one of the requested groups.

## Core Command

First run, using only:

- `lif_learned_input_fdi`
- `alif_b010_inc0125_biaslr05`
- `alif_b010_inc0125_biaslr10`

```bash
python scripts/diagnose_lowrank_runaway.py \
  --run-dirs \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_42_260615123903 \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_43_260615123908 \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_44_260615123913 \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_45_260615123919 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_42_260616014632 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_43_260616083935 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_44_260616083940 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_45_260616083948 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_42_260616005645 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_43_260616014512 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_44_260616014523 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_45_260616014531 \
  --output-dir reports/lowrank_runaway_diagnostics/core_lif_vs_alif_biaslr \
  --num-batches 4 \
  --batch-size 64 \
  --top-k 50 \
  --firing-threshold 0.9 \
  --theta-grad-threshold 50
```

## Optional Command: Core + ALIF b005/inc010 Spike Adaptation

```bash
python scripts/diagnose_lowrank_runaway.py \
  --run-dirs \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_42_260615123903 \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_43_260615123908 \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_44_260615123913 \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_45_260615123919 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_42_260616014632 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_43_260616083935 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_44_260616083940 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_45_260616083948 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_42_260616005645 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_43_260616014512 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_44_260616014523 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_45_260616014531 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_42_260616000131 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_43_260616000140 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_44_260616000147 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_45_260616000153 \
  --output-dir reports/lowrank_runaway_diagnostics/core_plus_alif_b005_inc010_spike_adaptation_concat \
  --num-batches 4 \
  --batch-size 64 \
  --top-k 50 \
  --firing-threshold 0.9 \
  --theta-grad-threshold 50
```

## Optional Command: Core + ALIF Spike Count

```bash
python scripts/diagnose_lowrank_runaway.py \
  --run-dirs \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_42_260615123903 \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_43_260615123908 \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_44_260615123913 \
  experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_45_260615123919 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_42_260616014632 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_43_260616083935 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_44_260616083940 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_45_260616083948 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_42_260616005645 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_43_260616014512 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_44_260616014523 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_45_260616014531 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_42_260615182043 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_43_260615182112 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_44_260615182116 \
  experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_45_260615182120 \
  --output-dir reports/lowrank_runaway_diagnostics/core_plus_alif_spike_count \
  --num-batches 4 \
  --batch-size 64 \
  --top-k 50 \
  --firing-threshold 0.9 \
  --theta-grad-threshold 50
```
