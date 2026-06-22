# SHD LSM ALIF Learned Input + FDI Summary

Date: 2026-06-15
Updated: 2026-06-16

이 문서는 `learned_lowrank r16 + validation rollback m50p10 + learned_sparse input projection + FDI` 기준선에 ALIF와 readout 변형을 적용한 4-seed 결과를 정리한다.

Primary metric:

- `test accuracy @ best validation accuracy`

Seeds:

- `42, 43, 44, 45`

Population std is reported.

## Experiment Status

완료된 비교:

- A. LIF + learned input + FDI + `spike_count`
- B. ALIF + learned input + FDI + `spike_count`
- C. ALIF + learned input + FDI + `spike_adaptation_concat`
- D. LIF + learned input + FDI + `non_spiking_lif_final_mem`, beta `0.99`, normalized v0.1
- E. ALIF + learned input + FDI + `spike_adaptation_concat`, `alif_beta_init=0.10`, `alif_adapt_increment=0.125`, `theta_bias_lr_scale=1.0`
- F. ALIF + learned input + FDI + `spike_adaptation_concat`, `alif_beta_init=0.10`, `alif_adapt_increment=0.125`, `theta_bias_lr_scale=0.5`

주의:

- 2026-06-15 22:29에 생성된 4개 rerun은 shell 줄바꿈 문제로 `liquid.readout_mode=spike_adaptation_concat`가 적용되지 않았다.
- 2026-06-16 00:01에 생성된 4개 run은 `config.yaml`과 `train.jsonl` 기준으로 `readout_mode: spike_adaptation_concat`가 정상 적용되었다.
- 2026-06-16 01:45 이후 생성된 8개 run은 `alif_beta_init=0.10`, `alif_adapt_increment=0.125` rerun이다. 두 묶음의 config 차이는 seed/experiment name을 제외하면 `liquid.theta_bias_lr_scale`뿐이다.

## Configurations

### LIF Baseline

Base config:

- `configs/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi.yaml`

Important settings:

- `liquid.recurrent_mode: learned_lowrank`
- `liquid.theta_rank: 16`
- `liquid.input_projection_mode: learned_sparse`
- `liquid.train_input_projection: true`
- `liquid.init_mode: fdi_calibrated`
- `liquid.neuron_type: lif`
- `liquid.readout_mode: spike_count`
- topology adaptive freeze/rollback: enabled, `min_epoch=50`, `patience=10`

### ALIF Shared Settings

Same base config with CLI overrides:

- `liquid.neuron_type=alif`
- `liquid.alif_rho_init=0.85`
- `liquid.alif_beta_init=0.05`
- `liquid.alif_adapt_increment=0.10`
- `liquid.alif_learn_rho=false`
- `liquid.alif_learn_beta=false`

Readout variants:

- `liquid.readout_mode=spike_count`
- `liquid.readout_mode=spike_adaptation_concat`

`spike_adaptation_concat`는 별도 LIF readout이 아니라, final classifier 입력을 `[time-averaged spike rate, time-averaged ALIF adaptation trace]`로 확장한 readout이다.

### ALIF Beta 0.10 Increment 0.125 Reruns

Both rerun groups use the same base config and the same ALIF/readout recipe:

- `liquid.neuron_type=alif`
- `liquid.alif_rho_init=0.85`
- `liquid.alif_beta_init=0.10`
- `liquid.alif_adapt_increment=0.125`
- `liquid.alif_learn_rho=false`
- `liquid.alif_learn_beta=false`
- `liquid.readout_mode=spike_adaptation_concat`
- `liquid.theta_lr_scale=0.3`
- `liquid.topology_freeze_min_epoch=50`
- `liquid.topology_freeze_patience=10`
- `liquid.topology_freeze_rollback_best=true`

The only substantive config difference between the two 4-seed groups is:

| Group | `liquid.theta_bias_lr_scale` |
| --- | ---: |
| `alif_beta_init_01_incre_0125` | `1.0` |
| `b010_inc0125_biaslr05` | `0.5` |

## Run Directories

### LIF Baseline

- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_42_260615123903`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_43_260615123908`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_44_260615123913`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_45_260615123919`

### ALIF Spike Count

- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_42_260615182043`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_43_260615182112`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_44_260615182116`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_count_45_260615182120`

### ALIF Spike Adaptation Concat

- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_42_260616000131`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_43_260616000140`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_44_260616000147`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_concat_45_260616000153`

### ALIF Spike Adaptation Concat, beta 0.10 increment 0.125

`theta_bias_lr_scale=1.0`:

- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_42_260616005645`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_43_260616014512`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_44_260616014523`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_alif_beta_init_01_incre_0125_45_260616014531`

`theta_bias_lr_scale=0.5`:

- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_42_260616014632`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_43_260616083935`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_44_260616083940`
- `experiments/lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_45_260616083948`

### Incorrect 22:29 Rerun

이 4개는 저장된 `config.yaml`과 `train.jsonl` 기준으로 `readout_mode: spike_count`였다. `spike_adaptation_concat` 결과로 해석하면 안 된다.

- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_260615222919`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_260615222925`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_260615222930`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_260615222934`

### Non-Spiking LIF Readout v0.1 beta 0.99

- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_nonspiking_lif_readout_260615170456`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_nonspiking_lif_readout_260615170504`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_nonspiking_lif_readout_260615170509`
- `experiments/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi_nonspiking_lif_readout_260615170515`

## Aggregate Results

| Setting | Mean test @ best val | Std | Worst | Best | Mean best test | Mean final test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LIF + learned input + FDI + spike_count | `0.7026` | `0.0104` | `0.6948` | `0.7204` | `0.7098` | `0.7041` |
| ALIF + learned input + FDI + spike_count | `0.6990` | `0.0136` | `0.6807` | `0.7182` | `0.7102` | `0.7016` |
| ALIF + learned input + FDI + spike_adaptation_concat, beta 0.05 inc 0.10 | `0.7037` | `0.0262` | `0.6608` | `0.7306` | `0.7161` | `0.7055` |
| ALIF + learned input + FDI + spike_adaptation_concat, beta 0.10 inc 0.125 bias LR 1.0 | `0.7195` | `0.0062` | `0.7125` | `0.7284` | `0.7289` | `0.7216` |
| ALIF + learned input + FDI + spike_adaptation_concat, beta 0.10 inc 0.125 bias LR 0.5 | `0.7239` | `0.0043` | `0.7186` | `0.7306` | `0.7308` | `0.7243` |
| LIF + learned input + FDI + non-spiking LIF readout v0.1 beta 0.99 | `0.6744` | `0.0088` | `0.6617` | `0.6842` | `0.6804` | `0.6681` |

Delta vs LIF baseline:

| Setting | Mean delta | Std delta | Worst delta | Best delta |
| --- | ---: | ---: | ---: | ---: |
| ALIF spike_count | `-0.0036` | `+0.0032` | `-0.0141` | `-0.0022` |
| ALIF spike_adaptation_concat, beta 0.05 inc 0.10 | `+0.0011` | `+0.0158` | `-0.0340` | `+0.0102` |
| ALIF spike_adaptation_concat, beta 0.10 inc 0.125 bias LR 1.0 | `+0.0169` | `-0.0042` | `+0.0177` | `+0.0080` |
| ALIF spike_adaptation_concat, beta 0.10 inc 0.125 bias LR 0.5 | `+0.0213` | `-0.0061` | `+0.0238` | `+0.0102` |
| Non-spiking LIF readout v0.1 beta 0.99 | `-0.0283` | `-0.0016` | `-0.0331` | `-0.0362` |

## Seed-Level Results

### Validation-Selected Test Accuracy

| Seed | LIF spike_count | ALIF spike_count | ALIF spike_adaptation_concat | Spike-adapt delta vs LIF |
| --- | ---: | ---: | ---: | ---: |
| 42 | `0.6966` | `0.6807` | `0.7306` | `+0.0340` |
| 43 | `0.6948` | `0.6943` | `0.7169` | `+0.0221` |
| 44 | `0.7204` | `0.7182` | `0.7067` | `-0.0137` |
| 45 | `0.6988` | `0.7027` | `0.6608` | `-0.0380` |

### ALIF Spike Adaptation Concat Details, beta 0.05 inc 0.10

| Seed | Best val epoch | Best val | Test @ best val | Best test epoch | Best test | Final test | Topology frozen epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `84` | `0.7966` | `0.7306` | `75` | `0.7385` | `0.7292` | `88` |
| 43 | `72` | `0.8039` | `0.7169` | `97` | `0.7266` | `0.7257` | `82` |
| 44 | `95` | `0.8174` | `0.7067` | `79` | `0.7279` | `0.7058` | none |
| 45 | `94` | `0.7941` | `0.6608` | `59` | `0.6714` | `0.6612` | `50` |

## FDI Calibration

### ALIF Spike Adaptation Concat, beta 0.05 inc 0.10

| Seed | Input scale | Recurrent scale | Threshold scale | Probe mean rate Hz | Silent frac | Overactive frac | Xi mean | Adapt/threshold ratio | Rec/input std ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `1.25` | `1.0` | `0.75` | `9.3023` | `0.156` | `0.000` | `0.9989` | `0.0036` | `0.0486` |
| 43 | `1.00` | `1.0` | `0.75` | `8.9445` | `0.214` | `0.000` | `1.0210` | `0.0035` | `0.0744` |
| 44 | `1.25` | `1.0` | `0.75` | `9.2054` | `0.182` | `0.000` | `1.0038` | `0.0036` | `0.0554` |
| 45 | `1.25` | `1.0` | `0.75` | `9.4559` | `0.170` | `0.000` | `0.9956` | `0.0036` | `0.0593` |

The adaptation/threshold ratio at FDI time is tiny. With `rho=0.85`, `adapt_increment=0.10`, and `beta=0.05`, the maximum adaptation state is approximately `0.10 / (1 - 0.85) = 0.6667`, but the maximum threshold lift is only `0.05 * 0.6667 = 0.0333`.

## Activity and Stability

### Final Activity Stats

These activity stats are for the beta `0.05`, increment `0.10` `spike_adaptation_concat` runs.

| Seed | Final mean firing | Final max firing | Final mean adaptation | Final max adaptation |
| --- | ---: | ---: | ---: | ---: |
| 42 | `0.2238` | `0.8287` | `0.1571` | `0.6667` |
| 43 | `0.1992` | `0.8192` | `0.1415` | `0.6667` |
| 44 | `0.2247` | `0.9454` | `0.1425` | `0.6667` |
| 45 | `0.0761` | `0.5408` | `0.0008` | `0.2140` |

### First `max_firing_rate > 0.9`

| Seed | First epoch | Val | Test | Max firing | Mean firing | Max adaptation | Topology grad pre-clip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `28` | `0.6225` | `0.5981` | `0.9217` | `0.2002` | `0.6667` | `1203.3` |
| 43 | none | - | - | - | - | - | - |
| 44 | `35` | `0.6250` | `0.5985` | `0.9038` | `0.2494` | `0.6667` | `24.1` |
| 45 | `40` | `0.6042` | `0.5623` | `0.9004` | `0.2163` | `0.6667` | `8.7` |

Interpretation:

- The beta `0.05`, increment `0.10` `spike_adaptation_concat` run improves the mean selected test slightly and raises the best seed, but variance increases sharply.
- Seed 45 is the failure case: validation remains high late, but selected test collapses to `0.6608`.
- Seed 42 and seed 44 show strong peaks, but repeated high-firing warnings indicate residual recurrent-loop instability.
- `max_adaptation` often saturates at `0.6667`, so the ALIF state is active. The problem is that current `alif_beta_init=0.05` makes the threshold correction too small to suppress overactive neurons.

## Beta 0.10 Increment 0.125 Reruns

The follow-up reruns increase the ALIF threshold feedback from the original beta `0.05`, increment `0.10` setting to beta `0.10`, increment `0.125`. With `rho=0.85`, the steady-state maximum adaptation state changes from approximately `0.10 / (1 - 0.85) = 0.6667` to `0.125 / (1 - 0.85) = 0.8333`, and the maximum threshold lift changes from `0.0333` to `0.0833`.

Two config-matched 4-seed groups were run. They differ only in `liquid.theta_bias_lr_scale`.

### Aggregate Comparison

| Setting | Mean best val | Mean test @ best val | Std | Worst | Best | Mean best test | Mean final test | Mean final density |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beta 0.10 inc 0.125, bias LR 1.0 | `0.8186` | `0.7195` | `0.0062` | `0.7125` | `0.7284` | `0.7289` | `0.7216` | `0.0514` |
| beta 0.10 inc 0.125, bias LR 0.5 | `0.8186` | `0.7239` | `0.0043` | `0.7186` | `0.7306` | `0.7308` | `0.7243` | `0.0555` |

Delta from bias LR `1.0` to bias LR `0.5`:

| Metric | Delta |
| --- | ---: |
| Mean best val | `+0.0000` |
| Mean test @ best val | `+0.0044` |
| Mean best test | `+0.0019` |
| Mean final val | `+0.0025` |
| Mean final test | `+0.0027` |
| Mean final density | `+0.0041` |
| Mean final firing | `+0.0050` |
| Mean final adaptation | `+0.0066` |

### Bias LR 1.0 Details

| Seed | Best val epoch | Best val | Test @ best val | Best test | Final val | Final test | Frozen epoch | Final density | Final mean firing | Final mean adaptation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `72` | `0.7978` | `0.7151` | `0.7292` @ `84` | `0.7929` | `0.7248` | `61` | `0.0430` | `0.2098` | `0.1858` |
| 43 | `98` | `0.8321` | `0.7222` | `0.7284` @ `63` | `0.8272` | `0.7213` | `79` | `0.0517` | `0.2182` | `0.1913` |
| 44 | `78` | `0.8235` | `0.7125` | `0.7275` @ `80` | `0.8125` | `0.7138` | `88` | `0.0434` | `0.2158` | `0.1633` |
| 45 | `87` | `0.8211` | `0.7284` | `0.7306` @ `84` | `0.8137` | `0.7266` | `88` | `0.0676` | `0.1732` | `0.1330` |

### Bias LR 0.5 Details

| Seed | Best val epoch | Best val | Test @ best val | Best test | Final val | Final test | Frozen epoch | Final density | Final mean firing | Final mean adaptation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `94` | `0.8076` | `0.7306` | `0.7398` @ `70` | `0.8051` | `0.7314` | `76` | `0.0460` | `0.2203` | `0.1933` |
| 43 | `94` | `0.8284` | `0.7186` | `0.7306` @ `69` | `0.8235` | `0.7191` | `87` | `0.0539` | `0.2134` | `0.1846` |
| 44 | `93` | `0.8223` | `0.7244` | `0.7266` @ `87` | `0.8137` | `0.7235` | `81` | `0.0575` | `0.2190` | `0.1815` |
| 45 | `99` | `0.8162` | `0.7222` | `0.7261` @ `77` | `0.8137` | `0.7231` | `78` | `0.0644` | `0.1844` | `0.1405` |

### FDI Calibration for Beta 0.10 Increment 0.125

The FDI-selected scale triplet is identical across the two rerun groups for each seed because the initial ALIF and FDI settings are the same. All seeds select `input_scale=1.25`, `recurrent_scale=1.0`, `threshold_scale=0.75`.

| Seed | Probe mean rate Hz | Silent frac | Xi mean | Adapt/threshold ratio | Rec/input std ratio | Warning |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 42 | `9.2162` | `0.156` | `1.0010` | `0.0089` | `0.0483` | Rec/input below target |
| 43 | `10.0795` | `0.164` | `0.9643` | `0.0097` | `0.0634` | Xi below target; rec/input below target |
| 44 | `9.1184` | `0.184` | `1.0060` | `0.0088` | `0.0550` | Rec/input below target |
| 45 | `9.3662` | `0.170` | `0.9979` | `0.0090` | `0.0590` | Xi slightly below target; rec/input below target |

Interpretation:

- Raising beta/increment converts the earlier `spike_adaptation_concat` result from an unstable peak into a stable improvement over the LIF learned-input FDI baseline.
- Bias LR `0.5` is slightly better than bias LR `1.0`: mean `test @ best val` improves from `0.7195` to `0.7239`, while std drops from `0.0062` to `0.0043`.
- The strongest validation seed is not the strongest test seed. In the bias LR `0.5` group, seed 43 has the best validation accuracy (`0.8284`), but seed 42 has the best selected test (`0.7306`) and best oracle test (`0.7398`).
- FDI still reports very weak recurrent/input current ratios, around `0.048-0.063` against the configured target range `[0.3, 1.5]`. The new result is strong despite this, but recurrent-drive calibration remains an unresolved diagnostic.

## Interpretation

ALIF spike-count is competitive but not better than the LIF baseline:

- Mean is lower than LIF: `0.6990` vs `0.7026`.
- Worst seed is lower: `0.6807` vs `0.6948`.
- Variance is higher: `0.0136` vs `0.0104`.

The original ALIF `spike_adaptation_concat` beta `0.05`, increment `0.10` run is more promising than count-only readout but unstable:

- Mean is slightly higher than LIF: `0.7037` vs `0.7026`.
- Best seed is higher: `0.7306` vs `0.7204`.
- Worst seed is much lower: `0.6608` vs `0.6948`.
- Std is much higher: `0.0262` vs `0.0104`.

The beta `0.10`, increment `0.125` reruns change the conclusion. Giving the readout access to adaptation trace still matters, but the stronger adaptation setting removes the seed-45 collapse seen in the beta `0.05` run. The best current ALIF recipe is `spike_adaptation_concat` with beta `0.10`, increment `0.125`, and `theta_bias_lr_scale=0.5`:

- Mean selected test is higher than LIF: `0.7239` vs `0.7026`.
- Worst seed is higher than LIF: `0.7186` vs `0.6948`.
- Std is lower than LIF: `0.0043` vs `0.0104`.
- Mean final test is also higher: `0.7243` vs `0.7041`.

The non-spiking LIF final-membrane readout remains rejected:

- Mean is materially below baseline: `0.6744` vs `0.7026`.
- Worst seed is also below baseline: `0.6617` vs `0.6948`.

## Decision

Current decisions:

- Keep `LIF + learned input + FDI + spike_count` as the conservative reference baseline.
- Reject `non_spiking_lif_final_mem` v0.1 beta `0.99`.
- Do not accept `ALIF + spike_count` as an improvement.
- Do not use the old beta `0.05`, increment `0.10` `ALIF + spike_adaptation_concat` run as the final ALIF judgment; it was an unstable peak.
- Treat `ALIF + spike_adaptation_concat`, beta `0.10`, increment `0.125`, `theta_bias_lr_scale=0.5` as the current strongest 4-seed SHD LSM candidate in this document.

## Next Follow-Up

The original next-step recommendation was to increase ALIF feedback strength. The beta `0.10`, increment `0.125` reruns partially answer that recommendation and should replace the beta `0.05` run as the ALIF starting point.

Useful next probes:

```bash
python scripts/train_lsm.py configs/lsm_shd_lowrank_r16_m50p10_learned_input_proj_fdi.yaml \
  seed=42 \
  liquid.neuron_type=alif \
  liquid.alif_rho_init=0.85 \
  liquid.alif_beta_init=0.10 \
  liquid.alif_adapt_increment=0.125 \
  liquid.alif_learn_rho=false \
  liquid.alif_learn_beta=false \
  liquid.readout_mode=spike_adaptation_concat \
  liquid.theta_bias_lr_scale=0.5 \
  experiment_name=lsm_shd_alif_lowrank_r16_m50p10_learned_input_proj_fdi_spike_adaptation_b010_inc0125_biaslr05_42
```

Recommended ablations from this point:

- Test whether `theta_bias_lr_scale=0.25` or `0.75` improves the already-stable bias LR `0.5` result.
- Test whether the FDI recurrent/input warning can be fixed by expanding recurrent scale candidates above `1.0`.
- Keep `test @ best val` as the primary metric; do not select by oracle best test.

Success criteria against the active LIF baseline:

- Strong success: mean up, std same/down, worst same/up.
- Stabilization success: mean similar, worst up.
- Unstable peak: mean up but worst down.
- Reject: mean down and worst down.

Under these criteria, the old beta `0.05`, increment `0.10` `spike_adaptation_concat` result was an unstable peak. The beta `0.10`, increment `0.125`, bias LR `0.5` result meets the strong-success pattern on this 4-seed batch: mean up, std down, and worst seed up.

## Caveats

- All statistics are based on seeds `42-45` only.
- `spike_adaptation_concat` runs completed on 2026-06-16, while the original summary date is 2026-06-15.
- The old ALIF hyperparameter point was `rho=0.85`, `beta=0.05`, `adapt_increment=0.10`.
- The latest stronger ALIF point is `rho=0.85`, `beta=0.10`, `adapt_increment=0.125`.
