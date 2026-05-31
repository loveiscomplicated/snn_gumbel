# ALIF Implementation and Experiment Notes

> Updated: 2026-05-31

이 문서는 SHD LSM 코드베이스에 ALIF 뉴런을 이식한 과정, 현재까지의 실험 결과, 성능 저하 원인 해석, readout ablation 결과, 그리고 다음 실험 결정을 정리한다.

핵심 결론은 다음과 같다.

- ALIF 구현 자체는 동작한다.
- 하지만 기존 `spike_count` readout과 결합하면 `learned_lowrank`에서 성능 이점이 거의 없거나 오히려 떨어진다.
- `random_sparse`에서는 ALIF가 LIF 대비 거의 중립에 가까워, ALIF 뉴런 자체가 완전히 나쁜 것은 아니다.
- 문제의 핵심은 `ALIF + learned_lowrank topology + count-only readout`의 상호작용으로 보인다.
- `membrane_trace`와 `spike_adaptation_concat` readout은 ALIF 성능을 상당히 복구한다.
- 현재 최우선 후속 실험은 `spike_adaptation_concat` readout의 seed 43-45 반복이다.

## 1. Background

Phase A에서 가장 안정적인 LIF 기준선은 다음 조합이었다.

```text
dataset = SHD
liquid = LIF
recurrent_mode = learned_lowrank
topology selection = validation rollback m50p10
selection metric = test@best-val
seed = 42, 43, 44, 45
```

이 기준선은 density-matched `random_sparse` control보다 명확히 강했다.

| Condition | test@best-val mean | Interpretation |
|---|---:|---|
| `learned_lowrank + LIF + m50p10` | `0.5919` | Phase A main baseline |
| density-matched `random_sparse` controls | `~0.5257` | density-only explanation rejected |
| no-recurrence baseline | `~0.5490` | random sparse recurrence보다 강함 |

따라서 Phase B의 질문은 다음이었다.

> LIF에서 좋았던 `learned_lowrank` liquid를 ALIF로 바꾸면 더 나은 temporal/adaptive representation을 얻을 수 있는가?

현재까지의 답은 보수적으로 다음과 같다.

> count-only readout을 그대로 쓰면 아니다. 다만 ALIF 상태를 더 직접적으로 읽는 readout을 쓰면 가능성이 다시 열린다.

## 2. ALIF Implementation

ALIF는 기존 LIF liquid path를 깨지 않는 opt-in 방식으로 추가되었다.

### Config Surface

`liquid` config에 다음 필드가 추가되었다.

```yaml
liquid:
  neuron_type: lif   # lif | alif
  alif_rho_init: 0.9
  alif_beta_init: 0.4
  alif_adapt_increment: 1.0
  alif_learn_rho: false
  alif_learn_beta: false
```

기본값은 기존 LIF 동작을 유지한다.

### ALIF Dynamics

ALIF는 기존 membrane state에 adaptation state `a`를 추가한다.

```text
a[t] = rho * a[t-1] + adapt_increment * z[t-1]
theta_eff[t] = threshold_base + beta_adapt * a[t]
v[t] = beta_mem * v[t-1] + input_current + recurrent_current
z[t] = spike_fn(v[t] - theta_eff[t])
v[t] = v[t] * (1 - z[t])
```

구현상 중요한 제약은 다음과 같다.

- `neuron_type: lif`는 기존 path를 유지한다.
- `neuron_type: alif`에서만 `liquid_a` adaptation state를 사용한다.
- effective threshold는 너무 낮아지지 않도록 clamp한다.
- membrane clamp는 기존 안정화 정책을 유지한다.
- truncated BPTT를 사용할 때 `liquid_mem`, `liquid_spike`, `liquid_a`를 함께 detach한다.

### Logging

epoch별 JSONL에는 다음 diagnostic이 기록된다.

- `neuron_type`
- `readout_mode`
- `mean_firing_rate`
- `max_firing_rate`
- `mean_adaptation`
- `max_adaptation`
- `hard_density`

ALIF 실험 분석에서는 accuracy만 보지 않고, 최소한 다음을 같이 본다.

- topology hard density가 LIF 기준선과 비슷한가
- mean firing rate가 지나치게 낮아지지 않았는가
- adaptation이 threshold를 과하게 밀어 올리지 않는가
- validation-best 선택 시 test generalization이 같이 따라오는가

## 3. Readout Implementation

초기 구조의 readout은 사실상 count-only readout이다.

```text
logits = mean_t Linear(spike_t)
       = Linear(mean_t spike_t)
```

이 방식은 시간 순서, subthreshold membrane signal, ALIF adaptation state를 거의 버린다.

이를 검증하기 위해 `liquid.readout_mode`가 추가되었다.

```yaml
liquid:
  readout_mode: spike_count
```

현재 지원하는 readout은 다음과 같다.

| Mode | Readout input | Purpose |
|---|---|---|
| `spike_count` | mean spike over time | 기존 동작 유지 |
| `membrane_trace` | mean membrane over time | spike로 드러나지 않은 subthreshold signal 읽기 |
| `spike_adaptation_concat` | concat(mean spike, mean adaptation) | ALIF 고유 상태를 readout에 직접 제공 |

### spike_count

기존 방식이다.

```text
readout_input = mean_t z[t]
```

장점은 단순하고 기존 LIF baseline과 직접 비교가 쉽다는 점이다.

단점은 ALIF가 burst와 반복 발화를 억제할 때 readout에 들어가는 signal 자체가 약해진다는 점이다.

### membrane_trace

spike 대신 membrane potential trace를 평균해서 읽는다.

```text
readout_input = mean_t v[t]
```

ALIF에서는 threshold가 올라가 spike가 줄어도 membrane에는 class-relevant subthreshold signal이 남아 있을 수 있다. 이 readout은 그 정보를 복구하려는 목적이다.

### spike_adaptation_concat

spike 평균과 adaptation 평균을 붙여서 읽는다.

```text
readout_input = concat(mean_t z[t], mean_t a[t])
```

이 방식은 ALIF를 단순 LIF 대체재가 아니라 추가 상태 변수를 가진 뉴런으로 취급한다.

해석상 adaptation `a`는 다음 정보를 담는다.

- 최근 발화 이력
- burst 억제 정도
- effective threshold 상승 정도
- 뉴런별 temporal fatigue / eligibility-like trace

현재 결과상 가장 중요한 후속 후보이다.

## 4. Configs

### Main ALIF Lowrank Configs

| Config | Purpose |
|---|---|
| `configs/lsm_shd_alif_learned_lowrank_m50p10.yaml` | 초기 ALIF learned_lowrank 이식 |
| `configs/lsm_shd_alif_lowrank_lifmatched_a010_b005.yaml` | LIF baseline에 맞춘 낮은 adaptation setting |
| `configs/lsm_shd_alif_lowrank_density_p05_a010_b005.yaml` | lowrank density를 더 올려보는 ablation |
| `configs/lsm_shd_alif_lowrank_density_p05_a025_b010.yaml` | density 증가 + adaptation 중간값 |

### Random Sparse Controls

| Config | Purpose |
|---|---|
| `configs/lsm_shd_random_sparse_p045_lif_control.yaml` | fixed random sparse LIF control |
| `configs/lsm_shd_alif_random_sparse_p045_a010_b005.yaml` | fixed random sparse ALIF control |

### Readout Ablation Configs

| Config | Purpose |
|---|---|
| `configs/lsm_shd_lif_lowrank_readout_spike_count.yaml` | LIF lowrank readout baseline |
| `configs/lsm_shd_alif_lowrank_readout_spike_count.yaml` | ALIF lowrank count-only baseline |
| `configs/lsm_shd_alif_lowrank_readout_membrane_trace.yaml` | ALIF lowrank membrane readout |
| `configs/lsm_shd_alif_lowrank_readout_spike_adaptation_concat.yaml` | ALIF lowrank spike + adaptation readout |
| `configs/lsm_shd_alif_random_sparse_p045_readout_membrane_trace.yaml` | random sparse에서 membrane readout 효과 확인 |

## 5. Experimental Results

모든 주 비교는 `test@best-val` 기준이다. `best test`는 참고용으로만 본다.

### 5.1 Initial ALIF Learned-Lowrank

초기 ALIF learned-lowrank 실험은 성능이 좋지 않았다.

주요 현상:

- hard density가 LIF lowrank 대비 너무 낮았다.
- mean firing rate가 LIF lowrank 대비 크게 낮았다.
- adaptation이 effective threshold를 올려 반복 발화를 억제했다.

초기 해석:

```text
LIF lowrank = high recurrent activity regime에서 강함
ALIF = burst / repeated firing 억제
count readout = spike count signal에 의존
결과 = readout이 먹던 신호가 약해짐
```

따라서 ALIF 자체가 나쁘다기보다, 기존 topology/readout과의 조합이 맞지 않을 가능성이 커졌다.

### 5.2 LIF-Matched ALIF Lowrank, Seeds 42-45

`configs/lsm_shd_alif_lowrank_lifmatched_a010_b005.yaml`을 seed 42-45로 반복했다.

| Condition | test@best-val mean | std | best | worst |
|---|---:|---:|---:|---:|
| LIF lowrank baseline | `0.5919` | `0.0145` | `0.6135` | `0.5826` |
| ALIF lowrank lifmatched | `0.5703` | `0.0406` | `0.6184` | `0.5292` |

paired difference, ALIF minus LIF:

| Seed | Difference |
|---:|---:|
| 42 | `-0.0565` |
| 43 | `+0.0049` |
| 44 | `+0.0027` |
| 45 | `-0.0371` |

해석:

- 일부 seed에서는 LIF에 근접하거나 넘지만 안정적이지 않다.
- 평균은 LIF보다 낮고 variance가 크다.
- 특히 seed 42 collapse가 결정적이다.

### 5.3 Random Sparse ALIF Control, Seeds 42-45

`random_sparse p=0.045`에서는 ALIF와 LIF 차이가 거의 없었다.

| Condition | test@best-val mean | std |
|---|---:|---:|
| LIF random sparse p045 | `~0.5255` | `~0.0023` |
| ALIF random sparse p045 | `0.5243` | `0.0066` |

paired difference, ALIF minus LIF:

| Seed | Difference |
|---:|---:|
| 42 | `-0.0071` |
| 43 | `+0.0013` |
| 44 | `+0.0053` |
| 45 | `-0.0044` |

해석:

- fixed random sparse에서는 ALIF 뉴런 자체가 성능을 크게 망치지 않는다.
- 따라서 문제는 `ALIF 자체`라기보다 `ALIF + learned_lowrank topology learning + current readout` 조합일 가능성이 크다.

### 5.4 Readout Ablation, Seed 42

readout mismatch 가설을 보기 위해 seed 42에서 5개 실험을 먼저 돌렸다.

| Condition | test@best-val | best test | Notes |
|---|---:|---:|---|
| LIF lowrank + spike count | `0.5857` | `0.5875` | baseline |
| ALIF lowrank + spike count | `0.5292` | `0.5309` | count-only failure |
| ALIF lowrank + membrane trace | `0.6003` | `0.6135` | seed 42에서는 LIF 초과 |
| ALIF lowrank + spike/adaptation concat | `0.6047` | `0.6109` | seed 42 최고 |
| ALIF random sparse + membrane trace | `0.5724` | `0.5795` | random sparse에서도 readout 효과 큼 |

activity diagnostics:

| Condition | hard density | mean firing rate | max adaptation |
|---|---:|---:|---:|
| LIF lowrank + spike count | `0.0415` | `0.2000` | `0.0000` |
| ALIF lowrank + spike count | `0.0039` | `0.0512` | `0.1061` |
| ALIF lowrank + membrane trace | `0.0173` | `0.0895` | `0.4981` |
| ALIF lowrank + spike/adaptation concat | `0.0459` | `0.2029` | `0.6667` |
| ALIF random sparse + membrane trace | `0.0451` | `0.0693` | `0.0768` |

해석:

- `ALIF + spike_count`는 seed 42에서 topology/activity collapse가 뚜렷하다.
- `membrane_trace`는 낮은 firing rate에서도 test 성능을 복구한다.
- `spike_adaptation_concat`은 LIF와 유사한 density/firing-rate regime까지 회복한다.
- readout이 단순 classifier만 바꾼 것이 아니라, topology learning gradient까지 바꾸고 있다.

### 5.5 ALIF Lowrank + Membrane Trace, Seeds 42-45

`configs/lsm_shd_alif_lowrank_readout_membrane_trace.yaml`을 seed 42-45로 반복했다.

| Seed | val@best-val | test@best-val | best test | density | mean firing rate |
|---:|---:|---:|---:|---:|---:|
| 42 | `0.7647` | `0.6003` | `0.6135` | `0.0173` | `0.0895` |
| 43 | `0.7843` | `0.5932` | `0.6321` | `0.0195` | `0.0965` |
| 44 | `0.7328` | `0.5897` | `0.6051` | `0.0175` | `0.0958` |
| 45 | `0.7390` | `0.5685` | `0.5901` | `0.0191` | `0.1008` |

Aggregate:

| Condition | test@best-val mean | std | best test mean | val mean |
|---|---:|---:|---:|---:|
| ALIF lowrank + membrane trace | `0.5879` | `0.0137` | `0.6102` | `0.7552` |
| ALIF lowrank + spike count | `0.5703` | `0.0406` | `0.5844` | `0.6210` |
| LIF lowrank baseline | `0.5919` | `0.0145` | `0.5991` | `0.6330` |

Paired comparison:

| Comparison | Mean diff |
|---|---:|
| membrane trace minus ALIF spike count | `+0.0176` |
| membrane trace minus LIF lowrank | `-0.0040` |

해석:

- `membrane_trace`는 ALIF count-only 대비 평균 성능을 올리고 variance를 줄인다.
- LIF lowrank baseline에는 거의 근접하지만, 안정적으로 넘지는 못한다.
- validation accuracy가 지나치게 높게 튀므로 over-selection 가능성을 주의해야 한다.

Validation-test gap:

| Seed | val@best-val | test@best-val | gap |
|---:|---:|---:|---:|
| 42 | `0.7647` | `0.6003` | `-0.1644` |
| 43 | `0.7843` | `0.5932` | `-0.1911` |
| 44 | `0.7328` | `0.5897` | `-0.1432` |
| 45 | `0.7390` | `0.5685` | `-0.1705` |

이 gap은 LIF lowrank baseline보다 크다. 따라서 `membrane_trace`는 성능 복구 후보이지만, 최종 선택 후보로 바로 고정하기에는 generalization risk가 있다.

## 6. Interpretation

현재 결과는 다음 가설을 지지한다.

### 6.1 ALIF 뉴런 자체 문제는 아니다

`random_sparse p045`에서 ALIF와 LIF의 차이는 거의 없다. 따라서 ALIF dynamics가 SHD에서 일반적으로 해로운 것은 아니다.

### 6.2 Count-only readout은 ALIF와 잘 맞지 않는다

ALIF는 반복 발화와 burst를 억제한다. 그런데 기존 readout은 평균 spike count에 강하게 의존한다.

결과적으로 ALIF가 만든 정보 중 다음이 버려진다.

- 발화 타이밍
- subthreshold membrane trajectory
- adaptation state
- effective threshold 변화

`membrane_trace`와 `spike_adaptation_concat`이 성능을 복구했다는 점은 이 해석과 잘 맞는다.

### 6.3 Topology learning과 readout은 독립이 아니다

readout을 바꾸면 단지 마지막 classifier만 바뀌는 것이 아니다.

`learned_lowrank`에서는 readout loss gradient가 topology parameter에도 영향을 준다. 따라서 readout mode는 liquid가 어떤 activity regime으로 학습되는지도 바꾼다.

가장 명확한 예:

| Condition | density | mean firing rate |
|---|---:|---:|
| ALIF lowrank + spike count, seed 42 | `0.0039` | `0.0512` |
| ALIF lowrank + spike/adaptation concat, seed 42 | `0.0459` | `0.2029` |

이는 `spike_adaptation_concat`이 ALIF 상태를 더 잘 읽을 뿐 아니라, learned topology가 LIF baseline과 비슷한 activity regime을 만들도록 돕고 있음을 시사한다.

### 6.4 Membrane trace는 강하지만 validation over-selection이 있다

`membrane_trace`는 test를 복구하지만 validation이 과도하게 높다. 이 현상은 다음 가능성을 의미한다.

- validation split에 membrane trace feature가 과적합됨
- readout capacity가 커진 효과가 test에는 제한적으로 전달됨
- topology freeze/rollback metric이 membrane readout에서는 너무 민감함
- SHD validation set이 temporal trace readout에 대해 test보다 쉬운 분포일 수 있음

따라서 `membrane_trace`는 메커니즘 증거로는 중요하지만, 최종 성능 후보로는 추가 검증이 필요하다.

## 7. Current Decision

현재 ALIF를 접을 단계는 아니다.

다만 결론은 분명하다.

```text
ALIF + spike_count readout = 현재 구조에서는 부적합
ALIF + membrane/adaptation-aware readout = 검증할 가치 있음
```

후속 우선순위:

| Priority | Experiment | Reason |
|---:|---|---|
| 1 | `ALIF lowrank + spike_adaptation_concat` seeds 43-45 | seed 42 최고, val-test gap이 membrane trace보다 작음 |
| 2 | `ALIF random sparse + membrane_trace` seeds 43-45 | readout effect가 topology-independent인지 확인 |
| 3 | `ALIF lowrank + membrane_trace` 추가 diagnostics | validation over-selection 원인 확인 |
| 4 | motor neuron readout | readout ablation 이후 진행 |

## 8. Next Commands

가장 먼저 반복할 실험:

```bash
uv run python scripts/train_lsm.py configs/lsm_shd_alif_lowrank_readout_spike_adaptation_concat.yaml seed=43
uv run python scripts/train_lsm.py configs/lsm_shd_alif_lowrank_readout_spike_adaptation_concat.yaml seed=44
uv run python scripts/train_lsm.py configs/lsm_shd_alif_lowrank_readout_spike_adaptation_concat.yaml seed=45
```

리소스가 허용되면 보조로 반복할 실험:

```bash
uv run python scripts/train_lsm.py configs/lsm_shd_alif_random_sparse_p045_readout_membrane_trace.yaml seed=43
uv run python scripts/train_lsm.py configs/lsm_shd_alif_random_sparse_p045_readout_membrane_trace.yaml seed=44
uv run python scripts/train_lsm.py configs/lsm_shd_alif_random_sparse_p045_readout_membrane_trace.yaml seed=45
```

## 9. Evaluation Rules Going Forward

ALIF 관련 실험은 다음 원칙으로 평가한다.

1. main metric은 `test@best-val`이다.
2. `best test`는 참고용이며 선택 기준으로 쓰지 않는다.
3. seed 42 단일 결과로 결론 내리지 않는다.
4. accuracy와 함께 density, firing rate, adaptation을 반드시 본다.
5. validation-test gap이 큰 readout은 성능 후보가 아니라 diagnostic 후보로 먼저 취급한다.
6. random sparse control에서 개선되는지 확인해 neuron/readout 효과와 topology-learning 효과를 분리한다.
7. motor neuron readout은 readout mismatch 검증 이후에 추가한다.

## 10. Open Questions

아직 닫히지 않은 질문은 다음이다.

- `spike_adaptation_concat`이 seed 43-45에서도 LIF lowrank를 넘거나 근접하는가?
- `spike_adaptation_concat`의 좋은 seed 42 결과는 adaptation feature 때문인가, activity regime 회복 때문인가?
- `membrane_trace`의 큰 validation-test gap은 split artifact인가, readout overfit인가?
- topology freeze metric을 `val_acc` 그대로 쓰는 것이 membrane/adaptation readout에서도 적절한가?
- ALIF에 맞는 topology selection 기준은 LIF와 달라야 하는가?
- motor neuron readout이 count-only mismatch를 더 자연스럽게 해결하는가?

현재까지의 가장 방어적인 결론:

> ALIF는 기존 count-only LSM readout에서는 이점이 보이지 않는다. 그러나 ALIF의 membrane/adaptation state를 읽도록 readout을 바꾸면 LIF lowrank 기준선에 근접하거나 일부 seed에서 넘는다. 따라서 문제는 ALIF 자체보다 ALIF가 만든 sparse/adaptive temporal code를 기존 readout과 topology-selection protocol이 제대로 사용하지 못하는 데 있을 가능성이 크다.
