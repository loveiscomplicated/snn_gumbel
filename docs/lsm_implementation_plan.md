# LSM 구현 상태 및 남은 작업

## 0. 개요

기존 feedforward Gumbel-SNN 위에 LSM(Liquid State Machine) 전용 코드가 추가되어 있다. 초기 계획서의 핵심 항목인 config 확장, SHD 로더, LSM 모델, 학습 CLI, gradient clipping, 모니터링은 현재 코드베이스에 구현되어 있다.

현재 목표는 SHD에서 다음 비교를 안정적으로 수행하는 것이다.

| 조건 | 코드 mode | 의미 |
|------|-----------|------|
| A | `fixed` | 전통 LSM: 구조 고정, recurrent weight도 고정 |
| B | `random_sparse` | 구조 고정, recurrent weight/readout 학습 |
| B* | `random_sparse` + C와 동일 density | 연결 수 통제 baseline |
| D | `grad_r` | hard threshold 구조 학습 baseline |
| C | `learned` | Gumbel/STE 기반 구조 + weight 동시 학습 |

---

## 1. 구현 완료 항목

### 1.1 Config

구현 파일: `src/utils/config.py`

완료:

- `LiquidConfig` 추가
- `Config.liquid` 추가
- YAML inheritance 및 CLI override에서 `liquid.*` 파싱
- `tau_hold_epochs`, `lr_min`, `weight_decay` 등 LSM 학습에 필요한 필드 지원

현재 주요 LSM 필드:

```python
n_liquid: int = 200
exc_ratio: float = 0.8
p_input: float = 0.1
recurrent_mode: str = "learned"
recurrent_sparsity: float = 0.2
self_connection: bool = False
theta_init_mean: float = 0.0
theta_init_std: float = 0.01
grad_clip_max_norm_w: float = 100.0
grad_clip_max_norm_theta: float = 10.0
input_weight_scale: float = 0.1
w_raw_max: float = -3.0
w_raw_init_mean: float = -4.0
w_raw_init_std: float = 0.01
train_w_raw: bool = True
bptt_truncate: int = 0
beta_min: float = 0.7
beta_max: float = 0.95
threshold_min: float = 0.8
threshold_max: float = 1.5
theta_warmup_epochs: int = 0
theta_lr_scale: float = 0.1
noise_scale: float = 0.1
```

현재 baseline YAML: `configs/lsm_shd_baseline.yaml`

- `n_liquid=500`
- `theta_init_mean=-2.2`, `theta_init_std=0.5`
- `w_raw_init_mean=-4.0`, `w_raw_init_std=0.01`, `train_w_raw=true`, `w_raw_max=-3.0`
- `bptt_truncate=25`
- `theta_warmup_epochs=10`
- `tau_hold_epochs=15`, `tau_anneal_epochs=40`
- `lambda_sparse=0.1`, `lambda_commit=0.01`

### 1.2 데이터 로더

구현 파일: `src/data/loaders.py`

지원 dataset:

- `mnist`
- `fashion_mnist`
- `nmnist`
- `dvs_gesture`
- `shd`

SHD 구현:

- `tonic.datasets.SHD(save_to=root, train=train)` 사용
- event time `events["t"]`를 microsecond로 보고 `dt_us=10_000.0` binning
- 출력 shape: `(T, 700)`
- 같은 time/channel bin은 binary spike `1.0`으로 처리

주의:

- 초기 계획서의 직접 HDF5 fallback은 현재 구현되어 있지 않다.
- `tonic`이 설치되어 있어야 NMNIST, DVS Gesture, SHD를 사용할 수 있다.

### 1.3 LSM 모델

구현 파일: `src/lsm/model.py`

구성:

```text
InputProjection
  fixed sparse randn input projection, shape (n_input, n_liquid)

LiquidLayer
  recurrent topology, theta/w_raw/beta/threshold, Dale's Law, self_conn_mask

LSMModel
  input projection + liquid recurrent loop + linear readout
```

`InputProjection`:

- `mask = Bernoulli(p_input)`
- `weight = randn(n_input, n_liquid) * input_weight_scale * mask`
- `register_buffer("weight", weight)`
- 학습하지 않음

`LiquidLayer`:

- `theta`: learned mode에서만 학습
- `w_raw`: fixed mode가 아니면 학습
- `w_raw` 초기화: `Normal(w_raw_init_mean, w_raw_init_std)`
- `train_w_raw=false`: `w_raw`만 freeze, `beta/threshold/readout`은 계속 학습
- `logit_beta`: beta를 올바르게 sigmoid(logit)로 복원
- `threshold`: 뉴런별 이질적 초기화
- `dale_sign`: `(N, 1)`, presynaptic 행 기준
- `self_conn_mask`: 기본 diagonal 0
- `fixed_mask`: `random_sparse`, `fixed` 모드에서 사용

유효 recurrent weight:

```python
w_clamped = torch.clamp(self.w_raw, max=self.w_raw_max)
signed_w = self.dale_sign * F.softplus(w_clamped)
w_eff = self.current_mask * self.self_conn_mask * signed_w
```

`w_raw_max`는 상한 clamp일 뿐이다. 초기 `w_raw`가 상한보다 작으면 recurrent scale을 키우지 않는다. 초기 recurrent scale은 `w_raw_init_mean/std`로 조정한다.

Forward:

- `LSMModel.forward(spikes, tau=1.0)`만 받는다. `hard` 인자는 없다.
- forward 시작 시 `self.liquid.sample_mask(tau=tau)` 1회 호출
- `bptt_truncate > 0`이면 마지막 K step 전에 `liquid_mem`, `liquid_spike`를 detach
- membrane clamp `[-3.0, 3.0]`
- readout은 `nn.Linear(n_liquid, n_output)`를 타임스텝마다 누적 후 `T`로 나눈다.

### 1.4 LSM 학습 루프

구현 파일: `src/lsm/trainer.py`

완료:

- experiment directory 생성
- resolved config snapshot 저장
- SHD dataloader 로드
- device 선택: MPS > CUDA > CPU
- 2-phase 학습
- theta/other parameter optimizer group 분리
- CosineAnnealingLR
- independent gradient clipping
- NaN loss 감지
- JSONL logging
- best checkpoint 저장
- early stopping

2-phase 학습:

```text
Phase 1: epoch < theta_warmup_epochs
  theta freeze
  deterministic hard mask
  w_raw/readout/beta/threshold 안정화

Phase 2: epoch >= theta_warmup_epochs
  theta unfreeze
  epoch마다 Gumbel noise 1회 샘플링
  batch마다 같은 noise로 STE mask 재계산
  eval 전 epoch noise unlock -> deterministic mask 평가
```

Dynamic warmup은 선택 옵션이다.

```text
theta_warmup_dynamic=false  # 기본값: 기존 고정 warmup 재현

if learned and P1 and theta_warmup_dynamic:
  최소 theta_warmup_min_epochs 동안 P1 유지
  theta_warmup_strategy=slope:
    최근 theta_warmup_window개 score의 평균 epoch당 개선폭이
    theta_warmup_min_delta 미만인 상태가 theta_warmup_patience번 지속되면 P2 진입
  theta_warmup_strategy=best:
    best metric 미갱신 상태가 theta_warmup_patience번 지속되면 P2 진입
  theta_warmup_epochs는 최대 P1 길이로 동작
```

이 옵션은 learned topology C에서 P1이 과도하게 길거나 짧은 경우를 분리하기 위한 ablation이다. 기존 baseline과 random sparse 결과는 `theta_warmup_dynamic=false` 기준으로 비교한다.

현재 tau schedule:

```python
phase2_epoch = max(epoch - effective_theta_warmup_epochs, 0)
anneal_epoch = max(phase2_epoch - tau_hold_epochs, 0)
```

즉 warmup 동안 tau는 계산되지만 topology 학습에는 실질적으로 쓰이지 않는다. Phase 2 시작 후 `tau_hold_epochs` 동안 `tau_start`를 유지한 뒤 cosine annealing한다.

### 1.5 CLI

구현 파일:

- `scripts/train_lsm.py`
- `scripts/diagnose_liquid.py`
- `scripts/visualize.py`

사용:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=random_sparse
python scripts/diagnose_liquid.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=fixed
```

---

## 2. 초기 계획 대비 변경된 설계

| 초기 계획 | 현재 구현 |
|----------|-----------|
| 입력 projection은 흥분성만 | `randn` 양/음수 혼재 random projection |
| gradient clipping 단일 `max_norm=1.0` | weights/readout `100.0`, theta `10.0` 분리 |
| `theta_init_mean=0`, `std=0.01` | SHD baseline은 `mean=-2.2`, `std=0.5` |
| 매 batch Gumbel-Sigmoid | Phase 2에서 epoch-level noise 고정 + batch별 STE graph 재계산 |
| full BPTT 기본 | SHD baseline은 `bptt_truncate=25` |
| HDF5 fallback 포함 | 현재는 Tonic 기반 SHD loader만 구현 |
| LSM 평가에서 `hard=True` 고려 | `LSMModel.forward()`는 eval 모드로 deterministic mask 처리 |

변경 이유:

- recurrent BPTT에서 topology가 batch마다 크게 흔들리면 gradient가 폭주할 수 있어 epoch-level noise로 안정화했다.
- SHD에서 `N=500`과 긴 sequence를 다루기 위해 truncated BPTT와 큰 gradient norm 허용이 필요했다.
- 입력이 전부 양수 projection이면 클래스 간 liquid response가 지나치게 비슷해지는 문제가 있어 mixed-sign projection을 사용한다.

---

## 3. 검증해야 할 항목

### 3.1 실행 검증

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml epochs=1 liquid.n_liquid=200
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml epochs=1 liquid.recurrent_mode=random_sparse
python scripts/diagnose_liquid.py configs/lsm_shd_baseline.yaml liquid.n_liquid=200
```

확인:

- SHD loader가 `(batch, 100, 700)`을 반환하는가
- tonic dependency 및 데이터 다운로드가 정상 동작하는가
- 1 epoch wall-clock이 감당 가능한가
- NaN 없이 train/test loop가 완료되는가
- `theta_warmup_epochs` 경계에서 theta가 unfreeze되는가

### 3.2 모델 sanity check

- `dale_sign.shape == (N, 1)`
- `self_conn_mask.diag().sum() == 0` when `self_connection=false`
- `fixed` mode에서 `theta`, `w_raw`, `logit_beta`, `threshold`가 freeze되는가
- `random_sparse`와 `fixed` mode에서 `sparsity_loss`, `commitment_loss`가 0인가
- `learned` mode eval에서 deterministic mask가 사용되는가
- `current_mask`가 한 forward의 모든 timestep에서 동일한가

### 3.3 학습 안정성

로그에서 확인:

- `grad_norm`
- `theta_grad_norm`
- `w_raw_grad_norm`
- `mean_firing_rate`
- `max_firing_rate`
- `theta_mean`, `theta_std`
- `sparsity`

경고 기준:

- `grad_norm > 100`: lr 또는 clipping 검토
- `theta_grad_norm > 50`: tau/noise/lambda 조정 검토
- `w_raw_grad_norm > 50`: recurrent weight 폭주 가능성 확인
- `max_firing_rate > 0.9`: 흥분성 loop runaway 가능성
- epoch 20 이후 `theta_std < 0.01`: theta 정체 가능성

---

## 4. 현재 실험 위치와 순서

현재 상태:

| 단계 | 상태 | 메모 |
|------|------|------|
| 1. 현재 코드 상태 고정 | 완료 | LSM config/model/trainer/evaluate 변경 반영 |
| 2. LSM 평가 CLI 버그 수정 | 완료 | LSM forward는 `hard=True` 없이 eval mode deterministic mask 사용 |
| 3. 진단 출력 보강 | 완료 | spike stats, current scale, firing rate, class separation, sparsity, w_raw magnitude |
| 4. 순환 없음 baseline 재현 | 완료 | `lsm_shd_rs_p000_260428165253`, best test acc 54.90% |
| 5. random_sparse p-sweep 재현 | 완료 | `p=0.02,w=-2.25` 54.55%, `p=0.03,w=-2.5` 52.69%; 둘 다 reject |
| 6. 원인 분리 | 완료 | `w_raw` 학습 포화가 주요 실패 모드, density 증가 시 inhibitory suppression 악화 |
| 7. 최소 세팅 탐색 | 완료 | `p=0.02,w=-2.25,train_w_raw=false`가 best random recurrent지만 baseline 수준 |
| 8. learned topology C 재시도 | 다음 단계 | `train_w_raw=false`로 먼저 시도 |
| 9. analysis 도구 추가 | 미완료 | learned topology 구조 분석 |
| 10. predictive coding loss 검토 | 미완료 | LSM baseline 이후 검토 |

현재는 learned C 학습보다 원인 분리가 우선이다.

### Step 1: No-recurrence baseline 재현

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.0 \
  experiment_name=lsm_shd_rs_p000
```

목표:

- 현재 코드 상태에서 no-recurrence baseline을 다시 고정 — 완료
- 이후 recurrent 후보가 실제로 개선하는지 비교할 기준 확보 — **best test acc 54.90%**

재현 결과:

| 항목 | 값 |
|------|---:|
| experiment dir | `experiments/lsm_shd_rs_p000_260428165253` |
| epochs | 100 완료 |
| best epoch | 75 |
| best test acc | 0.5490 |
| final test acc | 0.5468 |
| best train acc | 0.6452 |
| final train acc | 0.6471 |
| sparsity | 0.0 |
| max grad_norm | 0.8908 |
| max firing rate | 0.3054 |
| final firing rate | 0.0439 / 0.2263 |

판단:

- no-recurrence baseline 재현 성공.
- 현재 코드 기준 baseline은 **54.90%**로 고정한다.
- recurrent 후보는 이 값을 넘어야 의미가 있다.

### Step 2: Baseline B 후보 학습

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.02 \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  experiment_name=lsm_shd_rs_p002_w225
```

목표:

- recurrent weight/readout 학습만으로 no-recurrence baseline을 이기는지 확인 — 실패
- epoch 1부터 firing rate와 grad norm이 안정적인지 확인 — 안정적
- `max_firing_rate > 0.9`이면 runaway로 보고 중단 — 해당 없음

결과:

| 항목 | 값 |
|------|---:|
| best test acc | 0.5455 |
| no-recurrence best test acc | 0.5490 |
| delta | -0.0035 |
| epoch 1 firing rate | 0.0746 / 0.3262 |
| best epoch firing rate | 0.0402 / 0.4142 |
| max firing rate | 0.4146 |
| max grad_norm | 0.8105 |

판단:

- runaway는 아니었지만 baseline을 이기지 못했으므로 reject.
- train accuracy도 baseline보다 낮아 보여 과적합보다 표현력/동역학 이득 부재로 해석한다.
- 단순 random recurrence가 유용한 class separation을 만들지 못하고 약간 방해한 결과에 가깝다.

Fallback 후보:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.03 \
  liquid.w_raw_init_mean=-2.5 \
  liquid.w_raw_max=-2.0 \
  experiment_name=lsm_shd_rs_p003_w250
```

Fallback 결과:

| 항목 | 값 |
|------|---:|
| experiment dir | `experiments/lsm_shd_rs_p003_w250_260428212005` |
| best test acc | 0.5269 |
| no-recurrence best test acc | 0.5490 |
| delta | -0.0221 |
| firing rate | 약 0.041 / 0.38x |
| grad_norm | 약 1.2~1.5 |
| sparsity | 0.030 |
| runaway | 없음 |

판단:

- fallback도 baseline을 크게 밑돌았으므로 reject.
- 두 random_sparse 후보가 모두 no-recurrence baseline을 넘지 못했다.
- 문제는 강도 부족이나 폭주가 아니라, random recurrent dynamics가 useful input response를 흐리거나 class separation을 개선하지 못하는 쪽에 가깝다.
- learned C로 바로 넘어가기 전에 trained checkpoint 진단으로 표현 변화 원인을 확인한다.

### Step 3: Trained checkpoint 진단

No-recurrence checkpoint:

```bash
python scripts/diagnose_liquid.py \
  --checkpoint experiments/lsm_shd_rs_p000_260428165253/checkpoints/best.pt \
  --batches 1 \
  --classes 5 \
  --samples-per-class 8 \
  experiments/lsm_shd_rs_p000_260428165253/config.yaml \
  batch_size=8
```

First random recurrent checkpoint:

```bash
python scripts/diagnose_liquid.py \
  --checkpoint experiments/lsm_shd_rs_p002_w225_260428201309/checkpoints/best.pt \
  --batches 1 \
  --classes 5 \
  --samples-per-class 8 \
  experiments/lsm_shd_rs_p002_w225_260428201309/config.yaml \
  batch_size=8
```

Fallback random recurrent checkpoint:

```bash
python scripts/diagnose_liquid.py \
  --checkpoint experiments/lsm_shd_rs_p003_w250_260428212005/checkpoints/best.pt \
  --batches 1 \
  --classes 5 \
  --samples-per-class 8 \
  experiments/lsm_shd_rs_p003_w250_260428212005/config.yaml \
  batch_size=8
```

결과:

| 항목 | `p=0.0` | `p=0.02,w=-2.25` | `p=0.03,w=-2.5` |
|------|--------:|-----------------:|----------------:|
| best test acc | 0.5490 | 0.5455 | 0.5269 |
| `\|rec\|/\|input\|` | 0.0000 | 0.0962 | 0.1317 |
| firing mean | 0.0511 | 0.0459 | 0.0460 |
| firing max | 0.4200 | 0.6700 | 0.6600 |
| active neurons `>0.01` | 327 | 269 | 240 |
| active neurons `>0.05` | 182 | 127 | 125 |
| cosine mean | 0.9676 | 0.9798 | 0.9790 |
| cosine min | 0.9234 | 0.9549 | 0.9545 |
| clamped fraction | 1.0000 | 1.0000 | 0.9999 |

판단:

- recurrent current는 생겼지만 정확도와 class separation은 악화됐다.
- active neuron 수가 줄어들어 liquid response가 일부 뉴런으로 수축된다.
- random_sparse 후보는 폭주하지 않았으므로 실패 원인은 runaway가 아니다.
- 학습 후 active recurrent weight가 거의 모두 `w_raw_max=-2.0` 상한에 붙어, near-uniform recurrent magnitude로 동작한다.
- learned C는 이 문제가 그대로 전이될 수 있으므로 `w_raw`를 자유롭게 학습시키는 설정으로 바로 진행하지 않는다.

### Step 4: 원인 분리

우선 확인할 가설:

1. `w_raw` 학습이 recurrent edge를 상한으로 포화시켜 near-uniform recurrent magnitude를 만든다.
2. Random Dale recurrent current가 입력 기반 liquid response를 공통 방향으로 끌어당긴다.
3. Recurrent current가 active neuron 수를 줄이면서 class별 mean-rate vector를 더 비슷하게 만든다.

우선순위 높은 ablation:

- `fixed` mode 또는 `random_sparse`에서 `w_raw` freeze로 topology/weight 학습 효과 분리.
- 더 약한 `w_raw_max=-3.0` 유지 후보로 recurrent current를 낮춘 상태 비교.
- E/I별 recurrent current, in/out degree, active neuron 분포를 분석하는 도구 추가.

구현된 원인 분리 지원:

- `liquid.train_w_raw` config 추가.
- `scripts/diagnose_liquid.py`가 E/I edge balance, incoming E/I degree, E/I recurrent current scale을 출력.
- `diagnose_liquid.py`의 argparse가 `parse_known_args()` 기반으로 바뀌어 named option과 `key=value` override를 섞어도 처리 가능.

원인 분리 실행 예:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.02 \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.train_w_raw=false \
  experiment_name=lsm_shd_rs_p002_w225_freeze_w
```

원인 분리 결과:

| Run | Best test acc | 판단 |
|-----|--------------:|------|
| `p=0.02,w=-2.25,train_w_raw=true` | 0.5455 | clamp saturation, reject |
| `p=0.02,w=-2.25,train_w_raw=false` | 0.5499 | best random recurrent, baseline 수준 |
| `p=0.03,w=-2.5,train_w_raw=false` | 0.5367 | density 증가로 reject |
| `p=0.05,w=-3.5,w_raw_max=-3.0` | 0.5477 | near baseline, reject |

진단 결론:

- `train_w_raw=false`는 clamp saturation을 제거하고 `p=0.02`를 baseline 수준으로 회복시켰다.
- 그러나 `p=0.03`에서는 frozen weight에서도 inhibitory recurrent current가 커지고 active neuron 수가 감소해 성능이 하락했다.
- random recurrent edge 수를 늘리는 방향은 현재 구조에서 맞지 않는다.
- random recurrence의 최선은 약하고 sparse한 보조 dynamics이며, robust improvement는 아직 없다.

### Step 5: Proposed C 학습

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=learned \
  liquid.train_w_raw=false \
  experiment_name=lsm_shd_C_freeze_w
```

목표:

- warmup 이후 theta가 분산되는지 확인
- sparsity가 의도한 범위에 수렴하는지 확인
- `p=0.02,w=-2.25,train_w_raw=false`가 보여준 baseline 수준 성능을 learned topology가 넘어서는지 확인
- learned topology가 edge placement를 개선하되 recurrent magnitude saturation을 피할 수 있는지 확인

고정 warmup C를 먼저 기준선으로 삼고, 그 다음 dynamic warmup ablation을 비교한다.

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=learned \
  liquid.train_w_raw=false \
  liquid.theta_warmup_dynamic=true \
  liquid.theta_warmup_strategy=slope \
  liquid.theta_warmup_window=3 \
  liquid.theta_warmup_min_epochs=5 \
  liquid.theta_warmup_patience=2 \
  liquid.theta_warmup_min_delta=0.01 \
  liquid.theta_warmup_metric=test_acc \
  experiment_name=lsm_shd_C_freeze_w_dynwarm_slope
```

### Step 6: 동일 희소성 B*

C의 최종 `sparsity`를 확인한 뒤:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=<C_final_sparsity>
```

목표:

- 연결 수가 아니라 연결 위치의 효과를 분리

### Step 7: A와 D

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=fixed
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=grad_r
```

목표:

- A -> B: recurrent weight 학습의 효과
- D -> C: hard threshold 대비 Gumbel/STE + annealing의 효과

---

## 5. 남은 구현/정리 작업

우선순위 높은 항목:

- 원인 분리 ablation 결과를 표로 고정.
- SHD 직접 HDF5 fallback 추가 여부 결정. 현재는 Tonic 의존.
- `src/lsm/model.py`의 import 중 `gumbel_sigmoid`, `gumbel_sigmoid_ste`, `List`는 현재 직접 사용되지 않는다. 정리 가능.
- `InputProjection` 설명을 모든 문서/코드 주석에서 mixed-sign random projection으로 통일.
- `commands.txt`가 실험 로그/명령 기록 용도라면 docs에서 명시하거나, 아니면 별도 관리 대상에서 제외.

분석 확장:

- E/I별 in/out degree 분석
- learned mask의 loop 수/길이 분석
- random sparse와 learned의 clustering/path length 비교
- SHD 이후 SSC 확장 계획 구체화
- C final sparsity와 B p-sweep optimum 비교

---

## 6. 체크리스트

구현 완료:

- [x] `LiquidConfig` 추가
- [x] `src/data/loaders.py`에 SHD loader 추가
- [x] `InputProjection` 구현
- [x] `LiquidLayer` 구현
- [x] `LSMModel` 구현
- [x] LSM 전용 trainer 구현
- [x] `configs/lsm_shd_baseline.yaml` 추가
- [x] `scripts/train_lsm.py` 추가
- [x] gradient clipping과 확장 logging 추가
- [x] membrane clamp, threshold clamp 추가
- [x] warmup + epoch-level noise 기반 topology 학습 추가
- [x] diagnostic script 추가
- [x] LSM용 `scripts/evaluate.py` 경로 수정
- [x] `liquid.train_w_raw` ablation 옵션 추가
- [x] recurrent E/I balance 진단 출력 추가

남음:

- [ ] HDF5 fallback 필요 여부 결정
- [ ] B/B*/C/D/A 비교 실험 실행 및 표 정리
- [ ] `w_raw` freeze / weak cap 원인 분리 ablation 실행
- [ ] learned topology 구조 분석 코드 추가
- [ ] 문서의 실험 결과 섹션을 실제 LSM 결과로 갱신
