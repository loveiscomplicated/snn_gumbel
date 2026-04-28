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

현재 tau schedule:

```python
phase2_epoch = max(epoch - theta_warmup_epochs, 0)
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

## 4. 실험 순서

### Step 1: Baseline B 안정화

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.1
```

목표:

- recurrent weight/readout 학습만으로 SHD에서 합리적 정확도 확보
- firing rate와 grad norm 안정화
- `p` sweep 후보 결정: 0.05, 0.1, 0.2, 0.3

### Step 2: Proposed C 학습

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=learned
```

목표:

- warmup 이후 theta가 분산되는지 확인
- sparsity가 의도한 범위에 수렴하는지 확인
- B 대비 accuracy 또는 efficiency 개선 확인

### Step 3: 동일 희소성 B*

C의 최종 `sparsity`를 확인한 뒤:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=<C_final_sparsity>
```

목표:

- 연결 수가 아니라 연결 위치의 효과를 분리

### Step 4: A와 D

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

- LSM 평가 CLI 경로 점검: `src/evaluation/evaluate.py`의 `run_evaluation()`은 현재 `model(x, tau=..., hard=True)` 형태를 호출하므로 `LSMModel`과 시그니처가 맞지 않는다. LSM 평가는 trainer 내부 eval 또는 visualize 경로는 동작하지만, CLI 평가 유틸은 수정이 필요하다.
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

남음:

- [ ] LSM용 `scripts/evaluate.py` 경로 수정
- [ ] HDF5 fallback 필요 여부 결정
- [ ] B/B*/C/D/A 비교 실험 실행 및 표 정리
- [ ] learned topology 구조 분석 코드 추가
- [ ] 문서의 실험 결과 섹션을 실제 LSM 결과로 갱신
