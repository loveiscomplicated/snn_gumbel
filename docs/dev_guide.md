# 개발 가이드: Gumbel-Softmax SNN / LSM 코드베이스

## 0. 목적

이 문서는 현재 코드베이스를 기준으로 실험을 실행하고 수정할 때 필요한 구현 정보를 정리한다. 연구 배경과 논문 포지셔닝은 `context.md`, `context_2.md`를 참조한다.

---

## 1. 현재 파일 구조

| 경로 | 역할 |
|------|------|
| `src/models/layers.py` | feedforward SNN용 `GumbelLIFLayer`, surrogate spike, Gumbel/Sigmoid STE 유틸 |
| `src/models/snn.py` | 임의 hidden layer 수를 지원하는 feedforward `SNNModel` |
| `src/training/trainer.py` | MNIST/Fashion/NMNIST/DVS 등 feedforward 학습 루프 |
| `src/training/losses.py` | CE + sparsity + commitment loss |
| `src/data/loaders.py` | MNIST, Fashion-MNIST, NMNIST, DVS Gesture, SHD DataLoader |
| `src/lsm/model.py` | `InputProjection`, `LiquidLayer`, `LSMModel` |
| `src/lsm/trainer.py` | SHD LSM 학습 루프, 2-phase topology learning, gradient clipping |
| `src/evaluation/evaluate.py` | checkpoint 로드 및 평가 유틸 |
| `src/evaluation/visualize.py` | feedforward/LSM 시각화 |
| `scripts/train.py` | feedforward 학습 CLI |
| `scripts/train_lsm.py` | LSM 학습 CLI |
| `scripts/diagnose_liquid.py` | LSM beta/threshold/separation 진단 |
| `scripts/evaluate.py` | 평가 CLI |
| `scripts/visualize.py` | 시각화 CLI |

---

## 2. 실행 방법

### 2.1 Feedforward SNN

```bash
python scripts/train.py --config configs/mnist_baseline.yaml
python scripts/train.py --config configs/ablation_full.yaml epochs=5
python scripts/train.py --config configs/ablation_random_sparse.yaml topology.target_sparsity=0.38
```

지원 topology mode:

| mode | 동작 |
|------|------|
| `learned` | Gumbel-Sigmoid로 theta 학습 |
| `full` | 모든 연결 사용, theta 무시 |
| `random_sparse` | 초기 fixed random mask 사용 |
| `transfer` | checkpoint에서 theta 로드 후 freeze |

### 2.2 LSM

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=random_sparse liquid.recurrent_sparsity=0.2
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.n_liquid=200 epochs=1
```

지원 recurrent mode:

| mode | 구조 | 가중치 |
|------|------|--------|
| `learned` | theta 기반 학습 mask | 학습 |
| `random_sparse` | fixed random sparse mask | 학습 |
| `fixed` | fixed random sparse mask | freeze |
| `grad_r` | `(theta > 0)` hard threshold | 학습 |

진단:

```bash
python scripts/diagnose_liquid.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=fixed
```

시각화:

```bash
python scripts/visualize.py \
  --config experiments/<exp>/config.yaml \
  --checkpoint experiments/<exp>/checkpoints/best.pt \
  --figures-dir experiments/<exp>/figures
```

---

## 3. Config 체계

설정 로딩 순서:

1. `configs/base.yaml`
2. 실험 YAML의 `base: base.yaml`
3. CLI `key=value` override

중첩 override 예:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.n_liquid=300 \
  liquid.theta_warmup_epochs=5 \
  tau_hold_epochs=10
```

### 3.1 주요 공통 필드

| 필드 | 의미 |
|------|------|
| `dataset` | `mnist`, `fashion_mnist`, `nmnist`, `dvs_gesture`, `shd` |
| `n_input`, `n_output` | 입력/출력 차원 |
| `T` | 시뮬레이션 타임스텝 수 |
| `tau_start`, `tau_end`, `tau_anneal_epochs`, `tau_hold_epochs` | temperature schedule |
| `lambda_sparse`, `lambda_commit` | topology regularization |
| `lr`, `lr_min`, `weight_decay` | optimizer/scheduler |

### 3.2 현재 LSM 기본값

`configs/lsm_shd_baseline.yaml` 기준:

| 항목 | 값 |
|------|-----|
| dataset | `shd` |
| input/output | 700 → 20 |
| T | 100 |
| liquid size | 500 |
| E/I ratio | 0.8 |
| input projection | `p_input=0.1`, `input_weight_scale=0.3`, fixed `randn` |
| recurrent mode | `learned` |
| initial recurrent density bias | `theta_init_mean=-2.2`, `theta_init_std=0.5` |
| recurrent weight cap | `w_raw_max=-3.0` |
| BPTT | `bptt_truncate=25` |
| warmup | `theta_warmup_epochs=10` |
| theta LR | `lr * theta_lr_scale`, 현재 `0.3` |
| Gumbel exploration | `noise_scale=0.1` |
| gradient clipping | weights/readout `100.0`, theta `10.0` |
| tau schedule | start 1.0, hold 15 epochs, anneal 40 epochs, end 0.05 |

---

## 4. Feedforward SNN 구현 메모

`SNNModel`은 `[n_input] + hidden_layers + [n_output]`로 레이어를 구성한다. 마지막 출력층은 threshold를 학습하지 않는다.

Forward 흐름:

```python
for _ in range(T):
    spike = (torch.rand_like(x) < x).float()
    for layer in self.layers:
        current = layer(spike, tau=tau, hard=hard)
        mem = layer.beta * mem + current
        spike = spike_fn(mem - layer.threshold.clamp(min=0.01))
        mem = mem * (1.0 - spike)
    spike_sum += spike
return spike_sum / T
```

주의:

- MNIST/Fashion-MNIST는 flatten 후 0~1로 재정규화되어 rate coding 입력으로 사용된다.
- `GumbelLIFLayer.forward()`는 호출 때마다 mask를 계산한다. feedforward에서는 기존 실험 결과와 호환하기 위해 이 구조를 유지한다.
- 평가에서는 `hard=True`로 deterministic mask를 사용한다.
- `get_binary_mask()`는 현재 `> 0.5` threshold를 사용한다. 학습 중 sparsity 해석 시 경계값 0.5 근처의 차이에 유의한다.

---

## 5. LSM 구현 메모

### 5.1 모델 구성

```
SHD spikes (batch, T, 700)
  -> InputProjection: fixed sparse randn projection (700, N)
  -> LiquidLayer: recurrent mask * Dale's Law weight (N, N)
  -> Readout: nn.Linear(N, 20), spike count 평균
```

`InputProjection`은 `register_buffer("weight", weight)`로 고정된다. 현재 구현은 양수만이 아니라 `torch.randn(...) * weight_scale`을 사용해 입력별 흥분/억제 성분이 섞인 random projection을 만든다.

### 5.2 LiquidLayer

학습 파라미터:

| 이름 | shape | 조건 |
|------|-------|------|
| `theta` | `(N, N)` | `mode == "learned"`일 때 학습 |
| `w_raw` | `(N, N)` | `mode != "fixed"`일 때 학습 |
| `logit_beta` | `(N,)` | `mode != "fixed"`일 때 학습 |
| `threshold` | `(N,)` | `mode != "fixed"`일 때 학습 |

버퍼:

| 이름 | shape | 의미 |
|------|-------|------|
| `dale_sign` | `(N, 1)` | presynaptic 뉴런 기준 흥분(+1)/억제(-1) |
| `self_conn_mask` | `(N, N)` | 자기 연결 제외 시 diagonal 0 |
| `fixed_mask` | `(N, N)` 또는 `None` | `random_sparse`, `fixed` 모드용 |

유효 가중치:

```python
w_clamped = torch.clamp(self.w_raw, max=self.w_raw_max)
signed_w = self.dale_sign * F.softplus(w_clamped)
w_eff = self.current_mask * self.self_conn_mask * signed_w
```

행(row)이 presynaptic 뉴런이다. `dale_sign`을 `(1, N)`으로 만들면 postsynaptic 기준 부호가 되어 Dale's Law가 깨진다.

### 5.3 Mask 샘플링

LSM에서는 mask를 타임스텝마다 바꾸지 않는다. `LSMModel.forward()` 시작 시 `self.liquid.sample_mask(tau)`를 한 번 호출하고, 모든 `T` 동안 같은 `current_mask`를 사용한다.

`learned` 모드의 현재 학습 정책:

1. Phase 1 warmup: `theta.requires_grad_(False)`, deterministic hard mask로 weight/readout 먼저 안정화
2. Phase 2: epoch 시작 시 Gumbel noise를 한 번 샘플링해 `sample_epoch_mask()`에 저장
3. 각 batch forward에서 같은 epoch noise로 STE mask를 새 graph로 재계산
4. epoch 평가 전 `unlock_epoch_mask()`를 호출해 deterministic mask로 평가

이 설계의 목적은 recurrent BPTT에서 batch마다 topology가 크게 흔들리는 것을 막으면서, epoch 간에는 OFF edge가 ON으로 탐색될 기회를 주는 것이다.

### 5.4 LSM Forward

```python
self.liquid.sample_mask(tau=tau)

liquid_mem = zeros(batch, N)
liquid_spike = zeros(batch, N)
readout_mem = zeros(batch, n_output)

grad_start = T - bptt_truncate if bptt_truncate > 0 else 0

for t in range(T):
    if t == grad_start and t > 0:
        liquid_mem = liquid_mem.detach()
        liquid_spike = liquid_spike.detach()

    input_current = self.input_proj(spikes[:, t])
    recurrent_current = self.liquid(liquid_spike)
    liquid_mem = self.liquid.beta * liquid_mem + input_current + recurrent_current
    liquid_mem = torch.clamp(liquid_mem, -3.0, 3.0)
    liquid_spike = spike_fn(liquid_mem - self.liquid.threshold.clamp(min=0.01))
    liquid_mem = liquid_mem * (1.0 - liquid_spike)
    readout_mem += self.readout(liquid_spike)

return readout_mem / T
```

---

## 6. 데이터 파이프라인

`src/data/loaders.py`가 모든 DataLoader를 담당한다.

| dataset | loader | 출력 shape |
|---------|--------|------------|
| `mnist` | `torchvision.datasets.MNIST` | `(batch, 784)` |
| `fashion_mnist` | `torchvision.datasets.FashionMNIST` | `(batch, 784)` |
| `nmnist` | `tonic.datasets.NMNIST` + `ToFrame` | `(batch, T, 2312)` |
| `dvs_gesture` | `tonic.datasets.DVSGesture` + downsample 32x32x2 | `(batch, T, 2048)` |
| `shd` | `tonic.datasets.SHD` + 직접 binning | `(batch, T, 700)` |

SHD binning:

- `events["t"]`는 microsecond 단위로 처리한다.
- 기본 bin width는 `dt_us=10_000.0`, 즉 10ms.
- `T=100`이면 약 1초를 커버한다.
- 같은 bin/channel에 여러 event가 들어오면 현재 구현은 count가 아니라 binary 1로 저장한다.

---

## 7. Loss와 학습 안정성

Feedforward:

```python
loss = CE + lambda_sparse * model.sparsity_loss() + lambda_commit * model.commitment_loss()
```

LSM도 동일한 형태지만 sparsity/commitment는 liquid theta에만 적용된다. `random_sparse`, `fixed`, `grad_r`에서는 두 loss가 0을 반환한다.

LSM 학습 안정화 장치:

- `theta_warmup_epochs`: topology 고정 후 weight/readout 선학습
- theta와 나머지 파라미터 optimizer group 분리
- theta LR scale 적용
- weights/readout과 theta에 서로 다른 `clip_grad_norm_`
- `w_raw_max`로 recurrent magnitude 상한
- membrane clamp `[-3, 3]`
- threshold clamp `min=0.01`
- `bptt_truncate`로 마지막 K step에만 gradient 흐름
- NaN loss 감지 시 history와 exp_dir 반환 후 중단

로그 항목:

| 키 | 의미 |
|----|------|
| `phase` | `P1` warmup 또는 `P2` topology learning |
| `tau` | 현재 temperature |
| `sparsity` | deterministic binary liquid mask density |
| `theta_mean`, `theta_std` | theta 분포 |
| `grad_norm` | non-theta parameter grad norm |
| `theta_grad_norm`, `w_raw_grad_norm` | component별 grad norm |
| `mean_firing_rate`, `max_firing_rate` | 마지막 forward 기준 발화율 |

경고 기준:

- `grad_norm > 100`
- `theta_grad_norm > 50`
- `w_raw_grad_norm > 50`
- `max_firing_rate > 0.9`
- epoch 20 이후 `theta_std < 0.01`

---

## 8. 실험 결과 해석 체크리스트

Feedforward:

- 학습된 topology가 full/random_sparse보다 성능 또는 효율에서 나은가
- `sigma(theta)`가 bimodal인지
- MNIST input connectivity가 중앙 receptive field를 형성하는지
- hidden threshold/beta가 분화되는지

LSM:

- Phase 1에서 random/fixed topology로 separation property가 확보되는지
- Phase 2에서 theta가 정체하지 않고 분산되는지
- learned mask density가 과도하게 0 또는 1로 붕괴하지 않는지
- learned vs same-density random sparse 비교에서 차이가 나는지
- E/I out-degree, in-degree, hub, loop 구조가 random baseline과 다른지

---

## 9. 주의할 점

- `src/lsm/model.py`는 현재 작업 트리에 수정된 상태다. 문서 갱신 외 작업을 할 때 사용자의 변경을 덮어쓰지 않는다.
- LSM의 `InputProjection` docstring은 “Mixed excitatory/inhibitory”가 실제 동작과 맞다. 예전 문서의 “흥분성만” 설명은 더 이상 맞지 않는다.
- `grad_clip_max_norm` 단일 필드는 현재 사용하지 않는다. LSM에서는 `grad_clip_max_norm_w`, `grad_clip_max_norm_theta`를 쓴다.
- `weight_decay`는 `Config`에는 있으나 `configs/base.yaml`에는 명시되어 있지 않아 기본값 0.0이 적용된다. LSM YAML은 0.0001로 override한다.
- `scripts/train_lsm.py`는 argparse가 아니라 `sys.argv` 기반이다. 첫 인자는 config path, 이후는 `key=value` override다.
- LSM eval은 모델 내부에서 eval 모드일 때 deterministic mask를 사용한다. `LSMModel.forward()`는 `hard` 인자를 받지 않는다.
