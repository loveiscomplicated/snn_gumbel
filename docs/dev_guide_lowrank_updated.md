# 개발 가이드: Gumbel-Softmax SNN / LSM 코드베이스

## 0. 목적

이 문서는 현재 코드베이스를 기준으로 실험을 실행하고 수정할 때 필요한 구현 정보를 정리한다. 현재 실험 수치나 해석은 `docs/lsm_current_baseline_lowrank_updated.md`를 우선 참고하고, 이 문서는 구현과 실행 규약에 초점을 맞춘다.

---

## 1. 현재 파일 구조

| 경로 | 역할 |
|------|------|
| `src/models/layers.py` | feedforward SNN용 `GumbelLIFLayer`, surrogate spike, Gumbel/STE 유틸 |
| `src/models/snn.py` | 임의 hidden layer 수를 지원하는 feedforward `SNNModel` |
| `src/training/trainer.py` | MNIST/FashionMNIST/NMNIST/DVS/SHD 학습 루프 |
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

지원 `topology.mode`:

| mode | 동작 |
|------|------|
| `learned` | 학습 중 Gumbel-Sigmoid, 평가 시 hard binary mask |
| `full` | 항상 fully-connected mask |
| `random_sparse` | 초기 fixed random mask 사용 |
| `transfer` | checkpoint에서 `theta` 로드 후 freeze |

### 2.2 LSM

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=random_sparse liquid.recurrent_sparsity=0.2
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=learned_lowrank liquid.theta_rank=16
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.n_liquid=200 epochs=1
```

지원 `recurrent_mode`:

| mode | 구조 | 가중치 |
|------|------|--------|
| `learned` | dense `theta` + Gumbel/STE mask | 학습 |
| `learned_lowrank` | `src_embed @ dst_embed.T + theta_bias` | 학습 |
| `random_sparse` | fixed random sparse mask | 학습 |
| `fixed` | fixed random sparse mask | freeze |
| `grad_r` | hard threshold forward + sigmoid-STE backward | 학습 |

진단:

```bash
python scripts/diagnose_liquid.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.02 \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  batch_size=8 \
  --batches 1 \
  --classes 5 \
  --samples-per-class 8
```

평가:

```bash
python scripts/evaluate.py \
  --config experiments/<exp>/config.yaml \
  --checkpoint experiments/<exp>/checkpoints/best.pt
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
2. 실험 YAML의 `base: base.yaml` 상속
3. CLI `key=value` override

중첩 override 예:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.n_liquid=300 \
  liquid.theta_warmup_epochs=5 \
  tau_hold_epochs=10
```

### 3.1 공통 필드

| 필드 | 의미 |
|------|------|
| `dataset` | `mnist`, `fashion_mnist`, `nmnist`, `dvs_gesture`, `shd` |
| `n_input`, `n_output` | 입력/출력 차원 |
| `T` | 시뮬레이션 타임스텝 수 |
| `tau_start`, `tau_end`, `tau_anneal_epochs`, `tau_hold_epochs` | temperature schedule |
| `lambda_sparse`, `lambda_commit`, `lambda_pred` | regularization |
| `lr`, `lr_min`, `weight_decay` | optimizer/scheduler |
| `seed` | 재현성 seed |

### 3.2 현재 코드 기본값

`src/utils/config.py` 기준 주요 기본값:

| 항목 | 값 |
|------|-----|
| `architecture.hidden_layers` | `[512]` |
| `topology.mode` | `learned` |
| `topology.target_sparsity` | `0.5` |
| `liquid.n_liquid` | `200` |
| `liquid.exc_ratio` | `0.8` |
| `liquid.p_input` | `0.1` |
| `liquid.recurrent_mode` | `learned` |
| `liquid.recurrent_sparsity` | `0.2` |
| `liquid.self_connection` | `false` |
| `liquid.theta_init_mean/std` | `0.0 / 0.01` |
| `liquid.theta_rank` | `16` |
| `liquid.theta_lowrank_init_std` | `0.30` |
| `liquid.w_raw_init_mean/std` | `-4.0 / 0.01` |
| `liquid.train_w_raw` | `true` |
| `liquid.w_raw_max` | `-3.0` |
| `liquid.bptt_truncate` | `0` |
| `liquid.beta_min/max` | `0.7 / 0.95` |
| `liquid.threshold_min/max` | `0.8 / 1.5` |
| `liquid.theta_warmup_epochs` | `0` |
| `liquid.theta_warmup_dynamic` | `false` |
| `liquid.theta_lr_scale` | `0.1` |
| `liquid.theta_lr_ramp_epochs` | `1` |
| `liquid.theta_bias_lr_scale` | `1.0` |
| `liquid.theta_freeze_epoch` | `0` |
| `liquid.theta_adaptive_freeze` | `false` |
| `liquid.noise_scale` | `0.1` |
| `liquid.pred_aux_enabled` | `false` |

### 3.3 현재 SHD baseline override

`configs/lsm_shd_baseline.yaml`에서 덮어쓰는 값:

| 항목 | 값 |
|------|-----|
| dataset | `shd` |
| input/output | `700 -> 20` |
| `T` | `100` |
| `liquid.n_liquid` | `500` |
| `liquid.exc_ratio` | `0.8` |
| `liquid.p_input` | `0.1` |
| `liquid.recurrent_mode` | `learned` |
| `liquid.recurrent_sparsity` | `0.1` |
| `liquid.theta_init_mean/std` | `-2.2 / 0.5` |
| `liquid.w_raw_init_mean/std` | `-4.0 / 0.01` |
| `liquid.train_w_raw` | `true` |
| `liquid.w_raw_max` | `-3.0` |
| `liquid.bptt_truncate` | `25` |
| `liquid.theta_warmup_epochs` | `10` |
| `liquid.theta_lr_scale` | `0.3` |
| `liquid.theta_freeze_epoch` | `0` |
| `liquid.theta_adaptive_freeze` | `false` |
| `liquid.noise_scale` | `0.1` |
| `liquid.input_weight_scale` | `0.3` |
| `tau_hold_epochs / tau_anneal_epochs` | `15 / 40` |
| `epochs / patience / batch_size` | `100 / 40 / 64` |
| `lr / lr_min / weight_decay` | `0.001 / 0.00001 / 0.0001` |
| `lambda_sparse / lambda_commit` | `0.1 / 0.01` |

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
- `GumbelLIFLayer.forward()`는 `learned`, `full`, `random_sparse`, `transfer`를 지원한다.
- `hard=True`는 평가 시 deterministic binary mask를 만든다.
- `get_binary_mask()`는 `torch.sigmoid(theta) >= 0.5` 기준을 쓴다.
- `random_sparse`는 초기 mask를 buffer로 고정하고, `transfer`는 외부 checkpoint의 `theta`를 load한 뒤 freeze한다.

---

## 5. LSM 구현 메모

### 5.1 모델 구성

```text
SHD spikes (batch, T, 700)
  -> InputProjection: fixed sparse random projection (700, N)
  -> LiquidLayer: recurrent mask * Dale's Law weight (N, N)
  -> Readout: nn.Linear(N, 20), spike count 평균
```

`InputProjection`은 `register_buffer("weight", weight)`로 고정된다. 현재 구현은 양수만이 아니라 `torch.randn(...) * weight_scale`을 사용해 입력별 흥분/억제 성분이 섞인 random projection을 만든다.

### 5.2 LiquidLayer

학습 파라미터:

| 이름 | shape | 조건 |
|------|-------|------|
| `theta` | `(N, N)` | `mode == "learned"` 또는 `mode == "grad_r"` |
| `src_embed`, `dst_embed`, `theta_bias` | `(N, r)`, `(N, r)`, scalar | `mode == "learned_lowrank"` |
| `w_raw` | `(N, N)` | `mode != "fixed"`이고 `train_w_raw=true`일 때 학습 |
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

핵심 해석:

- `w_raw_max`는 상한 clamp다. 작은 초기값을 키우지 않는다.
- recurrent strength를 키우려면 `w_raw_init_mean/std`를 조정해야 한다.
- `train_w_raw=false`는 `w_raw`만 freeze하고, `beta`, `threshold`, `readout`은 계속 학습한다.
- `dale_sign`은 `(N, 1)`이어야 presynaptic row 기준 Dale's Law가 유지된다.

### 5.3 Mask 샘플링

`LSMModel.forward()`는 시작 시 `self.liquid.sample_mask(tau=tau)`를 1회 호출하고, 모든 timestep에서 같은 `current_mask`를 사용한다.

`learned`와 `learned_lowrank`의 기본 동작은 2-phase다.

| Phase | 조건 | 동작 |
|-------|------|------|
| P1 | `epoch < theta_warmup_epochs` | theta freeze, deterministic hard mask, non-theta params 학습 |
| P2 | `epoch >= theta_warmup_epochs` | theta unfreeze, epoch-level Gumbel noise 1회 샘플링, batch마다 동일 noise로 STE 재계산 |

추가 규칙:

- `sample_epoch_mask(tau, epoch_noise)`는 epoch 단위 noise만 저장한다.
- `unlock_epoch_mask()`는 eval 직전에 deterministic mask로 돌아가게 한다.
- `theta_freeze_epoch > 0`이면 해당 epoch 시작 시 topology를 freeze하고 deterministic mask로 고정한다.
- `theta_adaptive_freeze=true`이면 `theta_grad_norm` 기준으로 adaptive freeze를 적용한다.

### 5.3.1 `learned`

`learned`는 dense `theta`를 사용한다. 학습 중에는 epoch-level noise를 넣은 Gumbel-STE를 사용하고, eval 또는 warmup 중에는 deterministic mask를 사용한다.

### 5.3.2 `grad_r`

`grad_r`는 Gumbel noise를 쓰지 않는다. forward는 deterministic hard threshold, backward는 sigmoid-STE다.

```python
soft = torch.sigmoid(theta)
hard = (theta > 0).float()
mask = hard.detach() - soft.detach() + soft
```

주의:

- `grad_r`에서도 `theta.requires_grad=True`여야 한다.
- `grad_r`는 theta 전용 optimizer group, theta clipping, theta-grad logging 대상이다.
- 이전의 non-STE hard threshold 구현은 theta가 사실상 학습되지 않았기 때문에 현재 baseline으로 쓰지 않는다.

### 5.3.3 `learned_lowrank`

`learned_lowrank`는 edge마다 독립적인 dense `theta`를 두지 않는다. 대신 source neuron embedding, destination neuron embedding, scalar bias로 topology logits를 만든다.

```python
topology_logit = src_embed @ dst_embed.T + theta_bias
```

주의:

- `learned_lowrank`에서는 dense `self.theta`가 없어야 한다.
- `topology_parameters()`는 `src_embed`, `dst_embed`, `theta_bias`를 반환해야 한다.
- P1 warmup에서는 세 파라미터가 모두 freeze되어야 한다.
- `theta_lr_ramp_epochs`와 `theta_bias_lr_scale`가 low-rank 안정화에 사용된다.

### 5.3.4 `random_sparse` / `fixed`

- 두 모드 모두 init 시 `fixed_mask`를 만든다.
- `random_sparse`는 `w_raw`, `beta`, `threshold`, `readout`을 학습할 수 있다.
- `fixed`는 recurrent weight까지 freeze하는 전통적인 고정 LSM에 가깝다.
- 둘 다 timestep마다 mask를 다시 샘플하지 않는다.

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
- 기본 bin width는 `dt_us=10_000.0`, 즉 10ms다.
- `T=100`이면 약 1초를 커버한다.
- 같은 bin/channel에 여러 event가 들어오면 현재 구현은 count가 아니라 binary `1.0`으로 저장한다.

---

## 7. Loss와 학습 안정성

Feedforward:

```python
loss = CE + lambda_sparse * model.sparsity_loss() + lambda_commit * model.commitment_loss()
```

LSM도 같은 형태지만 sparsity/commitment/prediction loss는 liquid topology에만 적용된다. `random_sparse`, `fixed`, `grad_r`는 sparsity/commitment loss가 사실상 0이다.

LSM 학습 안정화 장치:

- `theta_warmup_epochs`: topology 고정 후 weight/readout 선학습
- `theta_warmup_dynamic`: learned mode에서 plateau 감지 시 P2로 조기 전환
- theta와 나머지 파라미터 optimizer group 분리
- theta LR scale 및 low-rank용 `theta_bias_lr_scale`
- weights/readout과 theta에 서로 다른 `clip_grad_norm_`
- `w_raw_max`로 recurrent magnitude 상한 설정
- `w_raw_init_mean/std`로 recurrent magnitude 초기 scale 조정
- membrane clamp `[-3, 3]`
- threshold clamp `min=0.01`
- `bptt_truncate`로 마지막 K step에만 gradient 흐름
- NaN loss 감지 시 학습 중단 및 기록 저장

로그 항목은 구현상 다음 계열을 중심으로 본다.

| 키 | 의미 |
|----|------|
| `phase` | `P1` 또는 `P2` |
| `tau` | 현재 temperature |
| `sparsity` | deterministic binary liquid mask density |
| `theta_mean`, `theta_std` | topology logits 통계 |
| `grad_norm` | non-theta parameter grad norm |
| `theta_grad_norm`, `w_raw_grad_norm` | component별 grad norm |
| `mean_firing_rate`, `max_firing_rate` | 마지막 forward 기준 발화율 |

---

## 8. 주의할 점

- `InputProjection`은 mixed-sign random projection이다. 예전처럼 excitatory-only라고 쓰면 틀리다.
- `grad_clip_max_norm` 단일 필드는 현재 사용하지 않는다. LSM에서는 `grad_clip_max_norm_w`, `grad_clip_max_norm_theta`를 쓴다.
- `w_raw_max`는 한쪽 clamp라서, 이것만 조정해도 recurrent current가 강해지지 않을 수 있다.
- `learned_lowrank`는 dense edge-wise theta를 대체하는 구조이므로, 기존 `theta_*` 해석을 그대로 적용하면 안 된다.
- 평가 CLI는 LSM이면 `cfg.liquid.n_liquid > 0 and not cfg.architecture.hidden_layers` 조건으로 모델을 고른다.

