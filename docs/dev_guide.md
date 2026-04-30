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
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.recurrent_mode=random_sparse liquid.recurrent_sparsity=0.02 liquid.w_raw_init_mean=-2.25 liquid.w_raw_max=-2.0
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
| recurrent raw weight init | `w_raw_init_mean=-4.0`, `w_raw_init_std=0.01` |
| recurrent raw weight training | `train_w_raw=true` |
| recurrent weight cap | `w_raw_max=-3.0` |
| BPTT | `bptt_truncate=25` |
| warmup | `theta_warmup_epochs=10` |
| theta LR | `lr * theta_lr_scale`, 현재 `0.3` |
| theta freeze | `theta_freeze_epoch=0` disables scheduled freeze |
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

`w_raw`는 `Normal(w_raw_init_mean, w_raw_init_std)`로 초기화된다. `w_raw_max`는 상한 clamp라서 작은 초기값을 키우지 않는다. recurrent strength를 키우려면 `w_raw_init_mean`을 덜 음수로 조정해야 한다. `liquid.train_w_raw=false`를 주면 `random_sparse`에서 `w_raw`만 freeze하고 `beta`, `threshold`, `readout`은 계속 학습할 수 있다.

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
- `theta_warmup_dynamic`: P1 성능 정체 시 P2로 조기 전환하는 learned topology 실험 옵션
- theta와 나머지 파라미터 optimizer group 분리
- theta LR scale 적용
- weights/readout과 theta에 서로 다른 `clip_grad_norm_`
- `w_raw_max`로 recurrent magnitude 상한
- `w_raw_init_mean/std`로 recurrent magnitude 초기 scale 조정
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

Dynamic warmup 옵션:

| 키 | 의미 |
|----|------|
| `liquid.theta_warmup_dynamic` | `true`이면 learned mode P1 중 plateau 감지 후 P2로 조기 전환 |
| `liquid.theta_warmup_strategy` | `slope` 또는 `best`; 기본은 최근 window 기울기를 보는 `slope` |
| `liquid.theta_warmup_window` | `slope` 전략에서 사용할 최근 P1 epoch 수 |
| `liquid.theta_warmup_min_epochs` | dynamic 전환 전 반드시 유지할 최소 P1 epoch |
| `liquid.theta_warmup_patience` | 둔화 또는 best 미갱신을 허용할 연속 check 수 |
| `liquid.theta_warmup_min_delta` | 개선 또는 평균 기울기로 인정할 최소 변화량 |
| `liquid.theta_warmup_metric` | `test_acc`, `train_acc`, `train_loss` 중 하나 |

기본값은 `theta_warmup_dynamic=false`다. 기존 실험 재현성은 고정 `theta_warmup_epochs`를 기준으로 유지하고, dynamic warmup은 learned topology C의 별도 ablation으로 비교한다. `slope` 전략은 metric을 score로 변환한 뒤 최근 window의 평균 epoch당 개선폭이 `theta_warmup_min_delta`보다 작은 상태가 `theta_warmup_patience`번 이어지면 P2로 넘어간다. `train_loss`는 작을수록 좋으므로 내부 score는 `-train_loss`를 사용한다.

경고 기준:

- `grad_norm > 100`
- `theta_grad_norm > 50`
- `w_raw_grad_norm > 50`
- `max_firing_rate > 0.9`
- epoch 20 이후 `theta_std < 0.01`

---

## 8. 현재 SHD LSM 결론과 다음 순서

현재 기준 결과:

| run | best test acc | 판단 |
|-----|--------------:|------|
| `p=0.0` no recurrence | 0.5490 | baseline |
| `p=0.02`, `w_raw_init_mean=-2.25` | 0.5455 | reject |
| `p=0.03`, `w_raw_init_mean=-2.5` | 0.5269 | reject |
| `p=0.02`, `w_raw_init_mean=-2.25`, `train_w_raw=false` | 0.5499 | best random recurrent so far |
| `p=0.03`, `w_raw_init_mean=-2.5`, `train_w_raw=false` | 0.5367 | reject |
| `p=0.05`, `w_raw_init_mean=-3.5`, `w_raw_max=-3.0` | 0.5477 | near baseline |
| learned C, `theta_init_mean=-2.2`, `train_w_raw=false` | 0.5433 | too sparse, reject |
| learned C, `theta=-1.0,std=0.5,w=-2.25,freeze_w`, seed 42 | 0.5689 | success |
| learned C, same setting, seed 43 | 0.5751 | best tau=0.05 run |
| learned C, same setting, seed 44 | 0.5331 | seed-sensitive failure |
| learned C, same setting, seed 45 | 0.5587 | success, late theta-grad instability |
| learned C, same setting, `tau_end=0.2`, seed 42 | 0.5782 | tau-stabilized |
| learned C, same setting, `tau_end=0.2`, seed 43 | 0.5795 | current best, late theta-grad still spikes |
| learned C, same setting, `tau_end=0.2`, `theta_lr_scale=0.05`, seed 43 | 0.5530 | stable but topology under-opens |
| learned C, same setting, `tau_end=0.2`, `theta_lr_scale=0.3`, `theta_freeze_epoch=64`, seed 42 | 0.5764 | peak mostly preserved |
| learned C, same setting, `tau_end=0.2`, `theta_lr_scale=0.3`, `theta_freeze_epoch=64`, seed 43 | 0.5795 | current stable candidate |
| learned C, `theta=-1.2,std=0.5,w=-2.25,freeze_w` | 0.5442 | reject |
| learned C, `theta=-0.8,std=0.3,w=-2.25,freeze_w` | 0.5389 | reject; not density-preserving |
| random sparse, `p=0.041,w=-2.25,freeze_w` | 0.5216 | same-density random control failed |

Random-sparse trained checkpoint 진단 결과, recurrent current는 증가했지만 active neuron 수와 class separation은 악화됐다. `w_raw`를 학습하면 대부분 `w_raw_max=-2.0`에 포화되어 active recurrent edge가 near-uniform magnitude로 동작한다. `train_w_raw=false`는 이 포화를 제거하고 `p=0.02`에서 baseline 수준 성능을 회복했지만, density를 `p=0.03`으로 올리면 inhibitory recurrent current가 커지고 active neuron 수가 더 줄어 성능이 하락했다.

Learned C는 hard density와 topology placement가 모두 중요하다. 기본 `theta_init_mean=-2.2`는 best checkpoint에서도 density가 `0.0006`, `|rec|/|input|=0.0005`라 사실상 no-recurrence였다. `theta_init_mean=-1.0,std=0.5`, `w_raw_init_mean=-2.25`, `train_w_raw=false`는 seed 42/43/45에서 각각 best test `0.5689`, `0.5751`, `0.5587`을 기록했다. 그러나 seed 44는 `0.5331`로 실패해 아직 seed-sensitive하다. `tau_end=0.2`는 seed 42/43을 각각 `0.5782`, `0.5795`로 올렸다. `theta_freeze_epoch=64`는 seed 42/43에서 peak 성능을 거의 유지하면서, seed 43의 후반 test collapse를 줄였다.

주의: `theta=-1.0,std=0.5`의 초기 hard density는 `0.05~0.06`이 아니라 약 `P(N(-1.0,0.5)>0)=0.023`이다. 성공 checkpoint에서 학습 후 hard density가 `0.04~0.06` 범위로 올라간다.

Learned C seed comparison:

| seed | best test acc | density | `\|rec\|/\|input\|` | firing mean/max | active `>0.05` | cosine mean/min |
|------|--------------:|--------:|--------------------:|----------------:|---------------:|----------------:|
| 42 | 0.5689 | 0.0578 | 0.3261 | 0.0804 / 0.7800 | 257 / 500 | 0.9556 / 0.9142 |
| 43 | 0.5751 | 0.0409 | 0.1788 | 0.0625 / 0.6800 | 194 / 500 | 0.9724 / 0.9430 |
| 44 | 0.5331 | 0.0581 | 0.2756 | 0.0712 / 0.8400 | 214 / 500 | 0.9707 / 0.9421 |
| 45 | 0.5587 | n/a | n/a | 0.091 / 0.563 late | n/a | n/a |
| 42, `tau_end=0.2` | 0.5782 | 0.0610 late | n/a | 0.083 / 0.678 final | n/a | n/a |
| 43, `tau_end=0.2` | 0.5795 | 0.0620 final | n/a | 0.100 / 0.660 final | n/a | n/a |
| 42, `tau_end=0.2`, `theta_freeze_epoch=64` | 0.5764 | 0.0579 | 0.3315 | 0.0815 / 0.7800 | 255 / 500 | 0.9594 / 0.9179 |
| 43, `tau_end=0.2`, `theta_freeze_epoch=64` | 0.5795 | 0.0597 | 0.3177 | 0.0776 / 0.8000 | 254 / 500 | 0.9573 / 0.9214 |

해석:

- Learned C는 random-sparse보다 높은 성능 가능성을 보였다.
- 실패한 seed 44는 recurrent가 약한 것이 아니라 오히려 더 dense하고 더 강하다.
- Seed 42-45 best accuracy의 중앙값은 `0.5638`이며, no-recurrence baseline `0.5490`보다 높다.
- 따라서 다음 병목은 edge count 증가가 아니라 edge placement, recurrent current scale, tau annealing 후반의 topology gradient 안정화다.
- `p ~= 0.041` same-density random_sparse control은 best test `0.5216`으로 실패했으므로 learned C의 이득은 density만으로 설명되지 않는다.
- `tau_end=0.2`는 seed 42/43 모두에서 best test를 개선했으므로 현재 기본 후보로 승격한다.
- seed 43에서는 `tau=0.2`에서도 후반 `theta_grad_norm > 50` spike가 반복됐으므로, 다음 병목은 tau 하한보다 theta update scale일 가능성이 높다.
- `theta_lr_scale=0.05`는 theta-gradient spike를 제거했지만 sparsity가 `~0.023`에 머물러 learned topology가 거의 열리지 않았고 best test도 `0.5530`으로 하락했다. `theta_lr_scale=0.075`도 초반/중반 sparsity가 `~0.024`에 머물렀으므로, 단순 LR 축소는 1차 해법이 아니다.
- `theta_lr_scale=0.1` 역시 `theta_freeze_epoch=60`과 함께 쓰면 sparsity가 `~0.025`에 머물러 under-opened 상태가 된다. 성공한 freeze 실험은 `theta_lr_scale=0.3`을 유지하고 epoch 64부터 topology를 고정한 설정이다.
- `theta_freeze_epoch=64`는 seed 43에서 best `0.5795`를 유지하고 final test를 기존 `0.5252`에서 `0.5663`으로 끌어올렸다. Freeze 이후 `theta_grad=0`, sparsity `~0.060`, test `~0.56~0.57` 범위를 유지했다.
- Seed 42/43의 freeze64 checkpoint는 density `~0.058~0.060`, `|rec|/|input| ~0.32`, firing mean `~0.08`, active neurons `>0.05` `~255/500`, cosine mean/min `~0.96/~0.92`로 유사한 regime에 수렴했다.
- Mean-rate cosine diagnostic은 유용하지만 충분하지 않다. seed 43/44의 cosine summary는 비슷한데 accuracy는 크게 다르다.

다음 순서:

1. 자유로운 `w_raw` 학습은 clamp saturation 문제 때문에 보류한다.
2. 현재 learned C 안정화 기준은 `tau_end=0.2`, `theta_lr_scale=0.3`, `liquid.theta_freeze_epoch=64`다.
3. `std=0.3` 계열을 다시 볼 경우 `theta=-0.8`이 아니라 `theta=-0.6` 또는 `theta=-0.5`를 사용한다.
4. 다음 확인은 seed 44에 freeze64를 적용해 seed-sensitive failure를 완화하는지 보는 것이다.

원인 분리용 `w_raw` freeze 예:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.02 \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.train_w_raw=false \
  experiment_name=lsm_shd_rs_p002_w225_freeze_w
```

현재 learned C 기준 후보:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=learned \
  liquid.train_w_raw=false \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.theta_init_mean=-1.0 \
  liquid.theta_init_std=0.5 \
  tau_end=0.2 \
  liquid.theta_lr_scale=0.3 \
  liquid.theta_freeze_epoch=64 \
  experiment_name=lsm_shd_C_freeze_w_theta100_w225_tau020_tlr030_freeze64
```

Same-density random control, already failed:

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=random_sparse \
  liquid.recurrent_sparsity=0.041 \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.train_w_raw=false \
  experiment_name=lsm_shd_rs_p0041_w225_freeze_w
```

`scripts/diagnose_liquid.py`는 E/I edge balance, incoming E/I degree, excitatory/inhibitory recurrent current scale을 함께 출력한다. 가장 안전한 호출 형태는 named args를 먼저 쓰고, config path를 둔 뒤, config override를 마지막에 두는 것이다.

---

## 9. 실험 결과 해석 체크리스트

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

## 10. 주의할 점

- 작업 트리가 dirty일 수 있으므로, 문서/실험 작업 중 사용자의 변경을 덮어쓰지 않는다.
- LSM의 `InputProjection` docstring은 “Mixed excitatory/inhibitory”가 실제 동작과 맞다. 예전 문서의 “흥분성만” 설명은 더 이상 맞지 않는다.
- `grad_clip_max_norm` 단일 필드는 현재 사용하지 않는다. LSM에서는 `grad_clip_max_norm_w`, `grad_clip_max_norm_theta`를 쓴다.
- `w_raw_max`는 recurrent weight의 상한이다. 초기 recurrent scale은 `w_raw_init_mean/std`가 결정한다.
- `liquid.train_w_raw=false`는 `w_raw`만 freeze한다. `fixed` mode처럼 `beta/threshold`까지 freeze하지 않는다.
- `weight_decay`는 `Config`에는 있으나 `configs/base.yaml`에는 명시되어 있지 않아 기본값 0.0이 적용된다. LSM YAML은 0.0001로 override한다.
- `scripts/train_lsm.py`는 argparse가 아니라 `sys.argv` 기반이다. 첫 인자는 config path, 이후는 `key=value` override다.
- LSM eval은 모델 내부에서 eval 모드일 때 deterministic mask를 사용한다. `LSMModel.forward()`는 `hard` 인자를 받지 않는다.
