# LSM 구현 계획서

## 0. 개요

기존 feedforward SNN Gumbel-Softmax 코드베이스 위에 **LSM(Liquid State Machine) 모듈**을 추가 구현한다. 기존 코드는 그대로 유지하고, `src/lsm/` 하위에 LSM 전용 모듈을 새로 만든다.

**목표:** SHD(Spiking Heidelberg Digits) 데이터셋에서 Gumbel-Sigmoid 기반 리퀴드 구조 학습의 유효성을 검증한다.

---

## 1. 파일 구조

```
snn_gumbel/
├── src/
│   ├── models/layers.py          # [기존] gumbel_sigmoid, SurrogateSpike — 공유
│   ├── models/snn.py             # [기존] Feedforward SNN — 수정 없음
│   ├── training/losses.py        # [기존] CE + sparsity + commitment — 수정 없음
│   ├── training/trainer.py       # [기존] FF 학습 루프 — 수정 없음
│   ├── utils/config.py           # [수정] LSM 관련 Config 필드 추가
│   ├── data/loaders.py           # [수정] SHD 데이터셋 추가
│   └── lsm/                      # [신규] LSM 전용 모듈
│       ├── __init__.py
│       ├── model.py              # LSMModel, LiquidLayer, InputProjection
│       └── trainer.py            # LSM 전용 학습 루프 (gradient clipping, 모니터링)
├── configs/
│   └── lsm_shd_baseline.yaml    # [신규] LSM 실험 설정
└── scripts/
    └── train_lsm.py             # [신규] LSM 학습 CLI 진입점
```

**설계 원칙:**
- 기존 FF 코드를 깨뜨리지 않는다
- `src/models/layers.py`의 `gumbel_sigmoid`, `SurrogateSpike`, `spike_fn`을 공유한다
- `src/training/losses.py`의 `ce_loss`를 공유한다
- LSM 전용 로직(순환 루프, Dale's Law, gradient clipping)은 `src/lsm/`에 격리한다

---

## 2. 구현 단계

### Step 1: Config 확장 + SHD 데이터 로더

**목표:** LSM 실험을 위한 설정 체계와 데이터 파이프라인 준비

#### 1-1. Config 확장 (`src/utils/config.py`)

`LiquidConfig` 데이터클래스를 추가하고, `Config`에 `liquid` 필드를 넣는다.

```python
@dataclass
class LiquidConfig:
    n_liquid: int = 200            # 리퀴드 뉴런 수
    exc_ratio: float = 0.8         # 흥분성 뉴런 비율 (80%)
    p_input: float = 0.1           # 입력→리퀴드 연결 확률
    recurrent_mode: str = "learned"  # learned | random_sparse | fixed | grad_r
    recurrent_sparsity: float = 0.2  # random_sparse 모드 시 연결 확률
    self_connection: bool = False    # 자기 연결 허용 여부
    theta_init_std: float = 0.01    # theta 초기화 표준편차
    grad_clip_max_norm: float = 1.0  # gradient clipping
    input_weight_scale: float = 0.1  # 입력 가중치 스케일
```

`Config`에 추가:
```python
liquid: LiquidConfig = field(default_factory=LiquidConfig)
```

`load_config`의 `_dict_to_config`에서 `liquid` 섹션도 파싱하도록 수정한다. 기존 FF 실험에서는 이 필드가 YAML에 없으므로 기본값이 사용되어 영향 없음.

#### 1-2. SHD 데이터 로더 (`src/data/loaders.py`)

`get_dataloaders`에 `"shd"` 분기를 추가한다.

```python
elif dataset == "shd":
    train_ds = _make_shd(cfg.data_dir, train=True, T=cfg.T, dt=0.01)
    test_ds  = _make_shd(cfg.data_dir, train=False, T=cfg.T, dt=0.01)
```

SHD 로딩 방법 (우선순위):
1. **Tonic 라이브러리** — `tonic.datasets.SHD` 존재 시 사용
2. **직접 HDF5 파싱** — h5py로 spike_times, spike_units, labels 읽기

binning 함수:
```python
def bin_spikes(spike_times, spike_units, T_max=100, dt=0.01, n_channels=700):
    """스파이크 이벤트를 (T_max, n_channels) 이진 텐서로 변환"""
    binned = torch.zeros(T_max, n_channels)
    time_bins = (spike_times / dt).long().clamp(max=T_max - 1)
    binned[time_bins, spike_units.long()] = 1.0
    return binned
```

**검증:** `T=100, dt=0.01(10ms), n_channels=700, n_classes=20`으로 DataLoader가 `(batch, 100, 700)` 텐서를 반환하는지 확인.

---

### Step 2: LSM 모델 구현 (`src/lsm/model.py`)

**목표:** LiquidLayer + InputProjection + LSMModel 구현

#### 2-1. InputProjection

입력→리퀴드 고정 랜덤 연결. **학습하지 않는다.**

```python
class InputProjection(nn.Module):
    """고정 랜덤 희소 연결 (입력 → 리퀴드). 흥분성만."""
    def __init__(self, n_input, n_liquid, p_input=0.1, weight_scale=0.1):
        # 연결 마스크: Bernoulli(p_input)
        mask = (torch.rand(n_input, n_liquid) < p_input).float()
        # 가중치: 양수만 (흥분성)
        weight = torch.rand(n_input, n_liquid) * weight_scale
        weight = weight * mask
        self.register_buffer('input_weight', weight)  # 학습 안 함
```

**핵심:** `register_buffer`로 등록하여 `requires_grad=False`. 디바이스 이동/체크포인트 저장에 자동 포함.

#### 2-2. LiquidLayer

리퀴드 내부 순환 연결. Gumbel-Sigmoid 마스크 + Dale's Law + self_conn_mask.

**파라미터:**
| 이름 | Shape | 학습 | 역할 |
|------|-------|------|------|
| `theta` | (N, N) | Yes (learned 모드) | 연결 존재 확률 logit |
| `w_raw` | (N, N) | Yes | 시냅스 크기 (softplus 전) |
| `threshold` | (N,) | Yes | LIF 임계값 |
| `log_beta` | scalar | Yes | 막전위 감쇠율 |

**버퍼 (학습 안 함):**
| 이름 | Shape | 역할 |
|------|-------|------|
| `dale_sign` | (N, 1) | 흥분(+1)/억제(-1) 부호 |
| `self_conn_mask` | (N, N) | 대각선=0, 나머지=1 |
| `fixed_mask` | (N, N) | random_sparse 모드용 고정 마스크 |

**메서드:**
- `sample_mask(tau)` — 시뮬레이션 전 1회 호출. `self.current_mask`에 저장.
- `forward(spike)` — 저장된 마스크로 순환 전류 계산. 타임스텝 루프 안에서 호출.
- `get_effective_weight()` — `current_mask * self_conn_mask * (dale_sign * softplus(w_raw))`
- `sparsity()` — 현재 연결 비율 반환

**모드별 동작:**
```
learned      → sample_mask에서 gumbel_sigmoid(theta, tau) 사용
random_sparse → sample_mask에서 fixed_mask 반환
fixed        → sample_mask에서 fixed_mask 반환, w_raw도 학습 안 함
grad_r       → sample_mask에서 (theta > 0).float() 사용
```

**Dale's Law 구현:**
```python
n_exc = int(exc_ratio * n_liquid)
dale_sign = torch.ones(n_liquid, 1)
dale_sign[n_exc:] = -1.0
self.register_buffer('dale_sign', dale_sign)

# forward에서:
w_eff = self.current_mask * self.self_conn_mask * (self.dale_sign * F.softplus(self.w_raw))
```

#### 2-3. LSMModel

세 컴포넌트를 조합한 최종 모델.

```python
class LSMModel(nn.Module):
    def __init__(self, n_input, n_liquid, n_output, T, liquid_cfg):
        self.input_proj = InputProjection(n_input, n_liquid, ...)
        self.liquid = LiquidLayer(n_liquid, ...)
        self.readout = nn.Linear(n_liquid, n_output)
    
    def forward(self, spikes, tau):
        """
        spikes: (batch, T, n_input) — SHD 스파이크 입력
        반환: (batch, n_output) — 클래스별 평균 발화율
        """
        # 1. 마스크 1회 생성
        self.liquid.sample_mask(tau)
        
        # 2. 상태 초기화
        liquid_mem = zeros(batch, N)
        liquid_spike = zeros(batch, N)
        readout_mem = zeros(batch, n_output)
        
        # 3. 타임스텝 루프
        for t in range(T):
            input_current = spikes[:, t] @ self.input_proj.input_weight
            recurrent_current = liquid_spike @ self.liquid.get_effective_weight().T
            
            liquid_mem = beta * liquid_mem + input_current + recurrent_current
            liquid_spike = spike_fn(liquid_mem - threshold)
            liquid_mem = liquid_mem * (1.0 - liquid_spike)
            
            readout_mem += self.readout(liquid_spike)
        
        # 4. 평균 발화율 출력
        return readout_mem / T
```

**Loss 메서드 (LSMModel 내장):**
- `sparsity_loss()` → `mean(sigmoid(self.liquid.theta))` — 리퀴드 theta에만 적용
- `commitment_loss()` → `mean(H(sigmoid(self.liquid.theta)))` — 리퀴드 theta에만 적용

---

### Step 3: LSM 학습 루프 (`src/lsm/trainer.py`)

**목표:** 기존 trainer.py를 기반으로 LSM 전용 학습 루프 작성

**기존 `src/training/trainer.py`와의 차이점:**

| 항목 | 기존 FF | LSM |
|------|--------|-----|
| 모델 | `SNNModel` | `LSMModel` |
| Gradient clipping | 없음 | `clip_grad_norm_(max_norm=1.0)` |
| 모니터링 | sparsity만 | sparsity + firing rate + grad norm |
| 입력 형태 | `(batch, 784)` 또는 `(batch, T, C)` | `(batch, T, 700)` 항상 |
| 조기 경보 | 없음 | grad_norm > 100, max_firing_rate > 0.9 |

**추가 로깅 항목:**
```python
row = {
    "epoch": epoch,
    "tau": tau,
    "train_loss": train_loss,
    "train_acc": train_acc,
    "test_acc": test_acc,
    "sparsity_recurrent": sparsity,
    "theta_mean": theta.mean().item(),
    "theta_std": theta.std().item(),
    "grad_norm": total_grad_norm,
    "mean_firing_rate": mean_fr,
    "max_firing_rate": max_fr,
}
```

**Gradient clipping 위치:**
```python
loss.backward()
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.liquid.grad_clip_max_norm)
optimizer.step()
```

**`_evaluate` 함수:** 기존과 동일 구조, `model(x, tau=tau)` 호출 시 `hard=True` 대신 tau를 매우 낮은 값으로 설정하거나, LSMModel 내부에서 eval 모드 시 hard mask를 사용하도록 처리.

---

### Step 4: Config YAML + CLI 진입점

#### 4-1. `configs/lsm_shd_baseline.yaml`

```yaml
base: base.yaml

experiment_name: lsm_shd
dataset: shd

n_input: 700
n_output: 20
T: 100
beta: 0.9

architecture:
  hidden_layers: []    # LSM에서는 사용 안 함

topology:
  mode: learned

liquid:
  n_liquid: 200
  exc_ratio: 0.8
  p_input: 0.1
  recurrent_mode: learned
  recurrent_sparsity: 0.2
  self_connection: false
  theta_init_std: 0.01
  grad_clip_max_norm: 1.0
  input_weight_scale: 0.1

tau_start: 1.0
tau_end: 0.05
tau_anneal_epochs: 40

epochs: 100
patience: 20
batch_size: 64
lr: 0.001
lr_min: 0.00001
lambda_sparse: 0.01
lambda_commit: 0.08
seed: 42
```

#### 4-2. `scripts/train_lsm.py`

```python
"""LSM 학습 CLI 진입점"""
import sys
from src.utils.config import load_config
from src.lsm.trainer import train

if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/lsm_shd_baseline.yaml"
    overrides = sys.argv[2:] if len(sys.argv) > 2 else []
    cfg = load_config(config_path, overrides)
    train(cfg)
```

**사용법:**
```bash
# 기본 실행
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml

# HP 오버라이드
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml liquid.n_liquid=300 lr=0.0005
```

---

### Step 5: Wall-Clock 측정 + Baseline B 검증

**목표:** 실행 가능성 확인, 순환 BPTT 안정성 검증

#### 5-1. Wall-Clock 측정 (Step 2~4 완료 후 즉시)

```bash
# N=200 1에폭 시간 측정
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml epochs=1 liquid.n_liquid=200

# N=300
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml epochs=1 liquid.n_liquid=300

# N=500
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml epochs=1 liquid.n_liquid=500
```

**판단 기준 (1에폭 기준):**
| 시간 | 판단 | 조치 |
|------|------|------|
| < 1분 | 문제 없음 | N=500도 시도 가능 |
| 5~10분 | 감당 가능 | N=200~300으로 진행 |
| 30분+ | 즉시 대응 | N 축소 / truncated BPTT / dt 증가 |

#### 5-2. Baseline B 학습

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
    liquid.recurrent_mode=random_sparse \
    liquid.recurrent_sparsity=0.2
```

**확인 사항:**
- [ ] BPTT + surrogate gradient가 순환 구조에서 작동하는가
- [ ] SHD에서 합리적 정확도 나오는가 (기존 LSM 논문: ~70-85%)
- [ ] gradient norm이 안정적인가 (폭발/소실 없는지)
- [ ] 뉴런 발화율이 합리적인가 (0.01~0.5 범위)

---

### Step 6: 제안 방법 C 학습 + Go/No-Go

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
    liquid.recurrent_mode=learned
```

**확인 사항:**
- [ ] theta 분포가 bimodal로 수렴하는가
- [ ] 흥분성 루프 폭주 없는가 (max_firing_rate < 0.9)
- [ ] B 대비 유의미한 정확도 차이가 있는가

**Go/No-Go 판단:**
```
B vs C 비교:
  정확도 차이 > 1~2%  → Go (Phase 2 진행)
  정확도 차이 < 1%    → HP 재탐색:
     - theta_init_std: 0.01 → 0.1 → 0.5
     - tau_anneal_epochs: 25 → 40 → 60
     - lambda_sparse: 0.005 → 0.01 → 0.02
     - lambda_commit: 0.05 → 0.08 → 0.1
     - n_liquid: 200 → 300
  재탐색 후에도 차이 없음 → negative result 분석
```

---

## 3. 구현 순서 요약

```
Step 1: Config 확장 + SHD 데이터 로더
   ├─ 1-1. LiquidConfig 추가, load_config 수정
   └─ 1-2. SHD 로더 추가, bin_spikes 구현, DataLoader 검증
          ↓
Step 2: LSM 모델 구현
   ├─ 2-1. InputProjection (고정 랜덤 입력 연결)
   ├─ 2-2. LiquidLayer (Gumbel + Dale's Law + 순환)
   └─ 2-3. LSMModel (3분리 조합 + forward + loss)
          ↓
Step 3: LSM 학습 루프
   └─ gradient clipping + 확장 로깅 + 조기 경보
          ↓
Step 4: Config YAML + CLI 진입점
   ├─ 4-1. lsm_shd_baseline.yaml
   └─ 4-2. scripts/train_lsm.py
          ↓
Step 5: Wall-Clock 측정 + Baseline B 검증
   ├─ 5-1. N=200/300/500 시간 측정
   └─ 5-2. random_sparse 모드로 BPTT 안정성 검증
          ↓
Step 6: 제안 방법 C + Go/No-Go
   └─ learned 모드, B와 비교, HP 튜닝
```

---

## 4. 주의사항 체크리스트

### 모델 구현
- [ ] `dale_sign` shape이 `(N, 1)`인지 확인 — `(1, N)`이면 Dale's Law가 postsynaptic에 잘못 적용됨
- [ ] `self_conn_mask`의 대각선이 0인지 확인
- [ ] 마스크가 타임스텝 루프 **바깥**에서 1회만 생성되는지 확인
- [ ] `w_raw` 초기화가 `randn * 0.01`인지 확인 (너무 크면 초기 폭주)
- [ ] `softplus(w_raw)`이 `abs(w_raw)`가 아닌지 확인 (gradient 안정성)
- [ ] 리드아웃 threshold는 학습하지 않는지 확인

### 학습 안정성
- [ ] `clip_grad_norm_`이 `loss.backward()` 뒤, `optimizer.step()` 앞에 위치하는지 확인
- [ ] grad_norm > 100일 때 경고 출력하는지 확인
- [ ] max_firing_rate > 0.9일 때 경고 출력하는지 확인
- [ ] NaN 감지 로직이 있는지 확인

### 데이터
- [ ] SHD 출력 shape이 `(batch, 100, 700)`인지 확인
- [ ] 레이블이 0~19 범위인지 확인
- [ ] dt=10ms, T=100으로 약 1초를 커버하는지 확인

### 기존 코드 호환
- [ ] `src/models/layers.py`를 수정하지 않았는지 확인 (import만 사용)
- [ ] `src/training/trainer.py`를 수정하지 않았는지 확인
- [ ] `src/training/losses.py`를 수정하지 않았는지 확인
- [ ] 기존 FF 실험이 여전히 동작하는지 확인 (`python scripts/train.py configs/mnist_baseline.yaml epochs=1`)
