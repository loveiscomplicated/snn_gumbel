# LSM 개발 가이드: Gumbel-Softmax 기반 리퀴드 구조 학습

## 0. 문서 목적

이 문서는 LSM 코드를 구현할 때 참조하는 **개발 전용 문서**이다. 연구 동기, 선행 연구, 논문 포지셔닝 등은 `context_2.md`를 참조하고, 이 문서에서는 **구현에 필요한 모든 구체적 결정, 코드 구조, 하이퍼파라미터, 주의사항**만 다룬다.

---

## 1. Feedforward 실험 결과 요약 — LSM 전환의 실험적 동기

### 1.1 실험 결과 일람

| 단계 | 데이터셋 | 구조 | Test Acc | Theta Bimodal | 핵심 발견 |
|------|---------|------|----------|---------------|----------|
| 1단계 | MNIST | 784→512→10 | 98.27% | ✅ 달성 | Receptive field 창발, 임계값 분화, 전문가 뉴런 집단 |
| 2단계 | MNIST 비교 | 학습 38% vs 완전 vs 랜덤 38% | 98.27 > 98.22 > 97.83 | — | A > B > C: 어떤 연결이 얼마나보다 중요 |
| 3단계 | Fashion-MNIST | 784→512→10 | 86.79% | ✅ 달성 | 태스크 일반화 확인, 1024로 늘려도 차이 없음 |
| 4단계 | MNIST→FMNIST 이식 | 토폴로지 고정 | ~51% | — | 이식 실패 → 토폴로지는 태스크 특화 구조 |
| 5단계-a | NMNIST | 2312→512→10 | 97.49% | ✅ (L2 완벽, L1 부분) | 이벤트 데이터 검증, saccade 영역 자동 확장 |
| 5단계-b | DVS Gesture | 2048→256→11 | ~67-70% | ❌ L1 실패 | **FC 스케일링 한계 실증** |

### 1.2 핵심 통찰: FC의 구조적 한계 → LSM이 해결

```
FC 구조의 문제:
  theta 파라미터 수 = O(n_input × n_hidden)
  입력이 커지면 theta가 폭발 → 학습 불가능

  MNIST(784):     784 × 512 = 401K   → bimodal 성공
  NMNIST(2312):  2312 × 512 = 1.18M  → 부분 성공
  DVS(2048):     2048 × 256 = 524K   → 실패 (데이터 1,200개로 부족)

LSM 구조의 해결:
  입력→리퀴드: 랜덤 고정 (theta 없음) → 입력 크기와 무관
  리퀴드 내부: theta = O(N²) → 리퀴드 크기에만 의존

  N=500: 250K theta → MNIST Layer 1(401K)보다 작음
  N=200:  40K theta → 매우 가벼움
```

**뇌와의 일관성:** 감각 수용기→피질 투사는 발달 과정에서 대략 결정, 시냅스 가소성에 의한 구조 변화는 주로 피질 내부에서 발생. LSM의 "입력 고정, 리퀴드 내부 학습"이 이와 대응.

---

## 2. 모델 아키텍처

### 2.1 전체 구조

```
SHD 스파이크 입력 (700채널)
    ↓  랜덤 고정 연결 (p_in=0.1~0.3, randn 양수/음수 혼재)
리퀴드 (N개 LIF 뉴런, 80% 흥분 + 20% 억제, 이질적 beta/threshold)
    ↻  순환 연결: Gumbel-Sigmoid 마스크 × Dale's Law 가중치
    ↓  스파이크 카운트 수집
리드아웃 (선형 분류기, 20클래스)
```

### 2.2 세 가지 연결과 역할

| 연결 | Shape | 학습 여부 | 구현 |
|------|-------|---------|------|
| 입력→리퀴드 | (700, N) | **고정** | randn 초기화(양수/음수 혼재), p_in 확률로 희소 연결 |
| 리퀴드→리퀴드 | (N, N) | **theta + weight 학습** | Gumbel-Sigmoid 마스크 × Dale's Law softplus 가중치 |
| 리퀴드→리드아웃 | (N, 20) | **weight만 학습** | 선형층 (스파이크 카운트 → logits) |

### 2.3 리드아웃 사양 (확정)

**스파이크 카운트 기반 선형 분류기**를 사용한다.

```python
# 리드아웃: membrane 누적 방식
readout_mem = torch.zeros(batch, 20)
for t in range(T):
    readout_mem += readout_weight @ liquid_spike[t]  # (20, N) @ (N,) 
output = readout_mem / T  # 평균 발화율
loss = CrossEntropy(output, label)
```

**이 방식을 확정한 이유:**
- 전통 LSM(Baseline A)과 동일한 리드아웃 → A/B/C 비교에서 리드아웃이 통제 변수
- memoryless (상태 없음) → 시간적 처리는 전적으로 리퀴드가 담당
- 기존 SHD LSM 논문들과 직접 비교 가능

### 2.4 자기 연결(Self-Connection) 처리 (확정)

**대각선(i→i)을 학습에서 제외한다.** 마스크에서 대각선을 0으로 고정.

```python
# 대각선 마스크: 리퀴드 초기화 시 1회 생성
self_conn_mask = 1.0 - torch.eye(N, device=device)  # (N, N), 대각선=0

# forward에서 적용
effective_W = gumbel_mask * self_conn_mask * (dale_sign * F.softplus(w_raw))
```

**근거:**
- 대부분의 LSM 구현에서 autapse(자기 시냅스) 제외
- 자기 연결이 있으면 뉴런이 자기 스파이크로 자기를 흥분시켜 항상 발화 가능 → 학습 불안정
- 향후 ablation으로 "자기 연결 허용 시 어떤 변화가 있는가" 분석 가능

---

## 3. 핵심 구현 디테일

### 3.1 Dale's Law — Softplus 방식

```python
# 초기화 (모델 __init__)
n_exc = int(0.8 * N)
dale_sign = torch.ones(N, 1)       # (N_pre, 1) — 브로드캐스팅
dale_sign[n_exc:, :] = -1.0        # 뒤 20%가 억제성
self.register_buffer('dale_sign', dale_sign)  # 학습 안 함

# w_raw: 자유 파라미터 (부호 제약 없음)
self.w_raw = nn.Parameter(torch.randn(N, N) * 0.01)

# forward에서 유효 가중치 계산
w_eff = self.dale_sign * F.softplus(self.w_raw)  # (N, N)
# dale_sign (N,1) × softplus(w_raw) (N,N) → 브로드캐스팅 → (N,N)
# 흥분성 뉴런(앞 80%)의 행: 양수
# 억제성 뉴런(뒤 20%)의 행: 음수
```

**주의: 행(row) = presynaptic 뉴런.** `dale_sign`을 `(1, N)` 즉 열(column) 방향으로 만들면 Dale's Law가 postsynaptic에 적용되는 오류. 반드시 `(N_pre, 1)` 형태.

### 3.2 Gumbel-Sigmoid 마스크

```python
def gumbel_sigmoid(theta, tau, hard=False):
    """
    theta: (N, N) 학습 가능 파라미터
    tau: temperature (학습 진행에 따라 감소)
    hard: True면 추론 시 이진 마스크
    """
    if hard:
        return (torch.sigmoid(theta) >= 0.5).float()
    
    # Gumbel noise 샘플링
    eps = torch.rand_like(theta).clamp(1e-6, 1 - 1e-6)
    gumbel_noise = torch.log(eps) - torch.log(1 - eps)
    y = torch.sigmoid((theta + gumbel_noise) / tau)
    return y
```

### 3.3 마스크 샘플링 타이밍 — 시뮬레이션 전 1회 (중요)

```python
class LiquidLayer(nn.Module):
    def sample_mask(self, tau):
        """시뮬레이션 시작 전 1회 호출"""
        self.current_mask = gumbel_sigmoid(self.theta, tau)
        return self.current_mask
    
    def forward(self, spike):
        """타임스텝 루프 안에서 호출 — 저장된 마스크 사용"""
        w_eff = self.current_mask * self.self_conn_mask * (self.dale_sign * F.softplus(self.w_raw))
        return w_eff @ spike
```

**기존 feedforward 코드와의 차이:** 기존 `GumbelLIFLayer.forward()`는 호출 시마다 `gumbel_sigmoid` 재호출(매 타임스텝 새 노이즈). LSM에서는 물리적 연결 구조가 시뮬레이션 중 변하지 않으므로 1회 고정.

**배치 처리:** theta는 `(N, N)` shape, 배치 차원 없음 → 배치 전체에 동일 마스크. 배치마다 다른 마스크를 쓰면 theta gradient에 불필요한 variance만 추가.

### 3.4 전체 Forward 수도코드

```python
class LSMModel(nn.Module):
    def forward(self, spikes, tau):
        """
        spikes: (batch, T, 700) — SHD 스파이크 입력
        tau: Gumbel temperature
        """
        batch_size, T, _ = spikes.shape
        
        # 마스크 1회 생성 (타임스텝 루프 바깥)
        recurrent_mask = self.liquid_layer.sample_mask(tau)
        
        # 상태 초기화
        liquid_mem = torch.zeros(batch_size, self.N, device=spikes.device)
        liquid_spike = torch.zeros(batch_size, self.N, device=spikes.device)
        readout_mem = torch.zeros(batch_size, self.n_classes, device=spikes.device)
        
        for t in range(T):
            # 입력 전류 (고정 연결)
            input_current = spikes[:, t] @ self.input_weight  # (batch, 700) @ (700, N) → (batch, N)
            
            # 순환 전류 (학습된 마스크 + Dale's Law 가중치)
            recurrent_current = self.liquid_layer(liquid_spike)  # (batch, N)
            
            # LIF 동역학
            liquid_mem = self.beta * liquid_mem + input_current + recurrent_current
            liquid_spike = surrogate_spike(liquid_mem - self.threshold)
            liquid_mem = liquid_mem * (1.0 - liquid_spike.detach())  # 리셋
            
            # 리드아웃 누적
            readout_mem = readout_mem + liquid_spike @ self.readout_weight.T
        
        output = readout_mem / T
        return output
```

---

## 4. 손실 함수

```python
L = L_CE + lambda_sparse * L_sparse + lambda_commit * L_commit
```

| 항목 | 수식 | 역할 |
|------|------|------|
| L_CE | CrossEntropy(output, label) | 분류 정확도 |
| L_sparse | mean(sigmoid(theta)) | 연결을 스파스하게 유도 (0으로 밀기) |
| L_commit | mean(H(sigmoid(theta))) | 중간값 패널티 → 0 또는 1로 양극화 |

**중요:** L_sparse는 **리퀴드 내부 theta에만** 적용. 입력→리퀴드, 리드아웃에는 적용하지 않음 (theta가 없으므로).

---

## 5. 역전파: BPTT + 마스크 고정

### 5.1 세 가지 독립적 문제와 해결

| 문제 | 해결 |
|------|------|
| 스파이크 함수 미분 불가 | Surrogate gradient |
| 순환 연결 시간 축 역전파 | BPTT |
| 연결 존재 여부 이산성 | Gumbel-Softmax |

### 5.2 Gradient 흐름

```
loss
  ↓ ∂L/∂output
readout_weight  ← gradient (학습)
  ↓ ∂L/∂spike (BPTT: T 타임스텝 역전파)
threshold       ← gradient (학습)
w_raw           ← gradient via (mask * dale_sign * softplus'(w_raw)) (학습)
theta           ← gradient via (∂L/∂mask × ∂mask/∂theta) (학습)
                   mask가 시간 상수이므로 모든 타임스텝의 gradient가 합산
```

### 5.3 Gradient Clipping (중요)

순환 구조에서 BPTT → gradient exploding 위험이 feedforward보다 높음.

```python
# 매 배치 업데이트 후
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**초기값 1.0으로 시작.** 학습 초기에 흥분성 루프 형성 시 gradient 폭발 가능. 로그에 gradient norm을 기록하여 모니터링:

```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if total_norm > 10.0:
    print(f"Warning: grad norm = {total_norm:.1f}")
```

### 5.4 BPTT 메모리 대응

**기본:** dt=10ms binning → T=100 (SHD 표준)

**메모리 부족 시 순서대로:**
1. gradient checkpointing (`torch.utils.checkpoint`)
2. Truncated BPTT (윈도우 200 스텝)
3. N 축소

---

## 6. 데이터 파이프라인: SHD

### 6.1 데이터 사양

| 항목 | 값 |
|------|-----|
| 포맷 | HDF5 |
| 입력 | 700채널 스파이크 트레인 (인공 와우 모델) |
| 클래스 | 20 (영어+독일어 숫자) |
| Train | 8,156 |
| Test | 2,264 |
| 샘플 길이 | ~1초 |

### 6.2 가변 길이 시퀀스 처리 (확정)

**최대 길이 고정 + zero-padding** 방식을 사용한다.

```python
# dt=10ms binning → 1초 = 100 타임스텝
T_MAX = 100

def bin_spikes(spike_times, spike_units, T_max=100, dt=0.01, n_channels=700):
    """
    spike_times: (num_spikes,) — 초 단위
    spike_units: (num_spikes,) — 채널 인덱스
    반환: (T_max, n_channels) — 이진 텐서
    """
    binned = torch.zeros(T_max, n_channels)
    for t, u in zip(spike_times, spike_units):
        time_bin = int(t / dt)
        if time_bin < T_max:
            binned[time_bin, int(u)] = 1.0  # 또는 += 1.0 (카운트)
    return binned
```

**짧은 샘플:** 뒷부분이 zero 프레임 → LIF의 leak으로 자연 감쇠, 문제 없음
**긴 샘플:** T_MAX=100(1초)에서 잘림 → SHD 대부분 1초 이내이므로 손실 미미

### 6.3 데이터 로딩 방법

**Tonic 라이브러리 우선 시도**, 안 되면 직접 HDF5 파싱.

```python
# 방법 1: Tonic
import tonic
dataset = tonic.datasets.SHD(save_to='./data', train=True)

# 방법 2: 직접 HDF5
import h5py
with h5py.File('shd_train.h5', 'r') as f:
    spike_times = f['spikes']['times']   # 리스트 of 배열
    spike_units = f['spikes']['units']   # 리스트 of 배열
    labels = f['labels'][()]             # (8156,)
```

---

## 7. 실험 구도

### 7.1 Baseline 설계

| 조건 | 리퀴드 구조 | 리퀴드 가중치 | 연결 밀도 | 역할 |
|------|-----------|-------------|---------|------|
| **A** | 랜덤 고정 | **고정** | 기존 LSM 기본값 | 전통 LSM |
| **B (p sweep)** | 랜덤 고정 | **학습** | p ∈ {0.1, 0.2, 0.3, 0.5} | 가중치 학습 효과 |
| **B*** | 랜덤 고정 | **학습** | **C와 동일 희소성** | 가장 공정한 비교 |
| **D (Grad R)** | Hard threshold | **학습** | 동적 변화 | 기존 rewiring 대비 |
| **C** | **Gumbel 학습** | **학습** | **자동 결정** | **제안 방법** |

### 7.2 코드 레벨에서의 Baseline 전환

```python
# A: 전통 LSM — 가중치 고정, 구조 고정
model = LSMModel(mode="fixed", weight_trainable=False, p_connect=0.2)

# B: 가중치만 학습
model = LSMModel(mode="random_sparse", weight_trainable=True, p_connect=0.2)

# B*: C와 동일 희소성의 랜덤 구조
learned_sparsity = get_sparsity_from_trained_C()
model = LSMModel(mode="random_sparse", weight_trainable=True, p_connect=learned_sparsity)

# C: 제안 방법
model = LSMModel(mode="learned", weight_trainable=True)

# D: Grad R
model = LSMModel(mode="grad_r", weight_trainable=True)
# 내부: gumbel_sigmoid 대신 (theta > 0).float()
```

---

## 8. 하이퍼파라미터

### 8.1 확정 사항

| 항목 | 값 | 근거 |
|------|-----|------|
| 프레임워크 | 순수 PyTorch | 기존 코드 재활용, 마스크 커스텀 로직 제어 |
| 입력 | SHD 700채널 | 표준 벤치마크 |
| dt | 10ms | T=100, SHD 표준 |
| E:I 비율 | 80:20 | 뇌 피질 비율 반영, LSM 표준 |
| 입력 연결 | **randn(양수/음수 혼재)**, 랜덤 고정 | separation property 확보에 필수 (아래 주의사항 참조) |
| 리드아웃 | 선형 (스파이크 카운트 평균) | A/B/C 통제 |
| 자기 연결 | 제외 (대각선 0) | 학습 안정성 |
| 마스크 타이밍 | 시뮬레이션 전 1회, T 동안 고정 | PGExplainer 설계 |
| 마스크 배치 | 배치 전체 동일 | variance 최소화 |
| Gradient clipping | max_norm=1.0 | 순환 BPTT 안정화 |
| Dale's Law | Softplus (abs 폐기) | gradient 안정성 |
| beta 초기화 | 뉴런별 이질적 (`linspace(beta_min, beta_max, N)`) | 시간 스케일 다양성 확보 |
| threshold 초기화 | 뉴런별 이질적 (`linspace(thr_min, thr_max, N)`) | 발화 민감도 다양성 확보 |
| 막전위 clamp | `clamp(-3.0, 3.0)` | 흥분성 루프 폭주 방지 안전장치 |

### 8.1.1 입력 연결 사양 — 주의사항 (Phase 1에서 확인됨)

**`torch.rand`(양수만)가 아닌 `torch.randn`(양수/음수 혼재)을 사용해야 한다.**

전통 LSM 문헌에서는 입력→리퀴드 연결이 흥분성만(양수만)인 것이 표준이다. 그러나 이는 리퀴드 내부의 억제성 순환 연결이 충분히 강해서 입력의 균일한 흥분을 분산시키는 동역학이 작동할 때의 이야기이다.

현재 구현에서는 순환 연결이 약하기 때문에(`softplus(-4.0) ≈ 0.018`), 입력이 전부 양수면 모든 뉴런이 입력 활성도에 비례하여 같은 방향으로 반응한다. 결과적으로 서로 다른 클래스의 스파이크 벡터 간 코사인 유사도가 0.999에 수렴하여 리퀴드가 입력을 구분하지 못한다.

`randn`으로 바꾸면 각 뉴런이 어떤 입력 채널에는 흥분하고 다른 채널에는 억제되는 **랜덤 프로젝션** 뉴런이 되어, LSM의 separation property가 확보된다.

### 8.1.2 beta 변환 — 주의사항 (Phase 1에서 발견된 버그)

beta를 (0, 1) 구간으로 제약하기 위해 `sigmoid`를 사용할 때, **`log`가 아닌 `logit`을 저장해야 한다.**

```python
# 잘못된 코드: sigmoid(log(0.9)) = sigmoid(-0.105) = 0.47
self.log_beta = nn.Parameter(torch.tensor(beta).log())

# 올바른 코드: sigmoid(logit(0.9)) = sigmoid(2.197) = 0.9
init_logit = torch.log(torch.tensor(beta) / (1.0 - torch.tensor(beta)))
self.logit_beta = nn.Parameter(init_logit)
```

### 8.2 미결정 — Phase 1에서 탐색

| 항목 | 탐색 범위 | 우선 시도 | 결정 시점 |
|------|---------|---------|---------|
| 뉴런 수 N | 200, 300, 500 | 200 | wall-clock 측정 후 |
| 입력 연결 확률 p_in | 0.1, 0.2, 0.3 | 0.1 | Phase 1 |
| theta 초기화 σ | 0.01, 0.1, 0.5 | 0.01 | Phase 1 |
| Gumbel τ 초기값 | 1.0 | 1.0 | FF 실험과 동일하게 시작 |
| Gumbel τ 최종값 | 0.05 | 0.05 | FF 실험과 동일하게 시작 |
| τ annealing 에폭 | 25, 40, 60 | 40 | Phase 1 |
| learning rate | 1e-3, 5e-4, 1e-4 | 1e-3 | Phase 1 |
| batch size | 32, 64, 128 | 64 | Phase 1 |
| lambda_sparse | 0.005, 0.01, 0.02 | 0.01 | Phase 1 |
| lambda_commit | 0.05, 0.08, 0.1 | 0.08 | Phase 1 |
| beta_min / beta_max | (0.7, 0.95), (0.85, 0.95), (0.7, 0.9) | 0.7 / 0.95 | Phase 3 |
| threshold_min / threshold_max | (0.8, 1.5), (0.8, 2.0), (1.0, 1.5) | 0.8 / 1.5 | Phase 3 |
| threshold 학습 | 리퀴드만 학습, 리드아웃 고정 | — | FF 실험 교훈 |

---

## 9. 구현 Phase 계획

### Phase 1: 핵심 검증 — Go/No-Go (1~2주)

#### Step 1: 기반 구축

```
[ ] LSM 모델 클래스 작성
    - LiquidLayer: theta(N,N) + w_raw(N,N) + dale_sign buffer + self_conn_mask
    - sample_mask() 메서드 (루프 바깥 1회)
    - forward(): 저장된 마스크 사용
    - mode 전환: learned / random_sparse / fixed / grad_r
[ ] SHD 데이터 로딩
    - Tonic 또는 직접 HDF5
    - bin_spikes 함수 (dt=10ms, T=100)
    - DataLoader + collate_fn (zero-padding)
[ ] 학습 루프
    - BPTT (PyTorch autograd 자동 처리)
    - gradient clipping (max_norm=1.0)
    - gradient norm 로깅
    - temperature annealing
    - sparsity / commitment loss
```

#### Step 1.5: Wall-Clock 측정 (첫날 수행)

```
N=200, batch_size=64, T=100으로 1에폭 시간 측정

판단 기준:
  < 1분   → 문제 없음, N=500도 시도 가능
  5~10분  → 감당 가능, N=200~300으로 진행
  30분+   → 즉시 대응: N 축소 / truncated BPTT / dt 증가

추가 측정: GPU 메모리 사용량
  nvidia-smi로 peak memory 확인
  N=200 / N=300 / N=500 각각 측정
```

#### Step 2: Baseline B 먼저 학습

```
[ ] mode="random_sparse", p=0.2, weight_trainable=True
[ ] BPTT + surrogate gradient가 순환 구조에서 작동하는지 확인
[ ] SHD에서 합리적 정확도 확인 (기존 LSM 논문 참조)
[ ] gradient norm이 안정적인지 확인 (폭발/소실 없는지)
```

#### Step 3: 제안 방법 C 학습

```
[ ] mode="learned"로 전환 (나머지 동일)
[ ] temperature annealing + commitment loss 적용
[ ] theta 분포 모니터링: bimodal로 수렴하는지
[ ] 흥분성 루프 폭주 여부 확인 (뉴런 발화율 모니터링)
```

#### Step 4: Go/No-Go 판단

```
B vs C 비교:
  ✅ 유의미한 차이 → Phase 2 진행
  ❌ 차이 없음 → 원인 분석:
     - theta 초기화 σ 변경
     - 온도 스케줄 변경
     - lambda_sparse / lambda_commit 조정
     - N 변경
     - 그래도 안 되면: negative result 논문 가능성 검토
```

### Phase 2: Baseline 강화 (1~2주)

```
[ ] A (전통 LSM): weight_trainable=False
[ ] B* (동일 희소성 랜덤): C의 최종 희소성으로 p 설정
[ ] D (Grad R): (theta > 0).float() 교체
[ ] B p sweep: p ∈ {0.1, 0.2, 0.3, 0.5}
[ ] 비교 테이블 작성: A, B*, D, C 정확도 + 희소성
```

### Phase 3: 분석 및 확장 (2~3주)

```
[ ] 학습된 구조 분석
    - 희소성 비율
    - E/I 연결 비율 (80:20에서 변화?)
    - 허브 뉴런 존재 여부 (in-degree/out-degree 분포)
    - 루프 길이 분포
    - Small-world 특성 (clustering coefficient, path length)
    - LSNN+DEEP R과 비교 (리드아웃 근처 밀집 여부)
    - E/I 균형 자동 발견 여부 (흥분성 루프 주변 억제성 배치)
[ ] SSC 확장 (35클래스, 75K+ 데이터)
[ ] Ablation
    - 초기화 σ (0.01, 0.1, 0.5)
    - Dale's Law 유무
    - 뉴런 수 (200, 300, 500)
    - 온도 스케줄 변화
    - 자기 연결 허용 vs 제외
```

---

## 10. 모니터링 항목

매 에폭 로깅해야 할 항목:

```python
log = {
    "epoch": epoch,
    "lr": current_lr,
    "tau": current_tau,
    "train_loss": train_loss,
    "train_acc": train_acc,
    "test_acc": test_acc,
    
    # 토폴로지 학습 상태
    "sparsity_recurrent": (sigmoid(theta) >= 0.5).float().mean().item(),
    "theta_mean": theta.mean().item(),
    "theta_std": theta.std().item(),
    
    # 학습 안정성
    "grad_norm": total_grad_norm,
    "max_firing_rate": liquid_spike.mean(dim=0).max().item(),  # 폭주 감지
    "mean_firing_rate": liquid_spike.mean().item(),
    
    # E/I 분석 (Phase 3에서 추가)
    "exc_connection_ratio": ...,
    "inh_connection_ratio": ...,
}
```

**조기 경보 조건:**
- `grad_norm > 100`: gradient 폭발 → max_norm 축소 또는 lr 감소
- `max_firing_rate > 0.9`: 특정 뉴런 과발화 → 흥분성 루프 폭주 가능
- `theta_std < 0.01` (epoch 20 이후): theta 정체 → commitment loss 가중치 증가

**필수 진단 (학습 시작 전 1회):**
- 클래스 간 스파이크 벡터 코사인 유사도 측정 (separation property 검증)
- 0.999 이상이면 리퀴드가 입력을 구분하지 못하는 상태 → 입력 연결 사양 점검
- 0.95~0.99면 정상, 학습으로 개선 가능

---

## 11. 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| SHD SOTA 96%+, 성능 개선 여지 좁음 | 효율성(적은 연결로 같은 성능) 기여 병행 |
| B→C 점프 미미 | 넓은 프레이밍(A→B도 기여), 구조 분석으로 가치 보완 |
| BPTT 메모리 부담 (N×T) | dt=10ms(T=100), checkpointing, truncated BPTT |
| N² theta 메모리/계산량 | N 축소(200), 공간 제약 |
| 학습된 구조가 랜덤과 유사 | 네트워크 특성 분석(degree 분포, clustering 등)으로 차이 시각화 |
| Gumbel + Surrogate 학습 불안정 | 온도 스케줄, lr 조절, gradient clipping |
| 흥분성 루프 막전위 폭주 | 고온 초기화(자연 감쇠), firing rate clamp, 막전위 clamp |
| Grad R과 차별화 질문 | 연속 최적화 vs hard threshold, temperature annealing, commitment loss |
| Phase 1 go/no-go 실패 | HP 재탐색 → negative result 논문 가능성 |
| 순환 구조 GPU 비효율 | Phase 1 첫날 wall-clock 측정으로 N 범위 사전 확정 |
| **SHD 8,156개로 N² theta 학습 충분한가** | N=200이면 40K theta, 204 샘플/파라미터 → 안전. N=500이면 250K, 32.6 샘플/파라미터 → Phase 1에서 확인. DVS Gesture 실패(2.3 샘플/파라미터)의 교훈 적용 |

---

## 12. 기존 코드 재활용 맵

| 기존 코드 | LSM에서 재활용 | 수정 필요 여부 |
|----------|-------------|-------------|
| `gumbel_sigmoid()` | 리퀴드 마스크 생성 | 그대로 |
| `SurrogateSpike` | LIF 스파이크 함수 | 그대로 |
| `sparsity_loss()` | 리퀴드 theta 정규화 | 리퀴드 theta에만 적용하도록 범위 조정 |
| `commitment_loss()` | theta 양극화 유도 | 그대로 |
| topology mode 전환 | A/B/B*/C/D 전환 | mode 목록에 "fixed", "grad_r" 추가 |
| topology transfer | (향후 사용 가능) | 그대로 |
| temperature annealing | tau 스케줄 | 그대로 |

**새로 구현해야 할 것:**
- `LiquidLayer` 클래스 (순환 연결 + Dale's Law + self_conn_mask + sample_mask)
- `LSMModel` 클래스 (입력/리퀴드/리드아웃 3분리 + BPTT 루프)
- SHD 데이터 로딩 + binning
- gradient clipping 통합
- 입력→리퀴드 고정 연결 초기화 (p_in, 흥분성만)
- Baseline A 모드 (가중치까지 고정)

---

## 13. 파일 구조 (권장)

```
lsm/
├── model.py          # LSMModel, LiquidLayer, InputProjection
├── layers.py         # 기존 GumbelLIFLayer에서 가져온 유틸 (gumbel_sigmoid, surrogate 등)
├── data.py           # SHD 로딩, binning, DataLoader
├── train.py          # 학습 루프, gradient clipping, 로깅
├── evaluate.py       # 테스트 평가, 구조 분석
├── config.yaml       # 하이퍼파라미터
├── analysis/
│   ├── topology.py   # 희소성, degree 분포, 루프 분석
│   ├── ei_balance.py # E/I 비율, 흥분성 루프 분석
│   └── visualize.py  # theta 분포, 연결 행렬 시각화
└── baselines/
    ├── traditional.py  # Baseline A (전통 LSM)
    └── grad_r.py       # Baseline D (Grad R)
```
