# Learnable SNN Topology via Gumbel-Softmax — 프로젝트 전체 정리

> **Update 2026-05-09 — same-density random controls closed**  
> Density-matched `random_sparse` controls around the learned-lowrank regime are now complete: `p ∈ {0.040, 0.045, 0.050}` across seeds `42/43/44/45`, with `train_w_raw=false`, `w_raw_init_mean=-2.25`, `w_raw_max=-2.0`, `val_fraction=0.1`, and `val_seed=42`. The controls reached only test@best-val mean `0.5257` with best single run `0.5406`, below the no-recurrence baseline `0.5490` and far below `learned_lowrank + validation rollback m50p10` mean `0.5919` / worst `0.5826`. Density-only explanation is now rejected; next phase is topology diagnostics, graph-structure analysis, and paper-claim/table cleanup.

> **Update 2026-05-08 — adaptive freeze policy closed**  
> Validation split and `learned_lowrank` topology snapshot/rollback are implemented. The validation-rollback policy search is now closed: `m50p10` is the main proposed policy, `m60p10` is redundant because all `m50p10` freeze events already occurred after epoch 60, and `m60p15` is rejected because it lowers mean and worst-seed generalization. At that point, the next phase was same-density random controls, topology diagnostics, and paper-claim/table cleanup; the density-control part is now closed by the 2026-05-09 update.


## Adaptive Freeze Policy Closure — 2026-05-08

This section closes the validation-based adaptive topology freeze search for `learned_lowrank`.

### Final policy decision

| Policy | Config | Result | Decision |
|---|---|---|---|
| `m50p10` | `topology_freeze_min_epoch=50`, `topology_freeze_patience=10` | test@best-val mean `0.5919`, median `0.5857`, worst `0.5826` | **main proposed validation-rollback result** |
| `m60p10` | `topology_freeze_min_epoch=60`, `topology_freeze_patience=10` | not effectively different from `m50p10`; all `m50p10` freeze epochs were already after epoch 60 | **redundant / not distinct** |
| `m60p15` | `topology_freeze_min_epoch=60`, `topology_freeze_patience=15` | test@best-val mean `0.5834`, median `0.5808`, worst `0.5574` | **rejected** |

### m60p15 completed result

| Seed | Topology rollback epoch | Freeze epoch | Best val epoch | Best val | Test @ best val | Comparison vs m50p10 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 78 | 93 | 78 | 0.6238 | 0.5852 | -0.0005 |
| 43 | 54 | 69 | 90 | 0.6654 | 0.6144 | +0.0009 |
| 44 | 65 | 80 | 97 | 0.6299 | 0.5764 | -0.0093 |
| 45 | 77 | 92 | 96 | 0.6164 | 0.5574 | -0.0252 |

Aggregate comparison:

| Policy | Mean | Median | Worst | Best |
|---|---:|---:|---:|---:|
| `m50p10` | **0.5919** | **0.5857** | **0.5826** | 0.6135 |
| `m60p15` | 0.5834 | 0.5808 | 0.5574 | **0.6144** |

Interpretation:

- Increasing patience from 10 to 15 does not reliably improve topology quality.
- The small seed43 gain is not worth the seed44/seed45 degradation.
- Seed45 is decisive: `m60p15` drops from `0.5826` to `0.5574`, indicating late topology drift or validation over-selection.
- The adaptive-freeze policy search should stop here to avoid post-hoc policy mining.

Final claim:

> `learned_lowrank + validation rollback m50p10` is the fair main-table result. It is not the absolute peak-search upper bound, but it is the strongest test-leakage-free adaptive policy currently validated across seeds 42/43/44/45. Further gains should come from density controls, topology diagnostics, or a new pre-registered protocol, not additional manual policy search.

### Next phase after closure

| Status / Priority | Work item | Purpose |
|---:|---|---|
| **Closed** | same-density random controls around learned_lowrank densities | block density-only explanation; completed at test@best-val mean `0.5257`, best single `0.5406` |
| 1 | topology diagnostics for `learned`, `grad_r`, and `learned_lowrank` | explain why latent role topology changes seed behavior |
| 2 | main table cleanup using `test @ best val` | avoid test leakage and post-hoc peak selection |
| 3 | graph-structure analysis: E/I degree, loop count/length, clustering, path length | support mechanism-level claims |
| 4 | prediction auxiliary / e-prop / predictive coding | defer until topology-selection claims are defended |


## Same-Density Random Control Closure — 2026-05-09

This section closes the density-matched `random_sparse` control batch for the `learned_lowrank` regime.

Common setting:

```text
recurrent_mode=random_sparse
recurrent_sparsity ∈ {0.040, 0.045, 0.050}
seed ∈ {42, 43, 44, 45}
train_w_raw=false
w_raw_init_mean=-2.25
w_raw_max=-2.0
val_fraction=0.1
val_seed=42
train / val / test = 7340 / 816 / 2264
selection metric = validation-best checkpoint
```

### Individual density-control results

| Density `p` | Seed | Best val_acc | Best epoch | Test acc @ best val | Run name |
|---:|---:|---:|---:|---:|---|
| 0.040 | 42 | 0.5956 | 72 | 0.5190 | `p040_s42` |
| 0.040 | 43 | 0.5895 | 93 | 0.5327 | `p040_s43` |
| 0.040 | 44 | 0.5968 | 87 | 0.5336 | `p040_s44` |
| 0.040 | 45 | 0.5564 | 75 | 0.5340 | `p040_s45` |
| 0.045 | 42 | 0.5846 | 83 | 0.5230 | `p045_s42` |
| 0.045 | 43 | 0.5821 | 70 | 0.5243 | `p045_s43` |
| 0.045 | 44 | **0.6066** | 76 | 0.5265 | `p045_s44` |
| 0.045 | 45 | 0.5686 | 82 | 0.5283 | `p045_s45` |
| 0.050 | 42 | 0.5895 | 83 | 0.5159 | `p050_s42` |
| 0.050 | 43 | 0.5895 | 99 | **0.5406** | `p050_s43` |
| 0.050 | 44 | 0.5870 | 68 | 0.5177 | `p050_s44` |
| 0.050 | 45 | 0.5662 | 81 | 0.5133 | `p050_s45` |

### Density-level aggregation

| Density `p` | Mean val_acc | Val std | Mean test acc | Test std | Best single test |
|---:|---:|---:|---:|---:|---:|
| 0.040 | 0.5846 | 0.0191 | **0.5298** | 0.0072 | 0.5340 |
| 0.045 | **0.5855** | 0.0157 | 0.5255 | **0.0023** | 0.5283 |
| 0.050 | 0.5831 | **0.0113** | 0.5219 | 0.0126 | **0.5406** |

### Seed-level aggregation

| Seed | Mean test acc | Test range | Mean val_acc | Interpretation |
|---:|---:|---:|---:|---|
| 42 | 0.5193 | 0.5159–0.5230 | 0.5899 | low but stable test generalization |
| 43 | **0.5325** | 0.5243–0.5406 | 0.5870 | strongest random-control seed |
| 44 | 0.5259 | 0.5177–0.5336 | **0.5968** | strong validation, only mid test generalization |
| 45 | 0.5252 | 0.5133–0.5340 | 0.5637 | weak validation, occasional acceptable test |

### Aggregate decision

| Condition | Mean / Best | Decision |
|---|---:|---|
| no-recurrence baseline | `0.5490` | stronger than density-matched random recurrence |
| random_sparse density-control batch | mean `0.5257 ± 0.0084`, best single `0.5406` | **reject as performance explanation** |
| learned_lowrank validation rollback `m50p10` | mean `0.5919`, worst `0.5826` | main proposed test-leakage-free result |

Interpretation:

- Increasing fixed random recurrent density from `0.040` to `0.045` to `0.050` does not improve test performance.
- The best density by mean test accuracy is `p=0.040`, but even this reaches only `0.5298`.
- The best single random-control run is `p=0.050, seed=43` with test@best-val `0.5406`, still below the no-recurrence baseline `0.5490`.
- Validation accuracy and test accuracy are weakly aligned for fixed random topologies: `p045_s44` gives the highest validation accuracy `0.6066` but only test `0.5265`.
- This closes the density-only counterargument. Matching the learned-lowrank density regime is not sufficient; the learned topology advantage must come from edge placement, latent role structure, or downstream representation geometry rather than recurrent density alone.

Final claim:

> Density-matched fixed random recurrence fails to match even the no-recurrence baseline, while `learned_lowrank + validation rollback m50p10` remains substantially higher. Therefore the learned-lowrank gain is not explained by edge density alone; it requires learned edge placement / latent neuron role topology.



---

## 0. 현재 코드베이스 상태

현재 저장소는 두 축으로 구성되어 있다.

| 축 | 구현 위치 | 상태 |
|----|-----------|------|
| Feedforward Gumbel-SNN | `src/models/`, `src/training/`, `scripts/train.py` | MNIST/Fashion-MNIST/NMNIST/DVS 계열 실험 인프라 구현 |
| LSM 확장 | `src/lsm/`, `src/data/loaders.py`, `scripts/train_lsm.py` | SHD용 recurrent liquid 모델과 학습 루프 구현 |

Feedforward 쪽은 기존 실험 결과를 만든 검증된 코드이며, LSM 쪽은 SHD에서 Gumbel 기반 리퀴드 구조 학습을 검증하기 위한 현재 개발 대상이다. LSM 구현은 `InputProjection -> LiquidLayer -> Linear Readout` 구조이고, 리퀴드 내부 연결만 theta로 학습한다.

현재 LSM의 핵심 설계:

- SHD 입력: `tonic.datasets.SHD`, 10ms binning, `(T=100, 700)` spike tensor
- 입력→리퀴드: fixed sparse `randn` projection
- 리퀴드 내부: Dale's Law + `softplus(w_raw)` + learned/fixed/random/grad_r mask
- 학습: warmup 동안 theta freeze, 이후 epoch-level Gumbel noise + STE로 topology 학습
- 안정화: truncated BPTT, membrane clamp, threshold clamp, theta/weight gradient clipping 분리

---

## 1. 문제 의식

### 1.1 출발점: 뇌와 인공 신경망의 구조적 괴리

현재의 딥러닝은 입력→은닉→출력이라는 고정된 레이어 구조 위에서 작동한다. 연결 방향은 단방향이고, 어떤 뉴런이 어떤 뉴런과 연결되는지는 사람이 설계한다.

뇌는 다르다. 뉴런들은 사방으로 연결되어 있고, "출력 레이어"라는 것이 따로 없다. 신호가 네트워크를 순환하다가 특정 패턴으로 수렴하는 것 자체가 곧 결과이며, 그 수렴 상태가 운동 뉴런 등의 효과기를 통해 행동으로 이어진다.

이 프로젝트는 이 차이를 좁히는 첫 번째 단계로, **네트워크의 연결 구조(토폴로지) 자체를 학습 가능하게 만든 SNN**을 구축한다.

### 1.2 핵심 질문의 흐름

이 프로젝트는 하나의 일관된 질문에서 출발한다:

> "사방으로 연결된 신경망에서 출력값은 어떻게 나오는가?"

이 질문이 다음과 같이 전개되었다:

1. **뇌의 신호 처리 이해** — 뉴런들의 신호를 뇌는 어떻게 리드아웃하는가?
2. **출력 경로 이해** — 처리된 결과 신호는 어떻게 전달되는가?
3. **리드아웃의 본질** — 리드아웃이 별도로 존재하지 않는다는 것 확인
4. **판단과 출력의 연결** — 판단이 내려지는 시점에서 결과값은 어떻게 출력되는가?
5. **SNN으로 확장** — 이걸 SNN에서 피드포워드 없이 구현하면 출력을 어떻게 정의할 수 있는가?
6. **최적 방법 탐색** — 가장 뇌에 가까운 방식은 무엇인가?

도달한 통찰:

```
DNN의 질문: "어떤 레이어가 출력인가?"
       ↓
뇌의 답: "출력 레이어는 없다, 수렴이 곧 출력이다"
       ↓
SNN의 과제: "그 수렴을 어떻게 학습시킬 것인가?"
```

### 1.3 기존 SNN 학습의 한계

SNN에서 스파이크(발화/비발화)는 이산적이라 직접 역전파가 불가능하다. 이를 해결하기 위해 surrogate gradient 방법이 존재하며, 스파이크 함수를 역전파 시에만 부드러운 근사 함수로 대체하여 가중치와 임계값을 학습할 수 있게 한다.

그러나 **"어떤 뉴런과 어떤 뉴런을 연결할 것인가"**라는 토폴로지 결정은 여전히 이산적(연결 있음/없음)이라 surrogate gradient으로도 학습할 수 없다. 기존에는 사람이 구조를 미리 정하거나, 전부 연결해놓고 pruning하는 간접적 방법에 의존했다.

### 1.4 사방 연결의 수렴 문제

사방 연결(recurrent) SNN을 구현하기 위해서는 세 가지 문제를 해결해야 한다:

1. **수렴의 정의** — 스파이킹 뉴런들의 발화 패턴이 "안정됐다"는 걸 어떤 기준으로 판단할 것인가
2. **수렴을 유도하는 학습 규칙** — 역전파는 출력 레이어의 오차를 기준으로 삼는데, 출력 레이어가 없으면 오차 신호를 어디서 주입할 것인가
3. **수렴 패턴의 해석** — 수렴된 상태를 외부에서 읽어내는 것이 다시 "리드아웃"이 되는 순환 문제

출력 방식은 **운동 뉴런(효과기) 방식**을 채택했다. 뇌도 완전히 리드아웃이 없는 게 아니라 운동 뉴런이라는 지정된 출력 경로가 있되, 그 뉴런이 고립된 출력 레이어가 아니라 네트워크에 완전히 통합되어 있다는 점이 핵심이다.

### 1.5 GNN 기반 접근의 검토와 폐기

초기에는 SNN을 그래프 데이터로 모델링한 다음 GNN으로 SNN의 구조를 예측하는 방법을 검토했다. 그러나 다음과 같은 논리적 귀결에 의해 폐기되었다:

```
GNN이 필요한 경우:
  - SNN 내부로 역전파 불가 → GNN이 외부에서 설계 (블랙박스 방식)
  - 그러나 블랙박스라 정확한 그래디언트 계산 불가 → 비용이 큼

surrogate gradient을 쓰는 경우:
  - SNN 내부로 역전파 가능 → SNN 직접 학습 → GNN 불필요
```

surrogate gradient으로 역전파가 가능하면 GNN 없이 SNN을 직접 최적화할 수 있다. 두 접근을 동시에 쓰는 것은 구조적으로 모순이므로, GNN을 제외하고 SNN 직접 학습으로 방향을 확정했다.

### 1.6 unroll 문제와 단방향 우선 검증

사방 연결은 순환성 때문에 역전파 시 시간 축으로 unroll(순환을 시간 순서대로 펼침)이 필요하다. 수렴까지 T 타임스텝이 걸리면 네트워크 복사본이 T개 필요하여 메모리가 폭발한다. 이 문제는 뉴로모듈레이션 기반 로컬 학습 규칙 등 별도의 연구가 필요하므로, 현재 단계에서는 **단방향(feedforward) SNN에서 토폴로지 학습이 작동하는지를 먼저 검증**한다.

---

## 2. 해결 방안

### 2.1 핵심 아이디어: Gumbel-Softmax로 토폴로지를 미분 가능하게

GNN 설명 기법인 PGExplainer에서 사용하는 Gumbel-Softmax(Gumbel-Sigmoid) 트릭을 SNN의 연결 구조 학습에 적용한다.

각 뉴런 쌍 (i, j) 사이의 연결 존재 여부를 학습 가능한 파라미터 θᵢⱼ로 두고, Gumbel-Sigmoid로 연속 근사한다:

```
mᵢⱼ = σ((log ε - log(1-ε) + θᵢⱼ) / τ),  ε ~ U(0,1)
```

이를 통해 "연결이 있다/없다"의 이산적 결정을 미분 가능한 연산으로 바꾸어, 역전파만으로 네트워크가 스스로 필요한 연결을 발견하게 한다.

### 2.2 통합 학습 프레임워크

하나의 프레임워크 안에서 네 가지가 동시에 학습된다:

| 파라미터 | 의미 | 학습 방법 |
|---------|------|----------|
| θᵢⱼ | 연결 존재 여부 | Gumbel-Sigmoid |
| wᵢⱼ | 시냅스 가중치 | gradient descent |
| vᵢ | 스파이크 임계값 | gradient descent (은닉층만; 출력층은 1.0 고정) |
| βₗ | 레이어별 막전위 감쇠율 | gradient descent (log_beta → sigmoid) |

### 2.3 네트워크 구조

```
입력 뉴런 (784개, 28×28 픽셀)
    ↓  Gumbel-Sigmoid 엣지 마스크 × 학습 가능 가중치
은닉 뉴런 (hidden_layers: List[int], 기본값 [512]; 복수 은닉층 지원)
    ↓  Gumbel-Sigmoid 엣지 마스크 × 학습 가능 가중치
효과기 뉴런 (10개, 각각 숫자 0~9 담당; 임계값=1.0 고정, 학습 안 함)
```

- 뉴런 모델: LIF (Leaky Integrate-and-Fire)
- 스파이크 미분 근사: surrogate gradient (sigmoid 기반: σ(x)·(1−σ(x)))
- 토폴로지 미분 근사: Gumbel-Sigmoid with temperature annealing

### 2.4 순전파 과정

```
1. MNIST 이미지 → Poisson rate coding으로 입력 스파이크 생성
   (픽셀값을 발화 확률로 사용: spike = rand() < pixel_value)
2. 여러 타임스텝(T=25) 동안 SNN 시뮬레이션:
   a. 각 연결에 대해 엣지 마스크 계산: mᵢⱼ = GumbelSigmoid(θᵢⱼ, τ)
   b. 실제 전달 신호 = mᵢⱼ × wᵢⱼ × 프리시냅틱 스파이크
   c. 포스트시냅틱 뉴런의 막전위 누적 (LIF)
   d. 막전위 > 임계값(vᵢ) → 스파이크 발화 (surrogate gradient)
   e. 발화 후 막전위 리셋
3. 효과기 뉴런 발화율 집계 → softmax → 크로스엔트로피 손실
```

### 2.5 손실 함수

```
L = L_CE + λ_sparse × L_sparse + λ_commit × L_commit
```

- **L_CE**: 크로스엔트로피 (분류 정확도)
- **L_sparse**: sigmoid(θᵢⱼ)의 평균 (연결을 스파스하게 유도)
- **L_commit**: Binary entropy H(p) = -p·log(p) - (1-p)·log(1-p)의 레이어별 가중 평균 (θ를 0 또는 1로 양극화 유도; Layer 1에 가중치 2×, 이후 레이어 1×)

commitment loss가 필요한 이유: L_sparse(0으로 밀기)와 L_CE(연결 유지)가 0.4~0.5에서 균형을 이루면 gradient가 소멸하여 theta가 bimodal 분포를 형성하지 못하는 구조적 문제가 있었다. Binary entropy를 정규화 항으로 추가하면 p=0.5일 때 최대 페널티, 0 또는 1일 때 0이므로 중간값에 패널티를 줘서 양극화를 유도한다.

### 2.6 Temperature 스케줄링

```
학습 초기: τ = 1.0  → 부드러운 연속값, 탐색 장려
학습 후기: τ = 0.05 → 이진에 가까움, 확정적 구조 형성
어닐링: 25 에폭에 걸쳐 cosine annealing으로 점진적 감소, 이후 τ_end 고정
  τ(epoch) = τ_end + (τ_start − τ_end) × 0.5 × (1 + cos(π × epoch / tau_anneal_epochs))
```

### 2.7 개발 과정에서 발견한 버그와 해결

| 버그 | 원인 | 해결 |
|-----|------|------|
| hard 모드에서 마스크가 매번 달라짐 | hard=True에서도 Gumbel 노이즈 샘플링 | hard=True일 때 노이즈 없이 sigmoid(θ)≥0.5로 결정론적 반환 |
| sparsity가 항상 0.000 | sigmoid(0)=0.5인데 `>0.5`는 False | `>=0.5`로 수정 |
| L_sparse가 L_CE를 압도 | Layer 1(200,704개)이 Layer 2(2,560개)보다 78배 커서 `.sum()` 사용 시 불균형 | `.mean()`으로 수정, lambda_sparse 조정 |
| 출력 뉴런 임계값이 음수로 발산 | 학습이 threshold를 음수로 밀어 항상 발화 | `clamp(min=0.01)` 추가, Layer 2 threshold 학습 비활성화 |
| theta bimodal 미달성 | L_sparse와 L_CE가 0.4~0.5에서 gradient 소멸 | commitment_loss(binary entropy) 추가 |

---

## 3. 실험 계획

### 3.1 검증 목표 (3가지)

1. **작동 여부** — Gumbel-Sigmoid 엣지 마스크를 포함한 SNN이 학습되어 합리적인 정확도를 달성하는가
2. **토폴로지 창발** — 학습 후 어떤 연결 구조가 형성되는가, 스파스한 구조가 자연스럽게 나타나는가
3. **뉴런 특성 분화** — 각 뉴런의 임계값이 서로 다르게 학습되는가

### 3.2 실험 단계 (5단계)

| 단계 | 실험 | 목적 |
|-----|------|------|
| 1단계 | MNIST 학습 (n_hidden=512) | 기본 작동 검증, 3가지 목표 확인 |
| 2단계 | 비교 실험 (A: 학습 토폴로지 / B: 완전 연결 / C: 랜덤 스파스) | 학습된 토폴로지의 의미 검증 |
| 3단계 | Fashion-MNIST 처음부터 학습 | 태스크 일반화 확인 |
| 4단계 | MNIST 토폴로지 → Fashion-MNIST 이식 | 토폴로지 전이 가능성 확인 |
| 5단계 | DVS 뉴로모픽 데이터셋 | SNN의 시간적 동역학 + 토폴로지 학습 시너지 탐색 |

단계 순서의 논리적 근거:
- 2단계가 먼저인 이유: 랜덤 38%가 학습된 38%와 비슷한 성능을 내면, 이후 모든 실험의 해석이 흔들린다. 토폴로지 학습의 의미를 먼저 확립해야 후속 실험이 가치를 갖는다.
- 3단계→4단계 순서: 먼저 새 태스크에서 독립적으로 토폴로지 학습이 작동하는지 확인한 뒤, 크로스 태스크 전이를 시도해야 결과 해석이 명확하다.
- 5단계가 마지막인 이유: 1~4단계는 정적 이미지라 SNN의 시간적 장점을 살리지 못함. DVS에서 처음으로 시간 축이 의미를 가지며, 구조 확장(레이어 추가)도 이 시점에서 고려.

---

## 4. 실험 결과

### 4.1 1단계: MNIST 기본 학습

**설정**: n_hidden=512, 40 에폭, tau 1.0→0.05

| 항목 | 결과 |
|------|------|
| Test accuracy | 98.27% |
| Layer 1 활성 연결 | 37.5% (150,399 / 401,408) |
| Layer 2 활성 연결 | 57.1% (2,923 / 5,120) |
| theta bimodal | ✅ 달성 (두 레이어 모두) |
| Receptive field 창발 | ✅ 중앙 선호, 모서리 자동 제거 |
| 뉴런 임계값 분화 | ✅ 0.05~1.4 범위 분산 |

**핵심 발견**:
- 3가지 검증 목표 모두 달성
- Receptive field: 아무도 가르치지 않았는데 네트워크가 "MNIST 숫자는 이미지 중앙에 있으니까 모서리 픽셀은 무시해도 된다"를 스스로 발견
- 히든 뉴런 임계값 분포: 낮은 임계값(민감한 뉴런)부터 높은 임계값(선택적 뉴런)까지 역할 분화 발생
- Layer 2에서 각 출력 뉴런(0~9)이 서로 다른 히든 뉴런 집합을 선택적으로 사용 → 숫자별 "전문가 뉴런 집단" 형성

### 4.2 2단계: 비교 실험

| 조건 | 설명 | Test accuracy |
|------|------|--------------|
| A | 학습된 토폴로지 (37.7%) | 98.27% |
| B | 완전 연결 (100%) | 98.22% |
| C | 랜덤 스파스 (37.7%) | 97.83% |

**핵심 발견**: A > B > C 순서

- **A > B**: 연결을 62% 줄였는데 완전 연결보다 오히려 높은 성능. 토폴로지 학습이 단순한 "불필요한 연결 제거"가 아니라 "정보 흐름 경로의 최적화"를 수행함. 완전 연결에서는 노이즈 경로가 되는 연결들이 오히려 신호를 방해.
- **A > C**: 같은 38%의 연결이라도 학습된 38%가 랜덤 38%보다 0.44% 높음. 학습된 토폴로지가 단순한 압축이 아님을 증명.

### 4.3 3단계: Fashion-MNIST 처음부터 학습

| 설정 | Test accuracy |
|------|--------------|
| hidden=512, LR 고정 | 86.79% |
| hidden=1024, cosine scheduler | 86.63% |

**핵심 발견**:
- hidden을 2배 늘려도 성능 차이 없음
- Fashion-MNIST는 이 구조에서 86~87%가 한계
- MNIST와 달리 클래스 간 시각적 유사도가 높아(샌들 vs 스니커즈, 풀오버 vs 코트) 본질적으로 어려운 태스크임을 확인
- 토폴로지 학습 자체는 Fashion-MNIST에서도 정상 작동 (bimodal theta, receptive field 등)

### 4.4 4단계: MNIST 토폴로지 → Fashion-MNIST 이식

| 조건 | Test accuracy |
|------|--------------|
| Fashion-MNIST 처음부터 학습 | 86.79% |
| MNIST 토폴로지 이식 후 weight만 fine-tuning | ~51% |

**핵심 발견**: 갭이 35% 이상. 두 가지를 동시에 증명:

1. **학습된 토폴로지는 태스크에 고도로 특화된 구조** — MNIST에서 학습된 "어떤 픽셀을 보고, 어떤 히든 뉴런을 쓸 것인가"의 구조가 Fashion-MNIST에는 전혀 맞지 않음
2. **잘못된 토폴로지는 weight fine-tuning으로 극복 불가능** — 연결 구조가 부적절하면 가중치를 아무리 조정해도 한계가 있음

이 실패는 오히려 긍정적 결과다. 만약 이식이 성공했다면 토폴로지가 범용적이라는 뜻이 되어, "토폴로지가 태스크 구조를 반영한다"는 주장이 약해진다.

---

## 4.5 현재 LSM/SHD 상태 업데이트

SHD LSM 실험의 중심 결론은 다음처럼 업데이트한다.

- `random_sparse` recurrence는 현재 구조에서 no-recurrence baseline을 안정적으로 넘지 못한다.
- same-density random control도 실패했으므로, 단순한 recurrent density만으로는 성능 향상을 설명할 수 없다.
- Gumbel learned C는 seed 42/43/45에서 baseline을 넘었지만, seed 44에서는 original `0.5331`, freeze64 `0.5477`에 머물렀다.
- corrected Grad R-STE + adaptive freeze는 강한 hard-threshold baseline이 되었고, seed 44를 `0.5866`까지 회복했다.
- 최신 `learned_lowrank`는 edge-wise theta 대신 latent source/destination neuron embedding으로 topology logits를 만들며, 4-seed 기준으로 Grad R-STE + adaptive freeze보다 강하다.
- seed45 freeze72/tau0.05 결과는 남은 문제가 low-rank parameterization 실패가 아니라 **좋은 topology를 언제 freeze할 것인가**라는 topology selection 문제임을 보여준다.

현재 핵심 결과:

| 조건 | seed 42 | seed 43 | seed 44 | seed 45 | mean | median | 해석 |
|---|---:|---:|---:|---:|---:|---:|---|
| no recurrence | - | - | - | - | 0.5490 | 0.5490 | baseline |
| random_sparse density controls | - | - | - | - | **0.5257** | - | density-only control; below no-recurrence |
| learned C original | 0.5689 | 0.5751 | 0.5331 | 0.5587 | 0.5590 | 0.5638 | seed-sensitive |
| Grad R-STE non-freeze | 0.6038 | 0.5711 | 0.5587 | 0.5486 | 0.5706 | 0.5649 | strong hard-threshold learner |
| Grad R-STE + adaptive freeze | 0.6051 | 0.5808 | 0.5866 | 0.5486 | 0.5803 | 0.5837 | strongest hard-threshold baseline |
| learned_lowrank r16, no-freeze/tau0.05 | 0.5941 | 0.5861 | **0.6444** | 0.5751 | **0.5999** | 0.5901 | strongest peak-search setting |
| learned_lowrank r16, ramp10+bias0.05+tau0.2+freeze64 | 0.5989 | 0.5932 | 0.6334 | 0.5605 | **0.5965** | **0.5961** | stable fixed-freeze setting |
| learned_lowrank r16, best stable schedule upper bound | 0.5989 | 0.5932 | 0.6334 | **0.5751** | **0.6002** | **0.5961** | oracle/topology-selection upper bound |
| learned_lowrank r16, validation rollback m50p10 | 0.5857 | **0.6135** | 0.5857 | 0.5826 | **0.5919** | 0.5857 | first test-leakage-free adaptive policy |

해석:

- seed 44 결과가 가장 중요하다. seed 44는 edge-wise Gumbel learned C에서는 실패했지만, Grad R-STE + adaptive freeze에서는 `0.5866`, learned_lowrank에서는 no-freeze `0.6444` / stabilized `0.6334`까지 올라갔다.
- seed 45도 residual weak seed가 아니다. no-freeze/tau0.05와 freeze72/tau0.05에서 `0.5751`까지 도달했다.
- 다만 no-freeze seed45는 final `0.5367`로 collapse했고, freeze72는 final을 `0.5663`으로 회복했다.
- 따라서 learned_lowrank의 다음 문제는 **parameterization이 가능한가**가 아니라 **학습 중 발견한 좋은 topology를 validation 기반으로 언제 고정할 것인가**이다.
- best stable schedule은 사람이 사후 선택한 것이므로 논문 메인 표에 그대로 쓰면 안 된다. 현재는 topology-selection upper bound로만 해석한다.
- validation split과 topology snapshot/rollback은 구현 완료됐다.
- 첫 validation policy `m50p10`은 test@best-val mean `0.5919`, worst `0.5826`으로 Grad R-STE + adaptive freeze보다 안정적이지만, 기존 low-rank peak를 완전히 회복하지는 못했다.

다음 단계:

- `m60p10`은 redundant, `m60p15`는 reject로 닫았다.
- `m50p10`을 main-table 후보로 고정하고, same-density random controls와 topology diagnostics로 주장 방어를 진행한다.
- same-density random control과 topology diagnostics로 density-only explanation을 배제한다.
- prediction auxiliary / e-prop / predictive coding은 topology freeze timing 문제가 정리된 뒤 후속 연구로 둔다.

## 5. 누적된 주장의 구조

```
1단계 결과:
  토폴로지 학습이 작동한다 (정확도 + bimodal + receptive field + 임계값 분화)

2단계 결과:
  A > B > C → "어떤 연결을 쓰느냐가 얼마나 많이 쓰느냐보다 중요하다"

4단계 결과:
  이식 실패 → "토폴로지는 태스크 특화 구조를 인코딩한다"

2단계 + 4단계 종합:
  학습된 토폴로지는 랜덤이 아니다 (A > C)
  학습된 토폴로지는 범용이 아니다 (이식 실패)
  ∴ 학습된 토폴로지는 해당 태스크에 특화된 의미 있는 구조다
```

**핵심 주장**: Gumbel-Sigmoid로 학습된 SNN 토폴로지는 태스크의 구조적 특성을 반영한 의미 있는 연결 패턴이다.

---

## 6. 다음 단계: LSM 기반 recurrent 검증

### 6.1 의미

1~4단계는 정적 이미지 기반이었다. 이후 NMNIST/DVS Gesture로 이벤트 데이터까지 확장하면서, fully-connected feedforward 구조의 한계가 드러났다. 입력 차원이 커지면 `input × hidden` theta 수가 급증하고, 데이터가 충분하지 않은 경우 topology가 bimodal하게 정착하지 못한다.

현재 다음 단계는 단순히 DVS feedforward 실험을 늘리는 것이 아니라, **LSM으로 전환하여 recurrent liquid 내부의 구조를 학습하는 것**이다. 입력→리퀴드는 고정하고, 리퀴드 내부의 `N²` 연결만 학습하면 입력 차원 증가에 따른 theta 폭발을 피할 수 있다.

### 6.2 후보 데이터셋

- **SHD (현재 LSM 기준 데이터셋)**: 700채널 음성 spike stream, 20클래스. `src/data/loaders.py`와 `configs/lsm_shd_baseline.yaml`에 구현되어 있다.
- **SSC (확장 후보)**: SHD보다 큰 speech command spike dataset. SHD에서 방법론이 안정화된 뒤 확장 후보.
- **NMNIST / DVS128 Gesture**: feedforward 이벤트 데이터 실험 인프라는 존재하지만, LSM 논문 방향에서는 보조 비교 또는 확장 실험 후보.

### 6.3 장기 비전

현재 코드베이스는 recurrent SNN으로의 1차 확장을 이미 시작했다. 다만 학습은 아직 BPTT 기반이므로, 더 긴 시퀀스와 더 큰 네트워크로 확장하려면 e-prop/RTRL 계열 online learning 또는 gradient checkpointing이 필요할 수 있다. 장기적으로는 LSM의 입력/리퀴드/리드아웃 분리를 약화시키고, 효과기 뉴런이 네트워크에 통합된 interconnected SNN으로 확장하는 것이 목표다.
