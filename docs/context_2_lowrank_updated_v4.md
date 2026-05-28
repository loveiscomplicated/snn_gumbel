---

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



## 1. 연구 비전 및 핵심 문제

### 1.1 궁극적 목표: 뇌를 닮은 Interconnected SNN

인간의 뇌에서 뉴런들은 “입력층 → 은닉층 → 출력층”이라는 단방향 구조로 연결되어 있지 않다. **모든 뉴런이 사방으로 서로 얽혀 있으며(interconnected)**, 신호가 네트워크를 순환하다가 특정 패턴으로 수렴하는 것 자체가 곧 결과이다.

```
기존 인공신경망 (DNN/SNN):
  입력 → 레이어1 → 레이어2 → ... → 출력
  단방향, 사람이 설계한 고정 구조

인간의 뇌:
  모든 뉴런이 사방으로 interconnected
  신호가 순환하며 수렴 → 그 자체가 출력
  연결 구조는 경험을 통해 학습됨
```

본 연구의 궁극적 목표는 **뉴런들이 사방으로 interconnected되어 있고, 그 연결 구조 자체가 학습을 통해 결정되는 SNN 모델**을 구축하고, 이런 구조가 어떤 계산적 이점을 가지는지 탐구하는 것이다.

### 1.2 논문의 위치: LSM을 첫 번째 검증 무대로

이 큰 비전을 학계에 제시하기 위해, 이미 순환 연결 구조를 갖춘 기존 개념인 **Liquid State Machine(LSM)**을 출발점으로 삼는다.

LSM을 선택한 이유:
- LSM은 이미 뉴런들이 순환 연결된 SNN이라는 구조를 갖고 있음
- SNN 커뮤니티에서 확립된 벤치마크와 비교 대상이 풍부함
- “interconnected SNN”이라는 개념을 기존 연구자들에게 설명하기 용이함

그러나 **궁극적으로 LSM의 영역에 갇혀 있을 필요는 없다.** LSM은 interconnected SNN의 한 가지 형태일 뿐이며, 본 연구에서 개발하는 구조 학습 방법론은 더 넓은 범위의 interconnected SNN에 적용 가능한 원리이다.

### 1.3 현재 LSM의 한계

LSM은 뇌의 피질 미세회로를 모방한 reservoir computing 모델로, 랜덤하게 연결된 스파이킹 뉴런의 저수조(liquid)와 학습 가능한 리드아웃으로 구성된다. 현재 LSM 연구의 근본적 한계는 **리퀴드 내부 연결이 랜덤으로 고정**되어 있다는 점이다.

- 기존 접근: 리퀴드는 고정, 리드아웃만 학습
- 성능 향상의 주된 방법이 리저버 크기를 키우는 것뿐 → 수확 체감
- “왜 그 구조여야 하는지”에 대한 근거 없음 → 성능 천장 존재

뇌에서는 시냅스 가소성을 통해 연결 구조가 끊임없이 변화하는데, LSM은 이 핵심적 특성을 반영하지 못하고 있다.

### 1.4 핵심 제안

**Gumbel-Softmax trick을 사용하여 리퀴드 내부 연결 구조 자체를 훈련 중에 학습하고, 추론 시에는 학습된 이산 구조를 그대로 사용한다.**

이는 interconnected SNN에서 “어떤 뉴런과 어떤 뉴런을 연결할 것인가”를 네트워크가 스스로 발견하게 하는 첫 번째 단계이다.

### 1.5 현재 실험 업데이트: Gradient-Based Topology Learning에서 Latent Topology Parameterization으로 확장

최근 SHD LSM 결과는 초기 프레이밍을 다시 한 단계 확장한다. Gumbel-Sigmoid learned C는 일부 seed에서 no-recurrence baseline을 넘었지만 seed 44에서 불안정했다. 이후 corrected Grad R-STE + adaptive freeze가 강한 hard-threshold baseline이 되었고, topology stabilization의 중요성을 확인했다. 그러나 최신 `learned_lowrank` 결과는 더 중요한 전환점을 만든다.

`learned_lowrank`는 edge마다 독립적인 `theta_ij`를 두지 않는다. 대신 각 뉴런에 source embedding과 destination embedding을 두고, edge logit을 다음처럼 materialize한다.

```text
topology_logit_ij = src_embed_i · dst_embed_j + theta_bias
```

즉 연결은 독립 edge parameter가 아니라 **latent neuron role 조합**으로 결정된다. 이는 기존 문제의식 — “엣지를 독립 객체로 학습하면 edge 조합이 seed에 따라 불안정하게 굳어질 수 있다” — 에 대한 직접적인 구조적 대응이다.

현재 핵심 수치:

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

따라서 현재 중심 주장은 다음처럼 업데이트한다.

> 랜덤 recurrent topology는 충분하지 않다. 핵심은 recurrent topology를 gradient로 학습하는 것이며, 더 나아가 edge-wise independent parameterization보다 **latent neuron role 기반 topology parameterization**이 더 강한 inductive bias가 될 수 있다. `learned_lowrank`는 4-seed 기준으로 Grad R-STE + adaptive freeze보다 강하며, seed44 failure를 강하게 뒤집었다. 다만 seed45 freeze72 결과가 보여주듯, 마지막 남은 문제는 topology parameterization 자체가 아니라 **좋은 topology를 validation 기반으로 언제 freeze할 것인가**이다.

이 업데이트는 “Gumbel trick 하나”에서 “gradient-based recurrent topology learning”으로, 다시 “edge-wise topology search vs latent role-based topology search vs adaptive topology selection”으로 연구 질문이 정교화되었음을 의미한다. 현재 논문의 가장 강한 포인트는 **연결을 독립 edge가 아니라 뉴런 role interaction으로 parameterize했을 때 topology formation이 달라지고, 기존 failure seed가 최고 성능 seed로 반전될 수 있다**는 것이다.

---

## 2. 배경 이론

### 2.1 Liquid State Machine 구조

LSM은 세 부분으로 **명확하게 분리**되어 있다:

### 입력층 (Input Layer)

- “뉴런 층”이 아니라 외부 스파이크를 리퀴드로 전달하는 **연결(시냅스)**
- SHD의 경우: 700개 입력 채널 → 리퀴드 뉴런에 희소하게 랜덤 연결
- 연결 확률 p_in으로 각 입력 채널-리퀴드 뉴런 연결 결정
- 가중치: 균등분포에서 랜덤 샘플링
- **항상 흥분성(excitatory) 시냅스**
- 전통적 LSM에서 **고정 (학습하지 않음)**

### 리퀴드 (Liquid/Reservoir)

- 순환적으로 연결된 스파이킹 뉴런의 집합
- 보통 **80% 흥분성 + 20% 억제성** LIF 뉴런 (뇌 피질 비율 반영)
- 뉴런 간 랜덤 순환 연결 → 시변 입력을 시공간적 활성화 패턴으로 변환
- 전통적 LSM에서 **가중치도, 구조도 완전히 고정**
- **→ 본 연구에서 Gumbel-Softmax로 연결 구조를 학습하는 부분**

### 리드아웃 (Readout Layer)

- 지도 학습으로 훈련되는 분류기
- 메모리가 없음 (memoryless)
- 리퀴드 뉴런의 스파이크 카운트를 특징 벡터로 사용
- 로지스틱 회귀 또는 선형 분류기
- **기존에도 학습하는 부분 → 그대로 유지**

### 데이터 흐름

```
스파이크 입력 → (랜덤 고정 연결) → 리퀴드에서 순환 처리 → (스파이크 카운트 추출) → 리드아웃이 분류
```

### 2.2 LIF 뉴런의 흥분성/억제성 구분

- LIF 뉴런의 동역학 자체는 동일 (막전위 적분, 임계값 초과 시 발화, 리셋, leak 감쇠)
- **차이는 시냅스 가중치의 부호**:
    - 흥분성(Excitatory) 뉴런 → 양수 가중치 → 받는 뉴런의 막전위 상승
    - 억제성(Inhibitory) 뉴런 → 음수 가중치 → 받는 뉴런의 막전위 하강
- **Dale’s Law**: 하나의 뉴런은 모든 시냅스에서 같은 종류의 신경전달물질만 분비
    - 생물학적으로 흥분/억제를 동시에 하는 뉴런은 불가능

### 2.3 Gumbel-Softmax Trick

- 원래 엣지 존재 여부: 있다(1) / 없다(0) → 미분 불가능
- Gumbel-Softmax:
    - 훈련 중: 0과 1 사이 연속값으로 근사 → 역전파 가능
    - 추론 시: 딱 0 또는 1로 확정
- PGExplainer에서 GNN 설명을 위해 사용된 아이디어를 LSM 구조 학습에 적용

### 2.4 Surrogate Gradient

- SNN의 스파이크 함수는 미분 불가능 (계단 함수)
- 역전파 시 대리 기울기(surrogate gradient)로 근사하여 학습 가능
- 본 연구에서는 리퀴드 내부 가중치도 surrogate gradient로 함께 학습

### 2.5 순환 구조와 타임스텝

### 두 가지 “타임스텝”의 구분

| 개념 | 설명 | 루프 영향 |
| --- | --- | --- |
| **시뮬레이션 해상도 (dt)** | 물리적 시간을 이산적으로 쪼개는 단위 (예: 1ms) | 루프 유무와 무관, 항상 사람이 설정 |
| **처리 깊이** | 네트워크가 입력을 충분히 처리하는 데 필요한 시간 | 루프 구조에 의해 자연스럽게 결정 가능 |
- 시뮬레이션 해상도 dt: 뉴런 동역학의 수치적 해를 구하기 위해 **반드시 사람이 설정**
- 처리 깊이: 루프 구조가 있으면 신호가 순환하며 자연스럽게 수렴 시점이 결정
    - 단순한 입력 → 빨리 수렴, 복잡한 입력 → 더 많이 순환
    - Adaptive Computation Time, Deep Equilibrium Models과 연결되는 개념

---

## 3. 선행 연구 서베이 및 포지셔닝

### 3.1 Novelty 확인: Gumbel-Softmax × SNN 연결 학습 = 선행 연구 없음

광범위한 서베이 결과, **Gumbel-Softmax(또는 Gumbel-Sigmoid)를 SNN의 뉴런 쌍 수준 연결 구조 학습에 적용한 논문은 발견되지 않았다.** Gumbel-Softmax는 GNN 설명(PGExplainer, Luo et al. 2020), GNN 그래프 재배선(Gumbel-MPNN, Hoffmann et al. 2025), ANN NAS(채널/연산 선택) 등에서 사용되었으나, SNN의 시냅스 존재 여부를 학습하는 데는 적용되지 않았다.

### 3.2 가장 가까운 경쟁 영역: Sparse Connectivity Learning in SNN

본 연구와 목적이 가장 유사한 기존 연구들은 SNN에서의 sparse training / rewiring 방법론이다.

**주요 경쟁 방법들:**

| 방법 | 연결 제거 | 연결 생성 | 미분 가능 | 핵심 한계 |
| --- | --- | --- | --- | --- |
| DEEP R (Bellec et al., 2018) | 부호 변경 시 | 랜덤 | ✗ | 새 연결 위치가 랜덤 |
| SET (Mocanu et al., 2018) | 최소 magnitude | 랜덤 | ✗ | 새 연결 위치가 랜덤 |
| RigL (Evci et al., 2021) | 최소 magnitude | 최대 gradient | △ | 이산적 결정 |
| Grad R (Chen et al., 2021) | θ ≤ 0 | θ > 0 regrowth | △ | hard threshold |
| ESL-SNNs (Shen et al., 2023) | magnitude/gradient | gradient/random | ✗ | 규칙 기반 |
| **본 연구** | **sigmoid(θ) → 0** | **sigmoid(θ) → 1** | **✓** | — |

**핵심 차별점**: 기존 방법들은 “제거 → 추가” 사이클을 반복하는 **이산적 rewiring**인 반면, Gumbel-Sigmoid는 연결 확률을 **연속 공간에서 직접 최적화**한다. 이로 인해 (1) gradient 흐름이 더 안정적이고, (2) temperature annealing으로 탐색↔︎확정을 자연스럽게 전환하며, (3) commitment loss 같은 정규화를 통해 구조의 양극화를 유도할 수 있다.

특히 **Grad R (Chen et al., 2021)**이 가장 가까운 경쟁자인데, θ > 0이면 연결 존재, θ ≤ 0이면 제거하는 hard threshold 방식이다. 본 연구의 Gumbel-Sigmoid는 이 결정을 soft하게 만들어 gradient가 더 안정적으로 흐르게 한다.

**Mészáros et al. (2024)**는 SHD 벤치마크에서 DEEP R + RigL을 결합하여 구조 학습이 지연 학습보다 더 중요하다는 발견을 보고했다. 이는 본 연구의 “토폴로지 학습이 중요하다”는 주장과 일치하며, 방법론 수준에서의 차별화(규칙 기반 vs 미분 가능)를 강조할 수 있다.

### 3.3 SNN NAS와의 구분

SNN NAS 연구(SNASNet, SpikeDHS, MA-DARTS 등)는 활발하나, 모두 **레이어/셀 수준의 매크로 구조**를 탐색한다. 본 연구는 **뉴런 쌍 수준의 마이크로 연결**을 학습하므로, 상호 보완적 관계이다. 원칙적으로 NAS로 매크로 구조를 정한 후 본 연구의 방법으로 마이크로 연결을 최적화할 수 있다.

### 3.4 LSM 구조 최적화와의 구분

LSM 리퀴드 구조를 최적화하려는 연구(CMA-ES: Zhou et al. 2019, EONS: Plank et al. 2019, DA-BCM: Wang et al. 2023)는 주로 **진화적 탐색(블랙박스)**에 의존한다. 본 연구는 **loss → θ → 구조로 직접 역전파 가능한 end-to-end gradient 기반**이라 차원이 높은 탐색 공간에서 훨씬 효율적이다.

### 3.5 논문 포지셔닝 문장

> “기존 SNN 구조 학습은 규칙 기반 rewiring(DEEP R, SET, ESL-SNNs)이나 진화적 탐색(EONS, CMA-ES)에 의존했다. 한편, Gumbel-Softmax는 GNN 설명(PGExplainer)이나 ANN의 NAS에서 이산적 구조 결정을 미분 가능하게 만드는 데 성공적으로 사용되었다. 그러나 이 두 연구 흐름의 교차점 — Gumbel-Softmax를 SNN의 시냅스 연결 학습에 적용하는 것 — 은 탐색되지 않았다. 본 연구는 이 간극을 메우며, surrogate gradient와 Gumbel-Sigmoid를 결합하여 SNN의 가중치, 임계값, 연결 구조를 동시에 end-to-end로 학습하는 첫 번째 프레임워크를 제시한다.”
> 

### 3.6 향후 연구와의 연결점

- **e-prop (Bellec et al., 2020)**: BPTT의 생물학적 근사. 향후 recurrent 확장 시 BPTT 대안으로 핵심적. e-prop + Gumbel-Sigmoid 결합이 향후 연구 방향.
- **LSNN + DEEP R (Bellec et al., 2018b)**: 12% 연결만으로 완전 연결보다 우수한 성능 → 본 연구의 “희소 구조가 더 낫다” 주장의 선행 증거.

---

## 4. 연구 설계

### 4.1 실험 구도: 세 단계 비교

논문의 설득력을 위해 **세 단계 비교**가 필수적이다. A→B의 성능 점프와 B→C의 성능 점프를 분리해서 보여줘야 “구조 학습의 추가적 가치”를 명확히 입증할 수 있다.

|  | 리퀴드 연결 구조 | 리퀴드 가중치 | 리드아웃 |
| --- | --- | --- | --- |
| **A. 전통 LSM** | 랜덤 고정 | **고정** | 학습 |
| **B. 가중치 학습 LSM** | 랜덤 고정 | **Surrogate gradient 학습** | 학습 |
| **C. 제안 방법** | **Gumbel-Softmax 학습** | **Surrogate gradient 학습** | 학습 |

### 각 비교의 의미

**A → B (가중치 학습의 효과):**
- 리퀴드를 고정하는 전통 LSM 대비, surrogate gradient로 가중치를 학습하면 얼마나 개선되는가
- 이 점프가 크면: “리퀴드를 학습 가능하게 만드는 것 자체가 가치 있다”

**B → C (구조 학습의 추가 효과):**
- 가중치를 이미 학습하는 상황에서, 구조까지 학습하면 추가 개선이 있는가
- 이 점프가 유의미하면: “가중치만 학습해선 한계가 있고, 구조까지 학습해야 한다”
- 이 점프가 미미하면: “구조 학습의 추가 가치가 뭔가?”라는 공격에 취약

**A → C (전체 효과):**
- 전통 LSM 대비 제안 방법의 전체 개선 폭

### Baseline B의 공정성 보장

B→C 비교의 신뢰성을 위해, Baseline B의 연결 확률 p가 결정적으로 중요하다. p를 부적절하게 설정하면 B가 불공정하게 약해져서 B→C 점프가 인위적으로 과대평가될 수 있다.

**문제**: B에서 p=0.1이면 너무 희소하고, p=0.5면 너무 밀집. 어느 쪽이든 B의 성능이 최적이 아닐 수 있으며, 리뷰어가 “B의 연결 확률을 최적화했으면 C와 비슷했을 텐데?”라고 공격 가능.

**해결 전략 — 두 가지 비교를 모두 수행:**

**(1) 메인 비교: 동일 희소성 비교 (가장 순수한 비교)**

C가 학습 후 최종적으로 k%의 연결을 살렸다면, B에서도 p=k/100으로 설정하여 비교.

```
B* (동일 희소성 랜덤): 연결 30%, 위치는 랜덤
C  (학습된 구조):      연결 30%, 위치는 학습으로 결정

→ 연결 수는 동일, 차이는 오직 "어디를 연결했느냐"
→ 구조 학습의 가치를 가장 순수하게 드러냄
```

이전 feedforward SNN 실험(학습된 38% vs 랜덤 38%)과 동일한 로직.

**(2) Ablation: p sweep**

B에서 p ∈ {0.1, 0.2, 0.3, 0.5}를 각각 실험하여 최적 p를 탐색.

이 ablation이 주는 추가 분석:
- B의 최적 p와 C가 자체적으로 찾은 희소성 비율을 비교
- 만약 일치하면: **“Gumbel-Softmax가 최적 연결 밀도까지 자동으로 발견한다”**는 추가 기여
- 만약 불일치하면: **“랜덤 구조의 최적 밀도와 학습된 구조의 최적 밀도가 다르다”**는 흥미로운 발견

**최종 실험 비교 테이블:**

|  | 리퀴드 구조 | 리퀴드 가중치 | 연결 밀도 | 역할 |
| --- | --- | --- | --- | --- |
| **A** | 랜덤 고정 | 고정 | 기존 LSM 기본값 | 전통 LSM baseline |
| **B (p sweep)** | 랜덤 고정 | 학습 | p ∈ {0.1, 0.2, 0.3, 0.5} | 가중치 학습 효과 + 최적 p 탐색 |
| **B*** | 랜덤 고정 | 학습 | C와 동일한 희소성 | **가장 공정한 비교** |
| **D (Grad R)** | Hard threshold rewiring | 학습 | 동적 변화 | **기존 gradient 기반 rewiring baseline** |
| **C** | Gumbel 학습 | 학습 | 자동 결정 | 제안 방법 |

### 현재 실험 설계 업데이트

초기 설계에서는 C(Gumbel)가 주된 제안 방법이고 D(Grad R)는 closest baseline이었다. 그러나 corrected Grad R-STE + adaptive freeze 결과가 강하게 나오면서, 실험 구도는 다음처럼 업데이트한다.

| 조건 | 의미 | 현재 역할 |
|---|---|---|
| A | 전통 LSM / fixed liquid | 전통 baseline |
| B | random topology + trainable dynamics | random recurrent baseline |
| B* | same-density random topology | density-only control |
| C | Gumbel-Sigmoid learned topology | stochastic/relaxed topology learner |
| D | Grad R-STE hard-threshold topology learner | strong competing topology-learning baseline |
| D+AF | Grad R-STE + adaptive freeze | current strongest recipe |
| D+AF+Pred | Grad R-STE + adaptive freeze + prediction auxiliary loss | next experiment |

핵심 비교는 이제 `B* < C`만이 아니라 `B* < {C, D, D+AF}`이다. 즉, 학습된 edge placement가 random edge placement보다 유리한지, 그리고 어떤 topology learner가 더 안정적인지를 함께 본다.


### Grad R을 추가 baseline으로 포함하는 이유

Grad R (Chen et al., 2021)은 본 연구와 가장 가까운 경쟁자이다. 둘 다 gradient 기반으로 연결을 제거/추가하며 SNN에 직접 적용한다. 차이는 “hard threshold (θ>0이면 연결) vs soft relaxation (Gumbel-Sigmoid)”뿐이다.

리뷰어가 “Grad R에 비해 실질적으로 뭐가 나은가?”라고 물으면 방법론적 우아함만으로는 부족하고, **실증적 우위**가 필요하다.

**구현 난이도**: 매우 낮음. 본 연구 코드에서 Gumbel-Sigmoid 부분만 hard threshold로 교체하면 됨. 사실상 ablation 수준.

**비교의 의미**: 같은 gradient 신호를 받는데, hard threshold로 처리 vs soft relaxation으로 처리. 변수 통제가 완벽한 비교.

**이론적 차이점** (Discussion에서 분석):
- Grad R: θ=0 근처에서 gradient 불연속 → 학습 불안정성
- Gumbel-Sigmoid: temperature로 매끄럽게 전환 → 안정적 gradient 흐름
- Gumbel-Sigmoid: temperature annealing으로 탐색(높은 τ)↔︎확정(낮은 τ)의 자연스러운 전환
- Gumbel-Sigmoid: commitment loss로 양극화 유도 가능 (Grad R에는 대응물 없음)

### 현재 Grad R-STE 결과에 따른 포지셔닝 수정

초기 기대와 달리, corrected Grad R-STE는 단순한 약한 baseline이 아니라 매우 강한 topology learner로 나타났다. 특히 adaptive freeze를 붙이면 seed 42/43/44에서 각각 `0.6051`, `0.5808`, `0.5866`을 기록했고, Gumbel learned C가 실패했던 seed 44를 크게 회복했다.

따라서 Discussion에서는 다음처럼 정리하는 것이 안전하다.

- Gumbel-Sigmoid의 장점: stochastic exploration, temperature-controlled relaxation, soft-to-hard transition.
- Grad R-STE의 장점: deterministic hard topology, simpler dynamics, no sampling noise.
- Adaptive freeze의 장점: seed마다 다른 topology instability 시점을 gradient signal로 감지하여 topology를 고정.

현재 결론은 “Gumbel이 Grad R보다 항상 낫다”가 아니라, **recurrent SNN에서는 topology learning과 topology stabilization이 모두 중요하다**는 것이다.

현재 최신 결과를 반영하면 여기에 한 가지 축이 추가된다. **learned_lowrank**는 Gumbel/Grad R의 soft-vs-hard 차이가 아니라, topology parameterization 자체를 바꾼다. edge별 독립 θ를 제거하고, 뉴런별 source/destination embedding으로 전체 edge logit field를 생성한다. 따라서 Discussion의 비교축은 다음 세 가지가 된다.

- Gumbel-Sigmoid: stochastic relaxed topology learner.
- Grad R-STE: deterministic hard-threshold topology learner.
- learned_lowrank: latent neuron role-based topology parameterization.

현재 관측상 learned_lowrank는 seed 44에서 `0.6444`를 기록하여 가장 강한 단일 run을 만들었고, 4-seed no-freeze mean `0.5999`, stabilized mean `0.5965`로 Grad R-STE + adaptive freeze를 넘어섰다. 다만 seed45 freeze72가 보여주듯 수동 freeze epoch 선택은 post-hoc이므로, 최종 주장은 validation-based adaptive freeze/rollback을 적용한 결과로 방어해야 한다.


### 학습된 구조의 패턴 분석: LSNN+DEEP R과의 비교

Bellec et al. (2018b)의 LSNN + DEEP R에서 두 가지 구조적 발견이 보고되었다:
1. 12% 연결만으로 완전 연결보다 우수한 성능
2. 리드아웃 뉴런으로의 연결이 네트워크 나머지보다 자발적으로 더 밀집

본 연구에서 Gumbel-Softmax가 학습한 구조를 분석할 때, 이 패턴들과 직접 비교한다:

```
분석 항목:
  1. 최종 희소성 비율 → DEEP R의 12%와 비교
  2. 리드아웃에 가까운 뉴런의 연결 밀도 → 자발적 밀집 여부
  3. 허브 뉴런의 위치 → 입력 근처인가, 리드아웃 근처인가

해석:
  유사한 패턴 → "방법이 다르지만 구조가 수렴" = 보편적 현상
  다른 패턴 → 차이의 원인 분석 = 흥미로운 발견
```

### 선행 연구와의 관계

B(리퀴드 가중치 학습) 자체의 위치:
- **전통 LSM의 핵심 정체성은 “리퀴드를 학습하지 않는다”는 것**
- 리퀴드 가중치를 학습하는 시도는 존재함 (Jin and Li, 2016; Ivanov and Michmizos, 2021; 진화적 최적화 Zhou et al., 2020; intrinsic plasticity 등)
- 그러나 이런 접근들은 **LSM 커뮤니티의 주류가 아니며**, BPTT SNN 대비 SOTA에 미도달
- Surrogate gradient로 리퀴드를 직접 학습하는 것은 **비표준적 접근**

### 논문 프레이밍의 확장

B 자체가 LSM 커뮤니티에서 표준이 아니므로, 논문의 기여를 더 넓게 잡을 수 있다:

```
좁은 프레이밍:
  "Gumbel-Softmax로 구조를 학습한다" (C만 기여)

넓은 프레이밍:
  "Surrogate gradient로 리퀴드를 학습 가능하게 만들고 (B)
   + 거기에 구조까지 동시에 학습한다 (C)"
  → 두 단계 모두 기여로 인정받을 수 있음
```

**전략**: 만약 B→C 점프가 작더라도, A→B 점프가 크면 논문 전체의 가치는 유지된다. 반대로 B→C 점프가 크면 핵심 기여가 더욱 명확해진다. 어느 쪽이든 세 단계를 모두 보여주면 리뷰어의 공격에 방어 가능.

### 4.2 리퀴드 내부 구조 학습 방법

### 파라미터 설계

- 각 뉴런 쌍 (i, j)마다 학습 가능한 파라미터 **θ_ij** 배정
- θ_ij를 sigmoid에 통과 → 연결 확률 (θ=0 → sigmoid(0) = 0.5)
- θ_ij를 **N(0, σ)로 초기화** → 모든 연결이 0.5 근처에서 시작 (중립적)
- Gumbel-Softmax가 θ를 기반으로 연결 존재 여부를 미분 가능하게 샘플링

### 초기화의 의미

- σ가 작으면 (예: 0.01): 거의 다 0.5에서 시작
- σ가 크면 (예: 0.5): 초기부터 다양성 존재
- 네트워크에게 **어떤 사전 편향도 주지 않음** (완전히 중립적 출발)
- 학습 후: 어떤 θ는 큰 양수 (연결 확률 → 1), 어떤 θ는 큰 음수 (확률 → 0)
- **최종 희소성 비율도 네트워크가 스스로 결정**

### Dale’s Law 적용: Softplus 방식

Dale’s Law에 따라 흥분성 뉴런의 모든 출력 시냅스는 양수, 억제성 뉴런의 모든 출력 시냅스는 음수여야 한다.

**구현 방식**: `abs()` 대신 **Softplus**를 사용하여 gradient 안정성을 확보.

```python
# w_raw: 자유롭게 학습되는 파라미터 (부호 제약 없음)
# dale_sign: 흥분성 뉴런이면 +1, 억제성이면 -1 (고정, 학습 안 함)
w_eff = dale_sign * F.softplus(w_raw)
```

- `softplus(x) = log(1 + exp(x))` → 항상 양수, 매끄러운 gradient
- `dale_sign`이 부호를 결정 → 흥분성은 양수, 억제성은 음수 보장
- `abs()` 방식의 0 근처 gradient 불연속 문제 완전 회피 (Bellec et al.의 e-prop에서도 유사 처리)

**Gumbel-Sigmoid 마스크와의 결합 — 세 가지 깔끔한 분리:**

```python
effective_W = mask * (dale_sign * F.softplus(w_raw))
```

| 요소 | 역할 | 학습 여부 |
| --- | --- | --- |
| mask | 연결 존재 여부 | Gumbel-Sigmoid (θ에서 학습) |
| dale_sign | 시냅스 부호 | 뉴런 유형에서 고정 (학습 안 함) |
| softplus(w_raw) | 시냅스 크기 | Surrogate gradient로 학습 |

**`dale_sign` 행렬의 형태 — 주의사항:**

Dale’s Law는 **시냅스전 뉴런**(presynaptic)의 유형에 의해 결정된다. 현재 코드의 가중치 행렬이 `(N_pre, N_post)` 형태이므로:

```python
# N개 뉴런 중 앞 80%가 흥분성, 뒤 20%가 억제성
n_exc = int(0.8 * N)
dale_sign = torch.ones(N, 1)       # (N_pre, 1) — 브로드캐스팅용
dale_sign[n_exc:, :] = -1.0        # 억제성 뉴런의 행 전체를 -1
dale_sign = dale_sign.detach()     # 학습하지 않음, buffer로 등록
```

`(N_pre, 1)` 형태로 만들면 `(N_pre, N_post)` 가중치 행렬에 브로드캐스팅으로 곱해져서, 억제성 뉴런에서 **출발하는 모든 시냅스**가 음수가 된다.

### 4.3 계산량 문제 및 대응 전략

뉴런 N개일 때 θ 파라미터 수 = N² (잠재적으로 매우 큼)

### 대응 우선순위

1. **일단 시도**: 현대 GPU 성능을 믿고 전체 N²으로 실험
2. **안 될 경우 옵션들**:
    - Gumbel-Softmax 업데이트를 매 타임스텝이 아닌 epoch/batch 단위로 수행
    - 뉴런 수를 200~300개로 축소 (기존 LSM 논문도 128~512 사용)
    - 공간적 제약: 뉴런을 3D 격자에 배치, 일정 거리 이내만 연결 후보 (뇌 모방)
3. **이런 제약 자체가 ablation이 될 수 있음**

### 4.4 역전파 전략: BPTT + 마스크 고정 방식

### 핵심 설계 결정

순환 SNN의 학습에는 두 가지 별개의 미분 불가능성 문제가 있다:

| 문제 | 원인 | 해결 방법 |
| --- | --- | --- |
| 스파이크 함수의 미분 불가능성 | 발화/비발화가 이산적 | **Surrogate gradient** |
| 순환 연결의 시간 축 역전파 | 루프로 인해 과거 상태에 의존 | **BPTT** (본 연구에서 채택) |
| 연결 존재 여부의 이산성 | 있다/없다가 이산적 | **Gumbel-Softmax** |

이 세 가지는 독립적인 문제이며, 각각의 해결책을 조합하여 사용한다.

### θ(구조 파라미터)의 gradient 계산 방식

PGExplainer의 설계를 따라, **마스크를 시뮬레이션 전에 한 번 샘플링하고 시뮬레이션 동안 고정**하는 방식을 채택한다.

```
[시뮬레이션 전]
  m = GumbelSigmoid(θ, τ)    ← θ에서 마스크 한 번 생성 (N×N 행렬)

[시뮬레이션 중 — 각 타임스텝 t = 1, 2, ..., T]
  전달 신호(t) = m ⊙ W ⊙ 스파이크(t)    ← m은 고정, W와 스파이크만 변화
  LIF 동역학 계산 → 막전위 업데이트 → 스파이크 생성 (surrogate gradient)

[시뮬레이션 후]
  loss = CrossEntropy(리드아웃 출력, 정답 레이블)
  BPTT로 역전파:
    ∂L/∂W → 가중치 업데이트 (1000 타임스텝 unroll)
    ∂L/∂θ = ∂L/∂m × ∂m/∂θ → 구조 파라미터 업데이트
```

### 왜 이 방식이 작동하는가

- **m은 시간에 대해 상수**: 모든 타임스텝에서 동일한 마스크가 사용됨
- **∂L/∂m**: BPTT가 시간 축을 따라 역전파하면서 모든 타임스텝의 gradient가 m에 대해 합산
- **∂m/∂θ**: Gumbel-Sigmoid의 미분으로, 시간과 무관한 단일 값
- → θ 때문에 BPTT에 추가 메모리가 필요하지 않음
- → θ의 gradient는 PyTorch autograd에 의해 자동 계산됨

### BPTT의 메모리 부담과 대응

θ와 무관하게, BPTT 자체의 메모리가 핵심 병목:
- N=500 뉴런, T=1000 타임스텝이면 각 타임스텝의 활성화 값(막전위, 스파이크)을 저장해야 함
- 메모리 ∝ N × T

**대응 방법 (기존 SNN 연구에서 표준적으로 사용):**

1. **시간 binning**: SHD를 dt=10ms로 binning하여 T=100으로 축소 (기존 논문 다수에서 채택)
2. **Truncated BPTT**: 일정 윈도우(예: 200 스텝)만 unroll. 장기 의존성 일부 손실
3. **Gradient checkpointing**: 일부 타임스텝만 저장, 나머지는 필요 시 재계산. 메모리↓ 계산량↑
4. **FPTT (Forward Propagation Through Time)**: 과거 상태에 대한 의존 없이 즉시 업데이트하는 근사 방법

**전략**: dt=10ms binning (T=100)으로 시작하는 것이 가장 현실적. 이는 기존 SHD 연구들과도 직접 비교 가능한 세팅.

---

## 5. 데이터셋

### 5.1 주요 데이터셋: SHD (Spiking Heidelberg Digits)

| 항목 | 내용 |
| --- | --- |
| **태스크** | 음성 숫자 분류 (0~9, 영어+독일어 = 20클래스) |
| **데이터 형태** | 음성 → 인공 와우 모델(Lauscher) → 700채널 스파이크 트레인 |
| **입력 차원** | [num_steps × 700] |
| **샘플 길이** | 약 1초 |
| **데이터 크기** | Train: 8,156 / Test: 2,264 |
| **화자** | 12명 (2명은 테스트셋에만 존재 → 일반화 테스트) |
| **포맷** | HDF5 (spikes/times, spikes/units, labels) |
| **라이선스** | Creative Commons Attribution 4.0 |

### SHD 선택 이유

- SNN 커뮤니티 표준 벤치마크 (Papers with Code 리더보드 존재)
- LSM 논문에서 가장 활발히 사용 (ELSM, RLSM 등)
- **스파이크 타이밍 정보 활용이 필수** → rate coding만으로는 성능 불충분
- Feedforward SNN이 명확한 한계를 보임 → 순환 구조의 효과 입증에 적합
- 입력이 이미 스파이크 → 별도 인코딩 불필요
- 크기가 작아 빠른 이터레이션 가능

### 5.2 확장 데이터셋: SSC (Spiking Speech Commands)

| 항목 | 내용 |
| --- | --- |
| **태스크** | 35클래스 음성 명령 분류 |
| **데이터 크기** | Train: 75,466 / Valid: 9,981 / Test: 20,382 |
| **난이도** | SHD보다 훨씬 어려움, SNN이 ANN 대비 아직 열세 |

### 전략

- **SHD로 먼저 개발/검증** (작아서 빠름)
- **SSC로 스케일업하여 본 실험** (논문의 설득력 강화)

### 5.3 LSM 논문에서 사용된 기타 데이터셋

| 데이터셋 | 유형 | LSM 논문 사용 빈도 | 비고 |
| --- | --- | --- | --- |
| N-MNIST | 비전 (DVS 변환) | 높음 | 너무 쉬움 (97%+) |
| DVS128 Gesture | 비전 (이벤트 카메라) | 중간 | 제스처 인식 11클래스 |
| MNIST | 비전 | 높음 (전통적) | 너무 쉬움 |
| Fashion-MNIST | 비전 | 중간 |  |
| FSDD | 음성 | 낮음 | 음성 숫자 |
| TI-alpha / TI-10 | 음성 | 낮음 | 전통적 LSM 벤치마크 |

---

## 6. 논문 기여 (Expected Contributions)

### 6.1 핵심 기여

1. **리퀴드 가중치 학습의 효과 입증 (A→B)**: Surrogate gradient로 리퀴드 내부 가중치를 학습하면 전통 LSM(리드아웃만 학습) 대비 성능이 개선됨을 보임. 기존에 비표준적이던 접근을 체계적으로 검증.
2. *구조 학습의 추가적 가치 입증 (B*→C)**: 동일 희소성의 랜덤 구조(B*)와 비교하여, 학습된 구조가 “어디를 연결하느냐”에서 우위를 보임을 입증. 가장 공정한 비교.
3. **최적 연결 밀도의 자동 발견**: C가 자체적으로 수렴한 희소성 비율과, B에서 p sweep으로 찾은 최적 p를 비교. 일치하면 “Gumbel-Softmax가 최적 밀도까지 자동 발견”이라는 추가 기여.
4. **학습된 구조의 특성 분석**:
    - 희소성 비율 (전체 연결 중 몇 %가 살아남는가)
    - 흥분성/억제성 연결의 비율 변화 (80:20에서 시작 → 학습 후?)
    - 허브 뉴런 존재 여부
    - 루프 길이 분포
    - Small-world 특성 여부
5. **수렴 속도 및 안정성**: 랜덤 대비 학습 효율 개선 여부

### 6.1.1 현재 실험 기준 기여 업데이트

현재 SHD 실험 결과를 반영하면, 핵심 기여는 다음처럼 재정렬된다.

1. **랜덤 recurrence의 한계 확인**: no-recurrence `0.5490`, best low-density random recurrent `0.5499`, and full same-density random control batch mean `0.5257 ± 0.0084` / best single `0.5406`으로, 단순 recurrent density는 충분하지 않음.
2. **Gradient-based topology learning의 효과 확인**: Gumbel learned C, Grad R-STE, learned_lowrank 모두 random topology보다 강한 결과를 보임.
3. **Latent neuron role parameterization의 우위**: learned_lowrank는 no-freeze/fixed-freeze upper-bound 기준으로 강한 잠재력을 보였고, validation rollback `m50p10` 기준으로도 Grad R-STE + adaptive freeze보다 높은 mean/worst를 보임.
4. **Failure seed reversal**: edge-wise learned C에서 실패했던 seed44가 learned_lowrank에서 no-freeze `0.6444`, stabilized `0.6334`로 반전됨.
5. **Topology selection/freeze timing의 중요성**: seed45는 no-freeze/freeze72에서 `0.5751`까지 도달하므로 residual weak seed가 아니라 unstable peak case임. 다음 단계는 validation-based adaptive topology freeze/rollback.
6. **Validation-based adaptive freeze 검증 완료**: validation split과 topology snapshot/rollback이 구현되었고, `m50p10` 4-seed 결과는 test@best-val mean `0.5919`, worst `0.5826`이다. 이는 Grad R-STE + adaptive freeze보다 안정적이지만, no-freeze/fixed-freeze low-rank peak를 완전히 복구하지는 못한다. `m60p10`은 redundant로 판정됐고, `m60p15`는 mean/worst 기준 악화되어 reject한다.



### 6.2 흥미로운 추가 분석

### 흥분/억제 비율의 자기 조직화

- Gumbel-Softmax에게 “흥분성을 몇 % 남겨라”고 지시하지 않음
- 각 연결을 독립적으로 살릴지 죽일지만 학습
- 학습 후 살아남은 연결에서 흥분/억제 비율을 사후 분석
- 다양한 실험에서 일관되게 특정 비율로 수렴한다면 → 해당 태스크의 최적 균형 발견
- **뇌의 80:20 비율에 계산적 근거가 있는지**에 대한 통찰

### Optuna와의 결합 (Ablation)

- Dale’s Law를 따르지 않는 세팅에서 Optuna로 최적 흥분/억제 비율 탐색
- 탐색된 비율이 80:20과 얼마나 유사한지 비교
- 또는 Gumbel-Softmax가 자체적으로 찾은 비율과 Optuna 결과 비교

### E/I 균형의 자동 발견 가능성

기존 LSM 연구에서는 흥분성 뉴런끼리 루프를 형성하면 막전위가 지속적으로 상승하는 문제가 알려져 있다. 이를 방지하기 위해 기존에는 **사람이 규칙으로** 흥분성 연결이 존재하는 곳에 반드시 억제성 연결도 함께 만드는 처리를 수동으로 적용했다.

**Gumbel-Softmax가 이를 자동으로 발견할 수 있는 이론적 경로:**

```
1. 학습 초기: Gumbel-Softmax가 우연히 흥분성 뉴런끼리만 루프 형성
2. 결과: 스파이크가 루프를 돌며 막전위 폭주 → 특정 뉴런이 과도하게 발화
3. 영향: 입력과 무관하게 같은 패턴 출력 → 분류 성능 하락 → loss 상승
4. 학습 신호: gradient가 "이 흥분성 루프를 제거하거나 억제성 연결을 추가하라" 방향으로 θ 업데이트
5. 결과: 학습이 자동으로 흥분성 루프 주변에 억제성 연결을 배치
```

**안정성을 위한 안전장치:**
- Gumbel 온도가 높은 학습 초기에는 모든 연결이 0.5 근처의 부드러운 값
- 신호가 루프를 돌 때마다 0.5씩 곱해져서 자연 감쇠 → 폭주 방지
- 온도가 점차 낮아지며 이진화될 때쯤에는 위험한 루프가 이미 제거된 상태
- 추가로 뉴런 발화율 상한(firing rate clamp)이나 막전위 clamp을 안전장치로 설정

**분석 방법:**
- 학습 후 흥분성 루프 주변의 억제성 연결 밀도를 정량 분석
- 전체 네트워크에서 E→E 루프 대비 E→I→E 경로의 비율 시각화
- 이 패턴이 일관되게 나타나면: **“기존에 사람이 규칙으로 강제하던 E/I 균형을 Gumbel-Softmax가 학습을 통해 자동 발견한다”**는 강력한 발견

### 6.3 효율성 기여

- SHD SOTA가 96%+ → 성능 개선 여지가 좁을 수 있음
- **“같은 성능을 더 적은 연결로 달성”** 즉 효율성 측면의 기여도 준비

---

## 7. 논문 구성 (Proposed Structure)

### 메인 스토리

1. **Introduction**:
    - 큰 비전: 뇌는 interconnected, 기존 DNN/SNN은 단방향 → 이 괴리를 좁히고 싶다
    - 좁히기: LSM이 기존 시도인데 연결 구조가 랜덤 고정이라는 한계
    - 제안: 연결 구조 자체를 Gumbel-Softmax로 학습
2. **Related Work**:
    - SNN sparse training/rewiring (DEEP R, SET, Grad R, ESL-SNNs) — 규칙 기반 vs 본 연구의 미분 가능 접근
    - SNN NAS (SNASNet, SpikeDHS) — 매크로 구조 탐색 vs 마이크로 연결 학습
    - LSM 구조 최적화 (CMA-ES, EONS) — 진화적 탐색 vs end-to-end gradient
    - Gumbel-Softmax 기반 구조 학습 (PGExplainer, Gumbel-MPNN) — GNN/ANN 도메인, SNN 적용 없음
    - **포지셔닝**: “두 연구 흐름(SNN rewiring + Gumbel-Softmax 구조 학습)의 교차점이 탐색되지 않았다”
3. **Method**: Gumbel-Softmax 기반 리퀴드 구조 학습 프레임워크
4. **Experiments**:
    - 세 단계 비교: A(전통 LSM) vs B*(동일 희소성 랜덤) vs C(구조+가중치 학습)
    - A→B 점프, B*→C 점프 분리 분석
    - **D(Grad R) vs C**: 기존 gradient 기반 rewiring 대비 Gumbel-Sigmoid의 실증적 우위
    - Ablation: B에서 p sweep → 최적 p와 C의 자동 희소성 비교
    - 학습된 구조 분석: 희소성, E/I 비율, 허브 뉴런, 루프 특성
    - **LSNN+DEEP R과의 구조 패턴 비교** (리드아웃 근처 밀집 여부 등)
    - Ablation studies (초기화 σ, Dale’s Law 유무, 뉴런 수 등)
5. **Discussion**:
    - LSM을 넘어선 확장: 본 결과는 LSM에 국한되지 않고 interconnected SNN 전반에 적용 가능
    - 학습된 루프 구조와 적응적 처리 시간의 관계 (향후 연구)
    - 구조 학습에 의한 희소화가 gradient 경로를 단순화하여 학습 효율 향상 가능성
    - E/I 균형의 자동 발견 여부 분석
6. **Conclusion**: LSM에서의 검증 결과 요약 + 궁극적 비전 (interconnected SNN) 재제시

### Discussion에서 다룰 향후 연구 방향

### LSM을 넘어선 Interconnected SNN으로의 확장

본 논문에서는 LSM을 첫 번째 검증 무대로 삼아 Gumbel-Softmax 기반 구조 학습의 유효성을 입증한다. 그러나 이 방법론은 **LSM에 국한되지 않으며, interconnected SNN 전반에 적용 가능한 원리**이다.

```
현재 논문 (첫 번째 단계):
  LSM 리퀴드 내부 연결 구조 학습
  → 입력→리퀴드는 고정, 리퀴드→리드아웃도 기존 방식

다음 단계:
  입력→네트워크, 네트워크 내부, 네트워크→출력 모든 연결을 학습
  → "리퀴드"와 "리드아웃"의 경계가 사라짐
  → 뇌처럼 출력 뉴런이 네트워크에 완전히 통합된 구조

궁극적 비전:
  모든 뉴런이 interconnected된 SNN에서
  Gumbel-Softmax가 태스크에 최적인 연결 구조를 자동 발견
  → "어떤 구조의 신경망을 쓸 것인가"를 사람이 아닌 학습이 결정
```

이 확장이 실현되면, 현재 인공신경망에서 사람이 결정하는 구조적 선택들(레이어 수, 연결 방향, 순환 여부 등)이 학습 과정에 통합되어, 태스크에 따라 네트워크가 스스로 최적의 토폴로지를 찾게 된다.

### Gradient 경로 단순화를 통한 학습 효율 향상

본 논문에서는 BPTT를 사용하여 리퀴드 가중치와 구조를 학습한다. 그러나 BPTT는 타임스텝 수에 비례하는 메모리를 요구하므로, 더 긴 시퀀스나 더 큰 네트워크로 확장할 때 한계가 있다. 향후 RTRL이나 e-prop 같은 online learning 알고리즘으로의 전환을 고려할 수 있다.

**초기 가설과 비판적 검토:**

처음에는 “Gumbel-Softmax가 불필요한 연결을 제거하면 루프가 특정 서브그래프에만 존재하게 되어 unroll 범위가 줄어든다”고 가정했다. 그러나 비판적 검토 결과, **기존 랜덤 LSM에서도 희소 연결(p=0.1~0.3)이라 해도 루프가 네트워크 전체에 분포하는 것이 확인되었다.**

- 뉴런 N=500, 연결 확률 p=0.2이면 평균 차수 ≈ 100
- 랜덤 그래프 이론의 퍼콜레이션 임계값(p > 1/N)을 훨씬 초과
- → 거대 연결 성분(giant connected component) 형성
- → 거의 모든 뉴런이 하나의 연결 덩어리에 속하며, 루프가 전체에 퍼져 있음
- 기존 연구에서도 흥분성 뉴런끼리 루프를 형성하면 막전위가 폭주하는 문제가 보고되어, 이를 방지하기 위한 별도 규칙을 적용하고 있음 → 랜덤 구조에서도 루프가 광범위하다는 반증

따라서 “기존은 전체 루프, 우리는 서브그래프 루프”라는 이분법적 주장은 증명 없이 할 수 없다. Gumbel-Softmax가 학습한 구조도 충분히 밀집돼 있다면 마찬가지로 전체에 루프가 존재할 수 있다.

**수정된 주장 (향후 RTRL/e-prop 전환 시의 이점):**

본 논문에서는 BPTT를 사용하지만, 향후 더 큰 네트워크나 더 긴 시퀀스에서 RTRL/e-prop으로 전환할 경우, Gumbel-Softmax가 불필요한 연결을 제거하여 전체 연결 수가 줄어들고, 이에 따라 **루프의 수와 길이도 줄어든다**. 이는 RTRL/e-prop 적용 시 gradient가 지나가는 경로 자체를 단순화하여, **같은 알고리즘이라도 근사 오차가 줄어들 수 있다.** 또한 BPTT에서도 희소한 구조는 gradient vanishing/exploding 문제를 완화할 가능성이 있다.

```
기존 랜덤 LSM (연결 확률 p=0.2):
  루프 다수, 길이 다양 → gradient 경로 복잡 → 근사 오차 큼

Gumbel-Softmax LSM (학습된 희소 구조):
  불필요한 연결 제거 → 루프 수 감소, 평균 루프 길이 단축
  → gradient 경로 단순화 → 같은 RTRL/e-prop에서도 근사 오차 감소
```

이는 “서브그래프 vs 전체”라는 이분법이 아니라, 희소화에 따른 연속적인 개선이므로 더 방어 가능한 주장이다. **실험적 검증 방법**: 학습 전후의 루프 수, 평균 루프 길이, 연결 밀도를 정량적으로 비교.

### 적응적 타임스텝 (처리 깊이의 자율 결정)

- 루프 형성 시 신호가 순환하며 처리 깊이가 자연스럽게 결정
- LIF 뉴런의 leak으로 신호가 자연 감쇠 → 수렴 시점 자동 결정
    - 단순한 입력 → 빨리 수렴, 복잡한 입력 → 더 많이 순환
- 수렴 감지 방법: 리드아웃 뉴런 막전위 변화량이 임계값 이하
- Adaptive Computation Time / Deep Equilibrium Models과의 연결

### 기타 확장 방향

- **입력→리퀴드 연결까지 학습**: 현재는 리퀴드 내부만 학습 → 입력 연결도 학습하면 “리퀴드”와 “입력층”의 경계가 흐려짐
- **리드아웃을 네트워크에 통합**: 별도의 리드아웃 대신 효과기 뉴런(운동 뉴런) 방식으로 출력 → LSM의 3단 구조(입력-리퀴드-리드아웃)를 넘어 뇌에 더 가까운 통합 구조
- **BPTT → Online Learning 전환**: 현재는 BPTT를 사용하지만, 더 큰 규모에서는 RTRL/e-prop 같은 online 방법으로 전환 필요. 학습된 희소 구조가 이 전환의 비용을 줄여줄 가능성



### Validation rollback m50p10 completed result

Common setting:

```text
learned_lowrank r16
train_w_raw=false
w_raw_init_mean=-2.25
w_raw_max=-2.0
theta_init_mean=-1.0
theta_lr_scale=0.3
val_fraction=0.1
val_seed=42
topology_freeze_metric=val_acc
topology_freeze_min_epoch=50
topology_freeze_patience=10
topology_freeze_rollback_best=true
```

| Seed | Topology snapshot used for rollback | Freeze epoch | Best val epoch | Best val | Test @ best val | Final test | Oracle best test | Final density |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 58 | 68 | 81 | 0.6225 | 0.5857 | 0.5861 | 0.5875 | 0.041 |
| 43 | 54 | 64 | 82 | 0.6654 | 0.6135 | 0.6109 | 0.6197 | 0.050 |
| 44 | 65 | 75 | 65 | 0.6225 | 0.5857 | 0.5716 | 0.5919 | 0.047 |
| 45 | 62 | 72 | 92 | 0.6213 | 0.5826 | 0.5813 | 0.5972 | 0.043 |

Aggregate:

| Metric | Value |
|---|---:|
| Test @ best validation mean | **0.5919** |
| Test @ best validation median | **0.5857** |
| Worst seed | **0.5826** |
| Best seed | **0.6135** |
| Final test mean | 0.5875 |
| Oracle best-test mean | 0.5991 |
| Mean freeze epoch | 69.75 |
| Mean topology snapshot epoch | 59.75 |

Interpretation:

- This is the first **test-leakage-free** learned_lowrank topology-selection result.
- It beats Grad R-STE + adaptive freeze on mean and worst-seed stability: `0.5919` mean vs `0.5803`, and `0.5826` worst vs `0.5486`.
- It does not recover the full no-freeze/fixed-freeze low-rank upside: no-freeze mean `0.5999`, fixed freeze64/tau0.2 mean `0.5965`.
- The policy is therefore valid and useful; after `m60p10` and `m60p15`, it is also the final preferred validation-rollback schedule for the current protocol.
- The largest weakness is that seed44's previously observed high trajectory (`0.6444` no-freeze, `0.6334` stabilized) is not recovered.

Policy follow-up status:

```text
m60p10 = redundant; not distinct from m50p10
m60p15 = rejected; mean and worst seed degraded
main adaptive policy = m50p10
```

No further validation-freeze policy search is recommended. Use `m50p10` for main-table reporting and move to density controls / topology diagnostics.

### Main m50p10 command template

```bash
python scripts/train_lsm.py configs/lsm_shd_baseline.yaml \
  liquid.recurrent_mode=learned_lowrank \
  liquid.theta_rank=16 \
  liquid.theta_lowrank_init_std=0.30 \
  liquid.train_w_raw=false \
  liquid.w_raw_init_mean=-2.25 \
  liquid.w_raw_max=-2.0 \
  liquid.theta_init_mean=-1.0 \
  liquid.theta_lr_scale=0.3 \
  liquid.topology_adaptive_freeze=true \
  liquid.topology_freeze_metric=val_acc \
  liquid.topology_freeze_min_epoch=50 \
  liquid.topology_freeze_patience=10 \
  liquid.topology_freeze_min_delta=0.0 \
  liquid.topology_freeze_rollback_best=true \
  val_fraction=0.1 \
  val_seed=42 \
  seed=<SEED> \
  experiment_name=lsm_shd_lowrank_r16_valrollback_m50p10_s<SEED>
```

Completed seeds: `42, 43, 44, 45`.

---

## 8. 구현 상태와 계획

### 현재 코드베이스 반영 사항

LSM 확장은 현재 별도 모듈로 구현되어 있다.

| 구성 | 현재 파일 | 구현 상태 |
|------|-----------|----------|
| 데이터 로더 | `src/data/loaders.py` | MNIST, Fashion-MNIST, NMNIST, DVS Gesture, SHD 지원 |
| Feedforward SNN | `src/models/layers.py`, `src/models/snn.py` | 기존 Gumbel topology 실험 유지 |
| LSM 모델 | `src/lsm/model.py` | `InputProjection`, `LiquidLayer`, `LSMModel` 구현 |
| LSM 학습 | `src/lsm/trainer.py` | warmup + epoch-level Gumbel/STE + BPTT 구현 |
| LSM 설정 | `configs/lsm_shd_baseline.yaml` | SHD baseline 설정 |
| 실행 | `scripts/train_lsm.py`, `scripts/diagnose_liquid.py` | 학습/진단 CLI 구현 |

현재 LSM 구현은 초기 구상과 몇 가지 점이 다르다.

- 입력→리퀴드는 흥분성만 쓰지 않고 fixed sparse `randn` projection을 쓴다.
- 리퀴드 내부 mask는 타임스텝마다 샘플링하지 않고, forward 시작 시 한 번 정한다.
- learned mode의 Phase 2에서는 epoch-level Gumbel noise를 고정하고 batch마다 STE graph를 새로 계산한다.
- beta는 `logit_beta`로 저장해 `sigmoid(logit_beta)`가 의도한 초기 beta 범위가 되도록 했다.
- SHD baseline은 `N=500`, `bptt_truncate=25`, `theta_warmup_epochs=10`을 사용한다.
- gradient clipping은 weight/readout 계열과 theta 계열을 분리한다.

남은 주요 코드 정리 항목:

- `scripts/evaluate.py` 경로에서 LSM forward signature(`hard` 인자 없음)를 처리하도록 수정 필요
- SHD 직접 HDF5 fallback은 현재 미구현
- learned topology 분석(E/I degree, loop, clustering 등) 코드는 아직 추가 필요

### 기존 코드베이스 (순수 PyTorch, 이미 구현 완료)

Feedforward SNN에서 Gumbel-Softmax 구조 학습을 검증한 기존 코드가 이미 존재하며, 순수 PyTorch로 직접 구현되어 있다. 프레임워크 종속성이 없어 LSM 확장 시 완전한 유연성을 갖는다.

**기존 구현 구조:**

| 파일 | 핵심 구성 요소 |
| --- | --- |
| `layers.py` | `GumbelLIFLayer`: Gumbel-Sigmoid 마스크 생성, 4가지 topology mode (learned/full/random_sparse/transfer), surrogate gradient spike 함수, 학습 가능한 threshold·beta |
| `snn.py` | `SNNModel`: 임의 개수의 hidden layer 지원, 타임스텝 루프 내에서 LIF 동역학(막전위 누적→스파이크→리셋) 처리, sparsity/commitment loss, topology transfer 헬퍼 |

**기존 코드의 forward 루프:**

```python
for t in range(T):
    spike = rate_coded_input
    for layer in layers:            # feedforward 순서
        current = layer(spike, tau)  # mask * weight @ spike
        mem = beta * mem + current
        spike = spike_fn(mem - threshold)
        mem = mem * (1 - spike)
```

**이미 검증된 기능:**
- Gumbel-Sigmoid + temperature annealing + commitment loss → theta bimodal 분포 달성
- topology mode 전환 (learned/full/random_sparse/transfer) → 비교 실험 인프라
- surrogate gradient 역전파 → 가중치·threshold·theta 동시 학습
- 마스크는 배치 전체에 동일하게 적용 (theta가 `(n_pre, n_post)` shape, 배치 차원 없음)

**LSM 확장에서 반영된 핵심 수정 — 마스크 샘플링 타이밍:**

기존 feedforward 코드에서는 `layer.forward()`가 타임스텝 루프 안에서 매번 호출되며, 그때마다 `gumbel_sigmoid`가 새로운 Gumbel 노이즈를 샘플링한다. 즉 **마스크가 매 타임스텝마다 달라지는 구조**이다. Feedforward에서는 레이어를 순차 통과하므로 큰 문제가 아니었지만, LSM에서는 문서 섹션 4.4의 BPTT 전략에서 설계한 대로 **시뮬레이션 시작 전에 마스크를 한 번 생성하고 T 타임스텝 동안 고정**해야 한다.

```
기존 (feedforward):
  for t in T:
    for layer in layers:
      mask = gumbel_sigmoid(theta)  ← 매 타임스텝, 매 레이어마다 새 마스크

수정 (LSM):
  mask = gumbel_sigmoid(theta)      ← 시뮬레이션 전 1회 생성
  for t in T:
    current = (mask * eff_w) @ spike ← 고정된 마스크 사용
```

현재 LSM 코드는 `GumbelLIFLayer`를 직접 개조하지 않고, `src/lsm/model.py`의 `LiquidLayer.sample_mask(tau)`와 `current_mask`로 이 요구사항을 구현한다.

### LSM으로의 확장: 현재 구현

초기 구상은 기존 `GumbelLIFLayer`를 최대한 재활용하는 것이었지만, 현재 코드는 recurrent 안정화와 Dale's Law를 명확히 분리하기 위해 `src/lsm/model.py`에 LSM 전용 레이어를 둔다.

**핵심 구성 3가지:**

1. **입력→리퀴드 연결**: `InputProjection(700, N_liquid)` — fixed sparse mixed-sign random projection
2. **리퀴드 내부 순환 연결**: `LiquidLayer(N_liquid, mode=learned/random_sparse/fixed/grad_r)` — 핵심 학습 대상
3. **리드아웃**: `nn.Linear(N_liquid, 20)` — 타임스텝별 liquid spike를 누적한 뒤 평균

**LSM forward 루프 (수도코드):**

```python
# 초기화
dale_sign = torch.ones(N, 1)           # (N_pre, 1)
dale_sign[n_exc:, :] = -1.0            # 뒤 20%가 억제성

# 마스크 샘플링 — 타임스텝 루프 바깥에서 1회
recurrent_mask = recurrent_layer.sample_mask(tau)  # Gumbel-Sigmoid, (N, N)

for t in range(T):
    input_spike = shd_spikes[t]                           # SHD 스파이크 (이미 스파이크 형태)
    input_current = input_proj(input_spike)                # 700 → N, 고정
    recurrent_current = liquid_layer(liquid_spike)          # N → N, current_mask 사용
    # liquid_layer 내부: eff_w = current_mask * self_conn_mask * (dale_sign * F.softplus(clamped_w_raw))

    liquid_mem = beta * liquid_mem + input_current + recurrent_current
    liquid_spike = spike_fn(liquid_mem - threshold)
    liquid_mem = liquid_mem * (1 - liquid_spike)

    readout_mem += readout_layer(liquid_spike)              # N → 20

output = readout_mem / T
```

**기존 코드 대비 변경점:**
- `rate_coded_input` (MNIST 픽셀→포아송 스파이크) → SHD 스파이크 직접 입력 (인코딩 불필요)
- feedforward `for layer in layers` 루프 → 입력/순환/리드아웃 3개 연결을 명시적으로 분리
- `recurrent_current`에 이전 타임스텝의 `liquid_spike`를 사용 (순환)
- **마스크 샘플링 타이밍 수정**: 기존에는 `forward()` 호출마다 매 타임스텝 새 마스크 생성 → LSM에서는 `sample_mask(tau)` 메서드를 타임스텝 루프 바깥에서 1회 호출하고, 루프 내에서는 저장된 마스크를 사용
- Dale’s Law: `GumbelLIFLayer`의 forward에서 `eff_w = mask * (dale_sign * F.softplus(w_raw))` 적용
- `dale_sign`은 `(N_pre, 1)` 형태의 buffer, 흥분성 +1 / 억제성 -1
- softplus로 gradient 안정성 확보 (abs() 방식의 0 근처 불연속 회피)

**기존 코드에서 재활용되는 부분:**
- `SurrogateSpike` / `spike_fn`
- Gumbel/Sigmoid STE 아이디어
- sparsity/commitment loss 구조
- config inheritance와 CLI override 방식

### 확정 사항

| 구성 요소 | 설정 |
| --- | --- |
| 프레임워크 | **순수 PyTorch** (기존 코드 재활용, 프레임워크 종속성 없음) |
| 입력 | SHD 700채널 스파이크 (`tonic.datasets.SHD`) |
| 입력→리퀴드 연결 | `InputProjection`: fixed sparse `randn` projection |
| 리퀴드 내부 연결 | `LiquidLayer`: Gumbel/STE 또는 baseline mask |
| 리퀴드→리드아웃 | `nn.Linear(N, 20)` |
| 리퀴드 뉴런 | LIF (기존 `SurrogateSpike` 재활용), 80% 흥분 + 20% 억제 (Dale’s Law) |
| 역전파 방법 | **BPTT + 마스크 고정**, SHD baseline은 truncated BPTT |
| 시간 binning | **dt=10ms, T=100 타임스텝** |
| 데이터셋 | SHD (개발) → SSC (확장) |

### 미결정 사항 (실험으로 결정)

| 항목 | 후보 |
| --- | --- |
| 뉴런 수 | 200~500 (GPU 성능에 따라) |
| 초기화 σ | 0.01, 0.1, 0.5 등 탐색 |
| Gumbel 온도 스케줄링 | 기존 MNIST 실험의 스케줄 참조하여 조정 |
| 계산량 대응 | 필요 시 뉴런 수 축소 / 공간 제약 / gradient checkpointing |
| BPTT 메모리 대응 | Truncated BPTT / gradient checkpointing (필요 시) |

### 구현 Phase 계획 업데이트

### Phase 1: 핵심 검증 — 진행 중

**목표**: B vs C에서 유의미한 차이가 나는지 확인. 이것이 프로젝트 전체의 존폐를 결정.

LSM 모델, SHD 로더, 학습 루프는 구현되어 있으므로 현재 Phase 1의 중심은 실행 안정화와 baseline 비교다.

```
Step 1: 구현 sanity check
  → SHD loader shape: (batch, 100, 700)
  → LiquidLayer의 dale_sign/self_conn_mask/mode별 requires_grad 확인
  → train_lsm.py 1 epoch smoke test

Step 1.5: Wall-clock 시간 측정
  → N=200, batch_size=64로 1에폭 시간 측정
  → 1분 이내 → 문제 없음, 그대로 진행
  → 5~10분 → 감당 가능, 하이퍼파라미터 탐색 범위 축소 고려
  → 30분 이상 → 뉴런 수 축소 또는 truncated BPTT 즉시 적용
  → 이 측정으로 N=500 사용 가능 여부가 결정됨

Step 2: B (mode="random_sparse") 학습
  → liquid.recurrent_mode="random_sparse", p sweep
  → BPTT + surrogate gradient 파이프라인이 순환 구조에서 작동하는지 검증
  → SHD에서 합리적인 정확도가 나오는지 확인

Step 3: C (mode="learned") 학습
  → warmup 이후 theta unfreeze 확인
  → epoch-level noise + STE가 안정적으로 동작하는지 확인
  → theta 분산, sparsity, firing rate 추적

Step 4: B vs C 비교
  → 유의미한 차이 있음 → Phase 2로 진행 ✓
  → 차이 없음 → 원인 분석 후 방향 재검토 ✗
```

**Phase 1이 실패할 경우의 대응:**
- θ 초기화, 온도 스케줄, 학습률 조정 등 하이퍼파라미터 재탐색
- sparsity/commitment loss 가중치 조정
- 뉴런 수 변경
- 그래도 차이 없으면: “가중치 학습으로 충분하다”는 것 자체가 발견 (negative result 논문 가능성)

### Phase 2: Baseline 강화 (1~2주)

**목표**: 리뷰어 공격에 대한 방어벽 구축

```
Step 4: A (전통 LSM, 리퀴드 고정) 추가
  → A→B 점프 측정

Step 5: B* (C와 동일 희소성의 랜덤 구조) 추가
  → B*→C 비교: 가장 순수한 "어디를 연결했느냐" 비교

Step 6: D (Grad R) 추가
  → GumbelLIFLayer에 mode="grad_r" 추가 (hard threshold: θ>0이면 연결)
  → C 코드와 거의 동일, gumbel_sigmoid 대신 (theta > 0).float() 사용
  → D vs C: 기존 gradient rewiring 대비 우위 실증

Step 7: B에서 p sweep (0.1, 0.2, 0.3, 0.5)
  → 최적 p 탐색, C의 자동 희소성과 비교
```

### Phase 3: 분석 및 확장 (2~3주)

**목표**: 논문의 깊이와 범위 확장

```
Step 8: 학습된 구조 분석
  - 희소성 비율, E/I 연결 비율
  - 허브 뉴런, 루프 특성, small-world 여부
  - LSNN+DEEP R 패턴과 비교 (리드아웃 근처 밀집 여부)
  - E/I 균형 자동 발견 여부

Step 9: SSC 확장
  → 더 어려운 태스크에서도 구조 학습의 효과 확인

Step 10: 나머지 Ablation
  - 초기화 σ 변화
  - Dale's Law 유무
  - 뉴런 수 변화
  - 온도 스케줄 변화
```

### 프레임워크: 순수 PyTorch (확정)

기존 feedforward SNN 코드가 순수 PyTorch로 구현되어 있으며, `GumbelLIFLayer`가 연결 행렬에 마스크를 곱하는 커스텀 로직을 직접 제어한다. 외부 SNN 프레임워크는 recurrent 연결 행렬을 추상화해버릴 수 있어 마스크 끼워넣기가 어려울 수 있으므로, **순수 PyTorch를 유지하는 것이 최적**이다. SHD 데이터 로딩에만 Tonic 또는 snnTorch의 spikedata 모듈을 활용.

---

## 9. 리스크 및 대응

| 리스크 | 대응 |
| --- | --- |
| SHD SOTA 96%+ 로 성능 개선 여지 좁음 | 효율성 (적은 연결로 같은 성능) 기여도 함께 준비 |
| B→C 점프가 미미할 가능성 | 넓은 프레이밍 적용 (A→B 자체도 기여), 효율성/구조 분석으로 가치 보완 |
| BPTT 메모리 부담 (N×T 활성화 저장) | dt=10ms binning (T=100), gradient checkpointing, truncated BPTT |
| N² 파라미터로 메모리/계산량 과다 | 뉴런 수 축소, 배치 단위 구조 업데이트, 공간 제약 |
| 학습된 구조가 랜덤과 크게 다르지 않을 가능성 | 구조 분석(네트워크 특성)을 통해 차이 시각화 |
| Gumbel-Softmax + Surrogate gradient 결합의 학습 불안정성 | 온도 스케줄링, 학습률 조절 등 하이퍼파라미터 튜닝 |
| 흥분성 루프 형성에 의한 막전위 폭주 | Gumbel 고온 초기화로 자연 감쇠, 발화율 clamp / 막전위 clamp 안전장치 |
| Grad R 등 기존 rewiring 방법과의 차별화 질문 | Gumbel-Sigmoid의 연속 최적화 vs hard threshold, 양방향 가능성, temperature annealing 장점 강조 |
| Phase 1에서 B vs C 차이 미미 | 하이퍼파라미터 재탐색 → 그래도 안 되면 negative result 논문 또는 방향 전환 |
| 순환 구조의 낮은 GPU 활용 효율 | 타임스텝이 순차적이라 feedforward 대비 느림. Phase 1 첫날 wall-clock 측정으로 N 범위 결정 |

---

## 10. 핵심 참고문헌

### 기초 방법론

- Jang et al. (2017) — Categorical Reparameterization with Gumbel-Softmax (ICLR)
- Maddison et al. (2017) — The Concrete Distribution (ICLR)
- Luo et al. (2020) — PGExplainer: Parameterized Explainer for GNN (NeurIPS)
- Neftci et al. (2019) — Surrogate Gradient Learning in SNNs (IEEE Signal Processing)

### 가장 가까운 경쟁 연구 (Sparse Training / Rewiring)

- Bellec et al. (2018) — DEEP R: Training very sparse deep networks (ICLR)
- Mocanu et al. (2018) — SET: Adaptive sparse connectivity (Nature Communications)
- Evci et al. (2021) — RigL: Making All Tickets Winners (ICML)
- Chen et al. (2021) — Grad R: Gradient Rewiring for SNN pruning (IJCAI)
- Shen et al. (2023) — ESL-SNNs: Evolutionary Structure Learning (AAAI)
- Mészáros et al. (2024) — Learning Delays Through Gradients and Structure (Frontiers)

### SNN NAS

- Kim et al. (2022) — SNASNet: Neural Architecture Search for SNN (ECCV)
- Che et al. (2022) — SpikeDHS: Differentiable Architecture Search for SNN

### LSM / Reservoir 구조 최적화

- Maass et al. (2002) — LSM 원논문 (Neural Computation)
- Zhou et al. (2019) — Evolutionary Optimization of LSMs (ISNN)
- Plank et al. (2019) — EONS: Intelligent Reservoir Generation (ICONS)
- Wang et al. (2023) — Adaptive structure evolution for recurrent SNN (Scientific Reports)

### Recurrent SNN 학습

- Bellec et al. (2020) — e-prop: Learning dilemma for recurrent SNN (Nature Communications)
- Bellec et al. (2018b) — LSNN + DEEP R (NeurIPS)
- Cramer et al. (2020) — Heidelberg Spiking Datasets: SHD, SSC (IEEE TNNLS)
