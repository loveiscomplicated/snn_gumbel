# Research Memo v0.5 — Vision-aligned update

> **Update 2026-05-17 — vision alignment / research_vision_roadmap_v0.2**  
> This document is now aligned to the project-level roadmap: the long-term target is a **modular cognitive system** whose core is **LSNN + topology learning + SSM**, and whose expression path is **adapter + decoder**. SHD/LSM topology learning is Phase A: it validates whether learned recurrent topology can create useful dynamics, but it is not the final architecture. The forward roadmap is **Phase B: ALIF 이식 → Phase C: e-prop 구현 → Phase D: 자연어 태스크, GPT-2 distillation, decoder 연결, SSM 탐색**. Biological inspiration remains an existence proof only; implementation choices are judged engineering-first.


> **Update 2026-05-17 — vision alignment / v0.5**  
> This memo remains the Phase A evidence ledger for SHD/LSM topology learning. It is now explicitly subordinate to `research_vision_roadmap_v0.2`: Phase A validates topology-driven dynamics; Phase B is ALIF integration; Phase C is e-prop for long sequences; Phase D is NLP via GPT-2 hidden-state distillation, adapter-decoder connection, and SSM exploration. Claims about cognitive SNN/local-LLM replacement remain long-term roadmap hypotheses, not conclusions from the SHD/LSM result.

> **Update 2026-05-17 — related-work / novelty repositioning**  
> A new literature pass narrows the novelty claim. Broad claims that *recurrent SNN topology learning*, *sparse rewiring*, or *LSM liquid-structure optimization* are novel are no longer safe: e-prop/LSNN/DEEP R, Grad R, ESL-SNNs, dynamic pruning with DEEP R+RigL on Heidelberg-style speech data, adaptive/evolutionary LSMs, EONS, and low-rank recurrent network theory already cover major parts of that space. The defensible contribution is narrower: in an SHD LSM setting, compare **edge-wise Gumbel/STE**, **Grad R-STE**, **learned_lowrank latent source/destination role parameterization**, and **validation-based topology selection** to test whether gains come from recurrent density, edge placement, topology parameterization, or freeze timing.

> **Claim hygiene rule**  
> Do not write: “first SNN topology learning method”, “first recurrent SNN structure learning method”, or “e-prop/LSNN did not address topology learning.” Safe wording: “This project studies topology *parameterization and selection* in recurrent SNN/LSM topology learning, building on prior sparse rewiring, LSNN/e-prop, and LSM structure-optimization work.”

> **Update 2026-05-17 — internal documentation lock / paper deferral**  
> The current findings are now locked as an internal research asset rather than a paper-ready final claim. There is no publication-pressure assumption for this project. The purpose of the current documents is to preserve the corrected related-work framing, the validated experimental facts, the claim-evidence boundary, and the next diagnostic roadmap so later experiments can build from a stable base. Paper writing is deferred until a visibly stronger result appears: broader seed/dataset robustness, causal mechanism evidence, readout/temporal separability evidence, successful ALIF/e-prop transfer, or a task where learned recurrent topology shows a clear advantage over structured alternatives.

> **Operating rule after this lock**  
> Treat the present result as a checkpoint in a longer research program. Do not inflate it into an external novelty claim. Use it to guide the next experiments: topology diagnostics, activity/readout diagnostics, causal graph interventions, and then Phase B/C/D extensions: ALIF integration, e-prop for long sequences, and SSM/NLP adapter-decoder experiments.

> **Update 2026-05-17 — diagnostics integration / v0.3**  
> The memo now incorporates the expanded performance and graph-diagnostic batch covering `R1`, `R2`, `R2v`, `R4`, `R4v`, and `R8`. The main result remains `learned_lowrank r16 + validation rollback m50p10` (`R8`) with mean test@best-val `0.5919`, median `0.5857`, worst `0.5826`, and best `0.6135`. The new diagnostics strengthen the claim that the `learned_lowrank` advantage is not a density effect: `R8` has nearly the same density as same-density random controls (`0.0455` vs `0.0451`) but much higher degree concentration, reciprocity, 3-cycle density, clustering, and more selective reachability. Edge-wise learned C (`R2/R2v`) is now included and remains weaker than `R8` in performance and graph organization.
>
> **Activity-diagnostics caveat**  
> The latest topology-diagnostics metadata still marks the integrated `activity_metrics.csv` path as placeholder-only. Therefore, formal activity claims in this memo use the separately generated `activity_diagnostics.csv` / `activity_group_summary.csv`, which currently cover `R1`, `R4`, `R4v`, and `R8`, but not `R2/R2v`. Activity rows are diagnostic observations, not independent seeds.


## 스파이킹 리퀴드 네트워크에서 잠재 뉴런 역할 기반 순환 토폴로지 학습

> **Update 2026-05-17 — internal lock / v0.4**  
> This memo now locks the current findings as internal research documentation. The project will not be pushed toward a paper until a stronger result or clearer mechanism evidence appears. The current role of the memo is to preserve claim hygiene, main-result semantics, diagnostic gaps, and paper-readiness triggers.

## 0. 한 문장 요약

본 연구는 스파이킹 순환 네트워크에서 리퀴드 내부 연결을 고정된 랜덤 구조로 두는 기존 LSM 방식의 한계를 검토하고, recurrent topology 자체를 학습 가능한 대상으로 바꾸었을 때 성능과 안정성이 어떻게 달라지는지를 분석한다. 현재 가장 방어 가능한 결론은 단순 recurrent density가 아니라 **어떤 뉴런이 어떤 뉴런과 연결되는가**, 더 구체적으로는 **edge-wise 독립 파라미터보다 latent neuron role 기반 parameterization이 더 강한 inductive bias를 제공할 수 있으며, 이 차이가 graph organization과 high-recurrent activity regime의 차이로 관찰된다**는 것이다.

---

## 1. 연구 질문

이 연구의 표면적 질문은 “SNN에서 연결 구조를 학습할 수 있는가?”이다. 그러나 실제 핵심 질문은 더 좁고 구조적이다.

> 순환 SNN에서 성능을 좌우하는 것은 recurrent 연결의 양인가, 아니면 recurrent 연결의 배치와 구조인가?

이 질문은 다음 세부 질문으로 분해된다.

1. 고정 랜덤 recurrent topology는 no-recurrence baseline보다 실제로 나은가?
2. 같은 연결 밀도에서 random topology와 learned topology는 다른 성능을 보이는가?
3. edge마다 독립적인 topology parameter를 두는 방식은 seed instability를 유발하는가?
4. source/destination latent neuron role을 통해 edge logit field를 생성하면 topology learning이 더 안정화되는가?
5. 학습 중 발견한 좋은 topology를 언제 고정해야 일반화 성능을 보존할 수 있는가?

이 중 현재 결과로 가장 강하게 답할 수 있는 것은 1, 2, 5이며, 3과 4도 R2/R2v 및 R8 graph diagnostics가 들어오면서 중간~강한 수준까지 올라왔다. 다만 edge-wise learned C와 learned_lowrank의 activity regime 차이는 아직 R2/R2v activity diagnostics가 없어 추가 보완이 필요하다.

---

## 2. 연구 배경과 문제의식

Liquid State Machine은 입력 스파이크를 순환적으로 연결된 spiking reservoir에 투입하고, 그 resulting liquid state를 readout으로 분류하는 구조다. 전통적 LSM에서 핵심 가정은 리퀴드 내부 연결을 랜덤하게 고정해도 충분히 풍부한 시공간 동역학이 생성된다는 것이다.

그러나 이 가정에는 두 가지 한계가 있다.

첫째, 랜덤 recurrent topology가 항상 유용한 representation을 만든다는 보장이 없다. 특히 SHD처럼 시간적 정보와 class separation이 중요한 데이터에서는 무작위 순환 연결이 입력 정보를 보존하기보다 흐릴 수 있다.

둘째, 성능 개선을 위해 reservoir 크기나 연결 밀도를 조정하는 방식은 구조적 설명력이 약하다. 연결을 더 많이 만들거나 뉴런 수를 늘리는 것은 “어떤 계산 구조가 필요한가”에 대한 답이 아니다.

따라서 본 연구는 LSM의 리퀴드 내부 연결을 고정된 랜덤 구조가 아니라 학습 가능한 topology로 본다. 핵심 전환은 다음과 같다.

```text
기존 LSM:
  recurrent topology = random fixed prior
  readout만 학습

본 연구의 방향:
  recurrent topology = trainable object
  topology, dynamics, readout을 함께 조정
```

다만 이 연구의 목표는 단순히 “Gumbel trick을 SNN에 적용했다”가 아니다. 현재까지의 실험은 연구 질문을 더 정교하게 바꾸었다.

```text
초기 질문:
  Gumbel-Sigmoid로 SNN topology를 학습할 수 있는가?

현재 질문:
  recurrent SNN에서 edge-wise topology search와 latent role-based topology search는 어떻게 다른가?

더 구체적인 질문:
  seed instability는 topology parameterization의 문제인가, topology selection timing의 문제인가?
```

---

## 3. 방법론 요약

### 3.1 기본 구조

현재 실험은 SHD 기반 LSM 구조를 사용한다.

```text
SHD spike input
  → fixed sparse input projection
  → recurrent liquid layer
  → linear readout
```

입력에서 리퀴드로 가는 projection은 고정한다. 학습의 중심은 리퀴드 내부 recurrent topology다. 리퀴드 내부에는 Dale's Law 기반 흥분성/억제성 부호 구조가 적용되며, recurrent weight magnitude와 topology mask가 결합되어 effective recurrent matrix를 구성한다.

### 3.2 비교한 topology 조건

현재 핵심 비교군은 다음과 같다.

| 조건 | 의미 | 현재 역할 |
|---|---|---|
| no recurrence | recurrent 연결 없음 | 최소 기준선 |
| random_sparse | 고정 랜덤 recurrent topology | random recurrence baseline |
| same-density random_sparse | learned_lowrank와 비슷한 density의 랜덤 구조 | density-only control |
| learned C | edge-wise dense theta + Gumbel/STE | 초기 제안 방식 |
| Grad R-STE | hard threshold topology learner | 강한 deterministic baseline |
| learned_lowrank | source/destination latent embedding 기반 topology logits | 현재 중심 후보 |
| learned_lowrank + validation rollback | validation 기준 topology snapshot/rollback | 현재 main-table 후보 |

### 3.3 Learned low-rank / latent neuron role topology

`learned_lowrank`는 edge마다 독립적인 `theta_ij`를 두지 않는다. 대신 각 뉴런에 source embedding과 destination embedding을 두고, edge logit을 다음처럼 생성한다.

```text
topology_logit_ij = src_embed_i · dst_embed_j + theta_bias
```

이 방식의 핵심은 edge를 독립 객체로 보지 않는다는 점이다. 한 뉴런의 source role이 바뀌면 그 뉴런에서 나가는 여러 edge logits가 함께 바뀌고, destination role이 바뀌면 그 뉴런으로 들어오는 여러 edge logits가 함께 바뀐다.

따라서 learned_lowrank는 단순한 파라미터 감소 기법이 아니다. 더 중요한 의미는 **edge decisions를 neuron role interaction으로 묶는 구조적 inductive bias**다.

이 해석은 현재 연구의 중심이다.

---

## 4. 현재까지 확정된 결과

### 4.1 Random recurrence는 충분하지 않다

no-recurrence baseline은 test accuracy `0.5490`으로 고정되어 있다. 반면 same-density random control은 `p ∈ {0.040, 0.045, 0.050}`와 seeds `42/43/44/45`의 12개 run에서 평균 test@best-val `0.5257`, median `0.5254`, worst `0.5133`, best single `0.5406`에 그쳤다.

이 결과는 단순히 recurrent edge가 존재하거나 learned topology와 유사한 density를 맞추는 것만으로는 성능이 오르지 않는다는 점을 보여준다. 현재 설정에서는 density-matched random recurrence가 no-recurrence baseline보다도 낮다.

따라서 다음 주장은 강하게 방어 가능하다.

> Recurrent density 자체는 `learned_lowrank` 성능 향상의 충분한 설명이 아니다.

다만 이 주장은 density-only explanation을 배제하는 것이지, 그 자체로 특정 회로 메커니즘을 입증하는 것은 아니다. 메커니즘 주장은 graph diagnostics와 activity diagnostics를 함께 봐야 한다.

### 4.2 Edge-wise learned C는 개선되지만, learned_lowrank에는 미치지 못한다

이번 업데이트에서 edge-wise learned C의 두 버전이 main summary에 포함되었다.

| ID | Method | Selection rule | Mean | Median | Worst | Best | 해석 |
|---:|---|---|---:|---:|---:|---:|---|
| R2 | learned C original | historical best-test/no-val | `0.5590` | `0.5638` | `0.5331` | `0.5751` | edge-wise Gumbel/STE, seed-sensitive |
| R2v | learned C valrollback | test@best-val + val rollback | `0.5603` | `0.5663` | `0.5349` | `0.5738` | validation rollback을 붙여도 큰 개선 없음 |
| R8 | learned_lowrank m50p10 | test@best-val + val rollback | **`0.5919`** | **`0.5857`** | **`0.5826`** | **`0.6135`** | current main result |

핵심은 `R2v`가 들어온 뒤에도 `R8`의 mean, worst, best가 모두 더 높다는 점이다. 따라서 이제 다음 표현은 가능하다.

> 현재 SHD-LSM validation protocol에서는 edge-wise independent topology parameterization보다 latent source/destination role 기반 `learned_lowrank`가 더 강하고 안정적인 후보로 나타났다.

단, 이 결론은 SHD-LSM, seeds 42/43/44/45, 현재 validation split과 training recipe에 한정된다. “항상 우월하다”는 일반 명제는 아직 주장하지 않는다.

### 4.3 Validation rollback m50p10이 현재 가장 공정한 main policy다

validation-based adaptive topology freeze policy 중 현재 main candidate는 `m50p10`이다.

```text
topology_freeze_min_epoch = 50
topology_freeze_patience = 10
topology_freeze_metric = val_acc
topology_freeze_rollback_best = true
```

이 정책은 `learned_lowrank`에서 test@best-val mean `0.5919`, median `0.5857`, worst `0.5826`을 기록했다. 반면 `m60p10`은 실질적으로 `m50p10`과 구분되는 정책이 아니었고, `m60p15`는 mean과 worst seed generalization을 악화시켜 reject되었다.

따라서 현재 논문형 보고서에서 사용할 수 있는 가장 안전한 성능 주장은 다음이다.

> `learned_lowrank + validation rollback m50p10`은 현재까지 검증된 test-leakage-free adaptive policy 중 가장 방어 가능한 main result다.

주의할 점은 이것이 absolute peak가 아니라는 점이다. no-freeze 또는 manually selected freeze timing에서는 더 높은 peak가 있었지만, 그것을 main result로 쓰면 post-hoc selection 문제가 생긴다.

### 4.4 Graph diagnostics: learned_lowrank는 density가 아니라 graph organization을 바꾼다

Graph diagnostics는 이제 `R1`, `R2`, `R2v`, `R4`, `R4v`, `R8`까지 포함한다. 가장 중요한 비교는 `R1` same-density random, `R2v` edge-wise learned C valrollback, `R8` learned_lowrank valrollback이다.

| 지표 | R1 random | R2v learned C | R8 learned_lowrank | 해석 |
|---|---:|---:|---:|---|
| density | `0.0451` | `0.0536` | `0.0455` | R8은 R1과 거의 같은 density, R2v보다 낮은 density |
| in-degree Gini | `0.1163` | `0.2205` | **`0.6127`** | R8은 incoming hub/role 집중이 매우 큼 |
| out-degree Gini | `0.1150` | `0.2911` | **`0.5758`** | R8은 outgoing role 집중도 큼 |
| reciprocity | `0.0231` | `0.0557` | **`0.0817`** | R8은 양방향 recurrent pair가 더 많음 |
| 3-cycle density | `0.000031` | `0.000099` | **`0.000157`** | R8은 short recurrent motif가 더 많음 |
| clustering | `0.0879` | `0.1590` | **`0.3719`** | R8은 local clustered circuit 구조가 강함 |
| largest SCC | `500.0` | `496.5` | **`396.75`** | R8은 전체 강연결 graph가 아니라 부분 구조화됨 |
| reachability ratio | `1.0000` | `0.9930` | **`0.7899`** | R8은 global mixing보다 선택적 propagation |
| effective diameter p90 | `3.0` | `3.0` | **`4.0`** | R8은 더 긴 경로/계층적 전달 구조 |

이 결과는 `learned_lowrank`의 이점이 recurrent edge 수에서 나온 것이 아님을 뒷받침한다. `R8`은 `R1`과 거의 같은 density를 유지하면서도 degree concentration, reciprocity, short cycle, clustering이 크게 증가한다. 또한 edge-wise learned C(`R2/R2v`)는 random보다 구조화되지만, `R8`만큼 강한 hub-like / clustered / partially modular structure를 만들지는 못한다.

안전한 해석은 다음이다.

> `learned_lowrank`는 sparse density 안에서 hub-like, clustered, partially modular recurrent topology를 형성한다. 이는 edge-wise learned C나 same-density random topology와 구별되는 graph organization이며, 현재 성능 차이와 함께 관찰된다.

그러나 이 단계에서 “특정 계산 회로를 발견했다”고 쓰면 과하다. graph metric은 구조 차이를 보여주지만, 그 구조가 정확히 어떤 계산을 수행하는지는 아직 별도 분석이 필요하다.

### 4.5 Activity diagnostics: high-recurrent regime은 확인되지만, R2/R2v activity는 아직 없다

Activity diagnostics는 별도 `activity_diagnostics.csv` / `activity_group_summary.csv` 기준으로 반영한다. 이 파일은 현재 `R1`, `R4`, `R4v`, `R8`을 포함하지만, `R2/R2v` edge-wise learned C의 activity diagnostics는 아직 없다. 또한 activity table의 `n=8` 또는 `n=24`는 independent seed 수가 아니라 diagnostic observation 수다.

| 지표 | R1 random | R4 Grad R historical | R4v Grad R valrollback | R8 learned_lowrank |
|---|---:|---:|---:|---:|
| mean firing rate | `0.0475` | **`0.2199`** | `0.0455` | `0.1975` |
| max firing rate | `0.7475` | **`0.9925`** | `0.7600` | **`0.9925`** |
| dead neurons | `81.25` | `1.00` | `42.75` | **`0.25`** |
| active neurons > 0.05 | `130.6` | **`385.8`** | `135.0` | `349.0` |
| overactive neurons > 0.20 | `33.5` | **`204.5`** | `23.25` | `145.75` |
| recurrent/input ratio | `0.1485` | `1.0363` | `0.0846` | **`1.3715`** |
| class mean cosine | `0.9737` | `0.9680` | `0.9681` | **`0.9882`** |
| class cosine min | `0.9570` | `0.9213` | `0.9484` | **`0.9781`** |

해석은 두 단계로 나눠야 한다.

첫째, `R8`은 low-activity random regime이 아니다. dead neuron이 거의 사라지고, active neuron coverage와 recurrent/input ratio가 크게 증가한다. 따라서 `learned_lowrank`는 sparse graph이면서도 강한 recurrent amplification을 동반하는 high-activity liquid regime을 만든다.

둘째, class mean-rate cosine은 `R8`에서 오히려 더 높다. 따라서 “R8이 평균 firing-rate vector의 class separation을 개선했다”고 말하면 안 된다. 현재 activity evidence가 지지하는 주장은 다음 정도다.

> `learned_lowrank`는 높은 recurrent/input ratio와 넓은 active neuron coverage를 갖는 high-recurrent activity regime을 형성한다. 그러나 raw class mean-rate cosine 기준으로 class separation이 좋아졌다고 볼 수는 없다. 성능 향상의 downstream explanation은 readout margin, temporal pattern, sample-level trajectory 분석으로 추가 확인해야 한다.

### 4.6 R4 vs R4v: selection rule이 activity regime을 크게 바꾼다

`R4` historical Grad R와 `R4v` validation rollback Grad R의 차이는 topology selection timing의 중요성을 보여준다.

| Group | Selection | Mean test | Mean firing | Active > 0.05 | Rec/input |
|---|---|---:|---:|---:|---:|
| R4 | historical best-test/no-val | `0.5803` | `0.2199` | `385.8` | `1.0363` |
| R4v | test@best-val + val rollback | `0.5330` | `0.0455` | `135.0` | `0.0846` |

`R4`는 high-activity recurrent regime에서 강한 성능을 보였지만, `R4v`는 low-activity regime으로 떨어지면서 same-density random control 수준에 가까워졌다. 이는 topology learner 자체뿐 아니라 **어떤 시점의 topology를 선택하고 rollback하는가**가 성능과 activity regime을 크게 바꿀 수 있음을 보여준다.

---

## 5. 현재 주장 가능한 것과 아직 주장하면 안 되는 것

### 5.1 현재 주장 가능한 것

| 주장 | 근거 수준 | 판단 |
|---|---:|---|
| Random recurrent density만으로는 성능 향상을 설명할 수 없다 | 강함 | same-density random control이 no-recurrence보다도 낮음 |
| `learned_lowrank + m50p10`은 현재 가장 방어 가능한 main result다 | 강함 | R8 mean/worst가 R1/R2/R2v/R4v보다 높음 |
| edge-wise learned C보다 latent role parameterization이 더 안정적인 후보일 수 있다 | 중간~강함 | R2/R2v 대비 R8의 성능 및 graph organization 우위 |
| learned topology가 random과 다른 graph organization을 만든다 | 강함에 가까움 | R8의 Gini, reciprocity, 3-cycle, clustering, reachability 차이 |
| topology selection timing은 핵심 문제다 | 강함 | R4 vs R4v의 performance/activity regime 차이 |
| learned_lowrank는 high-recurrent activity regime을 만든다 | 중간~강함 | R8의 active coverage, rec/input ratio 증가 |
| ALIF/e-prop/SSM language-path work은 Phase B/C/D로 분리해 진행해야 한다 | 강함 | Phase A claim 방어와 혼합하지 않되, 공식 로드맵의 다음 architecture 축이다 |

### 5.2 아직 주장하면 안 되는 것

| 주장 | 왜 아직 위험한가 | 필요한 추가 증거 |
|---|---|---|
| learned_lowrank가 생물학적으로 더 타당하다 | graph motif만으로 생물학적 타당성 부족 | E/I motif, role specialization, biological prior와의 대응 |
| latent role topology가 항상 edge-wise theta보다 우수하다 | seed 수와 task가 제한적 | 더 많은 seeds, SHD 외 task, 다른 split |
| 학습된 topology가 특정 계산 회로를 발견했다 | graph statistics는 회로 기능을 직접 설명하지 않음 | motif-level ablation, edge removal, causal intervention |
| learned_lowrank가 class separation을 rate-vector 수준에서 개선했다 | R8 class mean cosine이 오히려 높음 | readout margin, trajectory separation, temporal coding 분석 |
| e-prop으로 가면 바로 확장성이 해결된다 | 아직 구현·실험 없음 | BPTT vs e-prop 비교 실험 |
| 현재 SHD/LSM 결과만으로 cognitive SNN 또는 local LLM 대체 가능성이 입증됐다 | 현재 실험 범위 초과 | distillation, multiple-choice, decoder/interface, scaling evidence 필요 |

---

## 6. 해석: 현재 결과가 의미하는 것

현재 결과의 핵심은 “recurrence가 있으면 좋다”가 아니다. 오히려 random recurrence는 현재 설정에서 성능을 악화시킬 수 있다. 더 정확한 해석은 다음이다.

```text
1. Recurrent connection 자체는 충분하지 않다.
2. 같은 density라도 random placement는 성능을 설명하지 못한다.
3. Edge-wise learned C는 random보다 개선되지만 seed-sensitive하고 구조 집중도가 제한적이다.
4. Latent source/destination role 기반 learned_lowrank는 edge decisions를 공유 구조 안에 묶어 hub-like, clustered, partially modular topology를 형성한다.
5. 이 topology는 high-recurrent activity regime을 동반하지만, raw mean-rate cosine separation으로 성능을 단순 설명할 수는 없다.
6. 좋은 topology 또는 activity regime이 학습 중 발견되더라도, validation 기반으로 어떤 시점의 topology를 선택하느냐가 성능 보존에 결정적이다.
```

즉 현재 연구의 가장 좋은 프레이밍은 다음이다.

> 본 연구는 recurrent SNN topology learning이 단순한 sparsification 문제가 아니라, edge placement, topology parameterization, graph organization, activity regime, topology selection timing이 결합된 문제임을 보인다.

이 프레이밍은 과장되지 않으면서도 현재 결과를 가장 잘 설명한다.

---

## 7. 다음 실험: 무엇을 해야 주장력이 생기는가

현재 다음 단계는 무작정 성능을 더 올리는 실험이 아니다. 먼저 `R8`의 성능 차이를 더 직접적으로 설명해야 한다.

### 7.1 R2/R2v activity diagnostics 산출

현재 graph diagnostics에는 `R2/R2v`가 포함되었지만, activity diagnostics는 `R1/R4/R4v/R8`만 포함한다. 따라서 edge-wise learned C와 learned_lowrank의 activity regime 차이를 직접 말하려면 `R2/R2v`에 대해 다음 지표를 같은 batch protocol로 산출해야 한다.

| 항목 | 목적 |
|---|---|
| mean/max firing rate | recurrent activity scale 비교 |
| active neurons > threshold | representation coverage 비교 |
| recurrent/input current ratio | recurrence 영향력 비교 |
| class mean-rate cosine | rate-vector separation 확인 |
| readout logit margin | downstream separability 확인 |
| diagnostic batch accuracy | activity diagnostic과 성능 연결 |

### 7.2 Readout margin / temporal trajectory 분석

현재 `R8`의 class mean-rate cosine은 높다. 따라서 성능 향상을 rate-vector separation으로 설명하기 어렵다. 다음 분석은 readout이 어떤 정보를 활용하는지 확인하는 데 필요하다.

| 분석 | 목적 |
|---|---|
| mean logit margin | readout decision confidence 확인 |
| correct vs incorrect margin | 성능과 margin의 직접 연결 |
| class-wise margin | 특정 class가 이득을 보는지 확인 |
| temporal spike trajectory cosine / distance | 평균 rate가 아닌 시간 패턴 분리 확인 |
| readout weight와 hub degree correlation | graph hub가 readout에 연결되는지 확인 |

### 7.3 Causal graph intervention

Graph diagnostics는 상관 증거다. mechanism claim을 강화하려면 개입 실험이 필요하다.

| Intervention | 질문 |
|---|---|
| top-degree hub edge removal | hub-like structure가 성능에 필요한가? |
| reciprocal edge removal | reciprocity가 중요한가? |
| triangle/3-cycle motif disruption | short recurrent motif가 중요한가? |
| density-preserving edge shuffle | 같은 degree/density에서 edge placement가 중요한가? |
| learned_lowrank mask + random weights | topology와 weight magnitude의 기여 분리 |

### 7.4 통계적 방어

현재 seed 수는 4개다. 연구 메모 수준에서는 충분하지만, 논문형 주장에는 약하다. 최소한 다음 중 하나가 필요하다.

| 분석 | 목적 |
|---|---|
| bootstrap CI | 작은 seed 수에서 uncertainty 표기 |
| paired seed comparison | 같은 seed에서 method 차이 확인 |
| 추가 seeds | generalization 안정성 확인 |
| validation seed 변경 | `val_seed=42` 의존성 확인 |

---

## 8. 단기 로드맵

### Step 1. 문서 v0.4 고정

이번 업데이트의 목적은 새 실험을 추가하는 것이 아니라, 이미 확보된 성능·graph·activity evidence를 내부 문서에 고정하는 것이다.

### Step 2. R2/R2v activity diagnostics 보완

`learned_lowrank`와 edge-wise learned C의 activity regime 차이를 직접 비교할 수 있도록 동일 batch protocol로 `R2/R2v` activity metrics를 산출한다.

### Step 3. Readout margin 및 temporal diagnostics 추가

`class mean cosine`이 R8의 성능을 설명하지 못하므로, readout margin과 temporal trajectory 수준의 separability를 확인한다.

### Step 4. Causal graph intervention 설계

현재 graph evidence를 causal mechanism으로 강화하기 위해 density-preserving shuffle, hub removal, reciprocal edge removal 같은 intervention을 설계한다.

### Step 5. Phase B/C/D로 확장

현재 Phase A의 diagnostics는 계속 보완한다. 그러나 장기 로드맵의 다음 architecture 단계는 더 이상 모호한 “언젠가 ALIF/e-prop”이 아니다.

| Phase | 다음 질문 | 구현 원칙 |
|---|---|---|
| **Phase B — ALIF 이식** | ALIF가 topology-learning LSM/LSNN 구조와 호환되는가? | ALIF 자체를 증명하지 않고, 기존 검증된 adaptive threshold를 현재 구조에 안전하게 이식 |
| **Phase C — e-prop 구현** | 긴 시퀀스를 BPTT 없이 학습할 수 있는가? | 자연어 태스크 진입 전 robust implementation에 집중. SSM은 동시에 넣지 않음 |
| **Phase D — 자연어 + GPT-2 distillation + SSM** | SNN이 언어 표현공간을 담고 decoder와 연결될 수 있는가? | token-as-time, GPT-2 hidden-state distillation, multiple-choice evaluation, adapter-decoder 연결, 이후 SSM 접목 |

따라서 “prediction auxiliary / predictive coding”은 핵심 로드맵이 아니라 선택적 side track으로 둔다. 현재 공식 후속 축은 **ALIF → e-prop → NLP distillation/adapter/SSM**이다.

---

## 9. 현재 원고의 중심 문장 후보

다음 문장이 현재 연구의 중심 문장으로 가장 적합하다.

> 순환 SNN에서 성능 향상은 recurrent 연결의 존재나 밀도만으로 설명되지 않는다. SHD-LSM 실험에서 learned_lowrank는 same-density random recurrence와 edge-wise learned C보다 강한 validation-selected 성능을 보였고, 거의 같은 density에서도 훨씬 높은 degree concentration, reciprocity, short-cycle density, clustering을 갖는 partially modular recurrent topology를 형성했다. Activity diagnostics도 learned_lowrank가 high-recurrent activity regime을 만든다는 점을 보여주지만, raw mean-rate class separation만으로 성능을 설명할 수는 없다. 따라서 현재 가장 안전한 결론은 recurrent topology learning의 핵심이 density가 아니라 topology parameterization, graph organization, activity regime, selection timing의 결합에 있다는 것이다.

이 문장은 네 가지 점에서 안전하다.

1. 성능 결과를 인정하되 과장하지 않는다.
2. density-only explanation을 배제한다.
3. graph/activity diagnostics를 반영한다.
4. causal mechanism과 biological plausibility는 아직 보류한다.

---

## 10. 현재 버전의 결론

현재 연구는 “눈부신 성과” 단계는 아니지만, 내부 연구 보고서로 고정할 충분한 구조와 근거를 갖췄다. 가장 중요한 성과는 단일 accuracy 숫자가 아니라, 연구 질문이 다음처럼 정교화되었다는 점이다.

```text
SNN topology를 학습할 수 있는가?
  ↓
Random recurrent topology보다 learned topology가 나은가?
  ↓
Density가 아니라 edge placement가 중요한가?
  ↓
Edge-wise parameterization보다 latent role parameterization이 더 안정적인가?
  ↓
Graph organization과 activity regime은 어떻게 달라지는가?
  ↓
좋은 topology를 validation 기반으로 언제 고정해야 하는가?
```

현재 가장 방어 가능한 결론은 다음이다.

> `learned_lowrank + validation rollback m50p10`은 SHD-LSM에서 현재 가장 강한 test-leakage-free topology-learning result이며, 그 이점은 recurrent density가 아니라 graph organization과 high-recurrent activity regime의 차이와 함께 나타난다. 다만 이것은 아직 causal proof가 아니라 diagnostic evidence이므로, 다음 단계는 R2/R2v activity 보완, readout/temporal separability 분석, graph intervention이다.


## 11. Internal documentation lock and publication policy

### 11.1 현재 상태 판정

현재 결과는 외부 논문으로 바로 포장하기보다 내부 문서로 잠그는 것이 맞다. 이유는 명확하다.

| 항목 | 현재 상태 | 판단 |
|---|---|---|
| 성능 결과 | R8이 validation-selected main result로 가장 강함 | 내부 기준으로 의미 있음 |
| 선행연구 포지셔닝 | e-prop/LSNN/DEEP R/Grad R 위의 좁은 문제로 재정렬됨 | 안정화됨 |
| density-only 반론 | same-density random controls로 기각 가능 | 강함 |
| mechanism evidence | graph/activity diagnostics는 있으나 causal proof는 아님 | 추가 필요 |
| paper pressure | 없음 | 논문화 보류 가능 |

따라서 현재 문서의 역할은 논문 초안이 아니라 **claim hygiene + evidence ledger + next-step control document**다.

### 11.2 왜 선행연구가 많은 것이 긍정적인가

선행연구가 많다는 것은 현재 연구가 무의미하다는 뜻이 아니다. 반대로, 궁극 목표로 가기 위한 기반이 이미 존재한다는 뜻이다.

| 선행연구 흐름 | 레버리지 방식 |
|---|---|
| e-prop | BPTT 이후 online/local learning으로 확장할 경로 |
| LSNN / ALIF | 장기 기억과 adaptive threshold를 넣는 경로 |
| DEEP R / Grad R | sparse rewiring 및 hard topology-learning baseline |
| RigL / ESL-SNNs | dynamic pruning / sparse-from-scratch 비교축 |
| LSM structure optimization | random reservoir 한계를 논의할 기존 언어 |
| low-rank recurrent theory | learned_lowrank 해석의 이론적 배경 |

이제 연구 질문은 “내가 처음 topology learning을 했는가?”가 아니다. 더 정확한 질문은 다음이다.

> 기존 recurrent SNN structure-learning 흐름 위에서, 인지적 순환 계산으로 가려면 recurrent topology를 edge 단위로 학습해야 하는가, 아니면 neuron role / circuit / module 단위로 학습해야 하는가?

### 11.3 Paper-readiness trigger

paper writing은 다음 조건 중 하나 이상이 충족될 때 다시 연다.

| Trigger | 필요한 증거 |
|---|---|
| Robustness trigger | 더 많은 seeds, 다른 validation seed, 또는 SHD 외 dataset에서 R8 계열 우위 유지 |
| Mechanism trigger | hub removal, reciprocal edge disruption, 3-cycle disruption, density-preserving shuffle 등 causal graph intervention |
| Representation trigger | readout margin 또는 temporal trajectory separability가 R8 성능을 설명 |
| Learning-rule trigger | ALIF/LSNN/e-prop 확장에서도 topology-parameterization advantage 유지 |
| Cognitive-task trigger | working-memory / recurrent reasoning류 task에서 hand-structured recurrence 대비 learned topology 우위 |

이 trigger 전까지는 paper writing보다 diagnostics와 internal documentation을 우선한다.

### 11.4 현재 연구 운영 원칙

```text
1. 현재 발견은 내부 문서로 잠근다.
2. novelty claim은 과장하지 않는다.
3. performance result와 mechanism claim을 분리한다.
4. historical best-test 결과와 validation-selected 결과를 섞지 않는다.
5. 다음 연구는 Phase A diagnostics/intervention으로 claim을 잠그고, architecture는 ALIF -> e-prop -> NLP distillation/adapter/SSM 순서로 진행한다.
6. 눈에 띄는 성과가 나올 때만 paper framing을 재개한다.
```

이 원칙은 연구를 늦추기 위한 것이 아니라, 오히려 장기 목표로 더 빠르게 가기 위한 장치다. 선행연구가 이미 많이 진행되어 있으므로, 본 프로젝트는 그 위에 올라타 더 좁고 강한 질문을 밀어붙이면 된다.


# Appendix A. Claim–Evidence Matrix v0.3

## A.1 목적

이 섹션의 목적은 현재 연구에서 가능한 주장을 명시적으로 분리하는 것이다. 논문형 문서에서 가장 위험한 부분은 성능 수치가 있다는 이유만으로 메커니즘 주장까지 과도하게 확장하는 것이다. 따라서 각 주장을 다음 네 범주로 나눈다.

| 상태 | 의미 |
|---|---|
| **강함** | 현재 결과만으로도 비교적 방어 가능 |
| **중간** | 방향성은 있으나 추가 분석이 필요 |
| **약함** | 아이디어는 있으나 현재 결과로는 주장하면 위험 |
| **보류** | future work 또는 장기 비전으로만 언급 가능 |

## A.2 핵심 주장-증거 매트릭스

| ID | 주장 | 현재 증거 | 부족한 증거 | 상태 | 원고 내 위치 |
|---:|---|---|---|---|---|
| C1 | 고정 랜덤 recurrent topology는 현재 SHD-LSM 설정에서 충분하지 않다 | no-recurrence `0.5490`; same-density random mean `0.5257`, best `0.5406` | 다른 dataset 확인 | **강함** | Results |
| C2 | recurrent density 자체는 learned_lowrank의 성능 향상을 설명하지 못한다 | R8 density `0.0455`, R1 density `0.0451`; 성능은 `0.5919` vs `0.5257` | density-preserving shuffle | **강함** | Results / Diagnostics |
| C3 | learned recurrent topology는 random edge placement보다 유리한 후보이다 | R2/R2v/R4/R8이 R1보다 높음; R8이 가장 강함 | 더 많은 seeds, statistical CI | **중간~강함** | Results |
| C4 | learned_lowrank는 현재 가장 유망한 topology-learning family다 | R8 mean `0.5919`, worst `0.5826`; R2v mean `0.5603`, R4v mean `0.5330` | SHD 외 task, validation seed 변경 | **강함에 가까움** | Main Result |
| C5 | edge-wise independent theta보다 latent neuron role parameterization이 더 안정적일 수 있다 | R2/R2v보다 R8 성능 우세; R8 graph Gini/clustering/reciprocity 우세 | R2/R2v activity diagnostics, role embedding analysis | **중간~강함** | Discussion |
| C6 | topology selection timing은 성능 보존에 중요하다 | R4 historical mean `0.5803` vs R4v valrollback mean `0.5330`; activity regime도 크게 다름 | freeze/rollback trajectory 분석 | **강함** | Ablation / Discussion |
| C7 | m50p10은 현재 main-table에 쓰기 가장 안전한 validation-based policy다 | m60p10 redundant, m60p15 rejected; R8 m50p10 mean/worst 우세 | 다른 validation seed 확인 | **강함** | Method / Results |
| C8 | learned topology가 의미 있는 graph structure를 만든다 | R8의 in/out Gini, reciprocity, 3-cycle, clustering, reachability가 R1/R2v와 뚜렷하게 다름 | motif-level intervention | **강함에 가까움** | Diagnostics |
| C9 | learned_lowrank가 high-recurrent activity regime을 만든다 | R8 recurrent/input ratio `1.3715`, active >0.05 `349.0`, dead neurons `0.25` | R2/R2v activity, full test-set activity | **중간~강함** | Diagnostics |
| C10 | learned_lowrank가 class separation을 mean-rate 수준에서 개선한다 | R8 class mean cosine이 오히려 높음 | readout margin, temporal trajectory distance | **보류** | Diagnostics, 현재 주장 금지 |
| C11 | learned_lowrank가 생물학적으로 더 타당하다 | latent role / clustered topology라는 해석적 매력 | 생물학적 회로와의 직접 대응 없음 | **약함** | Future Work에서만 제한적으로 |
| C12 | e-prop은 긴 시퀀스 학습을 위한 Phase C 핵심 후보이다 | 이론적 관련성 및 로드맵상 필요성 있음 | 구현 및 BPTT 비교 실험 없음 | **보류** | Phase C |
| C13 | LSNN+topology+SSM cognitive core와 adapter-decoder 분리 구조로 확장할 수 있다 | 장기 비전과 architecture sketch는 확정 | distillation, decoder 연결, multiple-choice task, scaling evidence 없음 | **보류** | Long-term Vision / Phase D |

## A.3 현재 메인 주장 후보

현재 가장 안전한 메인 주장은 다음이다.

> 순환 SNN에서 성능 향상은 recurrent 연결의 존재나 밀도만으로 설명되지 않는다. SHD-LSM 실험에서 density-matched random recurrent topology는 no-recurrence baseline도 넘지 못한 반면, `learned_lowrank + validation rollback m50p10`은 edge-wise learned C와 Grad R validation comparator보다 높은 test@best-validation 성능을 보였다. Graph diagnostics는 이 차이가 단순 edge 수가 아니라 hub-like, clustered, partially modular topology와 연결되어 있음을 보여준다. Activity diagnostics도 R8이 high-recurrent regime을 형성함을 보이지만, raw mean-rate class separation만으로 성능을 설명할 수는 없다.

이 주장은 C1, C2, C4, C5, C6, C8, C9에 의해 지지된다. 아직 C10, C11, C12는 약하거나 보류 상태이므로 “특정 계산 회로를 발견했다”, “생물학적으로 더 타당하다”, “e-prop으로 확장성이 해결된다”는 식의 주장은 피한다.

## A.4 결과 표에 반드시 들어가야 할 비교군

| 그룹 | 방법 | 선택 기준 | 포함 이유 |
|---|---|---|---|
| R0 | no recurrence | fixed baseline | recurrent 연결 없이 가능한 기준 성능 |
| R1 | same-density random_sparse | test@best-val | density-only explanation 제거 |
| R2 | learned edge-wise Gumbel C original | historical best-test/no-val | 초기 edge-wise 방식, seed sensitivity 확인 |
| R2v | learned edge-wise Gumbel C valrollback | test@best-val + val rollback | fairer edge-wise comparator |
| R4 | Grad R-STE historical | historical best-test/no-val | 강한 hard-threshold historical comparator |
| R4v | Grad R-STE valrollback | test@best-val + val rollback | 같은 validation protocol comparator |
| R8 | learned_lowrank + m50p10 | test@best-val + val rollback | current main result |

주의: historical best-test/no-val 결과는 comparator 또는 diagnostic으로 다루고, main claim은 validation-selected 결과인 R8 중심으로 둔다.

## A.5 원고에서 피해야 할 표현

| 피해야 할 표현 | 이유 | 안전한 대체 표현 |
|---|---|---|
| learned_lowrank가 항상 우수하다 | task/seed 제한 | 현재 SHD-LSM 설정에서 가장 유망한 family로 나타났다 |
| 뇌와 더 유사한 구조를 학습했다 | 생물학적 검증 부족 | latent neuron role 기반 parameterization은 edge 결정을 공유 구조로 묶는다 |
| 학습된 회로를 발견했다 | 기능적 intervention 부족 | learned topology가 random/edge-wise와 다른 graph organization을 보였다 |
| class separation이 개선됐다 | R8 class mean cosine이 높음 | high-recurrent activity regime과 graph organization이 성능과 함께 관찰된다 |
| e-prop으로 해결하면 된다 | 구현·실험 없음 | e-prop은 Phase C에서 긴 시퀀스 학습을 검증해야 하는 후속 구현이다 |
| cognitive model로 확장 가능성이 입증됐다 | 현재 SHD/LSM 결과 범위 초과 | adapter-decoder 기반 장기 비전으로만 둔다 |

## A.6 다음 분석이 들어오면 강화되는 주장

| 추가 분석 | 강화되는 주장 | 기대 효과 |
|---|---|---|
| R2/R2v activity diagnostics | C5, C9 | edge-wise C와 learned_lowrank activity regime 직접 비교 |
| readout logit margin | C9, C10 | high-recurrent regime이 downstream separability로 이어지는지 확인 |
| temporal trajectory distance | C10 | mean-rate cosine으로 설명되지 않는 temporal code 확인 |
| role embedding visualization | C5 | latent neuron role parameterization의 구조적 해석 강화 |
| hub/motif intervention | C8 | graph statistics를 causal evidence로 강화 |
| density-preserving shuffle | C2, C8 | density/degree와 edge placement 기여 분리 |
| bootstrap CI / paired seed comparison | C4 | 작은 seed 수의 불확실성 방어 |
| validation seed 변경 | C6, C7 | validation rollback policy의 robustness 확인 |

## A.7 현재 버전의 결론

현재 연구는 다음 세 문장으로 요약할 수 있다.

1. Random recurrent topology는 현재 SHD-LSM 설정에서 충분한 성능 이득을 만들지 못했다.
2. `learned_lowrank + validation rollback m50p10`은 recurrent density만으로 설명되지 않는 성능 향상을 보였고, edge-wise learned C와 Grad R validation comparator보다 강한 main result다.
3. Graph/activity diagnostics는 learned_lowrank가 sparse but strongly recurrent, hub-like, clustered, partially modular regime을 형성함을 시사하지만, causal mechanism과 class-separation explanation은 아직 추가 검증이 필요하다.

---

# Appendix B. Main Result Table v0.3

## B.1 목적

이 섹션의 목적은 현재까지의 성능 결과를 논문형 표로 재정렬하는 것이다. 핵심 원칙은 다음이다.

> 수동 peak, historical best-test, oracle-style diagnostic best, validation-selected main result를 같은 지위로 취급하지 않는다.

순환 topology learning에서는 학습 중 특정 epoch에서 높은 test 성능이 나타났다가 후반에 collapse하는 경우가 있다. 따라서 단순 best test를 main result로 쓰면 post-hoc selection 또는 test leakage 비판을 피하기 어렵다. 본 문서에서는 결과를 다음 네 범주로 분리한다.

| 범주 | 의미 | 논문 내 사용 |
|---|---|---|
| **Main result** | validation 기준으로 선택된 test-leakage-free 결과 | 메인 표 가능 |
| **Fair comparator** | 동일한 validation-selected protocol을 가진 baseline | 메인 표 가능 |
| **Historical comparator** | 과거 실험의 best-test/no-val 기준 | 보조 표 또는 comparator로 명시 |
| **Diagnostic / upper bound** | 수동 freeze, no-freeze peak, oracle best 등 | Appendix 또는 Discussion |

## B.2 현재 핵심 성능 표

| ID | Method | Topology parameterization | Selection rule | Seeds / runs | Mean | Std | Median | Worst | Best | Status |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| R0 | no recurrence | none | fixed baseline | 1 | `0.5490` | `0.0000` | `0.5490` | `0.5490` | `0.5490` | baseline |
| R1 | same-density random_sparse | fixed random mask, `p ∈ {0.040, 0.045, 0.050}` | test@best-val | 12 | `0.5257` | `0.0084` | `0.5254` | `0.5133` | `0.5406` | density control |
| R2 | learned C original | edge-wise dense theta + Gumbel/STE | historical best-test/no-val | 4 | `0.5590` | `0.0185` | `0.5638` | `0.5331` | `0.5751` | historical comparator |
| R2v | learned C valrollback | edge-wise dense theta + Gumbel/STE | test@best-val + val rollback | 4 | `0.5603` | `0.0173` | `0.5663` | `0.5349` | `0.5738` | fair edge-wise comparator |
| R4 | Grad R-STE historical | hard threshold STE | historical best-test/no-val | 4 | `0.5803` | `0.0235` | `0.5837` | `0.5486` | `0.6051` | strong historical comparator |
| R4v | Grad R-STE valrollback | hard threshold STE | test@best-val + val rollback | 4 | `0.5330` | `0.0071` | `0.5347` | `0.5234` | `0.5393` | fair Grad R comparator |
| R8 | learned_lowrank r16 + m50p10 | latent source/destination embeddings | test@best-val + val rollback | 4 | **`0.5919`** | `0.0145` | **`0.5857`** | **`0.5826`** | **`0.6135`** | **main result** |

## B.3 본문용 압축 표

논문 본문에 들어갈 표는 과도하게 복잡하면 안 된다. 권장 본문 표는 다음이다.

| Method | Selection | Mean | Median | Worst | Best | Claim supported |
|---|---|---:|---:|---:|---:|---|
| No recurrence | fixed | `0.5490` | `0.5490` | `0.5490` | `0.5490` | baseline |
| Same-density random recurrence | test@best-val | `0.5257` | `0.5254` | `0.5133` | `0.5406` | density is insufficient |
| Edge-wise learned C | test@best-val + val rollback | `0.5603` | `0.5663` | `0.5349` | `0.5738` | edge-wise topology learning is seed-sensitive |
| Grad R-STE | test@best-val + val rollback | `0.5330` | `0.5347` | `0.5234` | `0.5393` | hard-threshold validation comparator |
| learned_lowrank + m50p10 | test@best-val + val rollback | **`0.5919`** | **`0.5857`** | **`0.5826`** | **`0.6135`** | current main result |

Historical Grad R (`R4`)는 strong comparator로 중요하지만, selection rule이 main result와 다르므로 본문 표에서는 별도 열 또는 footnote로 처리하는 것이 안전하다.

## B.4 Interpretation

Main Result Table이 전달해야 하는 메시지는 하나다.

> `learned_lowrank`의 성능 우위는 단순히 recurrent edge 수가 많아서 생긴 결과가 아니다. 같은 density의 random recurrent topology는 baseline보다도 낮았고, edge-wise learned C는 validation rollback을 붙여도 `0.5603`에 머물렀으며, 같은 validation-selected protocol에서 `learned_lowrank + m50p10`은 mean과 worst seed 모두 가장 강했다.

이 메시지를 넘어서 다음 주장까지 가면 아직 위험하다.

```text
위험한 확장:
  learned_lowrank가 어떤 구체적 회로 motif를 발견했다.
  learned_lowrank가 생물학적으로 더 타당하다.
  learned_lowrank가 모든 SNN topology learning에서 우월하다.
```

이런 주장은 topology intervention, R2/R2v activity, readout/temporal diagnostics 이후에만 강화할 수 있다.

## B.5 Remaining table work

| 항목 | 이유 | 우선순위 |
|---|---|---:|
| R2/R2v activity diagnostics | learned C와 lowrank activity 비교 | 높음 |
| bootstrap CI 또는 paired seed comparison | n=4의 통계적 방어 | 중간 |
| validation seed 변경 | m50p10 policy 의존성 확인 | 중간 |
| full test-set activity diagnostics | diagnostic batch 의존성 완화 | 중간 |

---

# Appendix C. Topology and Activity Diagnostics Results v0.2

## C.1 목적

이 섹션은 기존 `Topology Diagnostics Plan`을 결과 중심으로 갱신한다. 현재 확보된 것은 두 종류다.

| 자료 | 포함 그룹 | 상태 |
|---|---|---|
| graph diagnostics | R1, R2, R2v, R4, R4v, R8 | 사용 가능 |
| standalone activity diagnostics | R1, R4, R4v, R8 | 사용 가능, 단 R2/R2v 없음 |
| integrated `activity_metrics.csv` | R1, R2, R2v, R4, R4v, R8 row는 있으나 값은 placeholder | formal claim에 사용하지 않음 |

따라서 graph-level claim은 강하게 업데이트하고, activity-level claim은 R1/R4/R4v/R8 범위에서만 제한적으로 사용한다.

## C.2 Graph diagnostics summary

| Group | Method | Density | In Gini | Out Gini | Reciprocity | 3-cycle density | Clustering | Largest SCC | Reachability | Diam p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | random_sparse | `0.0451` | `0.1163` | `0.1150` | `0.0231` | `0.000031` | `0.0879` | `500.0` | `1.0000` | `3.0` |
| R2 | learned C original | `0.0538` | `0.2202` | `0.2946` | `0.0579` | `0.000101` | `0.1599` | `496.5` | `0.9930` | `3.0` |
| R2v | learned C valrollback | `0.0536` | `0.2205` | `0.2911` | `0.0557` | `0.000099` | `0.1590` | `496.5` | `0.9930` | `3.0` |
| R4 | Grad R historical | `0.0493` | `0.1856` | `0.2666` | `0.0365` | `0.000053` | `0.1382` | `500.0` | `1.0000` | `3.0` |
| R4v | Grad R valrollback | `0.0228` | `0.1621` | `0.1659` | `0.0114` | `0.000004` | `0.0449` | `500.0` | `1.0000` | `3.0` |
| R8 | learned_lowrank | `0.0455` | **`0.6127`** | **`0.5758`** | **`0.0817`** | **`0.000157`** | **`0.3719`** | **`396.75`** | **`0.7899`** | **`4.0`** |

### C.2.1 핵심 해석

1. **R8은 density effect가 아니다.** R8 density `0.0455`는 R1 `0.0451`과 거의 같지만 성능과 graph organization이 크게 다르다.
2. **R2/R2v는 random보다 구조화되지만 R8보다 약하다.** Edge-wise learned C는 degree concentration, reciprocity, clustering을 증가시키지만, R8만큼 강한 hub/cluster/selective-reachability 구조를 만들지는 못한다.
3. **R8은 partially modular graph다.** R1/R2/R4/R4v는 largest SCC가 거의 500이고 reachability가 1에 가깝다. 반면 R8은 largest SCC `396.75`, reachability `0.7899`, diameter p90 `4.0`이다. 이는 global mixing graph가 아니라 더 선택적인 propagation structure로 해석된다.
4. **R4v는 validation rollback 후 매우 sparse해진다.** R4v density는 `0.0228`로 낮고 clustering/3-cycle도 작다. 이는 R4 historical과 R4v validation comparator의 성능 차이를 설명할 후보다.

## C.3 Readout-degree correlation

| Group | readout-in corr | readout-out corr | readout-total corr | 해석 |
|---|---:|---:|---:|---|
| R1 | `-0.0510` | `-0.0433` | `-0.0666` | random topology와 readout weight의 구조적 정렬 없음 |
| R2 | `0.3518` | `0.6539` | `0.6033` | edge-wise C에서는 readout weight와 degree가 강하게 정렬 |
| R2v | `0.3701` | `0.6705` | `0.6242` | validation rollback 후에도 정렬 유지 |
| R4 | `0.2456` | `0.4472` | `0.4369` | Grad R historical도 degree-readout 정렬 존재 |
| R4v | `-0.0029` | `-0.0203` | `-0.0170` | valrollback Grad R에서는 정렬 사라짐 |
| R8 | `-0.1795` | `0.0926` | `-0.0628` | lowrank의 성능은 단순 degree-readout alignment로 설명되지 않음 |

이 결과는 중요하다. Edge-wise learned C는 readout weight와 graph degree가 강하게 정렬되지만, 성능은 R8보다 낮다. 반면 R8은 readout-degree correlation이 강하지 않은데도 성능이 가장 높다. 따라서 R8의 성능을 “readout이 hub를 직접 읽었기 때문”이라고 단순화하면 안 된다. R8은 degree-readout alignment보다 graph organization과 recurrent dynamics 자체가 더 중요할 가능성이 있다.

## C.4 Activity diagnostics summary

현재 standalone activity diagnostics는 R1/R4/R4v/R8만 포함한다. 표의 `n`은 diagnostic observation 수이며, independent seed 수가 아니다.

| Group | Method | n | Mean firing | Max firing | Dead neurons | Active >0.05 | Overactive >0.20 | Rec/input | Cosine mean | Cosine min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | random_sparse | 24 | `0.0475` | `0.7475` | `81.25` | `130.6` | `33.5` | `0.1485` | `0.9737` | `0.9570` |
| R4 | Grad R historical | 8 | **`0.2199`** | **`0.9925`** | `1.00` | **`385.8`** | **`204.5`** | `1.0363` | `0.9680` | `0.9213` |
| R4v | Grad R valrollback | 8 | `0.0455` | `0.7600` | `42.75` | `135.0` | `23.25` | `0.0846` | `0.9681` | `0.9484` |
| R8 | learned_lowrank | 8 | `0.1975` | **`0.9925`** | **`0.25`** | `349.0` | `145.8` | **`1.3715`** | **`0.9882`** | **`0.9781`** |

### C.4.1 핵심 해석

1. **R8은 high-recurrent activity regime이다.** R8은 dead neuron이 거의 없고, active neuron coverage가 크며, recurrent/input ratio가 가장 높다.
2. **R4와 R8은 모두 high-activity 계열이다.** R4 historical은 active >0.05와 overactive count가 가장 높다. R8은 rec/input ratio가 더 높고 dead neuron이 더 적다.
3. **R4v는 low-activity regime으로 떨어진다.** R4v는 R1과 비슷한 firing/activity 수준을 보이며, 성능도 낮다.
4. **Class mean cosine은 R8에서 높다.** 따라서 R8의 이점은 단순한 mean-rate class separation이 아니다.

## C.5 현재 가능한 mechanism statement

현재 가능한 가장 강한 mechanism statement는 다음이다.

> R8 learned_lowrank는 same-density random 및 edge-wise learned C와 다른 graph organization을 만들며, 별도 activity diagnostics에서 high-recurrent activity regime을 형성한다. 이 두 현상은 R8의 성능 우위와 함께 관찰된다. 그러나 graph/activity 차이가 성능을 직접 일으킨다는 causal proof는 아직 없고, class mean-rate separation으로 성능을 설명할 수도 없다.

## C.6 남은 diagnostics

| 항목 | 이유 | 우선순위 |
|---|---|---:|
| R2/R2v activity diagnostics | edge-wise C와 lowrank activity 직접 비교 | 높음 |
| readout logit margin | cosine으로 설명되지 않는 separability 확인 | 높음 |
| temporal trajectory diagnostics | rate 평균이 아닌 시간 패턴 분석 | 높음 |
| graph intervention | graph statistics의 causal role 확인 | 중간~높음 |
| role embedding analysis | lowrank latent role 해석 강화 | 중간 |
| validation trajectory / freeze timing | R4/R4v, R8 policy 차이 설명 | 중간 |

## C.7 Figure 계획

| Figure | 내용 | 현재 상태 |
|---|---|---|
| Figure 1 | Method schematic: LSM + topology parameterization | 작성 가능 |
| Figure 2 | Main performance comparison | 작성 가능 |
| Figure 3 | Graph diagnostics: density/Gini/clustering/reachability | 작성 가능 |
| Figure 4 | Activity regime: firing/active/rec-input/cosine | 부분 작성 가능; R2/R2v 없음 |
| Figure 5 | Topology trajectory / rollback analysis | 추가 로그 필요 |

## C.8 현재 버전의 결론

Topology diagnostics는 이제 단순 plan이 아니라 초기 결과로 전환되었다. 핵심 결론은 다음이다.

1. `learned_lowrank`는 same-density random과 거의 같은 density에서 매우 다른 graph organization을 만든다.
2. Edge-wise learned C도 random보다 구조화되지만, `learned_lowrank`만큼 강한 degree concentration, clustering, selective reachability를 만들지는 못한다.
3. Activity diagnostics는 `learned_lowrank`가 high-recurrent regime을 만든다는 점을 보이지만, class mean-rate cosine은 성능 향상을 설명하지 못한다.
4. 따라서 다음 단계는 R2/R2v activity, readout/temporal separability, graph intervention이다.

---
# Appendix D. Related Work Repositioning v0.2 — 2026-05-17

## D.1 핵심 수정

이 연구는 더 이상 “SNN topology learning 자체의 최초성”으로 포지셔닝하지 않는다. 관련 선행연구가 이미 넓게 존재한다.

| 축 | 대표 선행연구 | 본 연구에서의 의미 |
|---|---|---|
| recurrent SNN 학습 | e-prop | BPTT 병목을 줄이는 closest prior / future work |
| adaptive recurrent SNN | LSNN + DEEP R | ALIF, 장기 기억, BPTT+rewiring을 이미 결합 |
| sparse rewiring | DEEP R, Grad R, RigL, ESL-SNNs | topology learning 자체의 최초성 claim을 막는 직접 선행연구 |
| SHD/Heidelberg 구조 학습 | DEEP R + RigL dynamic pruning | 같은 계열 temporal speech SNN에서 structure learning의 중요성을 보여줌 |
| LSM liquid 구조 최적화 | EONS, evolutionary reservoir generation, adaptive/evolvable LSM | LSM 구조 최적화 자체의 최초성 claim을 막음 |
| low-rank recurrent dynamics | Mastrogiuseppe & Ostojic 계열 | `learned_lowrank` 해석 배경 |

## D.2 수정된 연구 질문

이전 질문:

> SNN에서 연결 구조를 학습할 수 있는가?

수정된 질문:

> 순환 SNN/LSM에서 topology learning의 성능 차이는 recurrent density 때문인가, 아니면 edge placement, topology parameterization, topology selection timing 때문인가?

## D.3 수정된 novelty claim

안전한 claim:

> 본 연구는 recurrent SNN topology learning을 단순 sparsification 문제가 아니라 **edge placement**, **topology parameterization**, **topology selection timing**이 결합된 문제로 다룬다. 특히 SHD-LSM에서 edge-wise Gumbel/STE, Grad R-STE, latent source/destination role 기반 learned_lowrank, validation-based topology rollback을 같은 실험 흐름 안에서 비교한다.

위험한 claim:

| 표현 | 판정 |
|---|---|
| SNN topology learning을 처음 제안한다 | 폐기 |
| recurrent SNN structure learning 최초 방법이다 | 폐기 |
| e-prop/LSNN이 topology learning을 다루지 않았다 | 폐기 |
| LSM 구조 최적화 최초 시도다 | 폐기 |
| learned_lowrank가 생물학적으로 더 타당하다 | diagnostics 전까지 보류 |

## D.4 Related Work 작성 순서

1. e-prop / LSNN / DEEP R: RSNN 학습과 sparse rewiring의 가장 가까운 역사적 선행연구.
2. Grad R / ESL-SNNs / RigL-style dynamic pruning: SNN sparse structure learning의 직접 비교군.
3. LSM structure optimization: EONS, adaptive/evolutionary LSM 등 liquid 구조 최적화 선행연구.
4. SNN NAS: macro-architecture search와 neuron-pair recurrent micro-topology learning의 차이.
5. Low-rank recurrent dynamics: `learned_lowrank`의 이론적 해석 배경.
6. 본 연구: density-matched random controls, topology parameterization comparison, validation rollback.

## D.5 현재 문서 전체에 적용할 문장

> 기존 연구는 recurrent SNN 학습, sparse rewiring, LSM 구조 최적화의 중요성을 이미 보여주었다. 본 연구는 이 흐름 위에서 SHD 기반 LSM-style recurrent SNN의 topology learning을 edge-wise Gumbel/STE, Grad R-STE, latent neuron-role learned_lowrank, validation-based topology rollback으로 분해해 비교한다. 핵심 질문은 성능 향상이 recurrent edge 수 때문인지, 아니면 edge placement, topology parameterization, topology selection timing 때문인지이다.



# Appendix E. Vision Alignment v0.5 — 2026-05-17

## E.1 기준 문서

이 메모는 `research_vision_roadmap_v0.2`의 하위 문서다. 기준 비전은 다음이다.

```text
토큰 인코더(freeze)
  → SNN 입력 어댑터
  → SNN cognitive core(LSNN + topology learning + SSM)
  → 표현공간 어댑터
  → decoder(freeze 또는 교체 가능)
```

## E.2 기존 메모와의 관계

| 기존 표현 | v0.5에서의 정리 |
|---|---|
| LSM/SHD 결과가 프로젝트의 중심 목표처럼 보이는 표현 | LSM/SHD는 Phase A evidence. 최종 목표는 cognitive core + adapter-decoder 구조 |
| ALIF/e-prop을 diagnostics 이후 먼 future work로만 둔 표현 | ALIF는 Phase B, e-prop은 Phase C. diagnostics는 Phase A claim을 잠그기 위한 병행/마무리 작업 |
| predictive coding을 주요 다음 단계처럼 보이게 하는 표현 | optional side track. 공식 로드맵은 ALIF → e-prop → NLP distillation/adapter/SSM |
| cognitive SNN/local LLM 대체 가능성이 입증된 듯한 표현 | 현재 SHD/LSM evidence로는 미입증. Phase D에서 distillation, multiple-choice, decoder 교체, scaling으로 검증 필요 |
| 생물학적 타당성을 강화하는 표현 | bio-inspired, engineering-first. 뇌는 existence proof이며 biological plausibility는 후순위 |

## E.3 즉시 다음 작업의 해석

Phase A diagnostics는 여전히 필요하다. 다만 이것은 ALIF/e-prop 진입을 막는 게 아니라, 현재 learned_lowrank 결과의 claim boundary를 깨끗하게 잠그는 작업이다. 실험 흐름은 다음처럼 병렬화할 수 있다.

1. Phase A closure: R2/R2v activity, readout margin, temporal trajectory, graph intervention.
2. Phase B branch: ALIF neuron을 현재 LSM 구조에 이식하고 LIF 대비 ablation.
3. Phase C preparation: e-prop 설계 문서와 최소 구현 스캐폴드 작성.
4. Phase D design: GPT-2 hidden-state distillation protocol, token-as-time input, adapter-decoder interface 설계.
