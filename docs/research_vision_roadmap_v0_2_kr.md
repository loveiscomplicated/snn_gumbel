# Research Vision & Roadmap v0.2
> 작성일: 2026-05-17  
> 변경: 로드맵 재구조화 (e-prop Phase C 편입, 자연어 태스크 구체화, 목표 아키텍처 확정, 개인 연구 전략 추가)  
> 위치: 이 문서는 SHD/LSM 실험 메모(research_memo_v0.4)와 별개로, 연구의 장기 비전과 전체 로드맵을 기록하는 문서다.

---

## 0. 한 문장 요약

LSNN + topology learning + SSM을 결합한 recurrent SNN을 인지 코어로, adapter + decoder를 발현부로 분리한 구조를 만들어, 경량 LLM과 대등한 성능을 더 적은 자원으로 달성하고 태스크에 따라 발현부를 교체할 수 있는 모듈형 인지 시스템을 구축한다.

---

## 1. 연구 동기

세 가지 문제의식에서 출발한다.

**첫째, 현재 LLM은 너무 무겁다.**  
Gemma 4, Qwen 3 같은 경량 모델도 로컬 실행에 상당한 자원을 요구한다. 연산 효율성의 근본적인 개선이 필요하다.

**둘째, 현재 LLM은 생각하는 부분과 발화하는 부분이 혼재되어 있다.**  
토큰을 생성하면서 동시에 추론해야 하는 구조는 추론 깊이와 유연성에 구조적 한계를 만든다.

**셋째, 인지와 발현을 분리하면 모듈성이 생긴다.**  
인지 코어가 고정되어 있으면, 발현부(언어 생성, 행동 제어, 분류 등)만 교체하면서 태스크에 맞게 확장할 수 있다.

---

## 2. 아이디어의 계보

이 연구는 여러 갈래의 선행 흐름을 연결해서 만들어졌다.

| 흐름 | 핵심 기여 | 이 연구에서의 역할 |
|---|---|---|
| Looped LLM (Ouro 등) | 파라미터 수가 아닌 latent state 반복 갱신으로도 추론 가능 | SNN recurrent core의 동기 |
| SSM / Mamba | 상태를 별도로 저장하는 방식의 유용성 입증 | 상태 분리 구조의 선례, Phase D에서 접목 |
| RSNN / LSNN | SNN으로 시계열 처리 가능성 입증 | recurrent SNN 기반의 근거 |
| SNN + SSM 연구 (SpikingSSMs 등) | SSM block과 spiking dynamics 결합, language modeling까지 실험 | Phase D 아키텍처의 직접 선행연구 |
| SNN 고유 특성 | 이벤트 스트림 처리 특화, 저전력, 연산 효율성 | 공학적 효율성의 근거 |
| 뇌 | 적은 에너지로 고도의 추론 수행 | existence proof |

핵심 도약: 이 흐름들을 연결해서, **Looped LLM이 Transformer block으로 하던 latent state 반복 갱신을 SNN recurrent dynamics로 대체할 수 있다**는 가설을 세운다.

---

## 3. 포지션

**Bio-inspired, engineering-first.**

뇌는 "이런 것이 가능하다"는 existence proof다. 구현은 철저히 공학적 기준으로 판단한다. 생물학적 타당성(biological plausibility)은 후순위다.

---

## 4. 목표 구조

```
입력 토큰
    ↓
토큰 인코더 (기존 모델 재사용, freeze)
    ↓
SNN 입력 어댑터 (직접 구현)
    ↓
SNN cognitive core (직접 구현) ← 핵심 연구 대상
(LSNN + topology learning + SSM)
    ↓
표현공간 어댑터 (직접 구현)
    ↓
Decoder (기존 모델 재사용, freeze 또는 교체 가능)
    ↓
언어, 행동, 분류, 제어
```

### 구성별 역할

| 구성 | 출처 | 역할 |
|---|---|---|
| 토큰 인코더 | GPT-2 등 재사용 | 텍스트 → 임베딩 |
| SNN 입력 어댑터 | 직접 구현 | 임베딩 → SNN 입력 형태 변환 |
| SNN cognitive core | 직접 구현 | 기억, 상태 갱신, 간섭 억제, 반복적 사유 |
| 표현공간 어댑터 | 직접 구현 | SNN state → decoder 표현공간 정렬 |
| Decoder | GPT-2 등 재사용 | SNN state를 언어·행동·예측으로 번역 |

### 어댑터가 필요한 이유

LSNN + topology learning + SSM이 결합된 인지 코어는 내부 dynamics가 GPT-2 표현공간과 달라질 수밖에 없다. 인지 코어 전체를 표현공간에 정렬하려 하면 SNN 고유의 dynamics가 망가진다. 따라서 인지 코어와 decoder 사이에 어댑터를 두어 표현공간을 정렬한다.

이 어댑터만 교체하면 decoder를 자유롭게 바꿀 수 있다. **어댑터가 모듈성의 기술적 근거다.**

"LLM이 생각하면서 말한다"가 아니라, **SNN cognitive core가 상태를 만들고 decoder가 발화한다.**

---

## 5. 성공 기준

| 기준 | 내용 |
|---|---|
| 효율성 | 경량 LLM(Gemma 4, Qwen 3 등)보다 적은 자원 |
| 성능 | 경량 LLM과 대등한 수준 |
| 모듈성 | 발현부(어댑터 + decoder) 교체로 태스크 확장 가능 |

세 가지를 동시에 달성하는 것이 최종 목표다. 개인 연구 단계에서는 스케일 독립적인 문제들을 먼저 해결하고, 스케일업은 외부 협력을 통해 진행한다. (→ 섹션 7 참조)

---

## 6. 연구 로드맵

### Phase A — 구조 자기조직화 (완료)

**핵심 질문:** Topology가 dynamics를 만드는가?  
**내용:** LIF + learned_lowrank topology learning, SHD 기반 LSM  
**현재 상태:** learned_lowrank가 same-density random보다 의미 있게 성능이 높음을 확인. graph diagnostics, activity diagnostics 진행 중.  
**상세:** research_memo_v0.4 참조

---

### Phase B — ALIF 이식

**핵심 질문:** ALIF가 우리 topology-learning LSM 구조와 호환되는가?  
**프레이밍:** ALIF를 증명하는 것이 아니라, 이미 검증된 ALIF를 우리 구조에 안전하게 이식한다.  
**성공 기준:** 성능 유지 또는 향상. 악화 시 topology learning과의 상호작용 원인 분석.  
**비고:** e-prop은 이 단계에서 후순위. 현재 T=100 수준에서 BPTT unroll이 병목이 아님.

---

### Phase C — e-prop 구현

**핵심 질문:** 긴 시퀀스를 BPTT 없이 안정적으로 학습할 수 있는가?  
**배경:** 자연어 태스크로 이동하면 긴 시퀀스는 피할 수 없다. e-prop은 Phase D 진입의 전제 조건이다.  
**프레이밍:** e-prop 구현 난이도가 높으므로 이 단계에서 robust한 구현에 집중한다. SSM 접목은 Phase D로 분리한다. 두 가지를 동시에 건드리면 성능 변화의 원인을 추적할 수 없다.  
**성공 기준:** e-prop이 BPTT와 비슷한 성능을 내면서 긴 시퀀스에서 안정적으로 작동한다.

---

### Phase D — 자연어 태스크 + GPT-2 Distillation + SSM 탐색

**핵심 질문:** SNN이 언어 표현공간을 담을 수 있는가? 인지 코어 + decoder 분리가 실제로 작동하는가?

#### D-1. GPT-2 Distillation (선행)

GPT-2를 freeze하고 SNN이 그 hidden state를 모방하도록 학습한다. 언어를 처음부터 배우는 것이 아니라 표현공간 근사 문제로 좁힌다.

이 실험은 "발화부 고정 + 인지 코어 교체"의 역방향이다. SNN이 언어 표현공간을 근사할 수 있다면, 나중에 SNN state → adapter → decoder 방향으로 뒤집을 수 있다.

**설계 원칙:** 토큰 순서 자체가 이미 시간이다. 토큰을 긴 spike train으로 펼치지 말고, 토큰이 들어올 때마다 SNN state를 업데이트하는 event로 취급한다.

**입력 인코딩 방향:**

| 방향 | 특징 | 역할 |
|---|---|---|
| Rate coding | 임베딩 값 → 발화 빈도 | baseline only |
| Token-as-time + current injection | 토큰 하나 = SNN timestep, 실수값 주입 | 깔끔한 출발점 |
| Distillation (유력) | GPT-2 freeze + SNN이 hidden state 모방 | 현실적, 성능 선례 있음 |

#### D-2. 1단계 검증 — 인지 코어가 상태를 제대로 만드는가

decoder 없이 SNN state만으로 평가한다. 다지선다 태스크 + linear classifier.

**후보 벤치마크 (난이도 순):**

| 벤치마크 | 특징 |
|---|---|
| BoolQ | Yes/No 이진 분류, 가장 단순한 출발점 |
| ARC Easy | 추론 다지선다, 난이도 조절 가능 |
| HellaSwag | 문맥 이해 + 다지선다, 상태 유지 필요 |
| MMLU | 다양한 도메인 다지선다, 경량 LLM과 직접 비교 가능 |

#### D-3. 2단계 검증 — decoder가 SNN state를 자연어로 번역할 수 있는가

표현공간 어댑터를 통해 SNN state를 decoder에 연결한다. decoder 교체 실험으로 모듈성을 검증한다.

#### D-4. SSM 접목 탐색

e-prop이 안정화된 이후, LSNN + topology learning 구조에 SSM block을 결합한다. 선행연구(SpikingSSMs, SPikE-SSM, P-SpikeSSM 등)를 레퍼런스로 삼는다. SSM은 경쟁자이면서 동시에 레버리지다.

---

## 7. 개인 연구 전략

### 스케일 독립적 문제를 먼저 해결한다

개인 연구자 + 클라우드 GPU 환경에서 직접 만들어야 하는 것은 세 가지다:

- SNN cognitive core (LSNN + topology learning + SSM)
- SNN 입력 어댑터
- 표현공간 어댑터

토큰 인코더와 decoder는 기존 공개 모델을 재사용한다. gradient는 어댑터와 SNN에만 흘러 메모리/연산량이 관리 가능한 범위 안에 있다.

### 스케일업은 외부 협력으로

개인 단계의 목표는 "구조가 작동한다, 스케일하면 더 좋아질 것이다"를 소규모에서 보여주는 것이다. 작은 스케일에서도 스케일링 경향이 관찰되면, 관심 있는 기관이나 연구자와의 협력으로 스케일업을 진행할 수 있다.

**개인 단계 마일스톤:**
1. 구조가 작동하는가 (파이프라인 end-to-end)
2. 표현공간 정렬이 되는가 (distillation)
3. 다지선다 태스크에서 의미 있는 성능이 나오는가
4. decoder 교체가 실제로 작동하는가 (모듈성)
5. 스케일을 올리면 성능이 오르는 경향이 보이는가

---

## 8. 전체 흐름 요약

```
Phase A  →  topology가 dynamics를 만드는가                     [완료]
Phase B  →  ALIF와 호환되는가
Phase C  →  e-prop으로 긴 시퀀스를 안정적으로 학습할 수 있는가
Phase D  →  SNN이 언어 표현공간을 담고, decoder와 연결될 수 있는가
            (distillation → 1단계 검증 → 2단계 검증 → SSM 접목)
```

각 단계가 다음 단계의 전제가 되는 구조. 개인 연구로 구조를 증명하고, 스케일업은 외부 협력으로 이어간다.
