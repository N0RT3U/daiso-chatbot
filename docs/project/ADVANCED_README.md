# ABSA (Aspect-Based Sentiment Analysis)

다이소 뷰티 리뷰 323K건에서 **제품 속성(8개)별 감성**을 추출하는 NLP 파이프라인

## 기술 스택

| 분류 | 기술 |
|------|------|
| 언어 모델 | KcELECTRA-base (beomi/KcELECTRA-base) |
| 라벨링 | GPT-4o Batch API |
| 프레임워크 | PyTorch, Transformers (HuggingFace) |
| 후처리 | 정규식 기반 Keyword Gate, Design Rule Override |
| 평가 | scikit-learn (F1, Precision, Recall) |

## 전체 파이프라인

```
[1] 층화 샘플링 (20,000개)
 │  └─ 대분류 → 소분류 → 감성 3단계 균형 추출
 ▼
[2] GPT-4o Batch API 라벨링 ($95)
 │  └─ 8 aspect × 4-class (none/positive/neutral/negative)
 ▼
[3] 사람 직접 검수 + EDA
 │  └─ 오류 패턴 파악 → 프롬프트 재수정
 ▼
[4] KcELECTRA Multi-task 학습
 │  ├─ Stage 1: 약지도 학습 (20K, lr=2e-5)
 │  └─ Stage 2: 골든셋 파인튜닝 (lr=2e-6)
 ▼
[5] Per-aspect Threshold 튜닝
 │  └─ F0.5 메트릭 Grid Search (Precision 4배 가중)
 ▼
[6] 후처리 체인
 │  ├─ Design Rule Override (디자인 aspect 키워드 규칙)
 │  ├─ Keyword Gate (정규식 기반 오탐 방지)
 │  └─ Keyword Force-On (재구매 FN 보정)
 ▼
[7] 전체 추론 (323,114건, 62.8분)
```

## 모델 아키텍처

```
Input: 리뷰 텍스트 (max_length=128)
        │
        ▼
┌─────────────────────────────────────┐
│   KcELECTRA Encoder (768-dim)       │
│   beomi/KcELECTRA-base              │
└───────────────┬─────────────────────┘
                │ [CLS] token
         ┌──────┴──────┐
         ▼             ▼
   ┌───────────┐ ┌─────────────────┐
   │ Sentiment │ │ Aspect-Sentiment│
   │  Head     │ │  Head           │
   │ [B, 3]    │ │ [B, 8, 4]      │
   │           │ │                 │
   │ negative  │ │ 8 aspects ×     │
   │ neutral   │ │ (none/pos/      │
   │ positive  │ │  neu/neg)       │
   └───────────┘ └─────────────────┘
```

**Multi-task Loss:**
- Sentiment: CrossEntropyLoss + class weight
- Aspect: Masked CrossEntropyLoss (mask=1인 셀만 학습)

## 운영 Aspect (8개)

| # | Aspect | 설명 | 후처리 |
|---|--------|------|--------|
| 0 | 배송/포장 | 배송 속도, 포장 상태 | Threshold |
| 1 | 가격/가성비 | 가격 만족도 | Keyword Gate |
| 2 | 사용감/성능 | 발림성, 지속력, 커버력 | Threshold |
| 3 | 용량/휴대 | 용량 만족도, 휴대성 | Keyword Gate |
| 4 | 디자인 | 외관, 패키지 디자인 | Design Rule (키워드 완전 대체) |
| 5 | 재질/냄새 | 텍스처, 향 | Keyword Gate |
| 6 | 재구매 | 재구매 의사 | Force-On (FN 보정) |
| 7 | 색상/발색 | 발색력, 색상 만족도 | Threshold |

## 후처리 3단계

### 1. Per-aspect None Threshold

```
P(none) ≥ threshold → none(0)
P(none) < threshold → non-none 중 argmax → pos(1), neu(2), neg(3)
```

Grid Search: 0.1~0.95 (step 0.05) × F0.5 최적화

### 2. Design Rule Override

디자인 aspect는 모델 예측 대신 **3-tier 키워드 규칙**으로 완전 대체:

```python
# Tier 1: Positive 키워드
['예쁘', '귀엽', '깔끔', '고급', '세련']

# Tier 2: Negative 키워드
['촌스럽', '유치', '투박', '별로']

# Tier 3: Structure 키워드 (존재하면 neutral)
['디자인', '패키지', '케이스']
```

### 3. Keyword Gate (정규식 기반)

non-none 예측이지만 관련 키워드가 없으면 → none으로 복원 (오탐 방지)

```python
# 예: 가격/가성비 aspect
KEYWORDS = [
    r'가격', r'가성비', r'비싸', r'저렴',
    r'싸[다고게서니]',  # 활용형 제한 (잽싸게 등 차단)
    r'(?<![가-힣])값',  # 경계 처리
]
```

## 학습 진행 요약

| Stage | 내용 | 핵심 결과 |
|-------|------|-----------|
| Stage 1 | 11 aspect 학습 | 미분류 붕괴, 과다 검출 발견 |
| Stage 2 | 8 aspect 구조 개편 + F0.5 재학습 | Precision 0.61 |
| Stage 3A | 후처리만 변경 (재학습 없음) | **Precision 0.67, GO 판정** |
| 전체 추론 | 323,114건 / 62.8분 | v2 Keyword Gate 적용 |

## 추론 결과

| 항목 | 수치 |
|------|------|
| 총 리뷰 수 | 323,114건 |
| 총 상품 수 | 937개 |
| 추론 속도 | 85.8 reviews/sec |
| 긍정 / 중립 / 부정 | 66.5% / 29.1% / 4.4% |
| Aspect 언급률 1위 | 사용감/성능 49.4% |
| 미분류 비율 | 28.7% |

## 디렉토리 구조

```
03_ABSA/
├── RQ_absa/                    # Python 패키지 (import RQ_absa)
│   ├── s1_config.py            # 중앙 설정 (328줄)
│   │                             ├─ ASPECT_LABELS (8개)
│   │                             ├─ 학습/추론 하이퍼파라미터
│   │                             ├─ Keyword Gate 정규식 패턴
│   │                             └─ Design Rule 3-tier 키워드
│   │
│   ├── s2_sampling.py          # 3단계 층화 샘플링 (587줄)
│   │                             └─ NaturalStratifiedSampler
│   │                                 대분류→소분류→감성 균형 추출
│   │
│   ├── s3_labeling.py          # GPT-4o Batch 라벨링 (311줄)
│   │                             └─ ABSALabeler
│   │                                 증분 라벨링 (resume 지원)
│   │
│   ├── s4_dataset.py           # PyTorch Dataset 생성 (584줄)
│   │                             ├─ ABSADataProcessor (CSV→그룹화→분할)
│   │                             └─ ABSADataset (4-class 라벨 벡터)
│   │
│   ├── s5_model.py             # Multi-task 모델 (373줄)
│   │                             ├─ MultiTaskABSAModel
│   │                             │   Sentiment Head [B,3]
│   │                             │   Aspect Head [B,8,4]
│   │                             └─ FocalLoss (불균형 대응)
│   │
│   ├── s6_train.py             # 2-Stage 학습 루프 (634줄)
│   │                             └─ ABSATrainer
│   │                                 Stage 1: 약지도 (lr=2e-5)
│   │                                 Stage 2: 골든셋 파인튜닝 (lr=2e-6)
│   │                                 Threshold 자동 튜닝
│   │
│   ├── s7_evaluation.py        # 평가 + 후처리 (818줄)
│   │                             ├─ ABSAEvaluator (다중 메트릭)
│   │                             ├─ tune_none_thresholds() F0.5 Grid Search
│   │                             ├─ apply_design_rule_override()
│   │                             └─ apply_full_postprocess() 전체 후처리 체인
│   │
│   └── s8_inference.py         # 배치 추론 (700줄)
│                                 └─ ABSAInference
│                                     ├─ from_bundle() 운영 번들 로드
│                                     ├─ predict_batch() 배치 추론
│                                     ├─ infer_dataframe() DataFrame 전체 추론
│                                     └─ Streaming 10K 청크 (메모리 효율)
│
└── requirements.txt
```

## 모듈 간 의존성

```
s1_config ←── 모든 모듈이 참조 (중앙 설정)
    │
    ├── s2_sampling ──→ 샘플링 설정
    ├── s3_labeling ──→ GPT API 설정
    ├── s4_dataset  ──→ 라벨/경로 설정
    ├── s5_model    ──→ 모델/라벨 설정
    ├── s6_train    ──→ s5_model + s7_evaluation
    ├── s7_evaluation ─→ threshold/keyword/design rule 설정
    └── s8_inference ──→ s5_model + s7_evaluation + transformers
```

## 비용 요약

| 단계 | 모델 | 데이터 수 | 비용 |
|------|------|-----------|------|
| GPT-4o-mini 1차 라벨링 | GPT-4o-mini | 20,000 | $2.87 |
| Golden Set 비교 | GPT-4o | 430 | ~$0.50 |
| Batch 라벨링 (최종) | GPT-4o Batch | 20,000 | $95.70 |
| ML 추론 | KcELECTRA (로컬) | 323,114 | $0 |
| **총합** | | | **~$99** |

## 코드 규모

| 파일 | 라인 수 | 핵심 클래스 |
|------|---------|------------|
| s1_config.py | 328 | (설정 변수) |
| s2_sampling.py | 587 | NaturalStratifiedSampler |
| s3_labeling.py | 311 | ABSALabeler |
| s4_dataset.py | 584 | ABSADataProcessor, ABSADataset |
| s5_model.py | 373 | MultiTaskABSAModel, FocalLoss |
| s6_train.py | 634 | ABSATrainer |
| s7_evaluation.py | 818 | ABSAEvaluator |
| s8_inference.py | 700 | ABSAInference |
| **합계** | **4,335줄** | **8개 클래스** |
