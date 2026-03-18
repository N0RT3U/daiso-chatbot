# 네이버 검색트렌드 분석

네이버 DataLab API + 검색 API를 활용한 **연착륙(SL) vs 비연착륙(Non-SL) 제품 검색 패턴 비교 분석** 시스템

## 기술 스택

| 분류 | 기술 |
|------|------|
| API | 네이버 DataLab 검색어 트렌드 API, 네이버 검색 API (블로그/쇼핑) |
| 데이터 처리 | Pandas, NumPy |
| 시각화 | Matplotlib |
| 통계 검정 | SciPy (Mann-Whitney U) |
| 캐시 | MD5 기반 JSON 로컬 캐시 |

## 핵심 기법: 앵커 기반 정규화

네이버 DataLab API는 배치(5개 키워드 그룹) 단위로 **상대적 ratio**를 반환하므로, 배치 간 스케일이 다릅니다. 이를 해결하기 위해 **앵커 제품**을 활용합니다.

```
[문제] 배치마다 ratio 스케일이 상이
       배치1: 제품A=100, 제품B=50    ← 배치1 기준 상대값
       배치2: 제품C=100, 제품D=70    ← 배치2 기준 상대값
       → 제품A와 제품C를 직접 비교 불가

[해결] 모든 배치에 동일한 앵커 제품 포함
       앵커 = "다이소 딥 클렌징 폼" (product_code: 1035082)

       배치1: 앵커=30, 제품A=100, 제품B=50
       배치2: 앵커=45, 제품C=100, 제품D=70

       normalized_ratio = product_ratio / anchor_ratio

       제품A = 100/30 = 3.33    제품C = 100/45 = 2.22
       제품B = 50/30  = 1.67    제품D = 70/45  = 1.56

       → 모든 제품을 같은 척도로 비교 가능!
```

## 전체 파이프라인

```
[SLI 통합 결과]
  SL 제품 158개 + Non-SL 제품 548개
       │
       ▼
┌─────────────────────────────────────────┐
│  1. 키워드 그룹 생성                     │
│     (keyword_builder.py)                │
│     ├─ 브랜드별: "다이소 {브랜드}" 등    │
│     └─ 제품별: "{브랜드} {제품}" 등      │
│     용량/단위 제거, 중복 해결            │
└────────────────┬────────────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌──────────────────┐ ┌──────────────────┐
│ DataLab API      │ │ 검색 API         │
│ (트렌드 ratio)   │ │ (블로그/쇼핑)    │
│                  │ │                  │
│ 앵커+4제품/배치  │ │ 키워드별 건수    │
│ 5개씩 분할 호출  │ │ 최대 1000건 수집 │
│                  │ │                  │
│ 키 로테이션     │ │ 키 로테이션     │
│ MD5 캐시        │ │ MD5 캐시        │
└────────┬─────────┘ └────────┬─────────┘
         │                    │
         ▼                    ▼
┌──────────────────────────────────────────┐
│  2. 앵커 기반 정규화                      │
│     normalized_ratio = ratio / anchor     │
│     → SL vs Non-SL 직접 비교 가능        │
└────────────────┬─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│  3. 비교 분석 (run_anchor_analysis.py)    │
│     ├─ 제품별 지표: 잔존율, CV, 트렌드   │
│     ├─ Mann-Whitney U 검정               │
│     ├─ 세그먼트별: 성별/연령대/기기      │
│     └─ 시각화 9종 자동 생성              │
└──────────────────────────────────────────┘
```

## 세그먼트 분석

각 제품에 대해 **11개 세그먼트**별 검색 트렌드를 수집:

| 세그먼트 타입 | 세그먼트 | 코드 |
|--------------|---------|------|
| base | 전체 | - |
| gender | 남성, 여성 | m, f |
| age | 10대, 20대, 30대, 40대, 50대, 60대 | 1~6 |
| device | PC, 모바일 | pc, mo |

## 분석 결과 (SL vs Non-SL 비교)

| 지표 | SL (연착륙) | Non-SL | 해석 |
|------|------------|--------|------|
| 잔존율 | 37.1% | 17.5% | SL이 2배 이상 높은 검색 유지력 |
| CV (변동계수) | 0.661 | 0.970 | SL이 더 안정적 |
| 안정 패턴 비율 | 34.1% | 10.5% | SL의 1/3이 안정적 트렌드 |
| 총 검색량 | 256만 건 | - | 블로그 92.3%, 쇼핑 7.3% |
| 검색 키워드 1위 | "후기" | - | "리뷰" 대비 50% 높음 |

## 디렉토리 구조

```
05_search_trend/
├── README.md
│
├── 05_src/                               # 핵심 API 클라이언트
│   ├── config.py                         # 설정 (86줄)
│   │                                       ├─ API 키 로테이션 (N개 키)
│   │                                       ├─ 성별/연령대/기기 코드 매핑
│   │                                       └─ 기본 조회 범위 (2024.01~2026.01)
│   │
│   ├── keyword_builder.py                # 키워드 그룹 생성 (284줄)
│   │                                       ├─ extract_search_keyword()  제품명 정제
│   │                                       ├─ build_brand_keyword_groups()  브랜드별
│   │                                       └─ build_product_keyword_groups() 제품별
│   │
│   ├── naver_trend_client.py             # DataLab 트렌드 API (273줄)
│   │                                       ├─ NaverTrendClient
│   │                                       ├─ search_trend()          5그룹/요청
│   │                                       ├─ search_trend_batch()    자동 분할
│   │                                       └─ MD5 캐시 + 키 로테이션
│   │
│   └── naver_search_client.py            # 검색 API (311줄)
│                                           ├─ NaverSearchClient
│                                           ├─ search()                단건 검색
│                                           ├─ search_total()          건수만 반환
│                                           ├─ search_all_pages()      최대 1000건
│                                           └─ search_bulk_keywords()  일괄 조회
│
└── 06_scripts/                           # 실행 스크립트
    ├── run_anchor_search_trend.py        # 앵커 기반 트렌드 수집 (568줄)
    │                                       앵커(딥클렌징폼) + 4제품/배치
    │                                       SL 158개 + Non-SL 548개
    │
    ├── run_anchor_analysis.py            # 앵커 비교 분석 + 시각화 (667줄)
    │                                       ├─ 잔존율, CV, 트렌드 분류
    │                                       ├─ Mann-Whitney U 검정
    │                                       └─ 시각화 9종 자동 생성
    │
    ├── run_soft_landing_search_trend.py  # SL 제품 트렌드 수집 (300줄)
    │
    ├── run_soft_landing_segment_analysis.py  # SL 세그먼트 분석 (300줄)
    │                                          11세그먼트 × 30배치
    │
    ├── collect_soft_landing_search_volume.py # SL 검색량 수집 (168줄)
    │                                          블로그/쇼핑/뉴스 건수
    │
    └── visualize_soft_landing.py         # 시각화 (150줄)
                                            바 차트, 시계열, 히트맵
```

## API 안정성 설계

### 키 로테이션 (Round-Robin)

```python
# 429 발생 시 다음 키로 자동 전환
NAVER_API_KEYS = [
    {"client_id": "id1", "client_secret": "secret1"},
    {"client_id": "id2", "client_secret": "secret2"},
    ...
]
# 모든 키 소진 시 60초 대기 후 재시도
```

### MD5 캐시

```python
# 동일 요청 중복 호출 방지
cache_key = hashlib.md5(json.dumps(request_body).encode()).hexdigest()
if cache_key in self.cache:
    return self.cache[cache_key]  # 캐시 히트
```

### 배치 분할 전략

```
API 제약: 최대 5개 키워드 그룹 / 요청

앵커 모드: 앵커 1개 + 제품 4개 = 5개 / 배치
  SL 158개 → 40배치
  Non-SL 548개 → 137배치
```

## 코드 규모

| 구분 | 파일 수 | 라인 수 |
|------|---------|---------|
| Core (05_src) | 4개 | 954줄 |
| Scripts (06_scripts) | 6개 | ~2,150줄 |
| **합계** | **10개** | **~3,100줄** |
