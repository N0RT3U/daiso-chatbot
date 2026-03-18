# BigQuery 데이터 인프라

크롤링 데이터의 **ETL 변환, BigQuery 적재, Tableau 시뮬레이터용 분석 뷰**를 제공하는 데이터 인프라 모듈

## 기술 스택

| 분류 | 기술 |
|------|------|
| 데이터 웨어하우스 | Google BigQuery |
| ETL | Python (Pandas) |
| 시각화 연동 | Tableau (SQL VIEW) |
| 인증 | GCP Service Account |

## 아키텍처

```
[크롤러 CSV]                          [ABSA 추론 결과]     [SLI 연착륙 판별]
     │                                      │                    │
     ▼                                      ▼                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ETL Layer (etl_loader.py)                         │
│  CSV → 정규화 변환 → FK 의존성 순서 → UPSERT                        │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 BigQuery (schema_v3.sql, 18개 테이블)                 │
│                                                                      │
│  [마스터]  brands, manufacturer, ingredients_dic, promotions          │
│  [제품]    products_core, products_category, products_stats           │
│            products_ingredients, functional                           │
│  [사용자]  user_id_map, users_profile, users_repurchase              │
│  [리뷰]    reviews_core, reviews_text                                │
│  [분석]    review_absa, review_aspects, sli_results                  │
│  [운영]    search_trends, pipeline_log                               │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────────────┐
│              Tableau 시뮬레이터 VIEW (10개+)                          │
│                                                                      │
│  v_score_distribution ──→ 제품별 기본점수 (축1+3+4+6)                │
│  v_golden_ingredients ──→ 카테고리별 골든/안티 성분                   │
│  v_ingredient_factors ──→ 축5/8 성분 적합성 양방향                   │
│  v_category_ingredient_type_score ──→ 축7 성분 타입 그룹             │
│  v_sim_category_benchmark ──→ 카테고리 BAN 카드                     │
│  v_sim_similar_sl_products ──→ 유사 연착륙 제품 Top-5               │
└──────────────────────────────────────────────────────────────────────┘
     │
     ▼
[Tableau 대시보드: 입점 성공 확률 시뮬레이터]
```

## 핵심 기능

### 1. BigQuery CRUD 클라이언트 (`bq_client.py`)

모든 BigQuery 작업을 추상화한 인터페이스:

| 함수 | 기능 |
|------|------|
| `query_to_df(sql)` | SQL 실행 → DataFrame 반환 |
| `insert_df(df, table)` | DataFrame → BigQuery INSERT |
| `upsert_df(df, table, keys)` | PK 기반 UPSERT (MERGE 패턴) |
| `list_tables()` | 데이터셋 테이블 목록 조회 |
| `get_table_schema(table)` | 스키마 정보 조회 |

**UPSERT 패턴:**
```
1. 임시 테이블에 DataFrame 로드
2. MERGE SQL로 PK 기반 UPDATE/INSERT
3. 타입 불일치 자동 CAST
4. 임시 테이블 삭제
```

### 2. ETL 자동화 (`etl_loader.py`)

크롤러 CSV를 ERD 정규화 테이블로 자동 변환:

```
[크롤러 CSV]
    │
    ├─ products_xxx.csv
    │     → brands (브랜드 마스터 자동 등록)
    │     → products_core (제품 기본)
    │     → products_stats (likes, shares, engagement_score)
    │     → products_category (카테고리 매핑)
    │
    ├─ reviews_xxx.csv
    │     → user_id_map + users_profile (사용자 자동 등록)
    │     → reviews_core (기본 정보)
    │     → reviews_text (텍스트 분리)
    │     → 파생변수: is_reorder, review_length 자동 생성
    │
    └─ ingredients_xxx.csv
          → ingredients_dic (성분 마스터 자동 등록)
          → products_ingredients (M:N 매핑)
```

**FK 의존성 순서 자동 처리:**
```
brands → manufacturer → ingredients_dic → promotions
    → products_core → products_category → products_stats → products_ingredients → functional
    → user_id_map → users_profile → users_repurchase
    → reviews_core → reviews_text
```

### 3. 스키마 마이그레이션 (`migrate_v3.py`)

v2(13개 테이블) → v3(18개 테이블) 무중단 업그레이드:

```python
# 신규 5개 테이블
review_absa       # ABSA 리뷰 레벨 감성 (sentiment, score, aspect_count)
review_aspects    # Aspect 레벨 감성 (1:N, aspect × sentiment × confidence)
sli_results       # SLI 연착륙 판별 (DTW/생존/규칙/ML 4방법론 투표)
search_trends     # 네이버 검색트렌드 (시계열)
pipeline_log      # 파이프라인 실행 이력
```

### 4. Tableau 시뮬레이터 뷰 (입점 성공 확률 점수화)

8축 100점 만점 스코어링 시스템:

| 축 | 배점 | 설명 | 계산 방식 |
|----|------|------|----------|
| 1-카테고리 | 5점 | 카테고리별 SL 비율 | 비선형 3단계 (≥35%→5, ≥20%→3, ≥5%→2) |
| 2-브랜드 | 25점 | 브랜드 파워 | 사용자 슬라이더 입력 |
| 3-가격 | 15점 | SL 중앙값 대비 편차 | 거리 비율 기반 |
| 4-성분수 | 15점 | 카테고리별 적정 범위 | Q1~Q3 비선형 매핑 |
| 5-성분적합성 | 20점 | 골든/실버 성분 매칭 | Gold×2 + Silver×1 |
| 6-기능성 | 5점 | 식약처 기능성 등록 | Binary (5 or 0) |
| 7-성분타입 | 15점 | 성분 그룹별 SL 비율 | 5그룹 매핑 (0~10점) |
| 8-안티성분 | -20점 | 안티 성분 페널티 | Red×2 + Orange×1 |

**base_score** = 축1 + 축3 + 축4 + 축6 (자동 계산, 최대 40점)
**최종 점수** = base_score + 축2 + 축5 + 축7 + 축8 (사용자 입력 포함, 최대 100점)

## 디렉토리 구조

```
02_bigquery/
├── bq_client.py                      # BigQuery CRUD 클라이언트 (309줄)
│                                       ├─ get_client()         인증 및 클라이언트 생성
│                                       ├─ query_to_df()        SQL → DataFrame
│                                       ├─ upsert_df()          PK 기반 MERGE UPSERT
│                                       └─ TABLE_KEYS           18개 테이블 PK 정의
│
├── etl_loader.py                     # 크롤러 CSV → ERD 정규화 ETL (465줄)
│                                       ├─ CrawlerETL           구 스키마 호환
│                                       └─ CrawlerETLv2         v2 스키마 (FK 순서 UPSERT)
│
├── migrate_v3.py                     # 스키마 마이그레이션 v2→v3 (271줄)
│
├── schema_v3.sql                     # ERD v3 DDL (18개 테이블)
│
├── tableau_scorecard_views_v2.sql    # Tableau 시뮬레이터 VIEW v2 (10개 VIEW)
│
├── v_ingredient_factors.sql          # 축5/8: 골든+안티 성분 양방향 분석
│
└── v_score_distribution_v2_patch.sql # 기본점수 PATCH (축5/8 추가)
```

## ERD (v3, 18개 테이블)

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────────┐
│   brands    │      │ manufacturer │      │   ingredients_dic   │
│ brand_id PK │      │manufacturer_ │      │ ingredient_id PK    │
│ name        │      │  id PK       │      │ ingredient_name     │
└──────┬──────┘      └──────┬───────┘      │ ingredient_type     │
       │                    │              │ is_allergic         │
       ▼                    ▼              └──────────┬──────────┘
┌──────────────────────────────────────┐              │
│           products_core              │              │
│ product_code PK                      │              │
│ brand_id FK, manufacturer_id FK      │              │
│ name, price, country                 │              │
└──────────────────┬───────────────────┘              │
                   │                                  │
       ┌───────────┼──────────┬──────────┐            │
       ▼           ▼          ▼          ▼            ▼
 products_    products_   products_   functional  products_
 category    stats       ingredients              ingredients
                                                  (M:N)
       │
       ▼
┌───────────────────────────────────────┐
│            reviews_core               │
│ review_id PK, product_code FK         │
│ user_id FK, rating, review_date       │
└──────────────┬────────────────────────┘
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
 reviews_  review_  review_     sli_results    search_trends
 text      absa    aspects     (product_code)  (product_code)
```

## 코드 규모

| 구분 | 파일 수 | 라인 수 |
|------|---------|---------|
| Python | 4개 | 1,316줄 |
| SQL DDL | 2개 | 305줄 |
| SQL VIEW | 4개 | 1,070줄 |
| **합계** | **10개** | **2,691줄** |
