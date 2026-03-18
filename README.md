# Daiso Beauty Data Project

다이소 뷰티 사업의 성장성과 리스크를 함께 분석해, `신뢰 확보 → 상품 최적화 → 물류 효율화`로 이어지는 데이터 기반 유통 전략을 설계한 프로젝트입니다.

## What This Repo Covers

- `src/acquisition`: 다이소몰 크롤링, OCR, 전성분 추출 모듈
- `src/absa`: ABSA 학습·평가·추론 파이프라인
- `src/trend`: 네이버 검색 트렌드 및 연착륙 분석 스크립트
- `src/gis`: 외국인 상권 및 물류 효율화 GIS 분석 코드
- `src/bigquery`: BigQuery 적재 및 ETL 코드
- `notebooks/`: 최종 분석 노트북
- `docs/`: 보고서, ERD, 파생변수, 스토리텔링 근거

## Business Impact

- **리스크 방어**: OCR + 성분 필터링으로 위험 제품 사전 차단
- **상품 전략**: SLI + ABSA + 검색트렌드로 연착륙 상품과 타겟 브랜드 도출
- **물류 효율화**: GIS 기반 Hub & Spoke 재고 전략 설계

## Repository Layout

```text
src/
  acquisition/
  absa/
  trend/
  gis/
  bigquery/
  common/
notebooks/
  eda/
  advanced/
  gis/
docs/
  reports/
  project/
  storytelling/
data/
```

## Data Policy

원천 데이터와 대용량 산출물은 GitHub에 포함하지 않습니다.
필요한 배경 설명은 `data/README.md`와 `docs/project/` 문서를 참고하면 됩니다.

## Recommended Commit Order

1. `chore: initialize repository structure and gitignore`
2. `feat: add crawling and OCR ingestion modules`
3. `feat: add EDA notebooks and preprocessing workflow`
4. `feat: add ABSA pipeline and search trend analysis`
5. `feat: add GIS and BigQuery integration modules`
6. `docs: add reports and project documentation`
