# Daiso Beauty Chatbot

다이소 뷰티 상품과 리뷰 데이터를 바탕으로, 사용자가 자연어로 피부 고민이나 원하는 조건을 입력하면 그에 맞는 제품을 추천해주는 웹 챗봇 프로젝트입니다.

이 저장소의 목적은 단순히 "챗봇 UI"를 보여주는 것이 아니라, 아래 전체 흐름을 실제로 구현하는 것입니다.

- 다이소 상품/리뷰 데이터 기반 추천 카탈로그 구축
- 자연어 질의를 추천 조건으로 해석
- 여러 지표를 결합한 추천 점수 계산
- 추천 이유와 실제 상품 링크 제공
- 로컬 실행, Render 배포, EC2 + DuckDNS 배포까지 지원

## 1. What This Project Does

사용자가 아래처럼 입력하면:

- `건성인데 보습 좋은 크림 추천해줘`
- `3천 원 이하 순한 스킨케어 보여줘`
- `리뷰 많은 클렌징 추천해줘`
- `재구매율 높은 다이소 선크림 추천해줘`

챗봇은 다음 단계를 거쳐 응답합니다.

1. 문장에서 피부 타입, 예산, 원하는 효과, 카테고리, 강조 포인트를 추출합니다.
2. 통합 카탈로그에서 조건에 맞는 상품 후보를 필터링합니다.
3. 리뷰 감성, 인기도, 재구매율, 가성비 등 여러 점수를 조합해 최종 순위를 계산합니다.
4. 상위 상품을 카드 형태로 보여주고 추천 이유를 자연어로 설명합니다.
5. 실제 다이소몰 상품 상세 페이지와 공식 매장 재고 확인 페이지로 이동할 수 있게 연결합니다.

## 2. Why This Exists

다이소 뷰티는 저렴하고 접근성이 좋지만, 제품 수가 많고 상품 정보가 파편화되어 있어서 "나한테 맞는 제품"을 고르기가 어렵습니다.

이 프로젝트는 이 문제를 아래 방식으로 풀었습니다.

- 단순 평점순 정렬 대신 다중 지표 기반 추천
- 일반 사용자가 이해하기 쉬운 질의 입력 방식
- 실제 구매 흐름과 연결되는 상품 링크/재고 확인 지원
- 배포 가능한 웹앱 형태 제공

## 3. How Recommendation Works

추천은 단일 평점이 아니라 여러 요인을 합쳐 계산합니다.

기본 가중치는 아래와 같습니다.

- 감성 분석: 30%
- 연착륙/꾸준한 인기 지표: 20%
- 인기도: 15%
- 평점: 10%
- 가성비: 10%
- 재구매율: 10%
- 성분/안전성 적합도: 5%

질문 내용에 따라 일부 가중치는 조정됩니다.

예:

- `가성비`가 들어오면 value 비중 상승
- `순한`, `민감성`이 들어오면 ingredient/safety 비중 상승
- `인기`, `리뷰 많은`이 들어오면 popularity 비중 상승
- `재구매`가 들어오면 repurchase 비중 상승

핵심 구현은 [engine.py](/g:/데이터 모음/Proj/daiso/src/chatbot/engine.py) 에 있습니다.

## 4. Query Parsing Strategy

질의 해석은 하나의 방식에 고정하지 않고, 아래 백엔드를 선택할 수 있게 설계했습니다.

- `rule`: 규칙 기반 파서만 사용
- `local`: 로컬 질의 파서 우선, 실패 시 규칙 기반 폴백
- `openai`: OpenAI로 질의 구조화, 실패 시 규칙 기반 폴백
- `auto`: local -> openai -> rule 순서로 폴백

중요한 점은 추천 엔진 자체와 질의 파서는 분리되어 있다는 것입니다.

- 질의 파서: 사용자의 문장을 구조화된 조건으로 바꿈
- 추천 엔진: 구조화된 조건을 실제 상품 점수 계산에 반영

즉 OpenAI를 쓰더라도 추천 자체는 데이터 기반 점수 계산으로 동작합니다.

관련 파일:

- [query_parser.py](/g:/데이터 모음/Proj/daiso/src/chatbot/query_parser.py)
- [local_query_parser.py](/g:/데이터 모음/Proj/daiso/src/chatbot/local_query_parser.py)
- [train_local_query_parser.py](/g:/데이터 모음/Proj/daiso/src/chatbot/train_local_query_parser.py)

## 5. User Experience

웹 UI는 일반 사용자 기준으로 어렵지 않게 설계했습니다.

- 첫 화면에서 자연어 예시 제공
- 예시 버튼 클릭 시 바로 검색 실행
- 결과 카드에 가격, 리뷰 수, 재구매율, 추천 점수 표시
- 실제 상품 상세 링크 제공
- 공식 매장 재고 확인 페이지 연결
- `/embed` 경로 제공으로 외부 임베드 시도 지원

관련 파일:

- [index.html](/g:/데이터 모음/Proj/daiso/templates/chatbot/index.html)
- [_exchange.html](/g:/데이터 모음/Proj/daiso/templates/chatbot/_exchange.html)
- [app.js](/g:/데이터 모음/Proj/daiso/static/chatbot/app.js)
- [styles.css](/g:/데이터 모음/Proj/daiso/static/chatbot/styles.css)

## 6. Project Structure

```text
daiso/
  data/
    chatbot/
      daiso_chatbot_catalog.csv
  deploy/
    Caddyfile
    .env.example
  docs/
    project/
      CHATBOT_APP.md
      CHATBOT_DEPLOY_QUICKSTART.md
  models/
    query_parser/
  src/
    chatbot/
      app.py
      build_catalog.py
      engine.py
      local_query_parser.py
      models.py
      query_parser.py
      train_local_query_parser.py
  static/
    chatbot/
  templates/
    chatbot/
  docker-compose.yml
  Dockerfile
  README.md
  render.yaml
  requirements-chatbot.txt
  run_chatbot.py
```

## 7. Core Files

- [run_chatbot.py](/g:/데이터 모음/Proj/daiso/run_chatbot.py)
  FastAPI 앱 실행 엔트리포인트

- [app.py](/g:/데이터 모음/Proj/daiso/src/chatbot/app.py)
  라우팅, 템플릿 렌더링, `/`, `/embed`, `/chat`, `/api/chat`, `/healthz`

- [engine.py](/g:/데이터 모음/Proj/daiso/src/chatbot/engine.py)
  추천 필터링/가중치 계산/카드 생성의 핵심

- [build_catalog.py](/g:/데이터 모음/Proj/daiso/src/chatbot/build_catalog.py)
  추천에 필요한 통합 카탈로그 생성 스크립트

- [docker-compose.yml](/g:/데이터 모음/Proj/daiso/docker-compose.yml)
  앱, Caddy, DuckDNS 실행 구성

- [deploy/Caddyfile](/g:/데이터 모음/Proj/daiso/deploy/Caddyfile)
  HTTPS reverse proxy 설정

## 8. Data and Catalog

챗봇은 [daiso_chatbot_catalog.csv](/g:/데이터 모음/Proj/daiso/data/chatbot/daiso_chatbot_catalog.csv) 기반으로 동작합니다.

이 파일은 여러 원천 데이터를 합쳐 만든 통합 추천 카탈로그입니다.

예를 들어 아래 정보들이 포함됩니다.

- 제품 기본 정보
- 가격
- 리뷰 수
- 평균 평점
- 재구매율
- 감성/ABSA 관련 집계
- 성분/효과 태그
- 검색용 텍스트
- 추천 카드용 snippet

카탈로그가 없으면 UI는 떠도 추천 요청은 실패합니다.

재생성:

```bash
python src/chatbot/build_catalog.py
```

## 9. Local Run

### 9-1. Minimum Requirements

- Python 3.11 권장
- 통합 카탈로그 파일 존재
- `requirements-chatbot.txt` 설치

### 9-2. Install

```bash
pip install -r requirements-chatbot.txt
```

### 9-3. Run

```bash
python -m uvicorn run_chatbot:app --reload
```

브라우저:

```text
http://127.0.0.1:8000
```

### 9-4. Recommended Local Environment Variables

PowerShell 예시:

```powershell
$env:DAISO_QUERY_PARSER_BACKEND="rule"
$env:OPENAI_API_KEY=""
$env:DAISO_CHATBOT_MODEL="gpt-4o-mini"
python -m uvicorn run_chatbot:app --reload
```

## 10. API Endpoints

- `GET /`
  메인 웹 UI

- `GET /embed`
  외부 임베드 시도용 진입 경로

- `GET /healthz`
  헬스체크

- `POST /chat`
  HTML partial 응답

- `POST /api/chat`
  JSON 응답

예시:

```json
{
  "message": "건성인데 보습 좋은 크림 추천해줘"
}
```

## 11. Deployment Options

이 저장소는 세 가지 흐름을 지원합니다.

### A. Local only

- 가장 빠른 개발/테스트용
- `uvicorn`으로 바로 실행

### B. GitHub + Render

- 초보자에게 가장 쉬운 배포
- [render.yaml](/g:/데이터 모음/Proj/daiso/render.yaml) 포함
- 도커 기반 자동 배포 가능

### C. EC2 + DuckDNS + Caddy

- 실제 공개 웹서비스 형태
- HTTPS, reverse proxy, DuckDNS 자동 업데이트 포함

## 12. Docker / EC2 Deployment

### 12-1. App only

도메인 없이 공인 IP + 포트 8000으로 먼저 확인할 때:

```bash
docker compose up -d --build
```

접속:

```text
http://<EC2_PUBLIC_IP>:8000
```

### 12-2. Domain + HTTPS

DuckDNS와 Caddy까지 같이 쓰는 경우:

```bash
docker compose --profile prod-domain up -d --build
```

필수 준비:

- `APP_DOMAIN`
- `LETSENCRYPT_EMAIL`
- `DUCKDNS_SUBDOMAINS`
- `DUCKDNS_TOKEN`

보안 그룹:

- `22` SSH
- `80` HTTP
- `443` HTTPS
- 필요 시 `8000` 직접 확인용

## 13. Environment Variables

자주 쓰는 값만 정리하면 아래와 같습니다.

- `DAISO_QUERY_PARSER_BACKEND`
  `rule`, `local`, `openai`, `auto`

- `OPENAI_API_KEY`
  OpenAI 파서 사용 시 필요

- `DAISO_LOCAL_QUERY_MODEL_DIR`
  로컬 질의 파서 모델 경로

- `DAISO_CHATBOT_MODEL`
  OpenAI 사용 시 모델명

- `APP_DOMAIN`
  DuckDNS 또는 실제 도메인

- `DUCKDNS_SUBDOMAINS`
  DuckDNS 서브도메인

- `DUCKDNS_TOKEN`
  DuckDNS 토큰

- `DAISO_APP_PORT`
  앱 포트 매핑

## 14. What Was Actually Debugged

이 프로젝트는 단순히 "코드만 만든 상태"가 아니라 실제 배포 과정에서 발생한 문제들을 직접 해결한 상태입니다.

대표적으로 아래 이슈를 정리했습니다.

### 14-1. Small EC2 disk / Docker build failure

초기에는 `torch`, `transformers`, CUDA 계열 의존성 때문에 EC2 작은 디스크에서 이미지 빌드가 실패했습니다.

해결:

- 런타임 의존성에서 무거운 ML 패키지를 분리
- 기본 배포는 `rule` 백엔드 중심의 경량 이미지로 구성

### 14-2. Mixed Content on HTTPS

도메인 HTTPS 환경에서 정적 파일 URL이 `http://...` 로 생성되면서 UI가 깨지는 문제가 있었습니다.

해결:

- 정적 리소스 경로를 루트 기준 `/static/...` 로 통일
- Uvicorn proxy header 신뢰 옵션 추가

### 14-3. Notion embedding issue

초기에는 아래 문제가 섞여 있었습니다.

- iframe 차단 헤더
- `HEAD /` 응답 문제
- 초기 실패 상태 캐시 가능성

해결:

- Caddy에서 iframe 차단 헤더 제거
- `/embed` 전용 경로 추가
- Caddy에서 `HEAD /` 확인용 응답 처리

주의:

- 서버 설정을 모두 맞춰도 Notion/Iframely가 특정 URL을 embed로 받아주지 않으면 북마크로 전환될 수 있습니다.
- 즉 임베드는 서버 설정과 Notion 쪽 판단이 둘 다 맞아야 합니다.

## 15. Limitations

- 추천 품질은 통합 카탈로그 품질에 직접 영향을 받습니다.
- 로컬 KcELECTRA 질의 파서는 모델 파일이 준비되어 있어야 합니다.
- OpenAI 경로는 선택 사항이지만 API 비용이 발생할 수 있습니다.
- Notion 임베드는 서버 설정만으로 100% 보장되지 않습니다.

## 16. Recommended Read Order

처음 보는 사람이라면 아래 순서로 보면 이해가 빠릅니다.

1. 이 README
2. [CHATBOT_APP.md](/g:/데이터 모음/Proj/daiso/docs/project/CHATBOT_APP.md)
3. [CHATBOT_DEPLOY_QUICKSTART.md](/g:/데이터 모음/Proj/daiso/docs/project/CHATBOT_DEPLOY_QUICKSTART.md)
4. [app.py](/g:/데이터 모음/Proj/daiso/src/chatbot/app.py)
5. [query_parser.py](/g:/데이터 모음/Proj/daiso/src/chatbot/query_parser.py)
6. [engine.py](/g:/데이터 모음/Proj/daiso/src/chatbot/engine.py)
7. [build_catalog.py](/g:/데이터 모음/Proj/daiso/src/chatbot/build_catalog.py)

## 17. One-line Summary

다이소 뷰티 리뷰 데이터와 상품 데이터를 기반으로, 사용자가 자연어로 원하는 조건을 말하면 실제 상품 추천과 이유 설명, 상품 상세 링크, 재고 확인 흐름까지 연결해주는 FastAPI 기반 웹 챗봇 프로젝트입니다.
