# Daiso Chatbot App

## 개요

이 앱은 `data/chatbot/daiso_chatbot_catalog.csv`를 기반으로 동작하는 다이소 뷰티 추천 챗봇입니다.  
리뷰 감성 분석, SLI, 인기도, 평점, 가성비, 재구매율, 성분 안전성을 합쳐 상위 제품을 추천합니다.

질의 파서는 세 가지 경로를 지원합니다.

- `local`: 로컬 KcELECTRA 질의 파서 우선, 없으면 규칙 기반으로 폴백
- `openai`: OpenAI 파서 우선, 실패하면 규칙 기반으로 폴백
- `auto`: 로컬 KcELECTRA -> OpenAI -> 규칙 기반 순서로 폴백

기본값은 `auto`입니다.

## 주요 파일

- `run_chatbot.py`
- `src/chatbot/app.py`
- `src/chatbot/query_parser.py`
- `src/chatbot/local_query_parser.py`
- `src/chatbot/train_local_query_parser.py`
- `src/chatbot/engine.py`
- `templates/chatbot/index.html`
- `templates/chatbot/_exchange.html`
- `static/chatbot/styles.css`
- `static/chatbot/app.js`
- `Dockerfile`
- `docker-compose.yml`
- `deploy/Caddyfile`
- `deploy/.env.example`

## 1. 카탈로그 생성

이미 `data/chatbot/daiso_chatbot_catalog.csv`가 있으면 바로 실행할 수 있습니다.  
다시 만들려면 아래를 실행합니다.

```bash
python src/chatbot/build_catalog.py
```

원본 데이터 루트를 직접 지정할 수도 있습니다.

```bash
python src/chatbot/build_catalog.py --source-root "G:/Final_proj/Total_clear/데이터data"
```

주의:

- 원격 저장소에 CSV를 커밋하지 않았다면 배포 서버로 `data/chatbot/daiso_chatbot_catalog.csv`를 별도로 복사해야 합니다.
- CSV가 없으면 웹 UI는 열리지만 추천 요청은 `503`으로 응답합니다.

## 2. 로컬 개발 실행

### 기본 실행

```bash
uvicorn run_chatbot:app --reload
```

브라우저에서 `http://127.0.0.1:8000`에 접속합니다.

### 환경 변수

PowerShell 예시:

```powershell
$env:DAISO_QUERY_PARSER_BACKEND="auto"
$env:OPENAI_API_KEY=""
$env:DAISO_CHATBOT_MODEL="gpt-4o-mini"
```

OpenAI 비용 없이 쓰려면 `OPENAI_API_KEY`를 비워 두고 아래 둘 중 하나를 권장합니다.

- `DAISO_QUERY_PARSER_BACKEND=local`
- `DAISO_QUERY_PARSER_BACKEND=auto`

## 3. KcELECTRA 로컬 질의 파서 학습

질의 파서 학습 결과는 기본적으로 `models/query_parser`에 저장됩니다.

```bash
python src/chatbot/train_local_query_parser.py --epochs 3 --batch-size 16
```

학습이 끝나면 아래 파일이 생성됩니다.

- `models/query_parser/query_parser_model.pt`
- `models/query_parser/parser_config.json`
- `models/query_parser/encoder/`
- 토크나이저 파일들

배포 서버에서 완전 로컬 우선으로 운영하려면 아래처럼 설정합니다.

```powershell
$env:DAISO_QUERY_PARSER_BACKEND="local"
$env:DAISO_LOCAL_QUERY_MODEL_DIR="g:/데이터 모음/Proj/daiso/models/query_parser"
```

## 4. Docker로 바로 배포

### 1) 환경 파일 준비

```bash
cp deploy/.env.example .env
```

`.env`에서 최소한 아래 값을 채웁니다.

- `APP_DOMAIN`
- `LETSENCRYPT_EMAIL`
- `DUCKDNS_SUBDOMAINS`
- `DUCKDNS_TOKEN`

완전 로컬 우선이면 아래처럼 둡니다.

```text
DAISO_QUERY_PARSER_BACKEND=local
OPENAI_API_KEY=
```

OpenAI도 보조 경로로 남겨두려면 아래처럼 둡니다.

```text
DAISO_QUERY_PARSER_BACKEND=auto
OPENAI_API_KEY=sk-...
```

### 2) 컨테이너 기동

```bash
docker compose up -d --build
```

구성은 다음과 같습니다.

- `app`: FastAPI 챗봇 앱
- `caddy`: HTTPS 종료와 리버스 프록시
- `duckdns`: 서버 공인 IP를 DuckDNS 도메인에 자동 반영

### 3) 상태 확인

```bash
docker compose ps
docker compose logs -f app
docker compose logs -f caddy
docker compose logs -f duckdns
```

헬스 체크:

```bash
curl https://your-subdomain.duckdns.org/healthz
```

## 5. DuckDNS + EC2 배포 체크리스트

### EC2 보안 그룹

아래 포트를 열어야 합니다.

- `22` SSH
- `80` HTTP
- `443` HTTPS

### 서버 준비

서버에 Docker와 Docker Compose를 설치한 뒤 이 프로젝트를 올립니다.

```bash
git clone <your-repo>
cd daiso
cp deploy/.env.example .env
docker compose up -d --build
```

### DuckDNS 연결

- `APP_DOMAIN`은 `your-subdomain.duckdns.org`
- `DUCKDNS_SUBDOMAINS`는 `your-subdomain`
- `DUCKDNS_TOKEN`은 DuckDNS에서 발급받은 토큰

`duckdns` 컨테이너가 주기적으로 공인 IP를 업데이트합니다.

## 5-1. 실제 배포 순서

아래 순서대로 하면 됩니다.

### 1) EC2 인스턴스 생성

- Ubuntu 22.04 권장
- 보안 그룹 인바운드 허용:
  - `22` SSH
  - `80` HTTP
  - `443` HTTPS

### 2) 서버 접속 후 Docker 설치

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

다시 로그인한 뒤 아래로 확인합니다.

```bash
docker --version
docker compose version
```

### 3) 프로젝트 업로드

서버에서 Git으로 받거나, 로컬에서 압축해서 업로드합니다.

```bash
git clone <your-repo-url> daiso
cd daiso
```

### 4) 카탈로그 CSV 업로드

아래 파일은 꼭 서버에 있어야 합니다.

- `data/chatbot/daiso_chatbot_catalog.csv`

로컬에서 서버로 복사 예시:

```bash
scp data/chatbot/daiso_chatbot_catalog.csv ubuntu@<EC2_IP>:/home/ubuntu/daiso/data/chatbot/
```

### 5) 로컬 KcELECTRA 모델을 쓸 경우 모델 폴더 업로드

완전 로컬 파서를 쓰려면 아래 폴더 내용을 서버에 같이 올립니다.

- `models/query_parser/query_parser_model.pt`
- `models/query_parser/parser_config.json`
- `models/query_parser/encoder/`
- 토크나이저 파일들

복사 예시:

```bash
scp -r models/query_parser ubuntu@<EC2_IP>:/home/ubuntu/daiso/models/
```

### 6) 환경 파일 작성

```bash
cp deploy/.env.example .env
nano .env
```

최소 수정 항목:

- `APP_DOMAIN=너의서브도메인.duckdns.org`
- `LETSENCRYPT_EMAIL=네이메일`
- `DUCKDNS_SUBDOMAINS=너의서브도메인`
- `DUCKDNS_TOKEN=DuckDNS토큰`

로컬 파서 우선이면:

```text
DAISO_QUERY_PARSER_BACKEND=local
OPENAI_API_KEY=
```

OpenAI도 보조로 둘 거면:

```text
DAISO_QUERY_PARSER_BACKEND=auto
OPENAI_API_KEY=sk-...
```

### 7) 서비스 기동

```bash
docker compose up -d --build
```

### 8) 로그 확인

```bash
docker compose ps
docker compose logs -f app
docker compose logs -f caddy
docker compose logs -f duckdns
```

### 9) 접속 확인

- `https://너의서브도메인.duckdns.org`
- `https://너의서브도메인.duckdns.org/healthz`

정상이라면 `/healthz`가 아래처럼 나옵니다.

```json
{"status":"ok"}
```

## 6. 추천 점수

기본 가중치:

- 감성 분석 30%
- 연착륙 지수 20%
- 인기도 15%
- 평점 10%
- 가성비 10%
- 재구매율 10%
- 성분 매칭 5%

질의에 `가성비`, `순한`, `인기`, `재구매` 같은 표현이 들어오면 해당 축의 가중치가 자동 조정됩니다.

## 7. API 엔드포인트

- `GET /`: 웹 UI
- `GET /healthz`: 헬스 체크
- `POST /chat`: HTML partial 응답
- `POST /api/chat`: JSON 응답

`/api/chat` 예시:

```json
{
  "message": "건성인데 보습 좋은 크림 추천해줘"
}
```
