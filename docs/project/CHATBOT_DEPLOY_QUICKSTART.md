# Daiso Chatbot Deploy Quickstart

## 가장 쉬운 배포 방법

처음 배포라면 `GitHub + Render`로 가는 게 가장 쉽습니다.

- `GitHub`는 코드를 올려두는 곳입니다.
- `Render`는 올린 코드를 실제 웹사이트로 실행해주는 곳입니다.

중요:

- `GitHub Pages`만으로는 안 됩니다.
- 이유는 이 프로젝트가 `FastAPI` 기반이라서 Python 서버가 계속 실행되어야 하기 때문입니다.
- `GitHub Pages`는 HTML/CSS/JS 같은 정적 사이트만 올릴 수 있습니다.

이 프로젝트는 이미 `render.yaml`이 들어 있어서 Render에 연결만 하면 됩니다.

## 추천 경로

처음에는 아래 순서로 하세요.

1. GitHub에 올린다.
2. Render에 연결한다.
3. Render가 자동으로 배포한다.
4. Render 주소로 접속해서 먼저 동작 확인한다.
5. 나중에 원하면 DuckDNS나 개인 도메인을 붙인다.

## 1. GitHub에 올리기

### 1-1. GitHub 저장소 만들기

GitHub에서 새 저장소를 하나 만듭니다.

- 저장소 이름 예시: `daiso-chatbot`
- `Public`이나 `Private` 아무거나 가능

### 1-2. 내 PC에서 업로드

PowerShell에서 프로젝트 폴더로 이동합니다.

```powershell
cd "g:\데이터 모음\Proj\daiso"
```

아래 순서대로 실행합니다.

```powershell
git init
git add .
git commit -m "Initial deploy"
git branch -M main
git remote add origin https://github.com/내아이디/daiso-chatbot.git
git push -u origin main
```

주의:

- 이미 Git 저장소라면 `git init`은 다시 안 해도 됩니다.
- 이미 `origin`이 연결되어 있으면 `git remote add origin ...` 대신 아래를 씁니다.

```powershell
git remote set-url origin https://github.com/내아이디/daiso-chatbot.git
git push -u origin main
```

## 2. Render에 배포하기

### 2-1. Render 가입

아래 사이트에 들어가서 GitHub 계정으로 가입합니다.

- `https://render.com/`

### 2-2. 저장소 연결

Render에서 아래 순서로 들어갑니다.

1. `New +`
2. `Blueprint`
3. 방금 만든 GitHub 저장소 선택

이 프로젝트에는 `render.yaml`이 있어서 Render가 배포 설정을 자동으로 읽습니다.

### 2-3. 환경변수 확인

처음 배포는 아래 값으로 두는 것을 권장합니다.

- `DAISO_QUERY_PARSER_BACKEND=rule`
- `OPENAI_API_KEY` 비워둠

이렇게 하면:

- OpenAI 비용이 들지 않습니다.
- 로컬 질의 파서 모델이 없어도 일단 배포는 됩니다.

### 2-4. Deploy 누르기

Render에서 `Apply` 또는 `Deploy`를 누르면 됩니다.

배포가 끝나면 `https://something.onrender.com` 같은 주소가 나옵니다.

## 3. 접속 확인

배포가 끝나면 아래 두 주소를 확인합니다.

- 메인 페이지: `https://너의앱주소.onrender.com`
- 상태 확인: `https://너의앱주소.onrender.com/healthz`

정상이면 `/healthz`에서 아래처럼 보입니다.

```json
{"status":"ok"}
```

## 4. 앞으로 수정할 때

코드를 수정한 뒤 아래만 하면 됩니다.

```powershell
git add .
git commit -m "Update chatbot"
git push
```

그러면 Render가 자동으로 다시 배포합니다.

## 5. OpenAI 없이 먼저 쓰는 것을 권장하는 이유

처음 배포는 복잡도를 줄이는 게 중요합니다.

- OpenAI 키 설정 안 해도 됩니다.
- 비용 걱정이 없습니다.
- 에러 원인이 줄어듭니다.

먼저 사이트가 뜨는 것부터 확인하고, 그다음에 필요하면 OpenAI나 로컬 KcELECTRA 모델을 붙이는 게 맞습니다.

## 6. 나중에 도메인 붙이기

처음에는 Render 주소 그대로 쓰는 게 제일 쉽습니다.

예:

- `https://daiso-chatbot-xxxx.onrender.com`

사이트가 잘 뜨는 걸 확인한 뒤에만 아래를 진행하세요.

- `DuckDNS` 붙이기
- 개인 도메인 연결
- EC2로 이전

## 7. 제일 쉬운 한 줄 요약

`GitHub에 올리고 -> Render에 연결하면 -> 홈페이지로 바로 배포할 수 있습니다.`

## 8. 막히기 쉬운 부분

### Q. GitHub만 쓰면 안 되나요?

안 됩니다.

- `GitHub Pages`는 정적 사이트용입니다.
- 이 프로젝트는 `FastAPI` 서버가 필요합니다.

### Q. OpenAI 키 꼭 넣어야 하나요?

아니요.

처음에는 안 넣는 게 낫습니다.

### Q. DuckDNS도 지금 바로 해야 하나요?

아니요.

처음에는 Render 기본 주소로 먼저 확인하세요.

### Q. EC2가 꼭 필요한가요?

아니요.

처음 배포는 `Render`가 훨씬 쉽습니다.
