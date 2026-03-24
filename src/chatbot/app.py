from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .engine import DEFAULT_CATALOG_PATH, recommend_products
from .models import RecommendationResponse
from .query_parser import parse_query


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "templates"
STATIC_DIR = REPO_ROOT / "static"

STARTER_PROMPTS = [
    {"label": "보습 제품 추천", "query": "건성인데 보습 좋은 크림 추천해줘"},
    {"label": "순한 스킨케어", "query": "3천 원 이하 순한 스킨케어 추천해줘"},
    {"label": "인기 클렌징", "query": "리뷰 많은 클렌징 제품 보여줘"},
    {"label": "재구매 선크림", "query": "재구매율 높은 다이소 선크림 추천해줘"},
]


def _run_recommendation(message: str, catalog_path: Path) -> RecommendationResponse:
    if not catalog_path.exists():
        raise FileNotFoundError("catalog file is missing")
    query = parse_query(message)
    return recommend_products(query, catalog_path)


def create_app(catalog_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="Daiso Beauty Chatbot")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    resolved_catalog = catalog_path or DEFAULT_CATALOG_PATH

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "chatbot/index.html",
            {
                "starter_prompts": STARTER_PROMPTS,
                "catalog_ready": resolved_catalog.exists(),
            },
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_class=HTMLResponse)
    async def chat(request: Request) -> HTMLResponse:
        body = (await request.body()).decode("utf-8")
        payload = parse_qs(body)
        message = payload.get("message", [""])[0].strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        try:
            response = _run_recommendation(message, resolved_catalog)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return templates.TemplateResponse(
            request,
            "chatbot/_exchange.html",
            {
                "message": message,
                "response": response,
            },
        )

    @app.post("/api/chat")
    async def chat_api(request: Request) -> JSONResponse:
        payload = await request.json()
        message = str(payload.get("message", "")).strip()
        if not message:
            raise HTTPException(status_code=400, detail="message is required")

        try:
            response = _run_recommendation(message, resolved_catalog)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JSONResponse(response.model_dump())

    return app
