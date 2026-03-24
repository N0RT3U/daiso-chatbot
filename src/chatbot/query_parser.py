from __future__ import annotations

import json
import os
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency
    OpenAI = None

from .local_query_parser import parse_with_local_model
from .models import ParsedQuery


LIST_FIELDS = (
    "category_1",
    "category_2",
    "product_keywords",
    "desired_aspects",
    "desired_effects",
    "avoid_effects",
    "skin_types",
    "focus",
    "notes",
)

CATEGORY_1_KEYWORDS = {
    "스킨케어": ["스킨케어", "기초", "토너", "에센스", "앰플", "세럼", "크림", "로션", "미스트", "클렌징", "선크림", "팩", "립밤"],
    "메이크업": ["메이크업", "쿠션", "컨실러", "파운데이션", "틴트", "립", "아이", "마스카라", "섀도우", "치크", "하이라이터"],
    "맨케어": ["남성", "남자", "맨즈", "맨케어", "쉐이빙", "면도", "애프터쉐이브"],
}

CATEGORY_2_KEYWORDS = {
    "기초스킨케어": ["기초", "토너", "스킨", "에센스", "앰플", "세럼", "크림", "로션", "미스트"],
    "립케어": ["립케어", "립밤", "립마스크"],
    "자외선차단제": ["자외선", "선크림", "선케어", "sunscreen", "uv"],
    "클렌징/필링": ["클렌징", "세안", "폼클렌징", "클렌징폼", "워터", "오일", "패드", "필링", "스크럽"],
    "팩/마스크": ["팩", "마스크", "시트팩", "마스크팩"],
    "베이스메이크업": ["베이스", "쿠션", "파운데이션", "컨실러", "프라이머"],
    "립메이크업": ["틴트", "립", "글로스", "립스틱"],
    "아이메이크업": ["아이", "섀도우", "아이라이너", "마스카라", "브로우"],
    "치크/하이라이터": ["치크", "블러셔", "하이라이터"],
    "남성스킨케어": ["남성 스킨케어", "남자 스킨", "남성 기초", "쉐이빙"],
    "남성향수": ["남성 향수", "향수", "코롱", "오드퍼퓸"],
    "남성용면도기": ["면도기", "쉐이빙", "면도", "면도용"],
    "클렌징/쉐이빙": ["쉐이빙폼", "쉐이빙젤", "면도 폼", "면도 젤"],
    "남성메이크업": ["남성 메이크업", "남자 메이크업", "맨즈 메이크업"],
}

PRODUCT_KEYWORDS = [
    "토너",
    "에센스",
    "앰플",
    "세럼",
    "크림",
    "로션",
    "미스트",
    "클렌징",
    "폼클렌징",
    "클렌징폼",
    "워터",
    "오일",
    "패드",
    "마스크",
    "선크림",
    "쿠션",
    "컨실러",
    "파운데이션",
    "틴트",
    "립밤",
    "립스틱",
    "섀도우",
    "마스카라",
    "브로우",
    "치크",
    "하이라이터",
    "면도기",
    "쉐이빙폼",
]

SKIN_TYPE_KEYWORDS = {
    "dry": ["건성", "건조", "속건조"],
    "oily": ["지성", "유분", "번들", "기름"],
    "sensitive": ["민감", "예민", "순한", "자극 적은"],
    "acne": ["트러블", "여드름", "모공", "피지"],
    "combination": ["복합성", "복합"],
}

EFFECT_KEYWORDS = {
    "moisturizing": ["보습", "수분", "촉촉", "건성", "건조", "속건조"],
    "soothing": ["진정", "순한", "민감", "예민", "자극 적은", "쿨링"],
    "cleansing": ["클렌징", "세안", "노폐물", "각질", "피지", "워터", "오일"],
    "brightening": ["미백", "브라이트닝", "톤업", "칙칙", "비타민"],
    "sun_care": ["자외선", "선크림", "선케어", "uv", "sunscreen"],
    "lip_care": ["립케어", "립밤", "입술", "립마스크"],
    "makeup_cover": ["커버", "컨실러", "쿠션", "파운데이션", "베이스"],
}

AVOID_EFFECT_KEYWORDS = {
    "fragrance": ["무향", "향료 없는", "향 없는", "향 강하지 않은", "향료 적은"],
}

ASPECT_KEYWORDS = {
    "가격/가성비": ["가성비", "저렴", "싼", "가격 좋은", "값어치"],
    "사용감/성능": ["사용감", "성능", "효과", "발림", "보습력", "지속력", "커버력"],
    "재구매": ["재구매", "또 살", "다시 살", "반복 구매"],
    "색상/발색": ["색상", "발색", "컬러"],
    "제형/향": ["제형", "향", "향기", "발림성"],
    "용량/휴대": ["용량", "휴대", "크기", "들고 다니기"],
}

FOCUS_KEYWORDS = {
    "가성비": ["가성비", "저렴", "싼", "가격 좋은", "합리적"],
    "순함": ["순한", "자극 적은", "민감", "예민"],
    "인기": ["인기", "유명", "베스트", "후기 많은", "리뷰 많은"],
    "재구매": ["재구매", "반복 구매", "또 사는", "다시 사는"],
}

SKIN_TYPE_LABELS = {
    "dry": "건성",
    "oily": "지성",
    "sensitive": "민감성",
    "acne": "트러블성 피부",
    "combination": "복합성",
}

EFFECT_LABELS = {
    "moisturizing": "보습",
    "soothing": "순함/진정",
    "cleansing": "클렌징",
    "brightening": "미백/톤업",
    "sun_care": "선케어",
    "lip_care": "립케어",
    "makeup_cover": "커버력",
}

BUDGET_PATTERN = re.compile(r"(?P<amount>\d+(?:[.,]\d+)?)\s*(?P<unit>천원|천 원|만원|만 원|원)")
TOP_K_PATTERN = re.compile(r"(?:(?:top|TOP)\s*(?P<top>\d+))|(?P<count>\d+)\s*개")


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _parse_budget(text: str) -> tuple[float | None, float | None]:
    min_budget: float | None = None
    max_budget: float | None = None

    for match in BUDGET_PATTERN.finditer(text):
        amount = float(match.group("amount").replace(",", ""))
        unit = match.group("unit").replace(" ", "")
        if unit == "천원":
            amount *= 1000
        elif unit == "만원":
            amount *= 10000

        left_context = text[max(0, match.start() - 8) : match.start()]
        right_context = text[match.end() : min(len(text), match.end() + 8)]
        window = f"{left_context}{right_context}"

        if any(token in window for token in ["이상", "부터", "최소", "넘는"]):
            min_budget = amount if min_budget is None else max(min_budget, amount)
        else:
            max_budget = amount if max_budget is None else min(max_budget, amount)

    return min_budget, max_budget


def _extract_top_k(text: str) -> int:
    match = TOP_K_PATTERN.search(text)
    if not match:
        return 5
    value = match.group("top") or match.group("count")
    try:
        return max(1, min(int(value), 10))
    except (TypeError, ValueError):
        return 5


def _build_notes(
    min_budget: float | None,
    max_budget: float | None,
    category_1: list[str],
    category_2: list[str],
    skin_types: list[str],
) -> list[str]:
    notes: list[str] = []
    if min_budget is not None:
        notes.append(f"{int(min_budget):,}원 이상")
    if max_budget is not None:
        notes.append(f"{int(max_budget):,}원 이하")
    if category_2:
        notes.append(", ".join(category_2))
    elif category_1:
        notes.append(", ".join(category_1))
    if skin_types:
        notes.append(", ".join(SKIN_TYPE_LABELS[item] for item in skin_types if item in SKIN_TYPE_LABELS))
    return _dedupe(notes)


def parse_query_rule_based(message: str) -> ParsedQuery:
    text = " ".join(message.strip().split())
    min_budget, max_budget = _parse_budget(text)

    category_1 = [label for label, keywords in CATEGORY_1_KEYWORDS.items() if _contains_any(text, keywords)]
    category_2 = [label for label, keywords in CATEGORY_2_KEYWORDS.items() if _contains_any(text, keywords)]
    product_keywords = [keyword for keyword in PRODUCT_KEYWORDS if keyword in text]
    skin_types = [label for label, keywords in SKIN_TYPE_KEYWORDS.items() if _contains_any(text, keywords)]
    desired_effects = [label for label, keywords in EFFECT_KEYWORDS.items() if _contains_any(text, keywords)]
    avoid_effects = [label for label, keywords in AVOID_EFFECT_KEYWORDS.items() if _contains_any(text, keywords)]
    desired_aspects = [label for label, keywords in ASPECT_KEYWORDS.items() if _contains_any(text, keywords)]
    focus = [label for label, keywords in FOCUS_KEYWORDS.items() if _contains_any(text, keywords)]

    if "건성" in text and "moisturizing" not in desired_effects:
        desired_effects.append("moisturizing")
    if any(token in text for token in ["민감", "예민", "순한", "자극 적은"]):
        if "soothing" not in desired_effects:
            desired_effects.append("soothing")
        if "fragrance" not in avoid_effects:
            avoid_effects.append("fragrance")
    if any(token in text for token in ["클렌징", "세안", "폼클렌징", "클렌징폼"]):
        if "클렌징/필링" not in category_2:
            category_2.append("클렌징/필링")
        if "cleansing" not in desired_effects:
            desired_effects.append("cleansing")
    if any(token in text for token in ["선크림", "자외선", "선케어"]):
        if "자외선차단제" not in category_2:
            category_2.append("자외선차단제")
        if "sun_care" not in desired_effects:
            desired_effects.append("sun_care")
    if any(token in text for token in ["립밤", "립케어", "입술"]):
        if "립케어" not in category_2:
            category_2.append("립케어")
        if "lip_care" not in desired_effects:
            desired_effects.append("lip_care")
    if any(token in text for token in ["커버", "쿠션", "컨실러", "파운데이션"]):
        if "makeup_cover" not in desired_effects:
            desired_effects.append("makeup_cover")
    if any(token in text for token in ["미백", "브라이트닝", "톤업", "칙칙"]):
        if "brightening" not in desired_effects:
            desired_effects.append("brightening")
    if not category_1 and not category_2 and any(token in text for token in ["보습", "수분", "토너", "크림", "세럼", "로션"]):
        category_1.append("스킨케어")

    notes = _build_notes(min_budget, max_budget, category_1, category_2, skin_types)

    return ParsedQuery(
        raw_query=text,
        budget_min=min_budget,
        budget_max=max_budget,
        category_1=_dedupe(category_1),
        category_2=_dedupe(category_2),
        product_keywords=_dedupe(product_keywords),
        desired_aspects=_dedupe(desired_aspects),
        desired_effects=_dedupe(desired_effects),
        avoid_effects=_dedupe(avoid_effects),
        skin_types=_dedupe(skin_types),
        focus=_dedupe(focus),
        notes=notes,
        top_k=_extract_top_k(text),
        parser="rule",
    )


def _strip_code_fence(payload: str) -> str:
    cleaned = payload.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def _merge_lists(seed_values: list[str], extra_values: list[str]) -> list[str]:
    return _dedupe(list(extra_values) + list(seed_values))


def _merge_structured(seed: ParsedQuery, payload: dict[str, Any], parser_name: str) -> ParsedQuery:
    merged: dict[str, Any] = seed.model_dump()
    for field in LIST_FIELDS:
        incoming = payload.get(field)
        if incoming:
            merged[field] = _merge_lists(merged.get(field, []), list(incoming))
    for field in ("budget_min", "budget_max", "top_k"):
        if payload.get(field) is not None:
            merged[field] = payload[field]

    merged["notes"] = _build_notes(
        merged.get("budget_min"),
        merged.get("budget_max"),
        merged.get("category_1", []),
        merged.get("category_2", []),
        merged.get("skin_types", []),
    )
    merged["parser"] = parser_name
    return ParsedQuery.model_validate(merged)


def parse_query_with_local_model(message: str, seed: ParsedQuery) -> ParsedQuery:
    payload = parse_with_local_model(message)
    if not payload:
        return seed
    return _merge_structured(seed, payload, "local")


def parse_query_with_openai(message: str, seed: ParsedQuery) -> ParsedQuery:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return seed

    model = os.getenv("DAISO_CHATBOT_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    system_prompt = """
You extract shopping constraints for a Korean Daiso beauty recommendation engine.
Return JSON only.
Allowed category_1: ["스킨케어", "메이크업", "맨케어"]
Allowed category_2: ["기초스킨케어", "립케어", "자외선차단제", "클렌징/필링", "팩/마스크", "베이스메이크업", "립메이크업", "아이메이크업", "치크/하이라이터", "남성스킨케어", "남성향수", "남성용면도기", "클렌징/쉐이빙", "남성메이크업"]
Allowed desired_effects: ["moisturizing", "soothing", "cleansing", "brightening", "sun_care", "lip_care", "makeup_cover"]
Allowed avoid_effects: ["fragrance"]
Allowed skin_types: ["dry", "oily", "sensitive", "acne", "combination"]
Allowed desired_aspects: ["가격/가성비", "사용감/성능", "재구매", "색상/발색", "제형/향", "용량/휴대"]
Allowed focus: ["가성비", "순함", "인기", "재구매"]
Schema:
{
  "budget_min": number|null,
  "budget_max": number|null,
  "category_1": string[],
  "category_2": string[],
  "product_keywords": string[],
  "desired_aspects": string[],
  "desired_effects": string[],
  "avoid_effects": string[],
  "skin_types": string[],
  "focus": string[],
  "notes": string[],
  "top_k": integer
}
Prefer empty arrays over invented values.
""".strip()

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"User query: {message}\n"
                    f"Rule-based seed: {json.dumps(seed.model_dump(), ensure_ascii=False)}"
                ),
            },
        ],
        max_output_tokens=500,
    )
    payload = _strip_code_fence(response.output_text)
    data = json.loads(payload)
    return _merge_structured(seed, data, "openai")


def _resolve_backend() -> str:
    backend = os.getenv("DAISO_QUERY_PARSER_BACKEND", "auto").strip().lower()
    if backend in {"local", "openai", "rule", "auto"}:
        return backend
    return "auto"


def parse_query(message: str) -> ParsedQuery:
    seed = parse_query_rule_based(message)
    backend = _resolve_backend()

    if backend == "rule":
        return seed

    if backend in {"local", "auto"}:
        try:
            local_result = parse_query_with_local_model(message, seed)
            if local_result.parser == "local":
                return local_result
        except Exception:
            if backend == "local":
                return seed

    if backend in {"openai", "auto"}:
        try:
            return parse_query_with_openai(message, seed)
        except Exception:
            return seed

    return seed
