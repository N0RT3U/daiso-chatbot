from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .models import ParsedQuery, RecommendationCard, RecommendationResponse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = REPO_ROOT / "data" / "chatbot" / "daiso_chatbot_catalog.csv"
PRODUCT_URL_TEMPLATE = "https://www.daisomall.co.kr/pd/pdr/SCR_PDR_0001?pdNo={product_code}&recmYn=N"
STORE_STOCK_URL = "https://www.daisomall.co.kr/ms/msg/SCR_MSG_0015"

BASE_WEIGHTS = {
    "sentiment": 0.30,
    "soft_landing": 0.20,
    "popularity": 0.15,
    "rating": 0.10,
    "value": 0.10,
    "repurchase": 0.10,
    "ingredient": 0.05,
}

WEIGHT_BOOSTS = {
    "가성비": {"value": 0.08, "popularity": -0.03, "soft_landing": -0.02, "rating": -0.02, "ingredient": -0.01},
    "순함": {"ingredient": 0.10, "sentiment": 0.05, "popularity": -0.04, "soft_landing": -0.03, "value": -0.03, "rating": -0.03, "repurchase": -0.02},
    "인기": {"popularity": 0.10, "soft_landing": 0.05, "value": -0.03, "ingredient": -0.03, "repurchase": -0.04, "rating": -0.02, "sentiment": -0.03},
    "재구매": {"repurchase": 0.10, "soft_landing": 0.02, "popularity": -0.03, "value": -0.03, "ingredient": -0.02, "rating": -0.02, "sentiment": -0.02},
}

ASPECT_COLUMN_MAP = {
    "가격/가성비": "price_value",
    "사용감/성능": "performance",
    "재구매": "repurchase",
    "색상/발색": "color",
    "제형/향": "material_smell",
    "용량/휴대": "size_portability",
}

DEFAULT_ASPECT_MIX = {
    "사용감/성능": 0.45,
    "가격/가성비": 0.20,
    "재구매": 0.15,
    "제형/향": 0.10,
    "용량/휴대": 0.10,
}

EFFECT_KEYWORDS = {
    "moisturizing": ["보습", "수분", "촉촉", "히알루론", "세라마이드", "판테놀", "모이스처"],
    "soothing": ["진정", "순한", "민감", "예민", "자극", "알로에", "병풀", "시카"],
    "cleansing": ["클렌징", "세안", "노폐물", "각질", "피지", "워터", "오일", "폼"],
    "brightening": ["미백", "톤업", "비타민", "브라이트닝", "칙칙"],
    "sun_care": ["선크림", "자외선", "선케어", "uv", "sunscreen"],
    "lip_care": ["립밤", "립", "입술", "립케어"],
    "makeup_cover": ["커버", "컨실러", "쿠션", "파운데이션", "베이스"],
}

CATEGORY_DISPLAY = {
    "스킨케어": "스킨케어",
    "메이크업": "메이크업",
    "맨케어": "남성용",
    "기초스킨케어": "스킨케어",
    "립케어": "립케어",
    "자외선차단제": "선크림",
    "클렌징/필링": "클렌징",
    "팩/마스크": "팩/마스크",
    "베이스메이크업": "베이스 메이크업",
    "립메이크업": "립 메이크업",
    "아이메이크업": "아이 메이크업",
    "치크/하이라이터": "치크/하이라이터",
    "남성스킨케어": "남성 스킨케어",
    "남성향수": "남성 향수",
    "남성용면도기": "면도용품",
    "클렌징/쉐이빙": "쉐이빙",
    "남성메이크업": "남성 메이크업",
}

SKIN_TYPE_DISPLAY = {
    "dry": "건성",
    "oily": "지성",
    "sensitive": "민감성",
    "acne": "트러블 피부",
    "combination": "복합성",
}

EFFECT_DISPLAY = {
    "moisturizing": "보습",
    "soothing": "순한 제품",
    "cleansing": "클렌징",
    "brightening": "미백/톤업",
    "sun_care": "선케어",
    "lip_care": "립케어",
    "makeup_cover": "커버력",
}

FOCUS_DISPLAY = {
    "가성비": "가성비",
    "순함": "순한 제품",
    "인기": "인기 많은 제품",
    "재구매": "재구매 많은 제품",
}


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(value, 0.02) for key, value in weights.items()}
    total = sum(clipped.values())
    return {key: value / total for key, value in clipped.items()}


def humanize_filter_label(label: str) -> str:
    if label in CATEGORY_DISPLAY:
        return CATEGORY_DISPLAY[label]
    if label in FOCUS_DISPLAY:
        return FOCUS_DISPLAY[label]
    if label == "조건 완화":
        return "조건을 조금 넓혀서 추천"
    return label


@lru_cache(maxsize=4)
def load_catalog(catalog_path: str = str(DEFAULT_CATALOG_PATH)) -> pd.DataFrame:
    frame = pd.read_csv(catalog_path)
    for column in ["effect_tags", "ingredient_types", "key_ingredients"]:
        frame[column] = frame[column].fillna("[]").map(json.loads)
    frame["final_soft_landing"] = frame["final_soft_landing"].fillna(False).astype(bool)
    frame["product_name"] = frame["product_name"].fillna("")
    frame["brand_name"] = frame["brand_name"].fillna("")
    frame["category_1"] = frame["category_1"].fillna("")
    frame["category_2"] = frame["category_2"].fillna("")
    frame["search_text"] = frame["search_text"].fillna("")
    return frame


def build_factor_weights(query: ParsedQuery) -> dict[str, float]:
    weights = BASE_WEIGHTS.copy()
    focus_items = list(query.focus)
    if query.budget_max is not None and "가성비" not in focus_items:
        focus_items.append("가성비")
    for focus in focus_items:
        boosts = WEIGHT_BOOSTS.get(focus)
        if not boosts:
            continue
        for factor, delta in boosts.items():
            weights[factor] = weights.get(factor, 0.0) + delta
    return _normalize_weights(weights)


def choose_aspect_mix(query: ParsedQuery) -> dict[str, float]:
    mix = DEFAULT_ASPECT_MIX.copy()
    if "사용감/성능" in query.desired_aspects:
        mix["사용감/성능"] += 0.15
    if "가격/가성비" in query.desired_aspects:
        mix["가격/가성비"] += 0.15
    if "재구매" in query.desired_aspects:
        mix["재구매"] += 0.20
    if "색상/발색" in query.desired_aspects:
        mix["색상/발색"] = mix.get("색상/발색", 0.20) + 0.25
    if "제형/향" in query.desired_aspects:
        mix["제형/향"] += 0.10
    if any(category in query.category_2 for category in ["립메이크업", "베이스메이크업", "아이메이크업", "치크/하이라이터"]):
        mix["색상/발색"] = mix.get("색상/발색", 0.15) + 0.15
    total = sum(mix.values())
    return {aspect: value / total for aspect, value in mix.items()}


def filter_candidates(frame: pd.DataFrame, query: ParsedQuery) -> tuple[pd.DataFrame, list[str]]:
    filtered = frame.copy()
    applied: list[str] = []

    if query.category_1:
        filtered = filtered[filtered["category_1"].isin(query.category_1)]
        applied.extend(query.category_1)
    if query.category_2:
        next_frame = filtered[filtered["category_2"].isin(query.category_2)]
        if not next_frame.empty:
            filtered = next_frame
            applied.extend(query.category_2)
    if query.budget_min is not None:
        filtered = filtered[filtered["price"] >= query.budget_min]
        applied.append(f"{int(query.budget_min):,}원 이상")
    if query.budget_max is not None:
        filtered = filtered[filtered["price"] <= query.budget_max]
        applied.append(f"{int(query.budget_max):,}원 이하")

    if query.product_keywords:
        pattern = "|".join(re.escape(keyword) for keyword in query.product_keywords)
        keyword_matches = filtered["product_name"].str.contains(pattern, case=False, na=False)
        if keyword_matches.any():
            filtered = filtered[keyword_matches]
            applied.extend(query.product_keywords)

    if filtered.empty:
        filtered = frame.copy()
        applied.append("조건 완화")

    return filtered, applied


def keyword_signal(frame: pd.DataFrame, effect: str) -> pd.Series:
    keywords = EFFECT_KEYWORDS.get(effect, [])
    if not keywords:
        return pd.Series(0.0, index=frame.index)
    pattern = "|".join(re.escape(keyword.lower()) for keyword in keywords)
    return frame["search_text"].str.contains(pattern, case=False, na=False).astype(float)


def compute_ingredient_score(frame: pd.DataFrame, query: ParsedQuery) -> pd.Series:
    score = (frame["safety_rank"] * 0.55) + (frame["fragrance_safe_rank"] * 0.15)

    effect_contributors: list[pd.Series] = []
    for effect in query.desired_effects:
        if effect == "moisturizing":
            effect_contributors.append((frame["effect_rank_moisturizing"] * 0.7) + (keyword_signal(frame, effect) * 0.3))
        elif effect == "soothing":
            effect_contributors.append((frame["safety_rank"] * 0.5) + (keyword_signal(frame, effect) * 0.5))
        elif effect == "cleansing":
            effect_contributors.append((frame["effect_rank_cleansing"] * 0.7) + (keyword_signal(frame, effect) * 0.3))
        elif effect == "brightening":
            effect_contributors.append((frame["effect_rank_active"] * 0.6) + (keyword_signal(frame, effect) * 0.4))
        else:
            effect_contributors.append(keyword_signal(frame, effect))

    if effect_contributors:
        merged = sum(effect_contributors) / len(effect_contributors)
        score = (score * 0.45) + (merged * 0.55)

    if "fragrance" in query.avoid_effects:
        score = score - (frame["effect_ratio_fragrance"] * 0.25)

    return score.clip(lower=0.0, upper=1.0)


def compute_sentiment_score(frame: pd.DataFrame, query: ParsedQuery) -> pd.Series:
    aspect_mix = choose_aspect_mix(query)
    score = frame["sentiment_rank"] * 0.35
    for aspect, weight in aspect_mix.items():
        slug = ASPECT_COLUMN_MAP[aspect]
        score = score + (frame[f"aspect_rank_{slug}"] * (0.65 * weight))
    return score.clip(lower=0.0, upper=1.0)


def build_badges(row: pd.Series) -> list[str]:
    badges: list[str] = []
    if bool(row["final_soft_landing"]):
        badges.append("꾸준히 인기")
    if row["value_rank"] >= 0.85:
        badges.append("가성비 좋아요")
    if row["repurchase_rank"] >= 0.85:
        badges.append("재구매 많아요")
    if row["safety_rank"] >= 0.85:
        badges.append("순한 편")
    if row["popularity_rank"] >= 0.90:
        badges.append("리뷰 많아요")
    return badges[:4]


def factor_reason(name: str, value: float, row: pd.Series) -> str:
    if name == "sentiment":
        return f"리뷰 반응이 좋아요. 긍정 리뷰가 {row['overall_pos_rate'] * 100:.1f}%예요."
    if name == "soft_landing":
        if bool(row["final_soft_landing"]):
            return "한때만 뜬 제품보다, 꾸준히 찾는 편에 가까워요."
        return f"요즘 반응이 비교적 안정적인 편이에요. 참고 점수는 {row['ml_prob']:.2f}예요."
    if name == "popularity":
        return f"리뷰가 {int(row['review_count']):,}개라서 많이 본 제품이에요."
    if name == "rating":
        return f"평점이 {row['avg_rating']:.2f}점으로 높은 편이에요."
    if name == "value":
        return f"가격 대비 만족도가 좋아요. 가성비 지표는 {row['cp_index']:.1f}이에요."
    if name == "repurchase":
        return f"재구매 비율이 {row['reorder_rate'] * 100:.1f}%로 높은 편이에요."
    return f"성분 쪽 반응이 무난한 편이에요. 주의 점수는 {row['risk_score']:.0f}점이에요."


def choose_snippet(row: pd.Series, query: ParsedQuery) -> str:
    if "가격/가성비" in query.desired_aspects and row.get("snippet_value"):
        return row["snippet_value"]
    if "재구매" in query.desired_aspects and row.get("snippet_repurchase"):
        return row["snippet_repurchase"]
    if any(effect in query.desired_effects for effect in ["moisturizing", "soothing", "cleansing"]) and row.get("snippet_performance"):
        return row["snippet_performance"]
    if row.get("snippet_positive"):
        return row["snippet_positive"]
    return ""


def build_warning(row: pd.Series, query: ParsedQuery) -> list[str]:
    warnings: list[str] = []
    if row["risk_score"] >= 8 and any(effect in query.desired_effects for effect in ["soothing", "moisturizing"]):
        warnings.append("민감한 피부라면 성분표를 한 번 더 확인해보는 게 좋아요.")
    if row["effect_ratio_fragrance"] >= 0.15 and "fragrance" in query.avoid_effects:
        warnings.append("향 관련 성분 비중이 조금 있어서 무향을 원하면 확인이 필요해요.")
    if row["overall_neg_rate"] >= 0.10:
        warnings.append("좋다는 반응도 많지만 아쉽다는 의견도 조금 있는 제품이에요.")
    return warnings[:1]


def summarize_query(query: ParsedQuery, result_count: int) -> str:
    if result_count == 0:
        return "딱 맞는 상품을 찾지 못했어요. 조건을 조금만 바꿔서 다시 찾아볼까요?"

    pieces: list[str] = []
    if query.skin_types:
        pieces.append(", ".join(SKIN_TYPE_DISPLAY[item] for item in query.skin_types if item in SKIN_TYPE_DISPLAY))
    if query.desired_effects:
        pieces.append(", ".join(EFFECT_DISPLAY[item] for item in query.desired_effects if item in EFFECT_DISPLAY))
    if query.budget_max is not None:
        pieces.append(f"{int(query.budget_max):,}원 이하")
    if query.category_2:
        pieces.append(", ".join(CATEGORY_DISPLAY.get(item, item) for item in query.category_2))
    elif query.category_1:
        pieces.append(", ".join(CATEGORY_DISPLAY.get(item, item) for item in query.category_1))
    if query.focus:
        pieces.append(", ".join(FOCUS_DISPLAY.get(item, item) for item in query.focus))

    if pieces:
        return f"{' · '.join(pieces)} 기준으로 보기 좋은 상품 {result_count}개를 골라봤어요!"
    return f"추천해볼 만한 상품 {result_count}개를 골라봤어요!"


def recommend_products(query: ParsedQuery, catalog_path: Path | None = None) -> RecommendationResponse:
    frame = load_catalog(str(catalog_path or DEFAULT_CATALOG_PATH))
    candidates, applied_filters = filter_candidates(frame, query)
    weights = build_factor_weights(query)

    working = candidates.copy()
    working["factor_sentiment"] = compute_sentiment_score(working, query)
    working["factor_soft_landing"] = working["sli_rank"]
    working["factor_popularity"] = working["popularity_rank"]
    working["factor_rating"] = working["rating_rank"]
    working["factor_value"] = working["value_rank"]
    working["factor_repurchase"] = working["repurchase_rank"]
    working["factor_ingredient"] = compute_ingredient_score(working, query)

    working["final_score"] = (
        working["factor_sentiment"] * weights["sentiment"]
        + working["factor_soft_landing"] * weights["soft_landing"]
        + working["factor_popularity"] * weights["popularity"]
        + working["factor_rating"] * weights["rating"]
        + working["factor_value"] * weights["value"]
        + working["factor_repurchase"] * weights["repurchase"]
        + working["factor_ingredient"] * weights["ingredient"]
    )

    if query.budget_max is not None:
        budget_bonus = 1 - (working["price"] / query.budget_max).clip(upper=1)
        working["final_score"] = working["final_score"] + (budget_bonus * 0.03)

    working = working.sort_values(
        ["final_score", "review_count", "avg_rating"],
        ascending=[False, False, False],
    )

    cards: list[RecommendationCard] = []
    for _, row in working.head(query.top_k).iterrows():
        breakdown = {
            "sentiment": float(row["factor_sentiment"]),
            "soft_landing": float(row["factor_soft_landing"]),
            "popularity": float(row["factor_popularity"]),
            "rating": float(row["factor_rating"]),
            "value": float(row["factor_value"]),
            "repurchase": float(row["factor_repurchase"]),
            "ingredient": float(row["factor_ingredient"]),
        }
        contributions = {factor: breakdown[factor] * weights[factor] for factor in breakdown}
        ranked_factors = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
        reasons = [factor_reason(name, breakdown[name], row) for name, _ in ranked_factors[:3]]

        cards.append(
            RecommendationCard(
                product_code=int(row["product_code"]),
                name=str(row["product_name"]),
                brand_name=str(row["brand_name"]),
                category_1=str(CATEGORY_DISPLAY.get(row["category_1"], row["category_1"])),
                category_2=str(CATEGORY_DISPLAY.get(row["category_2"], row["category_2"])),
                product_url=PRODUCT_URL_TEMPLATE.format(product_code=int(row["product_code"])),
                inventory_url=STORE_STOCK_URL,
                price=float(row["price"]),
                review_count=int(row["review_count"]),
                avg_rating=float(row["avg_rating"]),
                reorder_rate=float(row["reorder_rate"]),
                final_score=float(row["final_score"]),
                badges=build_badges(row),
                reasons=reasons,
                warnings=build_warning(row, query),
                snippet=choose_snippet(row, query),
                breakdown=breakdown,
                metrics={
                    "cp_index": round(float(row["cp_index"]), 1),
                    "risk_score": round(float(row["risk_score"]), 1),
                    "overall_pos_rate": round(float(row["overall_pos_rate"]) * 100, 1),
                    "sli_confidence": round(float(row["confidence"]), 2),
                },
            )
        )

    summary = summarize_query(query, len(cards))
    display_filters = [humanize_filter_label(item) for item in (applied_filters or query.notes)]
    display_filters = list(dict.fromkeys(display_filters))

    return RecommendationResponse(
        query=query,
        summary=summary,
        applied_filters=display_filters,
        results=cards,
        parser_note="",
    )
