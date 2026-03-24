from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE_ROOT = Path("G:/Final_proj/Total_clear/데이터/data")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "chatbot" / "daiso_chatbot_catalog.csv"

ASPECT_SLUGS = {
    "가격/가성비": "price_value",
    "디자인": "design",
    "배송/포장": "shipping",
    "사용감/성능": "performance",
    "색상/발색": "color",
    "용량/휴대": "size_portability",
    "재구매": "repurchase",
    "재질/냄새": "material_smell",
}

EFFECT_MAP = {
    "moisturizing": "Moisturizing",
    "active": "Active",
    "cleansing": "Cleansing",
    "texture": "Texture",
    "color": "Color",
    "base": "Base",
    "fragrance": "Fragrance",
    "preservative": "Preservative",
}


def percentile_rank(series: pd.Series) -> pd.Series:
    filled = series.fillna(series.min() if not series.dropna().empty else 0)
    return filled.rank(method="average", pct=True)


def squash_text(value: str) -> str:
    return " ".join(str(value or "").split())


def first_text(group: pd.DataFrame, mask: pd.Series) -> str:
    subset = group.loc[mask].copy()
    if subset.empty:
        return ""
    subset["text"] = subset["text"].map(squash_text)
    subset = subset[subset["text"].str.len() >= 12]
    if subset.empty:
        return ""
    return subset.iloc[0]["text"]


def build_catalog(source_root: Path, output_path: Path) -> pd.DataFrame:
    analysis_root = source_root / "analysis"
    erd_root = source_root / "ERD_final" / "final_fix"
    text_root = source_root / "text"
    ingredient_root = source_root / "ingred"

    product_kpi = pd.read_csv(text_root / "product_kpi.csv")
    products_core = pd.read_csv(erd_root / "products_core.csv")
    products_stats = pd.read_csv(erd_root / "products_stats.csv")
    risk_scores = pd.read_csv(ingredient_root / "product_risk_score.csv").rename(
        columns={"product_id": "product_code", "score": "risk_score_external"}
    )
    ratings = (
        pd.read_csv(
            analysis_root / "reviews_full.csv",
            usecols=["product_code", "rating", "is_reorder"],
        )
        .groupby("product_code", as_index=False)
        .agg(
            avg_rating=("rating", "mean"),
            reorder_rate=("is_reorder", "mean"),
            review_rows=("rating", "size"),
        )
    )
    sli_results = pd.read_csv(erd_root / "sli_results.csv")

    ingredient_frame = pd.read_csv(
        ingredient_root / "products_ingredients_merged.csv",
        usecols=[
            "product_code",
            "ingredient",
            "ingredient_type",
            "effect",
            "allergic",
        ],
    )
    ingredient_rows: list[dict[str, object]] = []
    for product_code, group in ingredient_frame.groupby("product_code", sort=False):
        group = group.copy()
        group["ingredient"] = group["ingredient"].fillna("").astype(str)
        group["ingredient_type"] = group["ingredient_type"].fillna("").astype(str)
        group["effect"] = group["effect"].fillna("").astype(str)
        ingredient_total = max(len(group), 1)

        effect_tags = sorted({value for value in group["effect"] if value})
        ingredient_types = (
            group.loc[group["ingredient_type"] != "", "ingredient_type"]
            .value_counts()
            .head(12)
            .index.tolist()
        )
        key_ingredients = (
            group.loc[group["ingredient"] != "", "ingredient"]
            .value_counts()
            .head(12)
            .index.tolist()
        )

        row: dict[str, object] = {
            "product_code": product_code,
            "ingredient_count": int(ingredient_total),
            "allergic_count": int(group["allergic"].fillna(0).astype(int).sum()),
            "effect_tags": json.dumps(effect_tags, ensure_ascii=False),
            "ingredient_types": json.dumps(ingredient_types, ensure_ascii=False),
            "key_ingredients": json.dumps(key_ingredients, ensure_ascii=False),
            "search_text_extra": " ".join(effect_tags + key_ingredients + ingredient_types),
        }
        for slug, effect_name in EFFECT_MAP.items():
            row[f"effect_ratio_{slug}"] = float((group["effect"] == effect_name).mean())
        row["fragrance_ratio"] = float(
            ((group["effect"] == "Fragrance") | (group["ingredient_type"] == "Fragrance")).mean()
        )
        ingredient_rows.append(row)

    ingredient_summary = pd.DataFrame(ingredient_rows)

    review_level = pd.read_csv(
        text_root / "review_level.csv",
        usecols=["product_code", "text", "summary", "sentiment", "sentiment_score", "rating"],
    )
    review_level["text"] = review_level["text"].fillna("").astype(str)
    review_level["summary"] = review_level["summary"].fillna("").astype(str)
    review_level = review_level.sort_values(
        ["rating", "sentiment_score", "text"],
        ascending=[False, False, True],
    )

    snippet_rows: list[dict[str, object]] = []
    for product_code, group in review_level.groupby("product_code", sort=False):
        positive_mask = group["sentiment"].eq("positive")
        snippet_rows.append(
            {
                "product_code": product_code,
                "snippet_positive": first_text(group, positive_mask),
                "snippet_performance": first_text(
                    group,
                    positive_mask & group["summary"].str.contains("사용감/성능 긍정적", na=False),
                ),
                "snippet_value": first_text(
                    group,
                    positive_mask & group["summary"].str.contains("가격/가성비 긍정적", na=False),
                ),
                "snippet_repurchase": first_text(
                    group,
                    positive_mask & group["summary"].str.contains("재구매 긍정적", na=False),
                ),
                "snippet_negative": first_text(group, group["sentiment"].eq("negative")),
            }
        )
    snippets = pd.DataFrame(snippet_rows)

    catalog = (
        product_kpi.merge(
            products_core[["product_code", "manufacturer_id", "country"]],
            on="product_code",
            how="left",
        )
        .merge(
            products_stats[
                [
                    "product_code",
                    "likes",
                    "shares",
                    "review_count",
                    "engagement_score",
                    "cp_index",
                    "review_density",
                    "risk_score",
                    "first_review_date",
                ]
            ],
            on="product_code",
            how="left",
        )
        .merge(ratings, on="product_code", how="left")
        .merge(
            risk_scores[["product_code", "risk_score_external"]],
            on="product_code",
            how="left",
        )
        .merge(
            sli_results[
                [
                    "product_code",
                    "final_soft_landing",
                    "confidence",
                    "ml_prob",
                    "total_votes",
                ]
            ],
            on="product_code",
            how="left",
        )
        .merge(ingredient_summary, on="product_code", how="left")
        .merge(snippets, on="product_code", how="left")
    )

    catalog["review_count"] = catalog["review_count"].fillna(catalog["n_reviews"]).fillna(0).astype(int)
    catalog["avg_rating"] = catalog["avg_rating"].fillna(0.0)
    catalog["reorder_rate"] = catalog["reorder_rate"].fillna(catalog["repurchase_mention_rate"]).fillna(0.0)
    catalog["risk_score"] = catalog["risk_score"].fillna(catalog["risk_score_external"]).fillna(0.0)
    catalog["confidence"] = catalog["confidence"].fillna(0.0)
    catalog["ml_prob"] = catalog["ml_prob"].fillna(0.0)
    catalog["final_soft_landing"] = catalog["final_soft_landing"].fillna(False)
    catalog["total_votes"] = catalog["total_votes"].fillna(0).astype(int)
    catalog["allergic_count"] = catalog["allergic_count"].fillna(0).astype(int)
    catalog["ingredient_count"] = catalog["ingredient_count"].fillna(0).astype(int)
    catalog["effect_tags"] = catalog["effect_tags"].fillna("[]")
    catalog["ingredient_types"] = catalog["ingredient_types"].fillna("[]")
    catalog["key_ingredients"] = catalog["key_ingredients"].fillna("[]")
    catalog["search_text_extra"] = catalog["search_text_extra"].fillna("")
    for column in [
        "snippet_positive",
        "snippet_performance",
        "snippet_value",
        "snippet_repurchase",
        "snippet_negative",
    ]:
        catalog[column] = catalog[column].fillna("")

    for slug in EFFECT_MAP:
        column = f"effect_ratio_{slug}"
        catalog[column] = catalog[column].fillna(0.0)

    catalog["sli_raw"] = np.where(
        catalog["final_soft_landing"],
        0.7 + (catalog["confidence"] * 0.3),
        catalog["ml_prob"] * 0.3,
    )
    catalog["safety_raw"] = 1 - np.clip(catalog["risk_score"] / 15, 0, 1)
    catalog["fragrance_safe_raw"] = 1 - np.clip(catalog["fragrance_ratio"], 0, 1)
    catalog["sentiment_raw"] = (
        (catalog["overall_pos_rate"].fillna(0.0) * 0.65)
        + ((1 - catalog["overall_neg_rate"].fillna(0.0)) * 0.35)
    )

    catalog["popularity_rank"] = percentile_rank(catalog["engagement_score"])
    catalog["rating_rank"] = percentile_rank(catalog["avg_rating"])
    catalog["value_rank"] = percentile_rank(catalog["cp_index"])
    catalog["repurchase_rank"] = percentile_rank(catalog["reorder_rate"])
    catalog["sli_rank"] = percentile_rank(catalog["sli_raw"])
    catalog["safety_rank"] = percentile_rank(catalog["safety_raw"])
    catalog["fragrance_safe_rank"] = percentile_rank(catalog["fragrance_safe_raw"])
    catalog["sentiment_rank"] = percentile_rank(catalog["sentiment_raw"])

    for aspect, slug in ASPECT_SLUGS.items():
        mention_col = f"mention_rate_{aspect}"
        neg_col = f"neg_rate_{aspect}"
        signal = catalog[mention_col].fillna(0.0) * (1 - catalog[neg_col].fillna(0.0))
        catalog[f"aspect_signal_{slug}"] = signal
        catalog[f"aspect_rank_{slug}"] = percentile_rank(signal)

    for slug in EFFECT_MAP:
        raw_col = f"effect_ratio_{slug}"
        catalog[f"effect_rank_{slug}"] = percentile_rank(catalog[raw_col])

    catalog["search_text"] = (
        catalog["product_name"].fillna("")
        + " "
        + catalog["brand_name"].fillna("")
        + " "
        + catalog["category_1"].fillna("")
        + " "
        + catalog["category_2"].fillna("")
        + " "
        + catalog["search_text_extra"].fillna("")
    ).str.lower()

    preferred_order = [
        "product_code",
        "product_name",
        "brand_name",
        "price",
        "category_1",
        "category_2",
        "review_count",
        "avg_rating",
        "reorder_rate",
        "overall_pos_rate",
        "overall_neg_rate",
        "engagement_score",
        "cp_index",
        "risk_score",
        "final_soft_landing",
        "confidence",
        "ml_prob",
        "effect_tags",
        "ingredient_types",
        "key_ingredients",
        "snippet_positive",
        "snippet_performance",
        "snippet_value",
        "snippet_repurchase",
        "snippet_negative",
    ]
    ordered_columns = preferred_order + [
        column for column in catalog.columns if column not in preferred_order
    ]
    catalog = catalog[ordered_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False, encoding="utf-8-sig")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Daiso chatbot catalog.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Path to the prepared Daiso data root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path for the chatbot catalog.",
    )
    args = parser.parse_args()

    catalog = build_catalog(args.source_root, args.output)
    print(f"catalog_rows={len(catalog)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
