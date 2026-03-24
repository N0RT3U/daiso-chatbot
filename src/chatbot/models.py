from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParsedQuery(BaseModel):
    raw_query: str
    budget_min: float | None = None
    budget_max: float | None = None
    category_1: list[str] = Field(default_factory=list)
    category_2: list[str] = Field(default_factory=list)
    product_keywords: list[str] = Field(default_factory=list)
    desired_aspects: list[str] = Field(default_factory=list)
    desired_effects: list[str] = Field(default_factory=list)
    avoid_effects: list[str] = Field(default_factory=list)
    skin_types: list[str] = Field(default_factory=list)
    focus: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    top_k: int = 5
    parser: str = "rule"


class RecommendationCard(BaseModel):
    product_code: int
    name: str
    brand_name: str
    category_1: str
    category_2: str
    product_url: str = ""
    inventory_url: str = ""
    price: float
    review_count: int
    avg_rating: float
    reorder_rate: float
    final_score: float
    badges: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    snippet: str = ""
    breakdown: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    query: ParsedQuery
    summary: str
    applied_filters: list[str] = Field(default_factory=list)
    results: list[RecommendationCard] = Field(default_factory=list)
    parser_note: str = ""
