from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL_DIR = REPO_ROOT / "models" / "query_parser"

FIELD_LABELS: dict[str, list[str]] = {
    "category_1": ["스킨케어", "메이크업", "맨케어"],
    "category_2": [
        "기초스킨케어",
        "립케어",
        "자외선차단제",
        "클렌징/필링",
        "팩/마스크",
        "베이스메이크업",
        "립메이크업",
        "아이메이크업",
        "치크/하이라이터",
        "남성스킨케어",
        "남성향수",
        "남성용면도기",
        "클렌징/쉐이빙",
        "남성메이크업",
    ],
    "desired_aspects": [
        "가격/가성비",
        "사용감/성능",
        "재구매",
        "색상/발색",
        "제형/향",
        "용량/휴대",
    ],
    "desired_effects": [
        "moisturizing",
        "soothing",
        "cleansing",
        "brightening",
        "sun_care",
        "lip_care",
        "makeup_cover",
    ],
    "avoid_effects": ["fragrance"],
    "skin_types": ["dry", "oily", "sensitive", "acne", "combination"],
    "focus": ["가성비", "순함", "인기", "재구매"],
}


class MultiHeadQueryParserModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        field_labels: dict[str, list[str]] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.field_labels = field_labels or FIELD_LABELS
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                field: nn.Linear(hidden_size, len(labels))
                for field, labels in self.field_labels.items()
            }
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {field: head(pooled) for field, head in self.heads.items()}


class LocalQueryParser:
    def __init__(
        self,
        model: MultiHeadQueryParserModel,
        tokenizer: AutoTokenizer,
        field_thresholds: dict[str, float] | None = None,
        max_length: int = 96,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.field_labels = model.field_labels
        self.field_thresholds = field_thresholds or {}
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "LocalQueryParser":
        model_path = Path(model_dir)
        config = json.loads((model_path / "parser_config.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        encoder_dir = model_path / "encoder"
        encoder_source = str(encoder_dir) if encoder_dir.exists() else config["encoder_name"]
        model = MultiHeadQueryParserModel(
            encoder_name=encoder_source,
            field_labels=config.get("field_labels") or FIELD_LABELS,
            dropout=float(config.get("dropout", 0.1)),
        )
        state = torch.load(model_path / "query_parser_model.pt", map_location="cpu")
        model.load_state_dict(state)
        return cls(
            model=model,
            tokenizer=tokenizer,
            field_thresholds=config.get("field_thresholds"),
            max_length=int(config.get("max_length", 96)),
        )

    def parse(self, text: str) -> dict[str, list[str]]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded)

        results: dict[str, list[str]] = {}
        for field, field_logits in logits.items():
            probs = torch.sigmoid(field_logits)[0].cpu().tolist()
            labels = self.field_labels[field]
            threshold = float(self.field_thresholds.get(field, 0.5))
            chosen = [label for label, prob in zip(labels, probs) if prob >= threshold]
            if not chosen and probs:
                top_index = int(max(range(len(probs)), key=lambda idx: probs[idx]))
                if probs[top_index] >= threshold * 0.9:
                    chosen = [labels[top_index]]
            results[field] = chosen
        return results


@lru_cache(maxsize=1)
def load_local_query_parser(model_dir: str | None = None) -> LocalQueryParser | None:
    candidate = Path(model_dir or os.getenv("DAISO_LOCAL_QUERY_MODEL_DIR", str(DEFAULT_LOCAL_MODEL_DIR)))
    if not candidate.exists():
        return None
    required = [candidate / "parser_config.json", candidate / "query_parser_model.pt"]
    if not all(path.exists() for path in required):
        return None
    return LocalQueryParser.from_dir(candidate)


def parse_with_local_model(text: str, model_dir: str | None = None) -> dict[str, Any] | None:
    parser = load_local_query_parser(model_dir)
    if parser is None:
        return None
    return parser.parse(text)
