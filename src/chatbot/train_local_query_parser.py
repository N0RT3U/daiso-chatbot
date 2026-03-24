from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chatbot.local_query_parser import FIELD_LABELS, MultiHeadQueryParserModel
else:  # pragma: no cover - import path depends on execution mode
    from .local_query_parser import FIELD_LABELS, MultiHeadQueryParserModel


SKIN_PHRASES = {
    "dry": ["건성인데", "속건조 심한데", "건조한 피부인데"],
    "oily": ["지성인데", "유분 많은 편인데", "번들거리는 피부인데"],
    "sensitive": ["민감성인데", "예민한 피부인데", "자극 적은 거 찾는데"],
    "acne": ["트러블 피부인데", "여드름 피부인데", "모공 고민인데"],
    "combination": ["복합성인데", "복합성 피부인데"],
}

EFFECT_PHRASES = {
    "moisturizing": ["보습 좋은", "촉촉한", "수분감 있는"],
    "soothing": ["순한", "진정에 좋은", "자극 적은"],
    "cleansing": ["세정력 좋은", "클렌징 잘 되는", "피지 정리되는"],
    "brightening": ["미백용", "톤업되는", "칙칙함 관리용"],
    "sun_care": ["선케어용", "자외선 차단되는", "선크림"],
    "lip_care": ["립케어용", "입술 보습용", "립밤"],
    "makeup_cover": ["커버력 좋은", "베이스용", "커버 잘 되는"],
}

FOCUS_PHRASES = {
    "가성비": ["가성비 좋은", "저렴한", "값어치 있는"],
    "순함": ["순한", "자극 적은", "민감성도 쓸 만한"],
    "인기": ["인기 많은", "후기 많은", "리뷰 많은"],
    "재구매": ["재구매 많은", "또 사는 사람 많은", "반복 구매 많은"],
}

CATEGORY_2_PRODUCT_PHRASES = {
    "기초스킨케어": ["스킨케어", "토너", "세럼", "크림"],
    "립케어": ["립케어", "립밤"],
    "자외선차단제": ["선크림", "자외선차단제"],
    "클렌징/필링": ["클렌징", "세안 제품", "폼클렌징", "필링"],
    "팩/마스크": ["팩", "마스크팩", "시트팩"],
    "베이스메이크업": ["베이스메이크업", "쿠션", "컨실러"],
    "립메이크업": ["립메이크업", "틴트", "립 제품"],
    "아이메이크업": ["아이메이크업", "마스카라", "섀도우"],
    "치크/하이라이터": ["치크", "하이라이터"],
    "남성스킨케어": ["남성 스킨케어", "남자 기초"],
    "남성향수": ["남성 향수", "코롱", "향수"],
    "남성용면도기": ["면도기", "면도 용품"],
    "클렌징/쉐이빙": ["쉐이빙폼", "쉐이빙젤", "면도 폼"],
    "남성메이크업": ["남성 메이크업", "남자 메이크업"],
}

BUDGET_PHRASES = [
    (None, 3000, "3천원 이하"),
    (None, 5000, "5천원 이하"),
    (3000, None, "3천원 이상"),
    (None, None, ""),
]

CATEGORY_1_BY_CATEGORY_2 = {
    "기초스킨케어": "스킨케어",
    "립케어": "스킨케어",
    "자외선차단제": "스킨케어",
    "클렌징/필링": "스킨케어",
    "팩/마스크": "스킨케어",
    "베이스메이크업": "메이크업",
    "립메이크업": "메이크업",
    "아이메이크업": "메이크업",
    "치크/하이라이터": "메이크업",
    "남성스킨케어": "맨케어",
    "남성향수": "맨케어",
    "남성용면도기": "맨케어",
    "클렌징/쉐이빙": "맨케어",
    "남성메이크업": "맨케어",
}


@dataclass
class QueryExample:
    text: str
    labels: dict[str, list[str]]


def _clone_labels(labels: dict[str, list[str]]) -> dict[str, list[str]]:
    return json.loads(json.dumps(labels, ensure_ascii=False))


def build_synthetic_examples(limit: int = 6000) -> list[QueryExample]:
    examples: list[QueryExample] = []

    for category_2, product_terms in CATEGORY_2_PRODUCT_PHRASES.items():
        category_1 = CATEGORY_1_BY_CATEGORY_2[category_2]
        for product_term, (_, _, budget_phrase) in itertools.product(product_terms, BUDGET_PHRASES):
            base_labels = {
                "category_1": [category_1],
                "category_2": [category_2],
                "desired_aspects": [],
                "desired_effects": [],
                "avoid_effects": [],
                "skin_types": [],
                "focus": [],
            }

            sentence = f"{budget_phrase} {product_term} 추천해줘".strip()
            examples.append(QueryExample(text=sentence, labels=_clone_labels(base_labels)))

            for focus, focus_terms in FOCUS_PHRASES.items():
                labels = _clone_labels(base_labels)
                labels["focus"] = [focus]
                if focus == "가성비":
                    labels["desired_aspects"] = ["가격/가성비"]
                elif focus == "재구매":
                    labels["desired_aspects"] = ["재구매"]
                elif focus == "순함":
                    labels["desired_effects"] = ["soothing"]
                    labels["avoid_effects"] = ["fragrance"]

                sentence = f"{budget_phrase} {random.choice(focus_terms)} {product_term} 있어?".strip()
                examples.append(QueryExample(text=sentence, labels=labels))

            for skin_type, skin_terms in SKIN_PHRASES.items():
                labels = _clone_labels(base_labels)
                labels["skin_types"] = [skin_type]
                if skin_type == "dry":
                    labels["desired_effects"] = ["moisturizing"]
                elif skin_type == "oily":
                    labels["desired_effects"] = ["cleansing"]
                elif skin_type == "sensitive":
                    labels["desired_effects"] = ["soothing"]
                    labels["avoid_effects"] = ["fragrance"]

                sentence = f"{random.choice(skin_terms)} {budget_phrase} {product_term} 추천".strip()
                examples.append(QueryExample(text=sentence, labels=labels))

            for effect, effect_terms in EFFECT_PHRASES.items():
                labels = _clone_labels(base_labels)
                labels["desired_effects"] = [effect]
                if effect == "soothing":
                    labels["avoid_effects"] = ["fragrance"]
                if effect == "makeup_cover":
                    labels["desired_aspects"] = ["사용감/성능"]

                sentence = f"{budget_phrase} {random.choice(effect_terms)} {product_term} 추천해줘".strip()
                examples.append(QueryExample(text=sentence, labels=labels))

    random.shuffle(examples)
    return examples[:limit]


class SyntheticQueryDataset(Dataset):
    def __init__(self, examples: list[QueryExample], tokenizer, max_length: int = 96) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        for field, labels in FIELD_LABELS.items():
            target = torch.zeros(len(labels), dtype=torch.float32)
            for label in example.labels.get(field, []):
                if label in labels:
                    target[labels.index(label)] = 1.0
            item[f"labels_{field}"] = target
        return item


def train_local_parser(
    output_dir: Path,
    encoder_name: str = "beomi/KcELECTRA-base",
    max_length: int = 96,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    examples = build_synthetic_examples()
    split_index = int(len(examples) * 0.9)
    train_examples = examples[:split_index]
    valid_examples = examples[split_index:]

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    train_dataset = SyntheticQueryDataset(train_examples, tokenizer, max_length=max_length)
    valid_dataset = SyntheticQueryDataset(valid_examples, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    model = MultiHeadQueryParserModel(encoder_name=encoder_name, field_labels=FIELD_LABELS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_valid_loss = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {key: value.to(device) for key, value in batch.items() if not key.startswith("labels_")}
            labels = {
                key.replace("labels_", ""): value.to(device)
                for key, value in batch.items()
                if key.startswith("labels_")
            }
            logits = model(**inputs)
            loss = sum(criterion(logits[field], labels[field]) for field in FIELD_LABELS) / len(FIELD_LABELS)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += float(loss.item())

        model.eval()
        total_valid_loss = 0.0
        with torch.inference_mode():
            for batch in valid_loader:
                inputs = {key: value.to(device) for key, value in batch.items() if not key.startswith("labels_")}
                labels = {
                    key.replace("labels_", ""): value.to(device)
                    for key, value in batch.items()
                    if key.startswith("labels_")
                }
                logits = model(**inputs)
                loss = sum(criterion(logits[field], labels[field]) for field in FIELD_LABELS) / len(FIELD_LABELS)
                total_valid_loss += float(loss.item())

        valid_loss = total_valid_loss / max(len(valid_loader), 1)
        train_loss = total_train_loss / max(len(train_loader), 1)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), output_dir / "query_parser_model.pt")
            model.encoder.save_pretrained(output_dir / "encoder")

    tokenizer.save_pretrained(output_dir)
    config = {
        "encoder_name": encoder_name,
        "field_labels": FIELD_LABELS,
        "field_thresholds": {
            "category_1": 0.45,
            "category_2": 0.40,
            "desired_aspects": 0.45,
            "desired_effects": 0.45,
            "avoid_effects": 0.35,
            "skin_types": 0.40,
            "focus": 0.45,
        },
        "max_length": max_length,
        "dropout": 0.1,
        "best_valid_loss": best_valid_loss,
    }
    (output_dir / "parser_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local KcELECTRA query parser.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/query_parser"))
    parser.add_argument("--encoder-name", type=str, default="beomi/KcELECTRA-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    train_local_parser(
        output_dir=args.output_dir,
        encoder_name=args.encoder_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
