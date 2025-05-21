"""Создание препроцессора и модели ViT."""

from __future__ import annotations

from typing import Sequence

from transformers import ViTForImageClassification, ViTImageProcessor

from .utils import get_device

__all__ = ["MODEL_NAME", "get_processor", "get_model"]

MODEL_NAME = "google/vit-base-patch16-224"


def get_processor(model_name: str = MODEL_NAME) -> ViTImageProcessor:
    """Возвращаем препроцессор для ViT."""

    return ViTImageProcessor.from_pretrained(model_name)


def get_model(labels: Sequence[str], model_name: str = MODEL_NAME):
    """Создаём ViT‑модель с нужным количеством классов и переносим её на устройство."""

    num_labels = len(labels)
    id2label = {str(i): c for i, c in enumerate(labels)}
    label2id = {c: str(i) for i, c in enumerate(labels)}

    device = get_device()
    return (
        ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        ).to(device)
    )