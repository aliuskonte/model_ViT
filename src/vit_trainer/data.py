"""Загрузка ImageFolder‑датасета и подготовка трансформаций."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import torch
from typing import Any, Dict, List

from datasets import DatasetDict, load_dataset
from transformers import ViTImageProcessor

__all__ = ["get_datasets"]


def _build_transform(processor: ViTImageProcessor) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Создаём функцию‑трансформер, которая применяется пакетом (batch)."""

    def _transform(batch: dict[str, Any]) -> dict[str, Any]:
        # Преобразуем PIL‑изображения в тензоры и добавляем метки
        inputs = processor([img.convert("RGB") for img in batch["image"]], return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    return _transform


def get_datasets(data_dir: str | Path, processor: ViTImageProcessor) -> DatasetDict:
    """Загружаем датасет из каталога и возвращаем сплиты с онлайновым трансформом."""

    ds = load_dataset("imagefolder", data_dir=str(data_dir))
    return ds.with_transform(_build_transform(processor))


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Собирает батч из пикселей и меток:
      - pixel_values: [batch_size, 3, H, W]
      - labels:       [batch_size]
    """
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.tensor([example["labels"] for example in batch])
    return {"pixel_values": pixel_values, "labels": labels}