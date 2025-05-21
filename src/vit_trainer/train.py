"""Функции высокого уровня: обучение и оценка модели."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from evaluate import load as load_metric
from transformers import TrainingArguments

from .clearml_task import load_clearml_config
from .data import _build_transform, collate_fn
from datasets import load_dataset
from .models import get_model, get_processor
from .weighted_trainer import WeightedTrainer
from .utils import get_device, seed_everything, setup_logger

__all__ = ["train", "evaluate"]


logger = setup_logger()

# Загружаем метрики один раз, чтобы не создавать их при каждом вызове
_accuracy = load_metric("accuracy")
_f1 = load_metric("f1")


def _compute_metrics(eval_pred):
    """Вычисляем accuracy и F1‑score (macro)."""

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = _accuracy.compute(predictions=preds, references=labels)
    f1 = _f1.compute(predictions=preds, references=labels, average="macro")
    return {**acc, **f1}


# --------------- API ---------------

def train(
    data_dir: str | Path,
    output_dir: str | Path = "checkpoints",
    epochs: int = 3,
    batch_size: int = 32,
    seed: int = 42,
    use_weights: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Обучаем ViT-модель и возвращаем словарь метрик тренировки."""

    load_clearml_config(epochs, seed, batch_size, data_dir)

    seed_everything(seed)

    processor = get_processor()

    # Вместо только data_dir делаем явный указатель на папки train/validation
    raw_ds = load_dataset(
        "imagefolder",
        data_files={
            "train": str(Path(data_dir) / "train" / "**/*"),
            "val": str(Path(data_dir) / "val" / "**/*"),
            "test": str(Path(data_dir) / "test" / "**/*"),
        },
    )

    labels = raw_ds["train"].features["label"].names

    # Вычисляем веса классов, если это требуется
    class_weights = None
    if use_weights:
        counts = np.bincount(raw_ds["train"]["label"])
        class_weights = torch.tensor(1.0 / counts, dtype=torch.float32)
        logger.info("Веса классов: %s", class_weights.tolist())

    # 3) Оборачиваем датасет трансформацией
    ds = raw_ds.with_transform(_build_transform(processor))

    model = get_model(labels)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to=['clearml'],
        **kwargs,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("val", None),
        data_collator=collate_fn,
        tokenizer=processor,
        compute_metrics=_compute_metrics,
        class_weights=class_weights,
    )

    logger.info("Начинаем обучение на %d эпох(и)...", epochs)
    metrics = trainer.train()
    trainer.save_model()
    return metrics


def evaluate(checkpoint: str | Path, data_dir: str | Path) -> Dict[str, Any]:
    """Оцениваем сохранённую модель на валидационном/тестовом датасете."""

    processor = get_processor()
    raw_ds = load_dataset(
                "imagefolder",
                data_files = {
                    "train": str(Path(data_dir) / "train" / "**"),
                    "val": str(Path(data_dir) / "val" / "**"),
                    "test": str(Path(data_dir) / "test" / "**"),
        },
        )
    ds = raw_ds.with_transform(_build_transform(processor))
    labels = ds["train"].features["label"].names

    device = get_device()
    model = get_model(labels)

    # Загружаем веса, поддерживаем разные форматы state_dict
    state_dict = torch.load(checkpoint, map_location=device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    trainer = WeightedTrainer(
        model = model,
        args = TrainingArguments(output_dir="/tmp/eval", do_train=False, do_eval=True),
        eval_dataset=ds.get("test", ds.get("val")),
        data_collator = collate_fn,
        tokenizer = processor,
        compute_metrics = _compute_metrics,
    )
    return trainer.evaluate()