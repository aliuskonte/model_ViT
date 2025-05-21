"""Вспомогательные функции: сиды, устройство и логирование."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

__all__ = ["seed_everything", "get_device", "setup_logger"]


def seed_everything(seed: int = 42) -> None:
    """Фиксируем генераторы случайных чисел для полной воспроизводимости."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Детерминированные алгоритмы могут замедлить обучение,
    # но гарантируют одинаковый результат на всех запусках.
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    """Возвращаем GPU, если он доступен, иначе CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(name: str = "vit_trainer") -> logging.Logger:
    """Создаём и настраиваем логгер с красивым выводом во время tqdm‑прогрессбаров."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Логгер уже настроен

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger