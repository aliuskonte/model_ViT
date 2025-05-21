"""vit_trainer — пакет для обучения и оценки ViT‑моделей с учётом дисбаланса классов."""

from importlib.metadata import version

__all__ = [
    "__version__",
    "seed_everything",
    "get_datasets",
    "get_model",
    "WeightedTrainer",
    "train",
    "evaluate",
]

__version__: str = version("vit_trainer")

# Импорт выполняется внизу, чтобы избежать циклических зависимостей
from .utils import seed_everything  # noqa: E402
from .data import get_datasets  # noqa: E402
from .models import get_model  # noqa: E402
from weighted_trainer import WeightedTrainer  # noqa: E402
from .train import train, evaluate  # noqa: E402