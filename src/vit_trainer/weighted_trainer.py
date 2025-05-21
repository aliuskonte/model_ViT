"""Расширенный Trainer, учитывающий веса классов при вычислении CrossEntropyLoss."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer

__all__ = ["WeightedTrainer"]


class WeightedTrainer(Trainer):
    """Trainer, который при необходимости использует взвешенную функцию потерь."""

    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=None)
        logits = outputs.logits

        # Выбираем обычную или взвешенную кросс‑энтропию
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=weights)
        else:
            loss_fct = CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss