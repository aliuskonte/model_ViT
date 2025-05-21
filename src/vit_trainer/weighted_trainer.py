# weighted_trainer.py
from transformers import Trainer
from torch.nn import CrossEntropyLoss

# Определить кастомный Trainer с переопределённым compute_loss
# В Trainer есть метод compute_loss(), который по умолчанию вызывает обычную CrossEntropyLoss без весов.
# Мы переопределим этот метод и добавим weight=class_weights


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # Сохраним веса классов

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        # Если есть веса классов, используем взвешенную CrossEntropyLoss
        if self.class_weights is not None:
            # Переносим веса классов на то же устройство, где и logits
            weights = self.class_weights.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=weights)
        else:
            loss_fct = CrossEntropyLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss