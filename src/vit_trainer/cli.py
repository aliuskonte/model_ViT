# cli.py

import requests
import torch
from PIL import Image, ImageOps
import random
import matplotlib.pyplot as plt

from transformers import ViTImageProcessor, ViTForImageClassification
from tqdm import tqdm

from datasets import load_dataset

import os
from pathlib import Path
import shutil

from weighted_trainer import WeightedTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model_name = "google/vit-base-patch16-224"
image_processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name).to(DEVICE)


output_path = Path("output")

# load the mine dataset
ds = load_dataset("imagefolder", data_dir=str(output_path))

labels = ds["train"].features["label"]
print(labels)

# Weighted Loss
unique_labels = labels
class_weights = 1.0 / torch.tensor(counts, dtype=torch.float32)

print("unique_labels:", labels)
print("counts:", counts)
print("class_weights:", class_weights)


def transform(examples):
  # convert all images to RGB format, then preprocessing it
  # using our image processor
  inputs = image_processor([img.convert("RGB") for img in examples["image"]], return_tensors="pt")
  # we also shouldn't forget about the labels
  inputs["labels"] = examples["label"]
  return inputs


# Используйте метод with_transform(), чтобы применить преобразование к набору данных на лету во время обучения
dataset = ds.with_transform(transform)

for item in dataset["train"]:
  print(item["pixel_values"].shape)
  print(item["labels"])
  break

# извлеките метки для нашего набора данных
labels = ds["train"].features["label"].names
print(labels)


def collate_fn(batch):
    """
    Объединяет данные в батчи для использования в Trainer.
    """
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


from evaluate import load
import numpy as np

# load the accuracy and f1 metrics from the evaluate module
accuracy = load("accuracy")
f1 = load("f1")

def compute_metrics(eval_pred):
    #Определение метрик
    # compute the accuracy and f1 scores & return them
    accuracy_score = accuracy.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)
    f1_score = f1.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids, average="macro")
    return {**accuracy_score, **f1_score}


# load the ViT model
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True,
)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/CDS/MonsterEye/models", # output directory
    # output_dir="./vit-base-skin-cancer",
    per_device_train_batch_size=32, # batch size per device during training
    evaluation_strategy="steps",    # evaluation strategy to adopt during training
    num_train_epochs=3,             # total number of training epochs
    fp16=True,                    # use mixed precision
    #save_steps=1000,                # number of update steps before saving checkpoint
    #eval_steps=1000,                # number of update steps before evaluating
    #logging_steps=1000,             # number of update steps before logging
    save_steps=50,
    eval_steps=50,
    logging_steps=50,
    save_total_limit=2,             # limit the total amount of checkpoints on disk
    remove_unused_columns=False,    # remove unused columns from the dataset
    push_to_hub=False,              # do not push the model to the hub
    report_to='tensorboard',        # report metrics to tensorboard
    load_best_model_at_end=True    # load the best model at the end of training
)

# обучение с весами
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=image_processor,
    class_weights=class_weights  # Передаем рассчитанные веса
)

# start training
trainer.train()


from sklearn.metrics import classification_report, f1_score
print(classification_report(df['target'], df['pred']))

trainer.evaluate()







