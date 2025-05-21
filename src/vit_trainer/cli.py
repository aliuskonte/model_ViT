# файл: src/vit_trainer/cli.py

"""
Точка входа для командной строки на основе Click.
Здесь описываются команды fit и eval.
"""

from __future__ import annotations

import click
from pathlib import Path

from .train import train, evaluate
from .utils import setup_logger

# настраиваем логгер
logger = setup_logger()


@click.group()
@click.version_option(package_name="vit-trainer")
def app():
    """🔧 vit-trainer — обучение и оценка ViT-моделей."""
    pass


@app.command(help="Запустить обучение на каталоге ImageFolder.")
@click.option("--data-dir",
              type=click.Path(exists=True, file_okay=False),
              required=True,
              help="Путь к папке с изображениями в формате ImageFolder.")
@click.option("--output-dir",
              type=str,
              default="checkpoints",
              help="Куда сохранять чекпойнты.")
@click.option("--epochs",
              type=int,
              default=3,
              help="Число эпох обучения.")
@click.option("--batch-size",
              type=int,
              default=32,
              help="Размер батча.")
@click.option("--no-weights/--with-weights",
              default=False,
              help="Отключить/включить взвешенную функцию потерь.")
def fit(data_dir, output_dir, epochs, batch_size, no_weights):
    """
    Команда fit:
      - загружает датасет из data_dir;
      - обучает модель;
      - сохраняет лучшие веса в output_dir.
    """
    metrics = train(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        use_weights=not no_weights,
    )
    logger.info("Завершено обучение — метрики: %s", metrics)


@app.command(help="Оценить сохранённый чекпойнт на валидации/тесте.")
@click.option("--checkpoint",
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help="Путь к файлу с весами модели (.bin или .pt).")
@click.option("--data-dir",
              type=click.Path(exists=True, file_okay=False),
              required=True,
              help="Папка с валидационными/тестовыми данными.")
def eval(checkpoint, data_dir):
    """
    Команда eval:
      - загружает модель из checkpoint;
      - оценивает её на data_dir;
      - выводит метрики.
    """
    metrics = evaluate(checkpoint=Path(checkpoint), data_dir=Path(data_dir))
    logger.info("Результаты оценки: %s", metrics)


if __name__ == "__main__":
    app()