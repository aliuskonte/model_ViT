# файл: setup.py

from setuptools import setup, find_packages

setup(
    # Название пакета (может совпадать с ключом в Pipfile)
    name="vit-trainer",
    version="0.1.0",
    description="Пакет для обучения ViT на несбалансированных датасетах изображений",
    author="Your Name",
    author_email="you@example.com",
    # Указываем, что исходники лежат в папке src/
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # В соответствии с вашим Pipfile
    install_requires=[
        "requests",       # HTTP-библиотека
        "torch",          # PyTorch
        "transformers",   # Hugging Face Transformers
        "matplotlib",     # Визуализация
        "datasets",       # Hugging Face Datasets
        "evaluate",       # Модуль метрик 🤗
        "clearml"
    ],
    python_requires=">=3.10",  # как в Pipfile
    entry_points={
        "console_scripts": [
            # Создаём две команды CLI (дефис/подчёркивание)
            "vit-trainer=vit_trainer.cli:app",
            "vit_trainer=vit_trainer.cli:app",
        ]
    },
)