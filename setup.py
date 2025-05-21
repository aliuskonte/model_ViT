# setup.py
from setuptools import setup, find_packages

setup(
    name="vit_trainer",
    version="0.1.0",
    # говорим setuptools, что исходники лежат в папке src
    package_dir={"": "src"},
    # автоматически найдёт пакет vit_trainer внутри src/
    packages=find_packages("src")
)