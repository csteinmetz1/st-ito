from pathlib import Path
from setuptools import setup, find_packages

NAME = "st-ito"
DESCRIPTION = "Style transfer with inference-time optimization"
URL = "https://github.com/csteinmetz1/st-ito"
EMAIL = "c.j.steinmetz@qmul.ac.uk"
AUTHOR = "Christian J. Steinmetz"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "0.0.1"

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchaudio",
        "torchvision",
        "torchlibrosa",
        "pytorch-lightning",
        "pyloudnorm",
        "pedalboard",
        "wandb",
        "soundfile",
        "matplotlib",
        "scikit-learn",
        "timm",
        "transformers",
        "wav2clip",
        "resampy",
        "laion_clap",
        "cma",
        "auraloss",
        "umap-learn",
        "dasp-pytorch",
    ],
    include_package_data=True,
    license="Apache License 2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
    ],
)
