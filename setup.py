"""Setup script for collective_model package."""

from setuptools import setup, find_packages

setup(
    name="collective_model",
    version="0.1.0",
    description="Collective Model Architecture for Fashion-MNIST",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "wandb>=0.15.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
    ],
)

