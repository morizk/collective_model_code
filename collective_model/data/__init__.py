"""
Data loading utilities.

Provides data loaders for various datasets with proper train/val/test splits.
"""

from .loaders import get_mnist_loaders, get_data_loaders

__all__ = [
    'get_mnist_loaders',
    'get_data_loaders'
]

