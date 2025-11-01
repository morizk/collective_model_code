"""
Training utilities for the Collective Model.

Includes loss functions, training strategies, and the main model class.
"""

from .losses import (
    diversity_loss_experts,
    diversity_loss_analysts,
    combined_loss
)

from .collective_model import CollectiveModel
from .trainer import train_strategy_c, train_epoch, validate

__all__ = [
    'diversity_loss_experts',
    'diversity_loss_analysts',
    'combined_loss',
    'CollectiveModel',
    'train_strategy_c',
    'train_epoch',
    'validate',
]

