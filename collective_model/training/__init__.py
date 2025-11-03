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

# Modular strategy dispatcher (A/B/C)
def get_strategy_runner(strategy: str):
    """Return a callable run(config, model, train_loader, val_loader, test_loader, device)."""
    strategy = (strategy or 'C').upper()
    if strategy == 'C':
        from .strategies.strategy_c import run as run_c
        return run_c
    if strategy == 'A':
        from .strategies.strategy_a import run as run_a
        return run_a
    if strategy == 'B':
        from .strategies.strategy_b import run as run_b
        return run_b
    raise ValueError(f"Unknown training strategy '{strategy}'. Supported: A, B, C")

__all__ = [
    'diversity_loss_experts',
    'diversity_loss_analysts',
    'combined_loss',
    'CollectiveModel',
    'train_strategy_c',
    'train_epoch',
    'validate',
    'get_strategy_runner',
]

