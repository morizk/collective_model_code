"""Strategy C (end-to-end) runner.

This wraps the existing train_strategy_c so we can swap strategies
without modifying the main training script.
"""

from typing import Any, Dict, Optional

from ..trainer import train_strategy_c as core_train_strategy_c


def run(
    config: Dict[str, Any],
    model,  # nn.Module (CollectiveModel)
    train_loader,
    val_loader,
    test_loader=None,
    device: Optional[str] = None,
):
    return core_train_strategy_c(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )


