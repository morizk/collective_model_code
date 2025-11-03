"""Strategy A (pretrain experts → freeze → train rest → fine-tune all).

Placeholder runner. To be implemented.
"""

from typing import Any, Dict, Optional


def run(
    config: Dict[str, Any],
    model,
    train_loader,
    val_loader,
    test_loader=None,
    device: Optional[str] = None,
):
    raise NotImplementedError(
        "Strategy A is not implemented yet. This placeholder exists so we can "
        "switch strategies without rewriting the training loop."
    )


