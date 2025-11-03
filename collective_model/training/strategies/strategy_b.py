"""Strategy B (layer-wise pretraining with encoder/decoder, then fine-tune).

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
        "Strategy B is not implemented yet. This placeholder exists so we can "
        "switch strategies without rewriting the training loop."
    )


