"""Training strategies (A/B/C) dispatch module."""

from .strategy_c import run as run_c  # noqa: F401

__all__ = [
    'run_c',
]


