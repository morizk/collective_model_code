"""
Collective Model - A hierarchical ensemble architecture.

This package implements the Collective Model, a novel deep learning architecture
that combines multiple expert models, encodes their outputs, synthesizes them
through analyst models, and produces a collective decision.

Architecture:
    Input → Experts → Encoder → [Input + Encoded] → Analysts → Collective → Output

Key Components:
- Expert Layer: N large models extract rich features
- Encoder: Compresses expert outputs
- Analyst Layer: M smaller models synthesize expert opinions with input
- Collective Layer: Aggregates analyst opinions into final prediction
"""

__version__ = '0.1.0'
__author__ = 'Collective Model Research'

# Import key modules
from . import models

__all__ = ['models']

