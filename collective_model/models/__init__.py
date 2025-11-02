"""
Collective Model - Model Components

This module contains all the building blocks for the Collective Model architecture:
- Base classes (BaseExpert, BaseAnalyst)
- Expert models (MLPExpert)
- Analyst models (MLPAnalyst)
- Encoder networks (Encoder, CollectiveEncoder)
- Collective aggregation layers (SimpleCollective, EncoderHeadCollective)
"""

# Base classes
from .base import BaseExpert, BaseAnalyst

# Expert models
from .experts import MLPExpert, create_diverse_experts

# Analyst models
from .analysts import MLPAnalyst, create_diverse_analysts

# Encoder networks
from .encoder import Encoder, CollectiveEncoder

# Collective aggregation
from .collective import (
    SimpleCollective,
    EncoderHeadCollective,
    create_collective
)

__all__ = [
    # Base classes
    'BaseExpert',
    'BaseAnalyst',
    
    # Experts
    'MLPExpert',
    'create_diverse_experts',
    
    # Analysts
    'MLPAnalyst',
    'create_diverse_analysts',
    
    # Encoders
    'Encoder',
    'CollectiveEncoder',
    
    # Collective
    'SimpleCollective',
    'EncoderHeadCollective',
    'create_collective',
]



