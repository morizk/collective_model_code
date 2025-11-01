"""
Base classes for Expert and Analyst models.

These base classes provide a common interface with projection layers
to ensure consistent output dimensions across all models.
"""

import torch
import torch.nn as nn


class BaseExpert(nn.Module):
    """
    Base class for all expert models.
    
    Experts are LARGE models that extract rich feature representations.
    All experts must output the same dimension for concatenation.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output feature dimension (same for all experts)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Subclasses must implement these
        self.backbone = None  # Main feature extraction network
        self.projection = None  # Projects backbone output to output_dim
    
    def forward(self, x):
        """
        Forward pass through expert model.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            features: Feature tensor [batch_size, output_dim]
        """
        if self.backbone is None or self.projection is None:
            raise NotImplementedError("Subclass must implement backbone and projection")
        
        features = self.backbone(x)
        return self.projection(features)


class BaseAnalyst(nn.Module):
    """
    Base class for all analyst models.
    
    Analysts are SMALLER models that synthesize expert opinions with input.
    All analysts must output the same dimension for concatenation.
    
    Args:
        input_dim (int): Combined input dimension [original_input + encoded_expert_features]
        output_dim (int): Output feature dimension (same for all analysts)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Subclasses must implement these
        self.backbone = None  # Main feature extraction network
        self.projection = None  # Projects backbone output to output_dim
    
    def forward(self, x):
        """
        Forward pass through analyst model.
        
        Args:
            x: Input tensor [batch_size, input_dim]
               input_dim = original_input_dim + encoded_expert_features_dim
            
        Returns:
            features: Feature tensor [batch_size, output_dim]
        """
        if self.backbone is None or self.projection is None:
            raise NotImplementedError("Subclass must implement backbone and projection")
        
        features = self.backbone(x)
        return self.projection(features)

