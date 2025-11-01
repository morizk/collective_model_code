"""
Collective aggregation layers - Final decision making.

The collective layer takes analyst opinions and produces the final prediction.

Two versions:
1. SimpleCollective: Direct MLP from analyst features → prediction
2. EncoderHeadCollective: Encoder compresses analyst features, then prediction head
"""

import torch
import torch.nn as nn
from .encoder import CollectiveEncoder


class SimpleCollective(nn.Module):
    """
    Simple Collective (Version 1) - Direct MLP aggregation.
    
    Takes concatenated analyst outputs and directly predicts the final output.
    This is the simpler, more interpretable version.
    
    Architecture:
        analyst_features → hidden_layer → output
    
    Args:
        input_dim (int): Concatenated analyst output dimension (n_analysts × analyst_output_dim)
        num_classes (int): Number of output classes (e.g., 10 for MNIST)
        hidden_scale (float): Scale factor for hidden layer size (default: 1.0)
                             hidden_dim = input_dim * hidden_scale
        dropout (float): Dropout rate (0 = no dropout)
    
    Example:
        >>> # 5 analysts, each outputs 64 features → 320 total
        >>> collective = SimpleCollective(input_dim=320, num_classes=10)
        >>> analyst_features = torch.randn(32, 320)
        >>> logits = collective(analyst_features)  # (32, 10)
    """
    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_scale=1.0,
        dropout=0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_scale = hidden_scale
        
        # Hidden dimension
        hidden_dim = int(input_dim * hidden_scale)
        hidden_dim = max(hidden_dim, num_classes * 2)  # At least 2x output size
        
        # Simple MLP: input → hidden → output
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, analyst_features):
        """
        Aggregate analyst opinions into final prediction.
        
        Args:
            analyst_features: Concatenated analyst outputs [batch_size, input_dim]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        return self.network(analyst_features)
    
    def get_num_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"SimpleCollective(input={self.input_dim}, "
                f"classes={self.num_classes}, "
                f"params={self.get_num_parameters():,})")


class EncoderHeadCollective(nn.Module):
    """
    Encoder-Head Collective (Version 2) - Two-stage aggregation.
    
    First compresses analyst outputs with an encoder, then uses a prediction head.
    This allows for more compression and potentially better generalization.
    
    Architecture:
        analyst_features → CollectiveEncoder → compressed_features → prediction_head → output
    
    Args:
        input_dim (int): Concatenated analyst output dimension (n_analysts × analyst_output_dim)
        num_classes (int): Number of output classes (e.g., 10 for MNIST)
        c_collective (float): Compression ratio for encoder (e.g., 0.25)
        hidden_scale (float): Scale factor for prediction head hidden layer
        dropout (float): Dropout rate
    
    Example:
        >>> # 5 analysts, each outputs 64 features → 320 total
        >>> # Compress to 80 (ratio=0.25), then predict
        >>> collective = EncoderHeadCollective(
        ...     input_dim=320, 
        ...     num_classes=10, 
        ...     c_collective=0.25
        ... )
        >>> analyst_features = torch.randn(32, 320)
        >>> logits = collective(analyst_features)  # (32, 10)
    """
    def __init__(
        self,
        input_dim,
        num_classes,
        c_collective=0.25,
        hidden_scale=1.0,
        dropout=0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.c_collective = c_collective
        
        # Encoder: compress analyst features
        encoded_dim = int(input_dim * c_collective)
        encoded_dim = max(encoded_dim, num_classes * 2)  # At least 2x output size
        
        self.encoder = CollectiveEncoder(
            input_dim=input_dim,
            output_dim=encoded_dim,
            use_batchnorm=True,
            dropout=dropout
        )
        
        # Prediction head: compressed features → classes
        hidden_dim = int(encoded_dim * hidden_scale)
        hidden_dim = max(hidden_dim, num_classes * 2)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, analyst_features):
        """
        Aggregate analyst opinions into final prediction.
        
        Args:
            analyst_features: Concatenated analyst outputs [batch_size, input_dim]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Compress analyst features
        compressed = self.encoder(analyst_features)
        
        # Predict from compressed features
        logits = self.prediction_head(compressed)
        
        return logits
    
    def get_num_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"EncoderHeadCollective(input={self.input_dim}, "
                f"compressed={self.encoder.output_dim}, "
                f"classes={self.num_classes}, "
                f"ratio={self.c_collective:.2f}, "
                f"params={self.get_num_parameters():,})")


def create_collective(
    collective_version,
    input_dim,
    num_classes,
    c_collective=0.25,
    hidden_scale=1.0,
    dropout=0.2
):
    """
    Factory function to create collective layer based on version.
    
    Args:
        collective_version (str): Either 'simple_mlp' or 'encoder_head'
        input_dim (int): Input dimension (concatenated analyst features)
        num_classes (int): Number of output classes
        c_collective (float): Compression ratio (only used for encoder_head)
        hidden_scale (float): Hidden layer scale factor
        dropout (float): Dropout rate
    
    Returns:
        nn.Module: Either SimpleCollective or EncoderHeadCollective
    
    Example:
        >>> collective = create_collective(
        ...     collective_version='simple_mlp',
        ...     input_dim=320,
        ...     num_classes=10
        ... )
    """
    if collective_version == 'simple_mlp':
        return SimpleCollective(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_scale=hidden_scale,
            dropout=dropout
        )
    elif collective_version == 'encoder_head':
        return EncoderHeadCollective(
            input_dim=input_dim,
            num_classes=num_classes,
            c_collective=c_collective,
            hidden_scale=hidden_scale,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown collective_version: {collective_version}. "
                        f"Must be 'simple_mlp' or 'encoder_head'")

