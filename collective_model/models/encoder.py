"""
Encoder networks - Compression layers.

Encoders compress high-dimensional feature vectors to lower dimensions
while preserving important information.

Two types:
1. ExpertEncoder: Compresses concatenated expert outputs
2. CollectiveEncoder: Compresses concatenated analyst outputs (for collective v2)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder for compressing expert outputs.
    
    Takes concatenated expert feature vectors and compresses them
    to a smaller representation for analysts to consume.
    
    Architecture: Single hidden layer bottleneck
        input_dim → hidden_dim → output_dim
    
    Args:
        input_dim (int): Concatenated expert output dimension (n_experts × expert_output_dim)
        output_dim (int): Compressed dimension (input_dim × c_expert)
        use_batchnorm (bool): Whether to use batch normalization
        dropout (float): Dropout rate (0 = no dropout)
    
    Example:
        >>> # 3 experts, each outputting 128 features
        >>> # Compression ratio c_expert = 0.25
        >>> encoder = Encoder(input_dim=384, output_dim=96)  # 384 * 0.25 = 96
        >>> expert_features = torch.randn(32, 384)
        >>> compressed = encoder(expert_features)  # (32, 96)
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        use_batchnorm=True,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.compression_ratio = output_dim / input_dim
        
        # Hidden dimension: geometric mean of input and output
        hidden_dim = int((input_dim * output_dim) ** 0.5)
        
        layers = []
        
        # First layer: input → hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Second layer: hidden → output
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Compress expert features.
        
        Args:
            x: Concatenated expert outputs [batch_size, input_dim]
        
        Returns:
            compressed: Compressed features [batch_size, output_dim]
        """
        return self.network(x)
    
    def get_num_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"Encoder(input={self.input_dim}, "
                f"output={self.output_dim}, "
                f"ratio={self.compression_ratio:.2f}, "
                f"params={self.get_num_parameters():,})")


class CollectiveEncoder(nn.Module):
    """
    Encoder for compressing analyst outputs (Collective Version 2).
    
    Used in collective_version='encoder_head' mode.
    Takes concatenated analyst feature vectors and compresses them
    before the final prediction head.
    
    Architecture: Single hidden layer bottleneck
        input_dim → hidden_dim → output_dim
    
    Args:
        input_dim (int): Concatenated analyst output dimension (n_analysts × analyst_output_dim)
        output_dim (int): Compressed dimension (input_dim × c_collective)
        use_batchnorm (bool): Whether to use batch normalization
        dropout (float): Dropout rate (0 = no dropout)
    
    Example:
        >>> # 5 analysts, each outputting 64 features
        >>> # Compression ratio c_collective = 0.25
        >>> encoder = CollectiveEncoder(input_dim=320, output_dim=80)  # 320 * 0.25 = 80
        >>> analyst_features = torch.randn(32, 320)
        >>> compressed = encoder(analyst_features)  # (32, 80)
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        use_batchnorm=True,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.compression_ratio = output_dim / input_dim
        
        # Hidden dimension: geometric mean of input and output
        hidden_dim = int((input_dim * output_dim) ** 0.5)
        
        layers = []
        
        # First layer: input → hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Second layer: hidden → output
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Compress analyst features.
        
        Args:
            x: Concatenated analyst outputs [batch_size, input_dim]
        
        Returns:
            compressed: Compressed features [batch_size, output_dim]
        """
        return self.network(x)
    
    def get_num_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"CollectiveEncoder(input={self.input_dim}, "
                f"output={self.output_dim}, "
                f"ratio={self.compression_ratio:.2f}, "
                f"params={self.get_num_parameters():,})")

