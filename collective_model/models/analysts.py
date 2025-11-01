"""
Analyst models - Smaller synthesizers of expert opinions.

Analysts are SMALLER models (smaller than experts) that synthesize
expert opinions with the original input. They read expert features
and the problem, then give their own analysis.
"""

import torch
import torch.nn as nn
from .base import BaseAnalyst


class MLPAnalyst(BaseAnalyst):
    """
    Multi-Layer Perceptron (MLP) Analyst.
    
    A smaller feedforward network that synthesizes expert opinions with input.
    Analysts receive concatenated input: [original_input + encoded_expert_features]
    
    Think of analysts as people who:
    1. Know the original problem (original_input)
    2. Read expert opinions (encoded_expert_features)
    3. Provide their own analysis based on both
    
    Args:
        input_dim (int): Combined input dimension 
                         = original_input_dim + encoded_expert_features_dim
        hidden_dims (list[int]): List of hidden layer sizes (e.g., [256, 128])
                                 Must have at least 2 hidden layers
                                 SMALLER than expert hidden layers
        output_dim (int): Output feature dimension (consistent across all analysts)
        use_batchnorm (bool): Whether to use batch normalization
        dropout (float): Dropout rate (0 = no dropout)
        analyst_id (int): Unique identifier for this analyst (used for logging)
    
    Example:
        >>> # Original input: 784, Encoded expert features: 64
        >>> analyst = MLPAnalyst(input_dim=848, hidden_dims=[256, 128], output_dim=64)
        >>> x = torch.randn(32, 848)  # batch with concatenated input
        >>> features = analyst(x)      # (32, 64)
    """
    def __init__(
        self, 
        input_dim, 
        hidden_dims, 
        output_dim,
        use_batchnorm=True,
        dropout=0.2,
        analyst_id=0
    ):
        super().__init__(input_dim, output_dim)
        
        # Validate architecture
        if len(hidden_dims) < 2:
            raise ValueError(f"Analysts must have at least 2 hidden layers, got {len(hidden_dims)}")
        
        self.hidden_dims = hidden_dims
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.analyst_id = analyst_id
        
        # Build backbone: input → hidden layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout (optional, not on last hidden layer)
            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Projection layer: final hidden → output_dim
        self.projection = nn.Linear(hidden_dims[-1], output_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_num_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        """String representation for debugging."""
        return (f"MLPAnalyst(id={self.analyst_id}, "
                f"input={self.input_dim}, "
                f"hidden={self.hidden_dims}, "
                f"output={self.output_dim}, "
                f"params={self.get_num_parameters():,})")


def create_diverse_analysts(
    n_analysts,
    input_dim,
    output_dim,
    base_hidden_dims,
    vary_architectures=False,
    use_batchnorm=True,
    dropout=0.2
):
    """
    Create a list of analyst models.
    
    Args:
        n_analysts (int): Number of analysts to create
        input_dim (int): Combined input dimension (original + encoded expert features)
        output_dim (int): Output feature dimension (same for all)
        base_hidden_dims (list[int]): Base architecture (e.g., [256, 128])
                                      SMALLER than expert hidden dims
        vary_architectures (bool): If True, create analysts with different architectures
                                   to promote diversity
        use_batchnorm (bool): Whether to use batch normalization
        dropout (float): Dropout rate
    
    Returns:
        list[MLPAnalyst]: List of analyst models
    
    Architecture Variation Strategy (if vary_architectures=True):
        - Analyst 0: Base architecture [256, 128]
        - Analyst 1: Wider [320, 160] (+25%)
        - Analyst 2: Deeper [256, 192, 128]
        - Analyst 3: Narrower [200, 100] (-20%)
        - Then cycle for more analysts
    """
    analysts = []
    
    for i in range(n_analysts):
        if vary_architectures and n_analysts > 1:
            # Create architectural diversity
            variant = i % 4  # Cycle through 4 variants
            
            if variant == 0:
                # Base architecture
                hidden_dims = base_hidden_dims.copy()
            elif variant == 1:
                # Wider (+25%)
                hidden_dims = [int(h * 1.25) for h in base_hidden_dims]
            elif variant == 2:
                # Deeper (add middle layer)
                if len(base_hidden_dims) >= 2:
                    mid = (base_hidden_dims[0] + base_hidden_dims[-1]) // 2
                    hidden_dims = [base_hidden_dims[0], mid, base_hidden_dims[-1]]
                else:
                    hidden_dims = base_hidden_dims.copy()
            else:  # variant == 3
                # Narrower (-20%)
                hidden_dims = [int(h * 0.8) for h in base_hidden_dims]
        else:
            # All analysts use same architecture
            hidden_dims = base_hidden_dims.copy()
        
        # Ensure minimum 2 layers
        if len(hidden_dims) < 2:
            hidden_dims = hidden_dims * 2
        
        analyst = MLPAnalyst(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            analyst_id=i
        )
        analysts.append(analyst)
    
    return analysts

