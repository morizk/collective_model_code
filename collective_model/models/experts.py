"""
Expert models - Large feature extractors.

Experts are LARGE models (larger than analysts) that extract rich
feature representations from raw input. Multiple experts see the same
input but can learn different, complementary features.
"""

import torch
import torch.nn as nn
from .base import BaseExpert


class MLPExpert(BaseExpert):
    """
    Multi-Layer Perceptron (MLP) Expert.
    
    A large feedforward network that extracts rich features from input.
    Experts are trained to be diverse and complementary.
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list[int]): List of hidden layer sizes (e.g., [512, 256])
                                 Must have at least 2 hidden layers
        output_dim (int): Output feature dimension (consistent across all experts)
        use_batchnorm (bool): Whether to use batch normalization
        dropout (float): Dropout rate (0 = no dropout)
        expert_id (int): Unique identifier for this expert (used for logging)
    
    Example:
        >>> expert = MLPExpert(input_dim=784, hidden_dims=[512, 256], output_dim=128)
        >>> x = torch.randn(32, 784)  # batch of 32 MNIST images
        >>> features = expert(x)      # (32, 128)
    """
    def __init__(
        self, 
        input_dim, 
        hidden_dims, 
        output_dim,
        use_batchnorm=True,
        dropout=0.2,
        expert_id=0
    ):
        super().__init__(input_dim, output_dim)
        
        # Validate architecture
        if len(hidden_dims) < 2:
            raise ValueError(f"Experts must have at least 2 hidden layers, got {len(hidden_dims)}")
        
        self.hidden_dims = hidden_dims
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.expert_id = expert_id
        
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
        return (f"MLPExpert(id={self.expert_id}, "
                f"input={self.input_dim}, "
                f"hidden={self.hidden_dims}, "
                f"output={self.output_dim}, "
                f"params={self.get_num_parameters():,})")


def create_diverse_experts(
    n_experts,
    input_dim,
    output_dim,
    base_hidden_dims,
    vary_architectures=False,
    use_batchnorm=True,
    dropout=0.2
):
    """
    Create a list of expert models.
    
    Args:
        n_experts (int): Number of experts to create
        input_dim (int): Input dimension
        output_dim (int): Output feature dimension (same for all)
        base_hidden_dims (list[int]): Base architecture (e.g., [512, 256])
        vary_architectures (bool): If True, create experts with different architectures
                                   to promote diversity
        use_batchnorm (bool): Whether to use batch normalization
        dropout (float): Dropout rate
    
    Returns:
        list[MLPExpert]: List of expert models
    
    Architecture Variation Strategy (if vary_architectures=True):
        - Expert 0: Base architecture [512, 256]
        - Expert 1: Wider [640, 320] (+25%)
        - Expert 2: Deeper [512, 384, 256]
        - Expert 3: Narrower [400, 200] (-20%)
        - Then cycle for more experts
    """
    experts = []
    
    for i in range(n_experts):
        if vary_architectures and n_experts > 1:
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
            # All experts use same architecture
            hidden_dims = base_hidden_dims.copy()
        
        # Ensure minimum 2 layers
        if len(hidden_dims) < 2:
            hidden_dims = hidden_dims * 2
        
        expert = MLPExpert(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            expert_id=i
        )
        experts.append(expert)
    
    return experts

