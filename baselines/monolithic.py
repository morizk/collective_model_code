"""
Baseline monolithic models for comparison with Collective Model.

Includes:
- MonolithicMLP: Standard MLP without skip connections
- ResNetMLP: MLP with residual (skip) connections
"""

import torch
import torch.nn as nn


class MonolithicMLP(nn.Module):
    """
    Standard MLP without skip connections.
    
    Simple sequential architecture for baseline comparison.
    All layers are the same width (or varying based on hidden_dims).
    """
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1, use_batchnorm=True):
        """
        Args:
            input_dim: Input feature dimension (e.g., 784 for MNIST)
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128])
            num_classes: Number of output classes
            dropout: Dropout probability
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build layers
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batchnorm else None
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_batchnorm and i < len(dims) - 2:  # No BN on output layer
                self.norms.append(nn.BatchNorm1d(dims[i+1]))
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor [batch_size, ...] (will be flattened)
        
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers[:-1]):  # All except output
            x = layer(x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        
        # Output layer (no activation, no dropout)
        x = self.layers[-1](x)
        return x


class ResNetMLP(nn.Module):
    """
    MLP with residual (skip) connections.
    
    Adds skip connections between layers, similar to ResNet.
    Useful for training deeper networks and fair comparison with Collective Model
    (which also has skip connections: input → analysts).
    """
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1, use_batchnorm=True):
        """
        Args:
            input_dim: Input feature dimension (e.g., 784 for MNIST)
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256, 128])
            num_classes: Number of output classes
            dropout: Dropout probability
            use_batchnorm: Whether to use batch normalization
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build layers with skip connections
        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batchnorm else None
        self.projections = nn.ModuleList()  # For dimension matching in skip connections
        
        for i in range(len(hidden_dims)):
            # Main layer
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if use_batchnorm:
                self.norms.append(nn.BatchNorm1d(dims[i+1]))
            
            # Projection layer if dimensions don't match (for skip connection)
            if dims[i] != dims[i+1]:
                self.projections.append(nn.Linear(dims[i], dims[i+1]))
            else:
                self.projections.append(None)
        
        # Output layer (no skip connection)
        self.output = nn.Linear(hidden_dims[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with skip connections.
        
        Args:
            x: Input tensor [batch_size, ...] (will be flattened)
        
        Returns:
            Output logits [batch_size, num_classes]
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # First layer with skip connection from input
        identity = x
        x = self.layers[0](x)
        if self.norms is not None:
            x = self.norms[0](x)
        
        # Add skip connection (with projection if needed)
        if self.projections[0] is not None:
            identity = self.projections[0](identity)
        x = x + identity  # Skip connection
        x = self.relu(x)
        x = self.dropout(x)
        
        # Remaining layers with skip connections
        for i in range(1, len(self.layers)):
            identity = x
            x = self.layers[i](x)
            if self.norms is not None:
                x = self.norms[i](x)
            
            # Add skip connection (with projection if needed)
            if self.projections[i] is not None:
                identity = self.projections[i](identity)
            x = x + identity  # Skip connection
            x = self.relu(x)
            x = self.dropout(x)
        
        # Output layer (no skip connection, no activation)
        x = self.output(x)
        return x


if __name__ == '__main__':
    # Test both models
    print("Testing MonolithicMLP...")
    model1 = MonolithicMLP(input_dim=784, hidden_dims=[512, 256, 128], num_classes=10)
    x = torch.randn(2, 784)
    out1 = model1(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out1.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    print("\nTesting ResNetMLP...")
    model2 = ResNetMLP(input_dim=784, hidden_dims=[512, 256, 128], num_classes=10)
    x = torch.randn(2, 784)
    out2 = model2(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out2.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    print("\n✓ Both models working correctly!")

