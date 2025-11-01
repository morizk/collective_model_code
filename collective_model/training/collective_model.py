"""
CollectiveModel - Main model class that combines all components.

This is the end-to-end (Strategy C) implementation where all components
are trained together.
"""

import torch
import torch.nn as nn

# Use try/except to handle both package import and direct execution
try:
    from ..models import (
        create_diverse_experts,
        create_diverse_analysts,
        Encoder,
        create_collective
    )
except ImportError:
    from collective_model.models import (
        create_diverse_experts,
        create_diverse_analysts,
        Encoder,
        create_collective
    )


class CollectiveModel(nn.Module):
    """
    Main Collective Model that combines experts, encoder, analysts, and collective.
    
    Architecture:
        Input → Experts → Encoder → [Input + Encoded] → Analysts → Collective → Output
    
    Args:
        config (dict): Configuration dictionary (must be prepared with prepare_config())
    
    Example:
        >>> from collective_model.config import CONFIG_DEBUG, prepare_config
        >>> config = prepare_config(CONFIG_DEBUG)
        >>> model = CollectiveModel(config)
        >>> x = torch.randn(32, 784)
        >>> output = model(x)
        >>> print(output.shape)  # (32, 10)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Validate config
        required_keys = [
            'input_dim', 'num_classes', 'n_experts', 'n_analysts',
            'expert_hidden', 'expert_output', 'analyst_hidden', 'analyst_output',
            'expert_encoder_output_dim', 'analyst_input_dim', 'collective_input_dim',
            'c_expert', 'collective_version'
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config missing required key: {key}. Did you call prepare_config()?")
        
        # Create experts
        self.experts = nn.ModuleList(
            create_diverse_experts(
                n_experts=config['n_experts'],
                input_dim=config['input_dim'],
                output_dim=config['expert_output'],
                base_hidden_dims=config['expert_hidden'],
                vary_architectures=config.get('vary_expert_architectures', False),
                use_batchnorm=True,
                dropout=0.2
            )
        )
        
        # Create expert encoder
        expert_concat_dim = config['n_experts'] * config['expert_output']
        self.expert_encoder = Encoder(
            input_dim=expert_concat_dim,
            output_dim=config['expert_encoder_output_dim'],
            use_batchnorm=True,
            dropout=0.1
        )
        
        # Create analysts
        self.analysts = nn.ModuleList(
            create_diverse_analysts(
                n_analysts=config['n_analysts'],
                input_dim=config['analyst_input_dim'],
                output_dim=config['analyst_output'],
                base_hidden_dims=config['analyst_hidden'],
                vary_architectures=config.get('vary_analyst_architectures', False),
                use_batchnorm=True,
                dropout=0.2
            )
        )
        
        # Create collective layer
        self.collective = create_collective(
            collective_version=config['collective_version'],
            input_dim=config['collective_input_dim'],
            num_classes=config['num_classes'],
            c_collective=config.get('c_collective', 0.25),
            hidden_scale=config.get('collective_hidden_scale', 1.0),
            dropout=0.2
        )
    
    def forward(self, x, return_intermediates=False):
        """
        Forward pass through the entire Collective Model.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            return_intermediates (bool): If True, return expert and analyst outputs
                                         for diversity loss computation
        
        Returns:
            torch.Tensor: Output logits [batch_size, num_classes]
            OR
            tuple: (logits, expert_outputs, analyst_outputs) if return_intermediates=True
        """
        batch_size = x.size(0)
        
        # Ensure input is flattened
        if x.dim() > 2:
            x = x.view(batch_size, -1)
        
        # 1. Expert layer: Extract rich features
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_outputs.append(expert_out)
        
        # Concatenate expert outputs
        expert_concat = torch.cat(expert_outputs, dim=1)
        
        # 2. Encoder: Compress expert features
        encoded_experts = self.expert_encoder(expert_concat)
        
        # 3. Analyst layer: Synthesize expert opinions with input
        analyst_input = torch.cat([x, encoded_experts], dim=1)
        
        analyst_outputs = []
        for analyst in self.analysts:
            analyst_out = analyst(analyst_input)
            analyst_outputs.append(analyst_out)
        
        # Concatenate analyst outputs
        analyst_concat = torch.cat(analyst_outputs, dim=1)
        
        # 4. Collective layer: Final aggregation
        logits = self.collective(analyst_concat)
        
        if return_intermediates:
            return logits, expert_outputs, analyst_outputs
        
        return logits
    
    def get_num_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        """String representation for debugging."""
        params = self.get_num_parameters()
        return (f"CollectiveModel(\n"
                f"  experts={self.config['n_experts']}, "
                f"analysts={self.config['n_analysts']},\n"
                f"  collective_version='{self.config['collective_version']}',\n"
                f"  total_params={params:,}\n"
                f")")


if __name__ == '__main__':
    # Test the CollectiveModel
    print("Testing CollectiveModel...")
    
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from collective_model.config import CONFIG_DEBUG, prepare_config
    
    # Prepare config
    config = prepare_config(CONFIG_DEBUG)
    print("\nConfig prepared:")
    print(f"  n_experts={config['n_experts']}, n_analysts={config['n_analysts']}")
    print(f"  expert_encoder_output_dim={config['expert_encoder_output_dim']}")
    print(f"  analyst_input_dim={config['analyst_input_dim']}")
    
    # Create model
    print("\nCreating model...")
    model = CollectiveModel(config)
    print(model)
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(4, 784)
    
    # Regular forward
    logits = model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  ✓ Regular forward pass works")
    
    # Forward with intermediates
    logits, expert_outs, analyst_outs = model(x, return_intermediates=True)
    print(f"\n  With intermediates:")
    print(f"    Expert outputs: {len(expert_outs)} × {expert_outs[0].shape}")
    print(f"    Analyst outputs: {len(analyst_outs)} × {analyst_outs[0].shape}")
    print(f"    Final logits: {logits.shape}")
    print(f"  ✓ Forward with intermediates works")
    
    # Test gradient flow
    print("\n  Testing gradient flow...")
    loss = logits.sum()
    loss.backward()
    
    has_grad = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"  ✓ Gradients flow correctly: {has_grad}")
    
    print("\n✓ CollectiveModel tests passed!")

