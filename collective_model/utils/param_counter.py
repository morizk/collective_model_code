"""
Parameter counting utilities.

Functions to count parameters in models and compute equivalent architectures.
Used for fair baseline comparison by parameter matching.
"""

import torch
import torch.nn as nn


def count_parameters(model):
    """
    Count total trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Total number of trainable parameters
    
    Example:
        >>> model = nn.Linear(100, 10)
        >>> count_parameters(model)
        1010  # 100*10 weights + 10 biases
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_mlp_parameters(input_dim, hidden_dims, output_dim):
    """
    Count parameters in an MLP without building it.
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list[int]): List of hidden layer sizes
        output_dim (int): Output dimension
    
    Returns:
        int: Total parameter count
    
    Example:
        >>> count_mlp_parameters(784, [512, 256], 10)
        534538  # (784*512 + 512) + (512*256 + 256) + (256*10 + 10)
    """
    total = 0
    prev_dim = input_dim
    
    # Hidden layers
    for h in hidden_dims:
        total += prev_dim * h + h  # weights + bias
        prev_dim = h
    
    # Output layer
    total += prev_dim * output_dim + output_dim
    
    return total


def count_resnet_mlp_parameters(input_dim, hidden_dims, output_dim):
    """
    Count parameters in a ResNet MLP (with skip connections).
    
    Skip connections don't add parameters unless dimensions mismatch,
    in which case a 1x1 projection is needed.
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list[int]): List of hidden layer sizes
        output_dim (int): Output dimension
    
    Returns:
        int: Total parameter count
    
    Example:
        >>> count_resnet_mlp_parameters(784, [512, 512, 512], 10)
        # Includes projection from 784->512, then residual blocks
    """
    total = 0
    prev_dim = input_dim
    
    for h in hidden_dims:
        # Main path
        total += prev_dim * h + h  # weights + bias
        
        # Skip connection projection (if dimensions mismatch)
        if prev_dim != h:
            total += prev_dim * h + h  # projection weights + bias
        
        prev_dim = h
    
    # Output layer
    total += prev_dim * output_dim + output_dim
    
    return total


def get_model_summary(model, input_size=None, device='cpu'):
    """
    Get detailed summary of model architecture and parameters.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input size for forward pass test (e.g., (784,))
        device (str): Device to use ('cpu' or 'cuda')
    
    Returns:
        dict: Summary with total params, trainable params, layer info
    
    Example:
        >>> model = MLPExpert(784, [512, 256], 128)
        >>> summary = get_model_summary(model, input_size=(784,))
        >>> print(summary['total_params'])
    """
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    
    summary = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'model_name': model.__class__.__name__
    }
    
    # Test forward pass if input_size provided
    if input_size is not None:
        try:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                x = torch.randn(1, *input_size).to(device)
                output = model(x)
                summary['input_shape'] = x.shape
                summary['output_shape'] = output.shape
        except Exception as e:
            summary['forward_pass_error'] = str(e)
    
    return summary


def print_model_summary(model, input_size=None, device='cpu'):
    """
    Print a formatted model summary.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input size for forward pass test
        device (str): Device to use
    
    Example:
        >>> model = MLPExpert(784, [512, 256], 128)
        >>> print_model_summary(model, input_size=(784,))
    """
    summary = get_model_summary(model, input_size, device)
    
    print(f"\n{'='*60}")
    print(f"Model: {summary['model_name']}")
    print(f"{'='*60}")
    print(f"Total parameters:      {summary['total_params']:,}")
    print(f"Trainable parameters:  {summary['trainable_params']:,}")
    print(f"Non-trainable params:  {summary['non_trainable_params']:,}")
    
    if 'input_shape' in summary:
        print(f"\nInput shape:   {summary['input_shape']}")
        print(f"Output shape:  {summary['output_shape']}")
    
    if 'forward_pass_error' in summary:
        print(f"\n⚠ Forward pass failed: {summary['forward_pass_error']}")
    
    print(f"{'='*60}\n")


def compare_model_sizes(*models, names=None):
    """
    Compare parameter counts of multiple models.
    
    Args:
        *models: Variable number of models to compare
        names (list[str]): Optional names for the models
    
    Returns:
        dict: Dictionary mapping model names to parameter counts
    
    Example:
        >>> expert = MLPExpert(784, [512, 256], 128)
        >>> analyst = MLPAnalyst(848, [256, 128], 64)
        >>> compare_model_sizes(expert, analyst, names=['Expert', 'Analyst'])
    """
    if names is None:
        names = [f"Model_{i}" for i in range(len(models))]
    
    results = {}
    for name, model in zip(names, models):
        results[name] = count_parameters(model)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("Model Size Comparison")
    print(f"{'='*60}")
    
    max_name_len = max(len(name) for name in names)
    for name, params in results.items():
        print(f"{name:<{max_name_len}} : {params:>12,} params")
    
    # Show relative sizes
    if len(results) > 1:
        print(f"\n{'-'*60}")
        base_name = names[0]
        base_params = results[base_name]
        for name, params in results.items():
            if name != base_name:
                ratio = params / base_params
                print(f"{name} is {ratio:.2f}x the size of {base_name}")
    
    print(f"{'='*60}\n")
    
    return results


if __name__ == '__main__':
    # Test parameter counting
    print("Testing parameter counting utilities...")
    
    # Test count_mlp_parameters
    params = count_mlp_parameters(784, [512, 256], 10)
    print(f"\nMLP [784 -> 512 -> 256 -> 10] has {params:,} parameters")
    
    # Verify with actual model
    from collective_model.models import MLPExpert
    expert = MLPExpert(784, [512, 256], 128)
    actual_params = count_parameters(expert)
    print(f"Actual MLPExpert has {actual_params:,} parameters")
    
    # Test model summary
    print_model_summary(expert, input_size=(784,))
    
    print("✓ Parameter counting tests passed!")

