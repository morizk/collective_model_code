"""
Configuration system for Collective Model experiments.

Provides standard configurations for debugging and experiments,
plus a prepare_config() function to compute derived values.
"""

import copy


# =============================================================================
# Configuration Templates
# =============================================================================

CONFIG_DEBUG = {
    # Model architecture
    'n_total': 8,  # Total number of models (experts + analysts)
    'expert_ratio': 0.2,  # Ratio of experts (will compute n_experts, n_analysts)
    'expert_hidden': [128, 64],  # Expert hidden layers (LARGE)
    'expert_output': 32,  # Expert output dimension
    'analyst_hidden': [64, 32],  # Analyst hidden layers (SMALLER)
    'analyst_output': 16,  # Analyst output dimension
    'c_expert': 0.25,  # Expert encoder compression ratio
    'c_collective': 1.0,  # Multiplier for adaptive collective input dimension
    'adaptive_collective_min_dim': 64,  # Extreme min bound (will be manually engineered)
    'adaptive_collective_max_dim': 2048,  # Extreme max bound (will be manually engineered)
    'collective_version': 'simple_mlp',  # 'simple_mlp' or 'encoder_head'
    'collective_hidden_scale': 1.0,  # Scale for collective hidden layer
    
    # Diversity
    'use_expert_diversity': False,  # Enable diversity loss for experts
    'use_analyst_diversity': False,  # Enable diversity loss for analysts (independent)
    'diversity_lambda': 0.01,  # Weight for diversity loss
    'diversity_temperature': 1.0,  # Temperature for diversity (higher = less aggressive)
    'vary_expert_architectures': False,  # Use different expert architectures
    'vary_analyst_architectures': False,  # Use different analyst architectures
    
    # Model-level Dropout (ensemble regularization)
    'use_drop_models': False,  # Enable model-level dropout (drop entire models during training)
    'drop_models_expert_rate': 0.1,  # Dropout rate for experts (0.0-1.0, each expert has this prob of being dropped)
    'drop_models_analyst_rate': 0.1,  # Dropout rate for analysts (0.0-1.0, each analyst has this prob of being dropped)
    
    # Training
    'batch_size': 128,  # Optimized for accuracy
    'gradient_accumulation_steps': 4,  # Accumulate over 4 steps → effective batch=512 (better GPU utilization)
    'eval_batch_size': 512,  # Larger batch size for validation/test (smoother metrics, no gradients)
    'use_mixed_precision': True,  # Use FP16 for 1.5-2x speedup
    'learning_rate': 0.001,
    'epochs': 5,
    'optimizer': 'adam',  # 'adam', 'sgd', 'adamw' - Adam is good default for deep models
    'weight_decay': 0.0,
    'use_augmentation': True,  # Default: Use data augmentation (doesn't apply to val/test)
    'topk': (1,),  # Top-k accuracy: (1,) for top-1, (1, 5) for top-1 and top-5
    
    # Data
    'dataset': 'fashion_mnist',  # Options: 'mnist' (too easy), 'fashion_mnist' (recommended), 'cifar10' (hard for MLPs)
    'num_workers': 6,  # Increased for faster data loading (multithreading)
    'val_split': 0.1,
    
    # System
    'device': 'cuda',  # 'cuda' or 'cpu'
    'seed': 42,
    'log_interval': 10,  # Log every N batches
}


CONFIG_PHASE1 = {
    # Model architecture
    'n_total': 8,  # Total number of models
    'expert_ratio': 0.2,  # 20% experts (will be ~2 experts, 6 analysts for n_total=8)
    'expert_hidden': [512, 256],  # Expert hidden layers (LARGE)
    'expert_output': 128,  # Expert output dimension
    'analyst_hidden': [256, 128],  # Analyst hidden layers (SMALLER)
    'analyst_output': 64,  # Analyst output dimension
    'c_expert': 0.25,  # Expert encoder compression ratio
    'c_collective': 1.0,  # Multiplier for adaptive collective input dimension
    'adaptive_collective_min_dim': 64,  # Extreme min bound (will be manually engineered)
    'adaptive_collective_max_dim': 2048,  # Extreme max bound (will be manually engineered)
    'collective_version': 'simple_mlp',  # Start with simple version
    'collective_hidden_scale': 1.0,
    
    # Diversity
    'use_expert_diversity': False,  # Enable diversity loss for experts
    'use_analyst_diversity': False,  # Enable diversity loss for analysts (independent)
    'diversity_lambda': 0.01,
    'diversity_temperature': 1.0,
    'vary_expert_architectures': False,  # Start with uniform architectures
    'vary_analyst_architectures': False,
    
    # Model-level Dropout (ensemble regularization)
    'use_drop_models': False,  # Enable model-level dropout (drop entire models during training)
    'drop_models_expert_rate': 0.1,  # Dropout rate for experts (0.0-1.0, each expert has this prob of being dropped)
    'drop_models_analyst_rate': 0.1,  # Dropout rate for analysts (0.0-1.0, each analyst has this prob of being dropped)
    
    # Training
    'batch_size': 128,  # Optimized for accuracy (wandb shows better than 512)
    'gradient_accumulation_steps': 4,  # Accumulate over 4 steps → effective batch=512 (better GPU utilization)
    'eval_batch_size': 1024,  # Larger batch size for validation/test (smoother metrics, no gradients)
    'use_mixed_precision': True,  # Use FP16 for 1.5-2x speedup
    'learning_rate': 0.001,
    'epochs': 100,
    'optimizer': 'adam',  # 'adam', 'sgd', 'adamw' - Adam is good default for deep models
    'weight_decay': 1e-4,
    'use_augmentation': True,  # Default: Use data augmentation (only for training, not val/test)
    'topk': (1,),  # Top-k accuracy: (1,) for top-1, (1, 5) for top-1 and top-5
    'use_torch_compile': False,  # Use torch.compile() for 20-30% speedup (PyTorch 2.0+)
    'use_mixed_precision': False,  # Use FP16 training for 1.5-2x speedup (requires compatible GPU)
    
    # Data
    'dataset': 'fashion_mnist',  # Options: 'mnist' (too easy), 'fashion_mnist' (recommended), 'cifar10' (hard for MLPs)
    'num_workers': 8,  # Increased for faster data loading (multithreading)
    'val_split': 0.1,
    
    # System
    'device': 'cuda',
    'seed': 42,
    'log_interval': 50,
}


# =============================================================================
# Configuration Helper Functions
# =============================================================================

def prepare_config(config):
    """
    Prepare configuration by computing derived values.
    
    This function takes a base config and computes:
    - n_experts and n_analysts from n_total and expert_ratio
    - expert_encoder_output_dim from n_experts, expert_output, and c_expert
    - analyst_input_dim (original_input + expert_encoder_output_dim)
    - collective_input_dim from n_analysts and analyst_output
    - Any other derived values
    
    Args:
        config (dict): Base configuration dictionary
    
    Returns:
        dict: Updated configuration with derived values
    
    Example:
        >>> config = prepare_config(CONFIG_DEBUG)
        >>> print(config['n_experts'], config['n_analysts'])
        2 6
    """
    # Make a copy to avoid modifying original
    config = copy.deepcopy(config)
    
    # Compute number of experts and analysts
    n_total = config['n_total']
    
    # Support both expert_ratio (old) and n_experts (new, preferred)
    # IMPORTANT: Always ensure at least 2 experts (needed for diversity loss)
    if 'n_experts' in config:
        # New way: n_experts specified directly
        n_experts = max(2, min(int(config['n_experts']), n_total - 1))  # Minimum 2 experts, ensure at least 1 analyst
        # Also compute expert_ratio for backwards compatibility
        if 'expert_ratio' not in config:
            config['expert_ratio'] = n_experts / n_total
    elif 'expert_ratio' in config:
        # Old way: expert_ratio specified, compute n_experts
        expert_ratio = config['expert_ratio']
        n_experts_calculated = round(n_total * expert_ratio)
        n_experts = max(2, n_experts_calculated)  # Minimum 2 experts (needed for diversity loss)
    else:
        raise ValueError("Either 'n_experts' or 'expert_ratio' must be specified in config")
    
    # Ensure we have at least 1 analyst (adjust n_experts if needed to maintain n_total)
    if n_experts >= n_total:
        n_experts = n_total - 1  # Leave room for at least 1 analyst
    
    n_analysts = n_total - n_experts  # Adjust analysts to maintain n_total
    
    config['n_experts'] = n_experts
    config['n_analysts'] = n_analysts
    
    # Get dataset info for input dimension
    dataset = config['dataset'].lower()
    
    if dataset == 'mnist':
        config['input_dim'] = 784
        config['num_classes'] = 10
    elif dataset in ['fashion_mnist', 'fashion-mnist', 'fmnist']:
        config['input_dim'] = 784  # Same as MNIST (28x28)
        config['num_classes'] = 10
    elif dataset in ['cifar10', 'cifar-10']:
        config['input_dim'] = 3072  # 3 * 32 * 32 (RGB)
        config['num_classes'] = 10
    else:
        raise NotImplementedError(f"Dataset {dataset} not yet supported. Supported: ['mnist', 'fashion_mnist', 'cifar10']")
    
    # Compute expert encoder output dimension (compressed expert features)
    expert_concat_dim = n_experts * config['expert_output']
    expert_encoder_output_dim = int(expert_concat_dim * config['c_expert'])
    config['expert_encoder_output_dim'] = expert_encoder_output_dim
    
    # Compute analyst input dimension (original + expert encoder output)
    analyst_input_dim = config['input_dim'] + expert_encoder_output_dim
    config['analyst_input_dim'] = analyst_input_dim
    
    # Compute adaptive collective input dimension using logarithmic scaling
    import math
    analyst_output = config['analyst_output']
    n_analysts = config['n_analysts']
    min_dim = config.get('adaptive_collective_min_dim', 64)
    max_dim = config.get('adaptive_collective_max_dim', 2048)
    
    # Adaptive formula: min(max_size, max(min_size, analyst_output * (1 + log2(n_analysts))))
    log_calc = analyst_output * (1 + math.log2(max(n_analysts, 1)))  # Prevent log2(0)
    adaptive_dim = min(max_dim, max(min_dim, int(log_calc)))
    
    # Apply c_collective multiplier
    collective_input_dim = int(adaptive_dim * config['c_collective'])
    config['collective_input_dim'] = collective_input_dim
    
    # If using encoder_head collective, this is now just for compatibility (not used if analyst encoder is added)
    if config['collective_version'] == 'encoder_head':
        # For v2, we might still compress further, but this is now secondary
        compressed_collective_dim = int(collective_input_dim * config.get('c_collective_v2', 0.5))
        config['compressed_collective_dim'] = compressed_collective_dim
    
    return config


def validate_config(config):
    """
    Validate configuration for consistency and feasibility.
    
    Args:
        config (dict): Configuration to validate
    
    Returns:
        tuple: (is_valid, error_messages)
    
    Example:
        >>> config = prepare_config(CONFIG_DEBUG)
        >>> is_valid, errors = validate_config(config)
        >>> if not is_valid:
        ...     print("Errors:", errors)
    """
    errors = []
    
    # Check required fields
    required_fields = [
        'n_total', 'expert_ratio', 'expert_hidden', 'expert_output',
        'analyst_hidden', 'analyst_output', 'batch_size', 'learning_rate', 'epochs'
    ]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, errors
    
    # Check value ranges
    if config['n_total'] < 2:
        errors.append("n_total must be at least 2")
    
    # Validate expert_ratio if provided (backwards compatibility)
    if 'expert_ratio' in config:
        if not (0 < config['expert_ratio'] < 1):
            errors.append("expert_ratio must be between 0 and 1")
    
    # Validate n_experts if provided (new preferred method)
    if 'n_experts' in config:
        n_total = config.get('n_total', 0)
        n_experts = config['n_experts']
        if not (2 <= n_experts < n_total):  # Minimum 2 experts (needed for diversity loss)
            errors.append(f"n_experts must be between 2 and {n_total-1} (minimum 2 for diversity, must leave room for at least 1 analyst)")
    
    # Ensure at least one method is provided
    if 'expert_ratio' not in config and 'n_experts' not in config:
        errors.append("Either 'expert_ratio' or 'n_experts' must be specified")
    
    if config['batch_size'] < 1:
        errors.append("batch_size must be positive")
    
    if config['learning_rate'] <= 0:
        errors.append("learning_rate must be positive")
    
    if config['epochs'] < 1:
        errors.append("epochs must be at least 1")
    
    # Check that experts are larger than analysts (if config is prepared)
    if 'n_experts' in config:
        # Check hidden layer sizes
        if len(config['expert_hidden']) < 2:
            errors.append("expert_hidden must have at least 2 layers")
        if len(config['analyst_hidden']) < 2:
            errors.append("analyst_hidden must have at least 2 layers")
        
        # Experts should generally be larger than analysts
        expert_avg = sum(config['expert_hidden']) / len(config['expert_hidden'])
        analyst_avg = sum(config['analyst_hidden']) / len(config['analyst_hidden'])
        if expert_avg < analyst_avg:
            errors.append(
                f"Warning: Experts (avg={expert_avg:.0f}) are smaller than "
                f"analysts (avg={analyst_avg:.0f}). Experts should be LARGER."
            )
    
    # Check diversity settings
    if config.get('use_diversity_loss', False):
        if config.get('diversity_lambda', 0) <= 0:
            errors.append("diversity_lambda must be positive when using diversity loss")
    
    return len(errors) == 0, errors


def print_config(config):
    """
    Print configuration in a readable format.
    
    Args:
        config (dict): Configuration to print
    """
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    
    sections = {
        'Architecture': [
            'n_total', 'expert_ratio', 'n_experts', 'n_analysts',
            'expert_hidden', 'expert_output', 'analyst_hidden', 'analyst_output',
            'c_expert', 'c_collective', 'adaptive_collective_min_dim', 'adaptive_collective_max_dim', 'collective_version', 'collective_hidden_scale'
        ],
        'Derived Dimensions': [
            'input_dim', 'num_classes', 'expert_encoder_output_dim',
            'analyst_input_dim', 'collective_input_dim'
        ],
        'Diversity': [
            'use_diversity_loss', 'diversity_lambda', 'diversity_temperature',
            'vary_expert_architectures', 'vary_analyst_architectures'
        ],
        'Model Dropout': [
            'use_drop_models', 'drop_models_expert_rate', 'drop_models_analyst_rate'
        ],
        'Training': [
            'batch_size', 'learning_rate', 'epochs', 'optimizer',
            'weight_decay', 'use_augmentation'
        ],
        'Data': [
            'dataset', 'num_workers', 'val_split'
        ],
        'System': [
            'device', 'seed', 'log_interval'
        ]
    }
    
    for section_name, keys in sections.items():
        print(f"\n{section_name}:")
        for key in keys:
            if key in config:
                value = config[key]
                print(f"  {key:30s} : {value}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    # Test configuration system
    print("Testing configuration system...\n")
    
    # Test DEBUG config
    print("1. Testing CONFIG_DEBUG:")
    config_debug = prepare_config(CONFIG_DEBUG)
    is_valid, errors = validate_config(config_debug)
    if is_valid:
        print("✓ DEBUG config is valid")
        print(f"  n_experts={config_debug['n_experts']}, n_analysts={config_debug['n_analysts']}")
        print(f"  expert_encoder_output_dim={config_debug['expert_encoder_output_dim']}")
        print(f"  analyst_input_dim={config_debug['analyst_input_dim']}")
        print(f"  collective_input_dim={config_debug['collective_input_dim']}")
    else:
        print("✗ DEBUG config has errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Test PHASE1 config
    print("\n2. Testing CONFIG_PHASE1:")
    config_phase1 = prepare_config(CONFIG_PHASE1)
    is_valid, errors = validate_config(config_phase1)
    if is_valid:
        print("✓ PHASE1 config is valid")
        print(f"  n_experts={config_phase1['n_experts']}, n_analysts={config_phase1['n_analysts']}")
        print(f"  expert_encoder_output_dim={config_phase1['expert_encoder_output_dim']}")
        print(f"  analyst_input_dim={config_phase1['analyst_input_dim']}")
        print(f"  collective_input_dim={config_phase1['collective_input_dim']}")
    else:
        print("✗ PHASE1 config has errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Print full config
    print("\n3. Full DEBUG config:")
    print_config(config_debug)
    
    print("✓ Configuration system tests passed!")

