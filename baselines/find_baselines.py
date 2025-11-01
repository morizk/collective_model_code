"""
Automatically find baseline architectures matching collective model parameter count.

Finds 6 different baseline types:
1. Shallow (2-3 layers, wide)
2. Balanced (4-5 layers, geometric progression)
3. Deep (8-10 layers, constant width)
4. Very Deep (12-15 layers, constant width)
5. Deep ResNet (8-10 layers with skip connections)
6. Very Deep ResNet (12-15 layers with skip connections)
"""

from collective_model.training import CollectiveModel
from collective_model.config import prepare_config
from collective_model.utils.param_counter import count_parameters

# Handle both package import and direct script execution
try:
    from .monolithic import MonolithicMLP, ResNetMLP
except ImportError:
    from baselines.monolithic import MonolithicMLP, ResNetMLP


def find_all_baselines(config):
    """
    Find 6 different baseline architectures matching collective param count.
    
    Args:
        config: Collective model configuration (will be prepared automatically)
        
    Returns:
        dict: {baseline_type: {hidden_dims, params, description, use_skip_connections}}
        
    Example:
        >>> from collective_model.config import CONFIG_DEBUG, prepare_config
        >>> config = prepare_config(CONFIG_DEBUG)
        >>> baselines = find_all_baselines(config)
        >>> print(baselines['shallow']['hidden_dims'])
    """
    # Step 1: Build collective and count params
    print("=" * 60)
    print("BUILDING COLLECTIVE MODEL")
    print("=" * 60)
    
    config = prepare_config(config)  # Add n_experts, n_analysts
    collective = CollectiveModel(config)
    target_params = count_parameters(collective)
    
    print(f"✓ Collective model built")
    print(f"  Total parameters: {target_params:,}")
    print(f"  Experts: {config['n_experts']}")
    print(f"  Analysts: {config['n_analysts']}")
    print()
    
    # Step 2: Find each baseline type
    print("=" * 60)
    print("SEARCHING FOR MATCHING BASELINES")
    print("=" * 60)
    
    baselines = {}
    tolerance = 0.05  # Within 5% of target
    
    # Type 1: SHALLOW (2-3 layers, wide)
    print("\n1. Shallow baseline (2-3 layers, wide)...")
    baselines['shallow'] = _find_shallow_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 2: BALANCED (4-5 layers, geometric)
    print("\n2. Balanced baseline (4-5 layers, geometric)...")
    baselines['balanced'] = _find_balanced_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 3: DEEP (8-10 layers, narrow)
    print("\n3. Deep baseline (8-10 layers, narrow)...")
    baselines['deep'] = _find_deep_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 4: VERY DEEP (12-15 layers, constant width)
    print("\n4. Very Deep baseline (12-15 layers, constant)...")
    baselines['very_deep'] = _find_very_deep_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 5: DEEP RESNET (8-10 layers with skip connections)
    print("\n5. Deep ResNet baseline (8-10 layers + skip connections)...")
    baselines['deep_resnet'] = _find_deep_resnet_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 6: VERY DEEP RESNET (12-15 layers with skip connections)
    print("\n6. Very Deep ResNet baseline (12-15 layers + skip connections)...")
    baselines['very_deep_resnet'] = _find_very_deep_resnet_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Step 3: Print summary
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"{'Type':<20} {'Layers':<8} {'Params':<15} {'Diff %':<10}")
    print("-" * 60)
    
    for name, baseline in baselines.items():
        diff_pct = abs(baseline['params'] - target_params) / target_params * 100
        print(f"{name:<20} {len(baseline['hidden_dims']):<8} "
              f"{baseline['params']:>12,}  {diff_pct:>6.2f}%")
    
    print(f"{'Target (collective)':<20} {'Multi':<8} {target_params:>12,}  {'0.00%':>10}")
    print("=" * 60)
    
    return baselines


def find_all_baselines_with_target(config, target_params):
    """
    Find 6 different baseline architectures matching a specific target parameter count.
    
    This is useful for sweeps where target_params is specified directly (no collective model needed).
    
    Args:
        config: Minimal config with input_dim and num_classes
        target_params: Target parameter count to match
        
    Returns:
        dict: {baseline_type: {hidden_dims, params, description, use_skip_connections}}
    """
    print("=" * 60)
    print(f"FINDING BASELINES FOR TARGET: {target_params:,} parameters")
    print("=" * 60)
    
    baselines = {}
    tolerance = 0.05  # Within 5% of target
    
    # Type 1: SHALLOW (2-3 layers, wide)
    print("\n1. Shallow baseline (2-3 layers, wide)...")
    baselines['shallow'] = _find_shallow_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 2: BALANCED (4-5 layers, geometric)
    print("\n2. Balanced baseline (4-5 layers, geometric)...")
    baselines['balanced'] = _find_balanced_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 3: DEEP (8-10 layers, narrow)
    print("\n3. Deep baseline (8-10 layers, narrow)...")
    baselines['deep'] = _find_deep_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 4: VERY DEEP (12-15 layers, very narrow)
    print("\n4. Very Deep baseline (12-15 layers, very narrow)...")
    baselines['very_deep'] = _find_very_deep_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 5: DEEP RESNET (8-10 layers with skip connections)
    print("\n5. Deep ResNet baseline (8-10 layers with skip connections)...")
    baselines['deep_resnet'] = _find_deep_resnet_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Type 6: VERY DEEP RESNET (12-15 layers with skip connections)
    print("\n6. Very Deep ResNet baseline (12-15 layers with skip connections)...")
    baselines['very_deep_resnet'] = _find_very_deep_resnet_baseline(
        config['input_dim'], 
        config['num_classes'], 
        target_params, 
        tolerance
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE SEARCH SUMMARY")
    print("=" * 60)
    print(f"Target: {target_params:,} parameters")
    print()
    for name, info in baselines.items():
        diff_pct = abs(info['params'] - target_params) / target_params * 100
        print(f"  {name:20s}: {info['params']:>10,} params ({diff_pct:>5.2f}% diff)")
        print(f"{'':22s}  {info['hidden_dims']}")
    print("=" * 60)
    
    return baselines


def _find_shallow_baseline(input_dim, num_classes, target_params, tolerance):
    """
    Find shallow baseline: 2-3 wide layers.
    
    Strategy: Few layers, many neurons per layer.
    Tests if width beats depth.
    """
    best_config = None
    best_diff = float('inf')
    
    # Adaptive search: For 2 layers [w1, w2], params ≈ input_dim*w1 + w1*w2 + w2*num_classes
    # Rough estimate: w1 ≈ sqrt(target / input_dim), w2 ≈ w1/2
    # Use wider search ranges to ensure we find something
    est_w1 = int((target_params / input_dim) ** 0.5)
    min_width1 = max(100, est_w1 // 3)
    max_width1 = min(5000, est_w1 * 4)
    
    print(f"  Searching shallow baseline ({min_width1}-{max_width1})...", end='', flush=True)
    
    # Try 2-layer networks (wide) with adaptive step (FASTER: fewer iterations)
    step1 = max(50, (max_width1 - min_width1) // 40)  # 40 iterations max instead of 100
    for width1 in range(min_width1, max_width1, step1):
        # w2 should be roughly w1/2 to w1*0.9
        min_width2 = max(50, width1 // 3)
        max_width2 = min(width1, int(width1 * 0.95))
        step2 = max(50, (max_width2 - min_width2) // 20)  # 20 iterations max instead of 50
        
        for width2 in range(min_width2, max_width2, step2):
            hidden_dims = [width1, width2]
            
            model = MonolithicMLP(input_dim, hidden_dims, num_classes)
            params = count_parameters(model)
            diff = abs(params - target_params)
            
            if diff < best_diff and diff < target_params * tolerance:
                best_diff = diff
                best_config = {
                    'hidden_dims': hidden_dims,
                    'params': params,
                    'description': f'Shallow & Wide ({len(hidden_dims)} layers)',
                    'use_skip_connections': False
                }
            
            # Early stop if very close
            if diff < target_params * 0.01:
                print(f" ✓ Found: {hidden_dims}, {params:,} params")
                return best_config
    
    if best_config is not None:
        print(f" ✓ Found!")
    
    if best_config is None:
        # Last resort: try even wider search
        print(f"\n  ⚠ Expanding search...")
        for width1 in range(100, 2000, 40):  # Faster: step=40 instead of 20
            for width2 in range(50, width1, 40):  # Faster: step=40 instead of 20
                hidden_dims = [width1, width2]
                model = MonolithicMLP(input_dim, hidden_dims, num_classes)
                params = count_parameters(model)
                diff = abs(params - target_params)
                if diff < best_diff and diff < target_params * tolerance:
                    best_diff = diff
                    best_config = {
                        'hidden_dims': hidden_dims,
                        'params': params,
                        'description': f'Shallow & Wide ({len(hidden_dims)} layers)',
                        'use_skip_connections': False
                    }
                if diff < target_params * 0.01:
                    print(f"  ✓ Found: {hidden_dims}, {params:,} params")
                    return best_config
    
    if best_config is None:
        raise ValueError(f"Could not find shallow baseline within {tolerance*100}% tolerance")
    
    print(f"  ✓ Found: {best_config['hidden_dims']}, {best_config['params']:,} params")
    return best_config


def _find_balanced_baseline(input_dim, num_classes, target_params, tolerance):
    """
    Find balanced baseline: 4-5 layers, geometric progression.
    
    Strategy: Standard practice, each layer ~half of previous.
    """
    best_config = None
    best_diff = float('inf')
    
    # Adaptive search range based on target params
    # For geometric progression [s, s/2, s/4, s/8]:
    # params ≈ input*s + s*s/2 + s/2*s/4 + s/4*s/8 + s/8*output
    # Rough estimate: params ≈ s^2, so s ≈ sqrt(target_params / 2)
    est_scale = int((target_params / 2) ** 0.5)
    max_scale = min(5000, est_scale * 2)
    min_scale = max(200, est_scale // 2)
    step = max(50, (max_scale - min_scale) // 50)
    
    print(f"  Searching scale {min_scale}-{max_scale}...", end='', flush=True)
    
    # Try different starting scales
    for scale in range(min_scale, max_scale, step):
        # Geometric progression: scale → scale/2 → scale/4 → scale/8
        hidden_dims = [scale, scale//2, scale//4, scale//8]
        # Filter out too-small layers
        hidden_dims = [h for h in hidden_dims if h > num_classes * 2]
        
        if len(hidden_dims) < 3:  # Need at least 3 layers for "balanced"
            continue
        
        model = MonolithicMLP(input_dim, hidden_dims, num_classes)
        params = count_parameters(model)
        diff = abs(params - target_params)
        
        if diff < best_diff and diff < target_params * tolerance:
            best_diff = diff
            best_config = {
                'hidden_dims': hidden_dims,
                'params': params,
                'description': f'Balanced Geometric ({len(hidden_dims)} layers)',
                'use_skip_connections': False
            }
        
        if diff < target_params * 0.01:
            break
    
    if best_config is None:
        # Last resort: try much wider search
        print(f"  ⚠ Initial search failed, trying wider range...")
        wider_max = min(8000, max_scale * 4)
        for scale in range(100, wider_max, 100):  # Larger step for speed
            hidden_dims = [scale, scale//2, scale//4, scale//8]
            hidden_dims = [h for h in hidden_dims if h > num_classes * 2]
            if len(hidden_dims) < 3:
                continue
            model = MonolithicMLP(input_dim, hidden_dims, num_classes)
            params = count_parameters(model)
            diff = abs(params - target_params)
            if diff < best_diff and diff < target_params * tolerance:
                best_diff = diff
                best_config = {
                    'hidden_dims': hidden_dims,
                    'params': params,
                    'description': f'Balanced Geometric ({len(hidden_dims)} layers)',
                    'use_skip_connections': False
                }
            if diff < target_params * 0.01:
                break
    
    if best_config is None:
        raise ValueError(f"Could not find balanced baseline within {tolerance*100}% tolerance")
    
    print(f"  ✓ Found: {best_config['hidden_dims']}, {best_config['params']:,} params")
    return best_config


def _find_deep_baseline(input_dim, num_classes, target_params, tolerance):
    """
    Find deep baseline: 8-10 layers, constant width.
    
    Strategy: Test if depth helps (hierarchical features).
    """
    best_config = None
    best_diff = float('inf')
    
    print(f"  Searching deep baseline...", end='', flush=True)
    
    # Adaptive search: for constant width, params ≈ 8 * width^2 + (input_dim + num_classes) * width
    # Estimate width from target params
    num_layers = 8
    est_width = int(((target_params - input_dim * num_classes) / (num_layers + 1)) ** 0.5)
    max_width = min(1000, est_width * 2)
    min_width = max(100, est_width // 3)
    step = max(30, (max_width - min_width) // 30)  # Faster: 30 iterations max
    
    # Try different constant widths for 8 layers
    for width in range(min_width, max_width, step):
        hidden_dims = [width] * 8  # 8 identical layers
        
        model = MonolithicMLP(input_dim, hidden_dims, num_classes)
        params = count_parameters(model)
        diff = abs(params - target_params)
        
        if diff < best_diff and diff < target_params * tolerance:
            best_diff = diff
            best_config = {
                'hidden_dims': hidden_dims,
                'params': params,
                'description': f'Deep & Narrow ({len(hidden_dims)} layers)',
                'use_skip_connections': False
            }
        
        if diff < target_params * 0.01:
            break
    
    if best_config is None:
        # Last resort: try wider search
        print(f"  ⚠ Initial search failed, trying wider range...")
        for width in range(50, 1500, 20):
            hidden_dims = [width] * 8
            model = MonolithicMLP(input_dim, hidden_dims, num_classes)
            params = count_parameters(model)
            diff = abs(params - target_params)
            if diff < best_diff and diff < target_params * tolerance:
                best_diff = diff
                best_config = {
                    'hidden_dims': hidden_dims,
                    'params': params,
                    'description': f'Deep & Narrow ({len(hidden_dims)} layers)',
                    'use_skip_connections': False
                }
            if diff < target_params * 0.01:
                break
    
    if best_config is None:
        raise ValueError(f"Could not find deep baseline within {tolerance*100}% tolerance")
    
    print(f"  ✓ Found: {best_config['hidden_dims']}, {best_config['params']:,} params")
    return best_config


def _find_very_deep_baseline(input_dim, num_classes, target_params, tolerance):
    """
    Find very deep baseline: 12-15 layers, constant width.
    
    Strategy: Extreme depth test (may suffer from vanishing gradients).
    """
    best_config = None
    best_diff = float('inf')
    
    print(f"  Searching very deep baseline...", end='', flush=True)
    
    # Adaptive search: for constant width, params ≈ 12 * width^2 + (input_dim + num_classes) * width
    num_layers = 12
    est_width = int(((target_params - input_dim * num_classes) / (num_layers + 1)) ** 0.5)
    max_width = min(800, est_width * 2)
    min_width = max(100, est_width // 3)
    step = max(30, (max_width - min_width) // 30)  # Faster: 30 iterations max
    
    # Try different widths for 12 layers
    for width in range(min_width, max_width, step):
        hidden_dims = [width] * 12  # 12 identical layers
        
        model = MonolithicMLP(input_dim, hidden_dims, num_classes)
        params = count_parameters(model)
        diff = abs(params - target_params)
        
        if diff < best_diff and diff < target_params * tolerance:
            best_diff = diff
            best_config = {
                'hidden_dims': hidden_dims,
                'params': params,
                'description': f'Very Deep ({len(hidden_dims)} layers)',
                'use_skip_connections': False
            }
        
        if diff < target_params * 0.01:
            break
    
    if best_config is None:
        # Last resort: try wider search
        print(f"  ⚠ Initial search failed, trying wider range...")
        for width in range(50, 1200, 20):
            hidden_dims = [width] * 12
            model = MonolithicMLP(input_dim, hidden_dims, num_classes)
            params = count_parameters(model)
            diff = abs(params - target_params)
            if diff < best_diff and diff < target_params * tolerance:
                best_diff = diff
                best_config = {
                    'hidden_dims': hidden_dims,
                    'params': params,
                    'description': f'Very Deep ({len(hidden_dims)} layers)',
                    'use_skip_connections': False
                }
            if diff < target_params * 0.01:
                break
    
    if best_config is None:
        raise ValueError(f"Could not find very deep baseline within {tolerance*100}% tolerance")
    
    print(f"  ✓ Found: {best_config['hidden_dims']}, {best_config['params']:,} params")
    return best_config


def _find_deep_resnet_baseline(input_dim, num_classes, target_params, tolerance):
    """
    Find deep ResNet-style baseline: 8-10 layers with skip connections.
    
    Strategy: Modern architecture with residual connections.
    Uses constant width (standard ResNet practice) - only needs one projection (input→first layer).
    """
    best_config = None
    best_diff = float('inf')
    
    print(f"  Searching deep ResNet baseline...", end='', flush=True)
    
    # Adaptive search: ResNet adds ~150K params (BatchNorm + projection) vs regular MLP
    # So ResNet needs smaller width for same param count
    # Estimate: ResNet params ≈ MLP params + 150K, so find MLP width that gives (target - 150K)
    num_layers = 8
    adjusted_target = max(target_params - 150000, target_params // 2)  # Account for ResNet overhead
    est_width = int(((adjusted_target - input_dim * num_classes) / (num_layers + 1)) ** 0.5)
    max_width = min(1000, est_width * 3)
    min_width = max(50, est_width // 4)
    step = max(20, (max_width - min_width) // 60)
    
    # Try different constant widths for 8 layers with skips
    # Constant width ensures minimal projections (only input→first layer)
    for width in range(min_width, max_width, step):
        hidden_dims = [width] * 8
        
        model = ResNetMLP(input_dim, hidden_dims, num_classes)
        params = count_parameters(model)  # Includes BatchNorm and projections
        diff = abs(params - target_params)
        
        if diff < best_diff and diff < target_params * tolerance:
            best_diff = diff
            best_config = {
                'hidden_dims': hidden_dims,
                'params': params,
                'description': f'Deep ResNet ({len(hidden_dims)} layers + skips)',
                'use_skip_connections': True
            }
        
        if diff < target_params * 0.01:
            break
    
    if best_config is None:
        # Last resort: try wider search (ResNet needs smaller widths)
        print(f"  ⚠ Initial search failed, trying wider range...")
        for width in range(50, 500, 15):
            hidden_dims = [width] * 8
            model = ResNetMLP(input_dim, hidden_dims, num_classes)
            params = count_parameters(model)
            diff = abs(params - target_params)
            if diff < best_diff and diff < target_params * tolerance:
                best_diff = diff
                best_config = {
                    'hidden_dims': hidden_dims,
                    'params': params,
                    'description': f'Deep ResNet ({len(hidden_dims)} layers + skips)',
                    'use_skip_connections': True
                }
            if diff < target_params * 0.01:
                break
    
    if best_config is None:
        raise ValueError(f"Could not find deep ResNet baseline within {tolerance*100}% tolerance")
    
    print(f"  ✓ Found: {best_config['hidden_dims']}, {best_config['params']:,} params")
    return best_config


def _find_very_deep_resnet_baseline(input_dim, num_classes, target_params, tolerance):
    """
    Find very deep ResNet baseline: 12-15 layers with skip connections.
    
    Strategy: Best possible deep baseline (modern practice).
    Uses constant width (standard ResNet practice) - only needs one projection (input→first layer).
    """
    best_config = None
    best_diff = float('inf')
    
    print(f"  Searching very deep ResNet baseline...", end='', flush=True)
    
    # Adaptive search: ResNet adds ~150K params (BatchNorm + projection) vs regular MLP
    # So ResNet needs smaller width for same param count
    num_layers = 12
    adjusted_target = max(target_params - 150000, target_params // 2)  # Account for ResNet overhead
    est_width = int(((adjusted_target - input_dim * num_classes) / (num_layers + 1)) ** 0.5)
    max_width = min(800, est_width * 3)
    min_width = max(50, est_width // 4)
    step = max(20, (max_width - min_width) // 60)
    
    # Try different widths for 12 layers with skips
    # Constant width ensures minimal projections (only input→first layer)
    for width in range(min_width, max_width, step):
        hidden_dims = [width] * 12
        
        model = ResNetMLP(input_dim, hidden_dims, num_classes)
        params = count_parameters(model)  # Includes BatchNorm and projections
        diff = abs(params - target_params)
        
        if diff < best_diff and diff < target_params * tolerance:
            best_diff = diff
            best_config = {
                'hidden_dims': hidden_dims,
                'params': params,
                'description': f'Very Deep ResNet ({len(hidden_dims)} layers + skips)',
                'use_skip_connections': True
            }
        
        if diff < target_params * 0.01:
            break
    
    if best_config is None:
        # Last resort: try wider search (ResNet needs smaller widths for 12 layers)
        print(f"  ⚠ Initial search failed, trying wider range...")
        # Reset best_diff for fallback search
        best_diff = float('inf')
        for width in range(50, 250, 10):
            hidden_dims = [width] * 12
            model = ResNetMLP(input_dim, hidden_dims, num_classes)
            params = count_parameters(model)
            diff = abs(params - target_params)
            if diff < best_diff and diff < target_params * tolerance:
                best_diff = diff
                best_config = {
                    'hidden_dims': hidden_dims,
                    'params': params,
                    'description': f'Very Deep ResNet ({len(hidden_dims)} layers + skips)',
                    'use_skip_connections': True
                }
            if diff < target_params * 0.01:
                break
    
    if best_config is None:
        raise ValueError(f"Could not find very deep ResNet baseline within {tolerance*100}% tolerance")
    
    print(f"  ✓ Found: {best_config['hidden_dims']}, {best_config['params']:,} params")
    return best_config


if __name__ == '__main__':
    # Test with debug config
    from collective_model.config import CONFIG_DEBUG, prepare_config
    
    print("Testing baseline finding with CONFIG_DEBUG...")
    config = prepare_config(CONFIG_DEBUG)
    
    baselines = find_all_baselines(config)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Found {len(baselines)} baseline architectures")
    for name, baseline in baselines.items():
        print(f"  {name}: {baseline['hidden_dims']} → {baseline['params']:,} params")

