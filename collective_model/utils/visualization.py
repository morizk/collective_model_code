"""
Model visualization utilities.

Includes functions to plot model architecture graphs.
"""

import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np


def plot_model_architecture(model, config, output_path='model_architecture.png', figsize=(18, 12)):
    """
    Plot a detailed architecture diagram of the Collective Model.
    
    Accurately shows: Input → Experts → Encoder → [Input + Encoded] → Analysts → Collective → Output
    
    Args:
        model: CollectiveModel instance
        config: Configuration dictionary (prepared)
        output_path: Path to save the image
        figsize: Figure size (width, height)
    
    Returns:
        str: Path to saved image
    
    Example:
        >>> from collective_model.training import CollectiveModel
        >>> from collective_model.config import CONFIG_DEBUG, prepare_config
        >>> config = prepare_config(CONFIG_DEBUG)
        >>> model = CollectiveModel(config)
        >>> plot_model_architecture(model, config, 'collective_architecture.png')
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#E8F4F8',
        'expert': '#FFE5B4',
        'encoder': '#FFCCCB',
        'analyst': '#E0E0FF',
        'collective': '#D4EDDA',
        'output': '#F0E68C',
        'arrow': '#333333',
        'skip': '#4169E1'
    }
    
    # Get config values
    n_experts = config.get('n_experts', 2)
    n_analysts = config.get('n_analysts', 6)
    input_dim = config.get('input_dim', 784)
    expert_output = config.get('expert_output', 32)
    expert_concat_dim = n_experts * expert_output
    expert_encoder_output_dim = config.get('expert_encoder_output_dim', 16)
    analyst_input_dim = config.get('analyst_input_dim', 800)  # input_dim + expert_encoder_output_dim
    analyst_output = config.get('analyst_output', 16)
    collective_input_dim = config.get('collective_input_dim', 96)  # n_analysts * analyst_output
    num_classes = config.get('num_classes', 10)
    collective_version = config.get('collective_version', 'simple_mlp')
    
    # Layer positions (x, y, width, height) - Better organized layout
    layers = {
        'input': (1, 6, 1.8, 1.0),
        'experts': (4, 7.5, 2.0, 2.0),
        'encoder': (4, 5, 2.0, 1.0),
        'analysts': (8, 7.5, 2.0, 2.0),
        'collective': (8, 4, 2.0, 1.0),
        'output': (11, 6, 1.8, 1.0),
    }
    
    # ===== Draw Layers =====
    
    # 1. Input Layer
    x, y, w, h = layers['input']
    input_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                facecolor=colors['input'], edgecolor='black', linewidth=2.5)
    ax.add_patch(input_box)
    ax.text(x + w/2, y + h/2, f'Input\n{input_dim}D', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # 2. Expert Layer
    x, y, w, h = layers['experts']
    expert_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                facecolor=colors['expert'], edgecolor='black', linewidth=2.5)
    ax.add_patch(expert_box)
    ax.text(x + w/2, y + h - 0.15, f'Expert Layer\n({n_experts} Experts)', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Individual expert boxes
    expert_spacing = (h - 0.5) / (n_experts + 1)
    for i in range(n_experts):
        ey = y + 0.3 + expert_spacing * (i + 1)
        small_box = FancyBboxPatch((x + 0.15, ey - 0.12), w - 0.3, 0.2, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='white', edgecolor='black', linewidth=1.5)
        ax.add_patch(small_box)
        ax.text(x + w/2, ey, f'E{i+1}: {expert_output}D', 
                ha='center', va='center', fontsize=9)
    
    # Concat label for experts
    ax.text(x + w/2, y - 0.3, f'Concat → {expert_concat_dim}D', 
            ha='center', fontsize=9, style='italic', color='#666')
    
    # 3. Encoder Layer
    x, y, w, h = layers['encoder']
    encoder_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                 facecolor=colors['encoder'], edgecolor='black', linewidth=2.5)
    ax.add_patch(encoder_box)
    ax.text(x + w/2, y + h/2, 
            f'Encoder\n{expert_concat_dim}D → {expert_encoder_output_dim}D', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 4. Analyst Layer
    x, y, w, h = layers['analysts']
    analyst_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                 facecolor=colors['analyst'], edgecolor='black', linewidth=2.5)
    ax.add_patch(analyst_box)
    ax.text(x + w/2, y + h - 0.15, f'Analyst Layer\n({n_analysts} Analysts)', 
            ha='center', va='top', fontsize=11, fontweight='bold')
    
    # Show analyst input dimension
    ax.text(x + w/2, y + h + 0.15, f'Input: {analyst_input_dim}D\n(Input + Encoded)', 
            ha='center', fontsize=9, style='italic', color='#666')
    
    # Individual analyst boxes
    analyst_spacing = (h - 0.5) / (n_analysts + 1)
    for i in range(n_analysts):
        ay = y + 0.3 + analyst_spacing * (i + 1)
        small_box = FancyBboxPatch((x + 0.15, ay - 0.12), w - 0.3, 0.2, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='white', edgecolor='black', linewidth=1.5)
        ax.add_patch(small_box)
        ax.text(x + w/2, ay, f'A{i+1}: {analyst_output}D', 
                ha='center', va='center', fontsize=9)
    
    # Concat label for analysts
    ax.text(x + w/2, y - 0.3, f'Concat → {collective_input_dim}D', 
            ha='center', fontsize=9, style='italic', color='#666')
    
    # 5. Collective Layer
    x, y, w, h = layers['collective']
    collective_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                   facecolor=colors['collective'], edgecolor='black', linewidth=2.5)
    ax.add_patch(collective_box)
    collective_label = f'Collective\n({collective_version})\n{collective_input_dim}D → {num_classes}D'
    ax.text(x + w/2, y + h/2, collective_label, 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 6. Output Layer
    x, y, w, h = layers['output']
    output_box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                facecolor=colors['output'], edgecolor='black', linewidth=2.5)
    ax.add_patch(output_box)
    ax.text(x + w/2, y + h/2, f'Output\n{num_classes} Classes', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # ===== Draw Arrows (Data Flow) =====
    arrow_props = dict(arrowstyle='->', lw=2.5, color=colors['arrow'])
    skip_arrow_props = dict(arrowstyle='->', lw=2.5, color=colors['skip'], linestyle='--', alpha=0.8)
    
    # Input → Experts (parallel)
    input_x_center = layers['input'][0] + layers['input'][2] / 2
    input_y = layers['input'][1] + layers['input'][3]
    experts_x = layers['experts'][0]
    experts_y_center = layers['experts'][1] + layers['experts'][3] / 2
    ax.annotate('', xy=(experts_x, experts_y_center), xytext=(input_x_center, input_y), 
                arrowprops=arrow_props)
    
    # Experts → Encoder (via concat)
    experts_x_right = layers['experts'][0] + layers['experts'][2]
    experts_y_bottom = layers['experts'][1]
    encoder_x = layers['encoder'][0]
    encoder_y_center = layers['encoder'][1] + layers['encoder'][3] / 2
    ax.annotate('', xy=(encoder_x, encoder_y_center), xytext=(experts_x_right, experts_y_bottom - 0.2), 
                arrowprops=arrow_props)
    
    # Encoder → Analysts (to concatenate with input)
    encoder_x_right = layers['encoder'][0] + layers['encoder'][2]
    encoder_y = layers['encoder'][1] + layers['encoder'][3]
    analysts_x = layers['analysts'][0]
    analysts_y_bottom = layers['analysts'][1]
    ax.annotate('', xy=(analysts_x, analysts_y_bottom - 0.2), xytext=(encoder_x_right, encoder_y), 
                arrowprops=arrow_props)
    
    # Skip Connection: Input → Analysts (for concatenation)
    input_y_bottom = layers['input'][1]
    skip_arrow = FancyArrowPatch((input_x_center, input_y_bottom), (analysts_x, analysts_y_bottom - 0.2), 
                                connectionstyle="arc3,rad=0.3", **skip_arrow_props)
    ax.add_patch(skip_arrow)
    ax.text((input_x_center + analysts_x) / 2, input_y_bottom - 0.5, 'Skip Connection', 
            ha='center', fontsize=9, style='italic', color=colors['skip'], fontweight='bold')
    
    # Analysts → Collective (via concat)
    analysts_x_right = layers['analysts'][0] + layers['analysts'][2]
    analysts_y_bottom = layers['analysts'][1]
    collective_x = layers['collective'][0]
    collective_y_top = layers['collective'][1] + layers['collective'][3]
    ax.annotate('', xy=(collective_x, collective_y_top), xytext=(analysts_x_right, analysts_y_bottom - 0.2), 
                arrowprops=arrow_props)
    
    # Collective → Output
    collective_x_right = layers['collective'][0] + layers['collective'][2]
    collective_y_center = layers['collective'][1] + layers['collective'][3] / 2
    output_x = layers['output'][0]
    output_y_center = layers['output'][1] + layers['output'][3] / 2
    ax.annotate('', xy=(output_x, output_y_center), xytext=(collective_x_right, collective_y_center), 
                arrowprops=arrow_props)
    
    # ===== Title and Legend =====
    total_params = sum(p.numel() for p in model.parameters())
    title = f'Collective Model Architecture\n{n_experts} Experts + {n_analysts} Analysts | {total_params:,} Parameters'
    ax.text(7, 11.2, title, ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['input'], label='Input'),
        mpatches.Patch(facecolor=colors['expert'], label='Expert Layer'),
        mpatches.Patch(facecolor=colors['encoder'], label='Encoder'),
        mpatches.Patch(facecolor=colors['analyst'], label='Analyst Layer'),
        mpatches.Patch(facecolor=colors['collective'], label='Collective'),
        mpatches.Patch(facecolor=colors['output'], label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Model architecture diagram saved to: {output_path}")
    return str(output_path)


def plot_model_graph(model, input_shape, output_path='model_graph.png', format='png', show_params=False):
    """
    Plot and save the model computation graph using torchviz.
    
    Shows the actual computation graph with operations. Good for debugging.
    Note: This shows the detailed computation graph, not the architecture diagram.
    Use plot_model_architecture() for a clean architecture diagram.
    
    Args:
        model: PyTorch model
        input_shape: Tuple of input shape (e.g., (1, 784) for batch_size=1)
        output_path: Path to save the graph image
        format: Image format ('png', 'pdf', 'svg')
        show_params: Whether to show parameter names (can be very large)
    
    Returns:
        str: Path to saved graph image
    
    Example:
        >>> from collective_model.training import CollectiveModel
        >>> from collective_model.config import CONFIG_DEBUG, prepare_config
        >>> config = prepare_config(CONFIG_DEBUG)
        >>> model = CollectiveModel(config)
        >>> plot_model_graph(model, input_shape=(1, 784), output_path='model_graph.png')
    """
    try:
        from torchviz import make_dot
    except ImportError:
        raise ImportError(
            "torchviz not installed. Install with: pip install torchviz graphviz\n"
            "Also install graphviz system package:\n"
            "  Ubuntu/Debian: sudo apt-get install graphviz\n"
            "  Mac: brew install graphviz\n"
            "  Windows: Download from https://graphviz.org/download/"
        )
    
    # Set model to eval mode (required for batch_size=1 due to BatchNorm)
    model.eval()
    
    # Create dummy input that requires gradients (for better graph tracing)
    # Use batch_size=2 to avoid BatchNorm issues
    if len(input_shape) == 2:
        batch_size = 2
        dummy_shape = (batch_size, input_shape[1])
    else:
        dummy_shape = input_shape
        batch_size = dummy_shape[0]
    
    dummy_input = torch.randn(*dummy_shape, requires_grad=True)
    
    # Forward pass to build computation graph
    try:
        output = model(dummy_input)
    except Exception as e:
        # If model requires return_intermediates, try that
        try:
            output, _, _ = model(dummy_input, return_intermediates=True)
        except Exception:
            raise RuntimeError(f"Failed to run model forward pass: {e}")
    
    # Create visualization
    # For cleaner graph, show only operations, not all parameters
    if show_params:
        # Full graph with parameters (can be very large)
        dot = make_dot(output, params=dict(list(model.named_parameters())), 
                      show_attrs=False, show_saved=False)
    else:
        # Simplified graph showing only operations
        dot = make_dot(output, show_attrs=True, show_saved=False)
    
    # Save original dot source for modification
    dot_source = str(dot.source)
    
    # Color mappings
    expert_color = '#FFE5B4'  # Light orange
    analyst_color = '#E0E0FF'  # Light purple/blue
    encoder_color = '#FFCCCB'  # Light red
    collective_color = '#D4EDDA'  # Light green
    input_color = '#E8F4F8'  # Light blue
    output_color = '#F0E68C'  # Light yellow
    
    # Parse dot source and add colors/clusters
    import re
    lines = dot_source.split('\n')
    modified_lines = []
    in_graph = False
    
    # Create clusters for organization
    expert_nodes = []
    analyst_nodes = []
    encoder_nodes = []
    collective_nodes = []
    input_nodes = []
    output_nodes = []
    all_edges = []
    
    # Parse nodes and categorize them, preserve edges
    for line in lines:
        original_line = line
        line = line.strip()
        
        # Skip empty lines
        if not line:
            modified_lines.append(original_line)
            continue
        
        # Detect graph start
        if line.startswith('digraph'):
            modified_lines.append('digraph {')
            modified_lines.append('    rankdir=LR;')
            modified_lines.append('    size="18,12";')
            modified_lines.append('    nodesep=0.4;')
            modified_lines.append('    ranksep=0.6;')
            modified_lines.append('    splines=ortho;')
            modified_lines.append('    node [shape=box, style="rounded,filled", fontsize=9, fillcolor="#F5F5F5", color="#424242"];')
            modified_lines.append('')
            in_graph = True
            continue
        
        if line == '{' or line == '}':
            continue
        
        # Find edge definitions (preserve them)
        if '->' in line:
            all_edges.append(original_line)
            continue
        
        # Find node definitions
        if '[' in line and 'label=' in line:
            # Extract node identifier (before [)
            node_match = re.match(r'"?([^"\[\s]+)"?\s*\[', line)
            if node_match:
                node_id = node_match.group(1)
                
                # Get label to determine category
                label_match = re.search(r'label="([^"]+)"', line)
                if label_match:
                    label = label_match.group(1).lower()
                    
                    # Categorize and color
                    if any(kw in label for kw in ['expert', 'experts.0', 'experts.1']):
                        expert_nodes.append(node_id)
                        if 'fillcolor=' not in line:
                            line = line.replace('[', f'[fillcolor="{expert_color}", color="#D97706", ')
                    
                    elif any(kw in label for kw in ['analyst', 'analysts.0', 'analysts.1', 'analysts.2', 'analysts.3', 'analysts.4', 'analysts.5']):
                        analyst_nodes.append(node_id)
                        if 'fillcolor=' not in line:
                            line = line.replace('[', f'[fillcolor="{analyst_color}", color="#6366F1", ')
                    
                    elif any(kw in label for kw in ['encoder', 'expert_encoder']):
                        encoder_nodes.append(node_id)
                        if 'fillcolor=' not in line:
                            line = line.replace('[', f'[fillcolor="{encoder_color}", color="#DC2626", ')
                    
                    elif any(kw in label for kw in ['collective', 'simplecollective', 'encoderheadcollective']):
                        collective_nodes.append(node_id)
                        if 'fillcolor=' not in line:
                            line = line.replace('[', f'[fillcolor="{collective_color}", color="#059669", ')
                    
                    elif 'input' in label and 'backward' not in label:
                        input_nodes.append(node_id)
                        if 'fillcolor=' not in line:
                            line = line.replace('[', f'[fillcolor="{input_color}", color="#0288D1", ')
                    
                    elif ('output' in label or 'result' in label) and 'backward' not in label:
                        output_nodes.append(node_id)
                        if 'fillcolor=' not in line:
                            line = line.replace('[', f'[fillcolor="{output_color}", color="#F59E0B", ')
        
        modified_lines.append('    ' + line)
    
    # Add edges back
    for edge in all_edges:
        modified_lines.append('    ' + edge.strip())
    
    # Add clusters (subgraphs) for better organization
    modified_lines.append('')
    if expert_nodes:
        modified_lines.append('    subgraph cluster_experts {')
        modified_lines.append('        label="Expert Layer";')
        modified_lines.append('        style=filled;')
        modified_lines.append(f'        fillcolor="{expert_color}80";')  # Semi-transparent
        modified_lines.append('        color="#D97706";')
        for node in expert_nodes:
            modified_lines.append(f'        "{node}";')
        modified_lines.append('    }')
    
    if analyst_nodes:
        modified_lines.append('')
        modified_lines.append('    subgraph cluster_analysts {')
        modified_lines.append('        label="Analyst Layer";')
        modified_lines.append('        style=filled;')
        modified_lines.append(f'        fillcolor="{analyst_color}80";')
        modified_lines.append('        color="#6366F1";')
        for node in analyst_nodes:
            modified_lines.append(f'        "{node}";')
        modified_lines.append('    }')
    
    if encoder_nodes:
        modified_lines.append('')
        modified_lines.append('    subgraph cluster_encoder {')
        modified_lines.append('        label="Encoder";')
        modified_lines.append('        style=filled;')
        modified_lines.append(f'        fillcolor="{encoder_color}80";')
        modified_lines.append('        color="#DC2626";')
        for node in encoder_nodes:
            modified_lines.append(f'        "{node}";')
        modified_lines.append('    }')
    
    if collective_nodes:
        modified_lines.append('')
        modified_lines.append('    subgraph cluster_collective {')
        modified_lines.append('        label="Collective";')
        modified_lines.append('        style=filled;')
        modified_lines.append(f'        fillcolor="{collective_color}80";')
        modified_lines.append('        color="#059669";')
        for node in collective_nodes:
            modified_lines.append(f'        "{node}";')
        modified_lines.append('    }')
    
    modified_lines.append('}')
    
    # Recreate dot from modified source
    try:
        from graphviz import Source
        modified_dot = Source('\n'.join(modified_lines))
        dot = modified_dot
    except Exception as e:
        # If modification fails, use original with basic styling
        print(f"⚠ Could not apply color coding: {e}")
        print("  Using original graph with basic styling...")
        dot.attr('graph', rankdir='LR', size='18,12', nodesep='0.4', ranksep='0.6', 
                splines='ortho', concentrate='false')
        dot.attr('node', shape='box', style='rounded,filled', fontsize='9', 
                fillcolor='#F5F5F5', color='#424242')
    
    # Save the graph
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dot.render(output_path.stem, format=format, cleanup=True, directory=output_path.parent)
    
    full_path = output_path.parent / f"{output_path.stem}.{format}"
    
    print(f"✓ Computation graph saved to: {full_path}")
    print(f"  Note: This shows the computation graph (operations), not the architecture diagram.")
    print(f"  Use plot_model_architecture() for a clean architecture visualization.")
    
    return str(full_path)


def print_model_summary_text(model, input_shape):
    """
    Print a text summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Tuple of input shape
    
    Example:
        >>> print_model_summary_text(model, (1, 784))
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    
    # Model class name
    print(f"\nModel Type: {model.__class__.__name__}")
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total:      {total_params:,}")
    print(f"  Trainable:  {trainable_params:,}")
    print(f"  Frozen:     {total_params - trainable_params:,}")
    
    # Model structure
    print(f"\nModel Structure:")
    print(model)
    
    # Test forward pass
    print(f"\nForward Pass Test:")
    model.eval()
    dummy_input = torch.randn(*input_shape)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ✓ Forward pass successful")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
    
    # Precision info
    print(f"\nPrecision:")
    sample_param = next(model.parameters())
    dtype = sample_param.dtype
    print(f"  Data type: {dtype}")
    if dtype == torch.float32:
        print(f"  Mode: FP32 (Full Precision)")
        print(f"  Note: For FP16 (half precision), enable mixed precision training")
    elif dtype == torch.float16:
        print(f"  Mode: FP16 (Half Precision)")
    else:
        print(f"  Mode: {dtype}")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    # Test visualization
    print("Testing model visualization...")
    
    from collective_model.training import CollectiveModel
    from collective_model.config import CONFIG_DEBUG, prepare_config
    
    # Prepare config and create model
    config = prepare_config(CONFIG_DEBUG)
    model = CollectiveModel(config)
    
    # Print text summary
    print_model_summary_text(model, input_shape=(1, 784))
    
    # Generate architecture diagram
    try:
        arch_path = plot_model_architecture(
            model=model,
            config=config,
            output_path='collective_model_architecture.png'
        )
        print(f"\n✓ Architecture diagram complete!")
        print(f"  Diagram saved to: {arch_path}")
    except Exception as e:
        print(f"\n⚠ Architecture diagram failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Also try the old torchviz graph (for comparison)
    try:
        graph_path = plot_model_graph(
            model=model,
            input_shape=(1, 784),
            output_path='collective_model_graph_torchviz.png'
        )
        print(f"\n✓ Torchviz graph saved to: {graph_path}")
    except ImportError as e:
        print(f"\n⚠ Torchviz graph skipped: {e}")
    except Exception as e:
        print(f"\n⚠ Torchviz graph failed: {e}")

