"""
Utility functions for the Collective Model.

Includes parameter counting, metrics tracking, and visualization.
"""

from .param_counter import (
    count_parameters,
    count_mlp_parameters,
    count_resnet_mlp_parameters,
    get_model_summary,
    print_model_summary,
    compare_model_sizes
)

from .metrics import (
    AverageMeter,
    MetricTracker,
    Timer,
    compute_accuracy,
    format_time
)

from .visualization import (
    plot_model_graph,
    plot_model_architecture,
    print_model_summary_text
)

__all__ = [
    # Parameter counting
    'count_parameters',
    'count_mlp_parameters',
    'count_resnet_mlp_parameters',
    'get_model_summary',
    'print_model_summary',
    'compare_model_sizes',
    
    # Metrics tracking
    'AverageMeter',
    'MetricTracker',
    'Timer',
    'compute_accuracy',
    'format_time',
    
    # Visualization
    'plot_model_graph',
    'plot_model_architecture',
    'print_model_summary_text',
]

