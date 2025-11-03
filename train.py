"""
Main training script for Collective Model and Baselines.

Supports:
- Collective Model (Strategy C - end-to-end)
- 6 Baseline architectures (shallow, balanced, deep, very_deep, deep_resnet, very_deep_resnet)

Usage:
    # Train collective model with debug config
    python train.py --config debug --model collective
    
    # Train shallow baseline with phase1 config
    python train.py --config phase1 --model shallow
    
    # Train all baselines (run separately)
    python train.py --config phase1 --model balanced
    python train.py --config phase1 --model deep
    python train.py --config phase1 --model very_deep
    python train.py --config phase1 --model deep_resnet
    python train.py --config phase1 --model very_deep_resnet
"""

import argparse
import torch
import wandb
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from collective_model.config import CONFIG_DEBUG, CONFIG_PHASE1, prepare_config
from collective_model.data.loaders import get_data_loaders
from collective_model.training import CollectiveModel, train_strategy_c
from collective_model.utils.param_counter import count_parameters
from baselines.monolithic import MonolithicMLP, ResNetMLP
from baselines.find_baselines import find_all_baselines, find_all_baselines_with_target


def train_baseline(config, model, train_loader, val_loader, test_loader, device):
    """
    Train a baseline model (monolithic MLP or ResNet).
    
    Uses the same training loop structure as collective model but simpler
    (no diversity loss, no intermediate outputs).
    """
    from collective_model.training.trainer import train_epoch, validate
    from collective_model.utils.metrics import compute_accuracy
    import torch.nn.functional as F
    
    # Setup optimizer
    optimizer_name = config.get('optimizer', 'adam').lower()
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Import metrics utilities for consistent calculation
    from collective_model.utils.metrics import AverageMeter, compute_accuracy
    
    # Training loop
    best_val_acc = 0.0
    epochs = config['epochs']
    topk = config.get('topk', (1,))  # Top-k accuracy for consistent metrics
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_meter = AverageMeter('Train Loss', ':.4f')
        train_acc_meter = AverageMeter('Train Acc', ':.2f')
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            # Update metrics using AverageMeter
            batch_size = data.size(0)
            train_loss_meter.update(loss.item(), batch_size)
            accuracies = compute_accuracy(output, target, topk=topk)
            train_acc_meter.update(accuracies[0], batch_size)  # Top-1 accuracy
        
        # Validate (using AverageMeter for consistency)
        model.eval()
        val_loss_meter = AverageMeter('Val Loss', ':.4f')
        val_acc_meter = AverageMeter('Val Acc', ':.2f')
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # Update metrics using AverageMeter
                batch_size = data.size(0)
                val_loss_meter.update(loss.item(), batch_size)
                accuracies = compute_accuracy(output, target, topk=topk)
                val_acc_meter.update(accuracies[0], batch_size)  # Top-1 accuracy
        
        # Test (using AverageMeter for consistency)
        test_loss_meter = AverageMeter('Test Loss', ':.4f')
        test_acc_meter = AverageMeter('Test Acc', ':.2f')
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # Update metrics using AverageMeter
                batch_size = data.size(0)
                test_loss_meter.update(loss.item(), batch_size)
                accuracies = compute_accuracy(output, target, topk=topk)
                test_acc_meter.update(accuracies[0], batch_size)  # Top-1 accuracy
        
        # Extract final metrics (clear variable names, no duplicates)
        train_loss_final = train_loss_meter.avg
        train_acc_final = train_acc_meter.avg
        val_loss_final = val_loss_meter.avg
        val_acc_final = val_acc_meter.avg
        test_loss_final = test_loss_meter.avg
        test_acc_final = test_acc_meter.avg
        
        # Log to wandb (same metrics as collective for consistency)
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss_final,
            'train/loss_prediction': train_loss_final,  # For baselines, total loss = prediction loss
            'train/accuracy': train_acc_final,
            'val/loss': val_loss_final,
            'val/accuracy': val_acc_final,
            'test/loss': test_loss_final,
            'test/accuracy': test_acc_final,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Train Loss: {train_loss_final:.4f}, Train Acc: {train_acc_final:.2f}%, '
              f'Val Loss: {val_loss_final:.4f}, Val Acc: {val_acc_final:.2f}%, '
              f'Test Loss: {test_loss_final:.4f}, Test Acc: {test_acc_final:.2f}%')
        
        # Save best model
        if val_acc_final > best_val_acc:
            best_val_acc = val_acc_final
    
    print(f'\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%')
    
    # Calculate parameter efficiency metrics (using final test metrics)
    import math
    model_params = count_parameters(model)
    param_efficiency = (test_acc_final / model_params) * 100000  # Accuracy per 100K parameters
    accuracy_per_million = (test_acc_final / model_params) * 1000000  # Accuracy per 1M params
    accuracy_per_log10_params = test_acc_final / math.log10(model_params + 1)  # Log-scaled (better for large ranges)
    
    # Log final metrics including parameter efficiency (use final test metrics)
    wandb.log({
        'final/best_val_accuracy': best_val_acc,
        'final/test_accuracy': test_acc_final,
        'final/test_loss': test_loss_final,
        'final/model_params': model_params,
        'final/param_efficiency': param_efficiency,  # Acc per 100K params
        'final/accuracy_per_million_params': accuracy_per_million,  # Acc per 1M params
        'final/accuracy_per_log10_params': accuracy_per_log10_params,
        'param_efficiency/test_accuracy_per_log10_params': accuracy_per_log10_params,  # For sweep optimization
        'final/param_efficiency/test_accuracy_per_log10_params': accuracy_per_log10_params  # For sweep optimization
    })
    
    # Log final summary metrics to wandb summary
    wandb.summary.update({
        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': test_acc_final,
        'final_test_loss': test_loss_final,
        'model_params': model_params,
        'param_efficiency': param_efficiency,
        'accuracy_per_million_params': accuracy_per_million,
        'accuracy_per_log10_params': accuracy_per_log10_params,
        'param_efficiency/test_accuracy_per_log10_params': accuracy_per_log10_params
    })
    
    print(f'Parameter Efficiency: {param_efficiency:.2f} accuracy per 100K parameters')
    
    return {
        'best_val_acc': best_val_acc,
        'final_test_acc': test_acc,
        'model_params': model_params,
        'param_efficiency': param_efficiency
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train Collective Model or Baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='debug',
        choices=['debug', 'phase1'],
        help='Configuration preset (ignored in wandb sweep mode)'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='collective',
        choices=['collective', 'shallow', 'balanced', 'deep', 'very_deep', 
                 'deep_resnet', 'very_deep_resnet'],
        help='Model type to train (ignored in wandb sweep mode)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu). Auto-detects if not specified.'
    )
    
    # Parse known args first to detect if there are extra args (sweep params)
    known_args, unknown_args = parser.parse_known_args()
    args = known_args
    sweep_mode = len(unknown_args) > 0
    
    # Initialize wandb (will detect sweep mode automatically if called by wandb agent)
    if sweep_mode:
        # Wandb agent passes sweep params as command-line args
        # Initialize wandb to read them from environment/config
        wandb.init()
        
        # Check if this is actually a sweep run
        # wandb.config is a Config object, check if it has items
        if wandb.run is None:
            # Not a real sweep run, might be test mode
            sweep_mode = False
            wandb.finish()
            wandb.init(mode='disabled')  # Disable wandb for test
        else:
            # Check if config has sweep parameters (more than just default keys)
            config_dict = dict(wandb.config)
            if len(config_dict) < 5:
                # Not enough params, might not be a sweep
                sweep_mode = False
    
    if sweep_mode and wandb.run is not None:
        # Sweep mode: read all parameters from wandb.config
        print("="*60)
        print("WANDB SWEEP MODE")
        print("="*60)
        sweep_config = dict(wandb.config)
        
        # Start with base config (use phase1 as default for sweeps)
        config = CONFIG_PHASE1.copy()
        
        # Override with sweep parameters
        for key, value in sweep_config.items():
            # Handle string conversions for boolean-like values
            if isinstance(value, str):
                if value.lower() == 'true':
                    config[key] = True
                elif value.lower() == 'false':
                    config[key] = False
                elif key in ['expert_hidden', 'analyst_hidden']:
                    # Parse list strings like "[512, 256]"
                    import ast
                    try:
                        config[key] = ast.literal_eval(value)
                    except:
                        config[key] = value
                else:
                    config[key] = value
            else:
                config[key] = value
        
        # Extract model_type from sweep config
        model_type = sweep_config.get('model_type', 'collective')
        
        print(f"Model type from sweep: {model_type}")
        print(f"Key sweep parameters:")
        print(f"  batch_size: {config.get('batch_size')}")
        print(f"  learning_rate: {config.get('learning_rate')}")
        print(f"  optimizer: {config.get('optimizer')}")
        if model_type == 'collective':
            print(f"  n_total: {config.get('n_total')}")
            print(f"  expert_ratio: {config.get('expert_ratio')}")
        print("="*60)
    else:
        # Normal mode: use command-line arguments
        if args.config == 'debug':
            config = CONFIG_DEBUG.copy()
        else:
            config = CONFIG_PHASE1.copy()
        
        model_type = args.model
    
    # Prepare config (computes n_experts, n_analysts, etc.)
    config = prepare_config(config)
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Get data loaders first (needed for both collective and baselines)
    dataset_name = config.get('dataset', 'mnist')
    print(f"\nLoading data: {dataset_name.upper()}...")
    loaders, dataset_info = get_data_loaders(
        dataset_name=dataset_name,
        batch_size=config['batch_size'],
        eval_batch_size=config.get('eval_batch_size', None),  # Use larger batch for val/test (smoother metrics)
        val_split=config.get('val_split', 0.1),
        num_workers=config.get('num_workers', 2),
        use_augmentation=config.get('use_augmentation', False)
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # Update config with dataset info
    config['input_dim'] = dataset_info['input_dim']
    config['num_classes'] = dataset_info['num_classes']
    config['device'] = device
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Train based on model type
    if model_type == 'collective':
        # Collective model: dispatch to selected training strategy (default C)
        from collective_model.training import get_strategy_runner
        strategy = str(config.get('training_strategy', 'C')).upper()
        print(f"\nTraining Collective Model (strategy {strategy})...")
        if not sweep_mode:
            config['wandb_project'] = config.get('wandb_project', 'collective-architecture')
            config['wandb_name'] = f"{args.config}-{args.model}-strategy-{strategy}"
        run_strategy = get_strategy_runner(strategy)
        run_strategy(
            config=config,
            model=None,  # Strategy C builds the model internally; A/B may use this later
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=str(device)
        )
    else:
        # Baseline: we handle everything
        print(f"\nCreating {model_type} baseline model...")
        
        # Determine target parameter count
        if sweep_mode and 'target_params' in config:
            # Sweep mode: use target_params from sweep config
            target_params = int(config['target_params'])
            print(f"  Target parameter count from sweep: {target_params:,}")
            
            # Build a minimal config for baseline finding
            baseline_search_config = {
                'input_dim': config['input_dim'],
                'num_classes': config['num_classes'],
                'target_params': target_params  # Use this directly
            }
            
            # Find baseline matching this target
            print(f"  Finding {model_type} baseline matching {target_params:,} parameters...")
            baselines = find_all_baselines_with_target(baseline_search_config, target_params)
        else:
            # Normal mode or no target_params: use collective model to determine target
            print("  Finding matching baseline architectures (using collective reference)...")
            baselines = find_all_baselines(config)
        
        if model_type not in baselines:
            raise ValueError(f"Unknown baseline type: {model_type}")
        
        baseline_config = baselines[model_type]
        hidden_dims = baseline_config['hidden_dims']
        use_skip = baseline_config.get('use_skip_connections', False)
        
        if use_skip:
            model = ResNetMLP(
                input_dim=config['input_dim'],
                hidden_dims=hidden_dims,
                num_classes=config['num_classes'],
                dropout=config.get('dropout', 0.1),
                use_batchnorm=True
            )
            model_type_name = f"ResNet Baseline ({model_type})"
        else:
            model = MonolithicMLP(
                input_dim=config['input_dim'],
                hidden_dims=hidden_dims,
                num_classes=config['num_classes'],
                dropout=config.get('dropout', 0.1),
                use_batchnorm=True
            )
            model_type_name = f"MLP Baseline ({model_type})"
        
        model = model.to(device)
        total_params = count_parameters(model)
        
        print(f"  Architecture: {hidden_dims}")
        print(f"  Model params: {total_params:,} (target: {baseline_config['params']:,})")
        
        # Initialize wandb for baseline (only if not in sweep mode)
        if not sweep_mode:
            wandb.init(
                project="collective-architecture",
                name=f"{args.config}-{model_type}",
                config={
                    **config,
                    'model_type': model_type,
                    'model_params': total_params,
                    'baseline_hidden_dims': hidden_dims,
                    'baseline_target_params': baseline_config['params'],
                    'use_skip_connections': use_skip
                },
                tags=[args.config, model_type, config.get('dataset', 'mnist'), "baseline"]
            )
        else:
            # In sweep mode, wandb is already initialized, just update config
            wandb.config.update({
                'model_type': model_type,
                'model_params': total_params,
                'baseline_hidden_dims': hidden_dims,
                'baseline_target_params': baseline_config['params'],
                'use_skip_connections': use_skip
            })
        
        # Train baseline
        print(f"\nStarting training for {config['epochs']} epochs...")
        print("=" * 60)
        train_baseline(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device
        )
    
    print("=" * 60)
    print("Training complete!")
    
    # wandb.finish() is called by train_strategy_c for collective, or by train_baseline if needed
    if model_type != 'collective' and not sweep_mode:
        wandb.finish()


if __name__ == '__main__':
    main()

