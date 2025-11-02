"""
Training utilities for the Collective Model.

Includes training and validation loops with wandb integration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from .collective_model import CollectiveModel
from .losses import combined_loss
from ..utils.metrics import AverageMeter, compute_accuracy


def train_epoch(model, train_loader, optimizer, config, device):
    """
    Train for one epoch.
    
    Args:
        model: CollectiveModel instance
        train_loader: DataLoader for training data
        optimizer: Optimizer
        config: Configuration dictionary
        device: torch.device
    
    Returns:
        dict: Training metrics
    """
    model.train()
    
    # Metrics
    losses = {
        'prediction': AverageMeter('PredLoss', ':.4f'),
        'diversity_expert': AverageMeter('DivExpert', ':.4f'),
        'diversity_analyst': AverageMeter('DivAnalyst', ':.4f'),
        'total': AverageMeter('TotalLoss', ':.4f')
    }
    acc_meter = AverageMeter('Acc', ':.2f')
    
    # Gradient accumulation setup
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    effective_batch_size = config['batch_size'] * gradient_accumulation_steps
    
    # Mixed precision (FP16) setup
    use_mixed_precision = config.get('use_mixed_precision', False)
    scaler = None
    if use_mixed_precision:
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            autocast_context = autocast
        except ImportError:
            print("⚠ Mixed precision not available (requires PyTorch 1.6+)")
            use_mixed_precision = False
    
    pbar = tqdm(train_loader, desc='Train', leave=False)
    
    optimizer.zero_grad()  # Zero gradients at start of epoch
    
    num_batches = len(train_loader)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass (with mixed precision if enabled)
        # Check which diversity losses to compute (can be independent)
        # Backward compatibility: if old 'use_diversity_loss' flag exists, use it for both
        if 'use_diversity_loss' in config and 'use_expert_diversity' not in config:
            use_expert_diversity = config.get('use_diversity_loss', False)
        else:
            use_expert_diversity = config.get('use_expert_diversity', False)
        
        if 'use_diversity_loss' in config and 'use_analyst_diversity' not in config:
            use_analyst_diversity = config.get('use_diversity_loss', False)
        else:
            use_analyst_diversity = config.get('use_analyst_diversity', False)
        return_intermediates = use_expert_diversity or use_analyst_diversity
        
        # Mixed precision forward pass
        if use_mixed_precision:
            with autocast_context():
                if return_intermediates:
                    logits, expert_outputs, analyst_outputs = model(data, return_intermediates=True)
                else:
                    logits = model(data)
                    expert_outputs = None
                    analyst_outputs = None
        else:
            if return_intermediates:
                logits, expert_outputs, analyst_outputs = model(data, return_intermediates=True)
            else:
                logits = model(data)
                expert_outputs = None
                analyst_outputs = None
        
        # Compute loss
        loss_dict = combined_loss(
            prediction_logits=logits,
            targets=target,
            expert_outputs=expert_outputs if use_expert_diversity else None,
            analyst_outputs=analyst_outputs if use_analyst_diversity else None,
            diversity_lambda=config.get('diversity_lambda', 0.01),
            diversity_temperature=config.get('diversity_temperature', 1.0),
            use_expert_diversity=use_expert_diversity,
            use_analyst_diversity=use_analyst_diversity
        )
        
        # Scale loss for gradient accumulation
        scaled_loss = loss_dict['total'] / gradient_accumulation_steps
        
        # Backward pass (with mixed precision if enabled)
        if use_mixed_precision:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Update optimizer every N steps (gradient accumulation)
        # NOTE: Loop processes ALL batches, but optimizer.step() only happens
        # every gradient_accumulation_steps batches. Gradients accumulate
        # in memory between steps, giving us effective_batch_size = batch_size × steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        # Handle remaining gradients if epoch ends before completing accumulation
        elif batch_idx == num_batches - 1 and (batch_idx + 1) % gradient_accumulation_steps != 0:
            # Last batch and we have accumulated gradients that weren't stepped
            if use_mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Update metrics (use actual batch size for logging, effective batch for training)
        batch_size = data.size(0)
        losses['prediction'].update(loss_dict['prediction'].item(), batch_size)
        losses['diversity_expert'].update(loss_dict['diversity_expert'].item(), batch_size)
        losses['diversity_analyst'].update(loss_dict['diversity_analyst'].item(), batch_size)
        # Log total loss (use unscaled loss_dict['total'] for consistency)
        losses['total'].update(loss_dict['total'].item(), batch_size)
        
        # Accuracy (configurable top-k)
        topk = config.get('topk', (1,))  # Default: top-1 for classification
        accuracies = compute_accuracy(logits, target, topk=topk)
        acc_meter.update(accuracies[0], batch_size)  # Use top-1 for metric
        
        # Update progress bar
        if batch_idx % config.get('log_interval', 50) == 0:
            pbar.set_postfix({
                'loss': f'{losses["total"].avg:.4f}',
                'acc': f'{acc_meter.avg:.2f}%'
            })
    
    return {
        'loss': losses['total'].avg,
        'loss_prediction': losses['prediction'].avg,
        'loss_diversity_expert': losses['diversity_expert'].avg,
        'loss_diversity_analyst': losses['diversity_analyst'].avg,
        'accuracy': acc_meter.avg
    }


def validate(model, val_loader, config, device):
    """
    Validate the model.
    
    Args:
        model: CollectiveModel instance
        val_loader: DataLoader for validation data
        config: Configuration dictionary
        device: torch.device
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.2f')
    
    # Mixed precision for validation (if enabled)
    use_mixed_precision = config.get('use_mixed_precision', False)
    if use_mixed_precision:
        try:
            from torch.cuda.amp import autocast
            autocast_context = autocast
        except ImportError:
            use_mixed_precision = False
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Val', leave=False):
            data, target = data.to(device), target.to(device)
            
            # Forward pass (with mixed precision if enabled)
            if use_mixed_precision:
                with autocast_context():
                    logits = model(data)
            else:
                logits = model(data)
            
            # Loss (no diversity in validation)
            loss = nn.functional.cross_entropy(logits, target)
            
            # Metrics
            loss_meter.update(loss.item(), data.size(0))
            topk = config.get('topk', (1,))  # Default: top-1 for classification
            accuracies = compute_accuracy(logits, target, topk=topk)
            acc_meter.update(accuracies[0], data.size(0))  # Use top-1 for metric
    
    return {
        'loss': loss_meter.avg,
        'accuracy': acc_meter.avg
    }


def train_strategy_c(config, train_loader, val_loader, test_loader=None):
    """
    Train Collective Model using Strategy C (end-to-end training).
    
    Args:
        config: Configuration dictionary (must be prepared)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Optional test DataLoader
    
    Returns:
        dict: Training history and best model info
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.get('seed', 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.get('seed', 42))
    
    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating CollectiveModel...")
    model = CollectiveModel(config).to(device)
    
    # Print optimization settings
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    use_mixed_precision = config.get('use_mixed_precision', False)
    effective_batch_size = config['batch_size'] * gradient_accumulation_steps
    
    if gradient_accumulation_steps > 1:
        print(f"✓ Gradient accumulation: {gradient_accumulation_steps} steps")
        print(f"  Effective batch size: {config['batch_size']} × {gradient_accumulation_steps} = {effective_batch_size}")
    
    if use_mixed_precision:
        print("✓ Mixed precision (FP16) enabled for 1.5-2x speedup")
    
    # Optional: Compile model (may help for large models, but adds overhead for small ones)
    # Benchmark showed it's slower for our model size, so disabled by default
    if config.get('use_torch_compile', False):
        try:
            # Try different modes - 'default' might work better than 'reduce-overhead'
            model = torch.compile(model, mode='default')
            print("⚠ Model compiled with torch.compile() - may be slower for small models!")
            print("   Benchmark showed overhead > benefit. Use only if you test and see improvement.")
        except AttributeError:
            print("⚠ torch.compile() not available (requires PyTorch 2.0+)")
        except Exception as e:
            print(f"⚠ torch.compile() failed: {e}, continuing without compilation")
    
    model_params_true = model.get_num_parameters()  # TRUE parameter count (not calculated)
    print(f"Model parameters: {model_params_true:,}")
    
    # Optimizer
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Must be 'adam', 'adamw', or 'sgd'")
    
    # Initialize wandb
    wandb.init(
        project=config.get('wandb_project', 'collective-model'),
        name=config.get('wandb_name', None),
        config=config,
        reinit=True
    )
    
    # Log comprehensive model architecture and training config
    # NOTE: Do NOT log model_params or target_params as hyperparameters!
    # These are outputs/metrics, not inputs. Only log model structure specs.
    wandb.config.update({
        # Model architecture (structure specifications only - NOT parameter counts!)
        'n_total': config.get('n_total'),
        'n_experts': config['n_experts'],
        'n_analysts': config['n_analysts'],
        'expert_ratio': config.get('expert_ratio'),
        'expert_hidden': config.get('expert_hidden'),
        'expert_output': config.get('expert_output'),
        'analyst_hidden': config.get('analyst_hidden'),
        'analyst_output': config.get('analyst_output'),
        'c_expert': config.get('c_expert'),
        'c_collective': config.get('c_collective'),
        'collective_version': config['collective_version'],
        'collective_hidden_scale': config.get('collective_hidden_scale'),
        # Diversity settings
        'use_expert_diversity': config.get('use_expert_diversity', False),
        'use_analyst_diversity': config.get('use_analyst_diversity', False),
        'diversity_lambda': config.get('diversity_lambda'),
        'diversity_temperature': config.get('diversity_temperature'),
        'vary_expert_architectures': config.get('vary_expert_architectures', False),
        'vary_analyst_architectures': config.get('vary_analyst_architectures', False),
        # Training settings
        'optimizer': config.get('optimizer'),
        'learning_rate': config.get('learning_rate'),
        'weight_decay': config.get('weight_decay'),
        'batch_size': config.get('batch_size'),
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        'use_mixed_precision': config.get('use_mixed_precision', False),
        'eval_batch_size': config.get('eval_batch_size', None),
        'topk': config.get('topk', (1,)),
        # Dataset info
        'dataset': config.get('dataset'),
        'input_dim': config.get('input_dim'),
        'num_classes': config.get('num_classes'),
    })
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Training loop
    epochs = config.get('epochs', 200)
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config, device)
        
        # Validate
        val_metrics = validate(model, val_loader, config, device)
        
        # Test (every epoch for fair comparison with baselines)
        test_metrics_epoch = None
        if test_loader is not None:
            test_metrics_epoch = validate(model, test_loader, config, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Check for best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
        
        # Log to wandb
        log_dict = {
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/loss_prediction': train_metrics['loss_prediction'],
            'train/loss_diversity_expert': train_metrics['loss_diversity_expert'],
            'train/loss_diversity_analyst': train_metrics['loss_diversity_analyst'],
            'train/accuracy': train_metrics['accuracy'],
            'val/loss': val_metrics['loss'],
            'val/accuracy': val_metrics['accuracy'],
            'lr': optimizer.param_groups[0]['lr']  # Log learning rate
        }
        
        # Add test metrics if available (accuracy AND loss every epoch!)
        if test_metrics_epoch is not None:
            log_dict['test/accuracy'] = test_metrics_epoch['accuracy']
            log_dict['test/loss'] = test_metrics_epoch['loss']  # Also log test loss every epoch!
        
        # Log parameter efficiency metrics EVERY EPOCH (for graphs in wandb)
        # Use TRUE parameter count from model (not calculated)
        # Better scaling: Use log scale to handle large parameter differences (1M vs 7M)
        import math
        
        # Accuracy efficiency (higher is better)
        param_efficiency_100k = (val_metrics['accuracy'] / model_params_true) * 100000  # Accuracy per 100K params
        param_efficiency_1m = (val_metrics['accuracy'] / model_params_true) * 1000000  # Accuracy per 1M params
        # Log-scaled efficiency (accounts for large param differences better)
        param_efficiency_log = val_metrics['accuracy'] / math.log10(model_params_true + 1)  # Accuracy per log10(params)
        
        # Loss efficiency (lower loss = better, so calculate loss per param directly)
        # FIXED: Lower is better, so we want to minimize (loss / params)
        loss_efficiency_100k = (val_metrics['loss'] / model_params_true) * 100000  # Loss per 100K params (lower is better)
        loss_efficiency_1m = (val_metrics['loss'] / model_params_true) * 1000000  # Loss per 1M params (lower is better)
        loss_efficiency_log = val_metrics['loss'] / math.log10(model_params_true + 1)  # Loss per log10(params) (lower is better)
        
        # Only log log-scaled efficiency metrics (reduce clutter, better for large param ranges)
        log_dict['param_efficiency/accuracy_per_log10_params'] = param_efficiency_log
        log_dict['param_efficiency/loss_per_log10_params'] = loss_efficiency_log  # Lower is better
        log_dict['model_params'] = model_params_true  # Log true param count as metric (not hyperparameter)
        
        # Also log test parameter efficiency if available (only log-scaled)
        if test_metrics_epoch is not None:
            test_param_efficiency_log = test_metrics_epoch['accuracy'] / math.log10(model_params_true + 1)
            test_loss_efficiency_log = test_metrics_epoch['loss'] / math.log10(model_params_true + 1)  # FIXED: Lower is better
            
            log_dict['param_efficiency/test_accuracy_per_log10_params'] = test_param_efficiency_log
            log_dict['param_efficiency/test_loss_per_log10_params'] = test_loss_efficiency_log
        
        wandb.log(log_dict)
        
        # Print epoch summary
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
        if test_metrics_epoch is not None:
            print(f"  Test:  Loss={test_metrics_epoch['loss']:.4f}, Acc={test_metrics_epoch['accuracy']:.2f}%")
        if val_metrics['accuracy'] == best_val_acc:
            print(f"  ✓ New best validation accuracy!")
    
    # Final test evaluation (if provided) - use consistent variable names
    final_test_metrics = None
    if test_loader is not None:
        print("\nEvaluating on test set...")
        final_test_metrics = validate(model, test_loader, config, device)
        test_loss_final = final_test_metrics['loss']
        test_acc_final = final_test_metrics['accuracy']
        print(f"  Test: Loss={test_loss_final:.4f}, Acc={test_acc_final:.2f}%")
        
        # Log final test metrics
        wandb.log({
            'final/test_loss': test_loss_final,
            'final/test_accuracy': test_acc_final
        })
    
    # Calculate parameter efficiency metrics (use TRUE parameter count already stored)
    # model_params_true was calculated at model creation
    
    # Final wandb summary (clear variable names, no duplicates)
    final_summary = {
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_accuracy': history['train_acc'][-1],
        'final_val_accuracy': history['val_acc'][-1],
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'model_params': model_params_true,  # TRUE parameter count
    }
    
    if final_test_metrics:
        import math
        test_acc_final = final_test_metrics['accuracy']
        test_loss_final = final_test_metrics['loss']
        
        # Multiple efficiency metrics with better scaling
        param_efficiency = (test_acc_final / model_params_true) * 100000  # Accuracy per 100K parameters (TRUE count)
        accuracy_per_million = (test_acc_final / model_params_true) * 1000000  # Accuracy per 1M parameters (TRUE count)
        accuracy_per_log_params = test_acc_final / math.log10(model_params_true + 1)  # Log-scaled (better for large ranges)
        
        # Loss efficiency (lower loss = better, so calculate loss per param directly)
        # FIXED: Lower is better
        loss_efficiency = (test_loss_final / model_params_true) * 100000  # Loss per 100K params (lower is better)
        loss_efficiency_million = (test_loss_final / model_params_true) * 1000000  # Loss per 1M params (lower is better)
        loss_efficiency_log = test_loss_final / math.log10(model_params_true + 1)  # Loss per log10(params) (lower is better)
        
        final_summary.update({
            'final_test_accuracy': test_acc_final,
            'final_test_loss': test_loss_final,
            'final/test_accuracy': test_acc_final,
            'final/test_loss': test_loss_final,
            # Accuracy efficiency metrics
            'final/param_efficiency': param_efficiency,  # Acc per 100K params
            'param_efficiency': param_efficiency,
            'final/accuracy_per_million_params': accuracy_per_million,
            'accuracy_per_million_params': accuracy_per_million,
            'final/accuracy_per_log10_params': accuracy_per_log_params,
            'final/param_efficiency/test_accuracy_per_log10_params': accuracy_per_log_params,  # For sweep optimization
            # Loss efficiency metrics (inverse - higher is better)
            'final/loss_efficiency_per_100k_params': loss_efficiency,
            'final/loss_efficiency_per_1m_params': loss_efficiency_million,
            'final/loss_efficiency_per_log10_params': loss_efficiency_log,
            'loss_efficiency_per_100k_params': loss_efficiency,
            'loss_efficiency_per_1m_params': loss_efficiency_million,
            'loss_efficiency_per_log10_params': loss_efficiency_log
        })
        
        # Also log as metrics for sweep optimization
        wandb.log({
            'final/test_accuracy': test_acc_final,
            'final/test_loss': test_loss_final,
            'final/param_efficiency': param_efficiency,
            'final/accuracy_per_million_params': accuracy_per_million,
            'final/accuracy_per_log10_params': accuracy_per_log_params,
            'final/param_efficiency/test_accuracy_per_log10_params': accuracy_per_log_params,  # For sweep optimization
            'final/loss_per_log10_params': loss_efficiency_log  # Only log-scaled (reduce clutter)
        })
        
        print(f"\nParameter Efficiency (Accuracy - log-scaled):")
        print(f"  Per log10(params): {accuracy_per_log_params:.2f} accuracy")
        print(f"\nParameter Efficiency (Loss - log-scaled, lower is better):")
        print(f"  Per log10(params): {loss_efficiency_log:.4f} loss")
    
    wandb.summary.update(final_summary)
    wandb.finish()
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_metrics': final_test_metrics,
        'model': model
    }


if __name__ == '__main__':
    # Quick test
    print("Testing training utilities...")
    
    from collective_model.config import CONFIG_DEBUG, prepare_config
    from collective_model.data import get_mnist_loaders
    
    # Prepare config
    config = prepare_config(CONFIG_DEBUG)
    config['epochs'] = 1  # Just 1 epoch for testing
    config['wandb_project'] = 'collective-model-test'
    config['wandb_name'] = 'debug-test'
    
    # Get data
    loaders, info = get_mnist_loaders(
        batch_size=config['batch_size'],
        use_augmentation=config.get('use_augmentation', False)
    )
    
    print(f"\nDataset info: {info['train_size']} train, {info['val_size']} val")
    
    # Train (mini test)
    print("\nRunning mini training test...")
    try:
        results = train_strategy_c(
            config=config,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            test_loader=loaders['test']
        )
        
        print("\n✓ Training test completed!")
        print(f"  Best val acc: {results['best_val_acc']:.2f}%")
        if results['test_metrics']:
            print(f"  Test acc: {results['test_metrics']['accuracy']:.2f}%")
    except Exception as e:
        print(f"\n✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()

