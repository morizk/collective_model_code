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
    
    pbar = tqdm(train_loader, desc='Train', leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
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
        
        # Backward pass
        loss_dict['total'].backward()
        optimizer.step()
        
        # Update metrics
        losses['prediction'].update(loss_dict['prediction'].item(), data.size(0))
        losses['diversity_expert'].update(loss_dict['diversity_expert'].item(), data.size(0))
        losses['diversity_analyst'].update(loss_dict['diversity_analyst'].item(), data.size(0))
        losses['total'].update(loss_dict['total'].item(), data.size(0))
        
        # Accuracy (configurable top-k)
        topk = config.get('topk', (1,))  # Default: top-1 for classification
        accuracies = compute_accuracy(logits, target, topk=topk)
        acc_meter.update(accuracies[0], data.size(0))  # Use top-1 for metric
        
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
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Val', leave=False):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
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
    print(f"Model parameters: {model.get_num_parameters():,}")
    
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
    wandb.config.update({
        # Model architecture
        'model_params': model.get_num_parameters(),
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
        'use_augmentation': config.get('use_augmentation', False),
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
        
        wandb.log(log_dict)
        
        # Print epoch summary
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
        if test_metrics_epoch is not None:
            print(f"  Test:  Loss={test_metrics_epoch['loss']:.4f}, Acc={test_metrics_epoch['accuracy']:.2f}%")
        if val_metrics['accuracy'] == best_val_acc:
            print(f"  ✓ New best validation accuracy!")
    
    # Final test evaluation (if provided)
    test_metrics = None
    if test_loader is not None:
        print("\nEvaluating on test set...")
        test_metrics = validate(model, test_loader, config, device)
        print(f"  Test: Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.2f}%")
        
        wandb.log({
            'test/loss': test_metrics['loss'],
            'test/accuracy': test_metrics['accuracy']
        })
    
    # Calculate parameter efficiency metrics
    model_params = model.get_num_parameters()
    
    # Final wandb summary
    final_summary = {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'final/model_params': model_params,
    }
    
    if test_metrics:
        test_acc = test_metrics['accuracy']
        param_efficiency = (test_acc / model_params) * 100000  # Accuracy per 100K parameters
        
        final_summary.update({
            'test_acc': test_acc,
            'final/test_accuracy': test_acc,
            'final/param_efficiency': param_efficiency,  # Acc per 100K params
            'final/accuracy_per_million_params': (test_acc / model_params) * 1000000  # Acc per 1M params
        })
        
        # Also log as metrics for sweep optimization
        wandb.log({
            'final/test_accuracy': test_acc,
            'final/param_efficiency': param_efficiency
        })
        
        print(f"\nParameter Efficiency: {param_efficiency:.2f} accuracy per 100K parameters")
    
    wandb.summary.update(final_summary)
    wandb.finish()
    
    return {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics,
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

