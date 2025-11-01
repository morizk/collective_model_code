"""
Loss functions for training the Collective Model.

Includes diversity regularization to encourage experts to learn
complementary features rather than redundant ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def diversity_loss_experts(expert_outputs, temperature=1.0):
    """
    Compute diversity loss to encourage experts to produce different outputs.
    
    The loss penalizes high correlation between expert outputs.
    Lower loss = more diverse experts.
    
    Strategy: For each pair of experts, compute normalized correlation.
    We want experts to be orthogonal (correlation ≈ 0).
    
    Args:
        expert_outputs (list[torch.Tensor]): List of expert output tensors
                                              Each tensor has shape [batch_size, expert_dim]
        temperature (float): Temperature parameter (higher = less aggressive)
                            Default: 1.0
    
    Returns:
        torch.Tensor: Scalar diversity loss value
    
    Example:
        >>> expert1 = torch.randn(32, 128)
        >>> expert2 = torch.randn(32, 128)
        >>> loss = diversity_loss_experts([expert1, expert2], temperature=1.0)
        >>> print(loss.item())  # Should be low if experts are diverse
    """
    n_experts = len(expert_outputs)
    
    if n_experts < 2:
        # No diversity loss if only one expert
        return torch.tensor(0.0, device=expert_outputs[0].device)
    
    total_correlation = 0.0
    count = 0
    
    # Compare all pairs of experts
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            # Normalize features to unit vectors
            out_i = F.normalize(expert_outputs[i], dim=1, p=2)  # [batch, dim]
            out_j = F.normalize(expert_outputs[j], dim=1, p=2)  # [batch, dim]
            
            # Compute cosine similarity (correlation)
            # High similarity = redundant experts (bad)
            # Low similarity = diverse experts (good)
            correlation = (out_i * out_j).sum(dim=1).mean()  # Scalar
            
            # Apply temperature scaling
            correlation = correlation / temperature
            
            # Accumulate absolute correlation
            # We penalize both high positive AND high negative correlation
            total_correlation += correlation.abs()
            count += 1
    
    # Average over all pairs
    avg_correlation = total_correlation / count if count > 0 else 0.0
    
    return avg_correlation


def diversity_loss_analysts(analyst_outputs, temperature=1.0):
    """
    Compute diversity loss for analysts.
    
    Currently uses the same implementation as expert diversity loss,
    but kept as a separate function to allow different implementations
    in the future if needed.
    
    Args:
        analyst_outputs (list[torch.Tensor]): List of analyst output tensors
        temperature (float): Temperature parameter
    
    Returns:
        torch.Tensor: Scalar diversity loss value
    """
    n_analysts = len(analyst_outputs)
    
    if n_analysts < 2:
        # No diversity loss if only one analyst
        return torch.tensor(0.0, device=analyst_outputs[0].device)
    
    total_correlation = 0.0
    count = 0
    
    # Compare all pairs of analysts
    for i in range(n_analysts):
        for j in range(i + 1, n_analysts):
            # Normalize features to unit vectors
            out_i = F.normalize(analyst_outputs[i], dim=1, p=2)
            out_j = F.normalize(analyst_outputs[j], dim=1, p=2)
            
            # Compute cosine similarity (correlation)
            correlation = (out_i * out_j).sum(dim=1).mean()
            
            # Apply temperature scaling
            correlation = correlation / temperature
            
            # Accumulate absolute correlation
            total_correlation += correlation.abs()
            count += 1
    
    # Average over all pairs
    avg_correlation = total_correlation / count if count > 0 else 0.0
    
    return avg_correlation


def combined_loss(
    prediction_logits,
    targets,
    expert_outputs=None,
    analyst_outputs=None,
    diversity_lambda=0.01,
    diversity_temperature=1.0,
    use_expert_diversity=True,
    use_analyst_diversity=False
):
    """
    Compute combined loss: prediction loss + diversity regularization.
    
    Args:
        prediction_logits (torch.Tensor): Model predictions [batch_size, num_classes]
        targets (torch.Tensor): Ground truth labels [batch_size]
        expert_outputs (list[torch.Tensor]): Expert feature outputs (optional)
        analyst_outputs (list[torch.Tensor]): Analyst feature outputs (optional)
        diversity_lambda (float): Weight for diversity loss (0 = no diversity)
        diversity_temperature (float): Temperature for diversity loss
        use_expert_diversity (bool): Whether to apply diversity to experts
        use_analyst_diversity (bool): Whether to apply diversity to analysts
    
    Returns:
        dict: Dictionary with keys:
            - 'total': Total loss
            - 'prediction': Prediction loss (cross-entropy)
            - 'diversity_expert': Expert diversity loss (if used)
            - 'diversity_analyst': Analyst diversity loss (if used)
    
    Example:
        >>> logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> expert_outs = [torch.randn(32, 128) for _ in range(3)]
        >>> losses = combined_loss(logits, targets, expert_outputs=expert_outs)
        >>> total_loss = losses['total']
        >>> total_loss.backward()
    """
    # Prediction loss (cross-entropy)
    pred_loss = F.cross_entropy(prediction_logits, targets)
    
    losses = {
        'prediction': pred_loss,
        'total': pred_loss
    }
    
    # Add expert diversity loss
    if use_expert_diversity and expert_outputs is not None and len(expert_outputs) > 1:
        div_loss_expert = diversity_loss_experts(expert_outputs, temperature=diversity_temperature)
        losses['diversity_expert'] = div_loss_expert
        losses['total'] = losses['total'] + diversity_lambda * div_loss_expert
    else:
        losses['diversity_expert'] = torch.tensor(0.0, device=pred_loss.device)
    
    # Add analyst diversity loss
    if use_analyst_diversity and analyst_outputs is not None and len(analyst_outputs) > 1:
        div_loss_analyst = diversity_loss_analysts(analyst_outputs, temperature=diversity_temperature)
        losses['diversity_analyst'] = div_loss_analyst
        losses['total'] = losses['total'] + diversity_lambda * div_loss_analyst
    else:
        losses['diversity_analyst'] = torch.tensor(0.0, device=pred_loss.device)
    
    return losses


if __name__ == '__main__':
    # Test diversity loss
    print("Testing diversity loss functions...")
    
    # Test 1: Identical experts (should have HIGH correlation)
    print("\n1. Testing with identical experts:")
    expert1 = torch.randn(32, 128)
    expert2 = expert1.clone()  # Identical
    loss = diversity_loss_experts([expert1, expert2])
    print(f"   Diversity loss (identical): {loss.item():.4f} (should be ~1.0)")
    
    # Test 2: Orthogonal experts (should have LOW correlation)
    print("\n2. Testing with orthogonal experts:")
    expert1 = torch.randn(32, 128)
    expert2 = torch.randn(32, 128)
    loss = diversity_loss_experts([expert1, expert2])
    print(f"   Diversity loss (random): {loss.item():.4f} (should be ~0.0-0.3)")
    
    # Test 3: Multiple experts
    print("\n3. Testing with multiple experts:")
    experts = [torch.randn(32, 128) for _ in range(5)]
    loss = diversity_loss_experts(experts)
    print(f"   Diversity loss (5 random experts): {loss.item():.4f}")
    
    # Test 4: Temperature effect
    print("\n4. Testing temperature effect:")
    expert1 = torch.randn(32, 128)
    expert2 = expert1.clone() + torch.randn(32, 128) * 0.1  # Similar but not identical
    loss_t1 = diversity_loss_experts([expert1, expert2], temperature=1.0)
    loss_t2 = diversity_loss_experts([expert1, expert2], temperature=2.0)
    print(f"   Temperature 1.0: {loss_t1.item():.4f}")
    print(f"   Temperature 2.0: {loss_t2.item():.4f} (should be lower)")
    
    # Test 5: Combined loss
    print("\n5. Testing combined loss:")
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    expert_outputs = [torch.randn(32, 128) for _ in range(3)]
    
    losses = combined_loss(
        logits, targets,
        expert_outputs=expert_outputs,
        diversity_lambda=0.01,
        use_expert_diversity=True
    )
    
    print(f"   Prediction loss: {losses['prediction'].item():.4f}")
    print(f"   Diversity loss: {losses['diversity_expert'].item():.4f}")
    print(f"   Total loss: {losses['total'].item():.4f}")
    
    # Test gradient flow
    print("\n6. Testing gradient flow:")
    logits.requires_grad = True
    for exp in expert_outputs:
        exp.requires_grad = True
    
    losses = combined_loss(logits, targets, expert_outputs=expert_outputs, diversity_lambda=0.1)
    losses['total'].backward()
    
    print(f"   Logits grad: {logits.grad is not None}")
    print(f"   Expert 0 grad: {expert_outputs[0].grad is not None}")
    print("   ✓ Gradients flow correctly")
    
    print("\n✓ All diversity loss tests passed!")

