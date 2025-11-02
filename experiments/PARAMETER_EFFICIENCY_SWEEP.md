# Parameter Efficiency Sweep Guide

## Overview

This sweep optimizes for **parameter efficiency** (accuracy per 100K parameters) instead of raw accuracy. The Bayesian optimizer will learn from **ALL previous runs** (both accuracy-focused and efficiency-focused) to guide the search!

## Key Innovation: Cross-Sweep Learning

The Bayesian optimizer uses **all runs in the same wandb project** to learn:
- Which hyperparameters lead to high accuracy? (from accuracy sweep)
- Which hyperparameters lead to parameter efficiency? (from efficiency sweep)
- What are the trade-offs? (learns both!)

This means:
- **Accuracy sweep runs** help the efficiency sweep understand what architectures work
- **Efficiency sweep runs** help find parameter-efficient configurations
- **Both sweeps benefit from each other's data!**

## Current Sweeps

### 1. Accuracy Optimization
**Sweep ID**: `morizk/collective_model_code/svwsin93`
- **Metric**: `final/test_accuracy` (maximize)
- **Goal**: Find models with highest accuracy
- **Use case**: Best absolute performance

### 2. Parameter Efficiency Optimization
**Sweep ID**: `morizk/collective_model_code/r4zs0icj`
- **Metric**: `final/param_efficiency` (maximize)
- **Goal**: Find models with best accuracy per 100K parameters
- **Use case**: Best parameter efficiency (important for deployment!)

## How to Run Both Sweeps

You can run them **simultaneously** on different terminals/machines:

```bash
# Terminal 1: Optimize for accuracy
cd ~/Desktop/rizk/per/SE/collective_model/collective_model_code
source venv/bin/activate
wandb agent morizk/collective_model_code/svwsin93

# Terminal 2: Optimize for parameter efficiency
wandb agent morizk/collective_model_code/r4zs0icj
```

## What Gets Logged

Both sweeps log the same metrics, but optimize different ones:

**Every Epoch:**
- `train/loss`, `train/accuracy`
- `val/loss`, `val/accuracy`
- `test/loss`, `test/accuracy`
- `param_efficiency/accuracy_per_100k_params` ✨
- `param_efficiency/accuracy_per_1m_params` ✨
- `param_efficiency/test_accuracy_per_100k_params` ✨
- `model_params` (true count)

**Final Summary:**
- `final/test_accuracy` (used by accuracy sweep)
- `final/param_efficiency` (used by efficiency sweep)
- `final/accuracy_per_million_params`
- `model_params`

## Interpreting Results

### Accuracy Sweep Results:
- **Best run**: Highest `final/test_accuracy`
- **May have**: Many parameters (less efficient)
- **Good for**: When compute/memory is not a constraint

### Parameter Efficiency Sweep Results:
- **Best run**: Highest `final/param_efficiency`
- **May have**: Lower absolute accuracy but much fewer parameters
- **Good for**: Edge deployment, mobile, resource-constrained environments

### Combined Analysis:
1. Compare top models from each sweep
2. Find Pareto frontier (accuracy vs efficiency)
3. Choose based on deployment constraints

## Example Results

**Accuracy Sweep:**
- Best accuracy: 85.2% with 8.5M parameters
- Parameter efficiency: 10.0 accuracy per 100K params

**Efficiency Sweep:**
- Best efficiency: 18.5 accuracy per 100K params
- Accuracy: 82.1% with 4.4M parameters

**Trade-off**: 3.1% accuracy drop for 48% fewer parameters! 🎯

## Benefits of Dual Optimization

1. **Comprehensive Search**: Explores both objectives
2. **Cross-Learning**: Each sweep learns from the other
3. **Pareto Analysis**: Understand accuracy/efficiency trade-offs
4. **Deployment Flexibility**: Choose based on constraints
5. **Research Insight**: Understand parameter efficiency patterns

## Notes

- Both sweeps use the **same hyperparameter space**
- Bayesian optimizer uses **all runs** to guide search (cross-sweep learning!)
- Can run simultaneously or sequentially
- Results complement each other for full analysis

