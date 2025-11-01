# Wandb Sweep Configuration Guide

## Overview

This sweep tests **7 models** (1 collective + 6 baselines) across **different parameter counts** to measure **parameter efficiency**.

### Key Innovation: `target_params` Parameter

The sweep includes a `target_params` parameter with values: `[1.8M, 2.85M, 3.95M, 5M, 6.1M, 7.15M]` (matched to collective model range)

**How it works:**
- **Collective models**: Use `n_total` + `expert_ratio` to define architecture (ignore `target_params`)
- **Baseline models**: Use `target_params` to determine size (ignore `n_total`/`expert_ratio`)

This allows fair comparison across different model sizes!

## ✅ Correct Configuration

Use the provided YAML file: `experiments/configs/phase1_sweep.yaml`

### How to Create Sweep from YAML:

**Option 1: Using wandb CLI (Recommended)**
```bash
cd collective_model_code
source venv/bin/activate

# Create sweep from YAML file
wandb sweep experiments/configs/phase1_sweep.yaml
```

This will output a sweep ID like: `morizk/collective_model_code/ydnjncz0`

**Note**: The sweep ID format is `username/project_name/sweep_id`
**Current Sweep ID**: `morizk/collective_model_code/ydnjncz0`

**Option 2: Using Python**
```python
import wandb

# Load YAML config
sweep_config = wandb.sweep_from_yaml('experiments/configs/phase1_sweep.yaml')

# Create sweep
sweep_id = wandb.sweep(sweep_config, project="collective-architecture")
print(f"Sweep ID: {sweep_id}")
```

### Key Configuration Features:

✅ **Metric**: `goal: maximize`, `name: final/test_accuracy` (logged at end of training)
✅ **Models**: All 7 models (collective + 6 baselines)
✅ **Model Sizes**: `target_params` = [1.8M, 2.85M, 3.95M, 5M, 6.1M, 7.15M] (matches collective range)
✅ **Epochs**: Fixed at 200 (full training)
✅ **Batch Size**: 128-256 (optimized from GPU monitoring)
✅ **Fixed Parameters**: `input_dim=784`, `num_classes=10` (MNIST)
✅ **No Derived Parameters**: Only sweeps parameters that should vary

### Metrics Logged:

Every training run logs:
- `final/test_accuracy`: Test accuracy (used for sweep optimization)
- `final/param_efficiency`: Accuracy per 100K parameters
- `final/accuracy_per_million_params`: Accuracy per 1M parameters
- `final/model_params`: Total parameter count

**Parameter efficiency** shows how well models use their parameters!

### Running the Sweep:

After creating the sweep, run agents:

```bash
# Use the ACTUAL sweep ID from wandb output
# Current sweep ID: morizk/collective_model_code/ydnjncz0

# Terminal 1 (or script)
wandb agent morizk/collective_model_code/ydnjncz0

# Terminal 2 (if you have multiple GPUs)
wandb agent morizk/collective_model_code/ydnjncz0

# Terminal 3 (if you have multiple GPUs)
wandb agent morizk/collective_model_code/ydnjncz0
```

**Current Sweep ID**: `morizk/collective_model_code/ydnjncz0`

Each agent will:
- Pull a new hyperparameter configuration
- Train the model
- Report results back to wandb
- Get next configuration

### Expected Results:

- **~30-50 configs per model type** (Bayesian optimization)
- **Average ~30 epochs per config** (Hyperband early stopping)
- **Total: ~8,400 epochs** across all 7 models
- **Time**: ~20-40 hours on 1 GPU (can parallelize with multiple agents)

### Monitoring:

- View progress: https://wandb.ai/your-username/collective-architecture
- Each run shows: 
  - Training curves, validation accuracy, test accuracy
  - **Parameter count** and **parameter efficiency**
  - Model architecture details
- Best configs automatically identified by wandb
- **Key analysis**: Compare `param_efficiency` across model types and sizes

### Analyzing Parameter Efficiency:

In wandb, you can create custom plots:
1. **Accuracy vs Parameters**: Scatter plot showing efficiency
2. **Parameter Efficiency by Model Type**: Compare collective vs baselines
3. **Scaling Analysis**: How accuracy changes with model size

Example query:
```python
import wandb
api = wandb.Api()
runs = api.runs("your-username/collective-architecture")

for run in runs:
    model_type = run.config.get('model_type')
    params = run.summary.get('final/model_params')
    accuracy = run.summary.get('final/test_accuracy')
    efficiency = run.summary.get('final/param_efficiency')
    
    print(f"{model_type}: {params:,} params, {accuracy:.2f}% acc, {efficiency:.2f} eff")
```

## How Baseline Size Matching Works

When sweep provides `target_params` for a baseline run:

1. **Extract** `target_params` from sweep config (e.g., 500000)
2. **Search** for baseline architecture matching this count:
   - Shallow: 2-3 wide layers
   - Balanced: 4-5 layers with geometric progression
   - Deep: 8-10 constant-width layers
   - Very Deep: 12-15 constant-width layers
   - Deep ResNet: 8-10 layers with skip connections
   - Very Deep ResNet: 12-15 layers with skip connections
3. **Train** the found architecture
4. **Log** parameter efficiency metrics

For collective models, `n_total` and `expert_ratio` determine size naturally.

## Quick Setup Summary

1. **Create sweep**: `wandb sweep experiments/configs/phase1_sweep.yaml`
2. **Copy sweep ID** from output (format: `username/project/sweep_id`)
3. **Run agent(s)**: `wandb agent username/project/sweep_id`
4. **Monitor** at: https://wandb.ai/username/project/sweeps/sweep_id
5. **Analyze** parameter efficiency after completion

## Questions?

- Check `SETUP_AND_PLAN.md` for detailed hyperparameter strategy
- Check `BASELINE_MATCHING_EXPLAINED.md` for baseline architecture details
- Check `train.py` to see how `target_params` is used
- Check `GPU_OPTIMIZATION_FINDINGS.md` for batch size optimization

