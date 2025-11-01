# Collective Model Only Sweep Guide

## Overview

This sweep focuses **ONLY on the Collective Model** for hyperparameter tuning. 
After finding the best collective configuration, we'll test it against baselines separately.

## Current Sweep ID

**`morizk/collective_model_code/svwsin93`**

View at: https://wandb.ai/morizk/collective_model_code/sweeps/svwsin93

## How to Run

```bash
cd ~/Desktop/rizk/per/SE/collective_model/collective_model_code
source venv/bin/activate

# Run sweep agent
wandb agent morizk/collective_model_code/svwsin93
```

## Key Features

### ✅ Collective Model Only
- `model_type: collective` (fixed - no baselines)
- Tests only collective architecture hyperparameters
- Faster than full sweep (no baseline searches)

### ✅ Parameter Efficiency Metrics
- **Logged EVERY EPOCH** as graphs in wandb:
  - `param_efficiency/accuracy_per_100k_params` (val)
  - `param_efficiency/accuracy_per_1m_params` (val)
  - `param_efficiency/test_accuracy_per_100k_params` (test)
  - `param_efficiency/test_accuracy_per_1m_params` (test)
  - `model_params` (true parameter count from model)

### ✅ True Parameter Count
- Uses **actual model.parameters()** count (not calculated)
- Logged as **metric** (not hyperparameter)
- Shows up in wandb graphs over epochs

### ✅ Model Structure Only in Config
- **NO `model_params` or `target_params` in wandb.config**
- Only logs architectural specifications:
  - `n_total`, `expert_ratio`
  - `expert_hidden`, `analyst_hidden`
  - `expert_output`, `analyst_output`
  - etc.

## Sweep Configuration

- **Batch sizes**: `[256, 512]` (training), `[512, 1024]` (eval)
- **Augmentation**: `true` (default, enabled)
- **Epochs**: `200`
- **Model sizes**: `n_total: [6, 8, 10, 12]`
- **Optimizers**: `[adam, adamw]`

## Metrics Tracked

Every epoch:
- `train/loss`, `train/accuracy`
- `val/loss`, `val/accuracy`
- `test/loss`, `test/accuracy` (if available)
- `param_efficiency/accuracy_per_100k_params` ✨
- `param_efficiency/accuracy_per_1m_params` ✨
- `param_efficiency/test_accuracy_per_100k_params` ✨
- `model_params` (true count) ✨

Final summary:
- `final/test_accuracy`
- `final/param_efficiency`
- `final/accuracy_per_million_params`

## After This Sweep

Once you find the best collective model configuration:
1. Fix those hyperparameters
2. Run the **full sweep** (`phase1_sweep.yaml`) to compare against baselines
3. Compare parameter efficiency across all models

