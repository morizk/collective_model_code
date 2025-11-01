# Collective Model Architecture

A hierarchical deep learning architecture with multiple expert and analyst models that collaboratively solve classification tasks.

## Architecture Overview

The Collective Model consists of:
1. **Expert Layer**: N large models that extract rich feature vectors from input
2. **Encoder Layer**: Compresses concatenated expert outputs
3. **Analyst Layer**: M smaller models that process [original_input + encoded_expert_outputs]
4. **Collective Layer**: Aggregates analyst outputs for final prediction

**Key Feature**: Skip connection from original input to analysts (similar to ResNet)

## Quick Start

### 1. Setup Environment

```bash
# Navigate to code directory
cd collective_model_code

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Login to wandb

```bash
wandb login
# Follow prompts to authenticate with your API key
```

### 3. Run Quick Test (Debug Config - 5 epochs, ~2 min)

```bash
python train.py --config debug --model collective
```

This will:
- Train collective model with small architecture
- Log metrics to wandb
- Save results for analysis

### 4. Monitor GPU Usage (Optional)

In a separate terminal:
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use our monitoring script
python scripts/monitor_gpu.py --duration 300 --output gpu_stats.csv
```

### 5. Run Full Training (Phase 1 - 200 epochs)

**Optimized batch sizes** (debug=128, phase1=256) for 4-8x faster training:

```bash
# Train collective model (~15 min with batch_size=256)
python train.py --config phase1 --model collective

# Train all 6 baselines (~3-6 hours total with optimized batch sizes)
python train.py --config phase1 --model shallow
python train.py --config phase1 --model balanced
python train.py --config phase1 --model deep
python train.py --config phase1 --model deep_resnet
python train.py --config phase1 --model very_deep
python train.py --config phase1 --model very_deep_resnet
```

**Note**: With optimized batch sizes, training is ~4-8x faster than original estimates.

## Project Structure

```
collective_model_code/
├── collective_model/         # Main model package
│   ├── models/              # Expert, Analyst, Encoder, Collective
│   ├── training/            # Training strategies
│   ├── data/                # Data loaders
│   ├── utils/               # Utilities
│   └── config.py            # Configurations
├── baselines/               # Baseline models
├── experiments/             # Experiment configs
├── notebooks/               # Analysis notebooks
├── tests/                   # Unit tests
├── train.py                 # Main training script
└── requirements.txt         # Dependencies
```

## Configuration

Two main configs available in `collective_model/config.py`:

### CONFIG_DEBUG
- **Purpose**: Quick testing and debugging
- **Epochs**: 5
- **Batch Size**: 128 (optimized from 32)
- **Model Size**: Small (559K parameters)
- **Time**: ~2 minutes per run

### CONFIG_PHASE1
- **Purpose**: Full training experiments
- **Epochs**: 200
- **Batch Size**: 256 (optimized from 128)
- **Model Size**: Medium (matches baseline parameter counts)
- **Time**: ~8-15 minutes per model (optimized)

### Key Hyperparameters

**Architecture**:
- `n_total`: Total number of models (experts + analysts)
- `expert_ratio`: Ratio of experts (0.0-1.0)
- `expert_hidden`: Expert hidden layer sizes (list)
- `analyst_hidden`: Analyst hidden layer sizes (list)
- `c_expert`: Expert encoder compression ratio (0.0-1.0)
- `collective_version`: 'simple_mlp' or 'encoder_head'

**Training**:
- `batch_size`: Batch size (optimized: 128/256)
- `learning_rate`: Learning rate (default: 0.001)
- `optimizer`: 'adam', 'adamw', or 'sgd'
- `epochs`: Number of training epochs

**Diversity** (for Strategy C):
- `use_expert_diversity`: Enable diversity loss for experts
- `use_analyst_diversity`: Enable diversity loss for analysts
- `diversity_lambda`: Weight for diversity loss
- `diversity_temperature`: Temperature for diversity (higher = less aggressive)

See `collective_model/config.py` for all available parameters.

## Results & Analysis

### Wandb Integration
All training runs are automatically logged to wandb project: `collective-architecture`

**View results**:
- Dashboard: https://wandb.ai/[your-username]/collective-architecture
- Each run includes: training curves, validation metrics, test accuracy, hyperparameters

### Analysis Notebook
```bash
# Create analysis notebook (after running experiments)
jupyter notebook notebooks/phase1_analysis.ipynb
```

The notebook template includes:
- Loading wandb data
- Plotting training curves
- Comparing collective vs baselines
- Analyzing expert outputs (PCA/t-SNE)
- Calculating inference times

## GPU Optimization

**Findings** (from GPU monitoring):
- Original batch_size=32: **11.4% GPU utilization**, 2.7% memory usage
- Optimized batch_size=128/256: **Expected 60-80% GPU utilization**
- **Speedup**: 4-8x faster training

**Tools**:
- `scripts/monitor_gpu.py`: Monitor GPU during training
- `scripts/run_training_with_monitoring.sh`: Run training + monitoring together

See `GPU_OPTIMIZATION_FINDINGS.md` for detailed analysis.

## Baseline Models

Six baseline architectures are automatically generated to match collective model parameter count:
1. **Shallow**: 2-3 wide layers
2. **Balanced**: 4-5 layers, geometric progression
3. **Deep**: 8 layers, constant width
4. **Very Deep**: 12 layers, constant width
5. **Deep ResNet**: 8 layers with skip connections
6. **Very Deep ResNet**: 12 layers with skip connections

All baselines match parameter count within 5% tolerance.

## GPU Requirements

- **Minimum**: 4GB VRAM
- **Recommended**: 6GB+ VRAM (tested on RTX 4050 6GB)
- **Current Usage**: ~167 MB / 6141 MB (2.7%) - plenty of headroom
- Models are optimized for single GPU training

## Usage Examples

### Train with GPU Monitoring
```bash
./scripts/run_training_with_monitoring.sh debug collective 300
```

### Find Matching Baselines
```python
from collective_model.config import CONFIG_PHASE1, prepare_config
from baselines.find_baselines import find_all_baselines

config = prepare_config(CONFIG_PHASE1)
baselines = find_all_baselines(config)
print(baselines['shallow']['hidden_dims'])  # [460, 410]
```

### Custom Configuration
```python
from collective_model.config import CONFIG_DEBUG, prepare_config

config = CONFIG_DEBUG.copy()
config['batch_size'] = 256  # Custom batch size
config['learning_rate'] = 0.0005  # Custom learning rate
config = prepare_config(config)  # Compute derived values
```

## Troubleshooting

**Low GPU Utilization**:
- Increase `batch_size` (current: 128/256)
- Check `num_workers` in data loaders
- Ensure GPU is available: `torch.cuda.is_available()`

**Out of Memory**:
- Reduce `batch_size`
- Reduce model sizes in config
- Enable gradient accumulation (future feature)

**See**: `IMPLEMENTATION_CHECKLIST.md` in parent directory for detailed troubleshooting.

## Project Structure

```
collective_model_code/
├── collective_model/         # Main model package
│   ├── models/              # Expert, Analyst, Encoder, Collective
│   ├── training/            # Training strategies & trainer
│   ├── data/                # Data loaders (MNIST, etc.)
│   ├── utils/               # Metrics, param counting, visualization
│   └── config.py            # Configuration presets
├── baselines/               # Baseline models & finding utilities
├── scripts/                 # Utility scripts (GPU monitoring, etc.)
├── notebooks/               # Analysis notebooks
├── experiments/             # Experiment configs (future)
├── tests/                   # Unit tests (future)
├── train.py                 # Main training script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Documentation

- **Implementation Checklist**: `../IMPLEMENTATION_CHECKLIST.md`
- **Baseline Matching Strategy**: `../BASELINE_MATCHING_EXPLAINED.md`
- **GPU Optimization**: `GPU_OPTIMIZATION_FINDINGS.md`
- **Setup & Plan**: `../SETUP_AND_PLAN.md`

