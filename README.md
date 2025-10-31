# Collective Model Architecture

A hierarchical deep learning architecture with multiple expert and analyst models.

## Quick Start

### 1. Setup Environment

```bash
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
# Follow prompts to authenticate
```

### 3. Run Quick Test (Debug Config - 5 epochs, ~2 min)

```bash
python train.py --config debug --model collective
```

### 4. Run Full Training (Phase 1 - 200 epochs, ~30 min)

```bash
# Train collective model
python train.py --config phase1 --model collective

# Train all baselines (can run overnight - ~12 hours total)
python train.py --config phase1 --model shallow
python train.py --config phase1 --model balanced
python train.py --config phase1 --model deep
python train.py --config phase1 --model deep_resnet
python train.py --config phase1 --model very_deep
python train.py --config phase1 --model very_deep_resnet
```

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

Two main configs available:
- **CONFIG_DEBUG**: Fast testing (5 epochs, small models)
- **CONFIG_PHASE1**: Full training (200 epochs, production models)

Edit `collective_model/config.py` to modify hyperparameters.

## Results

Training results are logged to wandb project: `collective-architecture`

## GPU Requirements

- Minimum: 4GB VRAM
- Recommended: 8GB+ VRAM
- Models are optimized for single GPU training

## Questions?

Check the implementation checklist and documentation in the parent directory.

