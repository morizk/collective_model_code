# Fix Import Error on Other Devices

## Problem
`ModuleNotFoundError: No module named 'collective_model.data'` when running sweep on a different device.

## Solution

After pulling the latest code (which now includes `setup.py`), install the package in editable mode:

```bash
cd ~/Desktop/per/rizk/collective_model_code
source venv/bin/activate  # or your venv activation command

# Install the package in editable mode
pip install -e .
```

This will:
- Install `collective_model` and `baselines` as proper Python packages
- Make them importable from anywhere
- Link to the source code (editable mode) so changes are reflected immediately

## Verify Installation

Test that imports work:
```bash
python -c "from collective_model.config import prepare_config; print('✓ Imports work!')"
```

## Then Run Sweep

After installation, run the sweep normally:
```bash
wandb agent morizk/collective_model_code/751s938a
```

