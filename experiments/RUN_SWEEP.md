# How to Run W&B Sweep

## Quick Start

### 1. Navigate to Project Directory
```bash
cd ~/Desktop/rizk/per/SE/collective_model/collective_model_code
```

### 2. Activate Virtual Environment
```bash
source venv/bin/activate  # Linux/Mac
# OR
# venv\Scripts\activate  # Windows
```

### 3. Initialize Sweep
```bash
wandb sweep experiments/configs/collective_v4_sweep.yaml
```

This will output something like:
```
wandb: Creating sweep from: experiments/configs/collective_v4_sweep.yaml
wandb: Created sweep with ID: abc123xyz
wandb: View sweep at: https://wandb.ai/your-username/your-project/sweeps/abc123xyz
```

**Save the sweep ID** (e.g., `abc123xyz`) - you'll need it to run agents.

### 4. Run Sweep Agent(s)

**Single agent (one GPU):**
```bash
wandb agent your-username/your-project/abc123xyz
```

**Multiple agents (parallel, if you have multiple GPUs):**
```bash
# Terminal 1
wandb agent your-username/your-project/abc123xyz

# Terminal 2 (on another GPU)
CUDA_VISIBLE_DEVICES=1 wandb agent your-username/your-project/abc123xyz

# Terminal 3 (on another GPU)
CUDA_VISIBLE_DEVICES=2 wandb agent your-username/your-project/abc123xyz
```

**Replace:**
- `your-username` → Your W&B username (e.g., `morizk`)
- `your-project` → Your W&B project name (e.g., `collective_model_code`)
- `abc123xyz` → The sweep ID from step 3

---

## Detailed Instructions

### Prerequisites

1. **W&B Account**: Make sure you're logged in
   ```bash
   wandb login
   ```

2. **Check W&B Project**: The project name is set in `train.py` or via `wandb_project` config
   - Default: Check `train.py` for `wandb_project` parameter
   - Or set it in the sweep YAML if needed

3. **Verify Configuration**: Check the sweep file is valid
   ```bash
   wandb sweep --dry-run experiments/configs/collective_v4_sweep.yaml
   ```

### Running the Sweep

#### Option 1: Run Locally (Single Machine)

```bash
# Initialize sweep
wandb sweep experiments/configs/collective_v4_sweep.yaml

# Run agent (will keep running until you stop it)
wandb agent morizk/collective_model_code/SWEEP_ID
```

**To stop the agent:**
- Press `Ctrl+C` in the terminal
- The current run will finish, then it stops

#### Option 2: Run on Multiple GPUs (Parallel)

If you have multiple GPUs, you can run multiple agents in parallel:

```bash
# Terminal 1 - GPU 0
wandb agent morizk/collective_model_code/SWEEP_ID

# Terminal 2 - GPU 1
CUDA_VISIBLE_DEVICES=1 wandb agent morizk/collective_model_code/SWEEP_ID

# Terminal 3 - GPU 2
CUDA_VISIBLE_DEVICES=2 wandb agent morizk/collective_model_code/SWEEP_ID

# Terminal 4 - GPU 3
CUDA_VISIBLE_DEVICES=3 wandb agent morizk/collective_model_code/SWEEP_ID
```

**Note:** Each agent will pick up different hyperparameter configurations automatically.

#### Option 3: Run in Background (Detached)

```bash
# Run in background
nohup wandb agent morizk/collective_model_code/SWEEP_ID > sweep.log 2>&1 &

# Check if running
ps aux | grep wandb

# View logs
tail -f sweep.log

# Stop background process
pkill -f "wandb agent"
```

---

## Monitoring the Sweep

### View in W&B Dashboard

1. Go to: `https://wandb.ai/your-username/your-project/sweeps/SWEEP_ID`
2. You'll see:
   - **Overview**: Best runs, progress, parallel coordinates
   - **Runs**: All individual runs with their configs
   - **Charts**: Training curves, parameter importance, etc.

### Check Progress Locally

```bash
# View recent runs
wandb status

# View specific run
wandb run <run_id>
```

---

## Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'collective_model'"

**Solution:**
```bash
# Make sure you're in the project root
cd ~/Desktop/rizk/per/SE/collective_model/collective_model_code

# Install package in editable mode
pip install -e .
```

### Issue 2: "CUDA out of memory"

**Solution:**
- Reduce `batch_size` in the sweep config
- Or use `CUDA_VISIBLE_DEVICES` to use a different GPU with more memory

### Issue 3: "File not found or corrupted" (Fashion-MNIST download)

**Solution:**
```bash
# Remove corrupted data
rm -rf data/FashionMNIST

# Re-run - it will re-download
wandb agent morizk/collective_model_code/SWEEP_ID
```

### Issue 4: Sweep not progressing

**Check:**
1. Is the agent still running? (`ps aux | grep wandb`)
2. Are there errors in the logs?
3. Check W&B dashboard for failed runs

---

## Stopping the Sweep

### Stop Individual Agent
```bash
# Press Ctrl+C in the terminal running the agent
```

### Stop All Agents
```bash
# Kill all wandb agent processes
pkill -f "wandb agent"
```

### Pause Sweep (W&B Dashboard)
1. Go to sweep page
2. Click "Pause" or "Stop" button
3. Agents will stop picking up new runs

---

## Expected Runtime

**With current settings:**
- **Epochs**: 100 (max), but early termination kicks in at 50 if bad
- **Time per run**: ~20-40 minutes (depending on batch size and model size)
- **Expected runs**: ~30-50 configurations (Bayesian optimization)
- **Total time**: ~10-20 hours on single GPU, ~3-5 hours with 4 GPUs

**Early termination (Hyperband):**
- Bad runs stop at 50 epochs
- Good runs continue to 100 epochs
- This saves significant time!

---

## Tips

1. **Start with 1 agent** to verify everything works
2. **Monitor first few runs** to catch any issues early
3. **Use multiple GPUs** if available to speed up
4. **Check W&B dashboard regularly** to see progress
5. **Let it run overnight** - sweeps take time!

---

## Example Full Workflow

```bash
# 1. Navigate to project
cd ~/Desktop/rizk/per/SE/collective_model/collective_model_code

# 2. Activate venv
source venv/bin/activate

# 3. Initialize sweep
wandb sweep experiments/configs/collective_v4_sweep.yaml
# Output: wandb: Created sweep with ID: abc123xyz

# 4. Run agent (single GPU)
wandb agent morizk/collective_model_code/abc123xyz

# OR run multiple agents (4 GPUs)
# Terminal 1:
wandb agent morizk/collective_model_code/abc123xyz

# Terminal 2:
CUDA_VISIBLE_DEVICES=1 wandb agent morizk/collective_model_code/abc123xyz

# Terminal 3:
CUDA_VISIBLE_DEVICES=2 wandb agent morizk/collective_model_code/abc123xyz

# Terminal 4:
CUDA_VISIBLE_DEVICES=3 wandb agent morizk/collective_model_code/abc123xyz

# 5. Monitor in W&B dashboard
# Go to: https://wandb.ai/morizk/collective_model_code/sweeps/abc123xyz
```

---

## Next Steps After Sweep

1. **Analyze results** in W&B dashboard
2. **Identify best configuration** from sweep
3. **Run final validation** with best config
4. **Compare with baselines** if needed
5. **Document findings** for future sweeps

