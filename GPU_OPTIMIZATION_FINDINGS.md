# GPU Optimization Findings

## Summary
Training was monitored during debug run to identify GPU utilization bottlenecks and optimization opportunities for faster hyperparameter search.

## Test Configuration
- **Model**: Collective Model (debug config)
- **Batch Size**: 32
- **Model Parameters**: 559,738
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB)
- **Monitoring Duration**: 5 minutes (300 seconds, 90 samples)

## Findings

### GPU Utilization
- **Average**: 11.4%
- **Min**: 0.0%
- **Max**: 20.0%
- **Status**: ⚠️ **VERY LOW** - Training is CPU-bound or batch size too small

### Memory Usage
- **Peak Memory**: 167 MB / 6141 MB
- **Peak Usage**: 2.7%
- **Average Usage**: 2.2%
- **Status**: ⚠️ **VERY LOW** - Only using 2.7% of available GPU memory

### Power Consumption
- **Average**: 26.8 W
- **Peak**: 590.0 W (likely measurement error/spike)
- **Min**: 1.7 W
- **Status**: Low power draw indicates GPU is not heavily utilized

## Recommendations

### Immediate Actions
1. **Increase Batch Size**:
   - Current: 32 (debug), 128 (phase1)
   - Recommended: **128-256** (debug), **256-512** (phase1)
   - Expected improvement: **4-8x faster training**

2. **Memory Headroom**:
   - Available: 97.3% of GPU memory
   - Can safely increase batch size **4-8x** without memory issues
   - Estimated max batch size: **256-512** (with current model size)

3. **Training Speed**:
   - Current GPU utilization: 11.4%
   - With optimized batch size: Expected 60-80% utilization
   - **Potential speedup: 4-8x** for hyperparameter search

### Hyperparameter Search Optimization
- **Current**: Training is CPU-bound
- **After optimization**: GPU-bound (better)
- **Impact**: 
  - Faster individual training runs
  - Can run more parallel experiments
  - Reduce total hyperparameter search time from ~70 hours to ~10-20 hours

### Next Steps
1. Test with batch_size=256 in phase1 config
2. Monitor GPU utilization again
3. If still low (<50%), investigate:
   - Data loading bottlenecks (num_workers)
   - CPU preprocessing overhead
   - Consider mixed precision (FP16) training

## Implementation Notes
- Added TODO comments in `config.py` for batch size optimization
- GPU monitoring script: `scripts/monitor_gpu.py`
- CSV output: `gpu_stats_debug_collective.csv`

## Updated Configs
- `CONFIG_DEBUG`: batch_size=32 → **UPDATED to 128** ✅
- `CONFIG_PHASE1`: batch_size=128 → **UPDATED to 256** ✅

## Implementation Status
- ✅ Batch sizes optimized in `config.py`
- ⏳ Need to test optimized batch sizes to verify:
  - GPU utilization improves to 60-80%
  - Training speed improves 4-8x
  - Model accuracy remains stable (>97%)

