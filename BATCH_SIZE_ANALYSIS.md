# Batch Size Analysis: Why 128 Performs Better Than 512-1024

## Observation from Wandb Sweep Results

Smaller batch sizes (128) are showing **better test accuracy** than larger batch sizes (512-1024).

## Why Smaller Batch Sizes Often Perform Better

### 1. **Gradient Noise = Regularization** 📊
- **Small batches**: High gradient variance → acts as **implicit regularization**
- **Large batches**: Low gradient variance → smooth but can overfit
- The noise in small-batch gradients helps escape sharp minima and find **flatter, more generalizable solutions**

### 2. **More Frequent Updates** 🔄
- **Batch 128**: ~422 updates per epoch (for Fashion-MNIST train set ~54K samples)
- **Batch 512**: ~105 updates per epoch
- More frequent updates = better exploration of loss landscape
- Can find better solutions faster (more "steps" per epoch)

### 3. **Better Generalization** 🎯
- Smaller batches often lead to **better test accuracy** even if train accuracy is similar
- Related to finding wider minima (flat regions) vs sharp minima
- Sharp minima = overfitting, wide minima = generalization

### 4. **Learning Dynamics** ⚡
- **Small batches**: Model sees diverse examples more frequently
- **Large batches**: Averages over many examples, may miss important patterns
- The "fresh information" from smaller batches can help learning

### 5. **Optimizer Behavior** 🔧
- **Adam/AdamW** with smaller batches:
  - Gradient estimates are noisier but more diverse
  - Better exploration vs exploitation balance
- **Large batches**:
  - Gradient estimates are stable but may converge to worse local minima
  - Less exploration of the loss landscape

### 6. **Memory vs Performance Trade-off** 💾
- **GPU Memory**: Large batches use more memory (not always better)
- **Training Time**: Larger batches are faster per epoch BUT need more epochs
- **Total Time**: Often similar or worse (more epochs needed)

## Common Pattern in Deep Learning

```
Small Batch (32-128):  Better generalization, slower per epoch, fewer epochs needed
Medium Batch (256-512): Balanced (often good default)
Large Batch (1024+):   Faster per epoch, but may need many more epochs + worse generalization
```

## For Your Collective Model

The observation that **batch_size=128 performs better** is consistent with:
- **Complex architectures** (like collective model) often benefit from smaller batches
- **Regularization effect** of noisy gradients helps prevent overfitting
- **Better exploration** of the hierarchical feature space

## Recommendation

✅ **Use batch_size=128** (as wandb results show)
- Better test accuracy
- Better generalization
- Good balance of speed and performance

**Note**: Keep `eval_batch_size` large (512-1024) for:
- Smoother validation/test metrics (less variance)
- Faster evaluation (no gradients, can use more memory)

## References

- "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima" (Keskar et al., 2017)
- "Stochastic Gradient Descent with Large Minibatches" (Goyal et al., 2017)
- General wisdom: smaller batches = better generalization in many cases


