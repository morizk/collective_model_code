"""
Metrics tracking utilities.

Provides tools for tracking training/validation metrics efficiently.
"""

import torch
import time


class AverageMeter:
    """
    Computes and stores the average and current value.
    
    Useful for tracking metrics like loss, accuracy, etc. during training.
    
    Example:
        >>> loss_meter = AverageMeter('Loss', ':.4f')
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     loss_meter.update(loss.item(), batch_size)
        >>> print(loss_meter.avg)  # Average loss over all batches
    """
    def __init__(self, name='', fmt=':f'):
        """
        Args:
            name (str): Name of the metric
            fmt (str): Format string for displaying values
        """
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0  # Current value
        self.avg = 0  # Running average
        self.sum = 0  # Cumulative sum
        self.count = 0  # Number of updates
    
    def update(self, val, n=1):
        """
        Update statistics with new value(s).
        
        Args:
            val (float): New value to add
            n (int): Number of samples this value represents (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        """String representation for logging."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricTracker:
    """
    Track multiple metrics simultaneously.
    
    Example:
        >>> tracker = MetricTracker()
        >>> for epoch in range(10):
        ...     tracker.reset()
        ...     for batch in dataloader:
        ...         tracker.update('loss', loss.item(), batch_size)
        ...         tracker.update('acc', acc, batch_size)
        ...     print(tracker.summary())
    """
    def __init__(self):
        self.meters = {}
    
    def reset(self):
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()
    
    def update(self, name, val, n=1):
        """
        Update a specific metric.
        
        Args:
            name (str): Metric name
            val (float): New value
            n (int): Number of samples
        """
        if name not in self.meters:
            self.meters[name] = AverageMeter(name)
        self.meters[name].update(val, n)
    
    def get(self, name):
        """
        Get average value of a metric.
        
        Args:
            name (str): Metric name
        
        Returns:
            float: Average value
        """
        if name in self.meters:
            return self.meters[name].avg
        return None
    
    def summary(self):
        """
        Get summary of all metrics.
        
        Returns:
            dict: Dictionary mapping metric names to average values
        """
        return {name: meter.avg for name, meter in self.meters.items()}
    
    def __str__(self):
        """String representation for logging."""
        return ' | '.join(str(meter) for meter in self.meters.values())


class Timer:
    """
    Simple timer for measuring execution time.
    
    Example:
        >>> timer = Timer()
        >>> timer.start()
        >>> # ... do work ...
        >>> elapsed = timer.stop()
        >>> print(f"Took {elapsed:.2f} seconds")
    """
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
    
    def stop(self):
        """
        Stop the timer and return elapsed time.
        
        Returns:
            float: Elapsed time in seconds
        """
        if self.start_time is None:
            return 0
        self.elapsed = time.time() - self.start_time
        return self.elapsed
    
    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.elapsed = 0


def compute_accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy.
    
    Args:
        output (torch.Tensor): Model predictions [batch_size, num_classes]
        target (torch.Tensor): Ground truth labels [batch_size]
        topk (tuple): Which top-k accuracies to compute (e.g., (1, 5) for top-1 and top-5)
    
    Returns:
        list[float]: Accuracy values for each k in topk
    
    Example:
        >>> logits = torch.randn(32, 10)
        >>> targets = torch.randint(0, 10, (32,))
        >>> top1_acc, top5_acc = compute_accuracy(logits, targets, topk=(1, 5))
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # Get top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # Transpose: [maxk, batch_size]
        
        # Compare with targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        # Compute accuracy for each k
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


def format_time(seconds):
    """
    Format seconds into human-readable string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    
    Example:
        >>> format_time(3661.5)
        '1h 1m 1s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"


if __name__ == '__main__':
    # Test metrics utilities
    print("Testing metrics utilities...")
    
    # Test AverageMeter
    print("\n1. Testing AverageMeter:")
    meter = AverageMeter('Loss', ':.4f')
    for i in range(5):
        meter.update(1.0 / (i + 1), n=10)
    print(f"   {meter}")
    print(f"   Average: {meter.avg:.4f}")
    
    # Test MetricTracker
    print("\n2. Testing MetricTracker:")
    tracker = MetricTracker()
    for i in range(3):
        tracker.update('loss', 1.0 / (i + 1), n=10)
        tracker.update('acc', (i + 1) * 20, n=10)
    print(f"   {tracker}")
    print(f"   Summary: {tracker.summary()}")
    
    # Test Timer
    print("\n3. Testing Timer:")
    timer = Timer()
    timer.start()
    time.sleep(0.1)
    elapsed = timer.stop()
    print(f"   Elapsed: {elapsed:.3f}s")
    
    # Test accuracy computation
    print("\n4. Testing accuracy computation:")
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    top1_acc = compute_accuracy(logits, targets, topk=(1,))[0]
    print(f"   Top-1 accuracy: {top1_acc:.2f}%")
    
    # Test time formatting
    print("\n5. Testing time formatting:")
    print(f"   {format_time(45)} (should be ~45s)")
    print(f"   {format_time(125)} (should be ~2m 5s)")
    print(f"   {format_time(3661)} (should be ~1h 1m 1s)")
    
    print("\n✓ All metrics tests passed!")

