#!/bin/bash
# Script to run training with GPU monitoring
# Usage: ./scripts/run_training_with_monitoring.sh [config] [model] [monitor_duration]

CONFIG=${1:-debug}
MODEL=${2:-collective}
MONITOR_DURATION=${3:-300}

echo "=========================================="
echo "Training + GPU Monitoring Script"
echo "=========================================="
echo "Config: $CONFIG"
echo "Model: $MODEL"
echo "Monitor Duration: ${MONITOR_DURATION}s"
echo ""
echo "Starting GPU monitoring in background..."
python scripts/monitor_gpu.py --duration $MONITOR_DURATION --interval 1 --output "gpu_stats_${CONFIG}_${MODEL}.csv" &
MONITOR_PID=$!
echo "Monitor PID: $MONITOR_PID"
echo ""
echo "Waiting 2 seconds for monitor to start..."
sleep 2
echo ""
echo "Starting training..."
python train.py --config $CONFIG --model $MODEL
TRAIN_EXIT_CODE=$?
echo ""
echo "Training finished with exit code: $TRAIN_EXIT_CODE"
echo ""
echo "Waiting for monitoring to finish..."
wait $MONITOR_PID
echo ""
echo "=========================================="
echo "Done! Check gpu_stats_${CONFIG}_${MODEL}.csv for results"
echo "=========================================="

