#!/bin/bash
# Quick script to run W&B sweep
# Usage: ./run_sweep.sh [sweep_id] [num_agents]

# Default values
SWEEP_CONFIG="experiments/configs/collective_v4_sweep.yaml"
PROJECT="collective_model_code"  # Update this to your W&B project name
ENTITY="morizk"  # Update this to your W&B username

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}W&B Sweep Runner${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if sweep_id is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}No sweep ID provided. Initializing new sweep...${NC}\n"
    
    # Initialize sweep
    echo -e "${GREEN}Initializing sweep from: ${SWEEP_CONFIG}${NC}"
    SWEEP_OUTPUT=$(wandb sweep ${SWEEP_CONFIG} 2>&1)
    echo "$SWEEP_OUTPUT"
    
    # Extract sweep ID from output
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'sweeps/\K[^\s]+' | head -1)
    
    if [ -z "$SWEEP_ID" ]; then
        echo -e "${YELLOW}Could not extract sweep ID. Please check the output above.${NC}"
        echo -e "${YELLOW}Or provide sweep ID manually: ./run_sweep.sh YOUR_SWEEP_ID${NC}"
        exit 1
    fi
    
    echo -e "\n${GREEN}✓ Sweep created with ID: ${SWEEP_ID}${NC}"
    echo -e "${GREEN}✓ View at: https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}${NC}\n"
else
    SWEEP_ID="$1"
    echo -e "${GREEN}Using provided sweep ID: ${SWEEP_ID}${NC}\n"
fi

# Number of agents (default: 1)
NUM_AGENTS=${2:-1}

echo -e "${BLUE}Starting ${NUM_AGENTS} agent(s)...${NC}\n"

# Run agents
for i in $(seq 1 $NUM_AGENTS); do
    if [ $NUM_AGENTS -eq 1 ]; then
        echo -e "${GREEN}Running agent (single GPU)...${NC}"
        wandb agent ${ENTITY}/${PROJECT}/${SWEEP_ID}
    else
        GPU_ID=$((i - 1))
        echo -e "${GREEN}Starting agent ${i}/${NUM_AGENTS} on GPU ${GPU_ID}...${NC}"
        CUDA_VISIBLE_DEVICES=${GPU_ID} wandb agent ${ENTITY}/${PROJECT}/${SWEEP_ID} &
        sleep 2  # Small delay between starting agents
    fi
done

if [ $NUM_AGENTS -gt 1 ]; then
    echo -e "\n${GREEN}✓ All ${NUM_AGENTS} agents started!${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop all agents${NC}"
    
    # Wait for all background processes
    wait
fi

