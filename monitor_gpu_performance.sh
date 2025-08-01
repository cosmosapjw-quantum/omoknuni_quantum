#!/bin/bash
# Script to monitor GPU performance during MCTS test

echo "GPU Performance Monitoring Script"
echo "================================="
echo ""
echo "This script will help you monitor GPU utilization while running MCTS"
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo "Instructions:"
echo "1. This window will show real-time GPU utilization"
echo "2. Open another terminal and run: python test_actual_gpu_mcts_performance.py"
echo "3. Watch the GPU utilization percentage and memory usage"
echo "4. Target: 80%+ GPU utilization"
echo ""
echo "Starting monitoring in 3 seconds..."
echo "Press Ctrl+C to stop"
echo ""
sleep 3

# Monitor GPU utilization, memory, and power
# -d: delay between updates (0.1 seconds)
# -s: select metrics to display
#   u: GPU utilization
#   m: memory utilization  
#   p: power usage
#   c: SM clock
#   e: memory controller utilization
#   t: temperature
nvidia-smi dmon -d 0.1 -s pumcet