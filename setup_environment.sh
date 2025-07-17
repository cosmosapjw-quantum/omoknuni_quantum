#!/bin/bash
# Setup environment for MCTS physics analysis

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export library paths
export LD_LIBRARY_PATH="${SCRIPT_DIR}/python:${SCRIPT_DIR}/build_cpp/lib/Release:${LD_LIBRARY_PATH}"

# Activate virtual environment if not already activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    source ~/venv/bin/activate
fi

echo "Environment setup complete!"
echo "Library path: $LD_LIBRARY_PATH"
echo "You can now run: python run_mcts_physics_analysis.py --preset quick"