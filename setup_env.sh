#!/bin/bash

# Setup environment for running optimization

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Export library paths
export LD_LIBRARY_PATH="${SCRIPT_DIR}/python:${SCRIPT_DIR}/build_cpp/lib/Release:${LD_LIBRARY_PATH}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

echo "Environment setup complete:"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""

# Source this file to set environment in current shell:
# source setup_env.sh