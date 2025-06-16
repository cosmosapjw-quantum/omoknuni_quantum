#!/bin/bash

# Script to run all tests for the quantum-enhanced MCTS codebase

# Set the Python path
export PYTHONPATH=/home/cosmo/omoknuni_quantum/python:$PYTHONPATH

# Change to project directory
cd /home/cosmo/omoknuni_quantum

# Run all tests with pytest
echo "Running all tests..."
python -m pytest python/tests/ -v --tb=short "$@"

# To run specific test files:
# ./run_tests.sh python/tests/test_mcts_core.py

# To run with coverage:
# ./run_tests.sh --cov=mcts --cov-report=html

# To run only tests matching a pattern:
# ./run_tests.sh -k "test_quantum"

# To stop on first failure:
# ./run_tests.sh -x

# To run in parallel (requires pytest-xdist):
# ./run_tests.sh -n auto