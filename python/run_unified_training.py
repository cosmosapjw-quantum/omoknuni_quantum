#!/usr/bin/env python3
"""
Run the unified training pipeline from the command line.
This script ensures proper imports by being in the python directory.
"""

import sys
import os

# Ensure we're in the right directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the main function
from mcts.neural_networks.unified_training_pipeline import main

if __name__ == "__main__":
    # The main function doesn't exist yet, so we'll run the module directly
    import runpy
    runpy.run_module('mcts.neural_networks.unified_training_pipeline', run_name='__main__')