#!/usr/bin/env python3
"""CUDA Kernel Compilation Script

This script compiles the CUDA kernels for the MCTS project.
It's called by the main setup.py during installation.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path to import modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main compilation function"""
    print("🔧 Starting CUDA kernel compilation...")
    
    try:
        # Import the CUDA compilation module
        from mcts.gpu.cuda_compile import pre_compile_modular_kernels
        
        print("✓ Found CUDA compilation module")
        
        # Attempt to compile all kernels
        results = pre_compile_modular_kernels()
        success = len(results.get('compiled', {})) > 0
        
        if success:
            print("✅ CUDA kernel compilation completed successfully!")
            return 0
        else:
            print("⚠ CUDA kernel compilation completed with warnings")
            return 0  # Still return success since warnings are acceptable
            
    except ImportError as e:
        print(f"⚠ Could not import CUDA compilation module: {e}")
        print("This might be normal if CUDA is not available or alphazero_py is not built yet")
        return 0  # Don't fail installation for missing optional dependencies
        
    except Exception as e:
        print(f"⚠ CUDA kernel compilation failed: {e}")
        print("Installation will continue without CUDA acceleration")
        return 0  # Don't fail installation for CUDA compilation errors

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)