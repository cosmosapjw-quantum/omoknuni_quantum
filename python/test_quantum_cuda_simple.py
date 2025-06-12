#!/usr/bin/env python3
"""Simple test for quantum CUDA kernel compilation and availability"""

import torch
import os
import sys

def test_quantum_cuda_availability():
    """Test if quantum CUDA kernels can be loaded"""
    print("Testing Quantum CUDA Kernel Availability")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    
    print("\n1. Testing quantum CUDA extension loading:")
    try:
        from mcts.gpu.quantum_cuda_extension import load_quantum_cuda_kernels, _QUANTUM_CUDA_AVAILABLE
        loaded = load_quantum_cuda_kernels()
        print(f"   ✓ Quantum CUDA kernels loaded: {loaded}")
        print(f"   ✓ _QUANTUM_CUDA_AVAILABLE: {_QUANTUM_CUDA_AVAILABLE}")
        
        # Check if module is available
        from mcts.gpu import quantum_cuda_extension
        if hasattr(quantum_cuda_extension, '_QUANTUM_CUDA_MODULE'):
            module = quantum_cuda_extension._QUANTUM_CUDA_MODULE
            if module:
                print(f"   ✓ Module loaded: {module}")
                # List available functions
                funcs = [attr for attr in dir(module) if not attr.startswith('_')]
                print(f"   ✓ Available functions: {funcs}")
    except Exception as e:
        print(f"   ✗ Failed to load quantum CUDA extension: {e}")
    
    print("\n2. Testing unified kernels with quantum support:")
    try:
        from mcts.gpu.unified_kernels import get_unified_kernels
        kernels = get_unified_kernels(torch.device('cuda'))
        print(f"   ✓ Unified kernels loaded")
        print(f"   ✓ CUDA kernels available: {kernels.use_cuda}")
        
        # Check for quantum kernel method
        if hasattr(kernels, 'batch_ucb_selection'):
            print(f"   ✓ batch_ucb_selection available")
            # Check if it accepts quantum parameters
            import inspect
            sig = inspect.signature(kernels.batch_ucb_selection)
            quantum_params = [p for p in sig.parameters if 'quantum' in p.lower()]
            if quantum_params:
                print(f"   ✓ Quantum parameters supported: {quantum_params}")
    except Exception as e:
        print(f"   ✗ Failed to load unified kernels: {e}")
    
    print("\n3. Testing pre-compiled modules:")
    # Check for compiled modules
    module_paths = [
        "mcts/gpu/quantum_cuda_kernels.cpython-312-x86_64-linux-gnu.so",
        "mcts/gpu/unified_cuda_kernels.cpython-312-x86_64-linux-gnu.so",
        "build_cuda/unified_cuda_kernels.so",
    ]
    
    for path in module_paths:
        if os.path.exists(path):
            print(f"   ✓ Found: {path}")
        else:
            print(f"   ✗ Not found: {path}")
    
    print("\n4. Testing kernel compilation:")
    # Try to compile if not available
    if torch.cuda.is_available():
        try:
            # Check if setup.py has been run
            import subprocess
            result = subprocess.run(
                [sys.executable, "-c", "import mcts.gpu.quantum_cuda_kernels"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("   ✓ Pre-compiled quantum kernels importable")
            else:
                print("   ✗ Pre-compiled kernels not found")
                print(f"   ! Run: pip install -e . to compile CUDA extensions")
        except Exception as e:
            print(f"   ✗ Compilation check failed: {e}")
    
    print("\n5. Testing quantum kernel function directly:")
    if torch.cuda.is_available():
        try:
            # Try direct import
            import mcts.gpu.quantum_cuda_kernels as qk
            print("   ✓ Direct import successful")
            
            # Test basic function call
            batch_size = 10
            num_children = 20
            
            # Create test tensors with correct types
            q_values = torch.rand(batch_size * num_children, device='cuda', dtype=torch.float32)
            visit_counts = torch.randint(0, 100, (batch_size * num_children,), device='cuda', dtype=torch.int32)
            parent_visits = torch.randint(1, 1000, (batch_size,), device='cuda', dtype=torch.int32)
            priors = torch.rand(batch_size * num_children, device='cuda', dtype=torch.float32)
            row_ptr = torch.arange(0, (batch_size + 1) * num_children, num_children, device='cuda', dtype=torch.int32)
            col_indices = torch.arange(batch_size * num_children, device='cuda', dtype=torch.int32)
            
            # Empty quantum tensors
            quantum_phases = torch.empty(0, device='cuda')
            uncertainty_table = torch.empty(0, device='cuda')
            
            # Call function
            actions, scores = qk.batched_ucb_selection_quantum(
                q_values, visit_counts, parent_visits, priors,
                row_ptr, col_indices, 1.414,
                quantum_phases, uncertainty_table,
                0.05, 0.1, 0.05, True
            )
            
            print(f"   ✓ Quantum kernel call successful")
            print(f"   ✓ Output shapes: actions={actions.shape}, scores={scores.shape}")
            
        except ImportError:
            print("   ✗ Module not compiled - run: pip install -e .")
        except Exception as e:
            print(f"   ✗ Kernel test failed: {e}")

if __name__ == "__main__":
    test_quantum_cuda_availability()