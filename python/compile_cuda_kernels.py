#!/usr/bin/env python3
"""
CUDA Kernel Compilation Script
=============================

This script compiles the v5.0 quantum CUDA kernels for optimal performance.
Run this before testing to ensure kernels are compiled with latest optimizations.
"""

import os
import sys
import torch
import subprocess
from pathlib import Path

def check_cuda_availability():
    """Check if CUDA is available and compatible"""
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA available with {device_count} device(s)")
    
    for i in range(device_count):
        device_props = torch.cuda.get_device_properties(i)
        compute_capability = torch.cuda.get_device_capability(i)
        print(f"  Device {i}: {device_props.name}")
        print(f"    Compute capability: {compute_capability[0]}.{compute_capability[1]}")
        print(f"    Memory: {device_props.total_memory / 1024**3:.1f} GB")
    
    return True

def compile_v5_kernels():
    """Compile v5.0 quantum CUDA kernels"""
    print("\nCompiling v5.0 Quantum CUDA Kernels...")
    print("-" * 50)
    
    try:
        # Change to python directory
        python_dir = Path(__file__).parent
        os.chdir(python_dir)
        
        # Clean previous builds
        print("🧹 Cleaning previous builds...")
        build_dirs = [
            "build",
            "dist", 
            "mcts.egg-info",
            "mcts/gpu/__pycache__"
        ]
        
        for build_dir in build_dirs:
            if os.path.exists(build_dir):
                import shutil
                shutil.rmtree(build_dir)
                print(f"  Removed {build_dir}")
        
        # Set CUDA architecture for current GPU
        capability = torch.cuda.get_device_capability()
        arch = f"{capability[0]}.{capability[1]}"
        os.environ['TORCH_CUDA_ARCH_LIST'] = arch
        print(f"🎯 Set CUDA architecture: {arch}")
        
        # Compile with setup.py
        print("🔨 Compiling CUDA extensions...")
        cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ CUDA kernels compiled successfully!")
            
            # Check if compiled modules are available
            try:
                from mcts.gpu import quantum_v5_cuda_kernels
                print("✅ v5.0 quantum kernels loaded successfully")
                return True
            except ImportError as e:
                print(f"⚠️  Compilation succeeded but import failed: {e}")
                return False
        else:
            print("❌ CUDA compilation failed!")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CUDA compilation timed out (> 5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False

def test_compiled_kernels():
    """Test that compiled kernels work correctly"""
    print("\nTesting Compiled Kernels...")
    print("-" * 30)
    
    try:
        device = torch.device('cuda')
        
        # Test data
        batch_size = 256
        q_values = torch.randn(batch_size, device=device)
        visit_counts = torch.randint(1, 20, (batch_size,), device=device, dtype=torch.float32)
        priors = torch.softmax(torch.randn(batch_size, device=device), dim=0)
        
        # Test v5.0 kernel
        from mcts.gpu import quantum_v5_cuda_kernels
        
        result = quantum_v5_cuda_kernels.selective_quantum_v5(
            q_values, visit_counts, priors,
            kappa=1.5, beta=1.0, hbar_0=0.1, alpha=0.5,
            parent_visits=1000.0, simulation_count=500
        )
        
        assert result.shape == q_values.shape, f"Shape mismatch: {result.shape} vs {q_values.shape}"
        assert torch.all(torch.isfinite(result)), "Non-finite values in CUDA result"
        
        print("✅ v5.0 CUDA kernels working correctly")
        
        # Performance benchmark
        num_iterations = 1000
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        for _ in range(10):
            quantum_v5_cuda_kernels.selective_quantum_v5(
                q_values, visit_counts, priors,
                1.5, 1.0, 0.1, 0.5, 1000.0, 500
            )
        
        torch.cuda.synchronize()
        start_time.record()
        
        for i in range(num_iterations):
            result = quantum_v5_cuda_kernels.selective_quantum_v5(
                q_values, visit_counts, priors,
                1.5, 1.0, 0.1, 0.5, 1000.0, 500 + i
            )
        
        end_time.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_time.elapsed_time(end_time)
        ops_per_sec = (num_iterations * batch_size) / (elapsed_ms / 1000)
        
        print(f"🚀 Performance: {ops_per_sec:.0f} operations/second")
        print(f"   ({elapsed_ms:.2f}ms for {num_iterations} iterations)")
        
        return True
        
    except Exception as e:
        print(f"❌ Kernel test failed: {e}")
        return False

def validate_installation():
    """Validate that the selective quantum implementation works"""
    print("\nValidating v5.0 Implementation...")
    print("-" * 35)
    
    try:
        from mcts.quantum.selective_quantum_optimized import (
            create_ultra_performance_quantum_mcts
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        quantum_mcts = create_ultra_performance_quantum_mcts(device=device)
        
        print(f"✅ SelectiveQuantumMCTS created on {device}")
        
        # Test basic functionality
        test_device = torch.device(device)
        q_values = torch.randn(100, device=test_device)
        visit_counts = torch.randint(1, 20, (100,), device=test_device, dtype=torch.float32)
        priors = torch.softmax(torch.randn(100, device=test_device), dim=0)
        
        result = quantum_mcts.apply_selective_quantum(
            q_values, visit_counts, priors,
            simulation_count=500
        )
        
        assert result.shape == q_values.shape
        assert torch.all(torch.isfinite(result))
        
        print("✅ Selective quantum processing working")
        
        # Check if CUDA kernels are being used
        stats = quantum_mcts.get_performance_stats()
        if stats.get('cuda_kernels_available', False):
            print("✅ CUDA kernels are available and active")
        else:
            print("⚠️  Using PyTorch fallback (CUDA kernels not active)")
        
        return True
        
    except Exception as e:
        print(f"❌ Implementation validation failed: {e}")
        return False

def main():
    """Main compilation and testing workflow"""
    print("CUDA Kernel Compilation & Optimization")
    print("=" * 60)
    
    # Check CUDA availability
    if not check_cuda_availability():
        print("\n❌ Cannot proceed without CUDA")
        return False
    
    # Compile kernels
    if not compile_v5_kernels():
        print("\n❌ Compilation failed")
        return False
    
    # Test compiled kernels
    if not test_compiled_kernels():
        print("\n❌ Kernel testing failed")
        return False
    
    # Validate implementation
    if not validate_installation():
        print("\n❌ Implementation validation failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 SUCCESS: All CUDA kernels compiled and validated!")
    print("✅ Ready to run performance tests and proceed with todo tasks")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)