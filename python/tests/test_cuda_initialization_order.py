"""Test CUDA initialization order in multiprocessing"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)


def worker_check_cuda_before_import():
    """Check CUDA state before importing torch"""
    import os
    import sys
    
    print(f"[WORKER PRE-IMPORT] PID: {os.getpid()}", file=sys.stderr)
    print(f"[WORKER PRE-IMPORT] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}", file=sys.stderr)
    
    # Set environment BEFORE importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print(f"[WORKER PRE-IMPORT] Set CUDA_VISIBLE_DEVICES to empty", file=sys.stderr)
    
    # Now import torch
    import torch
    print(f"[WORKER PRE-IMPORT] Torch imported, CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    print(f"[WORKER PRE-IMPORT] CUDA device count: {torch.cuda.device_count()}", file=sys.stderr)
    
    return "Success - CUDA disabled before torch import"


def worker_check_cuda_after_import():
    """Check CUDA state after importing torch (problematic)"""
    import os
    import sys
    import torch  # Import torch BEFORE setting environment
    
    print(f"[WORKER POST-IMPORT] PID: {os.getpid()}", file=sys.stderr)
    print(f"[WORKER POST-IMPORT] Torch already imported", file=sys.stderr)
    print(f"[WORKER POST-IMPORT] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    print(f"[WORKER POST-IMPORT] CUDA device count: {torch.cuda.device_count()}", file=sys.stderr)
    
    # Try to set environment AFTER importing torch (too late!)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print(f"[WORKER POST-IMPORT] Set CUDA_VISIBLE_DEVICES to empty (too late!)", file=sys.stderr)
    
    # Check again
    print(f"[WORKER POST-IMPORT] CUDA still available: {torch.cuda.is_available()}", file=sys.stderr)
    
    return "CUDA might still be initialized"


def test_initialization_order():
    """Test different CUDA initialization orders"""
    print("=== Testing CUDA Initialization Order ===")
    
    # Test 1: Set environment before torch import
    print("\n--- Test 1: Environment before import ---")
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(worker_check_cuda_before_import)
            result = future.result(timeout=5)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Failed: {type(e).__name__}: {e}")
    
    # Test 2: Set environment after torch import
    print("\n--- Test 2: Environment after import ---")
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(worker_check_cuda_after_import)
            result = future.result(timeout=5)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Failed: {type(e).__name__}: {e}")


def worker_with_early_cuda_disable(data):
    """Worker that disables CUDA very early"""
    # This runs BEFORE any imports
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Now safe to import
    import sys
    import torch
    
    print(f"[EARLY_DISABLE] CUDA available: {torch.cuda.is_available()}", file=sys.stderr)
    print(f"[EARLY_DISABLE] Data type: {type(data)}", file=sys.stderr)
    
    # Check if data has CUDA tensors
    if isinstance(data, dict):
        for k, v in data.items():
            if torch.is_tensor(v):
                print(f"[EARLY_DISABLE] Tensor {k} device: {v.device}", file=sys.stderr)
    
    return "Early CUDA disable successful"


def test_with_tensor_data():
    """Test with actual tensor data"""
    import torch
    
    print("\n=== Testing with Tensor Data ===")
    
    # Create CPU tensors
    cpu_data = {
        'weight': torch.randn(10, 10),
        'bias': torch.randn(10)
    }
    
    print("Testing with CPU tensors...")
    with ProcessPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(worker_with_early_cuda_disable, cpu_data)
            result = future.result(timeout=5)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Failed: {type(e).__name__}: {e}")
    
    # Create CUDA tensors if available
    if torch.cuda.is_available():
        cuda_data = {
            'weight': torch.randn(10, 10).cuda(),
            'bias': torch.randn(10).cuda()
        }
        
        print("\nTesting with CUDA tensors (should fail)...")
        with ProcessPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(worker_with_early_cuda_disable, cuda_data)
                result = future.result(timeout=5)
                print(f"Unexpected success: {result}")
            except Exception as e:
                print(f"Expected failure: {type(e).__name__}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    test_initialization_order()
    test_with_tensor_data()