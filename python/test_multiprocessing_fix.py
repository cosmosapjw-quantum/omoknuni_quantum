#!/usr/bin/env python3
"""Test script to verify CUDA multiprocessing fixes"""

import os
import sys
import torch
import multiprocessing as mp
from typing import Dict, Any

def test_worker_process(worker_id: int, result_queue: mp.Queue):
    """Test worker process to verify CUDA is properly disabled"""
    import os
    import sys
    
    # Log startup
    print(f"[TEST WORKER {worker_id}] Starting, PID: {os.getpid()}", file=sys.stderr)
    
    # Check CUDA_VISIBLE_DEVICES
    cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"[TEST WORKER {worker_id}] CUDA_VISIBLE_DEVICES: '{cuda_env}'", file=sys.stderr)
    
    # Import torch
    import torch
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    
    print(f"[TEST WORKER {worker_id}] CUDA available: {cuda_available}, device count: {cuda_device_count}", file=sys.stderr)
    
    # Try to create tensors
    try:
        cpu_tensor = torch.randn(10, 10)
        print(f"[TEST WORKER {worker_id}] Created CPU tensor successfully", file=sys.stderr)
        
        if cuda_available:
            print(f"[TEST WORKER {worker_id}] WARNING: CUDA is available in worker!", file=sys.stderr)
            # Try CUDA tensor (should fail or be avoided)
            try:
                cuda_tensor = torch.randn(10, 10, device='cuda')
                print(f"[TEST WORKER {worker_id}] ERROR: Created CUDA tensor in worker!", file=sys.stderr)
                result_queue.put({
                    'worker_id': worker_id,
                    'cuda_available': True,
                    'cuda_tensor_created': True,
                    'error': None
                })
            except Exception as e:
                print(f"[TEST WORKER {worker_id}] Good: CUDA tensor creation failed: {e}", file=sys.stderr)
                result_queue.put({
                    'worker_id': worker_id,
                    'cuda_available': True,
                    'cuda_tensor_created': False,
                    'error': str(e)
                })
        else:
            print(f"[TEST WORKER {worker_id}] Good: CUDA is not available in worker", file=sys.stderr)
            result_queue.put({
                'worker_id': worker_id,
                'cuda_available': False,
                'cuda_tensor_created': False,
                'error': None
            })
            
    except Exception as e:
        print(f"[TEST WORKER {worker_id}] Error: {e}", file=sys.stderr)
        result_queue.put({
            'worker_id': worker_id,
            'cuda_available': cuda_available,
            'cuda_tensor_created': False,
            'error': str(e)
        })


def test_worker_with_cuda_disabled(worker_id: int, result_queue: mp.Queue):
    """Test worker that properly disables CUDA before importing torch"""
    import os
    import sys
    
    # Log startup
    print(f"[FIXED WORKER {worker_id}] Starting, PID: {os.getpid()}", file=sys.stderr)
    
    # CRITICAL: Disable CUDA before importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print(f"[FIXED WORKER {worker_id}] Set CUDA_VISIBLE_DEVICES to empty", file=sys.stderr)
    
    # Now import torch
    import torch
    
    # Check CUDA availability (should be False)
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    
    print(f"[FIXED WORKER {worker_id}] CUDA available: {cuda_available}, device count: {cuda_device_count}", file=sys.stderr)
    
    # Create CPU tensor
    try:
        cpu_tensor = torch.randn(10, 10)
        print(f"[FIXED WORKER {worker_id}] Created CPU tensor successfully", file=sys.stderr)
        
        result_queue.put({
            'worker_id': worker_id,
            'cuda_available': cuda_available,
            'cuda_tensor_created': False,
            'error': None,
            'fixed': True
        })
    except Exception as e:
        print(f"[FIXED WORKER {worker_id}] Error: {e}", file=sys.stderr)
        result_queue.put({
            'worker_id': worker_id,
            'cuda_available': cuda_available,
            'cuda_tensor_created': False,
            'error': str(e),
            'fixed': True
        })


def main():
    """Run multiprocessing tests"""
    print("="*70)
    print("CUDA Multiprocessing Test")
    print("="*70)
    
    # Check main process CUDA
    print(f"\nMain process CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Main process GPU: {torch.cuda.get_device_name()}")
    
    # Set spawn method
    try:
        mp.set_start_method('spawn', force=True)
        print("\nSet multiprocessing start method to 'spawn'")
    except RuntimeError:
        print("\nMultiprocessing start method already set")
    
    print("\n" + "-"*70)
    print("Test 1: Worker processes WITHOUT CUDA disabled (problematic)")
    print("-"*70)
    
    # Test without CUDA disabled
    result_queue = mp.Queue()
    processes = []
    
    for i in range(2):
        p = mp.Process(target=test_worker_process, args=(i, result_queue))
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for p in processes:
        p.join()
        results.append(result_queue.get())
    
    print("\nResults:")
    for r in results:
        print(f"  Worker {r['worker_id']}: CUDA={r['cuda_available']}, "
              f"CUDA tensor created={r['cuda_tensor_created']}")
        if r['error']:
            print(f"    Error: {r['error']}")
    
    print("\n" + "-"*70)
    print("Test 2: Worker processes WITH CUDA disabled (fixed)")
    print("-"*70)
    
    # Test with CUDA disabled
    result_queue = mp.Queue()
    processes = []
    
    for i in range(2):
        p = mp.Process(target=test_worker_with_cuda_disabled, args=(i, result_queue))
        p.start()
        processes.append(p)
    
    # Collect results
    results = []
    for p in processes:
        p.join()
        results.append(result_queue.get())
    
    print("\nResults:")
    for r in results:
        print(f"  Worker {r['worker_id']}: CUDA={r['cuda_available']}, "
              f"CUDA tensor created={r['cuda_tensor_created']}")
        if r['error']:
            print(f"    Error: {r['error']}")
    
    print("\n" + "="*70)
    print("Summary:")
    print("- Workers should have CUDA disabled (cuda_available=False)")
    print("- This prevents CUDA multiprocessing errors")
    print("- Neural network evaluation happens in main process via GPU service")
    print("="*70)


if __name__ == "__main__":
    main()