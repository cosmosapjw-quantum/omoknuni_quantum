#!/usr/bin/env python3
"""Check available CUDA kernel functions"""

import os

print("Checking available CUDA kernels")
print("=" * 60)

# Check compiled modules
print("\n1. Compiled modules:")
for f in os.listdir('mcts/gpu'):
    if f.endswith('.so'):
        print(f"   - {f}")

# Try to import and check functions
print("\n2. Unified CUDA kernels:")
try:
    import mcts.gpu.unified_cuda_kernels as unified
    funcs = [attr for attr in dir(unified) if not attr.startswith('_')]
    for f in funcs:
        print(f"   - {f}")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Quantum CUDA kernels:")
try:
    import mcts.gpu.quantum_cuda_kernels as quantum
    funcs = [attr for attr in dir(quantum) if not attr.startswith('_')]
    for f in funcs:
        print(f"   - {f}")
except Exception as e:
    print(f"   Error: {e}")

# Check the old mcts_cuda_kernels
print("\n4. Old MCTS CUDA kernels:")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mcts_cuda_kernels", 
        "mcts/gpu/mcts_cuda_kernels.cpython-312-x86_64-linux-gnu.so"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        funcs = [attr for attr in dir(module) if not attr.startswith('_')]
        for f in funcs:
            print(f"   - {f}")
except Exception as e:
    print(f"   Error: {e}")

# Check what unified_kernels.py is loading
print("\n5. What unified_kernels.py loads:")
from mcts.gpu.unified_kernels import _UNIFIED_KERNELS, _KERNELS_AVAILABLE
print(f"   _KERNELS_AVAILABLE: {_KERNELS_AVAILABLE}")
if _UNIFIED_KERNELS:
    print(f"   Module: {_UNIFIED_KERNELS}")
    funcs = [attr for attr in dir(_UNIFIED_KERNELS) if not attr.startswith('_') and callable(getattr(_UNIFIED_KERNELS, attr))]
    for f in funcs:
        print(f"   - {f}")