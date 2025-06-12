#!/usr/bin/env python3
"""Test direct import of unified kernels"""

print("Testing direct import of kernels")
print("=" * 60)

# Try direct import
try:
    import mcts.gpu.unified_cuda_kernels as unified
    print("✓ Successfully imported unified_cuda_kernels")
    funcs = [attr for attr in dir(unified) if not attr.startswith('_')]
    print(f"\nAvailable functions ({len(funcs)}):")
    for f in sorted(funcs):
        print(f"  - {f}")
    
    # Check for quantum function
    has_quantum = hasattr(unified, 'batched_ucb_selection_quantum')
    print(f"\nHas batched_ucb_selection_quantum: {has_quantum}")
    
except Exception as e:
    print(f"✗ Failed to import: {e}")

# Also try the old module
print("\n" + "=" * 60)
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "old_kernels", 
        "mcts/gpu/mcts_cuda_kernels.cpython-312-x86_64-linux-gnu.so"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✓ Successfully loaded old mcts_cuda_kernels")
        funcs = [attr for attr in dir(module) if not attr.startswith('_')]
        print(f"\nFunctions in old module ({len(funcs)}):")
        for f in sorted(funcs):
            print(f"  - {f}")
except Exception as e:
    print(f"✗ Failed to load old module: {e}")