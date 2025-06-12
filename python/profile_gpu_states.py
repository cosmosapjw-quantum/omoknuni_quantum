#!/usr/bin/env python3
"""Profile GPU state operations to find bottlenecks"""

import torch
import time
from mcts.gpu.gpu_game_states import GPUGameStates, GPUGameStatesConfig, GameType


def time_operation(name, func, *args, **kwargs):
    """Time a GPU operation"""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed*1000:.2f}ms")
    return result


def profile_gpu_states():
    """Profile GPU state operations"""
    print("=== GPU State Operations Profile ===\n")
    
    config = GPUGameStatesConfig(
        capacity=10000,
        game_type=GameType.GOMOKU,
        board_size=15,
        device='cuda'
    )
    
    states = GPUGameStates(config)
    
    # Test 1: State allocation
    print("1. State allocation:")
    indices1 = time_operation("  Allocate 1 state", states.allocate_states, 1)
    indices10 = time_operation("  Allocate 10 states", states.allocate_states, 10)
    indices100 = time_operation("  Allocate 100 states", states.allocate_states, 100)
    
    # Test 2: State cloning
    print("\n2. State cloning:")
    parent_idx = torch.tensor([0], device='cuda')
    
    clones1 = time_operation("  Clone 1 state", 
                            states.clone_states, parent_idx, torch.tensor([1], device='cuda'))
    clones10 = time_operation("  Clone to 10 states", 
                             states.clone_states, parent_idx, torch.tensor([10], device='cuda'))
    clones50 = time_operation("  Clone to 50 states", 
                             states.clone_states, parent_idx, torch.tensor([50], device='cuda'))
    
    # Test 3: Legal moves generation
    print("\n3. Legal moves generation:")
    indices = torch.arange(10, device='cuda')
    legal_mask = time_operation("  Get legal moves for 10 states", 
                               states.get_legal_moves_mask, indices)
    
    # Test 4: Apply moves
    print("\n4. Apply moves:")
    actions = torch.arange(10, device='cuda')
    time_operation("  Apply 10 moves", states.apply_moves, indices, actions)
    
    # Test 5: Feature extraction
    print("\n5. Neural network features:")
    features = time_operation("  Get features for 10 states", 
                             states.get_nn_features, indices)
    
    # Test 6: Batch operations
    print("\n6. Large batch operations:")
    large_indices = torch.arange(100, device='cuda')
    
    # Clone 100 states each to 50 children
    parent_indices = torch.arange(10, device='cuda')
    num_clones = torch.full((10,), 50, device='cuda')
    
    time_operation("  Clone 10 parents to 50 children each (500 total)", 
                  states.clone_states, parent_indices, num_clones)
    
    # Get legal moves for 100 states
    time_operation("  Get legal moves for 100 states", 
                  states.get_legal_moves_mask, large_indices)


def profile_specific_bottleneck():
    """Profile the specific expansion bottleneck"""
    print("\n=== Expansion Bottleneck Profile ===\n")
    
    config = GPUGameStatesConfig(
        capacity=10000,
        game_type=GameType.GOMOKU,
        board_size=15,
        device='cuda'
    )
    
    states = GPUGameStates(config)
    
    # Simulate root expansion
    print("Simulating root expansion (50 children):")
    
    # 1. Allocate root
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    
    root_idx = time_operation("  1. Allocate root", states.allocate_states, 1)
    
    # 2. Get legal moves
    legal_mask = time_operation("  2. Get legal moves", 
                               states.get_legal_moves_mask, root_idx)
    
    # 3. Clone to 50 children
    num_children = 50
    child_indices = time_operation("  3. Clone to 50 children", 
                                  states.clone_states, 
                                  root_idx, 
                                  torch.tensor([num_children], device='cuda'))
    
    # 4. Apply moves
    actions = torch.arange(num_children, device='cuda')
    time_operation("  4. Apply 50 moves", states.apply_moves, child_indices, actions)
    
    # 5. Get features for evaluation
    time_operation("  5. Get features for 50 states", 
                  states.get_nn_features, child_indices)
    
    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - total_start
    print(f"\nTotal expansion time: {total_elapsed*1000:.2f}ms")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
        
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Warmup
    torch.cuda.empty_cache()
    dummy = torch.zeros(1000, 1000, device='cuda')
    del dummy
    
    profile_gpu_states()
    profile_specific_bottleneck()


if __name__ == "__main__":
    main()