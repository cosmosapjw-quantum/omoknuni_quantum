#!/usr/bin/env python3
"""Test that configuration loading respects YAML values"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mcts.utils.config_system import AlphaZeroConfig

def test_yaml_override_preservation():
    """Test that YAML values are preserved during hardware adjustment"""
    print("Testing YAML override preservation...")
    
    config_path = "/home/cosmosapjw/omoknuni_quantum/configs/gomoku_improved_training.yaml"
    
    # Load config with hardware adjustment enabled
    config = AlphaZeroConfig.load(config_path, auto_adjust_hardware=True)
    
    print(f"Loaded config:")
    print(f"  max_tree_nodes: {config.mcts.max_tree_nodes}")
    print(f"  memory_pool_size_mb: {config.mcts.memory_pool_size_mb}")
    print(f"  min_wave_size: {config.mcts.min_wave_size}")
    print(f"  max_wave_size: {config.mcts.max_wave_size}")
    
    # Check if our YAML values are preserved
    expected_max_tree_nodes = 1500000
    expected_memory_pool_size = 6144
    expected_wave_size = 3072
    
    success = True
    
    if config.mcts.max_tree_nodes != expected_max_tree_nodes:
        print(f"❌ max_tree_nodes: expected {expected_max_tree_nodes}, got {config.mcts.max_tree_nodes}")
        success = False
    else:
        print(f"✅ max_tree_nodes correctly preserved: {config.mcts.max_tree_nodes}")
    
    if config.mcts.memory_pool_size_mb != expected_memory_pool_size:
        print(f"❌ memory_pool_size_mb: expected {expected_memory_pool_size}, got {config.mcts.memory_pool_size_mb}")
        success = False
    else:
        print(f"✅ memory_pool_size_mb correctly preserved: {config.mcts.memory_pool_size_mb}")
    
    if config.mcts.max_wave_size != expected_wave_size:
        print(f"❌ max_wave_size: expected {expected_wave_size}, got {config.mcts.max_wave_size}")
        success = False
    else:
        print(f"✅ max_wave_size correctly preserved: {config.mcts.max_wave_size}")
    
    return success

if __name__ == "__main__":
    success = test_yaml_override_preservation()
    if success:
        print("\n✅ All YAML values correctly preserved!")
        exit(0)
    else:
        print("\n❌ Some YAML values were overridden by hardware detection!")
        exit(1)