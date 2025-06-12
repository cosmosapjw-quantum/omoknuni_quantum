#!/usr/bin/env python3
"""
Quick diagnostic script to understand MCTS performance bottlenecks
"""

import torch
import time
import logging
from mcts.core.unified_mcts import UnifiedMCTS, UnifiedMCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.core.game_interface import GameInterface, GameType as InterfaceGameType

# Enable debug logging
logging.basicConfig(level=logging.INFO)

class SimpleEvaluator:
    """Ultra-simple evaluator for performance testing"""
    
    def __init__(self):
        self.device = torch.device('cuda')
        # Pre-allocate tensors to avoid allocation overhead
        self.max_batch = 10000
        self.dummy_policies = torch.ones(self.max_batch, 225, device=self.device) / 225
        self.dummy_values = torch.zeros(self.max_batch, 1, device=self.device)
        
    def evaluate_batch(self, features, legal_masks=None):
        """Instant evaluation with pre-allocated tensors"""
        if isinstance(features, torch.Tensor):
            batch_size = features.shape[0]
        elif isinstance(features, list):
            batch_size = len(features)
        else:
            batch_size = 1
        
        # Return pre-allocated slices - zero allocation cost
        return self.dummy_policies[:batch_size], self.dummy_values[:batch_size]


def profile_mcts_components():
    """Profile individual MCTS components"""
    print("🔍 Profiling MCTS Components")
    print("=" * 50)
    
    # Test 1: Simple wave-based search
    config = UnifiedMCTSConfig(
        num_simulations=1000,
        wave_size=1024,  # Start small
        c_puct=1.4,
        temperature=1.0,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_virtual_loss=False,  # Disable for initial test
    )
    
    evaluator = SimpleEvaluator()
    mcts = UnifiedMCTS(config, evaluator)
    
    print(f"Configuration:")
    print(f"  Wave Size: {config.wave_size}")
    print(f"  Game States Capacity: {mcts.game_states.capacity:,}")
    print(f"  Device: {config.device}")
    print()
    
    # Create initial state
    game_interface = GameInterface(InterfaceGameType.GOMOKU, board_size=15)
    state = game_interface.create_initial_state()
    
    # Test different simulation counts
    test_sizes = [100, 500, 1000, 2000]
    
    for num_sims in test_sizes:
        print(f"🧪 Testing {num_sims} simulations:")
        
        # Clear memory
        torch.cuda.empty_cache()
        
        # Warmup
        if num_sims == test_sizes[0]:
            mcts.search(state, 50)
        
        # Time the search
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        policy = mcts.search(state, num_sims)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        sims_per_sec = num_sims / elapsed
        memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
        print(f"  Performance: {sims_per_sec:,.0f} sims/s")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Memory: {memory_mb:.1f}MB")
        print(f"  Policy sum: {policy.sum():.6f}")
        
        # Performance analysis
        if sims_per_sec > 20000:
            print(f"  ✅ GOOD performance")
        elif sims_per_sec > 10000:
            print(f"  ⚠️  MODERATE performance")
        else:
            print(f"  ❌ POOR performance")
        print()


def test_wave_size_scaling():
    """Test how performance scales with wave size"""
    print("📊 Wave Size Scaling Test")
    print("=" * 50)
    
    evaluator = SimpleEvaluator()
    game_interface = GameInterface(InterfaceGameType.GOMOKU, board_size=15)
    state = game_interface.create_initial_state()
    
    wave_sizes = [256, 512, 1024, 2048, 3072, 4096]
    num_simulations = 2000
    
    for wave_size in wave_sizes:
        print(f"🔧 Testing wave_size={wave_size}:")
        
        config = UnifiedMCTSConfig(
            num_simulations=num_simulations,
            wave_size=wave_size,
            c_puct=1.4,
            temperature=1.0,
            device='cuda',
            game_type=GameType.GOMOKU,
            board_size=15,
            enable_virtual_loss=False,
        )
        
        try:
            mcts = UnifiedMCTS(config, evaluator)
            
            # Warmup
            mcts.search(state, 100)
            
            # Performance test
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            policy = mcts.search(state, num_simulations)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            
            sims_per_sec = num_simulations / elapsed
            memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
            print(f"  Performance: {sims_per_sec:,.0f} sims/s")
            print(f"  Memory: {memory_mb:.1f}MB")
            print(f"  Time: {elapsed:.3f}s")
            
            # Clear for next test
            del mcts
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        print()


def test_virtual_loss_impact():
    """Test impact of virtual loss on performance"""
    print("🎯 Virtual Loss Impact Test")
    print("=" * 50)
    
    evaluator = SimpleEvaluator()
    game_interface = GameInterface(InterfaceGameType.GOMOKU, board_size=15)
    state = game_interface.create_initial_state()
    
    configs = [
        ("Without Virtual Loss", False),
        ("With Virtual Loss", True)
    ]
    
    for name, enable_vl in configs:
        print(f"🧪 {name}:")
        
        config = UnifiedMCTSConfig(
            num_simulations=2000,
            wave_size=2048,
            c_puct=1.4,
            temperature=1.0,
            device='cuda',
            game_type=GameType.GOMOKU,
            board_size=15,
            enable_virtual_loss=enable_vl,
        )
        
        mcts = UnifiedMCTS(config, evaluator)
        
        # Performance test
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        policy = mcts.search(state, 2000)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        sims_per_sec = 2000 / elapsed
        
        print(f"  Performance: {sims_per_sec:,.0f} sims/s")
        print(f"  Time: {elapsed:.3f}s")
        
        del mcts
        torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    print("🚀 MCTS Performance Diagnostic")
    print("=" * 60)
    print()
    
    # Run all tests
    profile_mcts_components()
    test_wave_size_scaling()
    test_virtual_loss_impact()
    
    print("✅ Diagnostic complete!")