#!/usr/bin/env python3
"""
Test script to validate optimized MCTS performance with proper capacity settings
"""

import torch
import time
import logging
from mcts.core.unified_mcts import UnifiedMCTS, UnifiedMCTSConfig
from mcts.gpu.gpu_game_states import GameType
from mcts.core.game_interface import GameInterface, GameType as InterfaceGameType

logging.basicConfig(level=logging.INFO)


class HighPerformanceEvaluator:
    """Optimized evaluator for maximum performance"""
    
    def __init__(self):
        self.device = torch.device('cuda')
        # Pre-allocate large tensors for zero-allocation performance
        max_batch = 10000
        self.policies = torch.ones(max_batch, 225, device=self.device) / 225
        self.values = torch.zeros(max_batch, 1, device=self.device)
        
    def evaluate_batch(self, features, legal_masks=None):
        """Ultra-fast evaluation with pre-allocated tensors"""
        if isinstance(features, torch.Tensor):
            batch_size = features.shape[0]
        elif isinstance(features, list):
            batch_size = len(features)
        else:
            batch_size = 1
        
        return self.policies[:batch_size], self.values[:batch_size]


def test_optimized_config():
    """Test the optimized configuration with proper capacity"""
    print("🚀 Testing Optimized MCTS Configuration")
    print("=" * 60)
    
    # Your optimal configuration based on hardware analysis
    config = UnifiedMCTSConfig(
        num_simulations=3200,
        wave_size=4096,  # Maximum for RTX 3060 Ti
        c_puct=1.4,
        temperature=1.0,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_virtual_loss=True,
        virtual_loss_value=-1.0,
        # Key: Increase capacities to match hardware
        min_wave_size=1024,
        max_wave_size=4096
    )
    
    print(f"Configuration:")
    print(f"  Wave Size: {config.wave_size:,}")
    print(f"  Simulations: {config.num_simulations:,}")
    print(f"  Device: {config.device}")
    print()
    
    # Create evaluator
    evaluator = HighPerformanceEvaluator()
    
    # Create MCTS with optimized settings
    mcts = UnifiedMCTS(config, evaluator)
    
    print(f"MCTS Created:")
    print(f"  Tree max nodes: {mcts.tree.max_nodes}")
    print(f"  Game states capacity: {mcts.game_states.capacity:,}")
    print()
    
    # Create game state
    game_interface = GameInterface(InterfaceGameType.GOMOKU, board_size=15)
    state = game_interface.create_initial_state()
    
    # Test different simulation counts
    test_configs = [
        (1000, "Warmup"),
        (3200, "Target Performance"),
        (6400, "Stress Test"),
        (10000, "Maximum Load")
    ]
    
    results = []
    
    for num_sims, test_name in test_configs:
        print(f"🧪 {test_name} ({num_sims:,} simulations):")
        
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Performance test
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            policy = mcts.search(state, num_sims)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            
            sims_per_sec = num_sims / elapsed
            memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Get MCTS statistics
            stats = mcts.get_statistics()
            
            result = {
                'name': test_name,
                'num_sims': num_sims,
                'elapsed_ms': elapsed * 1000,
                'sims_per_sec': sims_per_sec,
                'memory_mb': memory_mb,
                'tree_nodes': stats.get('tree_nodes', 0),
                'policy_sum': float(policy.sum())
            }
            results.append(result)
            
            print(f"  ⏱️  Time: {elapsed:.3f}s")
            print(f"  🎯 Performance: {sims_per_sec:,.0f} sims/s")
            print(f"  💾 Memory: {memory_mb:.1f}MB")
            print(f"  🌳 Tree nodes: {stats.get('tree_nodes', 0):,}")
            print(f"  📊 Policy sum: {policy.sum():.6f}")
            
            # Performance rating
            if sims_per_sec >= 80000:
                print(f"  ✅ EXCELLENT performance!")
            elif sims_per_sec >= 50000:
                print(f"  ✅ VERY GOOD performance!")
            elif sims_per_sec >= 25000:
                print(f"  ✅ Good performance")
            else:
                print(f"  ⚠️  Below target")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            print()
    
    # Summary
    if results:
        print("📊 Performance Summary:")
        print("=" * 60)
        
        best_result = max(results, key=lambda x: x['sims_per_sec'])
        
        print(f"🏆 Best Performance:")
        print(f"  Test: {best_result['name']}")
        print(f"  Performance: {best_result['sims_per_sec']:,.0f} sims/s")
        print(f"  Simulations: {best_result['num_sims']:,}")
        print(f"  Memory Usage: {best_result['memory_mb']:.1f}MB")
        print(f"  Tree Nodes: {best_result['tree_nodes']:,}")
        print()
        
        # Calculate training throughput
        # Assume 50 moves per game average
        moves_per_game = 50
        sims_per_move = best_result['num_sims']
        time_per_move = best_result['elapsed_ms'] / 1000
        
        games_per_hour = 3600 / (moves_per_game * time_per_move)
        examples_per_hour = games_per_hour * moves_per_game
        
        print(f"📈 Training Throughput Estimate:")
        print(f"  Games per hour: {games_per_hour:.0f}")
        print(f"  Training examples per hour: {examples_per_hour:.0f}")
        print()
        
        # Hardware utilization analysis
        max_memory_gb = 8.0  # RTX 3060 Ti VRAM
        memory_utilization = (best_result['memory_mb'] / 1024) / max_memory_gb * 100
        
        print(f"💾 Hardware Utilization:")
        print(f"  GPU Memory: {memory_utilization:.1f}% of {max_memory_gb}GB VRAM")
        
        if memory_utilization < 50:
            print(f"  💡 Can increase wave_size or simulations for better utilization")
        elif memory_utilization > 85:
            print(f"  ⚠️  High memory usage - monitor for OOM errors")
        else:
            print(f"  ✅ Good memory utilization")
        
        print()
        
        # Final recommendations
        print(f"💡 Recommendations for your hardware:")
        if best_result['sims_per_sec'] >= 50000:
            print(f"  ✅ Performance excellent for training")
            print(f"  ✅ Consider using this config for production")
        elif best_result['sims_per_sec'] >= 25000:
            print(f"  ✅ Performance good for training")
            print(f"  💡 Consider increasing wave_size if memory allows")
        else:
            print(f"  ⚠️  Performance below optimal")
            print(f"  💡 Check GPU utilization and memory settings")
        
        print(f"  💡 For multi-process training, reduce wave_size to ~2048")
        print(f"  💡 Monitor GPU temperature and clock speeds during training")


if __name__ == "__main__":
    test_optimized_config()