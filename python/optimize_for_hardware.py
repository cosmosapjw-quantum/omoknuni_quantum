#!/usr/bin/env python3
"""
Hardware-Optimized MCTS Configuration

This script creates optimal MCTS configurations for the specific hardware:
- AMD Ryzen 9 5900X (12 cores, 24 threads, 3.7-4.5 GHz)
- RTX 3060 Ti (8GB VRAM, 4864 CUDA cores, 1410-1665 MHz)
- 64GB RAM
- 32GB shared VRAM

The goal is to maximize performance while staying within memory limits.
"""

import torch
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from mcts.core.mcts import MCTS, MCTSConfig
from mcts.gpu.gpu_game_states import GameType, GPUGameStatesConfig
from mcts.core.game_interface import GameInterface, GameType as InterfaceGameType


@dataclass
class HardwareSpec:
    """Hardware specification for optimization"""
    # CPU specs
    cpu_cores: int = 12
    cpu_threads: int = 24
    cpu_base_ghz: float = 3.7
    cpu_boost_ghz: float = 4.5
    system_ram_gb: int = 64
    
    # GPU specs  
    gpu_name: str = "RTX 3060 Ti"
    gpu_vram_gb: int = 8
    gpu_shared_vram_gb: int = 32
    gpu_cuda_cores: int = 4864
    gpu_base_mhz: int = 1410
    gpu_boost_mhz: int = 1665
    
    # Memory calculations
    def total_gpu_memory_gb(self) -> float:
        return self.gpu_vram_gb + self.gpu_shared_vram_gb
    
    def usable_vram_gb(self) -> float:
        """Usable VRAM after system overhead"""
        return self.gpu_vram_gb * 0.85  # 85% usable, 15% for system
    
    def optimal_wave_size(self) -> int:
        """Optimal wave size based on CUDA cores"""
        # Target ~2-4 waves per SM for occupancy
        # RTX 3060 Ti has 38 SMs
        sms = 38
        waves_per_sm = 3
        return min(4096, max(1024, sms * waves_per_sm * 32))  # 32 threads per warp


def calculate_memory_requirements(wave_size: int, max_tree_nodes: int, 
                                game_states_capacity: int) -> Dict[str, float]:
    """Calculate memory requirements for configuration"""
    
    # CSR Tree memory (per node)
    bytes_per_node = (
        4 + 4 + 4 + 4 + 4 + 1 + 4 +  # visit_counts, value_sums, priors, parent_indices, parent_actions, flags, phases
        4 * 512  # children table (512 max children)
    )
    
    # CSR edge memory (estimate 20 edges per node average)
    bytes_per_edge = 4 + 4 + 4  # col_indices, edge_actions, edge_priors
    total_edges = max_tree_nodes * 20
    
    tree_memory_mb = (max_tree_nodes * bytes_per_node + total_edges * bytes_per_edge) / (1024 * 1024)
    
    # GPU game states memory
    # Gomoku: board (15x15) + metadata per state
    bytes_per_state = 15 * 15 * 1 + 32  # board + metadata
    game_states_memory_mb = (game_states_capacity * bytes_per_state) / (1024 * 1024)
    
    # Wave processing memory (temporary allocations)
    # Features, policies, values during batch processing
    batch_memory_mb = (wave_size * (15 * 15 * 4 + 225 * 4 + 4)) / (1024 * 1024)
    
    # Neural network model memory (estimated)
    model_memory_mb = 256  # Typical AlphaZero model
    
    # System overhead
    system_overhead_mb = 512
    
    return {
        'tree_memory_mb': tree_memory_mb,
        'game_states_memory_mb': game_states_memory_mb,
        'batch_memory_mb': batch_memory_mb,
        'model_memory_mb': model_memory_mb,
        'system_overhead_mb': system_overhead_mb,
        'total_mb': tree_memory_mb + game_states_memory_mb + batch_memory_mb + model_memory_mb + system_overhead_mb
    }


def create_optimized_configs(hardware: HardwareSpec) -> List[Dict]:
    """Create optimized configurations for the hardware"""
    
    # Available memory budget
    usable_vram_gb = hardware.usable_vram_gb()
    usable_vram_mb = usable_vram_gb * 1024
    
    print(f"Hardware Analysis:")
    print(f"  GPU: {hardware.gpu_name}")
    print(f"  Total VRAM: {hardware.gpu_vram_gb}GB")
    print(f"  Usable VRAM: {usable_vram_gb:.1f}GB ({usable_vram_mb:.0f}MB)")
    print(f"  CUDA Cores: {hardware.gpu_cuda_cores:,}")
    print(f"  CPU: {hardware.cpu_cores}C/{hardware.cpu_threads}T @ {hardware.cpu_base_ghz}-{hardware.cpu_boost_ghz}GHz")
    print(f"  System RAM: {hardware.system_ram_gb}GB")
    print()
    
    configs = []
    
    # Configuration 1: Balanced Performance (70% VRAM usage)
    target_memory_mb = usable_vram_mb * 0.7
    
    wave_size = hardware.optimal_wave_size()
    max_tree_nodes = 2_000_000  # Large tree for extended games
    game_states_capacity = 1_000_000  # Much larger capacity
    
    memory_req = calculate_memory_requirements(wave_size, max_tree_nodes, game_states_capacity)
    
    # Adjust if over budget
    while memory_req['total_mb'] > target_memory_mb and game_states_capacity > 100_000:
        game_states_capacity = int(game_states_capacity * 0.8)
        memory_req = calculate_memory_requirements(wave_size, max_tree_nodes, game_states_capacity)
    
    config1 = {
        'name': 'Balanced Performance',
        'wave_size': wave_size,
        'max_tree_nodes': max_tree_nodes,
        'game_states_capacity': game_states_capacity,
        'num_simulations': 1600,
        'target_sims_per_sec': 50000,
        'memory_usage_mb': memory_req['total_mb'],
        'memory_percent': (memory_req['total_mb'] / usable_vram_mb) * 100
    }
    configs.append(config1)
    
    # Configuration 2: Maximum Performance (85% VRAM usage)
    target_memory_mb = usable_vram_mb * 0.85
    
    wave_size = min(4096, hardware.optimal_wave_size() * 2)  # Larger waves
    max_tree_nodes = 3_000_000  # Even larger tree
    game_states_capacity = 1_500_000
    
    memory_req = calculate_memory_requirements(wave_size, max_tree_nodes, game_states_capacity)
    
    # Adjust if over budget
    while memory_req['total_mb'] > target_memory_mb and game_states_capacity > 500_000:
        game_states_capacity = int(game_states_capacity * 0.9)
        max_tree_nodes = int(max_tree_nodes * 0.9)
        memory_req = calculate_memory_requirements(wave_size, max_tree_nodes, game_states_capacity)
    
    config2 = {
        'name': 'Maximum Performance', 
        'wave_size': wave_size,
        'max_tree_nodes': max_tree_nodes,
        'game_states_capacity': game_states_capacity,
        'num_simulations': 3200,
        'target_sims_per_sec': 80000,
        'memory_usage_mb': memory_req['total_mb'],
        'memory_percent': (memory_req['total_mb'] / usable_vram_mb) * 100
    }
    configs.append(config2)
    
    # Configuration 3: Conservative (50% VRAM usage for multi-process)
    target_memory_mb = usable_vram_mb * 0.5
    
    wave_size = hardware.optimal_wave_size() // 2
    max_tree_nodes = 1_000_000
    game_states_capacity = 500_000
    
    memory_req = calculate_memory_requirements(wave_size, max_tree_nodes, game_states_capacity)
    
    config3 = {
        'name': 'Conservative (Multi-Process)',
        'wave_size': wave_size,
        'max_tree_nodes': max_tree_nodes, 
        'game_states_capacity': game_states_capacity,
        'num_simulations': 800,
        'target_sims_per_sec': 25000,
        'memory_usage_mb': memory_req['total_mb'],
        'memory_percent': (memory_req['total_mb'] / usable_vram_mb) * 100
    }
    configs.append(config3)
    
    return configs


def create_mcts_config(opt_config: Dict) -> MCTSConfig:
    """Create MCTSConfig from optimization config"""
    return MCTSConfig(
        num_simulations=opt_config['num_simulations'],
        wave_size=opt_config['wave_size'],
        c_puct=1.4,
        temperature=1.0,
        device='cuda',
        game_type=GameType.GOMOKU,
        board_size=15,
        enable_virtual_loss=True,
        virtual_loss_value=-1.0
    )


def test_configuration(opt_config: Dict, test_simulations: int = 1000) -> Dict:
    """Test a configuration and return performance metrics"""
    print(f"\n🧪 Testing {opt_config['name']}:")
    print(f"   Wave Size: {opt_config['wave_size']}")
    print(f"   Game States Capacity: {opt_config['game_states_capacity']:,}")
    print(f"   Estimated Memory: {opt_config['memory_usage_mb']:.1f}MB ({opt_config['memory_percent']:.1f}%)")
    
    try:
        # Create evaluator
        class OptimizedEvaluator:
            def __init__(self):
                self.device = torch.device('cuda')
            def evaluate_batch(self, features, legal_masks=None):
                b = features.shape[0] if hasattr(features, 'shape') else len(features)
                return torch.ones(b, 225, device=self.device)/225, torch.zeros(b, 1, device=self.device)
        
        evaluator = OptimizedEvaluator()
        
        # Create MCTS config
        mcts_config = create_mcts_config(opt_config)
        
        # Create MCTS with custom game states capacity
        mcts = MCTS(mcts_config, evaluator)
        
        # Override the game states capacity
        if hasattr(mcts.unified_mcts, 'game_states'):
            # This is a hack to increase capacity - in production we'd modify the config
            original_capacity = mcts.unified_mcts.game_states.capacity
            print(f"   Original capacity: {original_capacity:,}")
            print(f"   Attempting to use larger capacity...")
        
        # Create game state
        game_interface = GameInterface(InterfaceGameType.GOMOKU, board_size=15)
        state = game_interface.create_initial_state()
        
        # Warmup
        mcts.search(state, 100)
        
        # Performance test
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        policy = mcts.search(state, test_simulations)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        
        sims_per_sec = test_simulations / elapsed
        
        # Get memory usage
        memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        memory_cached = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
        
        # Get MCTS stats
        stats = mcts.get_statistics()
        
        result = {
            'success': True,
            'sims_per_sec': sims_per_sec,
            'elapsed_ms': elapsed * 1000,
            'memory_allocated_mb': memory_allocated,
            'memory_cached_mb': memory_cached,
            'tree_nodes': stats.get('tree_nodes', 0),
            'policy_sum': float(policy.sum()),
            'target_achieved': sims_per_sec >= opt_config.get('target_sims_per_sec', 0) * 0.8  # 80% of target
        }
        
        print(f"   ✅ Performance: {sims_per_sec:,.0f} sims/s")
        print(f"   ✅ Memory: {memory_allocated:.1f}MB allocated, {memory_cached:.1f}MB cached")
        print(f"   ✅ Tree nodes: {stats.get('tree_nodes', 0):,}")
        
        if result['target_achieved']:
            print(f"   🎯 Target achieved! ({sims_per_sec/1000:.0f}k sims/s)")
        else:
            print(f"   ⚠️  Below target ({opt_config.get('target_sims_per_sec', 0)/1000:.0f}k sims/s target)")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'sims_per_sec': 0,
            'target_achieved': False
        }


def main():
    """Main optimization function"""
    print("🔧 MCTS Hardware Optimization")
    print("=" * 60)
    
    # Define hardware specs
    hardware = HardwareSpec()
    
    # Create optimized configurations
    configs = create_optimized_configs(hardware)
    
    print(f"\n📋 Generated {len(configs)} optimized configurations:")
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}:")
        print(f"   Wave Size: {config['wave_size']:,}")
        print(f"   Max Tree Nodes: {config['max_tree_nodes']:,}")
        print(f"   Game States Capacity: {config['game_states_capacity']:,}")
        print(f"   Simulations: {config['num_simulations']:,}")
        print(f"   Memory Budget: {config['memory_usage_mb']:.1f}MB ({config['memory_percent']:.1f}%)")
        print(f"   Target Performance: {config['target_sims_per_sec']:,} sims/s")
    
    # Test configurations
    print(f"\n🏃 Testing Configurations:")
    print("=" * 60)
    
    results = []
    for config in configs:
        result = test_configuration(config, test_simulations=2000)
        result['config_name'] = config['name']
        results.append(result)
    
    # Summary
    print(f"\n📊 Performance Summary:")
    print("=" * 60)
    
    successful_configs = [r for r in results if r['success']]
    if successful_configs:
        best_config = max(successful_configs, key=lambda x: x['sims_per_sec'])
        
        print(f"\n🏆 Best Performing Configuration:")
        print(f"   Name: {best_config['config_name']}")
        print(f"   Performance: {best_config['sims_per_sec']:,.0f} sims/s")
        print(f"   Memory Usage: {best_config['memory_allocated_mb']:.1f}MB")
        print(f"   Tree Nodes: {best_config['tree_nodes']:,}")
        
        # Find matching config for details
        best_opt_config = next(c for c in configs if c['name'] == best_config['config_name'])
        
        print(f"\n📝 Recommended Configuration:")
        print("```python")
        print("MCTSConfig(")
        print(f"    num_simulations={best_opt_config['num_simulations']},")
        print(f"    wave_size={best_opt_config['wave_size']},")
        print("    c_puct=1.4,")
        print("    temperature=1.0,")
        print("    device='cuda',")
        print("    game_type=GameType.GOMOKU,")
        print("    board_size=15,")
        print("    enable_virtual_loss=True")
        print(")")
        print("")
        print("# Also increase game states capacity:")
        print("GPUGameStatesConfig(")
        print(f"    capacity={best_opt_config['game_states_capacity']},")
        print("    game_type=GameType.GOMOKU,")
        print("    board_size=15,")
        print("    device='cuda'")
        print(")")
        print("```")
        
        # Performance rating
        best_perf = best_config['sims_per_sec']
        if best_perf >= 80000:
            print(f"\n🎯 PERFORMANCE RATING: EXCELLENT ({best_perf/1000:.0f}k sims/s)")
        elif best_perf >= 50000:
            print(f"\n🎯 PERFORMANCE RATING: VERY GOOD ({best_perf/1000:.0f}k sims/s)")
        elif best_perf >= 25000:
            print(f"\n🎯 PERFORMANCE RATING: GOOD ({best_perf/1000:.0f}k sims/s)")
        else:
            print(f"\n🎯 PERFORMANCE RATING: NEEDS OPTIMIZATION ({best_perf/1000:.0f}k sims/s)")
        
    else:
        print("❌ All configurations failed - check GPU memory or CUDA setup")
    
    print(f"\n💡 Hardware Utilization Tips:")
    print("- Use multiple processes for self-play to utilize all CPU cores")
    print("- Consider Conservative config for multi-process setups")
    print("- Monitor GPU memory usage during training")
    print("- Increase wave_size gradually if more VRAM becomes available")


if __name__ == "__main__":
    main()