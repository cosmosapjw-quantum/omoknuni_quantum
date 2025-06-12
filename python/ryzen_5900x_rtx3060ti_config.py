#!/usr/bin/env python3
"""
Optimized MCTS Configuration for Ryzen 9 5900X + RTX 3060 Ti

This configuration is specifically tuned for:
- AMD Ryzen 9 5900X (12 cores, 24 threads, 3.7-4.5 GHz)  
- RTX 3060 Ti (8GB VRAM, 4864 CUDA cores, 1410-1665 MHz)
- 64GB RAM
- 32GB shared VRAM

Performance target: 50k+ simulations/second
Memory usage: ~5-6GB VRAM (70-80% utilization)
"""

from dataclasses import dataclass
import torch
from mcts.core.mcts import MCTSConfig
from mcts.gpu.gpu_game_states import GameType, GPUGameStatesConfig


@dataclass
class OptimizedHardwareConfig:
    """Complete optimized configuration for the specific hardware"""
    
    # ============================================================================
    # MCTS CORE CONFIGURATION
    # ============================================================================
    
    @staticmethod
    def get_mcts_config(performance_level: str = "balanced") -> MCTSConfig:
        """Get optimized MCTS configuration
        
        Args:
            performance_level: "conservative", "balanced", or "maximum"
        """
        
        if performance_level == "conservative":
            # For multi-process training (8+ concurrent processes)
            return MCTSConfig(
                num_simulations=800,
                wave_size=1824,  # ~38 SMs * 3 waves * 16 threads
                c_puct=1.4,
                temperature=1.0,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                device='cuda',
                game_type=GameType.GOMOKU,
                board_size=15,
                enable_virtual_loss=True,
                virtual_loss_value=-1.0,
                # Memory settings
                max_tree_nodes=1_000_000,
                # Performance tuning
                min_wave_size=512,
                max_wave_size=2048,
                adaptive_wave_sizing=False  # Fixed wave size for consistency
            )
            
        elif performance_level == "balanced":
            # Recommended for most training scenarios
            return MCTSConfig(
                num_simulations=1600,
                wave_size=3648,  # Optimal for RTX 3060 Ti
                c_puct=1.4,
                temperature=1.0,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                device='cuda',
                game_type=GameType.GOMOKU,
                board_size=15,
                enable_virtual_loss=True,
                virtual_loss_value=-1.0,
                # Memory settings
                max_tree_nodes=2_000_000,
                # Performance tuning
                min_wave_size=1024,
                max_wave_size=4096,
                adaptive_wave_sizing=False
            )
            
        elif performance_level == "maximum":
            # For maximum single-process performance
            return MCTSConfig(
                num_simulations=3200,
                wave_size=4096,  # Maximum recommended for RTX 3060 Ti
                c_puct=1.4,
                temperature=1.0,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                device='cuda',
                game_type=GameType.GOMOKU,
                board_size=15,
                enable_virtual_loss=True,
                virtual_loss_value=-1.0,
                # Memory settings
                max_tree_nodes=3_000_000,
                # Performance tuning
                min_wave_size=2048,
                max_wave_size=4096,
                adaptive_wave_sizing=False
            )
        else:
            raise ValueError(f"Unknown performance level: {performance_level}")
    
    # ============================================================================
    # GPU GAME STATES CONFIGURATION
    # ============================================================================
    
    @staticmethod
    def get_gpu_states_config(performance_level: str = "balanced") -> GPUGameStatesConfig:
        """Get optimized GPU game states configuration"""
        
        capacity_map = {
            "conservative": 500_000,    # ~3GB usage
            "balanced": 1_000_000,      # ~5GB usage  
            "maximum": 1_500_000        # ~7GB usage
        }
        
        return GPUGameStatesConfig(
            capacity=capacity_map[performance_level],
            game_type=GameType.GOMOKU,
            board_size=15,
            device='cuda',
            dtype=torch.float32
        )
    
    # ============================================================================
    # SELF-PLAY CONFIGURATION
    # ============================================================================
    
    @staticmethod
    def get_selfplay_config(performance_level: str = "balanced") -> dict:
        """Get optimized self-play configuration"""
        
        if performance_level == "conservative":
            return {
                "num_games": 100,
                "num_processes": 8,  # Use 2/3 of CPU threads
                "batch_timeout": 0.01,
                "gpu_evaluator_batch_size": 256,
                "memory_pool_size_mb": 512,  # Per process
            }
            
        elif performance_level == "balanced":
            return {
                "num_games": 200,
                "num_processes": 12,  # Use 1/2 of CPU threads  
                "batch_timeout": 0.01,
                "gpu_evaluator_batch_size": 512,
                "memory_pool_size_mb": 1024,
            }
            
        elif performance_level == "maximum":
            return {
                "num_games": 50,
                "num_processes": 4,  # Single GPU process focused
                "batch_timeout": 0.005,
                "gpu_evaluator_batch_size": 1024,
                "memory_pool_size_mb": 2048,
            }
    
    # ============================================================================
    # TRAINING PIPELINE CONFIGURATION
    # ============================================================================
    
    @staticmethod
    def get_training_config(performance_level: str = "balanced") -> dict:
        """Get optimized training pipeline configuration"""
        
        base_config = {
            "device": "cuda",
            "mixed_precision": True,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            
            # Memory optimization
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            
            # Performance monitoring
            "log_interval": 100,
            "save_interval": 1000,
            "eval_interval": 5000,
        }
        
        if performance_level == "conservative":
            base_config.update({
                "batch_size": 64,
                "learning_rate": 0.001,
                "dataloader_workers": 4,
            })
            
        elif performance_level == "balanced":
            base_config.update({
                "batch_size": 128,
                "learning_rate": 0.002,
                "dataloader_workers": 8,
            })
            
        elif performance_level == "maximum":
            base_config.update({
                "batch_size": 256,
                "learning_rate": 0.003,
                "dataloader_workers": 12,
            })
        
        return base_config


# ============================================================================
# PERFORMANCE TARGETS AND EXPECTATIONS
# ============================================================================

PERFORMANCE_TARGETS = {
    "conservative": {
        "sims_per_sec": 25_000,
        "memory_usage_gb": 3.0,
        "parallel_processes": 8,
        "training_examples_per_hour": 12_000,
        "use_case": "Multi-process training with 8+ concurrent games"
    },
    
    "balanced": {
        "sims_per_sec": 50_000,
        "memory_usage_gb": 5.0,
        "parallel_processes": 4,
        "training_examples_per_hour": 20_000,
        "use_case": "Recommended for most training scenarios"
    },
    
    "maximum": {
        "sims_per_sec": 80_000,
        "memory_usage_gb": 7.0,
        "parallel_processes": 1,
        "training_examples_per_hour": 15_000,
        "use_case": "Maximum single-process performance"
    }
}


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def create_optimized_mcts(performance_level: str = "balanced"):
    """Create optimized MCTS instance for the hardware
    
    Example:
        mcts = create_optimized_mcts("balanced")
        policy = mcts.search(game_state, 1600)
    """
    from mcts.core.mcts import MCTS
    
    # Create evaluator (replace with your neural network)
    class OptimizedEvaluator:
        def __init__(self):
            import torch
            self.device = torch.device('cuda')
        def evaluate_batch(self, features, legal_masks=None):
            b = features.shape[0] if hasattr(features, 'shape') else len(features)
            import torch
            return torch.ones(b, 225, device=self.device)/225, torch.zeros(b, 1, device=self.device)
    
    evaluator = OptimizedEvaluator()
    config = OptimizedHardwareConfig.get_mcts_config(performance_level)
    
    return MCTS(config, evaluator)


def print_configuration_summary():
    """Print summary of all configurations"""
    print("🔧 Optimized MCTS Configurations for Ryzen 9 5900X + RTX 3060 Ti")
    print("=" * 80)
    
    for level in ["conservative", "balanced", "maximum"]:
        print(f"\n📊 {level.upper()} Configuration:")
        
        targets = PERFORMANCE_TARGETS[level]
        mcts_config = OptimizedHardwareConfig.get_mcts_config(level)
        
        print(f"   Target Performance: {targets['sims_per_sec']:,} sims/s")
        print(f"   Memory Usage: {targets['memory_usage_gb']:.1f}GB")
        print(f"   Wave Size: {mcts_config.wave_size:,}")
        print(f"   Simulations: {mcts_config.num_simulations:,}")
        print(f"   Use Case: {targets['use_case']}")
        print(f"   Training Examples/Hour: {targets['training_examples_per_hour']:,}")


if __name__ == "__main__":
    import torch
    print_configuration_summary()
    
    print(f"\n🧪 Testing Balanced Configuration:")
    mcts = create_optimized_mcts("balanced")
    
    from mcts.core.game_interface import GameInterface, GameType as InterfaceGameType
    game_interface = GameInterface(InterfaceGameType.GOMOKU, board_size=15)
    state = game_interface.create_initial_state()
    
    # Performance test
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    policy = mcts.search(state, 1600)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    sims_per_sec = 1600 / elapsed
    memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    
    print(f"   ✅ Performance: {sims_per_sec:,.0f} sims/s")
    print(f"   ✅ Memory: {memory_mb:.1f}MB")
    print(f"   ✅ Policy sum: {policy.sum():.6f}")
    
    target = PERFORMANCE_TARGETS["balanced"]["sims_per_sec"]
    if sims_per_sec >= target * 0.8:
        print(f"   🎯 Target achieved! ({sims_per_sec/target:.1%} of target)")
    else:
        print(f"   ⚠️  Below target ({sims_per_sec/target:.1%} of target)")