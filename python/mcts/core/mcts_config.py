"""Centralized MCTS Configuration

This module provides the official configuration defaults that match the documentation.
Based on the vectorized MCTS guide and performance benchmarks.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch


@dataclass
class MCTSConfig:
    """Main MCTS configuration matching documentation defaults"""
    
    # Wave Processing (from docs: 256-2048 adaptive)
    wave_size: int = 1024  # Default wave size
    min_wave_size: int = 256  # Minimum for adaptive sizing
    max_wave_size: int = 2048  # Maximum for adaptive sizing
    adaptive_wave_sizing: bool = True
    
    # Tree Parameters
    c_puct: float = 1.0  # UCB exploration constant
    temperature: float = 1.0  # Action selection temperature
    num_simulations: int = 800  # Per move
    max_tree_size: int = 1_000_000  # Same as max_nodes
    
    # Memory Configuration (from docs)
    max_nodes: int = 1_000_000  # Maximum nodes in tree
    node_size_bytes: int = 347  # Achieved node size
    gpu_memory_fraction: float = 0.9  # Use 90% of GPU memory
    enable_paging: bool = True  # CPU-GPU paging
    
    # Quantum Features (from docs: core to algorithm)
    enable_interference: bool = True  # MinHash interference
    interference_strength: float = 0.15  # Interference coefficient
    enable_phase_kicks: bool = True  # Phase-kicked priors
    phase_kick_strength: float = 0.1  # Phase kick amplitude
    enable_path_integral: bool = True  # Path integral formulation
    
    # GPU Configuration
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    enable_gpu: bool = torch.cuda.is_available()  # GPU enabled
    use_mixed_precision: bool = True  # FP16/FP32
    use_cuda_graphs: bool = True  # CUDA graph capture
    use_custom_kernels: bool = True  # Custom CUDA kernels
    
    # Neural Network
    batch_size: int = 512  # NN evaluation batch size
    nn_cache_size: int = 100_000  # Position cache
    use_fp16: bool = True  # Mixed precision inference
    
    # Performance Targets (from docs)
    target_sims_per_sec: int = 50_000  # Minimum target
    achieved_sims_per_sec: int = 52_442  # Current achievement
    
    # Advanced Features
    enable_transpositions: bool = True  # Transposition table
    enable_dag_support: bool = True  # DAG structure for transpositions
    enable_state_delta: bool = True  # State delta encoding
    enable_evaluator_pool: bool = True  # Ensemble evaluation
    ensemble_size: int = 3  # Number of models in ensemble
    
    # Hybrid Execution
    enable_hybrid_mode: bool = True  # CPU-GPU hybrid
    cpu_threads: int = 4  # CPU worker threads
    gpu_batch_threshold: int = 32  # Min batch for GPU
    
    # Virtual Loss for Leaf Parallelization
    enable_virtual_loss: bool = True  # Use virtual loss for path diversity
    virtual_loss_value: float = -1.0  # Penalty value (negative for deterrent)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'wave_size': self.wave_size,
            'min_wave_size': self.min_wave_size,
            'max_wave_size': self.max_wave_size,
            'adaptive_wave_sizing': self.adaptive_wave_sizing,
            'c_puct': self.c_puct,
            'temperature': self.temperature,
            'num_simulations': self.num_simulations,
            'max_nodes': self.max_nodes,
            'node_size_bytes': self.node_size_bytes,
            'gpu_memory_fraction': self.gpu_memory_fraction,
            'enable_paging': self.enable_paging,
            'enable_interference': self.enable_interference,
            'interference_strength': self.interference_strength,
            'enable_phase_kicks': self.enable_phase_kicks,
            'phase_kick_strength': self.phase_kick_strength,
            'enable_path_integral': self.enable_path_integral,
            'device': self.device,
            'use_mixed_precision': self.use_mixed_precision,
            'use_cuda_graphs': self.use_cuda_graphs,
            'use_custom_kernels': self.use_custom_kernels,
            'batch_size': self.batch_size,
            'nn_cache_size': self.nn_cache_size,
            'use_fp16': self.use_fp16,
            'target_sims_per_sec': self.target_sims_per_sec,
            'achieved_sims_per_sec': self.achieved_sims_per_sec,
            'enable_transpositions': self.enable_transpositions,
            'enable_dag_support': self.enable_dag_support,
            'enable_state_delta': self.enable_state_delta,
            'enable_evaluator_pool': self.enable_evaluator_pool,
            'ensemble_size': self.ensemble_size,
            'enable_hybrid_mode': self.enable_hybrid_mode,
            'cpu_threads': self.cpu_threads,
            'gpu_batch_threshold': self.gpu_batch_threshold,
            'enable_virtual_loss': self.enable_virtual_loss,
            'virtual_loss_value': self.virtual_loss_value
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MCTSConfig':
        """Create from dictionary"""
        return cls(**config_dict)
    
    def get_memory_config(self) -> 'MemoryConfig':
        """Get memory configuration subset"""
        from mcts.core.tree_arena import MemoryConfig
        return MemoryConfig(
            max_nodes=self.max_nodes,
            node_size_bytes=self.node_size_bytes,
            gpu_memory_fraction=self.gpu_memory_fraction,
            enable_paging=self.enable_paging
        )


@dataclass
class HardwareTierConfig:
    """Hardware tier configurations from documentation"""
    
    @staticmethod
    def desktop_64gb() -> MCTSConfig:
        """Desktop with 64GB RAM configuration"""
        config = MCTSConfig()
        config.max_nodes = 10_000_000  # 10M nodes
        config.wave_size = 2048
        config.max_wave_size = 4096
        config.batch_size = 1024
        return config
    
    @staticmethod
    def laptop_16gb() -> MCTSConfig:
        """Laptop with 16GB RAM configuration"""
        config = MCTSConfig()
        config.max_nodes = 1_000_000  # 1M nodes
        config.wave_size = 512
        config.max_wave_size = 1024
        config.batch_size = 256
        config.cpu_threads = 2
        return config
    
    @staticmethod
    def cloud_a10() -> MCTSConfig:
        """Cloud A10 GPU configuration"""
        config = MCTSConfig()
        config.max_nodes = 50_000_000  # 50M nodes
        config.wave_size = 4096
        config.max_wave_size = 8192
        config.batch_size = 2048
        config.cpu_threads = 8
        return config


def get_optimal_config(hardware_tier: Optional[str] = None) -> MCTSConfig:
    """Get optimal configuration for hardware tier
    
    Args:
        hardware_tier: One of 'desktop-64gb', 'laptop-16gb', 'cloud-a10', or None for auto
        
    Returns:
        Optimal MCTSConfig for the hardware
    """
    if hardware_tier == 'desktop-64gb':
        return HardwareTierConfig.desktop_64gb()
    elif hardware_tier == 'laptop-16gb':
        return HardwareTierConfig.laptop_16gb()
    elif hardware_tier == 'cloud-a10':
        return HardwareTierConfig.cloud_a10()
    else:
        # Auto-detect based on available memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > 20e9:  # > 20GB
                return HardwareTierConfig.cloud_a10()
            elif gpu_memory > 8e9:  # > 8GB
                return HardwareTierConfig.desktop_64gb()
        
        # Default to standard config
        return MCTSConfig()


# Global config instance
_global_config: Optional[MCTSConfig] = None


def get_config() -> MCTSConfig:
    """Get global MCTS configuration"""
    global _global_config
    if _global_config is None:
        _global_config = get_optimal_config()
    return _global_config


def set_config(config: MCTSConfig):
    """Set global MCTS configuration"""
    global _global_config
    _global_config = config