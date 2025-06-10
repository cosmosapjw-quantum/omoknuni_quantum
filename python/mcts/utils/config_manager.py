"""Configuration manager with hardware auto-detection

This module provides automatic hardware detection and configuration
optimization for the MCTS engine.
"""

import multiprocessing
import platform
import psutil
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import subprocess

logger = logging.getLogger(__name__)

# Try to import torch for GPU detection
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


@dataclass
class HardwareInfo:
    """Hardware information"""
    cpu_count: int
    cpu_freq_mhz: float
    total_memory_gb: float
    available_memory_gb: float
    
    # GPU info
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_compute_capability: tuple = (0, 0)
    cuda_version: str = ""
    
    # System info
    os_name: str = ""
    python_version: str = ""
    

@dataclass  
class OptimizedConfig:
    """Optimized configuration based on hardware"""
    # MCTS settings
    num_simulations: int = 800
    batch_size: int = 256
    wave_size: int = 512
    num_threads: int = 1
    
    # Memory settings
    gpu_memory_limit_gb: float = 4.0
    cpu_memory_limit_gb: float = 8.0
    page_size: int = 1024
    enable_mixed_precision: bool = False
    
    # Neural network settings
    nn_batch_size: int = 256
    cache_size: int = 100000
    
    # Features
    use_gpu: bool = True
    enable_transpositions: bool = True
    enable_interference: bool = True
    enable_phase_kicks: bool = True
    
    # Performance
    target_simulations_per_second: int = 100000
    

class ConfigManager:
    """Manages configuration with hardware auto-detection"""
    
    def __init__(self):
        self.hardware_info = self._detect_hardware()
        self.optimized_config = self._optimize_config()
        
    def _detect_hardware(self) -> HardwareInfo:
        """Detect hardware capabilities"""
        info = HardwareInfo(
            cpu_count=multiprocessing.cpu_count(),
            cpu_freq_mhz=psutil.cpu_freq().current if psutil.cpu_freq() else 2000,
            total_memory_gb=psutil.virtual_memory().total / (1024**3),
            available_memory_gb=psutil.virtual_memory().available / (1024**3),
            os_name=platform.system(),
            python_version=platform.python_version()
        )
        
        # Detect GPU if torch is available
        if HAS_TORCH and torch.cuda.is_available():
            info.has_gpu = True
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Get compute capability
            major = torch.cuda.get_device_capability(0)[0]
            minor = torch.cuda.get_device_capability(0)[1]
            info.gpu_compute_capability = (major, minor)
            
            # Get CUDA version
            info.cuda_version = torch.version.cuda or "Unknown"
            
            logger.info(
                f"GPU detected: {info.gpu_name} with {info.gpu_memory_gb:.1f}GB memory, "
                f"compute capability {major}.{minor}, CUDA {info.cuda_version}"
            )
        else:
            logger.debug("No GPU detected or PyTorch not available")
            
        logger.info(
            f"System: {info.cpu_count} CPUs @ {info.cpu_freq_mhz:.0f}MHz, "
            f"{info.total_memory_gb:.1f}GB RAM ({info.available_memory_gb:.1f}GB available)"
        )
        
        return info
        
    def _optimize_config(self) -> OptimizedConfig:
        """Create optimized configuration based on hardware"""
        config = OptimizedConfig()
        hw = self.hardware_info
        
        # CPU settings
        # Use 75% of cores for MCTS, leave some for system
        config.num_threads = max(1, int(hw.cpu_count * 0.75))
        
        # Memory settings
        # Use 50% of available memory for MCTS
        available_for_mcts = hw.available_memory_gb * 0.5
        
        if hw.has_gpu:
            config.use_gpu = True
            # Allocate 80% of GPU memory
            config.gpu_memory_limit_gb = hw.gpu_memory_gb * 0.8
            # CPU memory for overflow
            config.cpu_memory_limit_gb = min(available_for_mcts * 0.3, 16.0)
            
            # Enable mixed precision for newer GPUs
            if hw.gpu_compute_capability >= (7, 0):  # Volta and newer
                config.enable_mixed_precision = True
                logger.info("Enabling mixed precision for modern GPU")
        else:
            config.use_gpu = False
            config.gpu_memory_limit_gb = 0
            # All memory for CPU
            config.cpu_memory_limit_gb = min(available_for_mcts, 32.0)
            
        # Batch and wave sizes based on memory
        # Updated based on optimization testing - larger waves utilize GPU better
        if hw.has_gpu:
            # Larger batches for GPU
            if hw.gpu_memory_gb >= 8:
                config.batch_size = 2048
                config.wave_size = 4096  # Increased from 1024
                config.nn_batch_size = 2048
            elif hw.gpu_memory_gb >= 6:
                config.batch_size = 1024
                config.wave_size = 2048  # Increased from 512
                config.nn_batch_size = 1024
            elif hw.gpu_memory_gb >= 4:
                config.batch_size = 512
                config.wave_size = 1024  # Increased from 512
                config.nn_batch_size = 512
            else:
                config.batch_size = 256
                config.wave_size = 512   # Increased from 256
                config.nn_batch_size = 256
        else:
            # Smaller batches for CPU
            config.batch_size = 64
            config.wave_size = 128
            config.nn_batch_size = 32
            
        # Page size based on wave size
        config.page_size = config.wave_size * 2
        
        # Simulation count based on hardware capability
        if hw.has_gpu and hw.gpu_memory_gb >= 8:
            config.num_simulations = 1600
            config.target_simulations_per_second = 150000
        elif hw.has_gpu:
            config.num_simulations = 800
            config.target_simulations_per_second = 80000
        else:
            config.num_simulations = 400
            config.target_simulations_per_second = 20000
            
        # Cache size based on available memory
        cache_memory_mb = min(available_for_mcts * 100, 1000)  # Max 1GB for cache
        config.cache_size = int(cache_memory_mb * 1024 * 1024 / 200)  # ~200 bytes per entry
        
        logger.info(
            f"Optimized config: {config.num_threads} threads, "
            f"wave_size={config.wave_size}, batch_size={config.batch_size}, "
            f"GPU: {config.use_gpu}, mixed_precision: {config.enable_mixed_precision}"
        )
        
        return config
        
    def get_mcts_config(self) -> Dict[str, Any]:
        """Get MCTS configuration dict"""
        return {
            'num_simulations': self.optimized_config.num_simulations,
            'batch_size': self.optimized_config.batch_size,
            'num_threads': self.optimized_config.num_threads,
            'reuse_tree': True,
            'use_wave_engine': self.optimized_config.use_gpu,
            'c_puct': 1.5,
            'temperature': 1.0,
            'add_noise_at_root': True,
            'dirichlet_alpha': 0.3,
            'dirichlet_epsilon': 0.25,
        }
        
    def get_arena_config(self) -> Dict[str, Any]:
        """Get TreeArena configuration dict"""
        return {
            'gpu_memory_limit': int(self.optimized_config.gpu_memory_limit_gb * 1024**3),
            'cpu_memory_limit': int(self.optimized_config.cpu_memory_limit_gb * 1024**3),
            'page_size': self.optimized_config.page_size,
            'enable_mixed_precision': self.optimized_config.enable_mixed_precision,
            'gc_threshold': 0.9,
        }
        
    def get_wave_config(self) -> Dict[str, Any]:
        """Get WaveEngine configuration dict"""
        return {
            'initial_wave_size': self.optimized_config.wave_size,
            'max_wave_size': self.optimized_config.wave_size * 2,
            'min_wave_size': 64,
            'enable_interference': self.optimized_config.enable_interference,
            'interference_threshold': 0.5,
            'enable_phase_kicks': self.optimized_config.enable_phase_kicks,
            'enable_adaptive_sizing': True,
        }
        
    def get_evaluator_config(self) -> Dict[str, Any]:
        """Get Evaluator configuration dict"""
        return {
            'batch_size': self.optimized_config.nn_batch_size,
            'cache_size': self.optimized_config.cache_size,
            'use_mixed_precision': self.optimized_config.enable_mixed_precision,
            'device': 'cuda' if self.optimized_config.use_gpu else 'cpu',
        }
        
    def get_hardware_summary(self) -> str:
        """Get human-readable hardware summary"""
        hw = self.hardware_info
        lines = [
            f"CPU: {hw.cpu_count} cores @ {hw.cpu_freq_mhz:.0f}MHz",
            f"RAM: {hw.total_memory_gb:.1f}GB total, {hw.available_memory_gb:.1f}GB available",
            f"OS: {hw.os_name}, Python {hw.python_version}",
        ]
        
        if hw.has_gpu:
            lines.extend([
                f"GPU: {hw.gpu_name}",
                f"GPU Memory: {hw.gpu_memory_gb:.1f}GB",
                f"Compute Capability: {hw.gpu_compute_capability[0]}.{hw.gpu_compute_capability[1]}",
                f"CUDA Version: {hw.cuda_version}",
            ])
        else:
            lines.append("GPU: Not available")
            
        return "\n".join(lines)
        
    def get_optimization_summary(self) -> str:
        """Get human-readable optimization summary"""
        cfg = self.optimized_config
        lines = [
            f"Threads: {cfg.num_threads}",
            f"Simulations: {cfg.num_simulations}",
            f"Wave Size: {cfg.wave_size}",
            f"Batch Size: {cfg.batch_size}",
            f"Target Speed: {cfg.target_simulations_per_second:,} sims/s",
            f"GPU: {'Enabled' if cfg.use_gpu else 'Disabled'}",
            f"Mixed Precision: {'Enabled' if cfg.enable_mixed_precision else 'Disabled'}",
            f"Memory Limits: GPU={cfg.gpu_memory_limit_gb:.1f}GB, CPU={cfg.cpu_memory_limit_gb:.1f}GB",
        ]
        
        return "\n".join(lines)


# Singleton instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get singleton config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager