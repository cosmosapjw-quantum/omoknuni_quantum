"""
Unified Configuration System for Quantum MCTS
=============================================

This module provides a unified configuration system that standardizes all quantum
MCTS components with consistent interfaces, validation, and parameter management.

Features:
- Unified base configuration for all quantum components
- Automatic parameter validation and consistency checking
- Environment variable support for deployment configuration
- Configuration presets for different use cases
- Backward compatibility with existing component configs
- Type-safe parameter definitions with proper defaults
"""

import os
import json
import math
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, field, fields
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QuantumLevel(Enum):
    """Quantum enhancement levels"""
    CLASSICAL = "classical"      # No quantum effects
    MINIMAL = "minimal"          # Minimal quantum corrections
    SELECTIVE = "selective"      # Selective quantum application  
    FULL = "full"               # Full quantum features
    RESEARCH = "research"        # All experimental features

class PerformanceMode(Enum):
    """Performance optimization modes"""
    ACCURACY = "accuracy"        # Maximum accuracy, minimal optimizations
    BALANCED = "balanced"        # Balance accuracy and performance
    PERFORMANCE = "performance"  # Maximum performance, selective features
    ULTRA_FAST = "ultra_fast"   # Ultra-fast with minimal quantum

@dataclass
class UnifiedQuantumConfig:
    """
    Unified configuration for all quantum MCTS components
    
    This configuration unifies all quantum features under a single, consistent
    interface that can be used across all components while maintaining 
    backward compatibility.
    """
    
    # ========================================================================
    # Core Game Parameters
    # ========================================================================
    branching_factor: int = 30
    avg_game_length: int = 100
    
    # ========================================================================
    # Quantum Enhancement Level
    # ========================================================================
    quantum_level: QuantumLevel = QuantumLevel.SELECTIVE
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    
    # ========================================================================
    # v5.0 Physics Parameters (from docs/v5.0/new_quantum_mcts.md)
    # ========================================================================
    # Exploration and exploitation
    kappa: Optional[float] = None          # κ - exploration strength (stiffness)
    beta: float = 1.0                      # β - value weight (inverse temperature)
    
    # Quantum annealing schedule  
    hbar_0: float = 0.1                    # ℏ_0 - base Planck scale
    alpha: float = 0.5                     # α - annealing exponent for ℏ_eff
    
    # Alternative legacy parameters (for backward compatibility)
    c_puct: Optional[float] = None         # Legacy PUCT constant
    coupling_strength: float = 0.3         # Legacy quantum coupling
    
    # ========================================================================
    # Regime Detection and Adaptation
    # ========================================================================
    # Critical transition points
    critical_point_1: int = 1000          # Quantum -> Critical
    critical_point_2: int = 5000          # Critical -> Classical
    
    # Selective application thresholds
    exploration_visit_threshold: int = 10   # Apply quantum only below this visit count
    quantum_phase_threshold: int = 5000     # Apply quantum only below this simulation count
    
    # Adaptive behavior
    enable_adaptive_mode: bool = True       # Adaptive quantum/classical switching
    adaptive_update_interval: int = 100     # Update adaptation every N simulations
    
    # ========================================================================
    # Performance and Hardware Configuration
    # ========================================================================
    device: str = 'cuda'
    use_mixed_precision: bool = True
    enable_cuda_kernels: bool = True
    cuda_compile_timeout: int = 60         # CUDA compilation timeout (seconds)
    
    # Batch processing
    max_batch_size: int = 4096
    optimal_wave_size: int = 3072          # Optimal for GPU utilization
    enable_batch_processing: bool = True
    
    # Memory management
    use_tensor_pooling: bool = True
    tensor_pool_size: int = 1000
    enable_memory_optimization: bool = True
    
    # ========================================================================
    # State Management
    # ========================================================================
    enable_coherent_state_management: bool = True
    enable_causality_preservation: bool = True
    max_state_snapshots: int = 100
    state_snapshot_interval: int = 10
    enable_thread_safety: bool = True
    
    # ========================================================================
    # Feature Toggles
    # ========================================================================
    # Core quantum features
    enable_quantum_corrections: bool = True
    enable_path_integral: bool = False      # Disabled for performance
    enable_lindblad_dynamics: bool = False  # Disabled for performance
    enable_wave_processing: bool = True
    enable_interference: bool = False       # Disabled for performance
    
    # Advanced features (typically disabled for performance)
    enable_hamiltonian_dynamics: bool = False
    enable_rg_flow: bool = False
    enable_topological_suppression: bool = False
    
    # ========================================================================
    # Validation and Debugging
    # ========================================================================
    enable_mathematical_validation: bool = False
    enable_comprehensive_logging: bool = False
    log_quantum_statistics: bool = False
    enable_performance_monitoring: bool = True
    
    # Validation thresholds
    max_quantum_correction_ratio: float = 10.0  # Max quantum/classical ratio
    causality_violation_tolerance: int = 3      # Max causality violations
    
    # ========================================================================
    # Environment Integration
    # ========================================================================
    config_file: Optional[str] = None
    environment_prefix: str = "QUANTUM_MCTS"
    save_config_on_init: bool = False
    
    def __post_init__(self):
        """Post-initialization validation and parameter computation"""
        self._load_from_environment()
        self._compute_derived_parameters()
        self._validate_configuration()
        self._apply_performance_mode()
        
        if self.save_config_on_init:
            self.save_to_file("quantum_mcts_config.json")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        prefix = self.environment_prefix
        
        # Map environment variables to config fields
        env_mappings = {
            f"{prefix}_QUANTUM_LEVEL": ("quantum_level", lambda x: QuantumLevel(x)),
            f"{prefix}_PERFORMANCE_MODE": ("performance_mode", lambda x: PerformanceMode(x)),
            f"{prefix}_DEVICE": ("device", str),
            f"{prefix}_BRANCHING_FACTOR": ("branching_factor", int),
            f"{prefix}_HBAR_0": ("hbar_0", float),
            f"{prefix}_ALPHA": ("alpha", float),
            f"{prefix}_ENABLE_CUDA": ("enable_cuda_kernels", lambda x: x.lower() == 'true'),
        }
        
        for env_var, (field_name, converter) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    setattr(self, field_name, converter(env_value))
                    logger.debug(f"Set {field_name} = {env_value} from environment")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to set {field_name} from environment: {e}")
    
    def _compute_derived_parameters(self):
        """Compute derived parameters from base configuration"""
        # Compute κ if not provided
        if self.kappa is None:
            self.kappa = math.sqrt(2 * math.log(self.branching_factor))
        
        # Compute c_puct for backward compatibility
        if self.c_puct is None:
            self.c_puct = self.kappa
        
        # Adjust parameters based on quantum level
        if self.quantum_level == QuantumLevel.CLASSICAL:
            self.enable_quantum_corrections = False
            self.enable_coherent_state_management = False
            
        elif self.quantum_level == QuantumLevel.MINIMAL:
            self.hbar_0 *= 0.5  # Reduce quantum strength
            self.exploration_visit_threshold = 5
            
        elif self.quantum_level == QuantumLevel.RESEARCH:
            self.enable_path_integral = True
            self.enable_lindblad_dynamics = True
            self.enable_hamiltonian_dynamics = True
            self.enable_mathematical_validation = True
    
    def _validate_configuration(self):
        """Validate configuration parameters"""
        errors = []
        
        # Validate core parameters
        if self.branching_factor <= 0:
            errors.append("branching_factor must be positive")
        
        if self.avg_game_length <= 0:
            errors.append("avg_game_length must be positive")
        
        # Validate physics parameters
        if self.hbar_0 <= 0:
            errors.append("hbar_0 must be positive")
        
        if not 0 < self.alpha <= 1:
            errors.append("alpha must be in (0, 1]")
        
        if self.beta <= 0:
            errors.append("beta must be positive")
        
        # Validate thresholds
        if self.critical_point_1 >= self.critical_point_2:
            errors.append("critical_point_1 must be less than critical_point_2")
        
        # Validate performance parameters
        if self.max_batch_size <= 0:
            errors.append("max_batch_size must be positive")
        
        if errors:
            error_msg = f"Configuration validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _apply_performance_mode(self):
        """Apply performance mode optimizations"""
        if self.performance_mode == PerformanceMode.ULTRA_FAST:
            # Ultra-fast: minimal features for maximum speed
            self.quantum_level = QuantumLevel.MINIMAL
            self.enable_coherent_state_management = False
            self.enable_causality_preservation = False
            self.enable_performance_monitoring = False
            self.log_quantum_statistics = False
            self.exploration_visit_threshold = 5
            self.quantum_phase_threshold = 2000
            
        elif self.performance_mode == PerformanceMode.PERFORMANCE:
            # Performance: selective features
            self.enable_mathematical_validation = False
            self.enable_comprehensive_logging = False
            self.log_quantum_statistics = False
            
        elif self.performance_mode == PerformanceMode.ACCURACY:
            # Accuracy: enable validation and detailed logging
            self.enable_mathematical_validation = True
            self.enable_comprehensive_logging = True
            self.log_quantum_statistics = True
    
    def hbar_eff(self, N_tot: float) -> float:
        """Calculate ℏ_eff(N_tot) = ℏ_0 (1 + N_tot)^(-α/2)"""
        return self.hbar_0 * ((1.0 + N_tot) ** (-self.alpha * 0.5))
    
    def get_regime_factor(self, simulation_count: int) -> float:
        """Get quantum factor based on current regime"""
        if simulation_count < self.critical_point_1:
            return 1.0  # Full quantum
        elif simulation_count < self.critical_point_2:
            return 0.5  # Critical transition
        else:
            return 0.1 if self.quantum_level != QuantumLevel.CLASSICAL else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Enum):
                result[field.name] = value.value
            else:
                result[field.name] = value
        return result
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'UnifiedQuantumConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert enum strings back to enums
        if 'quantum_level' in data:
            data['quantum_level'] = QuantumLevel(data['quantum_level'])
        if 'performance_mode' in data:
            data['performance_mode'] = PerformanceMode(data['performance_mode'])
        
        return cls(**data)
    
    def create_component_config(self, component_type: str) -> Dict[str, Any]:
        """Create component-specific configuration"""
        base_config = {
            'device': self.device,
            'enable_quantum': self.quantum_level != QuantumLevel.CLASSICAL,
            'branching_factor': self.branching_factor,
            'avg_game_length': self.avg_game_length,
        }
        
        if component_type == "selective_quantum":
            return {
                **base_config,
                'kappa': self.kappa,
                'beta': self.beta,
                'hbar_0': self.hbar_0,
                'alpha': self.alpha,
                'exploration_visit_threshold': self.exploration_visit_threshold,
                'quantum_phase_threshold': self.quantum_phase_threshold,
                'enable_cuda_kernels': self.enable_cuda_kernels,
                'enable_coherent_state_management': self.enable_coherent_state_management,
                'enable_causality_preservation': self.enable_causality_preservation,
            }
        
        elif component_type == "state_manager":
            return {
                'critical_point_1': self.critical_point_1,
                'critical_point_2': self.critical_point_2,
                'enable_causality_preservation': self.enable_causality_preservation,
                'max_snapshots': self.max_state_snapshots,
                'snapshot_interval': self.state_snapshot_interval,
                'thread_safe': self.enable_thread_safety,
            }
        
        elif component_type == "legacy_v2":
            return {
                **base_config,
                'c_puct': self.c_puct,
                'coupling_strength': self.coupling_strength,
                'use_mixed_precision': self.use_mixed_precision,
                'fast_mode': self.performance_mode in [PerformanceMode.PERFORMANCE, PerformanceMode.ULTRA_FAST],
            }
        
        else:
            return base_config

# Configuration Presets
class ConfigPresets:
    """Pre-defined configuration presets for common use cases"""
    
    @staticmethod
    def development() -> UnifiedQuantumConfig:
        """Configuration for development and testing"""
        return UnifiedQuantumConfig(
            quantum_level=QuantumLevel.SELECTIVE,
            performance_mode=PerformanceMode.BALANCED,
            enable_mathematical_validation=True,
            enable_comprehensive_logging=True,
            device='cpu',  # Use CPU for development
            enable_cuda_kernels=False,
        )
    
    @staticmethod  
    def production_cpu() -> UnifiedQuantumConfig:
        """Production configuration for CPU deployment"""
        return UnifiedQuantumConfig(
            quantum_level=QuantumLevel.SELECTIVE,
            performance_mode=PerformanceMode.PERFORMANCE,
            device='cpu',
            enable_cuda_kernels=False,
            enable_mathematical_validation=False,
            enable_comprehensive_logging=False,
        )
    
    @staticmethod
    def production_gpu() -> UnifiedQuantumConfig:
        """Production configuration for GPU deployment"""
        return UnifiedQuantumConfig(
            quantum_level=QuantumLevel.SELECTIVE,
            performance_mode=PerformanceMode.PERFORMANCE,
            device='cuda',
            enable_cuda_kernels=True,
            max_batch_size=8192,
            optimal_wave_size=3072,
        )
    
    @staticmethod
    def ultra_performance() -> UnifiedQuantumConfig:
        """Ultra-high performance configuration"""
        return UnifiedQuantumConfig(
            quantum_level=QuantumLevel.MINIMAL,
            performance_mode=PerformanceMode.ULTRA_FAST,
            device='cuda',
            enable_cuda_kernels=True,
            exploration_visit_threshold=5,
            quantum_phase_threshold=2000,
            enable_coherent_state_management=False,
            enable_causality_preservation=False,
        )
    
    @staticmethod
    def research() -> UnifiedQuantumConfig:
        """Configuration for research with all features enabled"""
        return UnifiedQuantumConfig(
            quantum_level=QuantumLevel.RESEARCH,
            performance_mode=PerformanceMode.ACCURACY,
            enable_path_integral=True,
            enable_lindblad_dynamics=True,
            enable_hamiltonian_dynamics=True,
            enable_mathematical_validation=True,
            enable_comprehensive_logging=True,
            log_quantum_statistics=True,
        )

# Factory functions
def create_config(preset: str = "production_gpu", **overrides) -> UnifiedQuantumConfig:
    """Create configuration from preset with optional overrides"""
    if hasattr(ConfigPresets, preset):
        config = getattr(ConfigPresets, preset)()
    else:
        raise ValueError(f"Unknown preset: {preset}")
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    return config

def load_config(source: Union[str, Dict[str, Any], None] = None) -> UnifiedQuantumConfig:
    """Load configuration from various sources"""
    if source is None:
        return UnifiedQuantumConfig()
    elif isinstance(source, str):
        if os.path.exists(source):
            return UnifiedQuantumConfig.load_from_file(source)
        else:
            return create_config(source)
    elif isinstance(source, dict):
        return UnifiedQuantumConfig(**source)
    else:
        raise ValueError(f"Invalid config source: {type(source)}")

# Export main classes
__all__ = [
    'UnifiedQuantumConfig',
    'QuantumLevel',
    'PerformanceMode', 
    'ConfigPresets',
    'create_config',
    'load_config'
]