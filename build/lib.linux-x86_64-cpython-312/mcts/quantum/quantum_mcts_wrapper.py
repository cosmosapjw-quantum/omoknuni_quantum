"""
Quantum MCTS Wrapper - Migration Support for v1.0 to v2.0
========================================================

This module provides a unified wrapper that supports both v1.0 and v2.0
quantum MCTS implementations, enabling smooth migration between versions.

Key Features:
- Automatic version detection and selection
- Parameter mapping between v1 and v2 APIs
- Deprecation warnings for v1 usage
- Performance comparison utilities
"""

import warnings
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, List
from dataclasses import dataclass
import logging

from .quantum_features import QuantumMCTS, QuantumConfig
from .quantum_features_v2 import QuantumMCTSV2, QuantumConfigV2, MCTSPhase
from .qft_engine import create_qft_engine, create_qft_engine_v1, create_qft_engine_v2
from .path_integral import create_path_integral, create_path_integral_v1, create_path_integral_v2
from .decoherence import create_decoherence_engine, create_decoherence_engine_v1, create_decoherence_engine_v2

logger = logging.getLogger(__name__)


@dataclass
class UnifiedQuantumConfig:
    """Unified configuration supporting both v1 and v2 parameters"""
    # Version selection
    version: str = 'v2'  # 'v1' or 'v2' (default to v2)
    auto_detect_version: bool = True  # Auto-select based on parameters
    
    # Common parameters
    enable_quantum: bool = True
    quantum_level: str = 'tree_level'
    device: str = 'cuda'
    
    # v1.0 specific
    exploration_constant: Optional[float] = None  # Deprecated in v2
    
    # v2.0 specific
    branching_factor: Optional[int] = None
    avg_game_length: Optional[int] = None
    c_puct: Optional[float] = None
    use_neural_prior: bool = True
    enable_phase_adaptation: bool = True
    temperature_mode: str = 'annealing'
    envariance_threshold: float = 1e-3
    
    # Performance
    min_wave_size: int = 32
    optimal_wave_size: int = 3072
    use_mixed_precision: bool = True
    
    # Warnings
    suppress_deprecation_warnings: bool = False


class QuantumMCTSWrapper:
    """
    Unified wrapper for quantum MCTS supporting both v1 and v2
    
    This wrapper automatically detects and uses the appropriate version
    based on configuration and provides migration support.
    """
    
    def __init__(self, config: Optional[UnifiedQuantumConfig] = None):
        if config is None:
            config = UnifiedQuantumConfig()
        
        self.config = config
        self.version = self._determine_version()
        
        # Initialize the appropriate implementation
        if self.version == 'v1':
            self._show_v1_deprecation_warning()
            self.impl = self._create_v1_instance()
        else:
            self.impl = self._create_v2_instance()
        
        # Track usage statistics
        self.stats = {
            'version': self.version,
            'calls': 0,
            'phase_transitions': 0,
            'envariance_reached': False
        }
        
        logger.info(f"QuantumMCTSWrapper initialized with version {self.version}")
    
    def _determine_version(self) -> str:
        """Determine which version to use based on config"""
        if not self.config.auto_detect_version:
            return self.config.version
        
        # Auto-detect based on parameters
        has_v2_params = any([
            self.config.branching_factor is not None,
            self.config.avg_game_length is not None,
            self.config.enable_phase_adaptation,
            self.config.temperature_mode != 'fixed'
        ])
        
        has_v1_only_params = self.config.exploration_constant is not None
        
        # If only v1 params are set, use v1
        if has_v1_only_params and not has_v2_params:
            return 'v1'
        # If v2 params are set, use v2
        elif has_v2_params:
            return 'v2'
        else:
            # Default to v2 for new code
            return self.config.version
    
    def _show_v1_deprecation_warning(self):
        """Show deprecation warning for v1 usage"""
        if not self.config.suppress_deprecation_warnings:
            warnings.warn(
                "Quantum MCTS v1.0 is deprecated and will be removed in a future release. "
                "Please migrate to v2.0 for improved performance and features. "
                "Set version='v2' in your config or see docs/quantum-v1-to-v2-migration.md",
                DeprecationWarning,
                stacklevel=2
            )
    
    def _create_v1_instance(self) -> QuantumMCTS:
        """Create v1.0 quantum MCTS instance"""
        # Map exploration_constant to appropriate v1 parameters
        hbar_eff = 0.1  # Default
        if self.config.exploration_constant is not None:
            # Scale hbar_eff based on exploration constant
            # Higher exploration -> higher quantum effects
            hbar_eff = 0.1 * self.config.exploration_constant / 1.414
        
        v1_config = QuantumConfig(
            enable_quantum=self.config.enable_quantum,
            quantum_level=self.config.quantum_level,
            hbar_eff=hbar_eff,
            min_wave_size=self.config.min_wave_size,
            optimal_wave_size=self.config.optimal_wave_size,
            use_mixed_precision=self.config.use_mixed_precision,
            device=self.config.device
        )
        
        return QuantumMCTS(v1_config)
    
    def _create_v2_instance(self) -> QuantumMCTSV2:
        """Create v2.0 quantum MCTS instance"""
        # Ensure we have necessary v2 parameters
        branching_factor = self.config.branching_factor
        if branching_factor is None:
            # Estimate from other parameters or use default
            if self.config.exploration_constant is not None:
                # Reverse engineer from c_puct = sqrt(2 * log(b))
                # c^2 / 2 = log(b) => b = exp(c^2/2)
                c = self.config.exploration_constant
                branching_factor = int(np.exp(c * c / 2))
            else:
                branching_factor = 30  # Default
        
        # Compute c_puct if not provided
        c_puct = self.config.c_puct
        if c_puct is None:
            if self.config.exploration_constant is not None:
                c_puct = self.config.exploration_constant
            else:
                c_puct = np.sqrt(2 * np.log(branching_factor))
        
        # Map unified config to v2 config
        v2_config = QuantumConfigV2(
            enable_quantum=self.config.enable_quantum,
            quantum_level=self.config.quantum_level,
            branching_factor=branching_factor,
            avg_game_length=self.config.avg_game_length or 100,
            c_puct=c_puct,
            use_neural_prior=self.config.use_neural_prior,
            enable_phase_adaptation=self.config.enable_phase_adaptation,
            temperature_mode=self.config.temperature_mode,
            envariance_threshold=self.config.envariance_threshold,
            min_wave_size=self.config.min_wave_size,
            optimal_wave_size=self.config.optimal_wave_size,
            use_mixed_precision=self.config.use_mixed_precision,
            device=self.config.device
        )
        
        return QuantumMCTSV2(v2_config)
    
    def apply_quantum_to_selection(
        self,
        q_values: torch.Tensor,
        visit_counts: torch.Tensor,
        priors: torch.Tensor,
        c_puct: Optional[float] = None,
        total_simulations: Optional[int] = None,
        parent_visit: Optional[int] = None,
        is_root: bool = False
    ) -> torch.Tensor:
        """
        Apply quantum corrections to MCTS selection (unified interface)
        
        This method supports both v1 and v2 signatures and automatically
        adapts based on the version being used.
        """
        self.stats['calls'] += 1
        
        if self.version == 'v1':
            # v1 doesn't use total_simulations or c_puct in the same way
            if c_puct is not None and hasattr(self.impl.config, 'hbar_eff'):
                # Update hbar_eff based on c_puct for v1
                self.impl.config.hbar_eff = 0.1 * c_puct / 1.414
            
            # v1 has different parameter names
            return self.impl.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=c_puct or 1.414,
                parent_visits=torch.tensor([parent_visit]) if parent_visit is not None else None
            )
        else:
            # v2 uses additional parameters
            # Only update simulation count if it changed significantly
            if total_simulations is not None and (
                not hasattr(self, '_last_sim_count') or 
                abs(total_simulations - self._last_sim_count) >= 100
            ):
                self.impl.update_simulation_count(total_simulations)
                self._last_sim_count = total_simulations
            
            # v2 optimized path - avoid tensor creation
            # Pre-allocate parent_visits tensor if needed
            if parent_visit is not None:
                if not hasattr(self, '_parent_visit_tensor'):
                    self._parent_visit_tensor = torch.zeros(1, device=q_values.device)
                self._parent_visit_tensor[0] = parent_visit
                parent_visits_tensor = self._parent_visit_tensor
            else:
                parent_visits_tensor = None
            
            return self.impl.apply_quantum_to_selection(
                q_values, visit_counts, priors,
                c_puct=c_puct,
                parent_visits=parent_visits_tensor,
                simulation_count=total_simulations
            )
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get current phase information (v2 only)"""
        if self.version == 'v2' and hasattr(self.impl, 'current_phase'):
            return {
                'current_phase': self.impl.current_phase.value,
                'phase_transitions': self.impl.stats.get('phase_transitions', 0),
                'simulation_count': self.impl.total_simulations
            }
        else:
            return {
                'current_phase': 'not_available',
                'phase_transitions': 0,
                'simulation_count': 0
            }
    
    def check_convergence(self, tree: Any = None) -> bool:
        """Check envariance convergence (v2 only)"""
        if self.version == 'v2' and hasattr(self.impl, 'check_envariance'):
            converged = self.impl.check_envariance(tree)
            if converged:
                self.stats['envariance_reached'] = True
            return converged
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from wrapper and implementation"""
        impl_stats = {}
        if hasattr(self.impl, 'get_statistics'):
            impl_stats = self.impl.get_statistics()
        elif hasattr(self.impl, 'stats'):
            impl_stats = dict(self.impl.stats)
        
        return {**self.stats, **impl_stats}
    
    def migrate_v1_to_v2_config(self, v1_config: QuantumConfig) -> QuantumConfigV2:
        """Helper to migrate v1 config to v2"""
        logger.info("Migrating v1 config to v2...")
        
        # Estimate branching factor if not provided
        branching_factor = self.config.branching_factor
        if branching_factor is None:
            # Common defaults
            branching_factor = 30  # Reasonable default
            logger.warning(f"No branching_factor specified, using default: {branching_factor}")
        
        # Map exploration constant to c_puct
        c_puct = v1_config.exploration_constant
        if c_puct is None:
            c_puct = np.sqrt(2 * np.log(branching_factor))
        
        v2_config = QuantumConfigV2(
            enable_quantum=v1_config.enable_quantum,
            quantum_level=v1_config.quantum_level,
            branching_factor=branching_factor,
            c_puct=c_puct,
            min_wave_size=v1_config.min_wave_size,
            optimal_wave_size=v1_config.optimal_wave_size,
            use_mixed_precision=v1_config.use_mixed_precision,
            device=v1_config.device
        )
        
        return v2_config


def create_quantum_mcts(
    enable_quantum: bool = True,
    version: Optional[str] = None,
    **kwargs
) -> Union[QuantumMCTS, QuantumMCTSV2, QuantumMCTSWrapper]:
    """
    Factory function to create quantum MCTS instance
    
    Args:
        enable_quantum: Whether to enable quantum features
        version: Force specific version ('v1', 'v2', or None for auto)
        **kwargs: Additional configuration parameters
        
    Returns:
        Quantum MCTS instance (v1, v2, or wrapper)
    """
    # Create unified config
    config_dict = {
        'enable_quantum': enable_quantum,
    }
    if version is not None:
        config_dict['version'] = version
        config_dict['auto_detect_version'] = False
    
    config_dict.update(kwargs)
    config = UnifiedQuantumConfig(**config_dict)
    
    # Return wrapper for maximum compatibility
    return QuantumMCTSWrapper(config)


def compare_versions(
    q_values: torch.Tensor,
    visit_counts: torch.Tensor,
    priors: torch.Tensor,
    config: Optional[UnifiedQuantumConfig] = None
) -> Dict[str, Any]:
    """
    Compare v1 and v2 outputs for migration validation
    
    Returns dict with comparison results and performance metrics
    """
    if config is None:
        config = UnifiedQuantumConfig()
    
    # Create both versions
    v1_config = UnifiedQuantumConfig(**{**config.__dict__, 'version': 'v1', 'auto_detect_version': False})
    v2_config = UnifiedQuantumConfig(**{**config.__dict__, 'version': 'v2', 'auto_detect_version': False})
    
    v1_wrapper = QuantumMCTSWrapper(v1_config)
    v2_wrapper = QuantumMCTSWrapper(v2_config)
    
    # Apply quantum corrections
    import time
    
    t1_start = time.perf_counter()
    v1_output = v1_wrapper.apply_quantum_to_selection(q_values, visit_counts, priors)
    t1_end = time.perf_counter()
    
    t2_start = time.perf_counter()
    v2_output = v2_wrapper.apply_quantum_to_selection(q_values, visit_counts, priors)
    t2_end = time.perf_counter()
    
    # Compare outputs
    max_diff = torch.max(torch.abs(v1_output - v2_output)).item()
    mean_diff = torch.mean(torch.abs(v1_output - v2_output)).item()
    correlation = torch.corrcoef(torch.stack([v1_output.flatten(), v2_output.flatten()]))[0, 1].item()
    
    return {
        'max_difference': max_diff,
        'mean_difference': mean_diff,
        'correlation': correlation,
        'v1_time': t1_end - t1_start,
        'v2_time': t2_end - t2_start,
        'speedup': (t1_end - t1_start) / (t2_end - t2_start),
        'v1_stats': v1_wrapper.get_statistics(),
        'v2_stats': v2_wrapper.get_statistics()
    }