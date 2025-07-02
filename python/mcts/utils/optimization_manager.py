"""
Unified Optimization Manager for Classical and Quantum MCTS
==========================================================

This module provides a single, clean interface for managing both classical
and quantum optimizations in MCTS, ensuring fair performance comparison
and code maintainability.

The manager automatically selects the appropriate optimization strategy
based on the MCTS configuration and available resources.
"""

import torch
import logging
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """Optimization modes for MCTS"""
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    AUTO = "auto"  # Automatically select based on config


@dataclass
class OptimizationConfig:
    """Configuration for optimization manager"""
    mode: OptimizationMode = OptimizationMode.AUTO
    device: str = 'cuda'
    
    # Classical optimization settings
    classical_max_visits: int = 10000
    classical_enable_jit: bool = True
    classical_enable_triton: bool = True
    classical_max_batch_size: int = 4096
    classical_max_actions: int = 512
    
    # Quantum optimization settings  
    quantum_enable_lookup_tables: bool = True
    quantum_enable_jit: bool = True
    quantum_hbar_eff: float = 0.1
    quantum_phase_kick_strength: float = 0.1
    quantum_interference_alpha: float = 0.05
    
    # Performance settings
    enable_memory_pooling: bool = True
    enable_debug_logging: bool = False


class OptimizationManager:
    """
    Unified manager for classical and quantum MCTS optimizations
    
    This class provides a single interface for managing optimization strategies,
    ensuring fair performance comparison and eliminating code duplication.
    
    Features:
    - Automatic optimization strategy selection
    - Unified parameter interface
    - Performance monitoring and statistics
    - Graceful fallbacks for missing components
    """
    
    def __init__(self, config: OptimizationConfig, mcts_config: Any):
        """
        Initialize optimization manager
        
        Args:
            config: Optimization configuration
            mcts_config: MCTS configuration object
        """
        self.config = config
        self.mcts_config = mcts_config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Determine optimization mode
        self.optimization_mode = self._determine_optimization_mode()
        
        # Initialize performance statistics
        self.stats = {
            'optimization_mode': self.optimization_mode.value,
            'classical_calls': 0,
            'quantum_calls': 0,
            'fallback_calls': 0,
            'component_init_time': 0.0,
            'avg_ucb_time': 0.0
        }
        
        # Initialize optimization components
        self.classical_components = None
        self.quantum_components = None
        
        self._init_optimization_components()
        
        if config.enable_debug_logging:
            logger.debug(f"Optimization manager initialized: mode={self.optimization_mode.value}")
            logger.info(f"Device: {self.device}")
            
    def _determine_optimization_mode(self) -> OptimizationMode:
        """Determine the appropriate optimization mode based on configuration"""
        if self.config.mode != OptimizationMode.AUTO:
            return self.config.mode
            
        # Auto-determine based on MCTS config
        if hasattr(self.mcts_config, 'classical_only_mode') and self.mcts_config.classical_only_mode:
            return OptimizationMode.CLASSICAL
        elif hasattr(self.mcts_config, 'enable_quantum') and self.mcts_config.enable_quantum:
            return OptimizationMode.QUANTUM
        else:
            return OptimizationMode.CLASSICAL
    
    def _init_optimization_components(self):
        """Initialize the appropriate optimization components"""
        import time
        start_time = time.perf_counter()
        
        try:
            if self.optimization_mode in [OptimizationMode.CLASSICAL, OptimizationMode.AUTO]:
                self._init_classical_components()
                
            if self.optimization_mode in [OptimizationMode.QUANTUM, OptimizationMode.AUTO]:
                self._init_quantum_components()
                
        except Exception as e:
            logger.warning(f"Failed to initialize optimization components: {e}")
            # Set fallback mode
            self.optimization_mode = OptimizationMode.CLASSICAL
            if self.classical_components is None:
                self._init_classical_components()
        
        elapsed = time.perf_counter() - start_time
        self.stats['component_init_time'] = elapsed
        
        if self.config.enable_debug_logging:
            logger.info(f"Optimization components initialized in {elapsed:.3f}s")
    
    def _init_classical_components(self):
        """Initialize classical optimization components"""
        try:
            from ..gpu.classical_optimization_tables import (
                ClassicalOptimizationTables, ClassicalOptimizationConfig
            )
            from ..gpu.classical_memory_buffers import (
                ClassicalMemoryBuffers, ClassicalMemoryConfig
            )
            from ..gpu.classical_triton_kernels import (
                get_classical_triton_kernels
            )
            
            # Initialize lookup tables
            tables_config = ClassicalOptimizationConfig(
                max_visits=self.config.classical_max_visits,
                c_puct=getattr(self.mcts_config, 'c_puct', 1.414),
                device=self.config.device,
                enable_jit_compilation=self.config.classical_enable_jit
            )
            
            # Initialize memory buffers
            memory_config = ClassicalMemoryConfig(
                max_batch_size=self.config.classical_max_batch_size,
                max_actions_per_node=self.config.classical_max_actions,
                device=self.config.device,
                enable_memory_pooling=self.config.enable_memory_pooling
            )
            
            self.classical_components = {
                'optimization_tables': ClassicalOptimizationTables(tables_config),
                'memory_buffers': ClassicalMemoryBuffers(memory_config),
                'triton_kernels': None
            }
            
            # Initialize Triton kernels if enabled and on CUDA
            if self.config.classical_enable_triton and self.device.type == 'cuda':
                try:
                    self.classical_components['triton_kernels'] = get_classical_triton_kernels(self.device)
                except Exception as e:
                    logger.debug(f"Triton kernels initialization failed: {e}")
            
            if self.config.enable_debug_logging:
                logger.info("Classical optimization components initialized")
                
        except Exception as e:
            logger.warning(f"Failed to initialize classical components: {e}")
            self.classical_components = None
    
    def _init_quantum_components(self):
        """Initialize quantum optimization components"""
        try:
            # Import quantum components
            from ..quantum import QuantumConfig, QuantumMCTS, SearchPhase
            
            # Create quantum config if needed
            if hasattr(self.mcts_config, 'get_or_create_quantum_config'):
                quantum_config = self.mcts_config.get_or_create_quantum_config()
            else:
                # Create default quantum config
                from ..quantum import QuantumMode
                quantum_config = QuantumConfig(
                    quantum_mode=QuantumMode.PRAGMATIC,
                    base_c_puct=getattr(self.mcts_config, 'c_puct', 1.414),
                    device=self.config.device,
                    hbar_eff=self.config.quantum_hbar_eff,
                    phase_kick_strength=self.config.quantum_phase_kick_strength,
                    interference_alpha=self.config.quantum_interference_alpha
                )
            
            self.quantum_components = {
                'quantum_mcts': QuantumMCTS(quantum_config),
                'config': quantum_config,
                'total_simulations': 0,
                'current_phase': SearchPhase.EXPLORATION
            }
            
            if self.config.enable_debug_logging:
                logger.info("Quantum optimization components initialized")
                
        except Exception as e:
            logger.warning(f"Failed to initialize quantum components: {e}")
            self.quantum_components = None
    
    def get_ucb_parameters(self, active_nodes: torch.Tensor, children_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Get UCB parameters for the current optimization mode
        
        Args:
            active_nodes: Active nodes for UCB computation
            children_tensor: Children tensor for quantum phase computation
            
        Returns:
            Dictionary of parameters for UCB computation
        """
        if self.optimization_mode == OptimizationMode.CLASSICAL and self.classical_components:
            return self._get_classical_parameters(active_nodes, children_tensor)
        elif self.optimization_mode == OptimizationMode.QUANTUM and self.quantum_components:
            return self._get_quantum_parameters(active_nodes, children_tensor)
        else:
            # Fallback to disabled quantum parameters
            self.stats['fallback_calls'] += 1
            return self._get_fallback_parameters()
    
    def _get_classical_parameters(self, active_nodes: torch.Tensor, children_tensor: torch.Tensor) -> Dict[str, Any]:
        """Get classical optimization parameters"""
        self.stats['classical_calls'] += 1
        
        tables = self.classical_components['optimization_tables']
        buffers = self.classical_components['memory_buffers']
        
        return {
            'enable_classical_optimization': True,
            'classical_sqrt_table': tables.sqrt_table,
            'classical_exploration_table': tables.c_puct_factors,
            'classical_memory_buffers': buffers,
            'enable_quantum': False
        }
    
    def _get_quantum_parameters(self, active_nodes: torch.Tensor, children_tensor: torch.Tensor) -> Dict[str, Any]:
        """Get quantum optimization parameters"""
        self.stats['quantum_calls'] += 1
        
        quantum_mcts = self.quantum_components['quantum_mcts']
        total_sims = self.quantum_components['total_simulations']
        
        # Get quantum phases
        quantum_phases = self._get_quantum_phases(active_nodes, children_tensor)
        
        # Get quantum parameters based on version
        if hasattr(quantum_mcts, 'impl') and hasattr(quantum_mcts.impl, 'config'):
            # Wrapped implementation
            impl = quantum_mcts.impl
            
            if hasattr(impl, '_current_phase_config'):
                # v2.0 parameters
                phase_config = impl._current_phase_config
                
                # Compute hbar_eff using pre-computed factors
                if hasattr(impl, 'hbar_factors') and total_sims < len(impl.hbar_factors):
                    hbar_eff = (getattr(self.mcts_config, 'c_puct', 1.414) * 
                              impl.hbar_factors[total_sims] * 
                              phase_config['quantum_strength'])
                else:
                    hbar_eff = self.config.quantum_hbar_eff
                    
                return {
                    'quantum_phases': quantum_phases,
                    'uncertainty_table': getattr(impl, 'decoherence_table', torch.empty(0, device=self.device)),
                    'hbar_eff': hbar_eff,
                    'phase_kick_strength': phase_config.get('interference_strength', self.config.quantum_phase_kick_strength),
                    'interference_alpha': phase_config.get('interference_strength', self.config.quantum_interference_alpha) * 0.5,
                    'enable_quantum': True
                }
            else:
                # v1.0 parameters
                impl_config = impl.config
                return {
                    'quantum_phases': quantum_phases,
                    'uncertainty_table': getattr(impl, 'uncertainty_table', torch.empty(0, device=self.device)),
                    'hbar_eff': getattr(impl_config, 'hbar_eff', self.config.quantum_hbar_eff),
                    'phase_kick_strength': getattr(impl_config, 'phase_kick_strength', self.config.quantum_phase_kick_strength),
                    'interference_alpha': getattr(impl_config, 'interference_alpha', self.config.quantum_interference_alpha),
                    'enable_quantum': True
                }
        else:
            # Direct implementation
            if hasattr(quantum_mcts, '_current_phase_config'):
                # v2.0 direct
                phase_config = quantum_mcts._current_phase_config
                
                if hasattr(quantum_mcts, 'hbar_factors') and total_sims < len(quantum_mcts.hbar_factors):
                    hbar_eff = (getattr(self.mcts_config, 'c_puct', 1.414) * 
                              quantum_mcts.hbar_factors[total_sims] * 
                              phase_config['quantum_strength'])
                else:
                    hbar_eff = self.config.quantum_hbar_eff
                    
                return {
                    'quantum_phases': quantum_phases,
                    'uncertainty_table': getattr(quantum_mcts, 'decoherence_table', torch.empty(0, device=self.device)),
                    'hbar_eff': hbar_eff,
                    'phase_kick_strength': phase_config.get('interference_strength', self.config.quantum_phase_kick_strength),
                    'interference_alpha': phase_config.get('interference_strength', self.config.quantum_interference_alpha) * 0.5,
                    'enable_quantum': True
                }
            else:
                # v1.0 direct
                return {
                    'quantum_phases': quantum_phases,
                    'uncertainty_table': getattr(quantum_mcts, 'uncertainty_table', torch.empty(0, device=self.device)),
                    'hbar_eff': getattr(quantum_mcts.config, 'hbar_eff', self.config.quantum_hbar_eff),
                    'phase_kick_strength': getattr(quantum_mcts.config, 'phase_kick_strength', self.config.quantum_phase_kick_strength),
                    'interference_alpha': getattr(quantum_mcts.config, 'interference_alpha', self.config.quantum_interference_alpha),
                    'enable_quantum': True
                }
    
    def _get_fallback_parameters(self) -> Dict[str, Any]:
        """Get fallback parameters (disabled quantum)"""
        return {
            'enable_quantum': False,
            'quantum_phases': torch.empty(0, device=self.device),
            'uncertainty_table': torch.empty(0, device=self.device),
            'hbar_eff': 0.0,
            'phase_kick_strength': 0.0,
            'interference_alpha': 0.0
        }
    
    def _get_quantum_phases(self, active_nodes: torch.Tensor, children_tensor: torch.Tensor) -> torch.Tensor:
        """Get quantum phases for active nodes and children"""
        if self.quantum_components is None:
            return torch.empty(0, device=self.device)
            
        quantum_mcts = self.quantum_components['quantum_mcts']
        
        # Try to get phases from quantum implementation
        if hasattr(quantum_mcts, 'get_phases'):
            return quantum_mcts.get_phases(active_nodes, children_tensor)
        elif hasattr(quantum_mcts, 'impl') and hasattr(quantum_mcts.impl, 'get_phases'):
            return quantum_mcts.impl.get_phases(active_nodes, children_tensor)
        else:
            # Generate simple phases as fallback
            batch_size = len(active_nodes)
            max_children = children_tensor.shape[1] if len(children_tensor.shape) > 1 else children_tensor.shape[0]
            return torch.zeros((batch_size, max_children), device=self.device)
    
    def update_simulation_count(self, count: int):
        """Update simulation count for quantum phase tracking"""
        if self.quantum_components:
            self.quantum_components['total_simulations'] += count
    
    def get_classical_ucb_optimization(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, Any]]:
        """
        Get classical UCB optimization components
        
        Returns:
            Tuple of (sqrt_table, exploration_table, memory_buffers) or None
        """
        if self.classical_components is None:
            return None
            
        tables = self.classical_components['optimization_tables']
        buffers = self.classical_components['memory_buffers']
        
        return (tables.sqrt_table, tables.c_puct_factors, buffers)
    
    def get_quantum_features(self) -> Optional[Any]:
        """Get quantum features object for direct access"""
        if self.quantum_components is None:
            return None
        return self.quantum_components['quantum_mcts']
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        stats = self.stats.copy()
        
        # Add component-specific stats
        if self.classical_components:
            tables = self.classical_components['optimization_tables']
            buffers = self.classical_components['memory_buffers']
            
            stats['classical_table_stats'] = tables.get_stats()
            stats['classical_buffer_stats'] = buffers.get_stats()
            
            if self.classical_components['triton_kernels']:
                stats['classical_triton_stats'] = self.classical_components['triton_kernels'].get_stats()
        
        return stats
    
    def is_classical_mode(self) -> bool:
        """Check if using classical optimization mode"""
        return self.optimization_mode == OptimizationMode.CLASSICAL
    
    def is_quantum_mode(self) -> bool:
        """Check if using quantum optimization mode"""
        return self.optimization_mode == OptimizationMode.QUANTUM
    
    def has_classical_optimization(self) -> bool:
        """Check if classical optimization is available"""
        return self.classical_components is not None
    
    def has_quantum_optimization(self) -> bool:
        """Check if quantum optimization is available"""
        return self.quantum_components is not None


def create_optimization_manager(mcts_config: Any, config: Optional[OptimizationConfig] = None) -> OptimizationManager:
    """
    Factory function for creating optimization manager
    
    Args:
        mcts_config: MCTS configuration object
        config: Optional optimization configuration
        
    Returns:
        Configured optimization manager
    """
    if config is None:
        # Create default config based on MCTS config
        mode = OptimizationMode.AUTO
        if hasattr(mcts_config, 'classical_only_mode') and mcts_config.classical_only_mode:
            mode = OptimizationMode.CLASSICAL
        elif hasattr(mcts_config, 'enable_quantum') and mcts_config.enable_quantum:
            mode = OptimizationMode.QUANTUM
            
        config = OptimizationConfig(
            mode=mode,
            device=getattr(mcts_config, 'device', 'cuda'),
            classical_max_batch_size=getattr(mcts_config, 'max_wave_size', 4096),
            classical_max_actions=getattr(mcts_config, 'max_children_per_node', 512),
            enable_debug_logging=getattr(mcts_config, 'enable_debug_logging', False)
        )
    
    return OptimizationManager(config, mcts_config)


# Export main classes and functions
__all__ = [
    'OptimizationManager',
    'OptimizationConfig', 
    'OptimizationMode',
    'create_optimization_manager'
]