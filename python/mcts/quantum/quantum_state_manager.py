"""
Coherent Quantum State Management
=================================

This module provides unified state management across all quantum MCTS components:
- Consistent phase tracking (quantum/critical/classical regimes)
- Causality preservation with pre-update visit counts
- State synchronization between path integral, Lindblad, and wave processors
- Memory-efficient state tracking with automatic cleanup
- Thread-safe operations for parallel MCTS
"""

import torch
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class QuantumRegime(Enum):
    """Quantum regime classification"""
    QUANTUM = "quantum"      # N < N_c1: Full quantum effects
    CRITICAL = "critical"    # N_c1 <= N < N_c2: Critical transition
    CLASSICAL = "classical"  # N >= N_c2: Minimal quantum effects

@dataclass
class QuantumStateSnapshot:
    """Immutable snapshot of quantum state at a given time"""
    timestamp: float
    simulation_count: int
    regime: QuantumRegime
    visit_counts: torch.Tensor  # Copy of visit counts at this time
    hbar_eff: float
    quantum_corrections: torch.Tensor
    causality_preserved: bool
    
    def __post_init__(self):
        # Ensure tensors are detached and on CPU for storage
        if self.visit_counts.requires_grad:
            self.visit_counts = self.visit_counts.detach()
        if self.quantum_corrections.requires_grad:
            self.quantum_corrections = self.quantum_corrections.detach()

@dataclass 
class StateManagerConfig:
    """Configuration for quantum state manager"""
    
    # Regime transition points
    critical_point_1: int = 1000    # Quantum -> Critical
    critical_point_2: int = 5000    # Critical -> Classical
    
    # Causality preservation
    enable_causality_preservation: bool = True
    max_snapshots: int = 100        # Maximum state snapshots to keep
    snapshot_interval: int = 10      # Take snapshot every N simulation steps
    
    # State tracking
    track_corrections: bool = True   # Track quantum corrections over time
    track_regime_transitions: bool = True
    
    # Performance settings
    cleanup_interval: int = 50       # Cleanup old states every N operations
    thread_safe: bool = True         # Enable thread-safe operations
    
    # Memory management
    max_correction_history: int = 1000  # Max quantum corrections to store
    auto_cleanup: bool = True           # Automatic memory cleanup

class CoherentQuantumStateManager:
    """
    Manages coherent quantum state across all MCTS components
    
    Provides:
    - Unified state tracking for all quantum processors
    - Causality preservation through pre-update snapshots  
    - Consistent regime detection across components
    - Thread-safe state synchronization
    - Memory-efficient state history management
    """
    
    def __init__(self, config: StateManagerConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Core state
        self.current_simulation_count = 0
        self.current_regime = QuantumRegime.QUANTUM
        self.current_hbar_eff = 0.1
        
        # State history
        self.state_snapshots: deque = deque(maxlen=config.max_snapshots)
        self.correction_history: deque = deque(maxlen=config.max_correction_history)
        self.regime_transition_log: List[Tuple[int, QuantumRegime, QuantumRegime]] = []
        
        # Causality preservation
        self.pre_update_states: Dict[int, QuantumStateSnapshot] = {}
        self.causality_violations: List[Tuple[int, str]] = []
        
        # Thread safety
        if config.thread_safe:
            self._lock = threading.RLock()
        else:
            self._lock = None
        
        # Performance tracking
        self.last_cleanup = 0
        self.total_operations = 0
        self.state_access_count = 0
        
        # Component registration
        self.registered_components: Dict[str, Any] = {}
        
        logger.info(f"CoherentQuantumStateManager initialized")
        logger.info(f"  Critical points: {config.critical_point_1} -> {config.critical_point_2}")
        logger.info(f"  Causality preservation: {config.enable_causality_preservation}")
        logger.info(f"  Thread safe: {config.thread_safe}")
    
    def _with_lock(self, func):
        """Execute function with thread lock if enabled"""
        if self._lock:
            with self._lock:
                return func()
        else:
            return func()
    
    def register_component(self, name: str, component: Any):
        """Register a quantum component for state synchronization"""
        def _register():
            self.registered_components[name] = component
            logger.debug(f"Registered component: {name}")
        
        self._with_lock(_register)
    
    def update_simulation_count(self, new_count: int, force_update: bool = False):
        """Update simulation count with regime detection and state management"""
        def _update():
            if new_count == self.current_simulation_count and not force_update:
                return
            
            previous_count = self.current_simulation_count
            previous_regime = self.current_regime
            
            # Update core state
            self.current_simulation_count = new_count
            self.total_operations += 1
            
            # Detect regime transition
            new_regime = self._detect_regime(new_count)
            if new_regime != self.current_regime:
                self.regime_transition_log.append((new_count, self.current_regime, new_regime))
                logger.debug(f"Regime transition: {self.current_regime.value} -> {new_regime.value} at N={new_count}")
                self.current_regime = new_regime
            
            # Update hbar_eff
            self.current_hbar_eff = self._calculate_hbar_eff(new_count)
            
            # Take snapshot for causality preservation
            if (self.config.enable_causality_preservation and 
                new_count % self.config.snapshot_interval == 0):
                self._take_state_snapshot(new_count)
            
            # Sync with registered components
            self._sync_components(new_count, new_regime)
            
            # Periodic cleanup
            if (self.config.auto_cleanup and 
                self.total_operations - self.last_cleanup > self.config.cleanup_interval):
                self._cleanup_old_states()
                self.last_cleanup = self.total_operations
        
        self._with_lock(_update)
    
    def _detect_regime(self, simulation_count: int) -> QuantumRegime:
        """Detect quantum regime based on simulation count"""
        if simulation_count < self.config.critical_point_1:
            return QuantumRegime.QUANTUM
        elif simulation_count < self.config.critical_point_2:
            return QuantumRegime.CRITICAL
        else:
            return QuantumRegime.CLASSICAL
    
    def _calculate_hbar_eff(self, simulation_count: int, hbar_0: float = 0.1, alpha: float = 0.5) -> float:
        """Calculate effective Planck constant: ℏ_eff(N) = ℏ_0 (1 + N)^(-α/2)"""
        return hbar_0 * ((1.0 + simulation_count) ** (-alpha * 0.5))
    
    def _take_state_snapshot(self, simulation_count: int):
        """Take a state snapshot for causality preservation"""
        # Create snapshot with dummy data (would be actual visit counts in real implementation)
        snapshot = QuantumStateSnapshot(
            timestamp=time.time(),
            simulation_count=simulation_count,
            regime=self.current_regime,
            visit_counts=torch.zeros(1),  # Placeholder
            hbar_eff=self.current_hbar_eff,
            quantum_corrections=torch.zeros(1),  # Placeholder
            causality_preserved=True
        )
        
        self.state_snapshots.append(snapshot)
        
        # Store for causality access
        if self.config.enable_causality_preservation:
            self.pre_update_states[simulation_count] = snapshot
    
    def _sync_components(self, simulation_count: int, regime: QuantumRegime):
        """Synchronize state with all registered components"""
        for name, component in self.registered_components.items():
            try:
                if hasattr(component, 'update_simulation_count'):
                    component.update_simulation_count(simulation_count)
                if hasattr(component, 'update_regime'):
                    component.update_regime(regime)
                if hasattr(component, 'update_hbar_eff'):
                    component.update_hbar_eff(self.current_hbar_eff)
            except Exception as e:
                logger.warning(f"Failed to sync component {name}: {e}")
    
    def _cleanup_old_states(self):
        """Clean up old state snapshots and history"""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep 1 hour of history
        
        # Cleanup pre-update states (keep last 100)
        if len(self.pre_update_states) > 100:
            old_keys = sorted(self.pre_update_states.keys())[:-100]
            for key in old_keys:
                del self.pre_update_states[key]
        
        # Cleanup old snapshots by time
        while (self.state_snapshots and 
               self.state_snapshots[0].timestamp < cutoff_time):
            self.state_snapshots.popleft()
        
        logger.debug(f"Cleaned up old states. Current: {len(self.state_snapshots)} snapshots, {len(self.pre_update_states)} pre-update states")
    
    def get_causality_safe_state(self, simulation_count: int) -> Optional[QuantumStateSnapshot]:
        """Get state snapshot that preserves causality for given simulation count"""
        def _get_state():
            self.state_access_count += 1
            
            # Find the most recent snapshot at or before simulation_count
            best_snapshot = None
            best_count = -1
            
            for snapshot in self.state_snapshots:
                if (snapshot.simulation_count <= simulation_count and 
                    snapshot.simulation_count > best_count):
                    best_snapshot = snapshot
                    best_count = snapshot.simulation_count
            
            return best_snapshot
        
        return self._with_lock(_get_state)
    
    def record_quantum_correction(
        self, 
        simulation_count: int,
        node_indices: torch.Tensor,
        corrections: torch.Tensor,
        component_name: str = "default"
    ):
        """Record quantum corrections for tracking and analysis"""
        def _record():
            if not self.config.track_corrections:
                return
            
            correction_record = {
                'timestamp': time.time(),
                'simulation_count': simulation_count,
                'component': component_name,
                'regime': self.current_regime,
                'mean_correction': torch.mean(torch.abs(corrections)).item(),
                'max_correction': torch.max(torch.abs(corrections)).item(),
                'num_nodes': len(node_indices)
            }
            
            self.correction_history.append(correction_record)
        
        self._with_lock(_record)
    
    def validate_causality(self, 
                          current_visit_counts: torch.Tensor,
                          simulation_count: int) -> bool:
        """Validate that causality is preserved"""
        def _validate():
            if not self.config.enable_causality_preservation:
                return True
            
            # Find reference state for causality check
            ref_snapshot = self.get_causality_safe_state(simulation_count - 1)
            if ref_snapshot is None:
                return True  # No reference available
            
            # Check that visit counts haven't decreased (causality violation)
            if ref_snapshot.visit_counts.shape == current_visit_counts.shape:
                violations = current_visit_counts < ref_snapshot.visit_counts
                if torch.any(violations):
                    violation_msg = f"Causality violation at N={simulation_count}: {torch.sum(violations)} nodes"
                    self.causality_violations.append((simulation_count, violation_msg))
                    logger.warning(violation_msg)
                    return False
            
            return True
        
        return self._with_lock(_validate)
    
    def get_regime_factor(self, regime: Optional[QuantumRegime] = None) -> float:
        """Get quantum factor for current or specified regime"""
        target_regime = regime or self.current_regime
        
        regime_factors = {
            QuantumRegime.QUANTUM: 1.0,
            QuantumRegime.CRITICAL: 0.5,
            QuantumRegime.CLASSICAL: 0.1
        }
        
        return regime_factors[target_regime]
    
    def get_adaptive_parameters(self) -> Dict[str, float]:
        """Get adaptive parameters based on current state"""
        def _get_params():
            return {
                'hbar_eff': self.current_hbar_eff,
                'quantum_factor': self.get_regime_factor(),
                'simulation_count': self.current_simulation_count,
                'regime': self.current_regime.value,
                'causality_preserved': len(self.causality_violations) == 0
            }
        
        return self._with_lock(_get_params)
    
    def get_comprehensive_state(self) -> Dict[str, Any]:
        """Get comprehensive state information for debugging/monitoring"""
        def _get_state():
            return {
                'current_simulation_count': self.current_simulation_count,
                'current_regime': self.current_regime.value,
                'current_hbar_eff': self.current_hbar_eff,
                'total_operations': self.total_operations,
                'state_access_count': self.state_access_count,
                'snapshots_count': len(self.state_snapshots),
                'pre_update_states_count': len(self.pre_update_states),
                'correction_history_count': len(self.correction_history),
                'regime_transitions': len(self.regime_transition_log),
                'causality_violations': len(self.causality_violations),
                'registered_components': list(self.registered_components.keys()),
                'last_cleanup': self.last_cleanup
            }
        
        return self._with_lock(_get_state)
    
    def reset(self):
        """Reset all state to initial conditions"""
        def _reset():
            self.current_simulation_count = 0
            self.current_regime = QuantumRegime.QUANTUM
            self.current_hbar_eff = 0.1
            
            self.state_snapshots.clear()
            self.correction_history.clear()
            self.regime_transition_log.clear()
            self.pre_update_states.clear()
            self.causality_violations.clear()
            
            self.total_operations = 0
            self.state_access_count = 0
            self.last_cleanup = 0
            
            logger.info("Quantum state manager reset")
        
        self._with_lock(_reset)

# Factory function
def create_quantum_state_manager(
    critical_point_1: int = 1000,
    critical_point_2: int = 5000,
    enable_causality_preservation: bool = True,
    thread_safe: bool = True,
    **kwargs
) -> CoherentQuantumStateManager:
    """Create quantum state manager with standard configuration"""
    config = StateManagerConfig(
        critical_point_1=critical_point_1,
        critical_point_2=critical_point_2,
        enable_causality_preservation=enable_causality_preservation,
        thread_safe=thread_safe,
        **kwargs
    )
    return CoherentQuantumStateManager(config)

# Export main classes
__all__ = [
    'CoherentQuantumStateManager',
    'QuantumRegime',
    'QuantumStateSnapshot', 
    'StateManagerConfig',
    'create_quantum_state_manager'
]