"""
Thermodynamic Tree Operations for MCTS

This module implements thermodynamic principles for tree operations, including:
1. Landauer's principle for irreversible computation costs
2. Thermodynamic work calculations for tree modifications
3. Entropy production in tree evolution
4. Energy-efficient tree pruning and expansion
5. Heat dissipation in value propagation
6. Reversible vs irreversible operation classification

Mathematical Foundation:
- Landauer limit: kT ln(2) per bit erasure
- Thermodynamic work: W = ∫ F·dx
- Entropy production: dS = dQ/T + dS_irr
- Free energy: F = U - TS
- Maxwell's demon and information processing
- Fluctuation theorems for non-equilibrium processes

Physical Interpretation:
- Tree operations as thermodynamic processes
- Information erasure costs energy
- Reversible operations preserve entropy
- Irreversible operations dissipate heat
- Optimal algorithms minimize entropy production
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from scipy import integrate
from scipy.special import erfcinv
import networkx as nx
from collections import defaultdict
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of tree operations"""
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"
    QUASI_STATIC = "quasi_static"
    ADIABATIC = "adiabatic"


@dataclass
class ThermodynamicState:
    """Thermodynamic state of a tree node"""
    internal_energy: float  # Internal energy U
    entropy: float  # Entropy S
    temperature: float  # Temperature T
    volume: float  # Effective volume (tree size)
    pressure: float  # Effective pressure (exploration pressure)
    chemical_potential: float  # Chemical potential (value potential)
    
    def free_energy(self) -> float:
        """Helmholtz free energy F = U - TS"""
        return self.internal_energy - self.temperature * self.entropy
    
    def enthalpy(self) -> float:
        """Enthalpy H = U + PV"""
        return self.internal_energy + self.pressure * self.volume
    
    def gibbs_free_energy(self) -> float:
        """Gibbs free energy G = H - TS"""
        return self.enthalpy() - self.temperature * self.entropy


@dataclass
class ThermodynamicProcess:
    """Represents a thermodynamic process in tree operations"""
    operation_type: OperationType
    initial_state: ThermodynamicState
    final_state: ThermodynamicState
    work_done: float
    heat_transferred: float
    entropy_produced: float
    landauer_cost: float
    reversibility: float  # 0 = irreversible, 1 = reversible
    
    def efficiency(self) -> float:
        """Thermodynamic efficiency"""
        if self.heat_transferred == 0:
            return 1.0
        return 1.0 - abs(self.entropy_produced * self.initial_state.temperature) / abs(self.heat_transferred)


@dataclass
class LandauerCost:
    """Landauer cost analysis for information processing"""
    bits_erased: int
    logical_operations: int
    reversible_operations: int
    irreversible_operations: int
    minimum_energy: float  # kT ln(2) per bit
    actual_energy: float
    efficiency: float  # actual/minimum
    
    def energy_overhead(self) -> float:
        """Energy overhead above Landauer limit"""
        return self.actual_energy - self.minimum_energy


class ThermodynamicTreeOperator:
    """
    Implements thermodynamic tree operations with energy accounting.
    
    Key Features:
    - Landauer cost calculation for irreversible operations
    - Entropy production tracking
    - Energy-efficient algorithms
    - Reversible operation detection
    - Heat dissipation modeling
    """
    
    def __init__(self, temperature: float = 1.0, boltzmann_constant: float = 1.0):
        """
        Initialize thermodynamic tree operator.
        
        Args:
            temperature: System temperature
            boltzmann_constant: Boltzmann constant (natural units)
        """
        self.temperature = temperature
        self.kB = boltzmann_constant
        self.landauer_unit = self.kB * temperature * np.log(2)  # Energy per bit
        
        # Track cumulative thermodynamic quantities
        self.total_work = 0.0
        self.total_heat = 0.0
        self.total_entropy_produced = 0.0
        self.total_landauer_cost = 0.0
        
        # Operation history
        self.operation_history: List[ThermodynamicProcess] = []
        
    def compute_node_state(self, node_data: Dict[str, Any]) -> ThermodynamicState:
        """
        Compute thermodynamic state of a tree node.
        
        Args:
            node_data: Node information (visits, value, children, etc.)
            
        Returns:
            Thermodynamic state
        """
        # Extract node properties
        visits = node_data.get('visits', 1)
        value = node_data.get('value', 0.0)
        children = node_data.get('children', [])
        depth = node_data.get('depth', 0)
        
        # Internal energy: related to accumulated value
        internal_energy = value * visits
        
        # Entropy: related to visit distribution uncertainty
        if visits > 0:
            # Use visit distribution to compute entropy
            # Handle both cases: children as dicts or as IDs
            if children:
                if isinstance(children[0], dict):
                    # Children are node objects
                    child_visits = [child.get('visits', 0) for child in children]
                else:
                    # Children are node IDs - use uniform distribution
                    child_visits = [1] * len(children)
                
                total_child_visits = sum(child_visits)
                
                if total_child_visits > 0:
                    probs = [v / total_child_visits for v in child_visits if v > 0]
                    entropy = -sum(p * np.log(p) for p in probs)
                else:
                    entropy = 0.0
            else:
                entropy = 0.0
        else:
            entropy = 0.0
            
        # Volume: effective size (number of descendants)
        volume = len(children) + 1
        
        # Pressure: exploration pressure (related to UCB)
        pressure = np.sqrt(2 * np.log(visits + 1) / (visits + 1))
        
        # Chemical potential: value potential
        chemical_potential = value
        
        return ThermodynamicState(
            internal_energy=internal_energy,
            entropy=entropy,
            temperature=self.temperature,
            volume=volume,
            pressure=pressure,
            chemical_potential=chemical_potential
        )
    
    def landauer_cost_node_deletion(self, node_data: Dict[str, Any]) -> LandauerCost:
        """
        Calculate Landauer cost for deleting a node.
        
        Node deletion is irreversible information erasure.
        
        Args:
            node_data: Node to be deleted
            
        Returns:
            Landauer cost analysis
        """
        # Estimate information content
        visits = node_data.get('visits', 1)
        value = node_data.get('value', 0.0)
        children = node_data.get('children', [])
        
        # Bits erased: log2 of state space
        bits_erased = max(1, int(np.log2(visits + 1)))
        
        # Add bits for value information
        value_bits = 32 if value != 0.0 else 0  # Assume 32-bit float
        bits_erased += value_bits
        
        # Add bits for structural information
        structure_bits = len(children) * 8  # Assume pointer is 8 bits
        bits_erased += structure_bits
        
        # All operations are irreversible for deletion
        irreversible_operations = bits_erased
        
        # Minimum energy (Landauer limit)
        minimum_energy = bits_erased * self.landauer_unit
        
        # Actual energy (assume perfect efficiency for now)
        actual_energy = minimum_energy
        
        return LandauerCost(
            bits_erased=bits_erased,
            logical_operations=bits_erased,
            reversible_operations=0,
            irreversible_operations=irreversible_operations,
            minimum_energy=minimum_energy,
            actual_energy=actual_energy,
            efficiency=1.0
        )
    
    def landauer_cost_value_update(self, old_value: float, new_value: float) -> LandauerCost:
        """
        Calculate Landauer cost for value updates.
        
        Args:
            old_value: Previous value
            new_value: New value
            
        Returns:
            Landauer cost analysis
        """
        # If values are identical, no cost
        if abs(old_value - new_value) < 1e-10:
            return LandauerCost(
                bits_erased=0,
                logical_operations=1,
                reversible_operations=1,
                irreversible_operations=0,
                minimum_energy=0.0,
                actual_energy=0.0,
                efficiency=1.0
            )
        
        # Estimate precision changes
        old_precision = max(1, int(-np.log10(abs(old_value) + 1e-10)))
        new_precision = max(1, int(-np.log10(abs(new_value) + 1e-10)))
        
        # Bits erased depends on precision change
        if new_precision > old_precision:
            # More precise - reversible expansion
            bits_erased = 0
            reversible_ops = 32  # Assume 32-bit value
            irreversible_ops = 0
        else:
            # Less precise - irreversible compression
            bits_erased = (old_precision - new_precision) * 4  # Rough estimate
            reversible_ops = 0
            irreversible_ops = 32
        
        minimum_energy = bits_erased * self.landauer_unit
        actual_energy = minimum_energy  # Perfect efficiency assumption
        
        return LandauerCost(
            bits_erased=bits_erased,
            logical_operations=32,
            reversible_operations=reversible_ops,
            irreversible_operations=irreversible_ops,
            minimum_energy=minimum_energy,
            actual_energy=actual_energy,
            efficiency=1.0 if minimum_energy == 0 else 1.0
        )
    
    def compute_work_expansion(self, parent_state: ThermodynamicState,
                              child_state: ThermodynamicState) -> float:
        """
        Compute thermodynamic work for tree expansion.
        
        Tree expansion is like gas expansion: W = -∫P dV
        
        Args:
            parent_state: Parent node state
            child_state: New child node state
            
        Returns:
            Work done during expansion
        """
        # Volume change
        dV = child_state.volume - parent_state.volume
        
        # Average pressure during expansion
        P_avg = (parent_state.pressure + child_state.pressure) / 2
        
        # Work done by system (negative for expansion)
        work = -P_avg * dV
        
        return work
    
    def compute_heat_transfer(self, initial_state: ThermodynamicState,
                            final_state: ThermodynamicState,
                            work_done: float) -> float:
        """
        Compute heat transfer using first law of thermodynamics.
        
        ΔU = Q - W  =>  Q = ΔU + W
        
        Args:
            initial_state: Initial thermodynamic state
            final_state: Final thermodynamic state
            work_done: Work done during process
            
        Returns:
            Heat transferred
        """
        # Change in internal energy
        delta_U = final_state.internal_energy - initial_state.internal_energy
        
        # Heat transferred
        heat = delta_U + work_done
        
        return heat
    
    def compute_entropy_production(self, initial_state: ThermodynamicState,
                                 final_state: ThermodynamicState,
                                 heat_transferred: float) -> float:
        """
        Compute entropy production.
        
        ΔS = ΔS_system + ΔS_environment
        ΔS_environment = -Q/T
        
        Args:
            initial_state: Initial state
            final_state: Final state
            heat_transferred: Heat transferred
            
        Returns:
            Total entropy production
        """
        # System entropy change
        delta_S_system = final_state.entropy - initial_state.entropy
        
        # Environment entropy change
        delta_S_environment = -heat_transferred / self.temperature
        
        # Total entropy production (must be ≥ 0 for physical processes)
        entropy_produced = delta_S_system + delta_S_environment
        
        return max(0.0, entropy_produced)
    
    def execute_node_expansion(self, parent_data: Dict[str, Any],
                             child_data: Dict[str, Any]) -> ThermodynamicProcess:
        """
        Execute thermodynamic node expansion.
        
        Args:
            parent_data: Parent node data
            child_data: Child node data
            
        Returns:
            Thermodynamic process record
        """
        # Compute thermodynamic states
        parent_state = self.compute_node_state(parent_data)
        child_state = self.compute_node_state(child_data)
        
        # Compute work done
        work_done = self.compute_work_expansion(parent_state, child_state)
        
        # Compute heat transfer
        heat_transferred = self.compute_heat_transfer(parent_state, child_state, work_done)
        
        # Compute entropy production
        entropy_produced = self.compute_entropy_production(parent_state, child_state, heat_transferred)
        
        # Landauer cost (expansion is typically reversible)
        landauer_cost = 0.0  # Reversible expansion
        
        # Reversibility (high for expansion)
        reversibility = 0.9
        
        # Create process record
        process = ThermodynamicProcess(
            operation_type=OperationType.QUASI_STATIC,
            initial_state=parent_state,
            final_state=child_state,
            work_done=work_done,
            heat_transferred=heat_transferred,
            entropy_produced=entropy_produced,
            landauer_cost=landauer_cost,
            reversibility=reversibility
        )
        
        # Update totals
        self.total_work += work_done
        self.total_heat += heat_transferred
        self.total_entropy_produced += entropy_produced
        self.total_landauer_cost += landauer_cost
        
        # Record operation
        self.operation_history.append(process)
        
        return process
    
    def execute_node_deletion(self, node_data: Dict[str, Any]) -> ThermodynamicProcess:
        """
        Execute thermodynamic node deletion.
        
        Args:
            node_data: Node to delete
            
        Returns:
            Thermodynamic process record
        """
        # Compute states
        initial_state = self.compute_node_state(node_data)
        final_state = ThermodynamicState(
            internal_energy=0.0,
            entropy=0.0,
            temperature=self.temperature,
            volume=0.0,
            pressure=0.0,
            chemical_potential=0.0
        )
        
        # Landauer cost analysis
        landauer_analysis = self.landauer_cost_node_deletion(node_data)
        
        # Work done (compression work)
        work_done = -initial_state.pressure * initial_state.volume
        
        # Heat transfer
        heat_transferred = self.compute_heat_transfer(initial_state, final_state, work_done)
        
        # Entropy production (includes Landauer cost)
        entropy_produced = self.compute_entropy_production(initial_state, final_state, heat_transferred)
        entropy_produced += landauer_analysis.bits_erased * np.log(2)  # Additional entropy from erasure
        
        # Create process record
        process = ThermodynamicProcess(
            operation_type=OperationType.IRREVERSIBLE,
            initial_state=initial_state,
            final_state=final_state,
            work_done=work_done,
            heat_transferred=heat_transferred,
            entropy_produced=entropy_produced,
            landauer_cost=landauer_analysis.minimum_energy,
            reversibility=0.0  # Completely irreversible
        )
        
        # Update totals
        self.total_work += work_done
        self.total_heat += heat_transferred
        self.total_entropy_produced += entropy_produced
        self.total_landauer_cost += landauer_analysis.minimum_energy
        
        # Record operation
        self.operation_history.append(process)
        
        return process
    
    def execute_value_update(self, node_data: Dict[str, Any],
                           old_value: float, new_value: float) -> ThermodynamicProcess:
        """
        Execute thermodynamic value update.
        
        Args:
            node_data: Node being updated
            old_value: Previous value
            new_value: New value
            
        Returns:
            Thermodynamic process record
        """
        # Create temporary states with different values
        node_old = node_data.copy()
        node_old['value'] = old_value
        node_new = node_data.copy()
        node_new['value'] = new_value
        
        initial_state = self.compute_node_state(node_old)
        final_state = self.compute_node_state(node_new)
        
        # Landauer cost analysis
        landauer_analysis = self.landauer_cost_value_update(old_value, new_value)
        
        # Work done (minimal for value update)
        work_done = 0.1 * abs(new_value - old_value)
        
        # Heat transfer
        heat_transferred = self.compute_heat_transfer(initial_state, final_state, work_done)
        
        # Entropy production
        entropy_produced = self.compute_entropy_production(initial_state, final_state, heat_transferred)
        
        # Determine operation type
        if landauer_analysis.bits_erased > 0:
            op_type = OperationType.IRREVERSIBLE
            reversibility = 0.0
        else:
            op_type = OperationType.REVERSIBLE
            reversibility = 1.0
        
        # Create process record
        process = ThermodynamicProcess(
            operation_type=op_type,
            initial_state=initial_state,
            final_state=final_state,
            work_done=work_done,
            heat_transferred=heat_transferred,
            entropy_produced=entropy_produced,
            landauer_cost=landauer_analysis.minimum_energy,
            reversibility=reversibility
        )
        
        # Update totals
        self.total_work += work_done
        self.total_heat += heat_transferred
        self.total_entropy_produced += entropy_produced
        self.total_landauer_cost += landauer_analysis.minimum_energy
        
        # Record operation
        self.operation_history.append(process)
        
        return process
    
    def compute_efficiency_metrics(self) -> Dict[str, float]:
        """
        Compute overall thermodynamic efficiency metrics.
        
        Returns:
            Dictionary with efficiency metrics
        """
        if not self.operation_history:
            return {}
        
        # Overall efficiency
        total_useful_work = sum(p.work_done for p in self.operation_history if p.work_done > 0)
        total_heat_input = sum(p.heat_transferred for p in self.operation_history if p.heat_transferred > 0)
        
        overall_efficiency = (total_useful_work / total_heat_input) if total_heat_input > 0 else 0.0
        
        # Landauer efficiency
        theoretical_minimum = self.total_landauer_cost
        actual_energy = sum(abs(p.heat_transferred) for p in self.operation_history)
        landauer_efficiency = (theoretical_minimum / actual_energy) if actual_energy > 0 else 1.0
        
        # Reversibility ratio
        reversible_ops = sum(1 for p in self.operation_history if p.reversibility > 0.5)
        total_ops = len(self.operation_history)
        reversibility_ratio = reversible_ops / total_ops if total_ops > 0 else 0.0
        
        # Entropy production rate
        if self.operation_history:
            entropy_rate = self.total_entropy_produced / len(self.operation_history)
        else:
            entropy_rate = 0.0
        
        return {
            'overall_efficiency': overall_efficiency,
            'landauer_efficiency': landauer_efficiency,
            'reversibility_ratio': reversibility_ratio,
            'entropy_production_rate': entropy_rate,
            'total_work': self.total_work,
            'total_heat': self.total_heat,
            'total_entropy_produced': self.total_entropy_produced,
            'total_landauer_cost': self.total_landauer_cost
        }
    
    def optimize_tree_operations(self, tree_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize tree operations for energy efficiency.
        
        Args:
            tree_data: Tree structure data
            
        Returns:
            Optimization recommendations
        """
        # Analyze current tree
        nodes = tree_data.get('nodes', [])
        if not nodes:
            return {'error': 'No nodes in tree'}
        
        # Compute thermodynamic states for all nodes
        node_states = {}
        for i, node in enumerate(nodes):
            node_states[i] = self.compute_node_state(node)
        
        # Identify inefficient operations
        inefficient_nodes = []
        for i, state in node_states.items():
            # High entropy nodes are inefficient
            if state.entropy > 2.0:
                inefficient_nodes.append(i)
        
        # Recommend operations
        recommendations = []
        
        # 1. Pruning recommendations
        for node_id in inefficient_nodes:
            landauer_cost = self.landauer_cost_node_deletion(nodes[node_id])
            if landauer_cost.minimum_energy < 0.1:  # Low cost to delete
                recommendations.append({
                    'operation': 'prune',
                    'node_id': node_id,
                    'energy_saved': landauer_cost.minimum_energy,
                    'reason': 'high_entropy_low_cost'
                })
        
        # 2. Value update recommendations
        for i, node in enumerate(nodes):
            if i in node_states:
                state = node_states[i]
                current_value = node.get('value', 0.0)
                
                # Recommend value quantization for efficiency
                if abs(current_value) < 0.01:  # Very small values
                    recommendations.append({
                        'operation': 'quantize_value',
                        'node_id': i,
                        'old_value': current_value,
                        'new_value': 0.0,
                        'reason': 'value_quantization'
                    })
        
        # 3. Expansion recommendations
        expansion_candidates = []
        for i, state in node_states.items():
            if state.pressure > 1.0:  # High exploration pressure
                expansion_candidates.append((i, state.pressure))
        
        # Sort by pressure and recommend top candidates
        expansion_candidates.sort(key=lambda x: x[1], reverse=True)
        for node_id, pressure in expansion_candidates[:3]:
            recommendations.append({
                'operation': 'expand',
                'node_id': node_id,
                'pressure': pressure,
                'reason': 'high_exploration_pressure'
            })
        
        return {
            'recommendations': recommendations,
            'inefficient_nodes': inefficient_nodes,
            'node_states': {i: state.__dict__ for i, state in node_states.items()},
            'efficiency_metrics': self.compute_efficiency_metrics()
        }
    
    def simulate_heat_dissipation(self, tree_data: Dict[str, Any],
                                time_steps: int = 100) -> Dict[str, Any]:
        """
        Simulate heat dissipation in tree operations.
        
        Args:
            tree_data: Tree data
            time_steps: Number of simulation steps
            
        Returns:
            Heat dissipation analysis
        """
        nodes = tree_data.get('nodes', [])
        if not nodes:
            return {'error': 'No nodes in tree'}
        
        # Initialize temperatures
        temperatures = np.full(len(nodes), self.temperature)
        heat_capacities = np.ones(len(nodes))  # Assume unit heat capacity
        
        # Thermal conductivity matrix (based on tree structure)
        thermal_conductivity = np.zeros((len(nodes), len(nodes)))
        
        # Set up thermal connections
        for i, node in enumerate(nodes):
            children = node.get('children', [])
            for child_id in children:
                if child_id < len(nodes):
                    thermal_conductivity[i][child_id] = 0.1
                    thermal_conductivity[child_id][i] = 0.1
        
        # Simulation
        temperature_history = []
        heat_flow_history = []
        
        for step in range(time_steps):
            # Heat diffusion equation: dT/dt = α ∇²T
            new_temperatures = temperatures.copy()
            
            for i in range(len(nodes)):
                # Compute heat flow
                heat_flow = 0.0
                for j in range(len(nodes)):
                    if i != j:
                        heat_flow += thermal_conductivity[i][j] * (temperatures[j] - temperatures[i])
                
                # Update temperature
                new_temperatures[i] += 0.01 * heat_flow / heat_capacities[i]
            
            temperatures = new_temperatures
            temperature_history.append(temperatures.copy())
            
            # Calculate total heat flow
            total_heat_flow = 0.0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    flow = thermal_conductivity[i][j] * abs(temperatures[i] - temperatures[j])
                    total_heat_flow += flow
            
            heat_flow_history.append(total_heat_flow)
        
        # Analysis
        final_temp_variance = np.var(temperatures)
        equilibration_time = time_steps
        
        # Find equilibration time
        for i in range(1, len(heat_flow_history)):
            if heat_flow_history[i] < 0.01 * heat_flow_history[0]:
                equilibration_time = i
                break
        
        return {
            'final_temperatures': temperatures.tolist(),
            'temperature_history': temperature_history,
            'heat_flow_history': heat_flow_history,
            'equilibration_time': equilibration_time,
            'final_temperature_variance': final_temp_variance,
            'average_final_temperature': np.mean(temperatures)
        }
    
    def compute_maxwell_demon_efficiency(self, tree_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute Maxwell's demon efficiency for information processing.
        
        Relates to how efficiently the MCTS algorithm sorts good from bad moves.
        
        Args:
            tree_data: Tree data
            
        Returns:
            Maxwell's demon efficiency analysis
        """
        nodes = tree_data.get('nodes', [])
        if not nodes:
            return {'error': 'No nodes in tree'}
        
        # Extract values and visits
        values = np.array([node.get('value', 0.0) for node in nodes])
        visits = np.array([node.get('visits', 1) for node in nodes])
        
        # Compute information gained about values
        # This is the "demon's" knowledge about which nodes are good/bad
        
        # Initial entropy (no knowledge)
        initial_entropy = np.log(len(nodes))
        
        # Final entropy (based on visit distribution)
        visit_probs = visits / visits.sum()
        final_entropy = -np.sum(visit_probs * np.log(visit_probs + 1e-10))
        
        # Information gained
        information_gained = initial_entropy - final_entropy
        
        # Energy cost (Landauer cost for information processing)
        energy_cost = information_gained * self.landauer_unit
        
        # Theoretical minimum energy for sorting
        theoretical_minimum = np.log(2) * len(nodes) * self.landauer_unit
        
        # Demon efficiency
        demon_efficiency = theoretical_minimum / energy_cost if energy_cost > 0 else 1.0
        
        # Szilard engine efficiency (energy extracted vs information)
        energy_extracted = np.sum(visits * values)  # Weighted value
        szilard_efficiency = energy_extracted / (information_gained * self.temperature)
        
        return {
            'information_gained': information_gained,
            'energy_cost': energy_cost,
            'theoretical_minimum': theoretical_minimum,
            'demon_efficiency': demon_efficiency,
            'szilard_efficiency': szilard_efficiency,
            'initial_entropy': initial_entropy,
            'final_entropy': final_entropy
        }
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all thermodynamic operations.
        
        Returns:
            Summary statistics
        """
        if not self.operation_history:
            return {'message': 'No operations recorded'}
        
        # Operation type distribution
        op_types = {}
        for process in self.operation_history:
            op_type = process.operation_type.value
            op_types[op_type] = op_types.get(op_type, 0) + 1
        
        # Efficiency distribution
        efficiencies = [process.efficiency() for process in self.operation_history]
        
        # Reversibility distribution
        reversibilities = [process.reversibility for process in self.operation_history]
        
        return {
            'total_operations': len(self.operation_history),
            'operation_types': op_types,
            'total_work': self.total_work,
            'total_heat': self.total_heat,
            'total_entropy_produced': self.total_entropy_produced,
            'total_landauer_cost': self.total_landauer_cost,
            'average_efficiency': np.mean(efficiencies),
            'average_reversibility': np.mean(reversibilities),
            'efficiency_std': np.std(efficiencies),
            'reversibility_std': np.std(reversibilities),
            'landauer_unit': self.landauer_unit
        }


def apply_thermodynamic_operations(tree_data: Dict[str, Any],
                                 operations: List[Dict[str, Any]],
                                 temperature: float = 1.0) -> Dict[str, Any]:
    """
    Apply thermodynamic operations to tree with energy accounting.
    
    Args:
        tree_data: Tree structure data
        operations: List of operations to perform
        temperature: System temperature
        
    Returns:
        Results with thermodynamic analysis
    """
    operator = ThermodynamicTreeOperator(temperature=temperature)
    
    results = {
        'processes': [],
        'modified_tree': tree_data.copy(),
        'energy_analysis': {},
        'recommendations': {}
    }
    
    # Execute operations
    for operation in operations:
        op_type = operation.get('type')
        
        if op_type == 'expand':
            parent_id = operation.get('parent_id', 0)
            child_data = operation.get('child_data', {})
            
            if parent_id < len(tree_data.get('nodes', [])):
                parent_data = tree_data['nodes'][parent_id]
                process = operator.execute_node_expansion(parent_data, child_data)
                results['processes'].append(process)
        
        elif op_type == 'delete':
            node_id = operation.get('node_id', 0)
            
            if node_id < len(tree_data.get('nodes', [])):
                node_data = tree_data['nodes'][node_id]
                process = operator.execute_node_deletion(node_data)
                results['processes'].append(process)
        
        elif op_type == 'update_value':
            node_id = operation.get('node_id', 0)
            old_value = operation.get('old_value', 0.0)
            new_value = operation.get('new_value', 0.0)
            
            if node_id < len(tree_data.get('nodes', [])):
                node_data = tree_data['nodes'][node_id]
                process = operator.execute_value_update(node_data, old_value, new_value)
                results['processes'].append(process)
    
    # Compute analysis
    results['energy_analysis'] = operator.compute_efficiency_metrics()
    results['recommendations'] = operator.optimize_tree_operations(tree_data)
    results['operation_summary'] = operator.get_operation_summary()
    results['maxwell_demon'] = operator.compute_maxwell_demon_efficiency(tree_data)
    results['heat_dissipation'] = operator.simulate_heat_dissipation(tree_data)
    
    return results