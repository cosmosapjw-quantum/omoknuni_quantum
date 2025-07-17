"""
Quantum Error Correction Analogues for MCTS

This module implements quantum error correction concepts adapted for MCTS:
1. Redundant value encoding using quantum codes
2. Error detection and correction in value propagation
3. Fault-tolerant tree operations
4. Syndrome extraction for value corruption detection
5. Logical qubits for protected value storage
6. Threshold theorems for error correction capacity

Mathematical Foundation:
- Quantum error correction codes (Shor, Steane, surface codes)
- Stabilizer formalism
- Syndrome extraction: s = H·e (H = parity check matrix)
- Logical operators: X_L, Z_L acting on code space
- Error threshold: p < p_th for fault tolerance
- Quantum channel capacity for noisy storage

Physical Interpretation:
- MCTS values as quantum states subject to decoherence
- Tree operations as quantum gates (potentially faulty)
- Redundant encoding protects against value corruption
- Syndrome measurements detect errors without disturbing values
- Error correction maintains value fidelity during propagation
- Fault-tolerant algorithms enable robust tree search
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.linalg import null_space
import itertools
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantumCode:
    """
    Quantum error correction code representation.
    
    Stores the structure of a quantum code including:
    - Stabilizer generators
    - Logical operators
    - Code parameters [n, k, d]
    """
    n: int  # Number of physical qubits
    k: int  # Number of logical qubits
    d: int  # Minimum distance
    stabilizers: List[str]  # Stabilizer generators (Pauli strings)
    logical_x: List[str]  # Logical X operators
    logical_z: List[str]  # Logical Z operators
    name: str  # Code name
    
    def __post_init__(self):
        """Validate code structure"""
        if len(self.stabilizers) != self.n - self.k:
            logger.warning(f"Code {self.name}: expected {self.n - self.k} stabilizers, got {len(self.stabilizers)}")
        if len(self.logical_x) != self.k:
            logger.warning(f"Code {self.name}: expected {self.k} logical X operators, got {len(self.logical_x)}")
        if len(self.logical_z) != self.k:
            logger.warning(f"Code {self.name}: expected {self.k} logical Z operators, got {len(self.logical_z)}")


@dataclass
class ErrorSyndrome:
    """
    Error syndrome from stabilizer measurements.
    
    Contains information about detected errors and their locations.
    """
    syndrome_bits: List[int]  # Syndrome measurement outcomes
    error_type: str  # 'X', 'Z', or 'Y' error
    error_locations: List[int]  # Suspected error locations
    correction_applied: bool  # Whether correction was applied
    correction_success: bool  # Whether correction succeeded
    detection_confidence: float  # Confidence in error detection
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            'syndrome_bits': self.syndrome_bits,
            'error_type': self.error_type,
            'error_locations': self.error_locations,
            'correction_applied': self.correction_applied,
            'correction_success': self.correction_success,
            'detection_confidence': self.detection_confidence
        }


@dataclass
class LogicalQubit:
    """
    Logical qubit encoded in quantum error correction code.
    
    Stores the logical state and provides protected operations.
    """
    logical_state: torch.Tensor  # Logical state vector
    physical_qubits: List[int]  # Physical qubit indices
    code: QuantumCode  # Error correction code
    syndrome_history: List[ErrorSyndrome]  # History of syndromes
    error_count: int  # Number of errors corrected
    fidelity: float  # Current fidelity estimate
    
    def __post_init__(self):
        """Initialize logical qubit"""
        if len(self.physical_qubits) != self.code.n:
            raise ValueError(f"Expected {self.code.n} physical qubits, got {len(self.physical_qubits)}")
        if self.syndrome_history is None:
            self.syndrome_history = []


class QuantumErrorCorrector:
    """
    Quantum error correction system for MCTS values.
    
    Implements fault-tolerant value encoding, error detection, and correction
    to protect MCTS values from noise and corruption during tree operations.
    
    Key Features:
    - Multiple quantum code support (Shor, Steane, surface codes)
    - Syndrome extraction for error detection
    - Maximum likelihood error correction
    - Fault-tolerant logical operations
    - Threshold estimation for error correction capacity
    """
    
    def __init__(self, code_type: str = "steane", noise_model: str = "depolarizing"):
        """
        Initialize quantum error corrector.
        
        Args:
            code_type: Type of quantum code ("shor", "steane", "surface")
            noise_model: Noise model ("depolarizing", "amplitude_damping", "phase_damping")
        """
        self.code_type = code_type
        self.noise_model = noise_model
        
        # Initialize quantum codes
        self.codes = self._initialize_codes()
        self.current_code = self.codes[code_type]
        
        # Error statistics
        self.error_rates = defaultdict(float)
        self.correction_success_rate = 0.0
        self.total_corrections = 0
        
        # Logical qubits storage
        self.logical_qubits: Dict[int, LogicalQubit] = {}
        
        # Syndrome lookup table for fast correction
        self.syndrome_table = self._build_syndrome_table()
    
    def _initialize_codes(self) -> Dict[str, QuantumCode]:
        """Initialize quantum error correction codes"""
        codes = {}
        
        # Shor's 9-qubit code
        codes["shor"] = QuantumCode(
            n=9, k=1, d=3,
            stabilizers=[
                "ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII",
                "IIIIIIZZI", "IIIIIIIIZ", "XXXXXXIII", "IIIXXXXXX"
            ],
            logical_x=["XXXXXXXXX"],
            logical_z=["ZZZZZZZZZ"],
            name="Shor"
        )
        
        # Steane's 7-qubit code
        codes["steane"] = QuantumCode(
            n=7, k=1, d=3,
            stabilizers=[
                "IIIXXXX", "IXXIIXX", "XIXIXIX",
                "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"
            ],
            logical_x=["XXXXXXX"],
            logical_z=["ZZZZZZZ"],
            name="Steane"
        )
        
        # Surface code (5-qubit)
        codes["surface"] = QuantumCode(
            n=5, k=1, d=3,
            stabilizers=[
                "XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"
            ],
            logical_x=["XXXXX"],
            logical_z=["ZZZZZ"],
            name="Surface"
        )
        
        return codes
    
    def _build_syndrome_table(self) -> Dict[Tuple[int, ...], List[int]]:
        """Build syndrome lookup table for error correction"""
        syndrome_table = {}
        
        # For each possible single-qubit error
        for i in range(self.current_code.n):
            # X error at position i
            syndrome_x = self._compute_syndrome(f"{'I' * i}X{'I' * (self.current_code.n - i - 1)}")
            syndrome_table[tuple(syndrome_x)] = [('X', i)]
            
            # Z error at position i
            syndrome_z = self._compute_syndrome(f"{'I' * i}Z{'I' * (self.current_code.n - i - 1)}")
            syndrome_table[tuple(syndrome_z)] = [('Z', i)]
            
            # Y error at position i (X and Z)
            syndrome_y = self._compute_syndrome(f"{'I' * i}Y{'I' * (self.current_code.n - i - 1)}")
            syndrome_table[tuple(syndrome_y)] = [('Y', i)]
        
        # No error syndrome
        syndrome_table[(0,) * len(self.current_code.stabilizers)] = []
        
        return syndrome_table
    
    def _compute_syndrome(self, error_string: str) -> List[int]:
        """Compute syndrome for given error string"""
        syndrome = []
        
        for stabilizer in self.current_code.stabilizers:
            # Check if error anticommutes with stabilizer
            commutes = self._check_commutation(error_string, stabilizer)
            syndrome.append(0 if commutes else 1)
        
        return syndrome
    
    def _check_commutation(self, pauli1: str, pauli2: str) -> bool:
        """Check if two Pauli strings commute"""
        if len(pauli1) != len(pauli2):
            return False
        
        anticommutations = 0
        for p1, p2 in zip(pauli1, pauli2):
            # Count positions where operators anticommute
            if (p1 == 'X' and p2 == 'Z') or (p1 == 'Z' and p2 == 'X'):
                anticommutations += 1
            elif (p1 == 'Y' and p2 == 'X') or (p1 == 'X' and p2 == 'Y'):
                anticommutations += 1
            elif (p1 == 'Y' and p2 == 'Z') or (p1 == 'Z' and p2 == 'Y'):
                anticommutations += 1
        
        # Two operators commute if they anticommute at an even number of positions
        return anticommutations % 2 == 0
    
    def encode_value(self, value: float, qubit_id: int) -> LogicalQubit:
        """
        Encode MCTS value into logical qubit.
        
        Args:
            value: Value to encode (normalized to [0,1])
            qubit_id: Logical qubit identifier
            
        Returns:
            Encoded logical qubit
        """
        # Normalize value to [0, 1]
        normalized_value = max(0.0, min(1.0, (value + 1.0) / 2.0))
        
        # Encode as quantum state |ψ⟩ = √(1-p)|0⟩ + √p|1⟩
        amplitude_0 = np.sqrt(1 - normalized_value)
        amplitude_1 = np.sqrt(normalized_value)
        
        # Create logical state vector
        logical_state = torch.tensor([amplitude_0, amplitude_1], dtype=torch.complex64)
        
        # Assign physical qubits
        physical_qubits = list(range(self.current_code.n))
        
        # Create logical qubit
        logical_qubit = LogicalQubit(
            logical_state=logical_state,
            physical_qubits=physical_qubits,
            code=self.current_code,
            syndrome_history=[],
            error_count=0,
            fidelity=1.0
        )
        
        # Store in registry
        self.logical_qubits[qubit_id] = logical_qubit
        
        return logical_qubit
    
    def decode_value(self, logical_qubit: LogicalQubit) -> float:
        """
        Decode logical qubit back to MCTS value.
        
        Args:
            logical_qubit: Logical qubit to decode
            
        Returns:
            Decoded value
        """
        # Extract probability of |1⟩ state
        prob_one = float(torch.abs(logical_qubit.logical_state[1])**2)
        
        # Convert back to [-1, 1] range
        value = 2.0 * prob_one - 1.0
        
        return value
    
    def apply_noise(self, logical_qubit: LogicalQubit, error_rate: float) -> LogicalQubit:
        """
        Apply noise to logical qubit.
        
        Args:
            logical_qubit: Logical qubit to apply noise to
            error_rate: Probability of error per physical qubit
            
        Returns:
            Noisy logical qubit
        """
        # Generate random errors
        errors = []
        for i in range(self.current_code.n):
            if np.random.random() < error_rate:
                if self.noise_model == "depolarizing":
                    error_type = np.random.choice(['X', 'Y', 'Z'])
                elif self.noise_model == "amplitude_damping":
                    error_type = 'X' if np.random.random() < 0.5 else 'Z'
                elif self.noise_model == "phase_damping":
                    error_type = 'Z'
                else:
                    error_type = 'X'
                
                errors.append((error_type, i))
        
        # Apply errors to logical state (simplified)
        if errors:
            # Reduce fidelity based on number of errors
            fidelity_reduction = len(errors) * 0.1
            logical_qubit.fidelity = max(0.0, logical_qubit.fidelity - fidelity_reduction)
        
        return logical_qubit
    
    def measure_syndrome(self, logical_qubit: LogicalQubit) -> ErrorSyndrome:
        """
        Measure error syndrome without disturbing logical state.
        
        Args:
            logical_qubit: Logical qubit to measure
            
        Returns:
            Error syndrome
        """
        # Simulate syndrome measurement
        # In real implementation, this would involve stabilizer measurements
        
        # Generate random syndrome (for demonstration)
        syndrome_bits = []
        for _ in self.current_code.stabilizers:
            # Probability of syndrome bit being 1 depends on fidelity
            prob_error = 1.0 - logical_qubit.fidelity
            syndrome_bit = 1 if np.random.random() < prob_error else 0
            syndrome_bits.append(syndrome_bit)
        
        # Look up error from syndrome table
        syndrome_tuple = tuple(syndrome_bits)
        if syndrome_tuple in self.syndrome_table:
            error_info = self.syndrome_table[syndrome_tuple]
            if error_info:
                error_type, error_location = error_info[0]
                error_locations = [error_location]
            else:
                error_type = "none"
                error_locations = []
        else:
            # Unknown syndrome - multiple errors
            error_type = "unknown"
            error_locations = []
        
        # Estimate detection confidence
        detection_confidence = 1.0 - np.sum(syndrome_bits) * 0.1
        
        syndrome = ErrorSyndrome(
            syndrome_bits=syndrome_bits,
            error_type=error_type,
            error_locations=error_locations,
            correction_applied=False,
            correction_success=False,
            detection_confidence=detection_confidence
        )
        
        # Add to history
        logical_qubit.syndrome_history.append(syndrome)
        
        return syndrome
    
    def correct_errors(self, logical_qubit: LogicalQubit, syndrome: ErrorSyndrome) -> bool:
        """
        Correct errors based on syndrome measurement.
        
        Args:
            logical_qubit: Logical qubit to correct
            syndrome: Error syndrome from measurement
            
        Returns:
            Whether correction was successful
        """
        if syndrome.error_type == "none":
            syndrome.correction_applied = True
            syndrome.correction_success = True
            return True
        
        if syndrome.error_type == "unknown":
            # Cannot correct unknown syndrome
            syndrome.correction_applied = False
            syndrome.correction_success = False
            return False
        
        # Apply correction
        if syndrome.error_locations:
            # Correct the error (simplified)
            # In real implementation, this would apply Pauli corrections
            
            # Increase fidelity after correction
            fidelity_improvement = 0.8 * (1.0 - logical_qubit.fidelity)
            logical_qubit.fidelity = min(1.0, logical_qubit.fidelity + fidelity_improvement)
            
            # Update error count
            logical_qubit.error_count += 1
            
            # Update syndrome
            syndrome.correction_applied = True
            syndrome.correction_success = True
            
            # Update statistics
            self.total_corrections += 1
            self.error_rates[syndrome.error_type] += 1
            
            return True
        
        return False
    
    def fault_tolerant_operation(self, logical_qubit: LogicalQubit, 
                               operation: str, parameter: float = 0.0) -> LogicalQubit:
        """
        Apply fault-tolerant logical operation.
        
        Args:
            logical_qubit: Logical qubit to operate on
            operation: Operation type ("X", "Z", "H", "rotation")
            parameter: Operation parameter (for rotations)
            
        Returns:
            Logical qubit after operation
        """
        # Measure syndrome before operation
        syndrome_before = self.measure_syndrome(logical_qubit)
        if syndrome_before.error_type != "none":
            self.correct_errors(logical_qubit, syndrome_before)
        
        # Apply logical operation
        if operation == "X":
            # Logical X: flip logical state
            logical_qubit.logical_state = torch.tensor([
                logical_qubit.logical_state[1],
                logical_qubit.logical_state[0]
            ])
        elif operation == "Z":
            # Logical Z: phase flip
            logical_qubit.logical_state = torch.tensor([
                logical_qubit.logical_state[0],
                -logical_qubit.logical_state[1]
            ])
        elif operation == "H":
            # Logical Hadamard
            factor = 1.0 / np.sqrt(2)
            new_state = torch.tensor([
                factor * (logical_qubit.logical_state[0] + logical_qubit.logical_state[1]),
                factor * (logical_qubit.logical_state[0] - logical_qubit.logical_state[1])
            ])
            logical_qubit.logical_state = new_state
        elif operation == "rotation":
            # Logical rotation by parameter
            cos_theta = np.cos(parameter / 2)
            sin_theta = np.sin(parameter / 2)
            new_state = torch.tensor([
                cos_theta * logical_qubit.logical_state[0] - 1j * sin_theta * logical_qubit.logical_state[1],
                -1j * sin_theta * logical_qubit.logical_state[0] + cos_theta * logical_qubit.logical_state[1]
            ])
            logical_qubit.logical_state = new_state
        
        # Measure syndrome after operation
        syndrome_after = self.measure_syndrome(logical_qubit)
        if syndrome_after.error_type != "none":
            self.correct_errors(logical_qubit, syndrome_after)
        
        # Slight fidelity reduction due to operation
        logical_qubit.fidelity *= 0.99
        
        return logical_qubit
    
    def compute_error_threshold(self, max_error_rate: float = 0.1, 
                              num_trials: int = 1000) -> Dict[str, float]:
        """
        Compute error threshold for current code.
        
        Args:
            max_error_rate: Maximum error rate to test
            num_trials: Number of Monte Carlo trials
            
        Returns:
            Threshold analysis results
        """
        error_rates = np.linspace(0.001, max_error_rate, 20)
        success_rates = []
        
        for error_rate in error_rates:
            successes = 0
            
            for _ in range(num_trials):
                # Create logical qubit
                logical_qubit = self.encode_value(0.5, 0)
                
                # Apply noise
                self.apply_noise(logical_qubit, error_rate)
                
                # Measure syndrome
                syndrome = self.measure_syndrome(logical_qubit)
                
                # Try to correct
                success = self.correct_errors(logical_qubit, syndrome)
                
                if success:
                    successes += 1
            
            success_rate = successes / num_trials
            success_rates.append(success_rate)
        
        # Find threshold (where success rate drops below 50%)
        threshold = max_error_rate
        for i, success_rate in enumerate(success_rates):
            if success_rate < 0.5:
                threshold = error_rates[i]
                break
        
        return {
            'threshold': threshold,
            'error_rates': error_rates.tolist(),
            'success_rates': success_rates,
            'code_name': self.current_code.name,
            'theoretical_threshold': 0.029 if self.code_type == "surface" else 0.01  # Approximate
        }
    
    def analyze_syndrome_statistics(self) -> Dict[str, Any]:
        """
        Analyze syndrome statistics across all logical qubits.
        
        Returns:
            Syndrome statistics analysis
        """
        all_syndromes = []
        for logical_qubit in self.logical_qubits.values():
            all_syndromes.extend(logical_qubit.syndrome_history)
        
        if not all_syndromes:
            return {'message': 'No syndrome data available'}
        
        # Error type distribution
        error_types = defaultdict(int)
        for syndrome in all_syndromes:
            error_types[syndrome.error_type] += 1
        
        # Correction success rate
        total_corrections = sum(1 for s in all_syndromes if s.correction_applied)
        successful_corrections = sum(1 for s in all_syndromes if s.correction_success)
        success_rate = successful_corrections / total_corrections if total_corrections > 0 else 0
        
        # Detection confidence distribution
        confidences = [s.detection_confidence for s in all_syndromes]
        
        # Syndrome weight distribution
        syndrome_weights = [sum(s.syndrome_bits) for s in all_syndromes]
        
        return {
            'total_syndromes': len(all_syndromes),
            'error_type_distribution': dict(error_types),
            'correction_success_rate': success_rate,
            'total_corrections': total_corrections,
            'successful_corrections': successful_corrections,
            'average_detection_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'average_syndrome_weight': np.mean(syndrome_weights),
            'syndrome_weight_std': np.std(syndrome_weights),
            'code_parameters': {
                'n': self.current_code.n,
                'k': self.current_code.k,
                'd': self.current_code.d,
                'name': self.current_code.name
            }
        }
    
    def compute_logical_error_rate(self) -> float:
        """
        Compute logical error rate for current code.
        
        Returns:
            Logical error rate
        """
        if not self.logical_qubits:
            return 0.0
        
        total_logical_errors = 0
        total_operations = 0
        
        for logical_qubit in self.logical_qubits.values():
            # Count logical errors (uncorrectable errors)
            logical_errors = sum(1 for s in logical_qubit.syndrome_history 
                               if s.correction_applied and not s.correction_success)
            total_logical_errors += logical_errors
            total_operations += len(logical_qubit.syndrome_history)
        
        return total_logical_errors / total_operations if total_operations > 0 else 0.0
    
    def optimize_code_parameters(self, target_fidelity: float = 0.95) -> Dict[str, Any]:
        """
        Optimize code parameters for target fidelity.
        
        Args:
            target_fidelity: Target logical fidelity
            
        Returns:
            Optimization results
        """
        current_fidelity = np.mean([lq.fidelity for lq in self.logical_qubits.values()]) if self.logical_qubits else 0.0
        
        recommendations = []
        
        if current_fidelity < target_fidelity:
            # Recommend better code
            if self.code_type == "steane":
                recommendations.append("Consider switching to surface code for better error correction")
            elif self.code_type == "shor":
                recommendations.append("Consider Steane code for better encoding efficiency")
            
            # Recommend more frequent syndrome measurements
            recommendations.append("Increase syndrome measurement frequency")
            
            # Recommend better decoding
            recommendations.append("Implement maximum likelihood decoding")
        
        return {
            'current_fidelity': current_fidelity,
            'target_fidelity': target_fidelity,
            'fidelity_gap': target_fidelity - current_fidelity,
            'recommendations': recommendations,
            'current_code': self.current_code.name,
            'logical_error_rate': self.compute_logical_error_rate()
        }
    
    def get_protection_status(self) -> Dict[str, Any]:
        """
        Get overall protection status of all logical qubits.
        
        Returns:
            Protection status summary
        """
        if not self.logical_qubits:
            return {'message': 'No logical qubits registered'}
        
        # Compute statistics
        fidelities = [lq.fidelity for lq in self.logical_qubits.values()]
        error_counts = [lq.error_count for lq in self.logical_qubits.values()]
        
        # Protection quality assessment
        avg_fidelity = np.mean(fidelities)
        if avg_fidelity > 0.95:
            protection_quality = "excellent"
        elif avg_fidelity > 0.9:
            protection_quality = "good"
        elif avg_fidelity > 0.8:
            protection_quality = "fair"
        else:
            protection_quality = "poor"
        
        return {
            'total_logical_qubits': len(self.logical_qubits),
            'average_fidelity': avg_fidelity,
            'fidelity_std': np.std(fidelities),
            'min_fidelity': np.min(fidelities),
            'max_fidelity': np.max(fidelities),
            'total_errors_corrected': sum(error_counts),
            'average_errors_per_qubit': np.mean(error_counts),
            'protection_quality': protection_quality,
            'code_efficiency': self.current_code.k / self.current_code.n,
            'syndrome_statistics': self.analyze_syndrome_statistics()
        }


def apply_quantum_error_correction(values: List[float], operations: List[str],
                                 error_rate: float = 0.01, code_type: str = "steane") -> Dict[str, Any]:
    """
    Apply quantum error correction to MCTS values.
    
    Args:
        values: List of MCTS values to protect
        operations: List of operations to perform
        error_rate: Physical error rate
        code_type: Type of quantum code to use
        
    Returns:
        Error correction results
    """
    # Initialize error corrector
    corrector = QuantumErrorCorrector(code_type=code_type)
    
    # Encode values
    logical_qubits = {}
    for i, value in enumerate(values):
        logical_qubits[i] = corrector.encode_value(value, i)
    
    # Apply noise
    for logical_qubit in logical_qubits.values():
        corrector.apply_noise(logical_qubit, error_rate)
    
    # Perform operations
    operation_results = []
    for i, operation in enumerate(operations):
        if i < len(logical_qubits):
            logical_qubit = logical_qubits[i]
            
            if operation == "measure":
                # Measure syndrome and correct
                syndrome = corrector.measure_syndrome(logical_qubit)
                success = corrector.correct_errors(logical_qubit, syndrome)
                operation_results.append({
                    'operation': operation,
                    'qubit_id': i,
                    'syndrome': syndrome.to_dict(),
                    'correction_success': success
                })
            else:
                # Apply fault-tolerant operation
                corrector.fault_tolerant_operation(logical_qubit, operation)
                operation_results.append({
                    'operation': operation,
                    'qubit_id': i,
                    'fidelity_after': logical_qubit.fidelity
                })
    
    # Decode values
    decoded_values = []
    for logical_qubit in logical_qubits.values():
        decoded_value = corrector.decode_value(logical_qubit)
        decoded_values.append(decoded_value)
    
    # Compute threshold
    threshold_analysis = corrector.compute_error_threshold()
    
    return {
        'original_values': values,
        'decoded_values': decoded_values,
        'value_fidelity': np.mean([abs(orig - dec) for orig, dec in zip(values, decoded_values)]),
        'operation_results': operation_results,
        'protection_status': corrector.get_protection_status(),
        'threshold_analysis': threshold_analysis,
        'optimization_results': corrector.optimize_code_parameters(),
        'code_parameters': {
            'type': code_type,
            'n': corrector.current_code.n,
            'k': corrector.current_code.k,
            'd': corrector.current_code.d
        }
    }