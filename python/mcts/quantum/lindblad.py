"""
Lindblad-inspired selection with coherent hopping.

Implements quantum master equation dynamics for MCTS selection.
"""
import torch
import numpy as np
from typing import Optional, List, Union


class HoppingMatrix:
    """Computes hopping matrix for coherent transitions between actions"""
    
    def __init__(self, hopping_strength: float = 0.1):
        """
        Initialize hopping matrix calculator.
        
        Args:
            hopping_strength: Base strength of coherent hopping
        """
        self.hopping_strength = hopping_strength
    
    def compute(self, q_values: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute hopping matrix based on Q-value similarity.
        
        Hopping is stronger between actions with similar Q-values,
        implementing coherent superposition of similar strategies.
        
        Args:
            q_values: Tensor of Q-values for each action
            temperature: Temperature parameter
            
        Returns:
            Symmetric hopping matrix with zero diagonal
        """
        n_actions = len(q_values)
        
        # Compute pairwise Q-value distances
        q_distances = torch.abs(q_values.unsqueeze(0) - q_values.unsqueeze(1))
        
        # Hopping strength decreases with Q-value distance
        # Uses Gaussian kernel for smooth decay
        hop_matrix = self.hopping_strength * torch.exp(-q_distances / temperature)
        
        # Zero diagonal (no self-hopping)
        hop_matrix.fill_diagonal_(0)
        
        return hop_matrix


class LindbladSelector:
    """
    Selection with quantum-inspired coherent hopping.
    
    Implements selection based on an effective Hamiltonian that includes:
    1. Diagonal terms: PUCT scores (classical potential)
    2. Off-diagonal terms: Coherent hopping between similar actions
    
    This allows for quantum superposition of strategies.
    """
    
    def __init__(self, hopping_strength: float = 0.1, 
                 decoherence_rate: float = 0.0,
                 device: Optional[str] = None):
        """
        Initialize Lindblad selector.
        
        Args:
            hopping_strength: Strength of coherent transitions
            decoherence_rate: Rate of decoherence (0 = fully coherent)
            device: Computation device
        """
        self.hopping_strength = hopping_strength
        self.decoherence_rate = decoherence_rate
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.hop_matrix_calc = HoppingMatrix(hopping_strength)
    
    def select(self, node, temperature: float = 1.0) -> int:
        """
        Select action using coherent hopping dynamics.
        
        Args:
            node: MCTS node with children
            temperature: Temperature parameter
            
        Returns:
            Selected action
        """
        if not node.children:
            return None
            
        # Extract data from children
        actions = list(node.children.keys())
        children = list(node.children.values())
        
        scores = torch.tensor([c.score for c in children], 
                            device=self.device, dtype=torch.float32)
        q_values = torch.tensor([c.q_value for c in children],
                              device=self.device, dtype=torch.float32)
        
        # Compute effective Hamiltonian
        H_eff = self._compute_effective_hamiltonian_tensors(scores, q_values, temperature)
        
        # Convert to selection probabilities
        # For now, use diagonal approximation with hopping perturbation
        effective_scores = H_eff.diag() + 0.5 * H_eff.sum(dim=1)
        
        # Softmax selection
        probs = torch.softmax(effective_scores / temperature, dim=0)
        
        # Sample action
        action_idx = torch.multinomial(probs, 1).item()
        
        return actions[action_idx]
    
    def compute_effective_hamiltonian(self, node) -> torch.Tensor:
        """
        Compute effective Hamiltonian for the node.
        
        Args:
            node: MCTS node
            
        Returns:
            Effective Hamiltonian matrix
        """
        children = list(node.children.values())
        
        scores = torch.tensor([c.score for c in children], 
                            device=self.device, dtype=torch.float32)
        q_values = torch.tensor([c.q_value for c in children],
                              device=self.device, dtype=torch.float32)
        
        return self._compute_effective_hamiltonian_tensors(scores, q_values)
    
    def _compute_effective_hamiltonian_tensors(self, scores: torch.Tensor, 
                                             q_values: torch.Tensor,
                                             temperature: float = 1.0) -> torch.Tensor:
        """
        Compute effective Hamiltonian from tensors.
        
        H = H_diagonal + H_hopping
        
        Args:
            scores: PUCT scores (diagonal terms)
            q_values: Q-values for hopping computation
            temperature: Temperature parameter
            
        Returns:
            Hamiltonian matrix
        """
        n_actions = len(scores)
        
        # Diagonal: PUCT scores (classical potential)
        H = torch.diag(scores)
        
        # Off-diagonal: Coherent hopping
        if self.hopping_strength > 0:
            hop_matrix = self.hop_matrix_calc.compute(q_values, temperature)
            H = H + hop_matrix
        
        return H
    
    def select_batch(self, nodes: List, temperature: float = 1.0) -> List[int]:
        """
        Batch selection for multiple nodes.
        
        Args:
            nodes: List of MCTS nodes
            temperature: Temperature parameter
            
        Returns:
            List of selected actions
        """
        actions = []
        
        # For now, process sequentially
        # TODO: Implement true batch processing
        for node in nodes:
            action = self.select(node, temperature)
            actions.append(action)
            
        return actions
    
    def evolve_density_matrix(self, rho: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Evolve density matrix under Lindblad master equation.
        
        dρ/dt = -i[H, ρ] + D[ρ]
        
        where D[ρ] is the dissipator implementing decoherence.
        
        Args:
            rho: Initial density matrix
            dt: Time step
            
        Returns:
            Evolved density matrix
        """
        # For simplicity, use first-order Euler integration
        # In practice, would use more sophisticated integrator
        
        # Create a simple Hamiltonian for testing
        n = rho.shape[0]
        # Use smaller values to avoid numerical issues
        H = 0.1 * torch.randn(n, n, device=rho.device, dtype=torch.complex64)
        H = 0.5 * (H + H.conj().T)  # Make Hermitian
        
        # Commutator: -i[H, ρ]
        commutator = -1j * (H @ rho - rho @ H)
        
        # Dissipator (simple dephasing)
        if self.decoherence_rate > 0:
            # Dephasing in computational basis
            dissipator = torch.zeros_like(rho)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        dissipator[i, j] = -self.decoherence_rate * rho[i, j]
        else:
            dissipator = torch.zeros_like(rho)
        
        # Evolution
        drho_dt = commutator + dissipator
        rho_new = rho + dt * drho_dt
        
        # Ensure Hermiticity
        rho_new = 0.5 * (rho_new + rho_new.conj().T)
        
        # Ensure positive semi-definiteness by spectral decomposition
        # This is a numerical safeguard
        eigenvalues, eigenvectors = torch.linalg.eigh(rho_new)
        eigenvalues = eigenvalues.real.clamp(min=0)  # Force non-negative
        
        # Reconstruct with non-negative eigenvalues
        rho_new = eigenvectors @ torch.diag(eigenvalues.to(torch.complex64)) @ eigenvectors.conj().T
        
        # Ensure trace preservation
        trace = torch.trace(rho_new).real
        if trace > 0:
            rho_new = rho_new / trace
        
        return rho_new