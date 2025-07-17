"""
Tensor Network MCTS Enhancement

This module implements tensor network methods for efficient representation
and manipulation of MCTS trees, inspired by quantum many-body physics.

Key concepts:
- Tree nodes as tensors with indices for parent/children/state
- Matrix Product State (MPS) compression for paths
- Tensor Renormalization Group (TRG) for coarse-graining
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Import unified quantum definitions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from quantum_definitions import (
    UnifiedQuantumDefinitions,
    MCTSQuantumState,
    compute_von_neumann_entropy,
    compute_purity,
    construct_quantum_state_from_visits
)

logger = logging.getLogger(__name__)


@dataclass
class TensorNode:
    """
    Represents an MCTS node as a tensor with multiple indices.
    
    Tensor structure: T[parent, state, child1, child2, ...]
    - parent: index connecting to parent node
    - state: internal state representation 
    - child_i: indices for each child connection
    """
    state_dim: int
    bond_dim: int
    max_children: int
    tensor: Optional[torch.Tensor] = None
    singular_values: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        """Initialize tensor with appropriate dimensions"""
        if self.tensor is None:
            # Initialize with random values (will be updated with actual data)
            shape = [self.bond_dim, self.state_dim] + [self.bond_dim] * self.max_children
            self.tensor = torch.randn(*shape) * 0.1
            
    def compress(self, threshold: float = 1e-10) -> 'TensorNode':
        """
        Compress tensor using SVD to reduce bond dimension.
        
        In the tensor network formalism, we view the MCTS node as a tensor with indices:
        T[parent, state, child_1, ..., child_n]
        
        The compression performs a decomposition that identifies the essential
        degrees of freedom while discarding negligible correlations.
        
        Args:
            threshold: Cutoff for singular values
            
        Returns:
            Compressed TensorNode
        """
        original_shape = self.tensor.shape
        
        # Group indices for bipartition: (parent, state) vs (children)
        # This preserves the hierarchical structure of MCTS
        left_indices = self.bond_dim * self.state_dim
        right_indices = np.prod(original_shape[2:])
        
        # Reshape into matrix for SVD
        matrix = self.tensor.reshape(left_indices, right_indices)
        
        # Perform SVD decomposition
        # U contains parent-state correlations
        # S contains importance weights (singular values)
        # Vh contains children correlations
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        
        # Identify significant singular values using threshold
        # This implements an information-theoretic cutoff
        significant = s > threshold * s[0]  # Relative threshold
        n_keep = significant.sum().item()
        
        if n_keep == 0:
            n_keep = 1  # Keep at least one mode
        
        if n_keep < len(s):
            # Truncate to significant modes
            u_truncated = u[:, :n_keep]
            s_truncated = s[:n_keep]
            vh_truncated = vh[:n_keep, :]
            
            # Reconstruct compressed tensor
            # This preserves the most important correlations
            compressed_matrix = u_truncated @ torch.diag(s_truncated) @ vh_truncated
            
            # Determine new bond dimension
            # The compressed bond dimension reflects the effective rank
            new_bond_dim = max(1, int(np.sqrt(n_keep)))  # Geometric mean approach
            
            # Reshape back to tensor form
            # We need to carefully handle dimension changes
            new_shape = [new_bond_dim, self.state_dim] + [new_bond_dim] * self.max_children
            
            # The compressed matrix might not fit exactly, so we embed it
            target_size = np.prod(new_shape)
            actual_size = compressed_matrix.numel()
            
            if actual_size >= target_size:
                # Truncate if too large
                self.tensor = compressed_matrix.flatten()[:target_size].reshape(new_shape)
            else:
                # Pad with zeros if too small (represents no correlation)
                padded = torch.zeros(target_size, device=compressed_matrix.device, 
                                   dtype=compressed_matrix.dtype)
                padded[:actual_size] = compressed_matrix.flatten()
                self.tensor = padded.reshape(new_shape)
            
            self.singular_values = s_truncated
            self.bond_dim = new_bond_dim
            
        return self
    
    def contract_with_child(self, child_tensor: torch.Tensor, child_index: int) -> torch.Tensor:
        """
        Contract this tensor with a child tensor along specified index.
        
        Args:
            child_tensor: Child node tensor
            child_index: Which child connection to use
            
        Returns:
            Contracted tensor
        """
        # Contract along the appropriate child dimension
        contract_dim = 2 + child_index  # Skip parent and state dims
        return torch.tensordot(self.tensor, child_tensor, dims=([contract_dim], [0]))


class MatrixProductState:
    """
    Represents a path in MCTS tree as a Matrix Product State (MPS).
    
    This allows efficient compression and manipulation of paths.
    """
    
    def __init__(self, path_nodes: List[TensorNode], bond_dim: int = 50):
        """
        Initialize MPS from a path of tensor nodes.
        
        Args:
            path_nodes: List of TensorNode objects along path
            bond_dim: Maximum bond dimension for compression
        """
        self.tensors = []
        self.bond_dim = bond_dim
        self.length = len(path_nodes)
        
        # Convert path to MPS form
        # In path integral formulation, we trace a specific path through the tree
        # Each tensor in the MPS represents the quantum amplitude at that node
        for i, node in enumerate(path_nodes):
            # Extract relevant slice based on position in path
            # The key insight: we're tracing a specific path through the tree
            # Each node tensor has shape [parent_bond, state, child_bonds...]
            
            # Handle variable tensor dimensions based on max_children
            # TensorNode shape is [bond_dim, state_dim] + [bond_dim] * max_children
            
            if i == 0:
                # First tensor: [1, state_dim, bond_dim]
                state_dim = node.state_dim
                right_bond = min(self.bond_dim, node.bond_dim)
                
                if node.max_children == 1:
                    # Tensor shape is [bond_dim, state_dim, bond_dim]
                    base_tensor = node.tensor[0, :state_dim, :right_bond]  # [state_dim, right_bond]
                else:
                    # Tensor has more child dimensions
                    base_tensor = node.tensor[0, :state_dim, :right_bond, 0]  # [state_dim, right_bond]
                    
                tensor = base_tensor.unsqueeze(0)  # [1, state_dim, right_bond]
                
            elif i == len(path_nodes) - 1:
                # Last tensor: [bond_dim, state_dim, 1]
                left_bond = min(self.bond_dim, node.bond_dim)
                state_dim = node.state_dim
                
                if node.max_children == 1:
                    # Tensor shape is [bond_dim, state_dim, bond_dim]
                    base_tensor = node.tensor[:left_bond, :state_dim, 0]  # [left_bond, state_dim]
                else:
                    # Tensor has more child dimensions
                    base_tensor = node.tensor[:left_bond, :state_dim, 0, 0]  # [left_bond, state_dim]
                    
                tensor = base_tensor.unsqueeze(2)  # [left_bond, state_dim, 1]
                
            else:
                # Middle tensors: [bond_dim, state_dim, bond_dim]
                left_bond = min(self.bond_dim, node.bond_dim)
                state_dim = node.state_dim
                right_bond = min(self.bond_dim, node.bond_dim)
                
                if node.max_children == 1:
                    # Tensor shape is [bond_dim, state_dim, bond_dim]
                    tensor = node.tensor[:left_bond, :state_dim, :right_bond]
                else:
                    # Select first child path
                    tensor = node.tensor[:left_bond, :state_dim, :right_bond, 0]
                
            # Ensure tensor has exactly 3 dimensions
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0 if i == 0 else 2)
            elif tensor.ndim > 3:
                # Squeeze extra dimensions
                while tensor.ndim > 3:
                    # Find singleton dimension to squeeze
                    for dim in range(tensor.ndim):
                        if tensor.shape[dim] == 1 and dim > 2:
                            tensor = tensor.squeeze(dim)
                            break
                    else:
                        # No singleton found, reshape
                        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], -1)[:, :, :1]
                        
            self.tensors.append(tensor)
    
    def compress(self, max_bond: Optional[int] = None) -> 'MatrixProductState':
        """
        Compress MPS using canonical form and SVD truncation.
        
        This implements the canonical MPS compression algorithm:
        1. Left-canonical sweep: brings MPS into left-canonical form
        2. SVD truncation: keeps only significant singular values
        3. Information preservation: maintains quantum correlations
        
        The compression is crucial for computational efficiency while
        preserving the essential quantum information encoded in the path.
        
        Args:
            max_bond: Maximum bond dimension after compression
            
        Returns:
            Compressed MPS
        """
        if max_bond is None:
            max_bond = self.bond_dim
            
        # Left-to-right canonicalization sweep
        for i in range(len(self.tensors) - 1):
            # Current tensor has shape [left_bond, physical, right_bond]
            tensor = self.tensors[i]
            shape = tensor.shape
            
            # Validate tensor structure
            if len(shape) != 3:
                # Skip malformed tensors
                continue
                
            left_dim, phys_dim, right_dim = shape
            
            # Reshape for SVD: group (left, physical) indices
            # This preserves the local Hilbert space structure
            matrix = tensor.reshape(left_dim * phys_dim, right_dim)
            
            # Perform SVD decomposition
            # U: left-canonical tensor component
            # S: singular values (entanglement spectrum)
            # Vh: to be absorbed into next tensor
            try:
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            except RuntimeError:
                # Handle numerical instabilities
                continue
            
            # Determine truncation based on singular value spectrum
            # This implements an adaptive bond dimension reduction
            cumulative_weight = torch.cumsum(s**2, dim=0) / torch.sum(s**2)
            
            # Keep modes until we capture sufficient variance (e.g., 99.9%)
            variance_threshold = 0.999
            keep_variance = (cumulative_weight < variance_threshold).sum().item() + 1
            
            # Apply bond dimension constraint
            keep = min(max_bond, len(s), keep_variance)
            keep = max(1, keep)  # Ensure at least rank-1
            
            # Truncate to selected rank
            u_truncated = u[:, :keep]
            s_truncated = s[:keep]
            vh_truncated = vh[:keep, :]
            
            # Reshape U back to tensor form
            # The new right bond dimension is 'keep'
            new_u_shape = (left_dim, phys_dim, keep)
            
            # Proper reshaping that preserves tensor structure
            u_reshaped = u_truncated.reshape(new_u_shape)
            self.tensors[i] = u_reshaped
            
            # Create matrix to absorb into next tensor
            # This transfers the singular values and remaining correlations
            sv_matrix = torch.diag(s_truncated) @ vh_truncated
            
            # Update next tensor by contracting with sv_matrix
            next_tensor = self.tensors[i + 1]
            next_shape = next_tensor.shape
            
            if len(next_shape) == 3:
                # Standard MPS tensor
                # Contract sv_matrix with left bond of next tensor
                next_left, next_phys, next_right = next_shape
                
                # Check dimension compatibility
                if sv_matrix.shape[1] != next_left:
                    # Dimension mismatch - need to handle bond dimension change
                    if sv_matrix.shape[1] < next_left:
                        # Pad sv_matrix
                        padding = torch.zeros(sv_matrix.shape[0], next_left - sv_matrix.shape[1], 
                                            device=sv_matrix.device, dtype=sv_matrix.dtype)
                        sv_matrix = torch.cat([sv_matrix, padding], dim=1)
                    else:
                        # Truncate next tensor's left dimension
                        next_tensor = next_tensor[:sv_matrix.shape[1], :, :]
                        next_left = sv_matrix.shape[1]
                
                # Reshape next tensor for contraction
                next_matrix = next_tensor.reshape(next_left, next_phys * next_right)
                
                # Contract: sv_matrix @ next_matrix
                contracted = sv_matrix @ next_matrix
                
                # Reshape back to tensor form
                self.tensors[i + 1] = contracted.reshape(keep, next_phys, next_right)
            else:
                # Handle edge cases
                self.tensors[i + 1] = sv_matrix @ next_tensor.flatten()
                self.tensors[i + 1] = self.tensors[i + 1].reshape(keep, -1, 1)
                
        return self
    
    def compute_path_amplitude(self) -> torch.Tensor:
        """
        Compute the amplitude (probability) of this path.
        
        Returns:
            Scalar amplitude value
        """
        if len(self.tensors) == 0:
            return torch.tensor(0.0)
            
        # Contract all tensors in the MPS
        result = self.tensors[0]
        
        for i in range(1, len(self.tensors)):
            # Contract along the bond dimension
            if result.ndim >= 1 and self.tensors[i].ndim >= 1:
                result = torch.tensordot(result, self.tensors[i], dims=([-1], [0]))
            else:
                result = result * self.tensors[i]
                
        # Ensure scalar output by summing over all remaining dimensions
        while result.ndim > 0:
            result = result.sum(dim=-1)
            
        return result


class TensorRenormalizationGroup:
    """
    Implements Tensor Renormalization Group (TRG) for coarse-graining MCTS trees.
    
    This allows efficient handling of very deep trees by systematically
    combining layers while preserving important information.
    """
    
    def __init__(self, cutoff: float = 1e-8):
        """
        Initialize TRG.
        
        Args:
            cutoff: Truncation threshold for singular values
        """
        self.cutoff = cutoff
        
    def coarse_grain_layer(self, layer_tensors: List[TensorNode]) -> List[TensorNode]:
        """
        Apply one TRG step to coarse-grain a tree layer.
        
        Args:
            layer_tensors: Tensors at a given tree depth
            
        Returns:
            Coarse-grained tensors for next level
        """
        coarse_grained = []
        
        # Group tensors in pairs for coarse-graining
        for i in range(0, len(layer_tensors), 2):
            if i + 1 < len(layer_tensors):
                # Contract pair of tensors
                tensor1 = layer_tensors[i]
                tensor2 = layer_tensors[i + 1]
                
                # Contract along appropriate dimensions
                contracted = self._contract_pair(tensor1, tensor2)
                
                # Compress the result
                compressed = self._compress_tensor(contracted)
                coarse_grained.append(compressed)
            else:
                # Odd tensor, keep as is
                coarse_grained.append(layer_tensors[i])
                
        return coarse_grained
    
    def _contract_pair(self, tensor1: TensorNode, tensor2: TensorNode) -> torch.Tensor:
        """Contract two adjacent tensors"""
        # Simplified contraction - in practice would depend on tree structure
        t1 = tensor1.tensor
        t2 = tensor2.tensor
        
        # Contract along a shared dimension
        return torch.tensordot(t1, t2, dims=([-1], [0]))
    
    def _compress_tensor(self, tensor: torch.Tensor) -> TensorNode:
        """Compress tensor using SVD"""
        # Get shape info
        shape = tensor.shape
        if len(shape) < 2:
            # Need at least 2D tensor
            shape = (1,) + shape
            tensor = tensor.reshape(shape)
            
        # Ensure we have valid dimensions
        state_dim = shape[1] if len(shape) > 1 else 1
        
        # Reshape for SVD
        matrix = tensor.reshape(shape[0] * state_dim, -1)
        
        # Check matrix dimensions
        if matrix.numel() == 0 or min(matrix.shape) == 0:
            # Return trivial node
            node = TensorNode(state_dim=state_dim, bond_dim=1, max_children=2)
            return node
        
        # SVD with truncation
        try:
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        except:
            # Return trivial node on SVD failure
            node = TensorNode(state_dim=state_dim, bond_dim=1, max_children=2)
            return node
        
        # Keep only significant singular values
        keep_mask = s > self.cutoff
        n_keep = keep_mask.sum().item()
        
        if n_keep == 0:
            n_keep = 1  # Keep at least one
            
        u = u[:, :n_keep]
        s = s[:n_keep]
        vh = vh[:n_keep, :]
        
        # Create new TensorNode with appropriate dimensions
        bond_dim = n_keep
        node = TensorNode(
            state_dim=state_dim,
            bond_dim=bond_dim,
            max_children=2
        )
        
        # Reconstruct compressed tensor with correct shape
        compressed = u @ torch.diag(s) @ vh
        
        # Target shape for the node tensor
        target_shape = [bond_dim, state_dim, bond_dim, bond_dim]
        target_size = np.prod(target_shape)
        
        # Pad or truncate as needed
        if compressed.numel() >= target_size:
            node.tensor = compressed.flatten()[:target_size].reshape(target_shape)
        else:
            padded = torch.zeros(target_size, device=compressed.device)
            padded[:compressed.numel()] = compressed.flatten()
            node.tensor = padded.reshape(target_shape)
            
        node.singular_values = s
        
        return node


class TensorNetworkMCTS:
    """
    Main class for Tensor Network enhanced MCTS.
    
    Integrates tensor methods into standard MCTS for improved efficiency
    and better handling of deep trees.
    """
    
    def __init__(self, state_dim: int = 64, bond_dim: int = 50, 
                 compression_threshold: float = 0.9, device: Optional[str] = None):
        """
        Initialize TN-MCTS.
        
        Args:
            state_dim: Dimension of state representation
            bond_dim: Maximum bond dimension for tensors
            compression_threshold: When to trigger compression (based on memory)
            device: Computation device
        """
        self.state_dim = state_dim
        self.bond_dim = bond_dim
        self.compression_threshold = compression_threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize unified quantum definitions
        self.quantum_defs = UnifiedQuantumDefinitions(device=self.device)
        self.trg = TensorRenormalizationGroup()
        
        # Cache for tensor representations
        self.tensor_cache = {}
        
    def create_tensor_node(self, mcts_node) -> TensorNode:
        """
        Convert MCTS node to tensor representation.
        
        Args:
            mcts_node: Standard MCTS node
            
        Returns:
            TensorNode representation
        """
        # Extract relevant information from MCTS node
        n_children = len(mcts_node.children) if hasattr(mcts_node, 'children') else 0
        
        # Create tensor node
        tensor_node = TensorNode(
            state_dim=self.state_dim,
            bond_dim=self.bond_dim,
            max_children=max(n_children, 2)
        )
        
        # Encode node information into tensor
        # This is problem-specific - here's a generic encoding
        if hasattr(mcts_node, 'value_sum') and hasattr(mcts_node, 'visit_count'):
            value = mcts_node.value_sum / (mcts_node.visit_count + 1)
            visits = np.log(mcts_node.visit_count + 1)
            
            # Encode into state dimension
            tensor_node.tensor[0, :self.state_dim//2] = value
            tensor_node.tensor[0, self.state_dim//2:] = visits
            
        return tensor_node
    
    def compress_path(self, path_nodes: List) -> MatrixProductState:
        """
        Compress a path through the tree using MPS.
        
        Args:
            path_nodes: List of MCTS nodes along path
            
        Returns:
            Compressed MPS representation
        """
        # Convert to tensor nodes
        tensor_nodes = [self.create_tensor_node(node) for node in path_nodes]
        
        # Create and compress MPS
        mps = MatrixProductState(tensor_nodes, self.bond_dim)
        mps.compress()
        
        return mps
    
    def apply_trg_to_tree(self, tree_layers: List[List]) -> List[List]:
        """
        Apply TRG to entire tree structure.
        
        Args:
            tree_layers: Tree organized by depth layers
            
        Returns:
            Coarse-grained tree layers
        """
        coarse_grained_layers = []
        
        for layer in tree_layers:
            # Convert to tensor nodes
            tensor_layer = [self.create_tensor_node(node) for node in layer]
            
            # Apply TRG
            coarse_layer = self.trg.coarse_grain_layer(tensor_layer)
            coarse_grained_layers.append(coarse_layer)
            
        return coarse_grained_layers
    
    def compute_entanglement_spectrum(self, partition_nodes: List) -> Dict[str, Any]:
        """
        Compute entanglement spectrum across a tree partition.
        
        Args:
            partition_nodes: Nodes defining the partition boundary
            
        Returns:
            Entanglement spectrum and related quantities
        """
        if len(partition_nodes) == 0:
            return {
                'eigenvalues': torch.tensor([1.0]),
                'entanglement_entropy': torch.tensor(0.0),
                'schmidt_rank': 1,
                'participation_ratio': 1.0
            }
            
        # Collect visit counts from partition nodes
        visit_counts = []
        for node in partition_nodes:
            if hasattr(node, 'visit_count'):
                visit_counts.append(float(node.visit_count))
            else:
                visit_counts.append(1.0)
                
        visit_counts = torch.tensor(visit_counts, dtype=torch.float32, device=self.device)
        
        # Construct quantum state using unified definitions
        # This naturally creates mixed states with off-diagonal terms
        quantum_state = self.quantum_defs.construct_quantum_state_from_single_visits(
            visit_counts, outcome_uncertainty=0.1
        )
        
        # Get density matrix and compute entropy using unified definitions
        density_matrix = quantum_state.density_matrix
        entropy = compute_von_neumann_entropy(density_matrix)
        purity = compute_purity(density_matrix)
        
        # Compute participation ratio from eigenvalues
        eigenvalues = torch.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        participation = 1.0 / torch.sum(eigenvalues**2) if len(eigenvalues) > 0 else 1.0
        
        # Participation ratio
        participation = 1.0 / torch.sum(eigenvalues**2) if len(eigenvalues) > 0 else 1.0
        
        return {
            'eigenvalues': eigenvalues.tolist(),
            'entanglement_entropy': float(entropy),
            'schmidt_rank': len(eigenvalues),
            'participation_ratio': float(participation)
        }
    
    def should_compress(self) -> bool:
        """Check if compression should be triggered based on memory usage"""
        # Simplified check - in practice would monitor actual memory
        return len(self.tensor_cache) > 1000 * self.compression_threshold
    
    def clear_cache(self):
        """Clear tensor cache to free memory"""
        self.tensor_cache.clear()


def integrate_with_mcts(mcts_tree, config: Dict[str, Any]) -> TensorNetworkMCTS:
    """
    Integrate tensor network methods with existing MCTS tree.
    
    Args:
        mcts_tree: Existing MCTS tree structure
        config: Configuration parameters
        
    Returns:
        TensorNetworkMCTS instance
    """
    tn_mcts = TensorNetworkMCTS(
        state_dim=config.get('state_dim', 64),
        bond_dim=config.get('bond_dim', 50),
        compression_threshold=config.get('compression_threshold', 0.9)
    )
    
    # Convert high-value nodes to tensor representation
    def convert_subtree(node, depth=0, max_depth=10):
        if depth > max_depth or not hasattr(node, 'children'):
            return
            
        # Convert high-value nodes
        if hasattr(node, 'visit_count') and node.visit_count > 100:
            tensor_node = tn_mcts.create_tensor_node(node)
            tn_mcts.tensor_cache[id(node)] = tensor_node
            
        # Recurse
        for child in node.children.values():
            convert_subtree(child, depth + 1, max_depth)
            
    convert_subtree(mcts_tree)
    
    return tn_mcts