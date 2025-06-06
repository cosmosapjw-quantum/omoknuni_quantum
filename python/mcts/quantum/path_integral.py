"""Unified path integral formulation for MCTS

This module implements the quantum-inspired path integral formulation that
provides a variational principle for path selection in MCTS.
Includes both CPU and GPU implementations with automatic selection.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set, Union, Any
import numpy as np
from dataclasses import dataclass
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PathIntegralConfig:
    """Configuration for path integral formulation"""
    # Basic parameters
    temperature: float = 1.0
    phase_coupling: float = 0.1
    regularization: float = 0.01
    use_complex_action: bool = True
    max_path_length: int = 50
    
    # Physics parameters (from docs)
    hbar_eff: float = 1.0  # Effective Planck constant (ℏ_eff)
    beta: float = 1.0  # Inverse temperature (β)
    quantum_strength: float = 0.1  # Strength of quantum corrections
    
    # Advanced parameters (GPU mode)
    hbar: float = 1.0  # Reduced Planck constant
    mass: float = 1.0  # Effective mass in path space
    dt: float = 0.1  # Time step for discretization
    
    # Potential parameters
    value_potential_scale: float = 1.0
    visit_potential_scale: float = 0.5
    uncertainty_potential_scale: float = 0.3
    
    # Path sampling
    num_paths_sample: int = 1000
    
    # Variational parameters
    use_variational: bool = True
    num_variational_steps: int = 10
    learning_rate: float = 0.01
    
    # Quantum effects
    enable_tunneling: bool = True
    tunneling_strength: float = 0.1


class Path:
    """Represents a path through the MCTS tree (CPU mode)"""
    
    def __init__(self, nodes: List['Node'], actions: List[int]):
        """Initialize path
        
        Args:
            nodes: List of nodes in the path
            actions: List of actions taken
        """
        self.nodes = nodes
        self.actions = actions
        self.length = len(nodes)
        
        # Cached values
        self._action_value: Optional[complex] = None
        self._probability_amplitude: Optional[complex] = None
        
    def get_leaf(self) -> 'Node':
        """Get the leaf node of this path"""
        return self.nodes[-1] if self.nodes else None
        
    def get_visits(self) -> int:
        """Get total visits along path"""
        return sum(node.visit_count for node in self.nodes)
        
    def get_value(self) -> float:
        """Get path value (average of node values)"""
        if not self.nodes:
            return 0.0
        values = [node.value() for node in self.nodes if node.visit_count > 0]
        return np.mean(values) if values else 0.0


class PathIntegralMCTS:
    """Unified path integral formulation with CPU/GPU support
    
    This implements the discretized path integral approach where:
    - Action S[path] = ∫ dt L(x,ẋ,t) for GPU mode
    - Action S[path] = -log(N[path]) + iφ[path] for CPU mode
    - Probability ∝ exp(-S[path]/T)
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        config: Optional[PathIntegralConfig] = None
    ):
        """Initialize path integral MCTS
        
        Args:
            device: Device for computation
            config: Path integral configuration
        """
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.config = config or PathIntegralConfig()
        self.use_gpu = device.type == 'cuda'
        
        if self.use_gpu:
            logger.info("Using GPU-accelerated path integral formulation")
            self._init_gpu_state()
        else:
            logger.info("Using CPU path integral implementation")
            self._init_cpu_state()
        
        # Statistics
        self.stats = {
            'paths_evaluated': 0,
            'variational_steps': 0,
            'optimization_steps': 0,
            'avg_action': 0.0,
            'tunneling_events': 0
        }
    
    def _init_gpu_state(self):
        """Initialize GPU-specific state"""
        # Path cache for efficiency
        self.path_cache = {}
        self.action_cache = {}
    
    def _init_cpu_state(self):
        """Initialize CPU-specific state"""
        self.path_cache: Dict[str, Path] = {}
        self.action_cache: Dict[str, complex] = {}
    
    def compute_action_values(
        self,
        tree,
        root_idx: int,
        child_indices: Union[torch.Tensor, List[int]]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Compute action values using path integral
        
        Main interface for MCTS to get quantum-corrected action values.
        
        Args:
            tree: Tree structure (CSR tree for GPU, Node tree for CPU)
            root_idx: Root node index
            child_indices: Child node indices to evaluate
            
        Returns:
            Action values for each child
        """
        if self.use_gpu:
            # GPU implementation
            if not isinstance(child_indices, torch.Tensor):
                child_indices = torch.tensor(child_indices, device=self.device)
            return self._compute_action_values_gpu(tree, root_idx, child_indices)
        else:
            # CPU implementation
            return self._compute_action_values_cpu(tree, root_idx, child_indices)
    
    def _compute_action_values_gpu(
        self,
        tree,
        root_idx: int,
        child_indices: torch.Tensor
    ) -> torch.Tensor:
        """GPU implementation using full path integral"""
        num_children = len(child_indices)
        
        # Fully vectorized path sampling - all children processed in parallel
        all_paths, child_mapping = self._vectorized_multi_child_sampling(child_indices, tree, num_children)
        child_mapping = torch.tensor(child_mapping, device=self.device)
        
        # Get values and visits
        values = tree.value_sums[all_paths] / (tree.visit_counts[all_paths] + 1e-6)
        visits = tree.visit_counts[all_paths]
        
        # Mask invalid entries
        valid_mask = all_paths >= 0
        values = values * valid_mask
        visits = visits * valid_mask
        
        # Compute actions for all paths
        if self.config.use_variational:
            # Optimize paths
            optimized_paths = self._variational_optimization(
                all_paths.float(), values, visits
            )
            actions = self._compute_path_action_gpu(
                optimized_paths, values, visits
            )
        else:
            actions = self._compute_path_action_gpu(all_paths, values, visits)
        
        # Compute probabilities
        probabilities = self._compute_path_probability_gpu(actions)
        
        # Vectorized child aggregation (much faster than sequential loop)
        child_values = self._vectorized_child_aggregation(probabilities, values, child_mapping, num_children)
        
        return child_values
    
    def _vectorized_child_aggregation(
        self,
        probabilities: torch.Tensor,
        values: torch.Tensor,
        child_mapping: torch.Tensor,
        num_children: int
    ) -> torch.Tensor:
        """Vectorized aggregation using scatter operations"""
        # Use scatter_add for efficient aggregation
        child_values = torch.zeros(num_children, device=self.device)
        
        # Ensure indices are valid
        valid_mask = (child_mapping >= 0) & (child_mapping < num_children)
        valid_mapping = child_mapping[valid_mask]
        valid_probs = probabilities[valid_mask]
        
        if len(valid_mapping) == 0:
            return child_values
        
        # Handle values dimension
        if values.dim() > 1:
            valid_values = values[valid_mask].mean(dim=-1)
        else:
            valid_values = values[valid_mask] if valid_mask.sum() <= len(values) else values[:valid_mask.sum()]
        
        # Ensure compatible lengths
        min_len = min(len(valid_probs), len(valid_values))
        if min_len > 0:
            # Vectorized weighted aggregation using scatter_add
            weighted_values = valid_probs[:min_len] * valid_values[:min_len]
            child_values.scatter_add_(0, valid_mapping[:min_len], weighted_values)
        
        return child_values
    
    def _vectorized_multi_child_sampling(
        self,
        child_indices: torch.Tensor,
        tree,
        num_children: int
    ) -> Tuple[torch.Tensor, List[int]]:
        """Vectorized sampling for all children simultaneously"""
        paths_per_child = self.config.num_paths_sample // num_children
        max_path_length = min(self.config.max_path_length, 20)
        
        # Pre-allocate output tensors for maximum efficiency
        total_paths = num_children * paths_per_child
        all_paths = torch.full(
            (total_paths, max_path_length), -1,
            dtype=torch.long, device=self.device
        )
        
        # Create child mapping efficiently
        child_mapping = []
        for i in range(num_children):
            child_mapping.extend([i] * paths_per_child)
        
        # Vectorized initialization - all paths start at their respective children
        for i, child_idx in enumerate(child_indices):
            start_idx = i * paths_per_child
            end_idx = start_idx + paths_per_child
            all_paths[start_idx:end_idx, 0] = child_idx.item()
        
        # Parallel path extension for all paths simultaneously
        active_mask = torch.ones(total_paths, dtype=torch.bool, device=self.device)
        
        for depth in range(1, max_path_length):
            if not active_mask.any():
                break
            
            # Get all current nodes for active paths
            current_nodes = all_paths[active_mask, depth - 1]
            
            # Batch UCB selection for ALL paths at once (mega-vectorization)
            next_nodes = self._vectorized_ucb_selection(current_nodes, tree, active_mask.sum().item())
            
            # Update all active paths - ensure matching dtypes
            all_paths[active_mask, depth] = next_nodes.to(all_paths.dtype)
            
            # Update active mask
            valid_next = (next_nodes >= 0) & (next_nodes < tree.num_nodes)
            new_active = torch.zeros(total_paths, dtype=torch.bool, device=self.device)
            new_active[active_mask] = valid_next
            
            # Random stopping (vectorized for all paths)
            stop_prob = torch.rand(new_active.sum(), device=self.device) < 0.05
            if stop_prob.any():
                stop_indices = torch.where(new_active)[0][stop_prob]
                new_active[stop_indices] = False
            
            active_mask = new_active
        
        return all_paths, child_mapping
    
    def _compute_action_values_cpu(
        self,
        tree,
        root_idx: int,
        child_indices: List[int]
    ) -> np.ndarray:
        """CPU implementation using simplified path integral"""
        # Handle different tree structures
        if hasattr(tree, 'nodes') and root_idx in tree.nodes:
            root_node = tree.nodes[root_idx]
        else:
            # Fallback - return uniform values
            num_children = len(child_indices)
            return np.ones(num_children) / num_children
            
        # Get quantum corrections for each child
        corrections = self._get_quantum_corrections(root_node)
        
        # Convert to action values
        values = []
        for i, child_idx in enumerate(child_indices):
            if child_idx in corrections:
                values.append(1.0 + corrections[child_idx])
            else:
                values.append(1.0)
                
        values = np.array(values)
        # Normalize
        values = values / values.sum()
        
        return values
    
    def _compute_path_action_gpu(
        self,
        paths: torch.Tensor,
        values: torch.Tensor,
        visits: torch.Tensor
    ) -> torch.Tensor:
        """Compute physically correct complete action (classical + quantum)
        
        Based on docs: S[π] = S_R[π] + iS_I[π] where:
        - S_R[π] = -∑ᵢ log N(sᵢ, aᵢ) (real part: visit frequency)
        - S_I[π] = β·σ²(V[π]) (imaginary part: value uncertainty)
        """
        batch_size, path_length = paths.shape
        
        # Safety check: ensure paths contain valid indices
        valid_mask = (paths >= 0) & (paths < values.shape[0])
        safe_paths = torch.clamp(paths, 0, values.shape[0] - 1)
        
        # Get path data safely with proper shape handling
        try:
            # Ensure safe_paths is properly shaped
            if safe_paths.dim() == 2:
                safe_paths_flat = safe_paths.flatten()
                max_idx = min(values.shape[0] - 1, safe_paths_flat.max().item())
                safe_paths_clamped = torch.clamp(safe_paths_flat, 0, max_idx)
                
                # Gather values and reshape
                path_values_flat = values[safe_paths_clamped]
                path_visits_flat = visits[safe_paths_clamped]
                
                path_values = path_values_flat.view(batch_size, path_length)
                path_visits = path_visits_flat.view(batch_size, path_length)
            else:
                path_values = torch.zeros(batch_size, path_length, device=self.device)
                path_visits = torch.ones(batch_size, path_length, device=self.device)
                
        except (IndexError, RuntimeError) as e:
            # Fallback for any indexing issues
            path_values = torch.zeros(batch_size, path_length, device=self.device)
            path_visits = torch.ones(batch_size, path_length, device=self.device)
        
        # Apply validity mask (ensure shapes match)
        if path_values.shape == valid_mask.shape:
            path_values = path_values * valid_mask.float()
            path_visits = path_visits * valid_mask.float() + (~valid_mask).float()
        else:
            # If shapes don't match, create compatible tensors
            path_values = torch.zeros(batch_size, path_length, device=self.device)
            path_visits = torch.ones(batch_size, path_length, device=self.device)
        
        # Fused action computation for maximum GPU efficiency
        action = self._fused_action_kernel(path_values, path_visits, valid_mask)
        
        self.stats['paths_evaluated'] += batch_size
        self.stats['avg_action'] = action.mean().item()
        
        return action
        
    def _fused_action_kernel(self, path_values: torch.Tensor, path_visits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Fused GPU kernel for complete action computation"""
        # Get config parameters with defaults
        beta = getattr(self.config, 'beta', 1.0)
        hbar_eff = getattr(self.config, 'hbar_eff', 1.0)
        quantum_strength = getattr(self.config, 'quantum_strength', 0.1)
        
        # Try to use Triton if available for maximum performance
        try:
            return self._triton_fused_action(path_values, path_visits, valid_mask, beta, hbar_eff, quantum_strength)
        except (ImportError, AttributeError):
            # Fallback to PyTorch operations (still fused where possible)
            return self._pytorch_fused_action(path_values, path_visits, valid_mask, beta, hbar_eff, quantum_strength)
    
    def _pytorch_fused_action(self, path_values: torch.Tensor, path_visits: torch.Tensor, 
                             valid_mask: torch.Tensor, beta: float, hbar_eff: float, quantum_strength: float) -> torch.Tensor:
        """PyTorch implementation with maximum fusion"""
        eps = 1e-8
        
        # Fused classical action computation
        # S_R = -∑ᵢ log N(sᵢ, aᵢ) combined with masking
        log_visits = torch.log(path_visits + eps)
        S_R = -torch.sum(log_visits * valid_mask.float(), dim=1)
        
        # Fused quantum action computation
        # S_I = β·σ²(V[π]) - variance computation fused
        valid_float = valid_mask.float()
        masked_values = path_values * valid_float
        
        # Fused mean and variance computation
        sum_values = torch.sum(masked_values, dim=1, keepdim=True)
        count_values = torch.sum(valid_float, dim=1, keepdim=True).clamp(min=1)
        mean_values = sum_values / count_values
        
        # Variance in single operation
        centered_values = (masked_values - mean_values) * valid_float
        variance = torch.sum(centered_values * centered_values, dim=1) / count_values.squeeze(1)
        S_I = beta * variance
        
        # Fused final action computation
        # exp(-2S_R/ℏ) * (1 + strength*cos(S_I/ℏ)) -> log
        exp_arg = -2 * S_R / hbar_eff
        cos_arg = S_I / hbar_eff
        
        # Use torch.where for numerical stability
        classical_stable = torch.where(exp_arg > -20, torch.exp(exp_arg), torch.zeros_like(exp_arg))
        quantum_term = 1.0 + quantum_strength * torch.cos(cos_arg)
        
        # Final fused computation
        action_value = classical_stable * quantum_term
        action = torch.log(action_value + eps)
        
        return action
    
    def _triton_fused_action(self, path_values: torch.Tensor, path_visits: torch.Tensor, 
                           valid_mask: torch.Tensor, beta: float, hbar_eff: float, quantum_strength: float) -> torch.Tensor:
        """Triton kernel implementation for maximum performance"""
        try:
            import triton
            import triton.language as tl
            
            @triton.jit
            def fused_action_kernel(
                path_values_ptr, path_visits_ptr, valid_mask_ptr, output_ptr,
                batch_size, path_length,
                beta, hbar_eff, quantum_strength,
                BLOCK_SIZE: tl.constexpr
            ):
                # Get program ID
                pid = tl.program_id(0)
                
                # Load data for this batch
                if pid < batch_size:
                    # Calculate offsets
                    start_idx = pid * path_length
                    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < (start_idx + path_length)
                    
                    # Load tensors
                    values = tl.load(path_values_ptr + offsets, mask=mask, other=0.0)
                    visits = tl.load(path_visits_ptr + offsets, mask=mask, other=1.0)
                    valid = tl.load(valid_mask_ptr + offsets, mask=mask, other=0.0)
                    
                    # Classical action: S_R = -∑log(visits)
                    log_visits = tl.log(visits + 1e-8)
                    S_R = -tl.sum(log_visits * valid)
                    
                    # Quantum action: S_I = β·σ²(values)
                    masked_values = values * valid
                    sum_values = tl.sum(masked_values)
                    count_values = tl.sum(valid)
                    mean_val = sum_values / tl.maximum(count_values, 1.0)
                    
                    # Variance computation
                    centered = (masked_values - mean_val) * valid
                    variance = tl.sum(centered * centered) / tl.maximum(count_values, 1.0)
                    S_I = beta * variance
                    
                    # Final action
                    classical_exp = tl.exp(-2.0 * S_R / hbar_eff)
                    quantum_cos = tl.cos(S_I / hbar_eff)
                    action_val = classical_exp * (1.0 + quantum_strength * quantum_cos)
                    final_action = tl.log(action_val + 1e-8)
                    
                    # Store result
                    tl.store(output_ptr + pid, final_action)
            
            # Prepare tensors
            batch_size, path_length = path_values.shape
            output = torch.empty(batch_size, device=self.device, dtype=torch.float32)
            
            # Launch kernel
            BLOCK_SIZE = triton.next_power_of_2(path_length)
            grid = (batch_size,)
            
            fused_action_kernel[grid](
                path_values, path_visits, valid_mask.float(), output,
                batch_size, path_length,
                beta, hbar_eff, quantum_strength,
                BLOCK_SIZE=BLOCK_SIZE
            )
            
            return output
            
        except Exception:
            # Fall back to PyTorch if Triton fails
            return self._pytorch_fused_action(path_values, path_visits, valid_mask, beta, hbar_eff, quantum_strength)
    
    def _compute_potential(
        self,
        paths: torch.Tensor,
        values: torch.Tensor,
        visits: torch.Tensor
    ) -> torch.Tensor:
        """Compute potential energy along paths"""
        # Value-based potential (attractive to high-value regions)
        value_potential = -self.config.value_potential_scale * values
        
        # Visit-based potential (repulsive from over-explored regions)
        visit_potential = self.config.visit_potential_scale * torch.log(visits + 1.0)
        
        # Uncertainty potential (attractive to uncertain regions)
        uncertainty = 1.0 / torch.sqrt(visits + 1.0)
        uncertainty_potential = -self.config.uncertainty_potential_scale * uncertainty
        
        # Total potential
        potential = value_potential + visit_potential + uncertainty_potential
        
        return potential
    
    def _compute_quantum_correction(
        self,
        paths: torch.Tensor,
        potential: torch.Tensor
    ) -> torch.Tensor:
        """Compute quantum tunneling corrections"""
        batch_size = paths.shape[0]
        
        # Find potential barriers
        barriers = potential > potential.mean(dim=1, keepdim=True)
        
        # Compute tunneling amplitude
        barrier_height = torch.max(potential, dim=1).values - torch.min(potential, dim=1).values
        
        # WKB approximation
        tunneling_action = torch.sqrt(
            2 * self.config.mass * torch.relu(barrier_height)
        ) / self.config.hbar
        
        tunneling_amplitude = torch.exp(-tunneling_action)
        
        # Quantum correction
        correction = -self.config.tunneling_strength * torch.log(
            1 + tunneling_amplitude
        )
        
        # Track tunneling events
        self.stats['tunneling_events'] += (tunneling_amplitude > 0.1).sum().item()
        
        return correction
    
    def _compute_path_probability_gpu(self, action: torch.Tensor) -> torch.Tensor:
        """Compute probability from action (GPU)"""
        # Quantum amplitude with Wick rotation
        amplitude = torch.exp(-action / (self.config.hbar * self.config.temperature))
        
        # Normalize
        probability = F.softmax(amplitude, dim=0)
        
        return probability
    
    def _variational_optimization(
        self,
        initial_paths: torch.Tensor,
        values: torch.Tensor,
        visits: torch.Tensor
    ) -> torch.Tensor:
        """GPU-native variational optimization using evolutionary strategy"""
        # GPU-native evolutionary optimization (much faster than CPU scipy)
        return self._gpu_evolutionary_optimization(initial_paths, values, visits)
    
    def _gpu_evolutionary_optimization(
        self,
        initial_paths: torch.Tensor,
        values: torch.Tensor,
        visits: torch.Tensor
    ) -> torch.Tensor:
        """GPU-native evolutionary strategy for path optimization"""
        batch_size, path_length = initial_paths.shape
        population_size = min(batch_size * 4, 1024)  # GPU-friendly population size
        max_node = values.shape[0] - 1
        
        # Initialize population with variations of initial paths
        population = initial_paths.repeat(population_size // batch_size + 1, 1)[:population_size]
        
        # Add gaussian noise to create diversity (vectorized)
        noise = torch.randn_like(population.float()) * 2.0
        population = torch.clamp(population.float() + noise, 0, max_node).long()
        
        # Evolution loop - fully vectorized on GPU
        for generation in range(self.config.num_variational_steps):
            # Evaluate entire population in single call
            fitness = self._compute_path_action_gpu(population, values, visits)
            
            # Selection: keep top 50% (vectorized)
            _, top_indices = torch.topk(fitness, population_size // 2, largest=True)
            elite = population[top_indices]
            
            # Crossover: create offspring (vectorized)
            parent1_idx = torch.randint(0, len(elite), (population_size // 2,), device=self.device)
            parent2_idx = torch.randint(0, len(elite), (population_size // 2,), device=self.device)
            
            parent1 = elite[parent1_idx]
            parent2 = elite[parent2_idx]
            
            # Vectorized uniform crossover
            crossover_mask = torch.rand(parent1.shape, device=self.device) < 0.5
            offspring = torch.where(crossover_mask, parent1, parent2)
            
            # Mutation: small random changes (vectorized)
            mutation_mask = torch.rand(offspring.shape, device=self.device) < 0.1
            mutation_values = torch.randint(-2, 3, offspring.shape, device=self.device)
            offspring = torch.where(
                mutation_mask,
                torch.clamp(offspring + mutation_values, 0, max_node),
                offspring
            )
            
            # New population: elite + offspring
            population = torch.cat([elite, offspring], dim=0)
        
        # Return best paths (select top batch_size)
        final_fitness = self._compute_path_action_gpu(population, values, visits)
        _, best_indices = torch.topk(final_fitness, batch_size, largest=True)
        
        self.stats['optimization_steps'] += 1
        return population[best_indices]
    
    def _simulated_annealing_paths(
        self,
        initial_paths: torch.Tensor,
        values: torch.Tensor,
        visits: torch.Tensor
    ) -> torch.Tensor:
        """Simulated annealing fallback for discrete path optimization"""
        current_paths = initial_paths.clone()
        current_action = self._compute_path_action_gpu(current_paths, values, visits).mean()
        
        T = 1.0  # Initial temperature
        cooling_rate = 0.95
        
        for step in range(self.config.num_variational_steps):
            # Generate neighbor by modifying one random path element
            new_paths = current_paths.clone()
            batch_idx = torch.randint(0, current_paths.shape[0], (1,)).item()
            path_idx = torch.randint(0, current_paths.shape[1], (1,)).item()
            
            if current_paths[batch_idx, path_idx] >= 0:  # Only modify valid nodes
                # Perturb within bounds
                max_node = values.shape[0] - 1
                new_paths[batch_idx, path_idx] = torch.clamp(
                    current_paths[batch_idx, path_idx] + torch.randint(-2, 3, (1,)).item(),
                    0, max_node
                )
                
                # Evaluate new configuration
                new_action = self._compute_path_action_gpu(new_paths, values, visits).mean()
                
                # Accept/reject based on Metropolis criterion
                delta = new_action - current_action
                if delta > 0 or torch.rand(1).item() < torch.exp(delta / T).item():
                    current_paths = new_paths
                    current_action = new_action
                
                # Cool down
                T *= cooling_rate
        
        return current_paths
    
    def _sample_paths_importance(
        self,
        root_idx: int,
        num_paths: int,
        tree
    ) -> torch.Tensor:
        """Vectorized path sampling using batch UCB selection"""
        max_len = min(self.config.max_path_length, 20)  # Reasonable limit
        
        # Initialize all paths at root
        paths = torch.full((num_paths, max_len), root_idx, dtype=torch.long, device=self.device)
        active_mask = torch.ones(num_paths, dtype=torch.bool, device=self.device)
        
        # Vectorized path construction
        for depth in range(1, max_len):
            if not active_mask.any():
                break
            
            # Get current nodes for all active paths
            current_nodes = paths[active_mask, depth - 1]
            
            # Batch UCB selection for all active paths
            next_nodes = self._vectorized_ucb_selection(current_nodes, tree, active_mask.sum().item())
            
            # Update paths
            paths[active_mask, depth] = next_nodes
            
            # Update active mask (stop at leaves or invalid nodes)
            valid_next = (next_nodes >= 0) & (next_nodes < tree.num_nodes)
            
            # Create new active mask
            new_active = torch.zeros(num_paths, dtype=torch.bool, device=self.device)
            new_active[active_mask] = valid_next
            
            # Random stopping with vectorized probability
            stop_prob = torch.rand(new_active.sum(), device=self.device) < 0.05
            if stop_prob.any():
                stop_indices = torch.where(new_active)[0][stop_prob]
                new_active[stop_indices] = False
            
            active_mask = new_active
        
        # Set invalid nodes to -1
        invalid_mask = paths == root_idx
        invalid_mask[:, 0] = False  # Keep root valid
        paths[invalid_mask] = -1
        
        return paths
    
    def _vectorized_ucb_selection(self, parent_nodes: torch.Tensor, tree, batch_size: int) -> torch.Tensor:
        """Vectorized UCB selection for multiple parent nodes simultaneously"""
        if len(parent_nodes) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        # Ensure parent nodes are valid
        parent_nodes = torch.clamp(parent_nodes, 0, tree.num_nodes - 1)
        
        # Get all children for all parents at once
        all_children = tree.children[parent_nodes]  # Shape: (batch_size, max_children)
        valid_children_mask = all_children >= 0
        
        # Safety clamp for children
        safe_children = torch.clamp(all_children, 0, tree.num_nodes - 1)
        
        # Vectorized statistics gathering
        parent_visits = tree.visit_counts[parent_nodes].float().unsqueeze(1)  # (batch_size, 1)
        child_visits = tree.visit_counts[safe_children].float()  # (batch_size, max_children)
        child_values = tree.value_sums[safe_children].float()
        
        # Vectorized Q-values
        q_values = torch.where(
            child_visits > 0,
            child_values / child_visits,
            torch.zeros_like(child_values)
        )
        
        # Vectorized UCB computation
        c_puct = 1.0
        exploration = c_puct * torch.sqrt(torch.log(parent_visits + 1) / (child_visits + 1))
        
        # Quantum corrections (vectorized)
        T_eff = self.config.hbar_eff / self.config.beta
        quantum_uncertainty = T_eff * torch.sqrt(1.0 / (child_visits + 1))
        
        # Total UCB scores
        ucb_scores = q_values + exploration + quantum_uncertainty
        
        # Mask invalid children
        ucb_scores = torch.where(valid_children_mask, ucb_scores, torch.tensor(-float('inf'), device=self.device))
        
        # Vectorized selection - argmax for each row
        selected_child_indices = torch.argmax(ucb_scores, dim=1)
        
        # Gather selected children
        batch_indices = torch.arange(len(parent_nodes), device=self.device)
        selected_children = all_children[batch_indices, selected_child_indices]
        
        # Apply quantum tunneling (vectorized)
        tunneling_mask = torch.rand(len(parent_nodes), device=self.device) < self.config.tunneling_strength
        if tunneling_mask.any():
            # Vectorized tunneling selection
            tunneling_probs = torch.softmax(-child_visits / T_eff, dim=1)
            tunneling_indices = torch.multinomial(tunneling_probs[tunneling_mask], 1).squeeze(1)
            selected_children[tunneling_mask] = all_children[tunneling_mask.nonzero().squeeze(1), tunneling_indices]
        
        return selected_children
    
    def _sample_single_path(self, start_idx: int, tree) -> List[int]:
        """Sample a single path using UCB-based selection with quantum corrections
        
        Physical Justification: 
        - UCB naturally arises from the uncertainty principle in quantum mechanics
        - The exploration term captures quantum fluctuations (Heisenberg uncertainty)
        - The exploitation term represents the classical trajectory (stationary phase)
        """
        path = [start_idx]
        current = start_idx
        
        # Safety check
        if not hasattr(tree, 'children') or current < 0 or current >= tree.num_nodes:
            return path
        
        for _ in range(min(self.config.max_path_length, 20)):  # Safety limit
            # Get actual children from tree
            children = tree.children[current]
            valid_children = children[children >= 0]
            
            if len(valid_children) == 0:
                break
                
            # Safety check: ensure children are within bounds
            valid_children = valid_children[valid_children < tree.num_nodes]
            if len(valid_children) == 0:
                break
            
            # Quantum-corrected UCB selection
            next_node = self._ucb_select_with_quantum_corrections(current, valid_children, tree)
            
            # Ensure next node is valid
            if next_node < 0 or next_node >= tree.num_nodes:
                break
                
            path.append(next_node)
            current = next_node
            
            # Stop with small probability (quantum tunneling out of search)
            if torch.rand(1).item() < 0.05:  # Reduced probability
                break
        
        return path
    
    def _ucb_select_with_quantum_corrections(self, parent_idx: int, children: torch.Tensor, tree) -> int:
        """UCB selection with quantum corrections based on path integral formulation
        
        From docs: The exploration-exploitation tradeoff is controlled by T_eff = ℏ_eff/β
        UCB naturally emerges from quantum uncertainty principle.
        """
        if len(children) == 0:
            return -1
        
        # Get parent visit count
        parent_visits = tree.visit_counts[parent_idx].float()
        
        # Get child statistics
        child_visits = tree.visit_counts[children].float()
        child_values = tree.value_sums[children].float()
        
        # Q-values (exploitation)
        q_values = torch.where(
            child_visits > 0,
            child_values / child_visits,
            torch.zeros_like(child_values)
        )
        
        # Classical UCB exploration term
        c_puct = 1.0  # Could be from config
        exploration = c_puct * torch.sqrt(torch.log(parent_visits + 1) / (child_visits + 1))
        
        # Quantum correction: uncertainty principle ΔE·Δt ≥ ℏ/2
        # Higher uncertainty (lower visits) → larger quantum fluctuations
        T_eff = self.config.hbar_eff / self.config.beta
        quantum_uncertainty = T_eff * torch.sqrt(1.0 / (child_visits + 1))
        
        # Total UCB with quantum corrections
        ucb_scores = q_values + exploration + quantum_uncertainty
        
        # Add small quantum tunneling probability for exploration
        if torch.rand(1).item() < self.config.tunneling_strength:
            # Quantum tunneling: occasionally select suboptimal moves
            tunneling_probs = torch.softmax(-child_visits / T_eff, dim=0)
            selected_idx = torch.multinomial(tunneling_probs, 1).item()
        else:
            # Standard UCB selection (most probable classical trajectory)
            selected_idx = torch.argmax(ucb_scores).item()
        
        return children[selected_idx].item()
    
    # CPU-specific methods
    def compute_path_action(self, path: Path) -> complex:
        """Compute action for a path (CPU)"""
        # Check cache
        path_key = self._get_path_key(path)
        if path_key in self.action_cache:
            return self.action_cache[path_key]
            
        # Classical action: -log(visits)
        visits = path.get_visits()
        classical_action = -np.log(visits + self.config.regularization)
        
        # Quantum phase based on value fluctuations
        if self.config.use_complex_action:
            phase = self._compute_phase(path)
            action = classical_action + 1j * phase
        else:
            action = classical_action
            
        # Cache result
        self.action_cache[path_key] = action
        path._action_value = action
        
        self.stats['paths_evaluated'] += 1
        
        return action
    
    def _compute_phase(self, path: Path) -> float:
        """Compute quantum phase for path (CPU)"""
        if len(path.nodes) < 2:
            return 0.0
            
        # Compute value variance along path
        values = []
        for node in path.nodes:
            if node.visit_count > 0:
                values.append(node.value())
                
        if len(values) < 2:
            return 0.0
            
        # Phase proportional to value variance
        variance = np.var(values)
        phase = self.config.phase_coupling * np.sqrt(variance)
        
        # Add contribution from uncertainty
        uncertainties = []
        for node in path.nodes:
            if node.visit_count > 0:
                uncertainty = 1.0 / np.sqrt(node.visit_count)
                uncertainties.append(uncertainty)
                
        if uncertainties:
            avg_uncertainty = np.mean(uncertainties)
            phase += self.config.phase_coupling * avg_uncertainty
            
        return phase
    
    def _get_quantum_corrections(self, node: 'Node') -> Dict[int, float]:
        """Get quantum corrections to action values (CPU)"""
        corrections = {}
        
        for action, child in node.children.items():
            # Create path to child
            path = Path([node, child], [action])
            
            # Compute quantum amplitude
            amplitude = np.exp(-self.compute_path_action(path) / self.config.temperature)
            
            # Correction based on amplitude magnitude
            correction = np.log(np.abs(amplitude) + 1e-10)
            corrections[action] = correction * 0.1  # Scale factor
            
        return corrections
    
    def _get_path_key(self, path: Path) -> str:
        """Get unique key for path (for caching)"""
        return '-'.join(str(a) for a in path.actions)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get path integral statistics"""
        return dict(self.stats)