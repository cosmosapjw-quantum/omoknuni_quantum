"""
Renormalization Group Flow Optimizer for QFT-MCTS
=================================================

This module implements renormalization group (RG) flow analysis to optimize
MCTS parameters based on scale-invariant properties and fixed points.

Key Features:
- Beta function computation for parameter flow
- Wilson-Fisher fixed point identification
- Scale-dependent parameter optimization
- Critical exponent calculation
- GPU-accelerated flow evolution

Based on: docs/qft-mcts-math-foundations.md Section 5.2
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class RGConfig:
    """Configuration for RG flow optimizer"""
    # Flow parameters
    initial_scale: float = 1.0          # Initial energy scale μ
    final_scale: float = 0.01           # Final (IR) scale
    num_flow_steps: int = 100           # Number of RG steps
    flow_dt: float = 0.01               # Flow step size
    
    # Beta function parameters
    epsilon: float = 0.1                # ε = 4-d expansion parameter
    coupling_dimension: int = 3         # Number of couplings to track
    
    # Fixed point search
    fixed_point_tolerance: float = 1e-6 # Convergence criterion
    max_iterations: int = 1000          # Maximum iterations
    newton_damping: float = 0.5         # Newton method damping
    
    # Parameter bounds
    c_puct_min: float = 0.1            # Minimum c_puct
    c_puct_max: float = 10.0           # Maximum c_puct
    exploration_min: float = 0.01       # Minimum exploration
    exploration_max: float = 2.0        # Maximum exploration
    
    # Numerical stability
    regularization: float = 1e-8        # Numerical regularization
    gradient_clip: float = 10.0         # Gradient clipping threshold


class BetaFunction:
    """
    Computes beta functions for RG flow equations
    
    The beta function β(g) = μ ∂g/∂μ describes how couplings
    change with energy scale in the RG flow.
    """
    
    def __init__(self, config: RGConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize coupling constants
        self.initialize_couplings()
        
    def initialize_couplings(self):
        """Initialize perturbative coupling constants"""
        # Leading order coefficients for beta function
        # β(g) = -εg + ag³ + bg⁵ + ...
        
        # For c_puct coupling
        self.a_puct = 1.0   # One-loop coefficient
        self.b_puct = -0.5  # Two-loop coefficient
        
        # For exploration coupling
        self.a_explore = 0.8
        self.b_explore = -0.3
        
        # For interference coupling
        self.a_interference = 1.2
        self.b_interference = -0.6
        
    def compute_beta(
        self,
        couplings: torch.Tensor,
        scale: float
    ) -> torch.Tensor:
        """
        Compute beta function for coupling constants
        
        Args:
            couplings: Current coupling values [c_puct, exploration, interference]
            scale: Current RG scale μ
            
        Returns:
            Beta function values dg/d(log μ)
        """
        c_puct, exploration, interference = couplings[0], couplings[1], couplings[2]
        
        # Beta function for c_puct
        # β_puct = -εg + a*g³ + b*g⁵
        beta_puct = (
            -self.config.epsilon * c_puct +
            self.a_puct * c_puct**3 +
            self.b_puct * c_puct**5
        )
        
        # Beta function for exploration
        beta_explore = (
            -self.config.epsilon * exploration +
            self.a_explore * exploration**3 +
            self.b_explore * exploration**5
        )
        
        # Beta function for interference (with coupling to c_puct)
        beta_interference = (
            -self.config.epsilon * interference +
            self.a_interference * interference**3 +
            self.b_interference * interference**5 +
            0.1 * c_puct * interference**2  # Mixed coupling term
        )
        
        beta = torch.stack([beta_puct, beta_explore, beta_interference])
        
        return beta
    
    def compute_anomalous_dimensions(
        self,
        couplings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute anomalous dimensions for operators
        
        These determine how operator dimensions change under RG flow.
        """
        c_puct, exploration, interference = couplings[0], couplings[1], couplings[2]
        
        # Anomalous dimension for value operator
        gamma_value = 0.1 * c_puct**2 + 0.05 * exploration**2
        
        # Anomalous dimension for visit count operator  
        gamma_visits = 0.15 * c_puct**2 + 0.1 * interference**2
        
        # Anomalous dimension for path operator
        gamma_path = 0.2 * exploration**2 + 0.1 * interference**2
        
        return torch.stack([gamma_value, gamma_visits, gamma_path])


class FixedPointFinder:
    """
    Finds fixed points of RG flow
    
    Fixed points β(g*) = 0 correspond to scale-invariant theories
    and optimal parameter choices.
    """
    
    def __init__(self, config: RGConfig, device: torch.device):
        self.config = config
        self.device = device
        self.beta_function = BetaFunction(config, device)
        
    def find_wilson_fisher_fixed_point(
        self,
        initial_guess: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Find Wilson-Fisher fixed point using Newton's method
        
        The WF fixed point is the non-trivial IR stable fixed point
        that determines optimal MCTS parameters.
        
        Returns:
            Tuple of (fixed_point_couplings, convergence_info)
        """
        if initial_guess is None:
            # Theoretical prediction: g* ≈ √ε at leading order
            initial_guess = torch.tensor(
                [np.sqrt(self.config.epsilon)] * 3,
                device=self.device
            )
        
        couplings = initial_guess.clone()
        
        for iteration in range(self.config.max_iterations):
            # Compute beta function
            beta = self.beta_function.compute_beta(couplings, scale=1.0)
            
            # Check convergence
            if torch.norm(beta) < self.config.fixed_point_tolerance:
                logger.info(f"Fixed point found at iteration {iteration}")
                break
            
            # Compute Jacobian numerically
            jacobian = self._compute_beta_jacobian(couplings)
            
            # Newton step: g_new = g_old - J^(-1) * beta
            try:
                delta = torch.linalg.solve(jacobian, beta)
            except:
                # Fallback to gradient descent if singular
                delta = self.config.newton_damping * beta
            
            # Update with damping
            couplings = couplings - self.config.newton_damping * delta
            
            # Enforce bounds
            couplings = torch.clamp(couplings, min=0.01, max=5.0)
        
        # Compute critical exponents at fixed point
        critical_exponents = self._compute_critical_exponents(couplings)
        
        convergence_info = {
            'iterations': iteration,
            'final_beta_norm': torch.norm(beta).item(),
            'converged': torch.norm(beta) < self.config.fixed_point_tolerance,
            'critical_exponents': critical_exponents
        }
        
        return couplings, convergence_info
    
    def _compute_beta_jacobian(
        self,
        couplings: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """Compute Jacobian of beta function numerically"""
        dim = len(couplings)
        jacobian = torch.zeros((dim, dim), device=self.device)
        
        beta_0 = self.beta_function.compute_beta(couplings, scale=1.0)
        
        for i in range(dim):
            couplings_plus = couplings.clone()
            couplings_plus[i] += eps
            
            beta_plus = self.beta_function.compute_beta(couplings_plus, scale=1.0)
            jacobian[:, i] = (beta_plus - beta_0) / eps
        
        return jacobian
    
    def _compute_critical_exponents(
        self,
        fixed_point: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute critical exponents from RG eigenvalues
        
        Critical exponents determine universality class and scaling behavior.
        """
        # Compute Jacobian at fixed point
        jacobian = self._compute_beta_jacobian(fixed_point)
        
        # Eigenvalues give scaling dimensions
        eigenvalues = torch.linalg.eigvals(jacobian)
        
        # Sort by magnitude
        eigenvalues_real = eigenvalues.real
        sorted_indices = torch.argsort(torch.abs(eigenvalues_real), descending=True)
        eigenvalues_sorted = eigenvalues_real[sorted_indices]
        
        # Critical exponents from eigenvalues
        # ν (correlation length) from largest eigenvalue
        nu = 1.0 / torch.abs(eigenvalues_sorted[0]).item() if eigenvalues_sorted[0] != 0 else float('inf')
        
        # η (anomalous dimension) from anomalous dimensions at fixed point
        anomalous_dims = self.beta_function.compute_anomalous_dimensions(fixed_point)
        eta = anomalous_dims[0].item()  # Value operator anomalous dimension
        
        return {
            'nu': nu,                    # Correlation length exponent
            'eta': eta,                  # Anomalous dimension
            'omega': eigenvalues_sorted[1].item() if len(eigenvalues_sorted) > 1 else 0,  # Correction to scaling
            'eigenvalues': eigenvalues_sorted.cpu().numpy()
        }


class RGFlowEvolution:
    """
    Evolves parameters along RG flow trajectories
    
    This implements the scale-dependent optimization of MCTS parameters
    based on RG flow equations.
    """
    
    def __init__(self, config: RGConfig, device: torch.device):
        self.config = config
        self.device = device
        self.beta_function = BetaFunction(config, device)
        self.fixed_point_finder = FixedPointFinder(config, device)
        
    def evolve_parameters(
        self,
        initial_params: Dict[str, float],
        target_scale: float,
        return_trajectory: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Evolve parameters from UV to IR scale
        
        Args:
            initial_params: Initial parameter values
            target_scale: Target energy scale
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Tuple of (final_params, evolution_info)
        """
        # Convert to coupling vector
        couplings = torch.tensor([
            initial_params.get('c_puct', 1.414),
            initial_params.get('exploration_fraction', 0.25),
            initial_params.get('interference_strength', 0.15)
        ], device=self.device)
        
        # Initialize scale
        scale = self.config.initial_scale
        
        # Storage for trajectory
        trajectory = [] if return_trajectory else None
        
        # RG flow evolution
        num_steps = int(np.log(self.config.initial_scale / target_scale) / self.config.flow_dt)
        step = 0  # Initialize step counter
        
        for step in range(num_steps):
            if return_trajectory:
                trajectory.append({
                    'scale': scale,
                    'couplings': couplings.clone(),
                    'beta': self.beta_function.compute_beta(couplings, scale)
                })
            
            # Compute RG flow
            beta = self.beta_function.compute_beta(couplings, scale)
            
            # Update couplings: dg/d(log μ) = β(g)
            couplings = couplings + self.config.flow_dt * beta
            
            # Update scale
            scale = scale * np.exp(-self.config.flow_dt)
            
            # Enforce bounds
            couplings[0] = torch.clamp(couplings[0], self.config.c_puct_min, self.config.c_puct_max)
            couplings[1] = torch.clamp(couplings[1], self.config.exploration_min, self.config.exploration_max)
            couplings[2] = torch.clamp(couplings[2], 0.01, 1.0)  # Ensure non-zero interference
            
            # Check if we've reached target scale
            if scale <= target_scale * 1.001:  # Small tolerance
                break
        
        # Convert back to parameters
        final_params = {
            'c_puct': couplings[0].item(),
            'exploration_fraction': couplings[1].item(),
            'interference_strength': couplings[2].item()
        }
        
        # Add scale-dependent corrections
        final_params.update(self._compute_scale_corrections(couplings, scale))
        
        evolution_info = {
            'initial_scale': self.config.initial_scale,
            'final_scale': scale,
            'num_steps': step + 1,
            'trajectory': trajectory
        }
        
        return final_params, evolution_info
    
    def _compute_scale_corrections(
        self,
        couplings: torch.Tensor,
        scale: float
    ) -> Dict[str, float]:
        """Compute scale-dependent corrections to parameters"""
        # Anomalous dimensions affect operator scaling
        anomalous_dims = self.beta_function.compute_anomalous_dimensions(couplings)
        
        # Scale-dependent effective parameters
        scale_factor = scale / self.config.initial_scale
        
        corrections = {
            'value_scale_correction': scale_factor ** anomalous_dims[0].item(),
            'visit_scale_correction': scale_factor ** anomalous_dims[1].item(),
            'path_scale_correction': scale_factor ** anomalous_dims[2].item()
        }
        
        return corrections


class RGFlowOptimizer:
    """
    Main RG flow optimizer for MCTS parameter optimization
    
    This provides the interface for using RG flow analysis to optimize
    MCTS parameters based on tree statistics and performance.
    """
    
    def __init__(self, config: RGConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize components
        self.flow_evolution = RGFlowEvolution(config, device)
        self.fixed_point_finder = FixedPointFinder(config, device)
        
        # Cache for fixed points
        self.wilson_fisher_point = None
        self.gaussian_point = torch.zeros(3, device=device)  # Trivial fixed point
        
        # Statistics
        self.stats = {
            'optimizations_performed': 0,
            'fixed_points_found': 0,
            'average_convergence_steps': 0,
            'parameter_improvements': []
        }
        
        logger.info("RGFlowOptimizer initialized")
    
    def find_optimal_parameters(
        self,
        tree_statistics: Dict[str, float],
        current_params: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Find optimal parameters using RG flow analysis
        
        Args:
            tree_statistics: Current tree statistics (depth, branching, visits)
            current_params: Current parameter values
            
        Returns:
            Tuple of (optimal_params, optimization_info)
        """
        # Extract relevant scale from tree statistics
        tree_scale = self._extract_tree_scale(tree_statistics)
        
        # Find Wilson-Fisher fixed point if not cached
        if self.wilson_fisher_point is None:
            self.wilson_fisher_point, fp_info = self.fixed_point_finder.find_wilson_fisher_fixed_point()
            self.stats['fixed_points_found'] += 1
            logger.info(f"Wilson-Fisher fixed point found: {self.wilson_fisher_point}")
        
        # Determine which fixed point to flow towards
        if tree_scale < 0.1:  # Deep in IR
            target_point = self.wilson_fisher_point
            flow_type = "wilson_fisher"
        else:  # UV or crossover regime
            # Interpolate between fixed points
            alpha = np.exp(-tree_scale / 0.5)
            target_point = alpha * self.wilson_fisher_point + (1 - alpha) * self.gaussian_point
            flow_type = "crossover"
        
        # Add scale-dependent adjustments
        if flow_type == "crossover":
            # In UV, use different initial values to ensure flow
            current_params['c_puct'] *= (1 + 0.2 * tree_scale)
            current_params['exploration_fraction'] *= (1 + 0.1 * tree_scale)
        
        # Set initial parameters
        if current_params is None:
            current_params = {
                'c_puct': 1.414,  # √2 default
                'exploration_fraction': 0.25,
                'interference_strength': 0.15
            }
        
        # Evolve parameters to target scale
        optimal_params, evolution_info = self.flow_evolution.evolve_parameters(
            current_params,
            target_scale=tree_scale,
            return_trajectory=True
        )
        
        # Apply fixed point corrections
        if flow_type == "wilson_fisher":
            # At WF fixed point: c_puct = √2 exactly
            optimal_params['c_puct'] = np.sqrt(2.0)
        
        # Compute improvement metrics
        improvement_info = self._compute_parameter_improvements(
            current_params, optimal_params, tree_statistics
        )
        
        # Update statistics
        self.stats['optimizations_performed'] += 1
        self.stats['parameter_improvements'].append(improvement_info['total_improvement'])
        
        optimization_info = {
            'tree_scale': tree_scale,
            'flow_type': flow_type,
            'target_fixed_point': target_point.cpu().numpy(),
            'evolution_info': evolution_info,
            'improvement_info': improvement_info
        }
        
        return optimal_params, optimization_info
    
    def _extract_tree_scale(self, tree_statistics: Dict[str, float]) -> float:
        """Extract effective RG scale from tree statistics"""
        # Scale ~ 1/depth for tree exploration
        depth = tree_statistics.get('average_depth', 10)
        
        # Adjust for branching factor
        branching = tree_statistics.get('effective_branching_factor', 10)
        
        # Effective scale
        scale = 1.0 / (depth * np.log(branching + 1))
        
        return np.clip(scale, self.config.final_scale, self.config.initial_scale)
    
    def _compute_parameter_improvements(
        self,
        old_params: Dict[str, float],
        new_params: Dict[str, float],
        tree_statistics: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute expected improvements from parameter changes"""
        # Relative changes
        c_puct_change = (new_params['c_puct'] - old_params['c_puct']) / old_params['c_puct']
        exploration_change = (
            new_params['exploration_fraction'] - old_params['exploration_fraction']
        ) / old_params['exploration_fraction']
        
        # Expected improvements based on RG theory
        # Better exploration-exploitation balance
        balance_improvement = -abs(new_params['c_puct'] - np.sqrt(2)) / np.sqrt(2)
        
        # Reduced redundancy from interference
        redundancy_reduction = new_params['interference_strength'] * 0.5
        
        # Total improvement estimate
        total_improvement = balance_improvement + redundancy_reduction
        
        return {
            'c_puct_change': c_puct_change,
            'exploration_change': exploration_change,
            'balance_improvement': balance_improvement,
            'redundancy_reduction': redundancy_reduction,
            'total_improvement': total_improvement
        }
    
    def compute_running_couplings(
        self,
        scales: torch.Tensor,
        initial_couplings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute running of couplings at multiple scales
        
        This is useful for understanding parameter behavior across scales.
        
        Args:
            scales: Tensor of scales to evaluate
            initial_couplings: Initial coupling values
            
        Returns:
            Tensor of coupling values at each scale
        """
        if initial_couplings is None:
            initial_couplings = torch.tensor([1.414, 0.25, 0.15], device=self.device)
        
        num_scales = len(scales)
        running_couplings = torch.zeros((num_scales, 3), device=self.device)
        
        for i, scale in enumerate(scales):
            params, _ = self.flow_evolution.evolve_parameters(
                {
                    'c_puct': initial_couplings[0].item(),
                    'exploration_fraction': initial_couplings[1].item(),
                    'interference_strength': initial_couplings[2].item()
                },
                target_scale=scale.item()
            )
            
            running_couplings[i] = torch.tensor([
                params['c_puct'],
                params['exploration_fraction'],
                params['interference_strength']
            ], device=self.device)
        
        return running_couplings
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        stats = dict(self.stats)
        
        # Add average improvement
        if stats['parameter_improvements']:
            stats['average_improvement'] = np.mean(stats['parameter_improvements'])
        
        return stats


# Factory function
def create_rg_optimizer(
    device: Union[str, torch.device] = 'cuda',
    epsilon: float = 0.1,
    **kwargs
) -> RGFlowOptimizer:
    """
    Factory function to create RG flow optimizer
    
    Args:
        device: Device for computation
        epsilon: RG expansion parameter
        **kwargs: Override default config parameters
        
    Returns:
        Initialized RGFlowOptimizer
    """
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create config with overrides
    config_dict = {
        'epsilon': epsilon,
        'initial_scale': 1.0,
        'final_scale': 0.01,
    }
    config_dict.update(kwargs)
    
    config = RGConfig(**config_dict)
    
    return RGFlowOptimizer(config, device)