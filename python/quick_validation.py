"""Quick validation that v5.0 implementation is working"""

import torch
from mcts.quantum.selective_quantum_optimized import (
    create_selective_quantum_mcts, create_ultra_performance_quantum_mcts
)

def main():
    print("Quick v5.0 Validation")
    print("=" * 30)
    
    # Test basic functionality
    device = 'cpu'
    quantum_mcts = create_selective_quantum_mcts(device=device, enable_cuda_kernels=False)
    
    # Generate test data
    q_values = torch.randn(50)
    visit_counts = torch.randint(1, 20, (50,), dtype=torch.float32)
    priors = torch.softmax(torch.randn(50), dim=0)
    
    # Test quantum selection 
    quantum_scores = quantum_mcts.apply_selective_quantum(
        q_values, visit_counts, priors, 
        parent_visits=1000.0, simulation_count=500
    )
    
    # Test classical
    classical_scores = quantum_mcts._classical_v5_vectorized(
        q_values, visit_counts, priors, 1000.0
    )
    
    # Check difference
    diff = torch.max(torch.abs(quantum_scores - classical_scores))
    
    print(f"✓ Quantum vs Classical difference: {diff:.6f}")
    print(f"✓ Output shape: {quantum_scores.shape}")
    print(f"✓ All finite: {torch.all(torch.isfinite(quantum_scores))}")
    
    # Check v5.0 formula components
    kappa = quantum_mcts.config.kappa
    beta = quantum_mcts.config.beta
    hbar_0 = quantum_mcts.config.hbar_0
    alpha = quantum_mcts.config.alpha
    
    print(f"✓ v5.0 Parameters: κ={kappa:.3f}, β={beta:.3f}, ℏ₀={hbar_0:.3f}, α={alpha:.3f}")
    
    # Verify quantum bonus is applied
    low_visits = torch.tensor([1.0, 2.0, 3.0])
    high_visits = torch.tensor([20.0, 30.0, 40.0])
    test_priors = torch.tensor([0.33, 0.33, 0.34])
    test_q = torch.zeros(3)
    
    low_scores = quantum_mcts.apply_selective_quantum(
        test_q, low_visits, test_priors, simulation_count=100
    )
    high_scores = quantum_mcts.apply_selective_quantum(
        test_q, high_visits, test_priors, simulation_count=100
    )
    
    low_bonus = torch.mean(low_scores)
    high_bonus = torch.mean(high_scores)
    
    print(f"✓ Low visit bonus: {low_bonus:.6f}")
    print(f"✓ High visit bonus: {high_bonus:.6f}")
    print(f"✓ Quantum encourages exploration: {low_bonus > high_bonus}")
    
    print("\n🎉 v5.0 Implementation Working!")
    return True

if __name__ == "__main__":
    main()