// Quantum v5.0 CUDA kernels for MCTS - implements docs/v5.0 formula
// Score(k) = κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff(N_tot))/(3 N_k)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <math.h>

// ============================================================================
// v5.0 Quantum-Augmented Score Kernel (Selective Application)
// ============================================================================

__global__ void selective_quantum_v5_kernel(
    const float* __restrict__ q_values,      // Q_k - mean action-values [batch_size]
    const float* __restrict__ visit_counts,  // N_k - visit counts [batch_size]
    const float* __restrict__ priors,        // p_k - NN priors [batch_size]
    float* __restrict__ output,              // output scores [batch_size]
    const float kappa,                       // κ - exploration strength (stiffness)
    const float beta,                        // β - value weight (inverse temperature)
    const float hbar_0,                      // ℏ_0 - base Planck scale
    const float alpha,                       // α - annealing exponent
    const float parent_visits,               // N_tot - parent visit count
    const int simulation_count,              // Current simulation count
    const int batch_size,                    // Number of elements
    const int exploration_threshold,         // Apply quantum only below this visit count
    const int quantum_phase_threshold,       // Apply quantum only below this simulation count
    const bool enable_quantum                // Quantum features enable flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float N_k = visit_counts[idx];
    float p_k = priors[idx];
    float Q_k = q_values[idx];
    float N_tot = parent_visits;
    
    // Ensure N_k is positive for stability
    float safe_N_k = fmaxf(N_k, 1.0f);
    
    // v5.0 Quantum-Augmented Score Formula:
    // Score(k) = κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff(N_tot))/(3 N_k)
    
    // Component 1: Exploration term - κ p_k (N_k/N_tot)
    float exploration_term = kappa * p_k * (safe_N_k / N_tot);
    
    // Component 2: Exploitation term - β Q_k
    float exploitation_term = beta * Q_k;
    
    // Component 3: Selective quantum bonus - (4 ℏ_eff(N_tot))/(3 N_k)
    float quantum_bonus = 0.0f;
    
    // Apply quantum corrections selectively (only where beneficial)
    if (enable_quantum && 
        N_k < exploration_threshold && 
        simulation_count < quantum_phase_threshold) {
        
        // Calculate ℏ_eff(N_tot) = ℏ_0 (1 + N_tot)^(-α/2)
        float hbar_eff = hbar_0 * powf(1.0f + N_tot, -alpha * 0.5f);
        
        // v5.0 quantum bonus formula
        quantum_bonus = (4.0f * hbar_eff) / (3.0f * safe_N_k);
    }
    
    // Final v5.0 score
    output[idx] = exploration_term + exploitation_term + quantum_bonus;
}

// ============================================================================
// Batch Processing Kernel for Large-Scale Operations
// ============================================================================

__global__ void batch_quantum_v5_kernel(
    const float* __restrict__ q_values_batch,     // [batch_size, num_actions]
    const float* __restrict__ visit_counts_batch, // [batch_size, num_actions]
    const float* __restrict__ priors_batch,       // [batch_size, num_actions]
    const int* __restrict__ simulation_counts,    // [batch_size]
    float* __restrict__ output_batch,             // [batch_size, num_actions]
    const float kappa,
    const float beta,
    const float hbar_0,
    const float alpha,
    const float* __restrict__ parent_visits,      // [batch_size] or single value
    const int batch_size,
    const int num_actions,
    const int exploration_threshold,
    const int quantum_phase_threshold,
    const bool enable_quantum
) {
    int batch_idx = blockIdx.x;
    int action_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || action_idx >= num_actions) return;
    
    int idx = batch_idx * num_actions + action_idx;
    
    float N_k = visit_counts_batch[idx];
    float p_k = priors_batch[idx];
    float Q_k = q_values_batch[idx];
    float N_tot = parent_visits[batch_idx];  // Per-batch parent visits
    int sim_count = simulation_counts[batch_idx];
    
    float safe_N_k = fmaxf(N_k, 1.0f);
    
    // v5.0 formula components
    float exploration_term = kappa * p_k * (safe_N_k / N_tot);
    float exploitation_term = beta * Q_k;
    
    // Selective quantum bonus
    float quantum_bonus = 0.0f;
    if (enable_quantum && 
        N_k < exploration_threshold && 
        sim_count < quantum_phase_threshold) {
        
        float hbar_eff = hbar_0 * powf(1.0f + N_tot, -alpha * 0.5f);
        quantum_bonus = (4.0f * hbar_eff) / (3.0f * safe_N_k);
    }
    
    output_batch[idx] = exploration_term + exploitation_term + quantum_bonus;
}

// ============================================================================
// Vectorized Critical Point Detection Kernel
// ============================================================================

__global__ void quantum_regime_detection_kernel(
    const int* __restrict__ simulation_counts,  // [batch_size]
    const float* __restrict__ visit_counts,     // [batch_size]
    float* __restrict__ quantum_factors,        // [batch_size] output
    const int batch_size,
    const int critical_point_1,                 // Quantum -> Critical transition
    const int critical_point_2,                 // Critical -> Classical transition
    const int exploration_threshold             // Visit count threshold for quantum
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int sim_count = simulation_counts[idx];
    float visits = visit_counts[idx];
    
    // Determine quantum factor based on regime and visit count
    float quantum_factor = 0.0f;
    
    if (sim_count < critical_point_1 && visits < exploration_threshold) {
        quantum_factor = 1.0f;      // Full quantum regime
    } else if (sim_count < critical_point_2 && visits < exploration_threshold) {
        quantum_factor = 0.5f;      // Critical transition regime
    } else {
        quantum_factor = 0.1f;      // Minimal quantum (avoid pure classical)
    }
    
    quantum_factors[idx] = quantum_factor;
}

// ============================================================================
// C++ Interface Functions
// ============================================================================

// Single-batch selective quantum v5.0 processing
torch::Tensor selective_quantum_v5_cuda(
    torch::Tensor q_values,      // [batch_size]
    torch::Tensor visit_counts,  // [batch_size]
    torch::Tensor priors,        // [batch_size]
    float kappa,
    float beta,
    float hbar_0,
    float alpha,
    float parent_visits,
    int simulation_count,
    int exploration_threshold = 10,
    int quantum_phase_threshold = 5000,
    bool enable_quantum = true
) {
    const int batch_size = q_values.size(0);
    auto output = torch::zeros_like(q_values);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    selective_quantum_v5_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<float>(),
        priors.data_ptr<float>(),
        output.data_ptr<float>(),
        kappa,
        beta,
        hbar_0,
        alpha,
        parent_visits,
        simulation_count,
        batch_size,
        exploration_threshold,
        quantum_phase_threshold,
        enable_quantum
    );
    
    return output;
}

// Large-batch quantum v5.0 processing
torch::Tensor batch_quantum_v5_cuda(
    torch::Tensor q_values_batch,     // [batch_size, num_actions]
    torch::Tensor visit_counts_batch, // [batch_size, num_actions]
    torch::Tensor priors_batch,       // [batch_size, num_actions]
    torch::Tensor simulation_counts,  // [batch_size]
    torch::Tensor parent_visits,      // [batch_size]
    float kappa,
    float beta,
    float hbar_0,
    float alpha,
    int exploration_threshold = 10,
    int quantum_phase_threshold = 5000,
    bool enable_quantum = true
) {
    const int batch_size = q_values_batch.size(0);
    const int num_actions = q_values_batch.size(1);
    auto output = torch::zeros_like(q_values_batch);
    
    // Use batch_size blocks, num_actions threads per block
    const dim3 blocks(batch_size);
    const dim3 threads(min(num_actions, 1024));  // Max 1024 threads per block
    
    batch_quantum_v5_kernel<<<blocks, threads>>>(
        q_values_batch.data_ptr<float>(),
        visit_counts_batch.data_ptr<float>(),
        priors_batch.data_ptr<float>(),
        simulation_counts.data_ptr<int>(),
        output.data_ptr<float>(),
        kappa,
        beta,
        hbar_0,
        alpha,
        parent_visits.data_ptr<float>(),
        batch_size,
        num_actions,
        exploration_threshold,
        quantum_phase_threshold,
        enable_quantum
    );
    
    return output;
}

// Quantum regime detection
torch::Tensor quantum_regime_detection_cuda(
    torch::Tensor simulation_counts,  // [batch_size]
    torch::Tensor visit_counts,       // [batch_size]
    int critical_point_1 = 1000,
    int critical_point_2 = 5000,
    int exploration_threshold = 10
) {
    const int batch_size = simulation_counts.size(0);
    auto quantum_factors = torch::zeros({batch_size}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(simulation_counts.device()));
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    quantum_regime_detection_kernel<<<blocks, threads>>>(
        simulation_counts.data_ptr<int>(),
        visit_counts.data_ptr<float>(),
        quantum_factors.data_ptr<float>(),
        batch_size,
        critical_point_1,
        critical_point_2,
        exploration_threshold
    );
    
    return quantum_factors;
}

// ============================================================================
// Python Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Quantum v5.0 CUDA kernels for MCTS - Formula: κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff)/(3 N_k)";
    
    m.def("selective_quantum_v5", &selective_quantum_v5_cuda,
          "Selective quantum v5.0 scoring (CUDA)",
          py::arg("q_values"),
          py::arg("visit_counts"),
          py::arg("priors"),
          py::arg("kappa"),
          py::arg("beta"),
          py::arg("hbar_0"),
          py::arg("alpha"),
          py::arg("parent_visits"),
          py::arg("simulation_count"),
          py::arg("exploration_threshold") = 10,
          py::arg("quantum_phase_threshold") = 5000,
          py::arg("enable_quantum") = true);
    
    m.def("batch_quantum_v5", &batch_quantum_v5_cuda,
          "Batch quantum v5.0 processing (CUDA)",
          py::arg("q_values_batch"),
          py::arg("visit_counts_batch"),
          py::arg("priors_batch"),
          py::arg("simulation_counts"),
          py::arg("parent_visits"),
          py::arg("kappa"),
          py::arg("beta"),
          py::arg("hbar_0"),
          py::arg("alpha"),
          py::arg("exploration_threshold") = 10,
          py::arg("quantum_phase_threshold") = 5000,
          py::arg("enable_quantum") = true);
    
    m.def("quantum_regime_detection", &quantum_regime_detection_cuda,
          "Quantum regime detection (CUDA)",
          py::arg("simulation_counts"),
          py::arg("visit_counts"),
          py::arg("critical_point_1") = 1000,
          py::arg("critical_point_2") = 5000,
          py::arg("exploration_threshold") = 10);
}