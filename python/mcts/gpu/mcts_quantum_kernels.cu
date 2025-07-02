// Quantum-enhanced CUDA kernels for MCTS
// Part of the refactored unified kernel system for faster compilation

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// Quantum-Enhanced MCTS Kernels
// ============================================================================

// Quantum-enhanced UCB selection kernel
__global__ void batched_ucb_selection_quantum_kernel(
    const float* __restrict__ q_values,
    const int* __restrict__ visit_counts,
    const int* __restrict__ parent_visits,
    const float* __restrict__ priors,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    int* __restrict__ selected_actions,
    float* __restrict__ selected_scores,
    const int64_t num_nodes,
    const float c_puct,
    // Quantum parameters
    const float* __restrict__ quantum_phases,     // Pre-computed phases [num_edges]
    const float* __restrict__ uncertainty_table,  // Quantum uncertainty lookup [max_visits]
    const int max_table_size,                     // Size of uncertainty table
    const float hbar_eff,                         // Effective Planck constant
    const float phase_kick_strength,              // Phase kick amplitude
    const float interference_alpha,               // Interference strength
    const bool enable_quantum                     // Quantum features enable flag
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    int start = row_ptr[idx];
    int end = row_ptr[idx + 1];
    
    if (start == end) {
        selected_actions[idx] = -1;
        selected_scores[idx] = 0.0f;
        return;
    }
    
    float parent_visit = static_cast<float>(parent_visits[idx]);
    float sqrt_parent = sqrtf(parent_visit + 1.0f);
    
    float best_ucb = -1e10f;
    int best_action = 0;
    
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        
        float ucb;
        if (parent_visit > 0) {
            float exploration = c_puct * priors[i] * sqrt_parent / (1.0f + child_visit);
            ucb = q_value + exploration;
            
            // Apply quantum enhancements if enabled
            if (enable_quantum) {
                // Quantum uncertainty correction
                int visit_idx = min((int)child_visit, max_table_size - 1);
                float uncertainty = uncertainty_table[visit_idx];
                
                // Phase-kicked exploration
                float phase = quantum_phases[i];
                float phase_kick = phase_kick_strength * sinf(phase * hbar_eff);
                
                // Interference term
                float interference = interference_alpha * cosf(phase * 2.0f);
                
                ucb += phase_kick * exploration + interference * uncertainty;
            }
        } else {
            ucb = priors[i];
            
            if (enable_quantum) {
                float phase = quantum_phases[i];
                float phase_kick = phase_kick_strength * sinf(phase * hbar_eff);
                ucb += phase_kick;
            }
        }
        
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_action = i - start;
        }
    }
    
    selected_actions[idx] = best_action;
    selected_scores[idx] = best_ucb;
}

// Quantum interference kernel for wave function collapse
__global__ void quantum_interference_kernel(
    float* __restrict__ action_probs,
    const float* __restrict__ quantum_phases,
    const float* __restrict__ amplitudes,
    const int num_actions,
    const float interference_strength,
    const float decoherence_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_actions) return;
    
    float phase = quantum_phases[idx];
    float amplitude = amplitudes[idx];
    float prob = action_probs[idx];
    
    // Apply quantum interference
    float interference = interference_strength * cosf(phase) * amplitude;
    float decoherence = expf(-decoherence_rate * phase * phase);
    
    // Quantum-corrected probability
    prob = prob * (1.0f + interference * decoherence);
    
    // Ensure non-negative and normalized
    action_probs[idx] = fmaxf(prob, 1e-8f);
}

// Apply quantum interference to action probabilities
__global__ void apply_interference_kernel(
    float* __restrict__ action_probs,
    const float* __restrict__ phases,
    const int num_actions,
    const float interference_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_actions) return;
    
    float phase = phases[idx];
    float prob = action_probs[idx];
    
    // Interference modulation
    float interference = interference_strength * cosf(phase);
    prob = prob * (1.0f + interference);
    
    // Clamp to valid probability range
    action_probs[idx] = fmaxf(prob, 1e-8f);
}

// Phase-kicked policy kernel for quantum exploration
__global__ void phase_kicked_policy_kernel(
    float* __restrict__ policy,
    const float* __restrict__ quantum_phases,
    const int* __restrict__ visit_counts,
    const int num_actions,
    const float kick_strength,
    const float hbar_eff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_actions) return;
    
    float phase = quantum_phases[idx];
    float visits = static_cast<float>(visit_counts[idx]);
    
    // Phase kick strength decreases with visits (decoherence)
    float decoherence = expf(-visits * 0.1f);
    float kick = kick_strength * sinf(phase * hbar_eff) * decoherence;
    
    // Apply phase kick to policy
    policy[idx] = policy[idx] * (1.0f + kick);
    
    // Ensure non-negative
    policy[idx] = fmaxf(policy[idx], 1e-8f);
}

// Quantum path integral kernel for action selection
__global__ void quantum_path_integral_kernel(
    float* __restrict__ path_weights,
    const float* __restrict__ path_phases,
    const float* __restrict__ path_values,
    const int* __restrict__ path_lengths,
    const int num_paths,
    const float hbar_eff,
    const float beta_inverse_temp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;
    
    float phase = path_phases[idx];
    float value = path_values[idx];
    int length = path_lengths[idx];
    
    // Quantum action (phase contribution)
    float action = phase * hbar_eff;
    
    // Classical Boltzmann weight
    float boltzmann = expf(beta_inverse_temp * value);
    
    // Quantum path weight = |amplitude|^2
    float quantum_weight = cosf(action) * cosf(action) + sinf(action) * sinf(action);
    
    // Combined weight with path length penalty
    float length_penalty = expf(-length * 0.01f);
    path_weights[idx] = boltzmann * quantum_weight * length_penalty;
}

// Decoherence dynamics kernel
__global__ void decoherence_dynamics_kernel(
    float* __restrict__ coherence_matrix,
    const float* __restrict__ visit_counts,
    const int matrix_size,
    const float decoherence_rate,
    const float dt
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= matrix_size || col >= matrix_size) return;
    
    int idx = row * matrix_size + col;
    
    if (row == col) {
        // Diagonal elements (populations) - conserved
        return;
    }
    
    // Off-diagonal elements (coherences) decay
    float visits = visit_counts[row] + visit_counts[col];
    float decay = expf(-decoherence_rate * visits * dt);
    
    coherence_matrix[idx] *= decay;
}

// Quantum entanglement entropy calculation
__global__ void entanglement_entropy_kernel(
    const float* __restrict__ density_matrix,
    float* __restrict__ entropy_values,
    const int matrix_size,
    const int num_subsystems
) {
    int subsystem_idx = blockIdx.x;
    if (subsystem_idx >= num_subsystems) return;
    
    int tid = threadIdx.x;
    __shared__ float eigenvalues[256];  // Assuming max 256 eigenvalues
    
    // Simple diagonal approximation for entropy calculation
    if (tid < matrix_size) {
        int diag_idx = subsystem_idx * matrix_size + tid * matrix_size + tid;
        float prob = density_matrix[diag_idx];
        
        // von Neumann entropy contribution: -p * log(p)
        eigenvalues[tid] = (prob > 1e-10f) ? -prob * logf(prob) : 0.0f;
    } else {
        eigenvalues[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Parallel reduction to sum entropy contributions
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            eigenvalues[tid] += eigenvalues[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        entropy_values[subsystem_idx] = eigenvalues[0];
    }
}

// ============================================================================
// Python Interface Functions
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> batched_ucb_selection_quantum_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct,
    torch::Tensor quantum_phases,
    torch::Tensor uncertainty_table,
    float hbar_eff,
    float phase_kick_strength,
    float interference_alpha,
    bool enable_quantum
) {
    int num_nodes = q_values.size(0);
    int max_table_size = uncertainty_table.size(0);
    
    auto selected_actions = torch::zeros({num_nodes}, torch::dtype(torch::kInt32).device(q_values.device()));
    auto selected_scores = torch::zeros({num_nodes}, torch::dtype(torch::kFloat32).device(q_values.device()));
    
    const int threads = 256;
    const int blocks = (num_nodes + threads - 1) / threads;
    
    batched_ucb_selection_quantum_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        parent_visits.data_ptr<int>(),
        priors.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        selected_actions.data_ptr<int>(),
        selected_scores.data_ptr<float>(),
        num_nodes,
        c_puct,
        quantum_phases.data_ptr<float>(),
        uncertainty_table.data_ptr<float>(),
        max_table_size,
        hbar_eff,
        phase_kick_strength,
        interference_alpha,
        enable_quantum
    );
    
    return std::make_tuple(selected_actions, selected_scores);
}

torch::Tensor apply_quantum_interference_cuda(
    torch::Tensor action_probs,
    torch::Tensor phases,
    float interference_strength
) {
    int num_actions = action_probs.size(0);
    
    auto result = action_probs.clone();
    
    const int threads = 256;
    const int blocks = (num_actions + threads - 1) / threads;
    
    apply_interference_kernel<<<blocks, threads>>>(
        result.data_ptr<float>(),
        phases.data_ptr<float>(),
        num_actions,
        interference_strength
    );
    
    return result;
}

torch::Tensor phase_kicked_policy_cuda(
    torch::Tensor policy,
    torch::Tensor quantum_phases,
    torch::Tensor visit_counts,
    float kick_strength,
    float hbar_eff
) {
    int num_actions = policy.size(0);
    
    auto result = policy.clone();
    
    const int threads = 256;
    const int blocks = (num_actions + threads - 1) / threads;
    
    phase_kicked_policy_kernel<<<blocks, threads>>>(
        result.data_ptr<float>(),
        quantum_phases.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        num_actions,
        kick_strength,
        hbar_eff
    );
    
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_ucb_selection_quantum", &batched_ucb_selection_quantum_cuda, "Quantum UCB selection CUDA");
    m.def("apply_quantum_interference", &apply_quantum_interference_cuda, "Apply quantum interference CUDA");
    m.def("phase_kicked_policy", &phase_kicked_policy_cuda, "Phase-kicked policy CUDA");
}