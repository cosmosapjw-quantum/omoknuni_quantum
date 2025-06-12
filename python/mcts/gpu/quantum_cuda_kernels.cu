// Quantum-enhanced CUDA kernels for MCTS
// This file contains optimized quantum kernels that can be compiled separately

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>

// ============================================================================
// Quantum UCB Selection Kernel
// ============================================================================

// Quantum-enhanced UCB selection kernel with integrated quantum corrections
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
    
    // Generate node-specific quantum seed for reproducible randomness
    unsigned int quantum_seed = idx * 1337 + static_cast<unsigned int>(parent_visit);
    
    // Single pass UCB selection with quantum enhancements
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        
        float q_value = (child_visit > 0) ? 
            q_values[child_idx] : 0.0f;
        
        float ucb;
        if (parent_visit > 0) {
            float exploration = c_puct * priors[i] * sqrt_parent / (1.0f + child_visit);
            ucb = q_value + exploration;
            
            // Apply quantum enhancements if enabled
            if (enable_quantum) {
                // 1. Quantum uncertainty boost for low-visit nodes
                int visit_idx = min(static_cast<int>(child_visit), max_table_size - 1);
                float quantum_boost = uncertainty_table[visit_idx];
                
                // 2. Phase-kicked prior for exploration enhancement
                float phase_kick = 0.0f;
                if (child_visit < 10.0f) {  // Apply phase kick to low-visit nodes
                    // Use quantum seed to generate pseudo-random phase
                    quantum_seed = quantum_seed * 1664525u + 1013904223u;  // LCG
                    float rand_val = static_cast<float>(quantum_seed) / 4294967295.0f;
                    phase_kick = phase_kick_strength * sinf(2.0f * 3.14159265f * rand_val);
                }
                
                // 3. Interference based on pre-computed phases
                float interference = 0.0f;
                if (quantum_phases != nullptr && i < end) {
                    float phase = quantum_phases[i];
                    interference = interference_alpha * cosf(phase);
                }
                
                // Combine quantum corrections
                float quantum_correction = quantum_boost + phase_kick + interference;
                ucb += quantum_correction;
            }
        } else {
            // Parent not visited yet - use priors with potential quantum enhancement
            ucb = priors[i];
            if (enable_quantum && quantum_phases != nullptr) {
                float phase = quantum_phases[i];
                ucb += interference_alpha * 0.1f * cosf(phase);  // Small quantum boost
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

// ============================================================================
// Quantum Phase Generation Kernel
// ============================================================================

__global__ void generate_quantum_phases_kernel(
    const int* __restrict__ node_indices,
    const int* __restrict__ edge_indices,
    const float* __restrict__ visit_counts,
    float* __restrict__ quantum_phases,
    const int num_edges,
    const float base_frequency,
    const float visit_modulation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    
    int node_idx = node_indices[idx];
    int edge_idx = edge_indices[idx];
    float visits = visit_counts[node_idx];
    
    // Generate quantum phase based on node/edge properties
    float phase = base_frequency * static_cast<float>(edge_idx);
    phase += visit_modulation * logf(1.0f + visits);
    
    // Wrap phase to [0, 2π]
    phase = fmodf(phase, 2.0f * 3.14159265f);
    quantum_phases[idx] = phase;
}

// ============================================================================
// C++ Interface Functions
// ============================================================================

// Quantum-enhanced UCB selection with CUDA acceleration
std::tuple<torch::Tensor, torch::Tensor> batched_ucb_selection_quantum_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct,
    // Quantum parameters
    torch::Tensor quantum_phases,
    torch::Tensor uncertainty_table,
    float hbar_eff,
    float phase_kick_strength,
    float interference_alpha,
    bool enable_quantum
) {
    const int num_nodes = parent_visits.size(0);
    auto selected_actions = torch::zeros({num_nodes}, 
        torch::TensorOptions().dtype(torch::kInt32).device(q_values.device()));
    auto selected_scores = torch::zeros({num_nodes}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(q_values.device()));
    
    const int threads = 256;
    const int blocks = (num_nodes + threads - 1) / threads;
    
    // Prepare quantum parameters
    const float* quantum_phases_ptr = quantum_phases.numel() > 0 ? 
        quantum_phases.data_ptr<float>() : nullptr;
    const float* uncertainty_table_ptr = uncertainty_table.numel() > 0 ? 
        uncertainty_table.data_ptr<float>() : nullptr;
    const int max_table_size = uncertainty_table.numel();
    
    // Launch quantum-enhanced kernel
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
        quantum_phases_ptr,
        uncertainty_table_ptr,
        max_table_size,
        hbar_eff,
        phase_kick_strength,
        interference_alpha,
        enable_quantum
    );
    
    return std::make_tuple(selected_actions, selected_scores);
}

// Generate quantum phases for edges
torch::Tensor generate_quantum_phases_cuda(
    torch::Tensor node_indices,
    torch::Tensor edge_indices,
    torch::Tensor visit_counts,
    float base_frequency,
    float visit_modulation
) {
    const int num_edges = edge_indices.size(0);
    auto quantum_phases = torch::zeros({num_edges}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(node_indices.device()));
    
    const int threads = 256;
    const int blocks = (num_edges + threads - 1) / threads;
    
    generate_quantum_phases_kernel<<<blocks, threads>>>(
        node_indices.data_ptr<int>(),
        edge_indices.data_ptr<int>(),
        visit_counts.data_ptr<float>(),
        quantum_phases.data_ptr<float>(),
        num_edges,
        base_frequency,
        visit_modulation
    );
    
    return quantum_phases;
}

// ============================================================================
// Python Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Quantum-enhanced CUDA kernels for MCTS";
    
    m.def("batched_ucb_selection_quantum", &batched_ucb_selection_quantum_cuda,
          "Quantum-enhanced batched UCB selection (CUDA)",
          py::arg("q_values"),
          py::arg("visit_counts"),
          py::arg("parent_visits"),
          py::arg("priors"),
          py::arg("row_ptr"),
          py::arg("col_indices"),
          py::arg("c_puct"),
          py::arg("quantum_phases"),
          py::arg("uncertainty_table"),
          py::arg("hbar_eff") = 0.05f,
          py::arg("phase_kick_strength") = 0.1f,
          py::arg("interference_alpha") = 0.05f,
          py::arg("enable_quantum") = true);
    
    m.def("generate_quantum_phases", &generate_quantum_phases_cuda,
          "Generate quantum phases for edges (CUDA)",
          py::arg("node_indices"),
          py::arg("edge_indices"),
          py::arg("visit_counts"),
          py::arg("base_frequency") = 0.1f,
          py::arg("visit_modulation") = 0.01f);
}