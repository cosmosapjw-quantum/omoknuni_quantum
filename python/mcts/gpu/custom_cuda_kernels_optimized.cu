// Optimized CUDA kernels for high-performance MCTS
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// UCB selection kernel
__global__ void batched_ucb_selection_kernel(
    const float* __restrict__ q_values,
    const int* __restrict__ visit_counts,
    const int* __restrict__ parent_visits,
    const float* __restrict__ priors,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    int* __restrict__ selected_actions,
    const int num_nodes,
    const float c_puct
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    int start = row_ptr[idx];
    int end = row_ptr[idx + 1];
    
    if (start == end) {
        selected_actions[idx] = -1;
        return;
    }
    
    float parent_visit = static_cast<float>(parent_visits[idx]);
    float sqrt_parent = sqrtf(parent_visit);
    
    float best_ucb = -1e10f;
    int best_action = 0;
    
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        
        float q_value = (child_visit > 0) ? 
            q_values[child_idx] / child_visit : 0.0f;
        
        float exploration = c_puct * priors[i] * sqrt_parent / (1.0f + child_visit);
        float ucb = q_value + exploration;
        
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_action = i - start;
        }
    }
    
    selected_actions[idx] = best_action;
}

// Parallel backup kernel
__global__ void parallel_backup_kernel(
    const int* __restrict__ paths,
    const float* __restrict__ leaf_values,
    const int* __restrict__ path_lengths,
    float* __restrict__ value_sums,
    int* __restrict__ visit_counts,
    const int batch_size,
    const int max_depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float value = leaf_values[idx];
    int length = path_lengths[idx];
    
    for (int depth = 0; depth < length && depth < max_depth; depth++) {
        int node_idx = paths[idx * max_depth + depth];
        if (node_idx >= 0) {
            atomicAdd(&value_sums[node_idx], value);
            atomicAdd(&visit_counts[node_idx], 1);
            value = -value;
        }
    }
}

// Quantum interference kernel (simplified)
__global__ void quantum_interference_kernel(
    const float* __restrict__ q_values,
    const float* __restrict__ visit_counts,
    const float* __restrict__ priors,
    const float* __restrict__ phases,
    float* __restrict__ ucb_scores,
    const int batch_size,
    const int max_actions,
    const float c_puct,
    const float hbar_eff,
    const float lambda_qft
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / max_actions;
    int action_idx = idx % max_actions;
    
    if (batch_idx >= batch_size || action_idx >= max_actions) return;
    
    float visit = visit_counts[idx];
    float q_val = (visit > 0) ? q_values[idx] / visit : 0.0f;
    float prior = priors[idx];
    float phase = phases[idx];
    
    // Standard UCB
    float parent_visit = 0.0f;
    for (int i = 0; i < max_actions; i++) {
        parent_visit += visit_counts[batch_idx * max_actions + i];
    }
    float sqrt_parent = sqrtf(parent_visit + 1.0f);
    float exploration = c_puct * prior * sqrt_parent / (1.0f + visit);
    
    // Quantum correction with phase
    float quantum_factor = expf(-lambda_qft / (hbar_eff * hbar_eff));
    float interference = quantum_factor * cosf(phase);
    
    ucb_scores[idx] = q_val + exploration * (1.0f + 0.1f * interference);
}

// C++ interface functions
torch::Tensor batched_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct
) {
    const int num_nodes = parent_visits.size(0);
    auto selected_actions = torch::zeros({num_nodes}, 
        torch::TensorOptions().dtype(torch::kInt32).device(q_values.device()));
    
    const int threads = 256;
    const int blocks = (num_nodes + threads - 1) / threads;
    
    batched_ucb_selection_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        parent_visits.data_ptr<int>(),
        priors.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        selected_actions.data_ptr<int>(),
        num_nodes,
        c_puct
    );
    
    return selected_actions;
}

torch::Tensor parallel_backup_cuda(
    torch::Tensor paths,
    torch::Tensor leaf_values,
    torch::Tensor path_lengths,
    torch::Tensor value_sums,
    torch::Tensor visit_counts
) {
    const int batch_size = paths.size(0);
    const int max_depth = paths.size(1);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    parallel_backup_kernel<<<blocks, threads>>>(
        paths.data_ptr<int>(),
        leaf_values.data_ptr<float>(),
        path_lengths.data_ptr<int>(),
        value_sums.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        batch_size,
        max_depth
    );
    
    return value_sums;
}

torch::Tensor quantum_interference_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor priors,
    torch::Tensor phases,
    float c_puct,
    float hbar_eff,
    float lambda_qft
) {
    const int batch_size = q_values.size(0);
    const int max_actions = q_values.size(1);
    const int total_elements = batch_size * max_actions;
    
    auto ucb_scores = torch::zeros_like(q_values);
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    quantum_interference_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<float>(),
        priors.data_ptr<float>(),
        phases.data_ptr<float>(),
        ucb_scores.data_ptr<float>(),
        batch_size,
        max_actions,
        c_puct,
        hbar_eff,
        lambda_qft
    );
    
    return ucb_scores;
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_ucb_selection", &batched_ucb_selection_cuda, "Batched UCB selection (CUDA)");
    m.def("parallel_backup", &parallel_backup_cuda, "Parallel backup (CUDA)");
    m.def("quantum_interference", &quantum_interference_cuda, "Quantum interference (CUDA)");
}