// Custom CUDA kernels for high-performance MCTS
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for batched UCB selection
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

// CUDA kernel for parallel backup
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
            value = -value;  // Flip value for two-player games
        }
    }
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