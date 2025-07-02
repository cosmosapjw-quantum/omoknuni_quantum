// UCB Selection CUDA kernels for MCTS node selection
// Part of the refactored unified kernel system for faster compilation

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// UCB Selection Kernels
// ============================================================================

// Standard UCB selection kernel
__global__ void batched_ucb_selection_kernel(
    const float* __restrict__ q_values,
    const int* __restrict__ visit_counts,
    const int* __restrict__ parent_visits,
    const float* __restrict__ priors,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    int* __restrict__ selected_actions,
    float* __restrict__ selected_scores,
    const int64_t num_nodes,
    const float c_puct
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
    
    // Simple single pass - first maximum wins (moves already shuffled)
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        
        float q_value = (child_visit > 0) ? 
            q_values[child_idx] : 0.0f;  // Q-values already normalized
        
        float ucb;
        if (parent_visit > 0) {
            float exploration = c_puct * priors[i] * sqrt_parent / (1.0f + child_visit);
            ucb = q_value + exploration;
        } else {
            // Parent not visited yet - use priors directly
            ucb = priors[i];
        }
        
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_action = i - start;
        }
    }
    
    selected_actions[idx] = best_action;
    selected_scores[idx] = best_ucb;
}

// Optimized UCB selection with shared memory
__global__ void optimized_ucb_selection_kernel(
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
    const int max_children
) {
    int idx = blockIdx.x;
    if (idx >= num_nodes) return;
    
    int start = row_ptr[idx];
    int end = row_ptr[idx + 1];
    int num_children = end - start;
    
    if (num_children == 0) {
        if (threadIdx.x == 0) {
            selected_actions[idx] = -1;
            selected_scores[idx] = 0.0f;
        }
        return;
    }
    
    extern __shared__ float shared_data[];
    float* ucb_scores = shared_data;
    
    float parent_visit = static_cast<float>(parent_visits[idx]);
    float sqrt_parent = sqrtf(parent_visit + 1.0f);
    
    // Each thread computes UCB for one child
    int tid = threadIdx.x;
    if (tid < num_children) {
        int child_idx = col_indices[start + tid];
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        
        float ucb;
        if (parent_visit > 0) {
            float exploration = c_puct * priors[start + tid] * sqrt_parent / (1.0f + child_visit);
            ucb = q_value + exploration;
        } else {
            ucb = priors[start + tid];
        }
        
        ucb_scores[tid] = ucb;
    } else {
        ucb_scores[tid] = -1e10f;
    }
    
    __syncthreads();
    
    // Parallel reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < num_children) {
            if (ucb_scores[tid + stride] > ucb_scores[tid]) {
                ucb_scores[tid] = ucb_scores[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Find the actual index with maximum UCB
    if (tid == 0) {
        float max_ucb = ucb_scores[0];
        int best_action = 0;
        
        for (int i = 0; i < num_children; i++) {
            if (ucb_scores[i] == max_ucb) {
                best_action = i;
                break;
            }
        }
        
        selected_actions[idx] = best_action;
        selected_scores[idx] = max_ucb;
    }
}

// Vectorized UCB computation for multiple paths
__global__ void vectorized_ucb_selection_kernel(
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
    const int wave_size
) {
    int wave_idx = blockIdx.x;
    int node_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (wave_idx >= wave_size || node_idx >= num_nodes) return;
    
    int linear_idx = wave_idx * num_nodes + node_idx;
    int start = row_ptr[linear_idx];
    int end = row_ptr[linear_idx + 1];
    
    if (start == end) {
        if (tid == 0) {
            selected_actions[linear_idx] = -1;
            selected_scores[linear_idx] = 0.0f;
        }
        return;
    }
    
    float parent_visit = static_cast<float>(parent_visits[linear_idx]);
    float sqrt_parent = sqrtf(parent_visit + 1.0f);
    
    float best_ucb = -1e10f;
    int best_action = 0;
    
    // Each thread processes multiple children
    for (int i = start + tid; i < end; i += blockDim.x) {
        int child_idx = col_indices[i];
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        
        float ucb;
        if (parent_visit > 0) {
            float exploration = c_puct * priors[i] * sqrt_parent / (1.0f + child_visit);
            ucb = q_value + exploration;
        } else {
            ucb = priors[i];
        }
        
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_action = i - start;
        }
    }
    
    // Reduce across threads in block
    __shared__ float shared_ucb[256];
    __shared__ int shared_action[256];
    
    shared_ucb[tid] = best_ucb;
    shared_action[tid] = best_action;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_ucb[tid + stride] > shared_ucb[tid]) {
                shared_ucb[tid] = shared_ucb[tid + stride];
                shared_action[tid] = shared_action[tid + stride];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        selected_actions[linear_idx] = shared_action[0];
        selected_scores[linear_idx] = shared_ucb[0];
    }
}

// UCB selection with exploration bonus
__global__ void exploration_enhanced_ucb_kernel(
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
    const float exploration_bonus,
    const int* __restrict__ node_depths
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
    int depth = node_depths[idx];
    
    // Depth-dependent exploration bonus
    float depth_bonus = exploration_bonus * expf(-depth * 0.1f);
    
    float best_ucb = -1e10f;
    int best_action = 0;
    
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        
        float ucb;
        if (parent_visit > 0) {
            float exploration = c_puct * priors[i] * sqrt_parent / (1.0f + child_visit);
            float bonus = (child_visit == 0) ? depth_bonus : 0.0f;
            ucb = q_value + exploration + bonus;
        } else {
            ucb = priors[i] + depth_bonus;
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
// Python Interface Functions
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor> batched_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct
) {
    int num_nodes = q_values.size(0);
    
    auto selected_actions = torch::zeros({num_nodes}, torch::dtype(torch::kInt32).device(q_values.device()));
    auto selected_scores = torch::zeros({num_nodes}, torch::dtype(torch::kFloat32).device(q_values.device()));
    
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
        selected_scores.data_ptr<float>(),
        num_nodes,
        c_puct
    );
    
    return std::make_tuple(selected_actions, selected_scores);
}

std::tuple<torch::Tensor, torch::Tensor> optimized_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct,
    int max_children
) {
    int num_nodes = q_values.size(0);
    
    auto selected_actions = torch::zeros({num_nodes}, torch::dtype(torch::kInt32).device(q_values.device()));
    auto selected_scores = torch::zeros({num_nodes}, torch::dtype(torch::kFloat32).device(q_values.device()));
    
    const int threads = 256;
    const int shared_mem = threads * sizeof(float);
    
    optimized_ucb_selection_kernel<<<num_nodes, threads, shared_mem>>>(
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
        max_children
    );
    
    return std::make_tuple(selected_actions, selected_scores);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_ucb_selection", &batched_ucb_selection_cuda, "Batched UCB selection CUDA");
    m.def("optimized_ucb_selection", &optimized_ucb_selection_cuda, "Optimized UCB selection CUDA");
}