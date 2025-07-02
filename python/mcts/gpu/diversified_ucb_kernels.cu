// Enhanced CUDA kernels for diversified UCB selection with per-simulation Dirichlet noise
// This optimizes the critical path by applying different priors per simulation in the wave

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Diversified UCB selection kernel that uses different priors for each simulation
__global__ void batched_ucb_selection_diversified_kernel(
    const float* __restrict__ q_values,           // Q-values for all nodes
    const int* __restrict__ visit_counts,         // Visit counts for all nodes  
    const int* __restrict__ parent_visits,        // Parent visit counts
    const float* __restrict__ diversified_priors, // (wave_size, max_children) priors per simulation
    const int* __restrict__ row_ptr,              // CSR row pointers
    const int* __restrict__ col_indices,          // CSR column indices
    int* __restrict__ selected_actions,           // Output: selected actions per simulation
    float* __restrict__ selected_scores,          // Output: UCB scores for debugging
    const int64_t num_nodes,                      // Number of nodes to process
    const int64_t wave_size,                      // Number of parallel simulations
    const int64_t max_children,                   // Maximum children per node
    const float c_puct                            // UCB exploration parameter
) {
    // Each thread handles one (node, simulation) pair
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int node_idx = global_idx / wave_size;
    int sim_idx = global_idx % wave_size;
    
    if (node_idx >= num_nodes || sim_idx >= wave_size) return;
    
    int start = row_ptr[node_idx];
    int end = row_ptr[node_idx + 1];
    
    if (start == end) {
        selected_actions[global_idx] = -1;
        selected_scores[global_idx] = 0.0f;
        return;
    }
    
    float parent_visit = static_cast<float>(parent_visits[node_idx]);
    float sqrt_parent = sqrtf(parent_visit + 1.0f);
    
    // Find best action for this (node, simulation) pair
    float best_ucb = -1e10f;
    int best_action = 0;
    
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        int action_idx = i - start;
        
        // Get simulation-specific prior
        float prior = diversified_priors[sim_idx * max_children + action_idx];
        
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        
        // Compute UCB with simulation-specific prior
        float exploration = c_puct * prior * sqrt_parent / (1.0f + child_visit);
        float ucb = q_value + exploration;
        
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_action = action_idx;
        }
    }
    
    selected_actions[global_idx] = best_action;
    selected_scores[global_idx] = best_ucb;
}

// Vectorized kernel for generating diversified Dirichlet noise on GPU
__global__ void generate_diversified_dirichlet_kernel(
    float* __restrict__ diversified_priors,       // Output: (wave_size, num_actions) 
    const float* __restrict__ base_priors,        // Input: base priors from NN
    const float* __restrict__ dirichlet_samples,  // Input: (wave_size, num_actions) random samples
    const int64_t wave_size,
    const int64_t num_actions,
    const float epsilon                           // Mixing parameter
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sim_idx = idx / num_actions;
    int action_idx = idx % num_actions;
    
    if (sim_idx >= wave_size || action_idx >= num_actions) return;
    
    float base = base_priors[action_idx];
    float noise = dirichlet_samples[sim_idx * num_actions + action_idx];
    
    // Mix base prior with simulation-specific noise
    diversified_priors[idx] = (1.0f - epsilon) * base + epsilon * noise;
}

// Fused kernel: combines diversified UCB selection with path traversal
__global__ void fused_diversified_selection_traversal_kernel(
    const float* __restrict__ q_values,
    const int* __restrict__ visit_counts,
    const float* __restrict__ diversified_priors,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    int* __restrict__ paths,                      // Output: (wave_size, max_depth) paths
    int* __restrict__ path_lengths,               // Output: (wave_size,) path lengths
    int* __restrict__ leaf_nodes,                 // Output: (wave_size,) leaf nodes
    const int64_t wave_size,
    const int64_t max_children,
    const int64_t max_depth,
    const float c_puct
) {
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim_idx >= wave_size) return;
    
    int current_node = 0;  // Start at root
    int depth = 0;
    
    // Initialize path
    paths[sim_idx * max_depth + 0] = current_node;
    
    // Traverse until leaf or max depth
    while (depth < max_depth - 1) {
        int start = row_ptr[current_node];
        int end = row_ptr[current_node + 1];
        
        // If no children, this is a leaf
        if (start == end) break;
        
        // Find best action using simulation-specific priors
        float parent_visit = static_cast<float>(visit_counts[current_node]);
        float sqrt_parent = sqrtf(parent_visit + 1.0f);
        
        float best_ucb = -1e10f;
        int best_child = -1;
        
        for (int i = start; i < end; i++) {
            int child_idx = col_indices[i];
            int action_idx = i - start;
            
            // Use simulation-specific prior
            float prior = diversified_priors[sim_idx * max_children + action_idx];
            
            float child_visit = static_cast<float>(visit_counts[child_idx]);
            float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
            
            float exploration = c_puct * prior * sqrt_parent / (1.0f + child_visit);
            float ucb = q_value + exploration;
            
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = child_idx;
            }
        }
        
        if (best_child == -1) break;
        
        depth++;
        current_node = best_child;
        paths[sim_idx * max_depth + depth] = current_node;
    }
    
    path_lengths[sim_idx] = depth;
    leaf_nodes[sim_idx] = current_node;
}

// Python binding functions
torch::Tensor diversified_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor diversified_priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    int64_t wave_size,
    int64_t max_children,
    float c_puct
) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(q_values.device());
    auto selected_actions = torch::zeros({wave_size * q_values.size(0)}, options);
    auto selected_scores = torch::zeros({wave_size * q_values.size(0)}, q_values.options());
    
    const int threads = 256;
    const int blocks = (wave_size * q_values.size(0) + threads - 1) / threads;
    
    batched_ucb_selection_diversified_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        parent_visits.data_ptr<int>(),
        diversified_priors.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        selected_actions.data_ptr<int>(),
        selected_scores.data_ptr<float>(),
        q_values.size(0),
        wave_size,
        max_children,
        c_puct
    );
    
    return selected_actions;
}

torch::Tensor generate_diversified_dirichlet_cuda(
    torch::Tensor base_priors,
    torch::Tensor dirichlet_samples,
    int64_t wave_size,
    float epsilon
) {
    auto diversified_priors = torch::zeros({wave_size, base_priors.size(0)}, base_priors.options());
    
    const int threads = 256;
    const int blocks = (wave_size * base_priors.size(0) + threads - 1) / threads;
    
    generate_diversified_dirichlet_kernel<<<blocks, threads>>>(
        diversified_priors.data_ptr<float>(),
        base_priors.data_ptr<float>(),
        dirichlet_samples.data_ptr<float>(),
        wave_size,
        base_priors.size(0),
        epsilon
    );
    
    return diversified_priors;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_diversified_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor diversified_priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    int64_t wave_size,
    int64_t max_children,
    int64_t max_depth,
    float c_puct
) {
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(q_values.device());
    auto paths = torch::full({wave_size, max_depth}, -1, int_options);
    auto path_lengths = torch::zeros({wave_size}, int_options);
    auto leaf_nodes = torch::zeros({wave_size}, int_options);
    
    const int threads = 256;
    const int blocks = (wave_size + threads - 1) / threads;
    
    fused_diversified_selection_traversal_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        diversified_priors.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        paths.data_ptr<int>(),
        path_lengths.data_ptr<int>(),
        leaf_nodes.data_ptr<int>(),
        wave_size,
        max_children,
        max_depth,
        c_puct
    );
    
    return std::make_tuple(paths, path_lengths, leaf_nodes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("diversified_ucb_selection", &diversified_ucb_selection_cuda, "Diversified UCB Selection");
    m.def("generate_diversified_dirichlet", &generate_diversified_dirichlet_cuda, "Generate Diversified Dirichlet");
    m.def("fused_diversified_selection", &fused_diversified_selection_cuda, "Fused Diversified Selection");
}