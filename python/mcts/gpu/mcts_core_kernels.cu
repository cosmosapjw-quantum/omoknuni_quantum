// Core MCTS CUDA kernels for tree operations
// Part of the refactored unified kernel system for faster compilation

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// Core MCTS Tree Operations
// ============================================================================

// Kernel to find nodes needing expansion in parallel
__global__ void find_expansion_nodes_kernel(
    const int* __restrict__ current_nodes,     // Current nodes from all paths
    const int* __restrict__ children,          // Children lookup table [num_nodes * max_children]
    const int* __restrict__ visit_counts,      // Visit counts for all nodes
    const bool* __restrict__ valid_path_mask,  // Which paths are valid
    int* __restrict__ expansion_nodes,         // Output: nodes needing expansion
    int* __restrict__ expansion_count,         // Output: number of nodes to expand
    const int64_t wave_size,
    const int64_t max_children,
    const int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= wave_size || !valid_path_mask[tid]) return;
    
    int node_idx = current_nodes[tid];
    if (node_idx < 0 || node_idx >= num_nodes) return;
    
    // Check if node has children
    bool has_children = false;
    int base_idx = node_idx * max_children;
    
    #pragma unroll 8
    for (int i = 0; i < max_children; i++) {
        if (children[base_idx + i] >= 0) {
            has_children = true;
            break;
        }
    }
    
    // Check if node needs expansion (no children and unvisited)
    if (!has_children && visit_counts[node_idx] == 0) {
        // Atomic add to get unique index
        int idx = atomicAdd(expansion_count, 1);
        if (idx < wave_size) {  // Prevent overflow
            expansion_nodes[idx] = node_idx;
        }
    }
}

// Kernel to process legal moves and normalize priors in batch
__global__ void batch_process_legal_moves_kernel(
    const float* __restrict__ raw_policies,    // Raw NN output [num_states * action_size]
    const int* __restrict__ board_states,      // Board states [num_states * board_size]
    float* __restrict__ normalized_priors,     // Output: normalized priors
    int* __restrict__ legal_move_indices,      // Output: indices of legal moves
    int* __restrict__ legal_move_counts,       // Output: number of legal moves per state
    const int num_states,
    const int action_size
) {
    int state_idx = blockIdx.x;
    if (state_idx >= num_states) return;
    
    __shared__ float shared_sum;
    __shared__ int shared_count;
    
    if (threadIdx.x == 0) {
        shared_sum = 0.0f;
        shared_count = 0;
    }
    __syncthreads();
    
    int base_idx = state_idx * action_size;
    
    // First pass: find legal moves and compute sum
    for (int i = threadIdx.x; i < action_size; i += blockDim.x) {
        int idx = base_idx + i;
        // For Gomoku: legal move = empty square (0)
        if (board_states[idx] == 0) {
            float prior = raw_policies[idx];
            atomicAdd(&shared_sum, prior);
            int local_idx = atomicAdd(&shared_count, 1);
            if (local_idx < action_size) {
                legal_move_indices[base_idx + local_idx] = i;
            }
        }
    }
    __syncthreads();
    
    float sum = shared_sum;
    int count = shared_count;
    
    if (threadIdx.x == 0) {
        legal_move_counts[state_idx] = count;
    }
    
    // Second pass: normalize priors
    if (count > 0) {
        float norm_factor = (sum > 0) ? (1.0f / sum) : (1.0f / count);
        
        for (int i = threadIdx.x; i < action_size; i += blockDim.x) {
            int idx = base_idx + i;
            if (board_states[idx] == 0) {
                normalized_priors[idx] = (sum > 0) ? 
                    raw_policies[idx] * norm_factor : norm_factor;
            } else {
                normalized_priors[idx] = 0.0f;
            }
        }
    }
}

// Parallel backup kernel for MCTS value propagation
__global__ void parallel_backup_kernel(
    float* __restrict__ q_values,
    int* __restrict__ visit_counts,
    const int* __restrict__ path_nodes,
    const float* __restrict__ leaf_values,
    const int* __restrict__ path_lengths,
    const int num_paths
) {
    int path_idx = blockIdx.x;
    if (path_idx >= num_paths) return;
    
    int path_length = path_lengths[path_idx];
    if (path_length <= 0) return;
    
    float value = leaf_values[path_idx];
    
    // Backup along the path
    for (int depth = path_length - 1; depth >= 0; depth--) {
        int node_idx = path_nodes[path_idx * 256 + depth]; // Assuming max depth 256
        if (node_idx >= 0) {
            // Atomic updates for thread safety
            int old_visits = atomicAdd(&visit_counts[node_idx], 1);
            
            // Update Q-value using incremental mean
            float old_q = q_values[node_idx];
            float new_q = (old_q * old_visits + value) / (old_visits + 1);
            q_values[node_idx] = new_q;
            
            // Flip value for opponent's perspective
            value = -value;
        }
    }
}

// Kernel to apply moves in batch
__global__ void batch_apply_moves_kernel(
    int* __restrict__ board_states,
    const int* __restrict__ move_indices,
    const int* __restrict__ players,
    const int num_moves,
    const int board_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_moves) return;
    
    int move_idx = move_indices[idx];
    int player = players[idx];
    
    if (move_idx >= 0 && move_idx < board_size * board_size) {
        board_states[idx * board_size * board_size + move_idx] = player;
    }
}

// Generate legal moves mask kernel
__global__ void generate_legal_moves_mask_kernel(
    const int* __restrict__ board_states,
    bool* __restrict__ legal_moves_mask,
    const int num_states,
    const int board_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_positions = num_states * board_size * board_size;
    
    if (idx >= total_positions) return;
    
    // Legal move = empty square (0)
    legal_moves_mask[idx] = (board_states[idx] == 0);
}

// Compute child offsets for CSR representation
__global__ void compute_child_offsets_kernel(
    const int* __restrict__ children_per_node,
    int* __restrict__ row_ptr,
    const int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    // Exclusive prefix sum
    if (idx == 0) {
        row_ptr[0] = 0;
    }
    
    if (idx < num_nodes - 1) {
        row_ptr[idx + 1] = children_per_node[idx];
    }
    
    // Note: This requires a proper prefix sum implementation
    // For now, this is a simplified version
}

// ============================================================================
// Python Interface Functions
// ============================================================================

torch::Tensor find_expansion_nodes_cuda(
    torch::Tensor current_nodes,
    torch::Tensor children,
    torch::Tensor visit_counts,
    torch::Tensor valid_path_mask,
    int max_children
) {
    int wave_size = current_nodes.size(0);
    int num_nodes = visit_counts.size(0);
    
    auto expansion_nodes = torch::zeros({wave_size}, torch::dtype(torch::kInt32).device(current_nodes.device()));
    auto expansion_count = torch::zeros({1}, torch::dtype(torch::kInt32).device(current_nodes.device()));
    
    const int threads = 256;
    const int blocks = (wave_size + threads - 1) / threads;
    
    find_expansion_nodes_kernel<<<blocks, threads>>>(
        current_nodes.data_ptr<int>(),
        children.data_ptr<int>(),
        visit_counts.data_ptr<int>(),
        valid_path_mask.data_ptr<bool>(),
        expansion_nodes.data_ptr<int>(),
        expansion_count.data_ptr<int>(),
        wave_size,
        max_children,
        num_nodes
    );
    
    return expansion_nodes;
}

torch::Tensor batch_process_legal_moves_cuda(
    torch::Tensor raw_policies,
    torch::Tensor board_states,
    int action_size
) {
    int num_states = raw_policies.size(0);
    
    auto normalized_priors = torch::zeros_like(raw_policies);
    auto legal_move_indices = torch::zeros({num_states, action_size}, torch::dtype(torch::kInt32).device(raw_policies.device()));
    auto legal_move_counts = torch::zeros({num_states}, torch::dtype(torch::kInt32).device(raw_policies.device()));
    
    const int threads = 256;
    
    batch_process_legal_moves_kernel<<<num_states, threads>>>(
        raw_policies.data_ptr<float>(),
        board_states.data_ptr<int>(),
        normalized_priors.data_ptr<float>(),
        legal_move_indices.data_ptr<int>(),
        legal_move_counts.data_ptr<int>(),
        num_states,
        action_size
    );
    
    return normalized_priors;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("find_expansion_nodes", &find_expansion_nodes_cuda, "Find expansion nodes CUDA");
    m.def("batch_process_legal_moves", &batch_process_legal_moves_cuda, "Batch process legal moves CUDA");
}