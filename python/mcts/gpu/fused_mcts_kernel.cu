// Fused MCTS kernel for UCB selection, expansion check, and backup preparation
// This reduces kernel launch overhead by combining multiple operations

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// PHASE 2.1 OPTIMIZATION: Fused UCB-Expansion-Backup kernel
// Combines three operations into one kernel to reduce launch overhead
__global__ void fused_mcts_step_kernel(
    const int* __restrict__ root_nodes,          // Starting nodes for each simulation
    const int* __restrict__ children,            // Children lookup table
    const int* __restrict__ visit_counts,        // Visit counts
    const int* __restrict__ virtual_loss_counts, // Virtual losses
    const float* __restrict__ value_sums,        // Value sums  
    const float* __restrict__ priors,            // Node priors
    int* __restrict__ selected_paths,            // Output: selected paths
    int* __restrict__ path_lengths,              // Output: path lengths
    int* __restrict__ expansion_nodes,           // Output: nodes needing expansion
    int* __restrict__ expansion_count,           // Output: number of expansions
    float* __restrict__ path_values,             // Output: values for backup
    const int batch_size,
    const int max_children,
    const int max_depth,
    const float c_puct,
    const float virtual_loss_value
) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Shared memory for path storage
    extern __shared__ int shared_path[];
    int* local_path = &shared_path[threadIdx.x * max_depth];
    
    int current = root_nodes[batch_idx];
    int depth = 0;
    local_path[0] = current;
    
    // Tree traversal with fused operations
    bool found_leaf = false;
    
    while (depth < max_depth - 1 && !found_leaf) {
        const int child_base = current * max_children;
        
        // Get parent statistics
        int parent_visits = visit_counts[current];
        if (parent_visits == 0) {
            // Found unexpanded node
            found_leaf = true;
            break;
        }
        
        float sqrt_parent = sqrtf((float)(parent_visits + 1));
        
        // Find best child using UCB
        float best_ucb = -INFINITY;
        int best_child = -1;
        bool has_children = false;
        
        // Vectorized child evaluation
        #pragma unroll 8
        for (int i = 0; i < max_children; i++) {
            int child_idx = children[child_base + i];
            if (child_idx < 0) break;
            
            has_children = true;
            
            // Get child statistics with virtual loss
            int child_visits = visit_counts[child_idx] + virtual_loss_counts[child_idx];
            float child_value = value_sums[child_idx] + 
                               virtual_loss_counts[child_idx] * virtual_loss_value;
            
            // Calculate UCB score
            float q_value = (child_visits > 0) ? 
                           (child_value / child_visits) : 0.0f;
            float prior = priors[child_idx];
            float exploration = c_puct * prior * sqrt_parent / (1.0f + child_visits);
            float ucb = q_value + exploration;
            
            // Track best
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best_child = child_idx;
            }
        }
        
        // Check if we reached a leaf
        if (!has_children || best_child < 0) {
            found_leaf = true;
            break;
        }
        
        // Move to best child
        depth++;
        current = best_child;
        local_path[depth] = current;
        
        // Apply virtual loss atomically
        atomicAdd(&virtual_loss_counts[best_child], 1);
    }
    
    // Store results
    path_lengths[batch_idx] = depth + 1;
    
    // Copy path to global memory
    for (int i = 0; i <= depth; i++) {
        selected_paths[batch_idx * max_depth + i] = local_path[i];
    }
    
    // Fill rest with -1
    for (int i = depth + 1; i < max_depth; i++) {
        selected_paths[batch_idx * max_depth + i] = -1;
    }
    
    // Mark node for expansion if needed
    if (found_leaf && visit_counts[current] == 0) {
        int idx = atomicAdd(expansion_count, 1);
        expansion_nodes[idx] = current;
    }
}

// Optimized backup kernel with warp-level primitives
__global__ void fused_backup_kernel(
    const int* __restrict__ paths,
    const int* __restrict__ path_lengths,
    const float* __restrict__ values,
    const int* __restrict__ virtual_loss_nodes,  // Nodes with virtual loss
    int* __restrict__ visit_counts,
    float* __restrict__ value_sums,
    int* __restrict__ virtual_loss_counts,
    const int batch_size,
    const int max_depth,
    const int max_nodes
) {
    const int warp_size = 32;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    const int num_warps = (gridDim.x * blockDim.x) / warp_size;
    
    // Each warp processes multiple paths
    for (int path_idx = warp_id; path_idx < batch_size; path_idx += num_warps) {
        int path_length = path_lengths[path_idx];
        if (path_length == 0) continue;
        
        float path_value = values[path_idx];
        
        // Warp processes path in parallel
        for (int depth = lane_id; depth < path_length; depth += warp_size) {
            int node_idx = paths[path_idx * max_depth + depth];
            if (node_idx < 0 || node_idx >= max_nodes) continue;
            
            // Calculate backup value with correct sign
            int distance_from_leaf = path_length - 1 - depth;
            float sign = (distance_from_leaf % 2 == 0) ? 1.0f : -1.0f;
            float backup_value = path_value * sign;
            
            // Update node statistics
            atomicAdd(&visit_counts[node_idx], 1);
            atomicAdd(&value_sums[node_idx], backup_value);
            
            // Remove virtual loss
            atomicSub(&virtual_loss_counts[node_idx], 1);
        }
    }
}

// Python bindings
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_mcts_step_cuda(
    torch::Tensor root_nodes,
    torch::Tensor children,
    torch::Tensor visit_counts,
    torch::Tensor virtual_loss_counts,
    torch::Tensor value_sums,
    torch::Tensor priors,
    int max_depth,
    float c_puct,
    float virtual_loss_value
) {
    const int batch_size = root_nodes.size(0);
    const int max_children = children.size(1);
    const int max_nodes = visit_counts.size(0);
    
    // Allocate output tensors
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(root_nodes.device());
    auto selected_paths = torch::zeros({batch_size, max_depth}, options);
    auto path_lengths = torch::zeros({batch_size}, options);
    auto expansion_nodes = torch::zeros({batch_size}, options);
    auto expansion_count = torch::zeros({1}, options);
    auto path_values = torch::zeros({batch_size}, torch::kFloat32).to(root_nodes.device());
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    const int shared_mem = threads * max_depth * sizeof(int);
    
    fused_mcts_step_kernel<<<blocks, threads, shared_mem>>>(
        root_nodes.data_ptr<int>(),
        children.data_ptr<int>(),
        visit_counts.data_ptr<int>(),
        virtual_loss_counts.data_ptr<int>(),
        value_sums.data_ptr<float>(),
        priors.data_ptr<float>(),
        selected_paths.data_ptr<int>(),
        path_lengths.data_ptr<int>(),
        expansion_nodes.data_ptr<int>(),
        expansion_count.data_ptr<int>(),
        path_values.data_ptr<float>(),
        batch_size,
        max_children,
        max_depth,
        c_puct,
        virtual_loss_value
    );
    
    // Get actual expansion count
    int num_expansions = expansion_count.cpu().item<int>();
    expansion_nodes = expansion_nodes.slice(0, 0, num_expansions);
    
    return std::make_tuple(selected_paths, path_lengths, expansion_nodes);
}

void fused_backup_cuda(
    torch::Tensor paths,
    torch::Tensor path_lengths,
    torch::Tensor values,
    torch::Tensor visit_counts,
    torch::Tensor value_sums,
    torch::Tensor virtual_loss_counts
) {
    const int batch_size = paths.size(0);
    const int max_depth = paths.size(1);
    const int max_nodes = visit_counts.size(0);
    
    // Launch optimized backup kernel
    const int threads = 256;
    const int blocks = (batch_size * 32 + threads - 1) / threads;
    
    fused_backup_kernel<<<blocks, threads>>>(
        paths.data_ptr<int>(),
        path_lengths.data_ptr<int>(),
        values.data_ptr<float>(),
        nullptr,  // virtual_loss_nodes not used in this version
        visit_counts.data_ptr<int>(),
        value_sums.data_ptr<float>(),
        virtual_loss_counts.data_ptr<int>(),
        batch_size,
        max_depth,
        max_nodes
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_mcts_step", &fused_mcts_step_cuda, "Fused MCTS step (selection + expansion check)");
    m.def("fused_backup", &fused_backup_cuda, "Fused backup with virtual loss removal");
}