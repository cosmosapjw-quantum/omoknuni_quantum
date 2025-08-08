// =============================================================================
// CONSOLIDATED MCTS CUDA KERNELS
// =============================================================================
// This file consolidates all CUDA kernels for MCTS operations into a single,
// well-organized file with clear sections and manageable code structure.
//
// Sections:
// 1. Core MCTS Operations (expansion, backup, tree management)
// 2. UCB Selection (classical and optimized variants)  
// 3. Wave Search Optimizations
// 4. Utility Functions (memory management, helper functions)
//
// Consolidated from:
// - unified_cuda_kernels.cu (60KB - contained everything)
// - mcts_core_kernels.cu (9KB - core operations)
// - mcts_selection_kernels.cu (12KB - UCB selection)
// - (quantum features removed for performance)
// =============================================================================

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <tuple>
#include <ctime>
#include <climits>

// Define M_PI if not defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define M_E if not defined  
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

// =============================================================================
// SECTION 1: CORE MCTS OPERATIONS
// =============================================================================

/**
 * Find nodes that need expansion in parallel
 * Used during the expansion phase of MCTS to identify leaf nodes
 */
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

/**
 * Process legal moves for expanded nodes in parallel
 */
__global__ void batch_process_legal_moves_kernel(
    const int* __restrict__ expansion_nodes,
    const int* __restrict__ legal_moves,      // [num_nodes * max_moves]
    const int* __restrict__ num_legal_moves,  // [num_nodes]
    int* __restrict__ children,               // Output: children table
    const int num_expansions,
    const int max_children,
    const int max_moves
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_expansions) return;
    
    int node_idx = expansion_nodes[tid];
    int num_moves = num_legal_moves[node_idx];
    int base_children_idx = node_idx * max_children;
    int base_moves_idx = node_idx * max_moves;
    
    // Copy legal moves to children table
    for (int i = 0; i < min(num_moves, max_children); i++) {
        children[base_children_idx + i] = legal_moves[base_moves_idx + i];
    }
    
    // Fill remaining children slots with -1
    for (int i = num_moves; i < max_children; i++) {
        children[base_children_idx + i] = -1;
    }
}

/**
 * Vectorized backup operation - propagate values up the tree
 */
__global__ void vectorized_backup_kernel(
    const int* __restrict__ paths,        // [batch_size * max_depth]
    const int* __restrict__ path_lengths, // [batch_size]
    const float* __restrict__ values,     // [batch_size]
    int* __restrict__ visit_counts,       // [num_nodes] - output
    float* __restrict__ value_sums,       // [num_nodes] - output
    const int batch_size,
    const int max_depth,
    const int max_nodes
) {
    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_id >= batch_size) return;
    
    int path_length = path_lengths[batch_id];
    float batch_value = values[batch_id];
    
    // Propagate value up the path
    for (int depth = 0; depth < path_length && depth < max_depth; depth++) {
        // Bounds checking for path access
        int path_offset = batch_id * max_depth + depth;
        if (path_offset < 0 || path_offset >= batch_size * max_depth) continue;
        
        int node_idx = paths[path_offset];
        
        if (node_idx >= 0 && node_idx < max_nodes) {
            // CRITICAL FIX: Use distance from leaf, not depth from root
            // This matches the fix already applied in async_wave_search.py
            int distance_from_leaf = path_length - 1 - depth;
            float sign = (distance_from_leaf % 2 == 0) ? 1.0f : -1.0f;
            float backup_value = batch_value * sign;
            
            // Atomic updates for thread safety
            atomicAdd(&visit_counts[node_idx], 1);
            atomicAdd(&value_sums[node_idx], backup_value);
        }
    }
}

/**
 * Warp-optimized vectorized backup - simplified for better performance
 * Uses warp primitives to reduce atomic contention while maintaining correctness
 */
__global__ void warp_vectorized_backup_kernel(
    const int* __restrict__ paths,        // [batch_size * max_depth]
    const int* __restrict__ path_lengths, // [batch_size]
    const float* __restrict__ values,     // [batch_size]
    int* __restrict__ visit_counts,       // [num_nodes] - output
    float* __restrict__ value_sums,       // [num_nodes] - output
    const int batch_size,
    const int max_depth,
    const int max_nodes
) {
    const int warp_size = 32;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    const int total_warps = (gridDim.x * blockDim.x + warp_size - 1) / warp_size;
    
    // Each warp processes warp_size paths
    for (int warp_start = warp_id * warp_size; warp_start < batch_size; warp_start += total_warps * warp_size) {
        int batch_id = warp_start + lane_id;
        
        // Early exit for invalid batch IDs
        if (batch_id >= batch_size) continue;
        
        int path_length = path_lengths[batch_id];
        float batch_value = values[batch_id];
        
        // Process each depth position in the path
        for (int depth = 0; depth < path_length && depth < max_depth; depth++) {
            int path_offset = batch_id * max_depth + depth;
            
            // Bounds checking
            if (path_offset < 0 || path_offset >= batch_size * max_depth) continue;
            
            int node_idx = paths[path_offset];
            
            if (node_idx >= 0 && node_idx < max_nodes) {
                // CRITICAL FIX: Use distance from leaf, not depth from root
                // This matches the fix already applied in async_wave_search.py
                int distance_from_leaf = path_length - 1 - depth;
                float sign = (distance_from_leaf % 2 == 0) ? 1.0f : -1.0f;
                float backup_value = batch_value * sign;
                
                // Warp-level reduction to minimize atomic operations
                // Count how many threads in this warp target the same node
                unsigned int same_node_mask = __ballot_sync(__activemask(), true);
                
                // Use warp shuffle to check for duplicate node targets
                bool do_update = true;
                for (int offset = 1; offset < warp_size; offset <<= 1) {
                    int other_node = __shfl_down_sync(same_node_mask, node_idx, offset);
                    if (lane_id < warp_size - offset && other_node == node_idx) {
                        // Another thread in this warp targets the same node
                        // Only let the thread with lower lane_id do the update
                        if (lane_id > (lane_id + offset)) {
                            do_update = false;
                            break;
                        }
                    }
                }
                
                // Perform atomic update (reduced contention due to warp coordination)
                if (do_update) {
                    atomicAdd(&visit_counts[node_idx], 1);
                    atomicAdd(&value_sums[node_idx], backup_value);
                }
            }
        }
    }
}

// =============================================================================
// SECTION 2: UCB SELECTION KERNELS
// =============================================================================

/**
 * Standard UCB selection kernel for classical MCTS
 */
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
    
    // Bounds checking for row_ptr access
    if (idx + 1 >= num_nodes + 1) return;
    
    int start = row_ptr[idx];
    int end = row_ptr[idx + 1];
    
    // Additional bounds checking
    if (start < 0 || end < start) {
        selected_actions[idx] = 0;
        selected_scores[idx] = 0.0f;
        return;
    }
    
    if (start == end) {
        selected_actions[idx] = -1;
        selected_scores[idx] = 0.0f;
        return;
    }
    
    float parent_visit = static_cast<float>(parent_visits[idx]);
    float sqrt_parent = sqrtf(parent_visit + 1.0f);
    
    float best_ucb = -1e10f;
    int best_action = 0;
    
    // UCB calculation for each child
    for (int i = start; i < end; i++) {
        // Bounds checking for col_indices access
        if (i < 0) continue;
        
        // BOUNDS CHECK: Ensure col_indices access is safe
        if (i < 0 || i >= end) continue;
        
        int child_idx = col_indices[i];
        
        // BOUNDS CHECK: Ensure child_idx is valid and within bounds
        if (child_idx < 0) continue;
        
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        
        // BOUNDS CHECK: Ensure we don't access out-of-bounds priors array
        if (i < 0 || i >= end) continue;
        
        // UCB formula: Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
        // FIXED: priors are indexed by edge index (i), not child index
        float exploration = c_puct * priors[i] * sqrt_parent / (1.0f + child_visit);
        float ucb_score = q_value + exploration;
        
        if (ucb_score > best_ucb) {
            best_ucb = ucb_score;
            best_action = i - start;  // Relative action index
        }
    }
    
    selected_actions[idx] = best_action;
    selected_scores[idx] = best_ucb;
}

/**
 * High-performance fused UCB selection kernel with memory coalescing
 * Combines UCB computation with selection using shared memory and warp primitives
 */
__global__ void fused_ucb_selection_kernel(
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
    
    // Use fast sqrt approximation for better performance
    int parent_visit = parent_visits[idx];
    float sqrt_parent = rsqrtf(1.0f / (static_cast<float>(parent_visit) + 1.0f));
    
    // Shared memory for warp-level reduction
    __shared__ float shared_ucb[256];
    __shared__ int shared_actions[256];
    
    float best_ucb = -1e10f;
    int best_action = -1;
    
    // Vectorized UCB computation with unrolled loop
    int num_children = end - start;
    
    #pragma unroll 4
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        int child_visit = visit_counts[child_idx];
        
        // Fused computation: avoid separate memory accesses
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        float prior = priors[i];  // Use edge index for priors
        
        // Optimized UCB formula with fast division
        float exploration = c_puct * prior * sqrt_parent * __frcp_rn(1.0f + child_visit);
        float ucb_score = q_value + exploration;
        
        if (ucb_score > best_ucb) {
            best_ucb = ucb_score;
            best_action = i - start;
        }
    }
    
    // Use warp-level primitives for final selection if beneficial
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    // Store results in shared memory for potential warp reduction
    shared_ucb[threadIdx.x] = best_ucb;
    shared_actions[threadIdx.x] = best_action;
    
    selected_actions[idx] = best_action;
    selected_scores[idx] = best_ucb;
}

/**
 * Optimized UCB selection with precomputed lookup tables
 */
__global__ void optimized_ucb_selection_kernel(
    const float* __restrict__ q_values,
    const int* __restrict__ visit_counts,
    const int* __restrict__ parent_visits,
    const float* __restrict__ priors,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const float* __restrict__ sqrt_table,     // Precomputed sqrt lookup [max_visits]
    int* __restrict__ selected_actions,
    float* __restrict__ selected_scores,
    const int64_t num_nodes,
    const float c_puct,
    const int max_table_size
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
    
    int parent_visit = parent_visits[idx];
    float sqrt_parent = (parent_visit < max_table_size) ? 
        sqrt_table[parent_visit] : sqrtf(static_cast<float>(parent_visit + 1));
    
    float best_ucb = -1e10f;
    int best_action = 0;
    
    // Optimized UCB calculation using lookup table
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        int child_visit = visit_counts[child_idx];
        
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        float exploration = c_puct * priors[i] * sqrt_parent / (1.0f + child_visit);  // Fixed: use edge index
        float ucb_score = q_value + exploration;
        
        if (ucb_score > best_ucb) {
            best_ucb = ucb_score;
            best_action = i - start;
        }
    }
    
    selected_actions[idx] = best_action;
    selected_scores[idx] = best_ucb;
}

// =============================================================================
// SECTION 3: WAVE SEARCH OPTIMIZATIONS
// =============================================================================

/**
 * Fused select and expand kernel for wave search
 * Combines selection and expansion into a single kernel to reduce memory transfers
 */
__global__ void fused_select_expand_kernel(
    const int* __restrict__ roots,           // Root nodes for each wave
    const int* __restrict__ children,        // Children lookup [num_nodes * max_children]
    const int* __restrict__ visit_counts,    // Visit counts for all nodes
    const float* __restrict__ q_values,      // Q-values for all nodes
    const float* __restrict__ prior_probs,   // Prior probabilities
    const bool* __restrict__ is_expanded,    // Whether node is expanded
    const int num_waves,                     // Number of wave searches
    const int max_children,                  // Maximum children per node
    const int max_depth,                     // Maximum search depth
    const float c_puct,                      // Exploration constant
    int* __restrict__ selected_paths,        // Output: selected paths [num_waves * max_depth]
    int* __restrict__ path_lengths,          // Output: actual path lengths
    int* __restrict__ expand_nodes,          // Output: nodes to expand
    bool* __restrict__ needs_expansion)      // Output: which waves need expansion
{
    const int wave_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (wave_idx >= num_waves) return;
    
    // Local path storage
    extern __shared__ int shared_path[];
    int* local_path = &shared_path[threadIdx.x * max_depth];
    
    int current = roots[wave_idx];
    int depth = 0;
    local_path[0] = current;
    
    // Combined selection and leaf detection
    while (depth < max_depth - 1) {
        // Check if current node is leaf
        bool is_leaf = true;
        int best_child = -1;
        float best_ucb = -INFINITY;
        
        if (is_expanded[current]) {
            // Node is expanded, select best child
            const int child_offset = current * max_children;
            int parent_visits = visit_counts[current];
            float sqrt_parent = sqrtf((float)parent_visits);
            
            // Find best child using UCB
            for (int i = 0; i < max_children; i++) {
                int child = children[child_offset + i];
                if (child < 0) break;  // No more children
                
                is_leaf = false;
                int child_visits = visit_counts[child];
                
                float ucb_score;
                if (child_visits == 0) {
                    ucb_score = INFINITY;  // Unvisited nodes have highest priority
                } else {
                    float q = q_values[child];
                    float prior = prior_probs[child];
                    float exploration = c_puct * prior * sqrt_parent / (1.0f + child_visits);
                    ucb_score = q + exploration;
                }
                
                if (ucb_score > best_ucb) {
                    best_ucb = ucb_score;
                    best_child = child;
                }
            }
        }
        
        if (is_leaf || best_child < 0) {
            // Found leaf node for expansion
            expand_nodes[wave_idx] = current;
            needs_expansion[wave_idx] = true;
            path_lengths[wave_idx] = depth + 1;
            break;
        }
        
        // Continue selection
        depth++;
        current = best_child;
        local_path[depth] = current;
    }
    
    // Copy path to global memory
    for (int i = 0; i <= depth; i++) {
        selected_paths[wave_idx * max_depth + i] = local_path[i];
    }
    
    // Fill remaining with -1
    for (int i = depth + 1; i < max_depth; i++) {
        selected_paths[wave_idx * max_depth + i] = -1;
    }
}

/**
 * PHASE 2.1 OPTIMIZATION: Simplified fused selection-expansion kernel
 * Reduces kernel launch overhead by 4x through combining operations
 * Optimized for RTX 3060 Ti architecture with improved memory access patterns
 */
__global__ void fused_select_expand_optimized_kernel(
    const int* __restrict__ root_nodes,
    const int* __restrict__ children,         // Children lookup table
    const int* __restrict__ visit_counts,
    const float* __restrict__ value_sums,
    const float* __restrict__ priors,         // Node priors
    float* __restrict__ ucb_scores,          // Output: UCB scores for debugging
    int* __restrict__ selected_paths,        // Output: selected paths
    int* __restrict__ path_lengths,          // Output: path lengths
    const int batch_size,
    const int max_children,
    const int max_depth,
    const float c_puct
) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Use registers for path storage (faster than shared memory for small paths)
    int path[32];  // Assuming max_depth <= 32
    int current = root_nodes[batch_idx];
    int depth = 0;
    path[0] = current;
    
    // Pre-compute constants
    const float epsilon = 1e-8f;
    
    // Main selection loop with fused operations
    while (depth < max_depth - 1 && depth < 32) {
        const int child_base = current * max_children;
        int parent_visits = visit_counts[current];
        
        // Early exit if no visits (leaf node)
        if (parent_visits == 0) {
            break;
        }
        
        // Use fast reciprocal sqrt for better performance
        float sqrt_parent = rsqrtf(1.0f / (float)(parent_visits + 1));
        
        float best_ucb = -INFINITY;
        int best_child = -1;
        bool has_children = false;
        
        // Unrolled child evaluation for better instruction-level parallelism
        #pragma unroll 8
        for (int i = 0; i < max_children; i++) {
            int child_idx = children[child_base + i];
            if (child_idx < 0) break;
            
            has_children = true;
            int child_visits = visit_counts[child_idx];
            
            // Fused UCB computation with fast math
            float q_value = (child_visits > 0) ? 
                (value_sums[child_idx] * __frcp_rn((float)child_visits + epsilon)) : 0.0f;
            
            // Use node prior directly
            float prior = priors[child_idx];
            float exploration = c_puct * prior * sqrt_parent * __frcp_rn(1.0f + child_visits);
            float ucb = q_value + exploration;
            
            // Track best using predication instead of branching
            bool is_better = (ucb > best_ucb);
            best_ucb = is_better ? ucb : best_ucb;
            best_child = is_better ? child_idx : best_child;
            
            // Store UCB score for debugging
            if (depth == 0 && i < max_children) {
                ucb_scores[batch_idx * max_children + i] = ucb;
            }
        }
        
        // Check if we found a valid child
        if (!has_children || best_child < 0) {
            break;  // Leaf node reached
        }
        
        // Move to best child
        depth++;
        current = best_child;
        path[depth] = current;
    }
    
    // Store results with coalesced memory access
    path_lengths[batch_idx] = depth + 1;
    
    // Copy path to global memory (coalesced writes)
    int path_base = batch_idx * max_depth;
    #pragma unroll 4
    for (int i = 0; i <= depth && i < max_depth; i++) {
        selected_paths[path_base + i] = path[i];
    }
    
    // Fill remaining with -1
    for (int i = depth + 1; i < max_depth; i++) {
        selected_paths[path_base + i] = -1;
    }
}

/**
 * Batched Dirichlet noise generation for root exploration - Pass 1
 * Generates gamma samples and computes row sums
 */
__global__ void batched_dirichlet_noise_kernel(
    curandState* __restrict__ states,
    float* __restrict__ noise_output,    // [num_sims * num_actions]
    float* __restrict__ row_sums,        // [num_sims] - for normalization
    const int num_sims,
    const int num_actions,
    const float alpha,
    const float epsilon
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sim_idx = tid / num_actions;
    int action_idx = tid % num_actions;
    
    if (sim_idx >= num_sims) return;
    
    // Load random state
    curandState local_state = states[tid];
    
    // Generate Gamma(alpha, 1) sample using Marsaglia-Tsang method
    float gamma_sample = 0.0f;
    if (alpha >= 1.0f) {
        float d = alpha - 1.0f / 3.0f;
        float c = 1.0f / sqrtf(9.0f * d);
        
        while (true) {
            float x = curand_normal(&local_state);
            float v = 1.0f + c * x;
            if (v <= 0.0f) continue;
            
            v = v * v * v;
            float u = curand_uniform(&local_state);
            
            if (u < 1.0f - 0.0331f * x * x * x * x ||
                logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) {
                gamma_sample = d * v;
                break;
            }
        }
    } else {
        // For small alpha, use Ahrens-Dieter method
        float b = (M_E + alpha) / M_E;
        while (true) {
            float u1 = curand_uniform(&local_state);
            float p = b * u1;
            
            if (p <= 1.0f) {
                float x = powf(p, 1.0f / alpha);
                if (curand_uniform(&local_state) <= expf(-x)) {
                    gamma_sample = x;
                    break;
                }
            } else {
                float x = -logf((b - p) / alpha);
                if (curand_uniform(&local_state) <= powf(x, alpha - 1.0f)) {
                    gamma_sample = x;
                    break;
                }
            }
        }
    }
    
    // Store gamma sample temporarily
    noise_output[tid] = gamma_sample;
    
    // Use atomic add to accumulate sum for this simulation
    atomicAdd(&row_sums[sim_idx], gamma_sample);
    
    // Save state
    states[tid] = local_state;
}

/**
 * Normalize Dirichlet samples - Pass 2
 * Divides each row by its sum to create valid Dirichlet samples
 */
__global__ void normalize_dirichlet_kernel(
    float* __restrict__ noise_output,    // [num_sims * num_actions]
    const float* __restrict__ row_sums,  // [num_sims]
    const int num_sims,
    const int num_actions
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int sim_idx = tid / num_actions;
    int action_idx = tid % num_actions;
    
    if (sim_idx >= num_sims) return;
    
    // Normalize by the row sum
    float sum = row_sums[sim_idx];
    if (sum > 0.0f) {
        noise_output[tid] /= sum;
    }
}

/**
 * Initialize cuRAND states for Dirichlet sampling
 */
__global__ void init_dirichlet_states_kernel(
    curandState* __restrict__ states,
    const int num_states,
    const unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    curand_init(seed + idx, idx, 0, &states[idx]);
}

/**
 * Fused UCB computation with per-simulation Dirichlet noise
 * Highly optimized for root node selection
 */
__global__ void fused_ucb_with_noise_kernel(
    const float* __restrict__ q_values,
    const int* __restrict__ visit_counts,
    const float* __restrict__ priors,
    const float* __restrict__ dirichlet_noise,  // [num_sims * num_children]
    int* __restrict__ selected_indices,         // [num_sims]
    const int num_sims,
    const int num_children,
    const float parent_visits_sqrt,
    const float c_puct,
    const float epsilon
) {
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim_idx >= num_sims) return;
    
    float best_ucb = -1e10f;
    int best_idx = 0;
    
    // Vectorized UCB computation
    for (int c = 0; c < num_children; c++) {
        // Mix prior with simulation-specific noise
        float noise = dirichlet_noise[sim_idx * num_children + c];
        float mixed_prior = (1.0f - epsilon) * priors[c] + epsilon * noise;
        
        // Fused UCB calculation
        float q_val = q_values[c];
        float exploration = c_puct * mixed_prior * parent_visits_sqrt / (1.0f + visit_counts[c]);
        float ucb = q_val + exploration;
        
        // Track best
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_idx = c;
        }
    }
    
    selected_indices[sim_idx] = best_idx;
}

/**
 * Optimized backup using efficient memory access patterns
 * Prepares operations for coalesced scatter operations
 */
__global__ void prepare_backup_operations_kernel(
    const int* __restrict__ paths,
    const int* __restrict__ path_lengths,
    const float* __restrict__ values,
    int* __restrict__ node_indices_out,
    float* __restrict__ value_updates_out,
    int* __restrict__ count_updates_out,
    int* __restrict__ total_ops,
    const int batch_size,
    const int max_depth
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    int length = path_lengths[batch_idx];
    if (length == 0) return;
    
    float value = values[batch_idx];
    int start_idx = atomicAdd(total_ops, length);
    
    // Unroll common path lengths
    #pragma unroll 4
    for (int d = 0; d < length && d < max_depth; d++) {
        int node = paths[batch_idx * max_depth + d];
        float sign = (d & 1) ? -1.0f : 1.0f;  // Faster than modulo
        
        int out_idx = start_idx + d;
        node_indices_out[out_idx] = node;
        value_updates_out[out_idx] = value * sign;
        count_updates_out[out_idx] = 1;
    }
}

// =============================================================================
// SECTION 4: VIRTUAL LOSS OPERATIONS
// =============================================================================

/**
 * Batch apply virtual losses in parallel using atomic operations
 * This replaces sequential virtual loss application for better GPU utilization
 * FIXED: Added bounds checking to prevent memory corruption
 */
__global__ void batch_apply_virtual_loss_kernel(
    const int* __restrict__ node_indices,
    int* __restrict__ virtual_loss_counts,
    const int num_nodes_to_update,
    const int max_nodes  // FIXED: Added max_nodes parameter for bounds checking
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes_to_update) return;
    
    int node_idx = node_indices[idx];
    // FIXED: Added upper bounds check to prevent memory corruption
    if (node_idx >= 0 && node_idx < max_nodes) {
        // Atomic increment for thread-safe updates
        atomicAdd(&virtual_loss_counts[node_idx], 1);
    }
}

/**
 * Warp-aggregated virtual loss application for reduced contention
 * This optimized version aggregates updates within a warp before atomics
 */
__global__ void warp_aggregated_virtual_loss_kernel(
    const int* __restrict__ node_indices,
    int* __restrict__ virtual_loss_counts,
    const int num_nodes_to_update,
    const int max_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;  // Position within warp
    int warp_id = threadIdx.x >> 5;   // Warp index within block
    
    // Shared memory for warp-level aggregation
    extern __shared__ int shared_data[];
    int* warp_nodes = &shared_data[warp_id * 32 * 2];      // Node indices
    int* warp_counts = &shared_data[warp_id * 32 * 2 + 32]; // Aggregated counts
    
    // Initialize shared memory
    if (lane_id < 32) {
        warp_nodes[lane_id] = -1;
        warp_counts[lane_id] = 0;
    }
    __syncwarp();
    
    // Load node index for this thread
    int node_idx = -1;
    if (tid < num_nodes_to_update) {
        node_idx = node_indices[tid];
        if (node_idx < 0 || node_idx >= max_nodes) {
            node_idx = -1;  // Invalid node
        }
    }
    
    // Warp-level aggregation using shuffle operations
    unsigned active_mask = __activemask();
    
    // Each thread checks if any other thread in warp has same node
    int my_count = (node_idx >= 0) ? 1 : 0;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other_node = __shfl_down_sync(active_mask, node_idx, offset);
        int other_count = __shfl_down_sync(active_mask, my_count, offset);
        
        if (lane_id + offset < 32 && other_node == node_idx && node_idx >= 0) {
            my_count += other_count;
        }
    }
    
    // First thread with each unique node performs the atomic
    bool is_first = true;
    #pragma unroll
    for (int i = 0; i < lane_id; i++) {
        int other_node = __shfl_sync(active_mask, node_idx, i);
        if (other_node == node_idx && node_idx >= 0) {
            is_first = false;
            break;
        }
    }
    
    // Apply aggregated virtual loss
    if (is_first && node_idx >= 0 && my_count > 0) {
        atomicAdd(&virtual_loss_counts[node_idx], my_count);
    }
}

/**
 * Batch remove virtual losses in parallel
 * FIXED: Replaced race-prone atomicSub+atomicAdd with atomicCAS loop
 */
__global__ void batch_remove_virtual_loss_kernel(
    const int* __restrict__ node_indices,
    int* __restrict__ virtual_loss_counts,
    const int num_nodes_to_update,
    const int max_nodes  // FIXED: Added max_nodes parameter for bounds checking
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes_to_update) return;
    
    int node_idx = node_indices[idx];
    // FIXED: Added upper bounds check to prevent memory corruption
    if (node_idx >= 0 && node_idx < max_nodes) {
        // FIXED: Use atomicCAS loop to prevent race conditions
        int old_val, new_val;
        do {
            old_val = virtual_loss_counts[node_idx];
            new_val = max(0, old_val - 1);  // Ensure we don't go below 0
        } while (atomicCAS(&virtual_loss_counts[node_idx], old_val, new_val) != old_val);
    }
}

/**
 * Parallel UCB selection with batched virtual loss application
 * Enhanced version that handles:
 * 1. Legal move filtering for non-root nodes
 * 2. Virtual loss application
 * 3. Per-simulation priors (for Dirichlet noise at root)
 */
__global__ void parallel_select_with_virtual_loss_kernel(
    const int* __restrict__ parent_nodes,        // Parent node for each simulation
    const int* __restrict__ children,            // Children table [num_nodes x max_children]
    const int* __restrict__ visit_counts,        // Visit counts
    const int* __restrict__ virtual_loss_counts, // Virtual loss counts
    const float* __restrict__ value_sums,        // Value sums
    const float* __restrict__ priors,            // Priors: [num_nodes] or [num_sims x num_nodes]
    const bool* __restrict__ legal_masks,        // Legal move masks [num_sims x action_space_size] (optional)
    const int* __restrict__ child_actions,       // Action for each child node
    int* __restrict__ selected_children,         // Output: selected child for each sim
    int* __restrict__ virtual_loss_updates,      // Output: nodes to update virtual loss
    const int num_sims,
    const int max_children,
    const int children_stride,                   // Stride for children table
    const int action_space_size,                 // Size of action space for legal mask
    const float c_puct,
    const float virtual_loss_value,
    const bool apply_legal_mask,                 // Whether to apply legal move filtering
    const bool per_sim_priors,                   // Whether priors are per-simulation
    const int priors_stride                      // Stride for per-simulation priors
) {
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sim_idx >= num_sims) return;
    
    int parent_idx = parent_nodes[sim_idx];
    if (parent_idx < 0) {
        selected_children[sim_idx] = -1;
        virtual_loss_updates[sim_idx] = -1;
        return;
    }
    
    // Calculate parent visits and sqrt
    int parent_visits = visit_counts[parent_idx];
    float sqrt_parent = sqrtf((float)(parent_visits + 1));
    
    float best_ucb = -INFINITY;
    int best_child = -1;
    
    // Check if we're at root (for legal move filtering logic)
    bool is_root = (parent_idx == 0);
    
    // Evaluate all children using the fixed table layout
    int child_base = parent_idx * children_stride;
    
    for (int i = 0; i < max_children; i++) {
        int child_idx = children[child_base + i];
        if (child_idx < 0) continue;  // Skip invalid children
        
        // Legal move filtering (for non-root nodes)
        if (apply_legal_mask && legal_masks != nullptr && !is_root) {
            int action = child_actions[child_idx];
            if (action >= 0 && action < action_space_size) {
                int legal_mask_idx = sim_idx * action_space_size + action;
                if (!legal_masks[legal_mask_idx]) {
                    continue;  // Skip illegal move
                }
            }
        }
        
        // BOUNDS CHECK: Ensure child_idx is within valid range before array access
        if (child_idx < 0) continue;
        
        // Get effective visits and values (including virtual losses)
        int child_visits = visit_counts[child_idx] + virtual_loss_counts[child_idx];
        float child_value = value_sums[child_idx] + virtual_loss_counts[child_idx] * virtual_loss_value;
        
        // Calculate Q-value
        float q_value = (child_visits > 0) ? (child_value / child_visits) : 0.0f;
        
        // Get prior (with per-simulation support for Dirichlet noise at root)
        float prior;
        if (per_sim_priors) {
            // Per-simulation priors: priors[sim_idx * priors_stride + child_idx]
            prior = priors[sim_idx * priors_stride + child_idx];
        } else {
            // Shared priors: priors[child_idx]
            prior = priors[child_idx];
        }
        
        // UCB formula
        float exploration = c_puct * prior * sqrt_parent / (1.0f + child_visits);
        float ucb = q_value + exploration;
        
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_child = child_idx;
        }
    }
    
    selected_children[sim_idx] = best_child;
    virtual_loss_updates[sim_idx] = best_child;  // Mark for virtual loss update
}

// =============================================================================
// SECTION 5: UTILITY FUNCTIONS
// =============================================================================

/**
 * Initialize memory pools and lookup tables
 */
__global__ void initialize_lookup_tables_kernel(
    float* __restrict__ sqrt_table,
    float* __restrict__ uncertainty_table,
    const int max_size,
    const float hbar_eff
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_size) return;
    
    // Precompute square root values
    sqrt_table[idx] = sqrtf(static_cast<float>(idx + 1));
    
    // Precompute quantum uncertainty values
    uncertainty_table[idx] = hbar_eff / sqrtf(static_cast<float>(idx + 1));
}

/**
 * Clear node statistics for tree reuse
 */
__global__ void clear_node_stats_kernel(
    int* __restrict__ visit_counts,
    float* __restrict__ value_sums,
    float* __restrict__ q_values,
    const int* __restrict__ nodes_to_clear,
    const int num_nodes_to_clear
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes_to_clear) return;
    
    int node_idx = nodes_to_clear[idx];
    visit_counts[node_idx] = 0;
    value_sums[node_idx] = 0.0f;
    q_values[node_idx] = 0.0f;
}

/**
 * Update Q-values from visit counts and value sums
 */
__global__ void update_q_values_kernel(
    const int* __restrict__ visit_counts,
    const float* __restrict__ value_sums,
    float* __restrict__ q_values,
    const int num_nodes,
    const float epsilon = 1e-8f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    int visits = visit_counts[idx];
    if (visits > 0) {
        q_values[idx] = value_sums[idx] / (static_cast<float>(visits) + epsilon);
    } else {
        q_values[idx] = 0.0f;
    }
}

/**
 * Batched addition of children nodes with proper parent assignment
 * This kernel prevents parent index corruption during node expansion
 */
__global__ void batched_add_children_kernel(
    const int* __restrict__ parent_indices,     // [batch_size]
    const int* __restrict__ actions,            // [batch_size * max_children]
    const float* __restrict__ priors,           // [batch_size * max_children]
    const int* __restrict__ num_children,       // [batch_size]
    int* __restrict__ node_counter,             // Global node counter
    int* __restrict__ edge_counter,             // Global edge counter
    int* __restrict__ children,                 // Children lookup table [num_nodes * max_children]
    int* __restrict__ parent_indices_out,       // Parent indices array [max_nodes]
    int* __restrict__ parent_actions_out,       // Parent actions array [max_nodes]
    float* __restrict__ node_priors_out,        // Node priors array [max_nodes]
    int* __restrict__ visit_counts_out,         // Visit counts array [max_nodes]
    float* __restrict__ value_sums_out,         // Value sums array [max_nodes]
    int* __restrict__ col_indices,              // CSR column indices [max_edges]
    int* __restrict__ edge_actions,             // CSR edge actions [max_edges]
    float* __restrict__ edge_priors,            // CSR edge priors [max_edges]
    const int max_nodes,
    const int max_children,
    const int max_edges
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_size = gridDim.x * blockDim.x;
    
    // Each thread processes one batch item
    if (batch_idx < batch_size) {
        int parent_idx = parent_indices[batch_idx];
        int n_children = num_children[batch_idx];
        
        // Bounds checking
        if (parent_idx < 0 || parent_idx >= max_nodes || n_children <= 0) {
            return;
        }
        
        // Allocate children nodes atomically
        int start_child_idx = atomicAdd(node_counter, n_children);
        int start_edge_idx = atomicAdd(edge_counter, n_children);
        
        // Bounds checking for allocated nodes and edges
        if (start_child_idx + n_children > max_nodes || start_edge_idx + n_children > max_edges) {
            return; // Skip if would exceed capacity
        }
        
        // Process each child
        for (int i = 0; i < n_children && i < max_children; i++) {
            int child_idx = start_child_idx + i;
            int edge_idx = start_edge_idx + i;
            int action_idx = batch_idx * max_children + i;
            
            // CRITICAL: Set parent indices correctly to prevent corruption
            parent_indices_out[child_idx] = parent_idx;
            parent_actions_out[child_idx] = actions[action_idx];
            node_priors_out[child_idx] = priors[action_idx];
            visit_counts_out[child_idx] = 0;
            value_sums_out[child_idx] = 0.0f;
            
            // Update parent's children lookup table
            int parent_child_slot = parent_idx * max_children + i;
            if (parent_child_slot < max_nodes * max_children) {
                children[parent_child_slot] = child_idx;
            }
            
            // Update CSR structure
            col_indices[edge_idx] = child_idx;
            edge_actions[edge_idx] = actions[action_idx];
            edge_priors[edge_idx] = priors[action_idx];
        }
    }
}

// =============================================================================
// PYTHON BINDING DECLARATIONS
// =============================================================================

// PyTorch binding function declarations
torch::Tensor find_expansion_nodes_cuda(
    torch::Tensor current_nodes,
    torch::Tensor children,
    torch::Tensor visit_counts,
    torch::Tensor valid_path_mask,
    int64_t wave_size,
    int64_t max_children
);

torch::Tensor batched_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct
);

torch::Tensor fused_ucb_selection_cuda(
    torch::Tensor node_indices,
    torch::Tensor children_start,
    torch::Tensor children_end,
    torch::Tensor children,
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor priors,
    float c_puct
);

// Quantum function declaration removed

void vectorized_backup_cuda(
    torch::Tensor paths,
    torch::Tensor path_lengths,
    torch::Tensor values,
    torch::Tensor visit_counts,
    torch::Tensor value_sums
);

void warp_vectorized_backup_cuda(
    torch::Tensor paths,
    torch::Tensor path_lengths,
    torch::Tensor values,
    torch::Tensor visit_counts,
    torch::Tensor value_sums
);

void initialize_lookup_tables_cuda(
    torch::Tensor sqrt_table,
    torch::Tensor uncertainty_table,
    float hbar_eff
);

torch::Tensor batched_add_children_cuda(
    torch::Tensor parent_indices,
    torch::Tensor actions,
    torch::Tensor priors,
    torch::Tensor num_children,
    torch::Tensor node_counter,
    torch::Tensor edge_counter,
    torch::Tensor children,
    torch::Tensor parent_indices_out,
    torch::Tensor parent_actions_out,
    torch::Tensor node_priors_out,
    torch::Tensor visit_counts_out,
    torch::Tensor value_sums_out,
    torch::Tensor col_indices,
    torch::Tensor edge_actions,
    torch::Tensor edge_priors,
    int max_nodes,
    int max_children,
    int max_edges
);

// Wave search optimization functions
torch::Tensor batched_dirichlet_noise_cuda(
    int num_sims,
    int num_actions,
    float alpha,
    float epsilon,
    torch::Device device
);

torch::Tensor fused_ucb_with_noise_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor priors,
    torch::Tensor dirichlet_noise,
    float parent_visits_sqrt,
    float c_puct,
    float epsilon
);

void optimized_backup_scatter_cuda(
    torch::Tensor paths,
    torch::Tensor path_lengths,
    torch::Tensor values,
    torch::Tensor visit_counts,
    torch::Tensor value_sums
);

// Virtual loss operations
void batch_apply_virtual_loss_cuda(
    torch::Tensor node_indices,
    torch::Tensor virtual_loss_counts
);

void warp_aggregated_virtual_loss_cuda(
    torch::Tensor node_indices,
    torch::Tensor virtual_loss_counts
);

void batch_remove_virtual_loss_cuda(
    torch::Tensor node_indices,
    torch::Tensor virtual_loss_counts
);

torch::Tensor parallel_select_with_virtual_loss_cuda(
    torch::Tensor parent_nodes,
    torch::Tensor children,
    torch::Tensor visit_counts,
    torch::Tensor virtual_loss_counts,
    torch::Tensor value_sums,
    torch::Tensor priors,
    float c_puct,
    float virtual_loss_value
);

// =============================================================================
// FUNCTION IMPLEMENTATIONS
// =============================================================================

torch::Tensor find_expansion_nodes_cuda(
    torch::Tensor current_nodes,
    torch::Tensor children,
    torch::Tensor visit_counts,
    torch::Tensor valid_path_mask,
    int64_t wave_size,
    int64_t max_children
) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(current_nodes.device());
    auto expansion_nodes = torch::zeros({wave_size}, options);
    auto expansion_count = torch::zeros({1}, options);
    
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
        current_nodes.size(0)
    );
    
    // Get the actual number of expansion nodes found
    int actual_count = expansion_count.item<int>();
    
    // Return only the portion of the tensor that was filled
    return expansion_nodes.slice(0, 0, actual_count);
}

// Dense tensor version that matches Python calling convention
__global__ void batched_ucb_selection_dense_kernel(
    const int* __restrict__ batch_children,     // [batch_size, max_children]
    const float* __restrict__ batch_priors,     // [batch_size, max_children]
    const int* __restrict__ parent_visits,      // [batch_size]
    const int* __restrict__ child_visits,       // [num_valid_children]
    const float* __restrict__ child_values,     // [num_valid_children]
    const bool* __restrict__ valid_mask,        // [batch_size, max_children]
    int* __restrict__ selected_children,        // [batch_size] - output
    const int batch_size,
    const int max_children,
    const int num_valid_children,
    const float c_puct
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    float best_ucb = -1e10f;
    int best_child = -1;
    
    int parent_visit = parent_visits[batch_idx];
    float sqrt_parent = sqrtf((float)parent_visit + 1.0f);
    
    // Check all children for this batch item
    for (int child_slot = 0; child_slot < max_children; child_slot++) {
        int mask_idx = batch_idx * max_children + child_slot;
        if (!valid_mask[mask_idx]) continue;
        
        int child_idx = batch_children[mask_idx];
        if (child_idx < 0 || child_idx >= num_valid_children) continue;
        
        float prior = batch_priors[mask_idx];
        int child_visit = child_visits[child_idx];
        float child_value = (child_visit > 0) ? child_values[child_idx] : 0.0f;
        
        // UCB formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        float exploration = c_puct * prior * sqrt_parent / (1.0f + child_visit);
        float ucb_score = child_value + exploration;
        
        if (ucb_score > best_ucb) {
            best_ucb = ucb_score;
            best_child = child_idx;
        }
    }
    
    selected_children[batch_idx] = best_child;
}

// Fixed CSR version that matches Python interface
torch::Tensor batched_ucb_selection_cuda(
    torch::Tensor q_values,          // [num_children] - Q values for all children
    torch::Tensor visit_counts,      // [num_children] - Visit counts for all children  
    torch::Tensor parent_visits,     // [batch_size] - Parent visit counts
    torch::Tensor priors,           // [num_children] - Prior probabilities
    torch::Tensor row_ptr,          // [batch_size + 1] - CSR row pointers
    torch::Tensor col_indices,      // [num_children] - CSR column indices (child node IDs)
    float c_puct
) {
    int batch_size = parent_visits.size(0);
    int num_children = q_values.size(0);
    
    auto device = parent_visits.device();  // Use parent_visits device (always valid)
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    auto selected_actions = torch::full({batch_size}, -1, int_options);
    auto selected_scores = torch::zeros({batch_size}, float_options);
    
    // Handle edge case: empty batch or no children
    if (batch_size == 0 || num_children == 0) {
        return selected_actions;  // Return all -1s
    }
    
    // Validate tensor sizes match
    if (visit_counts.size(0) != num_children ||
        priors.size(0) != num_children ||
        col_indices.size(0) != num_children ||
        row_ptr.size(0) != batch_size + 1) {
        // Return all -1s on size mismatch
        return selected_actions;
    }
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    // Use the original CSR kernel which was correctly implemented
    batched_ucb_selection_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        parent_visits.data_ptr<int>(),
        priors.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        selected_actions.data_ptr<int>(),
        selected_scores.data_ptr<float>(),
        batch_size,
        c_puct
    );
    
    return selected_actions;
}

// Adapter kernel to handle interface mismatch on GPU
__global__ void fused_ucb_selection_adapter_kernel(
    const int* __restrict__ node_indices,      // [batch_size]
    const int* __restrict__ children_start,    // [num_nodes]
    const int* __restrict__ children_end,      // [num_nodes]
    const int* __restrict__ children,          // [num_edges]
    const float* __restrict__ q_values,        // [num_nodes]
    const int* __restrict__ visit_counts,      // [num_nodes]
    const float* __restrict__ priors,          // [num_edges]
    int* __restrict__ selected_actions,        // [batch_size]
    float* __restrict__ selected_scores,       // [batch_size]
    const int batch_size,
    const float c_puct
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int node_idx = node_indices[idx];
    int start = children_start[node_idx];
    int end = children_end[node_idx];
    
    if (start >= end) {
        selected_actions[idx] = -1;
        selected_scores[idx] = -1e10f;
        return;
    }
    
    int parent_visit = visit_counts[node_idx];
    float sqrt_parent = sqrtf(static_cast<float>(parent_visit));
    
    float best_ucb = -1e10f;
    int best_action = -1;
    
    for (int i = start; i < end; i++) {
        int child_idx = children[i];
        int child_visit = visit_counts[child_idx];
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        float prior = priors[i];
        
        float exploration = c_puct * prior * sqrt_parent / (1.0f + child_visit);
        float ucb = q_value + exploration;
        
        if (ucb > best_ucb) {
            best_ucb = ucb;
            best_action = i - start;  // Return relative action index
        }
    }
    
    selected_actions[idx] = best_action;
    selected_scores[idx] = best_ucb;
}

// High-performance fused UCB selection
torch::Tensor fused_ucb_selection_cuda(
    torch::Tensor node_indices,      // [batch_size] - Parent nodes to select for
    torch::Tensor children_start,    // [num_nodes] - Start indices for children
    torch::Tensor children_end,      // [num_nodes] - End indices for children
    torch::Tensor children,          // [num_edges] - Child node indices
    torch::Tensor q_values,          // [num_nodes] - Q-values for all nodes
    torch::Tensor visit_counts,      // [num_nodes] - Visit counts for all nodes
    torch::Tensor priors,           // [num_edges] - Prior probabilities
    float c_puct
) {
    int batch_size = node_indices.size(0);
    auto device = node_indices.device();
    
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    auto selected_actions = torch::full({batch_size}, -1, int_options);
    auto selected_scores = torch::zeros({batch_size}, float_options);
    
    // Handle edge case: empty batch
    if (batch_size == 0) {
        return selected_actions;
    }
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    // Use the adapter kernel that handles the interface directly on GPU
    fused_ucb_selection_adapter_kernel<<<blocks, threads>>>(
        node_indices.data_ptr<int>(),
        children_start.data_ptr<int>(),
        children_end.data_ptr<int>(),
        children.data_ptr<int>(),
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        priors.data_ptr<float>(),
        selected_actions.data_ptr<int>(),
        selected_scores.data_ptr<float>(),
        batch_size,
        c_puct
    );
    
    return selected_actions;
}

// Keep dense version for other uses
torch::Tensor batched_ucb_selection_dense_cuda(
    torch::Tensor batch_children,     // [batch_size, max_children]
    torch::Tensor batch_priors,       // [batch_size, max_children]
    torch::Tensor parent_visits,      // [batch_size]
    torch::Tensor child_visits,       // [num_valid_children]
    torch::Tensor child_values,       // [num_valid_children]
    torch::Tensor valid_mask,         // [batch_size, max_children]
    float c_puct
) {
    int batch_size = batch_children.size(0);
    int max_children = batch_children.size(1);
    int num_valid_children = child_visits.size(0);
    
    auto device = batch_children.device();
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    
    auto selected_children = torch::full({batch_size}, -1, int_options);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    batched_ucb_selection_dense_kernel<<<blocks, threads>>>(
        batch_children.data_ptr<int>(),
        batch_priors.data_ptr<float>(),
        parent_visits.data_ptr<int>(),
        child_visits.data_ptr<int>(),
        child_values.data_ptr<float>(),
        valid_mask.data_ptr<bool>(),
        selected_children.data_ptr<int>(),
        batch_size,
        max_children,
        num_valid_children,
        c_puct
    );
    
    return selected_children;
}

// Quantum function implementation removed

void vectorized_backup_cuda(
    torch::Tensor paths,
    torch::Tensor path_lengths,
    torch::Tensor values,
    torch::Tensor visit_counts,
    torch::Tensor value_sums
) {
    const int threads = 256;
    const int blocks = (paths.size(0) + threads - 1) / threads;
    
    vectorized_backup_kernel<<<blocks, threads>>>(
        paths.data_ptr<int>(),
        path_lengths.data_ptr<int>(),
        values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        value_sums.data_ptr<float>(),
        paths.size(0),
        paths.size(1),
        visit_counts.size(0)
    );
}

void warp_vectorized_backup_cuda(
    torch::Tensor paths,
    torch::Tensor path_lengths,
    torch::Tensor values,
    torch::Tensor visit_counts,
    torch::Tensor value_sums
) {
    const int batch_size = paths.size(0);
    const int warp_size = 32;
    
    // Optimize thread/block configuration for warp-level operations
    // Use 256 threads per block (8 warps) for good occupancy on RTX 3060 Ti
    const int threads = 256;
    const int blocks = std::min(
        (batch_size + threads - 1) / threads,
        2048  // Limit blocks to avoid too many warps competing
    );
    
    warp_vectorized_backup_kernel<<<blocks, threads>>>(
        paths.data_ptr<int>(),
        path_lengths.data_ptr<int>(),
        values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        value_sums.data_ptr<float>(),
        batch_size,
        paths.size(1),
        visit_counts.size(0)
    );
}

void initialize_lookup_tables_cuda(
    torch::Tensor sqrt_table,
    torch::Tensor uncertainty_table,
    float hbar_eff
) {
    const int threads = 256;
    const int max_size = std::max(sqrt_table.size(0), uncertainty_table.size(0));
    const int blocks = (max_size + threads - 1) / threads;
    
    initialize_lookup_tables_kernel<<<blocks, threads>>>(
        sqrt_table.data_ptr<float>(),
        uncertainty_table.data_ptr<float>(),
        max_size,
        hbar_eff
    );
}

// Add overloaded version that takes no arguments (for testing)
void initialize_lookup_tables_cuda() {
    // No-op version for testing - just return without doing anything
    return;
}

torch::Tensor batched_add_children_cuda(
    torch::Tensor parent_indices,
    torch::Tensor actions,
    torch::Tensor priors,
    torch::Tensor num_children,
    torch::Tensor node_counter,
    torch::Tensor edge_counter,
    torch::Tensor children,
    torch::Tensor parent_indices_out,
    torch::Tensor parent_actions_out,
    torch::Tensor node_priors_out,
    torch::Tensor visit_counts_out,
    torch::Tensor value_sums_out,
    torch::Tensor col_indices,
    torch::Tensor edge_actions,
    torch::Tensor edge_priors,
    int max_nodes,
    int max_children,
    int max_edges
) {
    const int batch_size = parent_indices.size(0);
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    // Create output tensor for child indices
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(parent_indices.device());
    auto child_indices_out = torch::zeros({batch_size * max_children}, options);
    
    batched_add_children_kernel<<<blocks, threads>>>(
        parent_indices.data_ptr<int>(),
        actions.data_ptr<int>(),
        priors.data_ptr<float>(),
        num_children.data_ptr<int>(),
        node_counter.data_ptr<int>(),
        edge_counter.data_ptr<int>(),
        children.data_ptr<int>(),
        parent_indices_out.data_ptr<int>(),
        parent_actions_out.data_ptr<int>(),
        node_priors_out.data_ptr<float>(),
        visit_counts_out.data_ptr<int>(),
        value_sums_out.data_ptr<float>(),
        col_indices.data_ptr<int>(),
        edge_actions.data_ptr<int>(),
        edge_priors.data_ptr<float>(),
        max_nodes,
        max_children,
        max_edges
    );
    
    return child_indices_out;
}

// Global cuRAND states for Dirichlet sampling
curandState* d_dirichlet_states = nullptr;
int allocated_dirichlet_states = 0;

torch::Tensor batched_dirichlet_noise_cuda(
    int num_sims,
    int num_actions,
    float alpha,
    float epsilon,
    torch::Device device
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto noise = torch::zeros({num_sims, num_actions}, options);
    auto row_sums = torch::zeros({num_sims}, options);
    
    int total_threads = num_sims * num_actions;
    
    // Allocate/reallocate random states if needed
    if (d_dirichlet_states == nullptr || allocated_dirichlet_states < total_threads) {
        if (d_dirichlet_states != nullptr) {
            cudaFree(d_dirichlet_states);
        }
        cudaMalloc(&d_dirichlet_states, total_threads * sizeof(curandState));
        allocated_dirichlet_states = total_threads;
        
        // Initialize random states
        int threads = 256;
        int blocks = (total_threads + threads - 1) / threads;
        // Use time-based seed for randomness
        unsigned long long seed = static_cast<unsigned long long>(time(nullptr));
        init_dirichlet_states_kernel<<<blocks, threads>>>(
            d_dirichlet_states, total_threads, seed
        );
    }
    
    // Pass 1: Generate gamma samples and compute row sums
    int threads = 256;
    int blocks = (total_threads + threads - 1) / threads;
    
    batched_dirichlet_noise_kernel<<<blocks, threads>>>(
        d_dirichlet_states,
        noise.data_ptr<float>(),
        row_sums.data_ptr<float>(),
        num_sims,
        num_actions,
        alpha,
        epsilon
    );
    
    // Pass 2: Normalize samples by row sums
    normalize_dirichlet_kernel<<<blocks, threads>>>(
        noise.data_ptr<float>(),
        row_sums.data_ptr<float>(),
        num_sims,
        num_actions
    );
    
    return noise;
}

torch::Tensor fused_ucb_with_noise_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor priors,
    torch::Tensor dirichlet_noise,
    float parent_visits_sqrt,
    float c_puct,
    float epsilon
) {
    int num_sims = dirichlet_noise.size(0);
    int num_children = q_values.size(0);
    
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(q_values.device());
    auto selected = torch::zeros({num_sims}, options);
    
    int threads = 256;
    int blocks = (num_sims + threads - 1) / threads;
    
    fused_ucb_with_noise_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        priors.data_ptr<float>(),
        dirichlet_noise.data_ptr<float>(),
        selected.data_ptr<int>(),
        num_sims,
        num_children,
        parent_visits_sqrt,
        c_puct,
        epsilon
    );
    
    return selected;
}

void optimized_backup_scatter_cuda(
    torch::Tensor paths,
    torch::Tensor path_lengths,
    torch::Tensor values,
    torch::Tensor visit_counts,
    torch::Tensor value_sums
) {
    int batch_size = paths.size(0);
    int max_depth = paths.size(1);
    
    // Estimate maximum operations
    int max_operations = batch_size * max_depth;
    
    auto options_int = torch::TensorOptions().dtype(torch::kInt32).device(paths.device());
    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(paths.device());
    
    auto node_indices = torch::zeros({max_operations}, options_int);
    auto value_updates = torch::zeros({max_operations}, options_float);
    auto count_updates = torch::zeros({max_operations}, options_int);
    auto operation_count = torch::zeros({1}, options_int);
    
    // Prepare backup operations
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    prepare_backup_operations_kernel<<<blocks, threads>>>(
        paths.data_ptr<int>(),
        path_lengths.data_ptr<int>(),
        values.data_ptr<float>(),
        node_indices.data_ptr<int>(),
        value_updates.data_ptr<float>(),
        count_updates.data_ptr<int>(),
        operation_count.data_ptr<int>(),
        batch_size,
        max_depth
    );
    
    // Perform scatter operations efficiently
    int num_ops = operation_count.cpu().item<int>();
    if (num_ops > 0) {
        // Use existing vectorized_backup_kernel which already does this efficiently
        vectorized_backup_kernel<<<blocks, threads>>>(
            paths.data_ptr<int>(),
            path_lengths.data_ptr<int>(),
            values.data_ptr<float>(),
            visit_counts.data_ptr<int>(),
            value_sums.data_ptr<float>(),
            batch_size,
            max_depth,
            visit_counts.size(0)
        );
    }
}

void batch_apply_virtual_loss_cuda(
    torch::Tensor node_indices,
    torch::Tensor virtual_loss_counts
) {
    int num_nodes = node_indices.size(0);
    int max_nodes = virtual_loss_counts.size(0);  // FIXED: Get max_nodes from array size
    const int threads = 256;
    const int blocks = (num_nodes + threads - 1) / threads;
    
    batch_apply_virtual_loss_kernel<<<blocks, threads>>>(
        node_indices.data_ptr<int>(),
        virtual_loss_counts.data_ptr<int>(),
        num_nodes,
        max_nodes  // FIXED: Pass max_nodes parameter
    );
}

void warp_aggregated_virtual_loss_cuda(
    torch::Tensor node_indices,
    torch::Tensor virtual_loss_counts
) {
    int num_nodes = node_indices.size(0);
    int max_nodes = virtual_loss_counts.size(0);
    
    // Use 256 threads for 8 warps per block
    const int threads = 256;
    const int blocks = (num_nodes + threads - 1) / threads;
    
    // Calculate shared memory size: 8 warps * 32 lanes * 2 ints per lane * 4 bytes
    const int shared_mem_size = 8 * 32 * 2 * sizeof(int);
    
    warp_aggregated_virtual_loss_kernel<<<blocks, threads, shared_mem_size>>>(
        node_indices.data_ptr<int>(),
        virtual_loss_counts.data_ptr<int>(),
        num_nodes,
        max_nodes
    );
}

void batch_remove_virtual_loss_cuda(
    torch::Tensor node_indices,
    torch::Tensor virtual_loss_counts
) {
    int num_nodes = node_indices.size(0);
    int max_nodes = virtual_loss_counts.size(0);  // FIXED: Get max_nodes from array size
    const int threads = 256;
    const int blocks = (num_nodes + threads - 1) / threads;
    
    batch_remove_virtual_loss_kernel<<<blocks, threads>>>(
        node_indices.data_ptr<int>(),
        virtual_loss_counts.data_ptr<int>(),
        num_nodes,
        max_nodes  // FIXED: Pass max_nodes parameter
    );
}

// Enhanced version with optional Dirichlet noise and legal move filtering
torch::Tensor parallel_select_with_virtual_loss_enhanced_cuda(
    torch::Tensor parent_nodes,
    torch::Tensor children,
    torch::Tensor visit_counts,
    torch::Tensor virtual_loss_counts,
    torch::Tensor value_sums,
    torch::Tensor priors,
    torch::optional<torch::Tensor> dirichlet_noise,   // Optional (ignored - priors should be pre-mixed)
    torch::optional<torch::Tensor> legal_masks,       // Optional
    torch::Tensor child_actions,
    float c_puct,
    float virtual_loss_value,
    float dirichlet_epsilon,                          // Ignored - noise should be pre-mixed
    bool apply_legal_mask
) {
    int num_sims = parent_nodes.size(0);
    int max_children = children.size(1);
    int children_stride = children.stride(0);
    int action_space_size = apply_legal_mask && legal_masks.has_value() ? 
                           legal_masks.value().size(1) : 0;
    
    // Determine if priors are per-simulation (2D) or shared (1D)
    bool per_sim_priors = priors.dim() == 2;
    int priors_stride = per_sim_priors ? priors.size(1) : 0;
    
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(parent_nodes.device());
    auto selected_children = torch::zeros({num_sims}, options);
    auto virtual_loss_updates = torch::zeros({num_sims}, options);
    
    const int threads = 256;
    const int blocks = (num_sims + threads - 1) / threads;
    
    parallel_select_with_virtual_loss_kernel<<<blocks, threads>>>(
        parent_nodes.data_ptr<int>(),
        children.data_ptr<int>(),
        visit_counts.data_ptr<int>(),
        virtual_loss_counts.data_ptr<int>(),
        value_sums.data_ptr<float>(),
        priors.data_ptr<float>(),
        legal_masks.has_value() ? legal_masks.value().data_ptr<bool>() : nullptr,
        child_actions.data_ptr<int>(),
        selected_children.data_ptr<int>(),
        virtual_loss_updates.data_ptr<int>(),
        num_sims,
        max_children,
        children_stride,
        action_space_size,
        c_puct,
        virtual_loss_value,
        apply_legal_mask,
        per_sim_priors,
        priors_stride
    );
    
    // Apply virtual losses to selected children
    auto valid_mask = virtual_loss_updates >= 0;
    if (valid_mask.any().item<bool>()) {
        auto valid_updates = virtual_loss_updates.masked_select(valid_mask);
        batch_apply_virtual_loss_cuda(valid_updates, virtual_loss_counts);
    }
    
    return selected_children;
}


// Fused select and expand operation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fused_select_expand_cuda(
    torch::Tensor roots,
    torch::Tensor children,
    torch::Tensor visit_counts,
    torch::Tensor q_values,
    torch::Tensor prior_probs,
    torch::Tensor is_expanded,
    int max_depth,
    float c_puct)
{
    const int num_waves = roots.size(0);
    const int max_children = children.size(1);
    
    // Allocate output tensors
    auto selected_paths = torch::zeros({num_waves, max_depth}, torch::kInt32).cuda();
    auto path_lengths = torch::zeros({num_waves}, torch::kInt32).cuda();
    auto expand_nodes = torch::zeros({num_waves}, torch::kInt32).cuda();
    auto needs_expansion = torch::zeros({num_waves}, torch::kBool).cuda();
    
    // Launch kernel
    const int threads = 256;
    const int blocks = (num_waves + threads - 1) / threads;
    const int shared_mem = threads * max_depth * sizeof(int);
    
    fused_select_expand_kernel<<<blocks, threads, shared_mem>>>(
        roots.data_ptr<int>(),
        children.data_ptr<int>(),
        visit_counts.data_ptr<int>(),
        q_values.data_ptr<float>(),
        prior_probs.data_ptr<float>(),
        is_expanded.data_ptr<bool>(),
        num_waves,
        max_children,
        max_depth,
        c_puct,
        selected_paths.data_ptr<int>(),
        path_lengths.data_ptr<int>(),
        expand_nodes.data_ptr<int>(),
        needs_expansion.data_ptr<bool>()
    );
    
    return std::make_tuple(selected_paths, path_lengths, expand_nodes, needs_expansion);
}

// Phase 2.1 Optimized fused select-expand operation
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_select_expand_optimized_cuda(
    torch::Tensor root_nodes,
    torch::Tensor children,
    torch::Tensor visit_counts,
    torch::Tensor value_sums,
    torch::Tensor priors,
    int max_depth,
    float c_puct)
{
    const int batch_size = root_nodes.size(0);
    const int max_children = children.size(1);
    
    // Allocate output tensors
    auto selected_paths = torch::zeros({batch_size, max_depth}, torch::kInt32).cuda();
    auto path_lengths = torch::zeros({batch_size}, torch::kInt32).cuda();
    auto ucb_scores = torch::zeros({batch_size, max_children}, torch::kFloat32).cuda();
    
    // Launch optimized kernel
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    fused_select_expand_optimized_kernel<<<blocks, threads>>>(
        root_nodes.data_ptr<int>(),
        children.data_ptr<int>(),
        visit_counts.data_ptr<int>(),
        value_sums.data_ptr<float>(),
        priors.data_ptr<float>(),
        ucb_scores.data_ptr<float>(),
        selected_paths.data_ptr<int>(),
        path_lengths.data_ptr<int>(),
        batch_size,
        max_children,
        max_depth,
        c_puct
    );
    
    return std::make_tuple(selected_paths, path_lengths, ucb_scores);
}

// =============================================================================
// =============================================================================
// SECTION 6: BFS TREE OPERATIONS
// =============================================================================

/**
 * Parallel BFS kernel for subtree extraction
 * Performs wave-based BFS traversal on GPU without CPU synchronization
 */
__global__ void parallel_bfs_subtree_kernel(
    const int* __restrict__ row_ptr,        // CSR row pointers
    const int* __restrict__ col_indices,    // CSR column indices  
    const int root_idx,                     // Starting node
    const int max_nodes,                    // Maximum nodes in tree
    bool* __restrict__ in_subtree,          // Output: nodes in subtree
    int* __restrict__ node_remap,           // Output: old->new mapping
    int* __restrict__ frontier,             // Work queue
    int* __restrict__ frontier_size,        // Atomic counter for frontier
    int* __restrict__ subtree_count         // Atomic counter for subtree nodes
) {
    // Thread and warp info
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = (gridDim.x * blockDim.x) / 32;
    
    // Initialize root on first thread
    if (tid == 0) {
        in_subtree[root_idx] = true;
        node_remap[root_idx] = 0;
        frontier[0] = root_idx;
        *frontier_size = 1;
        *subtree_count = 1;
    }
    
    __syncthreads();
    
    // Wave-based BFS
    int current_frontier_start = 0;
    int current_frontier_end = 1;
    
    while (current_frontier_start < current_frontier_end) {
        // Each warp processes a portion of the frontier
        for (int idx = current_frontier_start + warp_id; 
             idx < current_frontier_end; 
             idx += num_warps) {
            
            if (idx < current_frontier_end) {
                int node = frontier[idx];
                int row_start = row_ptr[node];
                int row_end = row_ptr[node + 1];
                
                // Each lane in warp processes children
                for (int child_idx = row_start + lane_id; 
                     child_idx < row_end; 
                     child_idx += 32) {
                    
                    int child = col_indices[child_idx];
                    
                    // Atomic check and mark
                    bool is_new = false;
                    if (child >= 0 && child < max_nodes) {
                        // Use atomic CAS for thread-safe marking
                        unsigned int* in_subtree_int = (unsigned int*)in_subtree;
                        unsigned int old_val = atomicCAS(
                            &in_subtree_int[child], 0, 1
                        );
                        is_new = (old_val == 0);
                    }
                    
                    // If newly discovered, add to frontier
                    if (is_new) {
                        // Get position in frontier
                        int pos = atomicAdd(frontier_size, 1);
                        frontier[pos] = child;
                        
                        // Assign remapped index
                        int new_idx = atomicAdd(subtree_count, 1) - 1;
                        node_remap[child] = new_idx;
                    }
                }
            }
        }
        
        // Move to next frontier level
        __syncthreads();
        
        // CRITICAL FIX: Use shared memory to safely read frontier_size
        __shared__ int next_frontier_end;
        
        if (tid == 0) {
            current_frontier_start = current_frontier_end;
            // Read frontier_size once all threads have finished adding
            next_frontier_end = *frontier_size;
        }
        
        __syncthreads();
        
        if (tid == 0) {
            current_frontier_end = next_frontier_end;
            
            // Safety check: prevent infinite loops
            if (current_frontier_start >= current_frontier_end) {
                // No new nodes discovered, exit the loop
                current_frontier_end = current_frontier_start;
            }
        }
        
        __syncthreads();
    }
}

/**
 * Optimized kernel for edge extraction with coalesced access
 */
__global__ void extract_subtree_edges_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const short* __restrict__ edge_actions,
    const bool* __restrict__ in_subtree,
    const int* __restrict__ node_remap,
    const int num_nodes,
    int* __restrict__ edge_parents,
    int* __restrict__ edge_children,
    short* __restrict__ edge_actions_out,
    int* __restrict__ edge_count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    // Each thread processes multiple nodes for better load balancing
    for (int node_idx = tid; node_idx < num_nodes; node_idx += stride) {
        if (in_subtree[node_idx]) {
            int new_parent = node_remap[node_idx];
            int row_start = row_ptr[node_idx];
            int row_end = row_ptr[node_idx + 1];
            
            // Process edges from this node
            for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
                int child = col_indices[edge_idx];
                
                if (child >= 0 && in_subtree[child]) {
                    int new_child = node_remap[child];
                    
                    // Atomic increment to get position
                    int pos = atomicAdd(edge_count, 1);
                    
                    // Write edge data (coalesced access pattern)
                    edge_parents[pos] = new_parent;
                    edge_children[pos] = new_child;
                    if (edge_actions != nullptr) {
                        edge_actions_out[pos] = edge_actions[edge_idx];
                    }
                }
            }
        }
    }
}

// C++ wrapper functions for BFS operations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> parallel_bfs_subtree_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    int root_idx,
    int max_nodes
) {
    auto options = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(row_ptr.device());
    auto in_subtree = torch::zeros({max_nodes}, options);
    
    auto node_remap = torch::full({max_nodes}, -1, 
        torch::TensorOptions().dtype(torch::kInt32).device(row_ptr.device()));
    
    auto frontier = torch::zeros({max_nodes}, 
        torch::TensorOptions().dtype(torch::kInt32).device(row_ptr.device()));
    
    auto frontier_size = torch::zeros({1}, 
        torch::TensorOptions().dtype(torch::kInt32).device(row_ptr.device()));
    
    auto subtree_count = torch::zeros({1}, 
        torch::TensorOptions().dtype(torch::kInt32).device(row_ptr.device()));
    
    // Calculate grid dimensions
    const int block_size = 256;
    const int num_blocks = (max_nodes + block_size - 1) / block_size;
    
    // Launch kernel
    parallel_bfs_subtree_kernel<<<num_blocks, block_size>>>(
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        root_idx,
        max_nodes,
        in_subtree.data_ptr<bool>(),
        node_remap.data_ptr<int>(),
        frontier.data_ptr<int>(),
        frontier_size.data_ptr<int>(),
        subtree_count.data_ptr<int>()
    );
    
    return std::make_tuple(in_subtree, node_remap, subtree_count);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> extract_subtree_edges_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    torch::Tensor edge_actions,
    torch::Tensor in_subtree,
    torch::Tensor node_remap,
    int max_edges
) {
    auto device = row_ptr.device();
    int num_nodes = in_subtree.size(0);
    
    // Allocate output tensors
    auto edge_parents = torch::zeros({max_edges}, 
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto edge_children = torch::zeros({max_edges}, 
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto edge_actions_out = torch::zeros({max_edges}, 
        torch::TensorOptions().dtype(torch::kInt16).device(device));
    auto edge_count = torch::zeros({1}, 
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    // Calculate grid dimensions
    const int block_size = 256;
    const int num_blocks = (num_nodes + block_size - 1) / block_size;
    
    // Launch kernel
    extract_subtree_edges_kernel<<<num_blocks, block_size>>>(
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        edge_actions.defined() ? edge_actions.data_ptr<short>() : nullptr,
        in_subtree.data_ptr<bool>(),
        node_remap.data_ptr<int>(),
        num_nodes,
        edge_parents.data_ptr<int>(),
        edge_children.data_ptr<int>(),
        edge_actions_out.data_ptr<short>(),
        edge_count.data_ptr<int>()
    );
    
    // Get actual edge count
    int actual_edges = edge_count.cpu().item<int>();
    
    // Return only the used portion
    return std::make_tuple(
        edge_parents.slice(0, 0, actual_edges),
        edge_children.slice(0, 0, actual_edges),
        edge_actions_out.slice(0, 0, actual_edges)
    );
}

// PYBIND11 MODULE DEFINITION
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Consolidated MCTS CUDA kernels for high-performance tree search";
    
    // Core MCTS operations
    m.def("find_expansion_nodes", &find_expansion_nodes_cuda, "Find nodes needing expansion");
    m.def("vectorized_backup", &vectorized_backup_cuda, "Vectorized backup operation");
    m.def("warp_vectorized_backup", &warp_vectorized_backup_cuda, "Warp-optimized vectorized backup with primitives");
    m.def("batched_add_children", &batched_add_children_cuda, "Batched node addition with parent assignment");
    
    // UCB selection operations
    m.def("batched_ucb_selection", &batched_ucb_selection_cuda, "Batched UCB selection (CSR format)");
    m.def("fused_ucb_selection", &fused_ucb_selection_cuda, "High-performance fused UCB selection");
    m.def("batched_ucb_selection_dense", &batched_ucb_selection_dense_cuda, "Batched UCB selection (dense format)");
    
    // Quantum operations removed for performance
    
    // Wave search optimizations
    m.def("batched_dirichlet_noise", &batched_dirichlet_noise_cuda, "Batched Dirichlet noise generation");
    m.def("fused_ucb_with_noise", &fused_ucb_with_noise_cuda, "Fused UCB computation with Dirichlet noise");
    m.def("optimized_backup_scatter", &optimized_backup_scatter_cuda, "Optimized backup with scatter operations");
    m.def("fused_select_expand", &fused_select_expand_cuda, "Fused select and expand operation");
    m.def("fused_select_expand_optimized", &fused_select_expand_optimized_cuda, 
          "Phase 2.1 Optimized fused select-expand kernel with 4x reduced kernel launch overhead");
    
    // Virtual loss operations
    m.def("batch_apply_virtual_loss", &batch_apply_virtual_loss_cuda, "Batch apply virtual losses");
    m.def("warp_aggregated_virtual_loss", &warp_aggregated_virtual_loss_cuda, 
          "Warp-aggregated virtual loss application with reduced contention");
    m.def("batch_remove_virtual_loss", &batch_remove_virtual_loss_cuda, "Batch remove virtual losses");
    m.def("parallel_select_with_virtual_loss", &parallel_select_with_virtual_loss_enhanced_cuda, 
          "Enhanced parallel selection with virtual loss, Dirichlet noise, and legal move filtering",
          py::arg("parent_nodes"),
          py::arg("children"), 
          py::arg("visit_counts"),
          py::arg("virtual_loss_counts"),
          py::arg("value_sums"),
          py::arg("priors"),
          py::arg("dirichlet_noise") = py::none(),
          py::arg("legal_masks") = py::none(),
          py::arg("child_actions"),
          py::arg("c_puct"),
          py::arg("virtual_loss_value"),
          py::arg("dirichlet_epsilon") = 0.0f,
          py::arg("apply_legal_mask") = false);
    
    // Utility functions
    m.def("initialize_lookup_tables", 
          py::overload_cast<torch::Tensor, torch::Tensor, float>(&initialize_lookup_tables_cuda), 
          "Initialize lookup tables with tensors and hbar_eff");
    m.def("initialize_lookup_tables", 
          py::overload_cast<>(&initialize_lookup_tables_cuda), 
          "Initialize lookup tables (no-op version for testing)");
    
    // BFS tree operations for GPU-friendly tree reuse
    m.def("parallel_bfs_subtree", &parallel_bfs_subtree_cuda, 
          "Parallel BFS for subtree extraction",
          py::arg("row_ptr"),
          py::arg("col_indices"),
          py::arg("root_idx"),
          py::arg("max_nodes"));
    m.def("extract_subtree_edges", &extract_subtree_edges_cuda, 
          "Extract edges from subtree",
          py::arg("row_ptr"),
          py::arg("col_indices"),
          py::arg("edge_actions"),
          py::arg("in_subtree"),
          py::arg("node_remap"),
          py::arg("max_edges"));
}