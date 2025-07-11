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
        int node_idx = paths[batch_id * max_depth + depth];
        
        if (node_idx >= 0 && node_idx < max_nodes) {
            // Alternate signs for different players
            float sign = (depth % 2 == 0) ? 1.0f : -1.0f;
            float backup_value = batch_value * sign;
            
            // Atomic updates for thread safety
            atomicAdd(&visit_counts[node_idx], 1);
            atomicAdd(&value_sums[node_idx], backup_value);
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
    
    // UCB calculation for each child
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        float child_visit = static_cast<float>(visit_counts[child_idx]);
        
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        
        // UCB formula: Q + c_puct * prior * sqrt(parent_visits) / (1 + child_visits)
        float exploration = c_puct * priors[child_idx] * sqrt_parent / (1.0f + child_visit);
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
        float exploration = c_puct * priors[child_idx] * sqrt_parent / (1.0f + child_visit);
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

// Quantum function declaration removed

void vectorized_backup_cuda(
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

torch::Tensor batched_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct
) {
    auto device = q_values.device();
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    auto selected_actions = torch::zeros({q_values.size(0)}, int_options);
    auto selected_scores = torch::zeros({q_values.size(0)}, float_options);
    
    const int threads = 256;
    const int blocks = (q_values.size(0) + threads - 1) / threads;
    
    batched_ucb_selection_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        parent_visits.data_ptr<int>(),
        priors.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        selected_actions.data_ptr<int>(),
        selected_scores.data_ptr<float>(),
        q_values.size(0),
        c_puct
    );
    
    return selected_actions;
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

// =============================================================================
// PYBIND11 MODULE DEFINITION
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Consolidated MCTS CUDA kernels for high-performance tree search";
    
    // Core MCTS operations
    m.def("find_expansion_nodes", &find_expansion_nodes_cuda, "Find nodes needing expansion");
    m.def("vectorized_backup", &vectorized_backup_cuda, "Vectorized backup operation");
    m.def("batched_add_children", &batched_add_children_cuda, "Batched node addition with parent assignment");
    
    // UCB selection operations
    m.def("batched_ucb_selection", &batched_ucb_selection_cuda, "Batched UCB selection");
    
    // Quantum operations removed for performance
    
    // Wave search optimizations
    m.def("batched_dirichlet_noise", &batched_dirichlet_noise_cuda, "Batched Dirichlet noise generation");
    m.def("fused_ucb_with_noise", &fused_ucb_with_noise_cuda, "Fused UCB computation with Dirichlet noise");
    m.def("optimized_backup_scatter", &optimized_backup_scatter_cuda, "Optimized backup with scatter operations");
    m.def("fused_select_expand", &fused_select_expand_cuda, "Fused select and expand operation");
    
    // Utility functions
    m.def("initialize_lookup_tables", &initialize_lookup_tables_cuda, "Initialize lookup tables");
}