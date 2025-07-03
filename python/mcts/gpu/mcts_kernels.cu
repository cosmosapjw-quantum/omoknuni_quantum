// =============================================================================
// CONSOLIDATED MCTS CUDA KERNELS
// =============================================================================
// This file consolidates all CUDA kernels for MCTS operations into a single,
// well-organized file with clear sections and manageable code structure.
//
// Sections:
// 1. Core MCTS Operations (expansion, backup, tree management)
// 2. UCB Selection (classical and optimized variants)  
// 3. Quantum-Enhanced Operations (quantum UCB, phase calculations)
// 4. Utility Functions (memory management, helper functions)
//
// Consolidated from:
// - unified_cuda_kernels.cu (60KB - contained everything)
// - mcts_core_kernels.cu (9KB - core operations)
// - mcts_selection_kernels.cu (12KB - UCB selection)
// - mcts_quantum_kernels.cu (12KB - quantum enhancements)
// =============================================================================

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <ctime>
#include <climits>

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
// SECTION 3: QUANTUM-ENHANCED OPERATIONS
// =============================================================================

/**
 * Quantum-enhanced UCB selection with phase interference and uncertainty
 */
__global__ void quantum_ucb_selection_kernel(
    const float* __restrict__ q_values,
    const int* __restrict__ visit_counts,
    const int* __restrict__ parent_visits,
    const float* __restrict__ priors,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const float* __restrict__ quantum_phases,     // Pre-computed phases [num_edges]
    const float* __restrict__ uncertainty_table,  // Quantum uncertainty lookup [max_visits]
    int* __restrict__ selected_actions,
    float* __restrict__ selected_scores,
    const int64_t num_nodes,
    const float c_puct,
    const int max_table_size,
    const float hbar_eff,                         // Effective Planck constant
    const float phase_kick_strength,              // Phase kick amplitude
    const float interference_alpha                // Interference strength
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
    float sqrt_parent = sqrtf(static_cast<float>(parent_visit + 1));
    
    float best_ucb = -1e10f;
    int best_action = 0;
    
    // Quantum-enhanced UCB calculation
    for (int i = start; i < end; i++) {
        int child_idx = col_indices[i];
        int child_visit = visit_counts[child_idx];
        
        float q_value = (child_visit > 0) ? q_values[child_idx] : 0.0f;
        
        // Classical UCB term
        float exploration = c_puct * priors[child_idx] * sqrt_parent / (1.0f + child_visit);
        float classical_ucb = q_value + exploration;
        
        // Quantum enhancements
        float quantum_phase = quantum_phases[i];
        float uncertainty = (child_visit < max_table_size) ? 
            uncertainty_table[child_visit] : hbar_eff / sqrtf(child_visit + 1.0f);
        
        // Phase interference term
        float phase_interference = phase_kick_strength * cosf(quantum_phase);
        
        // Uncertainty-based exploration boost
        float quantum_exploration = interference_alpha * uncertainty;
        
        // Combined quantum UCB score
        float quantum_ucb = classical_ucb + phase_interference + quantum_exploration;
        
        if (quantum_ucb > best_ucb) {
            best_ucb = quantum_ucb;
            best_action = i - start;
        }
    }
    
    selected_actions[idx] = best_action;
    selected_scores[idx] = best_ucb;
}

/**
 * Compute quantum phases for interference effects
 */
__global__ void compute_quantum_phases_kernel(
    const float* __restrict__ q_values,
    const int* __restrict__ visit_counts,
    const float* __restrict__ priors,
    float* __restrict__ phases,
    const int num_edges,
    const float hbar_eff,
    const float time_step
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    
    float q_val = q_values[idx];
    int visits = visit_counts[idx];
    float prior = priors[idx];
    
    // Quantum phase calculation based on action value and visit history
    float energy = q_val + prior * logf(visits + 1.0f);
    float phase = (energy * time_step) / hbar_eff;
    
    // Normalize phase to [0, 2π]
    phases[idx] = fmodf(phase, 2.0f * M_PI);
}

// =============================================================================
// SECTION 4: UTILITY FUNCTIONS
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

torch::Tensor quantum_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    torch::Tensor quantum_phases,
    torch::Tensor uncertainty_table,
    float c_puct,
    float hbar_eff,
    float phase_kick_strength,
    float interference_alpha
);

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
    
    return expansion_nodes;
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

torch::Tensor quantum_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts,
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    torch::Tensor quantum_phases,
    torch::Tensor uncertainty_table,
    float c_puct,
    float hbar_eff,
    float phase_kick_strength,
    float interference_alpha
) {
    auto device = q_values.device();
    auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    auto selected_actions = torch::zeros({q_values.size(0)}, int_options);
    auto selected_scores = torch::zeros({q_values.size(0)}, float_options);
    
    const int threads = 256;
    const int blocks = (q_values.size(0) + threads - 1) / threads;
    
    quantum_ucb_selection_kernel<<<blocks, threads>>>(
        q_values.data_ptr<float>(),
        visit_counts.data_ptr<int>(),
        parent_visits.data_ptr<int>(),
        priors.data_ptr<float>(),
        row_ptr.data_ptr<int>(),
        col_indices.data_ptr<int>(),
        quantum_phases.data_ptr<float>(),
        uncertainty_table.data_ptr<float>(),
        selected_actions.data_ptr<int>(),
        selected_scores.data_ptr<float>(),
        q_values.size(0),
        c_puct,
        uncertainty_table.size(0),
        hbar_eff,
        phase_kick_strength,
        interference_alpha
    );
    
    return selected_actions;
}

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

// =============================================================================
// PYBIND11 MODULE DEFINITION
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Consolidated MCTS CUDA kernels for high-performance tree search";
    
    // Core MCTS operations
    m.def("find_expansion_nodes", &find_expansion_nodes_cuda, "Find nodes needing expansion");
    m.def("vectorized_backup", &vectorized_backup_cuda, "Vectorized backup operation");
    
    // UCB selection operations
    m.def("batched_ucb_selection", &batched_ucb_selection_cuda, "Batched UCB selection");
    
    // Quantum-enhanced operations
    m.def("quantum_ucb_selection", &quantum_ucb_selection_cuda, "Quantum-enhanced UCB selection");
    
    // Utility functions
    m.def("initialize_lookup_tables", &initialize_lookup_tables_cuda, "Initialize lookup tables");
}