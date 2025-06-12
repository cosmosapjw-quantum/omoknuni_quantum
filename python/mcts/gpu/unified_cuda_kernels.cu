// Unified CUDA kernels for high-performance MCTS
// This file consolidates all CUDA kernels into a single, optimized implementation

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <ctime>
#include <climits>

// ============================================================================
// Core MCTS Kernels
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

// Quantum-enhanced UCB selection kernel
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
    int path_length = path_lengths[idx];
    
    // Traverse path from leaf to root
    for (int depth = 0; depth < path_length && depth < max_depth; depth++) {
        int node_idx = paths[idx * max_depth + depth];
        if (node_idx < 0) break;
        
        // Atomic updates for thread safety
        atomicAdd(&visit_counts[node_idx], 1);
        atomicAdd(&value_sums[node_idx], value);
        
        // Negate value for opponent's perspective
        value = -value;
    }
}

// ============================================================================
// Quantum-Enhanced Kernels
// ============================================================================

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
    
    // Standard UCB calculation
    float parent_visit = 0.0f;
    for (int i = 0; i < max_actions; i++) {
        parent_visit += visit_counts[batch_idx * max_actions + i];
    }
    float sqrt_parent = sqrtf(parent_visit + 1.0f);
    float exploration = c_puct * prior * sqrt_parent / (1.0f + visit);
    
    // Quantum correction with interference
    float quantum_factor = expf(-lambda_qft / (hbar_eff * hbar_eff));
    float interference = quantum_factor * cosf(phase);
    
    ucb_scores[idx] = q_val + exploration * (1.0f + 0.1f * interference);
}

// ============================================================================
// Game State Management Kernels
// ============================================================================

// Kernel to apply moves to game states in batch
__global__ void batch_apply_moves_kernel(
    int8_t* __restrict__ boards,           // Board states [num_states x board_size x board_size]
    int8_t* __restrict__ current_players,  // Current player for each state
    int16_t* __restrict__ move_counts,     // Move counter for each state
    int16_t* __restrict__ move_history,    // Move history [num_states x history_size]
    const int* __restrict__ state_indices, // Which states to update
    const int* __restrict__ actions,       // Actions to apply
    const int batch_size,
    const int board_size,
    const int history_size,
    const int game_type  // 0=chess, 1=go, 2=gomoku
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    int state_idx = state_indices[tid];
    int action = actions[tid];
    
    if (action < 0) return;  // Invalid action
    
    if (game_type == 2 || game_type == 1) {  // Gomoku or Go
        // Convert action to board coordinates
        int row = action / board_size;
        int col = action % board_size;
        
        // Get current player
        int8_t player = current_players[state_idx];
        
        // Place stone on board
        int board_offset = state_idx * board_size * board_size;
        boards[board_offset + row * board_size + col] = player;
        
        // Update move history (shift left and add new move)
        int history_offset = state_idx * history_size;
        for (int i = 0; i < history_size - 1; i++) {
            move_history[history_offset + i] = move_history[history_offset + i + 1];
        }
        move_history[history_offset + history_size - 1] = action;
        
        // Switch player
        current_players[state_idx] = (player == 1) ? 2 : 1;
        
        // Increment move count
        move_counts[state_idx]++;
    }
    // Chess would require more complex move application
}

// Kernel to generate legal moves mask for multiple states
__global__ void generate_legal_moves_mask_kernel(
    const int8_t* __restrict__ boards,     // Board states
    const int8_t* __restrict__ metadata,   // Game-specific metadata
    bool* __restrict__ legal_mask,         // Output: legal moves mask
    int* __restrict__ move_counts,         // Output: number of legal moves per state
    const int* __restrict__ state_indices, // States to process
    const int batch_size,
    const int board_size,
    const int action_space_size,
    const int game_type
) {
    // Grid-stride loop for efficiency
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple actions
    for (int idx = tid; idx < batch_size * action_space_size; idx += stride) {
        int batch_idx = idx / action_space_size;
        int action_idx = idx % action_space_size;
        
        if (batch_idx >= batch_size) continue;
        
        int state_idx = state_indices[batch_idx];
        int board_offset = state_idx * board_size * board_size;
        
        bool is_legal = false;
        
        if (game_type == 2) {  // Gomoku
            // For Gomoku, legal moves are empty squares
            if (action_idx < board_size * board_size) {
                is_legal = (boards[board_offset + action_idx] == 0);
            }
        } else if (game_type == 1) {  // Go
            // For Go, check empty squares and ko rule
            if (action_idx < board_size * board_size) {
                is_legal = (boards[board_offset + action_idx] == 0);
                // TODO: Check ko point from metadata
            }
        }
        // Chess would require complex legal move generation
        
        // Write result
        legal_mask[batch_idx * action_space_size + action_idx] = is_legal;
        
        // Count legal moves (first thread per batch)
        if (action_idx == 0) {
            atomicAdd(&move_counts[batch_idx], 0);  // Initialize
        }
        __syncthreads();
        
        if (is_legal) {
            atomicAdd(&move_counts[batch_idx], 1);
        }
    }
}

// ============================================================================
// Tree Building Kernels
// ============================================================================

// Helper kernel to compute prefix sums for child allocation
__global__ void compute_child_offsets_kernel(
    const int* __restrict__ num_children,
    int* __restrict__ child_offsets,
    int* __restrict__ total_children,
    const int batch_size
) {
    // Simple sequential scan for now - could optimize with parallel scan
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int offset = 0;
        for (int i = 0; i < batch_size; i++) {
            child_offsets[i] = offset;
            offset += num_children[i];
        }
        *total_children = offset;
    }
}

__global__ void batched_add_children_kernel(
    const int* __restrict__ parent_indices,
    const int* __restrict__ actions,
    const float* __restrict__ priors,
    const int* __restrict__ num_children,
    const int* __restrict__ child_offsets,
    int* __restrict__ node_counter,
    int* __restrict__ edge_counter,
    int* __restrict__ children_table,
    int* __restrict__ parent_idx_array,
    int* __restrict__ parent_action_array,
    float* __restrict__ node_priors_array,
    int* __restrict__ visit_counts_array,
    float* __restrict__ value_sums_array,
    int* __restrict__ child_indices_out,
    // CSR edge data arrays - FIXED: Added missing parameters
    int* __restrict__ col_indices,
    int* __restrict__ edge_actions,
    float* __restrict__ edge_priors,
    const int total_children,
    const int batch_size,
    const int max_nodes,
    const int max_children_per_node,
    const int max_edges
) {
    // Each thread processes one child (not one parent!)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_children) return;
    
    // Binary search to find which parent this child belongs to
    int parent_batch_idx = 0;
    int left = 0;
    int right = batch_size - 1;
    
    // Find parent index using offsets
    while (left <= right) {
        int mid = (left + right) / 2;
        if (tid < child_offsets[mid]) {
            right = mid - 1;
        } else if (mid + 1 < batch_size && tid >= child_offsets[mid + 1]) {
            left = mid + 1;
        } else {
            parent_batch_idx = mid;
            break;
        }
    }
    
    // Get parent info
    int parent_idx = parent_indices[parent_batch_idx];
    int child_local_idx = tid - child_offsets[parent_batch_idx];
    
    // Allocate global child index
    int child_idx = atomicAdd(node_counter, 1);
    
    // Check bounds
    if (child_idx >= max_nodes) {
        child_indices_out[tid] = -1;
        return;
    }
    
    // Get action and prior for this specific child
    int action = actions[parent_batch_idx * max_children_per_node + child_local_idx];
    float prior = priors[parent_batch_idx * max_children_per_node + child_local_idx];
    
    // Initialize child node - no contention here since each thread writes to unique location
    parent_idx_array[child_idx] = parent_idx;
    parent_action_array[child_idx] = action;
    node_priors_array[child_idx] = prior;
    visit_counts_array[child_idx] = 0;
    value_sums_array[child_idx] = 0.0f;
    
    // Store output
    child_indices_out[tid] = child_idx;
    
    // FIXED: Update CSR edge data - THIS WAS MISSING!
    int edge_idx = atomicAdd(edge_counter, 1);
    if (edge_idx < max_edges) {
        col_indices[edge_idx] = child_idx;
        edge_actions[edge_idx] = action;
        edge_priors[edge_idx] = prior;
    }
    
    // Update children table - still has some contention but distributed
    for (int j = 0; j < max_children_per_node; j++) {
        int old_val = atomicCAS(&children_table[parent_idx * max_children_per_node + j], -1, child_idx);
        if (old_val == -1) break;
    }
}

// ============================================================================
// Utility Kernels
// ============================================================================


__global__ void evaluate_gomoku_positions_kernel(
    const int8_t* __restrict__ boards,
    const int8_t* __restrict__ current_players,
    float* __restrict__ features,
    const int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * 15 * 15) return;
    
    int batch_idx = idx / (15 * 15);
    int pos_idx = idx % (15 * 15);
    
    int8_t cell = boards[batch_idx * 15 * 15 + pos_idx];
    int8_t current_player = current_players[batch_idx];
    
    // Feature extraction
    features[batch_idx * 3 * 15 * 15 + 0 * 15 * 15 + pos_idx] = 
        (cell == current_player) ? 1.0f : 0.0f;
    features[batch_idx * 3 * 15 * 15 + 1 * 15 * 15 + pos_idx] = 
        (cell != 0 && cell != current_player) ? 1.0f : 0.0f;
    features[batch_idx * 3 * 15 * 15 + 2 * 15 * 15 + pos_idx] = 
        (cell == 0) ? 1.0f : 0.0f;
}

// ============================================================================
// C++ Interface Functions
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
    const int num_nodes = parent_visits.size(0);
    auto selected_actions = torch::zeros({num_nodes}, 
        torch::TensorOptions().dtype(torch::kInt32).device(q_values.device()));
    auto selected_scores = torch::zeros({num_nodes}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(q_values.device()));
    
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

torch::Tensor batched_add_children_cuda(
    torch::Tensor parent_indices,
    torch::Tensor actions,
    torch::Tensor priors,
    torch::Tensor num_children,
    torch::Tensor node_counter,
    torch::Tensor edge_counter,
    torch::Tensor children_table,
    torch::Tensor parent_idx_array,
    torch::Tensor parent_action_array,
    torch::Tensor node_priors_array,
    torch::Tensor visit_counts_array,
    torch::Tensor value_sums_array,
    // FIXED: Added CSR edge data tensors
    torch::Tensor col_indices,
    torch::Tensor edge_actions,
    torch::Tensor edge_priors,
    int max_nodes,
    int max_children_per_node,
    int max_edges
) {
    const int batch_size = parent_indices.size(0);
    
    // First, compute offsets and total children count
    auto child_offsets = torch::zeros({batch_size}, 
        torch::TensorOptions().dtype(torch::kInt32).device(parent_indices.device()));
    auto total_children_tensor = torch::zeros({1}, 
        torch::TensorOptions().dtype(torch::kInt32).device(parent_indices.device()));
    
    // Launch offset computation kernel
    compute_child_offsets_kernel<<<1, 1>>>(
        num_children.data_ptr<int>(),
        child_offsets.data_ptr<int>(),
        total_children_tensor.data_ptr<int>(),
        batch_size
    );
    
    // Synchronize to get total children count
    cudaDeviceSynchronize();
    int total_children = total_children_tensor.item<int>();
    
    if (total_children == 0) {
        return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(parent_indices.device()));
    }
    
    // Allocate output tensor for all children
    auto child_indices_out = torch::full({total_children}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(parent_indices.device()));
    
    // Launch main kernel with one thread per child
    const int threads = 256;
    const int blocks = (total_children + threads - 1) / threads;
    
    batched_add_children_kernel<<<blocks, threads>>>(
        parent_indices.data_ptr<int>(),
        actions.data_ptr<int>(),
        priors.data_ptr<float>(),
        num_children.data_ptr<int>(),
        child_offsets.data_ptr<int>(),
        node_counter.data_ptr<int>(),
        edge_counter.data_ptr<int>(),
        children_table.data_ptr<int>(),
        parent_idx_array.data_ptr<int>(),
        parent_action_array.data_ptr<int>(),
        node_priors_array.data_ptr<float>(),
        visit_counts_array.data_ptr<int>(),
        value_sums_array.data_ptr<float>(),
        child_indices_out.data_ptr<int>(),
        // FIXED: Pass CSR edge data arrays
        col_indices.data_ptr<int>(),
        edge_actions.data_ptr<int>(),
        edge_priors.data_ptr<float>(),
        total_children,
        batch_size,
        max_nodes,
        max_children_per_node,
        max_edges
    );
    
    // Return the flat array for now - reshape can be done in Python if needed
    return child_indices_out;
}

torch::Tensor evaluate_gomoku_positions_cuda(
    torch::Tensor boards,
    torch::Tensor current_players
) {
    const int batch_size = boards.size(0);
    const int total_cells = batch_size * 15 * 15;
    
    auto features = torch::zeros({batch_size, 3, 15, 15},
        torch::TensorOptions().dtype(torch::kFloat32).device(boards.device()));
    
    const int threads = 256;
    const int blocks = (total_cells + threads - 1) / threads;
    
    evaluate_gomoku_positions_kernel<<<blocks, threads>>>(
        boards.data_ptr<int8_t>(),
        current_players.data_ptr<int8_t>(),
        features.data_ptr<float>(),
        batch_size
    );
    
    return features;
}

// Wrapper for finding expansion nodes
std::tuple<torch::Tensor, torch::Tensor> find_expansion_nodes_cuda(
    torch::Tensor current_nodes,
    torch::Tensor children,
    torch::Tensor visit_counts,
    torch::Tensor valid_path_mask,
    int64_t wave_size,
    int64_t max_children,
    int num_nodes
) {
    auto expansion_nodes = torch::zeros({wave_size}, 
        torch::TensorOptions().dtype(torch::kInt32).device(current_nodes.device()));
    auto expansion_count = torch::zeros({1}, 
        torch::TensorOptions().dtype(torch::kInt32).device(current_nodes.device()));
    
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
    
    cudaDeviceSynchronize();
    
    // Return only the valid expansion nodes
    int count = expansion_count.cpu().item<int>();
    if (count > 0) {
        expansion_nodes = expansion_nodes.slice(0, 0, count);
    } else {
        expansion_nodes = torch::empty({0}, expansion_nodes.options());
    }
    
    return std::make_tuple(expansion_nodes, expansion_count);
}

// Wrapper for batch processing legal moves
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> batch_process_legal_moves_cuda(
    torch::Tensor raw_policies,
    torch::Tensor board_states,
    int num_states,
    int action_size
) {
    auto normalized_priors = torch::zeros_like(raw_policies);
    auto legal_move_indices = torch::full({num_states * action_size}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(raw_policies.device()));
    auto legal_move_counts = torch::zeros({num_states},
        torch::TensorOptions().dtype(torch::kInt32).device(raw_policies.device()));
    
    const int threads = 256;
    const int blocks = num_states;
    
    batch_process_legal_moves_kernel<<<blocks, threads>>>(
        raw_policies.data_ptr<float>(),
        board_states.data_ptr<int>(),
        normalized_priors.data_ptr<float>(),
        legal_move_indices.data_ptr<int>(),
        legal_move_counts.data_ptr<int>(),
        num_states,
        action_size
    );
    
    cudaDeviceSynchronize();
    
    return std::make_tuple(normalized_priors, legal_move_indices, legal_move_counts);
}

// ============================================================================
// Additional Quantum Kernels
// ============================================================================

// Optimized kernel for MinHash computation
__global__ void compute_minhash_signatures_kernel(
    const int* __restrict__ paths,          // [batch_size, path_length]
    int* __restrict__ signatures,           // [batch_size, num_hashes]
    const int batch_size,
    const int path_length,
    const int num_hashes
) {
    // Use 2D grid: x for batch, y for hash function
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hash_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || hash_idx >= num_hashes) return;
    
    // Fixed primes for hash functions
    const int primes[16] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53};
    const int hash_mod = 10007;
    const int prime = primes[hash_idx % 16];
    
    // Compute minimum hash for this path and hash function
    int min_hash = INT_MAX;
    
    for (int i = 0; i < path_length; i++) {
        int elem = paths[batch_idx * path_length + i];
        if (elem >= 0) {  // Valid element
            int hash_val = (prime * elem + prime * 7919) % hash_mod;
            min_hash = min(min_hash, hash_val);
        }
    }
    
    signatures[batch_idx * num_hashes + hash_idx] = min_hash;
}

// Kernel to compute similarities in parallel
__global__ void compute_similarities_kernel(
    const int* __restrict__ signatures,     // [batch_size, num_hashes]
    float* __restrict__ similarities,       // [batch_size, batch_size]
    const int batch_size,
    const int num_hashes
) {
    // Use 2D grid for all pairs
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= batch_size || j >= batch_size) return;
    
    if (i == j) {
        similarities[i * batch_size + j] = 1.0f;
        return;
    }
    
    // Count matching signatures
    int matches = 0;
    for (int h = 0; h < num_hashes; h++) {
        if (signatures[i * num_hashes + h] == signatures[j * num_hashes + h]) {
            matches++;
        }
    }
    
    similarities[i * batch_size + j] = (float)matches / num_hashes;
}

// Kernel to apply interference in parallel
__global__ void apply_interference_kernel(
    const float* __restrict__ similarities, // [batch_size, batch_size]
    const float* __restrict__ scores,       // [batch_size]
    float* __restrict__ new_scores,         // [batch_size]
    const int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    // Compute interference as dot product of similarity row and scores
    float interference = 0.0f;
    for (int j = 0; j < batch_size; j++) {
        interference += similarities[tid * batch_size + j] * scores[j];
    }
    // Subtract self-contribution
    interference -= scores[tid];
    
    // Apply destructive interference
    new_scores[tid] = scores[tid] - 0.1f * interference;
}

__global__ void phase_kicked_policy_kernel(
    const float* __restrict__ priors,       // [batch_size, num_actions]
    const int* __restrict__ visits,         // [batch_size, num_actions]
    const float* __restrict__ values,       // [batch_size, num_actions]
    float* __restrict__ kicked_policy,      // [batch_size, num_actions]
    float* __restrict__ uncertainty,        // [batch_size, num_actions]
    float* __restrict__ phases,             // [batch_size, num_actions]
    const int batch_size,
    const int num_actions,
    const float kick_strength,
    const int64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_actions) return;
    
    int batch_idx = idx / num_actions;
    int action_idx = idx % num_actions;
    
    // Compute uncertainty
    float visit_count = (float)visits[idx];
    float uncert = 1.0f / sqrtf(visit_count + 1.0f);
    uncertainty[idx] = uncert;
    
    // Generate phase kick (simplified - should use curand for production)
    unsigned int local_seed = seed + idx;
    local_seed = local_seed * 1664525u + 1013904223u;
    float rand_val = (local_seed & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    float phase = kick_strength * uncert * (2.0f * rand_val - 1.0f);
    phases[idx] = phase;
    
    // Apply phase kick
    kicked_policy[idx] = priors[idx] * (1.0f + kick_strength * cosf(phase));
}

// Wrapper functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_minhash_interference_cuda(
    torch::Tensor paths,
    torch::Tensor scores,
    int num_hashes
) {
    const int batch_size = paths.size(0);
    const int path_length = paths.size(1);
    
    auto signatures = torch::zeros({batch_size, num_hashes}, 
        torch::TensorOptions().dtype(torch::kInt32).device(paths.device()));
    auto similarities = torch::zeros({batch_size, batch_size},
        torch::TensorOptions().dtype(torch::kFloat32).device(paths.device()));
    auto new_scores = torch::zeros_like(scores);
    
    // Step 1: Compute MinHash signatures in parallel
    {
        dim3 threads(256);
        dim3 blocks((batch_size + threads.x - 1) / threads.x, num_hashes);
        compute_minhash_signatures_kernel<<<blocks, threads>>>(
            paths.data_ptr<int>(),
            signatures.data_ptr<int>(),
            batch_size,
            path_length,
            num_hashes
        );
    }
    
    // Step 2: Compute similarities in parallel
    {
        dim3 threads(16, 16);
        dim3 blocks((batch_size + threads.x - 1) / threads.x,
                   (batch_size + threads.y - 1) / threads.y);
        compute_similarities_kernel<<<blocks, threads>>>(
            signatures.data_ptr<int>(),
            similarities.data_ptr<float>(),
            batch_size,
            num_hashes
        );
    }
    
    // Step 3: Apply interference
    {
        const int threads = 256;
        const int blocks = (batch_size + threads - 1) / threads;
        apply_interference_kernel<<<blocks, threads>>>(
            similarities.data_ptr<float>(),
            scores.data_ptr<float>(),
            new_scores.data_ptr<float>(),
            batch_size
        );
    }
    
    cudaDeviceSynchronize();
    
    return std::make_tuple(signatures, similarities, new_scores);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> phase_kicked_policy_cuda(
    torch::Tensor priors,
    torch::Tensor visits,
    torch::Tensor values,
    float kick_strength
) {
    const int batch_size = priors.size(0);
    const int num_actions = priors.size(1);
    const int total_elements = batch_size * num_actions;
    
    auto kicked_policy = torch::zeros_like(priors);
    auto uncertainty = torch::zeros_like(priors);
    auto phases = torch::zeros_like(priors);
    
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;
    
    // Random seed for phase generation
    int64_t seed = time(nullptr);
    
    phase_kicked_policy_kernel<<<blocks, threads>>>(
        priors.data_ptr<float>(),
        visits.data_ptr<int>(),
        values.data_ptr<float>(),
        kicked_policy.data_ptr<float>(),
        uncertainty.data_ptr<float>(),
        phases.data_ptr<float>(),
        batch_size,
        num_actions,
        kick_strength,
        seed
    );
    
    // Normalize kicked_policy
    auto sums = kicked_policy.sum(1, true);
    kicked_policy = kicked_policy / (sums + 1e-8f);
    
    return std::make_tuple(kicked_policy, uncertainty, phases);
}

// ============================================================================
// C++ Wrapper Functions for New Kernels
// ============================================================================

void batch_apply_moves_cuda(
    torch::Tensor boards,
    torch::Tensor current_players,
    torch::Tensor move_counts,
    torch::Tensor move_history,
    torch::Tensor state_indices,
    torch::Tensor actions,
    int game_type
) {
    const int batch_size = state_indices.size(0);
    const int board_size = boards.size(1);  // Assuming square board
    const int history_size = move_history.size(1);
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    batch_apply_moves_kernel<<<blocks, threads>>>(
        boards.data_ptr<int8_t>(),
        current_players.data_ptr<int8_t>(),
        move_counts.data_ptr<int16_t>(),
        move_history.data_ptr<int16_t>(),
        state_indices.data_ptr<int>(),
        actions.data_ptr<int>(),
        batch_size,
        board_size,
        history_size,
        game_type
    );
    
    cudaDeviceSynchronize();
}

torch::Tensor generate_legal_moves_mask_cuda(
    torch::Tensor boards,
    torch::Tensor metadata,
    torch::Tensor state_indices,
    int action_space_size,
    int game_type
) {
    const int batch_size = state_indices.size(0);
    const int board_size = boards.size(1);  // Assuming square board
    
    // Allocate output tensors
    auto legal_mask = torch::zeros({batch_size, action_space_size}, 
                                  torch::dtype(torch::kBool).device(boards.device()));
    auto move_counts = torch::zeros({batch_size}, 
                                   torch::dtype(torch::kInt32).device(boards.device()));
    
    // Launch kernel with grid-stride pattern
    const int threads = 256;
    const int total_work = batch_size * action_space_size;
    const int blocks = std::min((total_work + threads - 1) / threads, 65535);
    
    generate_legal_moves_mask_kernel<<<blocks, threads>>>(
        boards.data_ptr<int8_t>(),
        metadata.data_ptr<int8_t>(),
        legal_mask.data_ptr<bool>(),
        move_counts.data_ptr<int>(),
        state_indices.data_ptr<int>(),
        batch_size,
        board_size,
        action_space_size,
        game_type
    );
    
    cudaDeviceSynchronize();
    
    return legal_mask;
}

// ============================================================================
// Python Bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_ucb_selection", &batched_ucb_selection_cuda, 
          "Batched UCB selection with random tie-breaking (CUDA)");
    m.def("batched_ucb_selection_quantum", &batched_ucb_selection_quantum_cuda,
          "Quantum-enhanced batched UCB selection (CUDA)");
    m.def("parallel_backup", &parallel_backup_cuda, 
          "Parallel backup for MCTS (CUDA)");
    m.def("quantum_interference", &quantum_interference_cuda, 
          "Quantum interference for UCB (CUDA)");
    m.def("batched_add_children", &batched_add_children_cuda,
          "Batched add children to tree nodes (CUDA)");
    m.def("evaluate_gomoku_positions", &evaluate_gomoku_positions_cuda, 
          "Evaluate Gomoku board positions (CUDA)");
    m.def("find_expansion_nodes", &find_expansion_nodes_cuda,
          "Find nodes needing expansion in parallel (CUDA)");
    m.def("batch_process_legal_moves", &batch_process_legal_moves_cuda,
          "Batch process legal moves and normalize priors (CUDA)");
    m.def("fused_minhash_interference", &fused_minhash_interference_cuda,
          "Fused MinHash signature computation with interference (CUDA)");
    m.def("phase_kicked_policy", &phase_kicked_policy_cuda,
          "Apply phase kicks to policy based on uncertainty (CUDA)");
    m.def("batch_apply_moves", &batch_apply_moves_cuda,
          "Apply moves to game states in batch (CUDA)");
    m.def("generate_legal_moves_mask", &generate_legal_moves_mask_cuda,
          "Generate legal moves mask for multiple states (CUDA)");
}