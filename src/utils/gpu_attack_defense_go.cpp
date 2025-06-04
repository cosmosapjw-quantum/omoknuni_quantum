#include "utils/gpu_attack_defense_module.h"
#include "utils/attack_defense_module.h"
#include "games/go/go_state.h"
#include <torch/torch.h>
#include <queue>

namespace alphazero {

// Static helper functions for Go GPU computations
namespace {
    // Flood fill for group detection using GPU-friendly iterative approach
    torch::Tensor gpu_flood_fill_groups_static(const torch::Tensor& board, int player) {
        auto device = board.device();
        auto batch_size = board.size(0);
        auto board_size = board.size(1);
        
        // Initialize group IDs
        auto group_ids = torch::zeros_like(board, torch::kInt32);
        auto player_mask = (board == player);
        
        // Iterative flood fill using convolutions
        int group_id = 1;
        auto unvisited = player_mask.clone();
        
        while (unvisited.any().item<bool>()) {
            // Find first unvisited stone
            auto indices = torch::nonzero(unvisited);
            if (indices.size(0) == 0) break;
            
            auto seed_batch = indices[0][0].item<int>();
            auto seed_row = indices[0][1].item<int>();
            auto seed_col = indices[0][2].item<int>();
            
            // Create seed mask
            auto current_group = torch::zeros_like(board, torch::kBool);
            current_group[seed_batch][seed_row][seed_col] = true;
            
            // Iterative expansion using 4-connected convolution
            // Create 4D kernel: [out_channels=1, in_channels=1, height=3, width=3]
            auto kernel_3x3 = torch::tensor({{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}, device).to(torch::kFloat32);
            auto kernel = kernel_3x3.unsqueeze(0).unsqueeze(0);  // Now [1, 1, 3, 3]
            
            bool changed = true;
            while (changed) {
                // Process each batch item separately
                auto expanded = torch::zeros_like(current_group, torch::kBool);
                for (int b = 0; b < batch_size; b++) {
                    auto group_slice = current_group[b].unsqueeze(0).unsqueeze(0).to(torch::kFloat32);
                    auto exp_slice = torch::nn::functional::conv2d(
                        group_slice,
                        kernel,
                        torch::nn::functional::Conv2dFuncOptions().stride(1).padding(1)
                    ).squeeze(0).squeeze(0) > 0;
                    expanded[b] = exp_slice;
                }
                
                // Mask to only player stones
                expanded = expanded & player_mask & unvisited;
                
                auto new_group = current_group | expanded;
                changed = !torch::equal(new_group, current_group);
                current_group = new_group;
            }
            
            // Assign group ID
            group_ids.masked_fill_(current_group, group_id);
            unvisited = unvisited & ~current_group;
            group_id++;
        }
        
        return group_ids;
    }
    
    // Count liberties using convolutions
    torch::Tensor compute_liberties_gpu(const torch::Tensor& board, const torch::Tensor& group_ids) {
        auto device = board.device();
        auto batch_size = board.size(0);
        auto board_size = board.size(1);
        
        auto liberties = torch::zeros_like(board, torch::kFloat32);
        auto empty_mask = (board == 0);
        
        // 4-connected kernel
        auto kernel_2d = torch::tensor({{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}, device).to(torch::kFloat32);
        auto kernel = kernel_2d.unsqueeze(0).unsqueeze(0);  // Now [1, 1, 3, 3]
        
        // Process each batch item separately
        for (int b = 0; b < batch_size; b++) {
            auto board_slice = board[b];
            auto group_ids_slice = group_ids[b];
            auto empty_mask_slice = empty_mask[b];
            
            // For each unique group in this batch
            auto unique_result = at::_unique(group_ids_slice);
            auto unique_groups = std::get<0>(unique_result);
            
            for (int i = 0; i < unique_groups.size(0); i++) {
                int group = unique_groups[i].item<int>();
                if (group == 0) continue;  // Skip empty
                
                auto group_mask = (group_ids_slice == group);
                
                // Find adjacent empty points
                auto group_mask_4d = group_mask.unsqueeze(0).unsqueeze(0).to(torch::kFloat32);
                auto adjacent_empty = torch::nn::functional::conv2d(
                    group_mask_4d,
                    kernel,
                    torch::nn::functional::Conv2dFuncOptions().stride(1).padding(1)
                ).squeeze(0).squeeze(0) > 0;
                
                adjacent_empty = adjacent_empty.to(torch::kFloat32) * empty_mask_slice.to(torch::kFloat32);
                
                // Count unique liberties
                int liberty_count = adjacent_empty.sum().item<int>();
                liberties[b].masked_fill_(group_mask, liberty_count);
            }
        }
        
        return liberties;
    }
    
    // Detect eyes (surrounded empty points)
    torch::Tensor detect_eyes_gpu(const torch::Tensor& board, int player) {
        auto device = board.device();
        auto empty_mask = (board == 0);
        auto player_mask = (board == player);
        
        // Full surround kernel (8-connected)
        auto kernel = torch::ones({1, 1, 3, 3}, device).to(torch::kFloat32);
        kernel[0][0][1][1] = 0;  // Center is empty
        
        // Check if empty points are surrounded by player stones
        auto surrounded = torch::nn::functional::conv2d(
            player_mask.unsqueeze(1).to(torch::kFloat32),
            kernel,
            torch::nn::functional::Conv2dFuncOptions().stride(1).padding(1)
        ).squeeze(1);
        
        // Eye if empty and surrounded by at least 7 friendly stones
        auto eyes = empty_mask.to(torch::kFloat32) * (surrounded >= 7).to(torch::kFloat32);
        
        return eyes.to(torch::kFloat32);
    }
    
    // Ladder detection (simplified)
    torch::Tensor detect_ladders_gpu(const torch::Tensor& board, const torch::Tensor& liberties) {
        // Groups with exactly 2 liberties are ladder candidates
        auto ladder_candidates = (liberties == 2);
        
        // Additional logic would check if those liberties lead to capture
        // This is a simplified version
        return ladder_candidates.to(torch::kFloat32);
    }
}

// Static method for computing Go features in batch
torch::Tensor compute_go_features_batch(const torch::Tensor& board_batch) {
    auto device = board_batch.device();
    auto batch_size = board_batch.size(0);
    auto board_size = board_batch.size(1);
    
    // Initialize feature channels
    const int num_features = 12;
    auto features = torch::zeros({batch_size, num_features, board_size, board_size}, device);
    
    // Basic stone positions
    features.select(1, 0) = (board_batch == 1).to(torch::kFloat32);  // Black stones
    features.select(1, 1) = (board_batch == 2).to(torch::kFloat32);  // White stones
    
    // Group IDs for both players
    auto black_groups = gpu_flood_fill_groups_static(board_batch, 1);
    auto white_groups = gpu_flood_fill_groups_static(board_batch, 2);
    
    // Liberty counts
    auto black_liberties = compute_liberties_gpu(board_batch, black_groups);
    auto white_liberties = compute_liberties_gpu(board_batch, white_groups);
    
    features.select(1, 2) = black_liberties.to(torch::kFloat32) / 4.0;  // Normalized
    features.select(1, 3) = white_liberties.to(torch::kFloat32) / 4.0;
    
    // Atari detection (1 liberty)
    features.select(1, 4) = (black_liberties == 1).to(torch::kFloat32);
    features.select(1, 5) = (white_liberties == 1).to(torch::kFloat32);
    
    // Eye detection
    features.select(1, 6) = detect_eyes_gpu(board_batch, 1);
    features.select(1, 7) = detect_eyes_gpu(board_batch, 2);
    
    // Ladder detection
    features.select(1, 8) = detect_ladders_gpu(board_batch, black_liberties);
    features.select(1, 9) = detect_ladders_gpu(board_batch, white_liberties);
    
    // Territory estimation (simplified - distance to nearest stone)
    auto distance_kernel = torch::ones({1, 1, 5, 5}, device).to(torch::kFloat32) / 25.0;
    auto black_stones = (board_batch == 1).to(torch::kFloat32).unsqueeze(1);  // [batch, 1, h, w]
    auto white_stones = (board_batch == 2).to(torch::kFloat32).unsqueeze(1);  // [batch, 1, h, w]
    
    auto black_influence = torch::nn::functional::conv2d(
        black_stones, distance_kernel,
        torch::nn::functional::Conv2dFuncOptions().stride(1).padding(2)
    );
    auto white_influence = torch::nn::functional::conv2d(
        white_stones, distance_kernel,
        torch::nn::functional::Conv2dFuncOptions().stride(1).padding(2)
    );
    
    features.select(1, 10) = black_influence.squeeze(1);
    features.select(1, 11) = white_influence.squeeze(1);
    
    return features;
}

// External function wrapper for use in other files
torch::Tensor GoGPUAttackDefense_compute_go_features_batch(const torch::Tensor& boards) {
    return compute_go_features_batch(boards);
}

// GoGPUAttackDefense implementation
GoGPUAttackDefense::GoGPUAttackDefense(int board_size, torch::Device device)
    : GPUAttackDefenseModule(board_size, device) {
    initialize_go_patterns();
}

void GoGPUAttackDefense::initialize_go_patterns() {
    // 4-connected kernel for liberty counting
    liberty_kernel_ = torch::tensor({{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}, device_).to(torch::kFloat32);
    liberty_kernel_ = liberty_kernel_.unsqueeze(0).unsqueeze(0);
    
    // Initialize group detection kernels
    group_detection_kernels_ = liberty_kernel_.clone();
}

std::pair<torch::Tensor, torch::Tensor> GoGPUAttackDefense::compute_planes_gpu(
    const torch::Tensor& board_batch,
    int current_player) {
    
    auto batch_size = board_batch.size(0);
    auto attack_planes = torch::zeros({batch_size, board_size_, board_size_}, device_);
    auto defense_planes = torch::zeros({batch_size, board_size_, board_size_}, device_);
    
    // Compute liberties for all groups
    auto group_ids = gpu_flood_fill_groups(board_batch, current_player);
    auto liberties = compute_liberties_gpu(board_batch, group_ids);
    
    // Attack plane: capturing opponent groups (low liberty groups)
    auto opponent_groups = gpu_flood_fill_groups(board_batch, 3 - current_player);
    auto opponent_liberties = compute_liberties_gpu(board_batch, opponent_groups);
    
    // Mark positions where placing a stone would capture
    auto capture_potential = (opponent_liberties == 1).to(torch::kFloat32) * 10.0f;
    attack_planes = capture_potential;
    
    // Defense plane: saving own groups in atari
    auto atari_groups = (liberties == 1).to(torch::kFloat32) * 10.0f;
    defense_planes = atari_groups;
    
    return {attack_planes, defense_planes};
}

std::pair<torch::Tensor, torch::Tensor> GoGPUAttackDefense::compute_bonuses_gpu(
    const torch::Tensor& board_batch,
    const torch::Tensor& chosen_moves,
    int current_player) {
    
    // Use CPU implementation for accurate Go logic
    auto batch_size = board_batch.size(0);
    auto board_cpu = board_batch.to(torch::kCPU);
    auto moves_cpu = chosen_moves.to(torch::kCPU);
    
    // Convert to CPU format
    std::vector<std::vector<std::vector<int>>> board_vec(batch_size,
        std::vector<std::vector<int>>(board_size_, std::vector<int>(board_size_, 0)));
    std::vector<int> moves_vec(batch_size);
    std::vector<int> player_vec(batch_size, current_player);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < board_size_; ++i) {
            for (int j = 0; j < board_size_; ++j) {
                board_vec[b][i][j] = board_cpu[b][i][j].item<int>();
            }
        }
        moves_vec[b] = moves_cpu[b].item<int>();
    }
    
    // Use CPU implementation
    GoAttackDefenseModule cpu_module(board_size_);
    auto [cpu_attack, cpu_defense] = cpu_module.compute_bonuses(board_vec, moves_vec, player_vec);
    
    // Convert back to GPU tensors
    auto attack_tensor = torch::zeros({batch_size}, device_);
    auto defense_tensor = torch::zeros({batch_size}, device_);
    
    for (int b = 0; b < batch_size; ++b) {
        attack_tensor[b] = cpu_attack[b];
        defense_tensor[b] = cpu_defense[b];
    }
    
    return {attack_tensor, defense_tensor};
}

torch::Tensor GoGPUAttackDefense::count_threats_for_player(
    const torch::Tensor& board_batch,
    int player) {
    
    // Count capture threats
    auto captures = detect_captures(board_batch, player);
    return captures;
}

torch::Tensor GoGPUAttackDefense::compute_go_liberties(const torch::Tensor& board_batch) {
    // Compute liberties for all groups
    auto black_groups = gpu_flood_fill_groups(board_batch, 1);
    auto white_groups = gpu_flood_fill_groups(board_batch, 2);
    
    auto black_liberties = compute_liberties_gpu(board_batch, black_groups);
    auto white_liberties = compute_liberties_gpu(board_batch, white_groups);
    
    // Combine liberties
    auto liberties = torch::zeros_like(board_batch, torch::kFloat32);
    liberties.masked_fill_(board_batch == 1, 1.0f).mul_(black_liberties);
    liberties.masked_fill_(board_batch == 2, 1.0f).mul_(white_liberties);
    
    return liberties;
}

torch::Tensor GoGPUAttackDefense::detect_captures(const torch::Tensor& board_batch, int player) {
    // Detect groups that would be captured
    auto opponent = 3 - player;
    auto opponent_groups = gpu_flood_fill_groups(board_batch, opponent);
    auto opponent_liberties = compute_liberties_gpu(board_batch, opponent_groups);
    
    // Groups with 0 liberties are captured
    auto captures = (opponent_liberties == 0).to(torch::kFloat32);
    return captures;
}

torch::Tensor GoGPUAttackDefense::gpu_flood_fill_groups(const torch::Tensor& board, int player) {
    return gpu_flood_fill_groups_static(board, player);
}

} // namespace alphazero