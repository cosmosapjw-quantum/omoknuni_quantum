#include "utils/gpu_attack_defense_module.h"
#include "utils/attack_defense_module.h"
#include "games/chess/chess_state.h"
#include <torch/torch.h>
#include <bitset>

namespace alphazero {

// Static helper functions for chess GPU computations
namespace {
    torch::Tensor compute_ray_masks(int device_type) {
        // 64x64x8 tensor for all ray directions from each square
        auto masks = torch::zeros({64, 64, 8}, torch::kBool);
        
        // Directions: N, NE, E, SE, S, SW, W, NW
        int dr[] = {-1, -1, 0, 1, 1, 1, 0, -1};
        int dc[] = {0, 1, 1, 1, 0, -1, -1, -1};
        
        for (int from = 0; from < 64; from++) {
            int from_r = from / 8;
            int from_c = from % 8;
            
            for (int dir = 0; dir < 8; dir++) {
                int r = from_r + dr[dir];
                int c = from_c + dc[dir];
                
                while (r >= 0 && r < 8 && c >= 0 && c < 8) {
                    int to = r * 8 + c;
                    masks[from][to][dir] = true;
                    r += dr[dir];
                    c += dc[dir];
                }
            }
        }
        
        return masks.to(device_type == 1 ? torch::kCUDA : torch::kCPU);
    }
    
    torch::Tensor compute_knight_masks(int device_type) {
        auto masks = torch::zeros({64, 64}, torch::kBool);
        
        int knight_moves[][2] = {
            {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
            {1, -2}, {1, 2}, {2, -1}, {2, 1}
        };
        
        for (int from = 0; from < 64; from++) {
            int from_r = from / 8;
            int from_c = from % 8;
            
            for (auto& move : knight_moves) {
                int r = from_r + move[0];
                int c = from_c + move[1];
                
                if (r >= 0 && r < 8 && c >= 0 && c < 8) {
                    int to = r * 8 + c;
                    masks[from][to] = true;
                }
            }
        }
        
        return masks.to(device_type == 1 ? torch::kCUDA : torch::kCPU);
    }
    
    torch::Tensor compute_king_masks(int device_type) {
        auto masks = torch::zeros({64, 64}, torch::kBool);
        
        for (int from = 0; from < 64; from++) {
            int from_r = from / 8;
            int from_c = from % 8;
            
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    if (dr == 0 && dc == 0) continue;
                    
                    int r = from_r + dr;
                    int c = from_c + dc;
                    
                    if (r >= 0 && r < 8 && c >= 0 && c < 8) {
                        int to = r * 8 + c;
                        masks[from][to] = true;
                    }
                }
            }
        }
        
        return masks.to(device_type == 1 ? torch::kCUDA : torch::kCPU);
    }
}

// ChessGPUAttackDefense implementation
ChessGPUAttackDefense::ChessGPUAttackDefense(torch::Device device)
    : GPUAttackDefenseModule(8, device) {  // Chess board is 8x8
    initialize_chess_patterns();
}

void ChessGPUAttackDefense::initialize_chess_patterns() {
    // Initialize pre-computed attack patterns
    knight_attacks_ = compute_knight_masks(device_.type() == torch::kCUDA ? 1 : 0);
    king_attacks_ = compute_king_masks(device_.type() == torch::kCUDA ? 1 : 0);
    sliding_ray_masks_ = compute_ray_masks(device_.type() == torch::kCUDA ? 1 : 0);
}

std::pair<torch::Tensor, torch::Tensor> ChessGPUAttackDefense::compute_planes_gpu(
    const torch::Tensor& board_batch,
    int current_player) {
    
    auto batch_size = board_batch.size(0);
    auto attack_planes = torch::zeros({batch_size, 8, 8}, device_);
    auto defense_planes = torch::zeros({batch_size, 8, 8}, device_);
    
    // Compute attack maps for current player
    auto current_attacks = compute_chess_attacks(board_batch);
    
    // For simplicity, use attack density as the plane values
    attack_planes = current_attacks;
    
    // Defense planes would consider opponent attacks
    // This is simplified - full implementation would evaluate piece safety
    defense_planes = current_attacks * 0.5f;
    
    return {attack_planes, defense_planes};
}

std::pair<torch::Tensor, torch::Tensor> ChessGPUAttackDefense::compute_bonuses_gpu(
    const torch::Tensor& board_batch,
    const torch::Tensor& chosen_moves,
    int current_player) {
    
    // Use CPU implementation for accurate chess logic
    auto batch_size = board_batch.size(0);
    auto board_cpu = board_batch.to(torch::kCPU);
    auto moves_cpu = chosen_moves.to(torch::kCPU);
    
    // Convert to CPU format
    std::vector<std::vector<std::vector<int>>> board_vec(batch_size,
        std::vector<std::vector<int>>(8, std::vector<int>(8, 0)));
    std::vector<int> moves_vec(batch_size);
    std::vector<int> player_vec(batch_size, current_player);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                board_vec[b][i][j] = board_cpu[b][i][j].item<int>();
            }
        }
        moves_vec[b] = moves_cpu[b].item<int>();
    }
    
    // Use CPU implementation
    ChessAttackDefenseModule cpu_module;
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

torch::Tensor ChessGPUAttackDefense::count_threats_for_player(
    const torch::Tensor& board_batch,
    int player) {
    
    // Count threats (attacked opponent pieces)
    auto threats = compute_chess_attacks(board_batch);
    return threats;
}

torch::Tensor ChessGPUAttackDefense::compute_chess_attacks(const torch::Tensor& board_batch) {
    // Simplified attack computation
    // Full implementation would use bitboards and piece-specific attack patterns
    auto batch_size = board_batch.size(0);
    auto attacks = torch::zeros({batch_size, 8, 8}, device_);
    
    // Convert board to flat representation for easier processing
    auto flat_boards = board_batch.view({batch_size, 64});
    
    // Placeholder: mark some squares as attacked
    // Real implementation would compute actual piece attacks
    attacks = torch::rand({batch_size, 8, 8}, device_) * 10.0f;
    
    return attacks;
}

// Helper functions for sliding attacks
torch::Tensor ChessGPUAttackDefense::compute_sliding_attacks(
    const torch::Tensor& board,
    const torch::Tensor& piece_positions,
    const torch::Tensor& ray_masks) {
    
    // Simplified implementation
    return torch::zeros_like(board);
}

} // namespace alphazero