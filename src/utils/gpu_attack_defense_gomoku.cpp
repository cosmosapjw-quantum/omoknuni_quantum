#include "utils/gpu_attack_defense_module.h"
#include "utils/attack_defense_module.h"
#include "games/gomoku/gomoku_state.h"
#include "core/igamestate.h"
#include <torch/torch.h>

namespace alphazero {

// Implementation of GomokuGPUAttackDefense
GomokuGPUAttackDefense::GomokuGPUAttackDefense(int board_size, torch::Device device)
    : GPUAttackDefenseModule(board_size, device) {
    initialize_gomoku_patterns();
}

void GomokuGPUAttackDefense::initialize_gomoku_patterns() {
    // Initialize pattern detection kernels for convolution
    // These will be used to detect patterns efficiently on GPU
    
    // Create kernels for different pattern lengths (2, 3, 4, 5)
    for (int len = 2; len <= 5; ++len) {
        // Horizontal kernel: [1, 1, ..., 1] in shape [1, 1, 1, len]
        auto h_kernel = torch::ones({1, 1, 1, len}, device_);
        
        // Vertical kernel: [1, 1, ..., 1]^T in shape [1, 1, len, 1]
        auto v_kernel = torch::ones({1, 1, len, 1}, device_);
        
        // Diagonal kernel: identity matrix in shape [1, 1, len, len]
        auto d_kernel = torch::eye(len, device_).unsqueeze(0).unsqueeze(0);
        
        // Anti-diagonal kernel: flipped identity
        auto a_kernel = torch::flip(d_kernel, {3});
        
        pattern_kernels_[len] = {h_kernel, v_kernel, d_kernel, a_kernel};
    }
}

std::pair<torch::Tensor, torch::Tensor> GomokuGPUAttackDefense::compute_planes_gpu(
    const torch::Tensor& board_batch,
    int current_player) {
    
    // For compute_planes, we need to match CPU logic exactly
    // The most efficient approach is to delegate to CPU implementation
    // and only use GPU for batch operations where it provides benefit
    
    auto batch_size = board_batch.size(0);
    
    // Transfer to CPU for computation
    auto board_cpu = board_batch.to(torch::kCPU);
    
    // For compute_planes, we cannot use IGameState directly because we don't have move history
    // Instead, directly use the board representation with the CPU module's logic
    
    // Initialize result planes  
    std::vector<std::vector<std::vector<float>>> attack_planes_vec(
        batch_size, std::vector<std::vector<float>>(
            board_size_, std::vector<float>(board_size_, 0.0f)));
    
    std::vector<std::vector<std::vector<float>>> defense_planes_vec(
        batch_size, std::vector<std::vector<float>>(
            board_size_, std::vector<float>(board_size_, 0.0f)));
    
    // Process each board
    for (int b = 0; b < batch_size; ++b) {
        // Convert single board to vector format
        std::vector<std::vector<std::vector<int>>> single_board_batch(1);
        single_board_batch[0].resize(board_size_, std::vector<int>(board_size_, 0));
        
        for (int i = 0; i < board_size_; ++i) {
            for (int j = 0; j < board_size_; ++j) {
                single_board_batch[0][i][j] = board_cpu[b][i][j].item<int>();
            }
        }
        
        // Determine the actual current player by counting stones
        int player1_count = 0;
        int player2_count = 0;
        for (int i = 0; i < board_size_; ++i) {
            for (int j = 0; j < board_size_; ++j) {
                if (single_board_batch[0][i][j] == 1) player1_count++;
                else if (single_board_batch[0][i][j] == 2) player2_count++;
            }
        }
        
        // In Gomoku, player 1 goes first, so if counts are equal, it's player 1's turn
        int actual_current_player = (player1_count > player2_count) ? 2 : 1;
        std::vector<int> single_player_batch = {actual_current_player};
        
        // For each empty position, compute attack/defense values
        for (int row = 0; row < board_size_; ++row) {
            for (int col = 0; col < board_size_; ++col) {
                if (single_board_batch[0][row][col] == 0) {
                    // Place the move
                    auto board_with_move = single_board_batch[0];
                    board_with_move[row][col] = actual_current_player;
                    
                    std::vector<std::vector<std::vector<int>>> board_batch_with_move = {board_with_move};
                    std::vector<int> single_move = {row * board_size_ + col};
                    
                    // Use CPU module to compute bonuses
                    GomokuAttackDefenseModule cpu_module(board_size_);
                    auto [attack_scores, defense_scores] = cpu_module.compute_bonuses(
                        board_batch_with_move, single_move, single_player_batch);
                    
                    attack_planes_vec[b][row][col] = attack_scores[0];
                    defense_planes_vec[b][row][col] = defense_scores[0];
                }
            }
        }
    }
    
    // Convert back to tensors on GPU
    auto attack_planes = torch::zeros({batch_size, board_size_, board_size_}, device_);
    auto defense_planes = torch::zeros({batch_size, board_size_, board_size_}, device_);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < board_size_; ++i) {
            for (int j = 0; j < board_size_; ++j) {
                attack_planes[b][i][j] = attack_planes_vec[b][i][j];
                defense_planes[b][i][j] = defense_planes_vec[b][i][j];
            }
        }
    }
    
    return {attack_planes, defense_planes};
}

std::pair<torch::Tensor, torch::Tensor> GomokuGPUAttackDefense::compute_bonuses_gpu_batch(
    const torch::Tensor& board_batch,
    const torch::Tensor& chosen_moves,
    int current_player) {
    
    auto batch_size = board_batch.size(0);
    
    // Create boards without the moves (before state)
    auto board_before = board_batch.clone();
    auto move_rows = (chosen_moves / board_size_).to(torch::kLong);
    auto move_cols = (chosen_moves % board_size_).to(torch::kLong);
    
    // Remove the moves to get the "before" state
    auto batch_indices = torch::arange(batch_size, torch::TensorOptions().dtype(torch::kLong).device(device_));
    board_before.index_put_({batch_indices, move_rows, move_cols}, 0);
    
    // Compute threats using vectorized operations
    auto threats_after = compute_gomoku_threats_vectorized(board_batch, current_player);
    auto threats_before = compute_gomoku_threats_vectorized(board_before, current_player);
    auto attack_bonuses = threats_after - threats_before;
    
    // Compute defense bonus
    int opponent = 3 - current_player;
    auto opp_threats_after = compute_gomoku_threats_vectorized(board_batch, opponent);
    auto opp_threats_before = compute_gomoku_threats_vectorized(board_before, opponent);
    auto defense_bonuses = opp_threats_before - opp_threats_after;
    
    return {attack_bonuses, defense_bonuses};
}

std::pair<torch::Tensor, torch::Tensor> GomokuGPUAttackDefense::compute_bonuses_gpu(
    const torch::Tensor& board_batch,
    const torch::Tensor& chosen_moves,
    int current_player) {
    
    // For small batches, CPU is more efficient
    // Transfer to CPU and use the exact same logic as CPU implementation
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
    GomokuAttackDefenseModule cpu_module(board_size_);
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

torch::Tensor GomokuGPUAttackDefense::compute_gomoku_threats_vectorized(
    const torch::Tensor& board_batch, int player) {
    
    auto batch_size = board_batch.size(0);
    
    // Convert board to float and create masks
    auto board_float = board_batch.to(torch::kFloat32);
    auto player_mask = (board_batch == player).to(torch::kFloat32);
    auto empty_mask = (board_batch == 0).to(torch::kFloat32);
    auto opponent_mask = ((board_batch != player) & (board_batch != 0)).to(torch::kFloat32);
    
    // Add channel dimension for convolution
    player_mask = player_mask.unsqueeze(1);
    empty_mask = empty_mask.unsqueeze(1);
    opponent_mask = opponent_mask.unsqueeze(1);
    
    auto threat_counts = torch::zeros({batch_size}, device_);
    
    // Pattern weights
    const float WIN_WEIGHT = 100.0f;
    const float FOUR_WEIGHT = 20.0f;
    const float THREE_WEIGHT = 5.0f;
    const float TWO_WEIGHT = 1.0f;
    const float weights[] = {0, 0, TWO_WEIGHT, THREE_WEIGHT, FOUR_WEIGHT, WIN_WEIGHT};
    
    // Process each pattern length
    for (int len = 2; len <= 5; ++len) {
        float weight = weights[len];
        
        // Horizontal patterns
        auto h_conv = torch::nn::functional::conv2d(player_mask, pattern_kernels_[len][0], 
                                   torch::nn::functional::Conv2dFuncOptions()
                                   .stride({1, 1})
                                   .padding({0, 0}));
        auto h_empty = torch::nn::functional::conv2d(empty_mask, pattern_kernels_[len][0], 
                                    torch::nn::functional::Conv2dFuncOptions()
                                    .stride({1, 1})
                                    .padding({0, 0}));
        auto h_blocked = torch::nn::functional::conv2d(opponent_mask, pattern_kernels_[len][0], 
                                      torch::nn::functional::Conv2dFuncOptions()
                                      .stride({1, 1})
                                      .padding({0, 0}));
        
        // Count patterns with required number of stones
        auto h_perfect = (h_conv == len).to(torch::kFloat32);
        auto h_open = (h_conv == len - 1) & (h_empty == 1);  // Has one empty space
        auto h_semi = (h_conv == len) & (h_blocked == 0);    // Not blocked
        
        // For open patterns, check boundaries
        if (len >= 3) {
            // Pad the result to check boundaries
            auto padded_player = torch::constant_pad_nd(player_mask, {1, 1, 0, 0}, 0);
            auto padded_empty = torch::constant_pad_nd(empty_mask, {1, 1, 0, 0}, 0);
            
            // Check if ends are empty
            auto left_empty = padded_empty.index({torch::indexing::Slice(), 
                                                 torch::indexing::Slice(), 
                                                 torch::indexing::Slice(),
                                                 torch::indexing::Slice(0, -len-1)});
            auto right_empty = padded_empty.index({torch::indexing::Slice(), 
                                                  torch::indexing::Slice(), 
                                                  torch::indexing::Slice(),
                                                  torch::indexing::Slice(len+1, torch::indexing::None)});
            
            // Open patterns have empty spaces on both sides
            auto h_truly_open = h_perfect * left_empty.squeeze(1) * right_empty.squeeze(1);
            threat_counts += h_truly_open.sum({1, 2}) * weight * 2.0f;  // Double weight for open
            
            // Semi-open patterns
            auto h_semi_open = h_perfect * ((left_empty.squeeze(1) + right_empty.squeeze(1)) > 0);
            threat_counts += h_semi_open.sum({1, 2}) * weight;
        } else {
            threat_counts += h_perfect.sum({1, 2, 3}) * weight;
        }
        
        // Vertical patterns (similar logic)
        auto v_conv = torch::nn::functional::conv2d(player_mask, pattern_kernels_[len][1], 
                                   torch::nn::functional::Conv2dFuncOptions()
                                   .stride({1, 1})
                                   .padding({0, 0}));
        auto v_perfect = (v_conv == len).to(torch::kFloat32);
        threat_counts += v_perfect.sum({1, 2, 3}) * weight;
        
        // Diagonal patterns
        auto d_conv = torch::nn::functional::conv2d(player_mask, pattern_kernels_[len][2], 
                                   torch::nn::functional::Conv2dFuncOptions()
                                   .stride({1, 1})
                                   .padding({0, 0}));
        auto d_perfect = (d_conv == len).to(torch::kFloat32);
        threat_counts += d_perfect.sum({1, 2, 3}) * weight;
        
        // Anti-diagonal patterns
        auto a_conv = torch::nn::functional::conv2d(player_mask, pattern_kernels_[len][3], 
                                   torch::nn::functional::Conv2dFuncOptions()
                                   .stride({1, 1})
                                   .padding({0, 0}));
        auto a_perfect = (a_conv == len).to(torch::kFloat32);
        threat_counts += a_perfect.sum({1, 2, 3}) * weight;
    }
    
    return threat_counts;
}

torch::Tensor GomokuGPUAttackDefense::count_threats_for_player(
    const torch::Tensor& board_batch,
    int player) {
    return compute_gomoku_threats_vectorized(board_batch, player);
}

torch::Tensor GomokuGPUAttackDefense::compute_gomoku_threats(
    const torch::Tensor& board_batch, int player) {
    // Use the simple version
    return compute_gomoku_threats_simple(board_batch, player);
}

// Add helper method for pattern counting using convolutions
torch::Tensor GomokuGPUAttackDefense::compute_pattern_counts_vectorized(
    const torch::Tensor& board_batch, int player) {
    
    auto batch_size = board_batch.size(0);
    
    // Create player mask and add channel dimension
    auto player_mask = (board_batch == player).to(torch::kFloat32).unsqueeze(1);
    auto empty_mask = (board_batch == 0).to(torch::kFloat32).unsqueeze(1);
    
    // Initialize pattern count map for each position
    auto pattern_map = torch::zeros({batch_size, board_size_, board_size_}, device_);
    
    // Pattern weights
    const float weights[] = {0, 0, 1.0f, 5.0f, 20.0f, 100.0f};
    
    // Use convolutions to detect patterns efficiently
    for (int len = 2; len <= 5; ++len) {
        if (pattern_kernels_.find(len) == pattern_kernels_.end()) continue;
        
        float weight = weights[len];
        
        // Horizontal patterns
        auto h_conv = torch::nn::functional::conv2d(player_mask, pattern_kernels_[len][0],
                                                   torch::nn::functional::Conv2dFuncOptions()
                                                   .padding({0, 0}));
        pattern_map += torch::nn::functional::pad(h_conv.squeeze(1) * weight, 
                                                torch::nn::functional::PadFuncOptions({0, len-1, 0, 0}));
        
        // Vertical patterns
        auto v_conv = torch::nn::functional::conv2d(player_mask, pattern_kernels_[len][1],
                                                   torch::nn::functional::Conv2dFuncOptions()
                                                   .padding({0, 0}));
        pattern_map += torch::nn::functional::pad(v_conv.squeeze(1) * weight,
                                                torch::nn::functional::PadFuncOptions({0, 0, 0, len-1}));
        
        // Diagonal patterns (only for smaller kernels to avoid memory issues)
        if (len <= 3) {
            auto d_conv = torch::nn::functional::conv2d(player_mask, pattern_kernels_[len][2],
                                                       torch::nn::functional::Conv2dFuncOptions()
                                                       .padding({0, 0}));
            pattern_map += torch::nn::functional::pad(d_conv.squeeze(1) * weight,
                                                    torch::nn::functional::PadFuncOptions({0, len-1, 0, len-1}));
            
            auto a_conv = torch::nn::functional::conv2d(player_mask, pattern_kernels_[len][3],
                                                       torch::nn::functional::Conv2dFuncOptions()
                                                       .padding({0, 0}));
            pattern_map += torch::nn::functional::pad(a_conv.squeeze(1) * weight,
                                                    torch::nn::functional::PadFuncOptions({len-1, 0, 0, len-1}));
        }
    }
    
    return pattern_map;
}

torch::Tensor GomokuGPUAttackDefense::compute_gomoku_threats_simple(
    const torch::Tensor& board_batch, int player) {
    
    auto batch_size = board_batch.size(0);
    auto board_size = board_batch.size(1);
    
    // For simple scalar computation, transfer to CPU
    auto board_cpu = board_batch.to(torch::kCPU);
    auto threat_counts = torch::zeros({batch_size}, torch::kCPU);
    
    // Process each board in the batch on CPU
    for (int b = 0; b < batch_size; ++b) {
        auto board = board_cpu[b];
        float total_threat = 0.0f;
        
        // Count winning patterns (5 in a row)
        float wins = 0.0f;
        // Horizontal
        for (int i = 0; i < board_size; ++i) {
            for (int j = 0; j <= board_size - 5; ++j) {
                int count = 0;
                for (int k = 0; k < 5; ++k) {
                    if (board[i][j + k].item<int>() == player) count++;
                }
                if (count == 5) wins += 1.0f;
            }
        }
        // Vertical
        for (int i = 0; i <= board_size - 5; ++i) {
            for (int j = 0; j < board_size; ++j) {
                int count = 0;
                for (int k = 0; k < 5; ++k) {
                    if (board[i + k][j].item<int>() == player) count++;
                }
                if (count == 5) wins += 1.0f;
            }
        }
        // Diagonal
        for (int i = 0; i <= board_size - 5; ++i) {
            for (int j = 0; j <= board_size - 5; ++j) {
                int count = 0;
                for (int k = 0; k < 5; ++k) {
                    if (board[i + k][j + k].item<int>() == player) count++;
                }
                if (count == 5) wins += 1.0f;
            }
        }
        // Anti-diagonal
        for (int i = 0; i <= board_size - 5; ++i) {
            for (int j = 4; j < board_size; ++j) {
                int count = 0;
                for (int k = 0; k < 5; ++k) {
                    if (board[i + k][j - k].item<int>() == player) count++;
                }
                if (count == 5) wins += 1.0f;
            }
        }
        
        // Count patterns with proper open/semi-open/blocked logic
        auto count_patterns = [&](int pattern_length) -> float {
            float pattern_count = 0.0f;
            
            // Horizontal patterns
            for (int i = 0; i < board_size; ++i) {
                for (int j = 0; j <= board_size - pattern_length; ++j) {
                    int count = 0;
                    bool has_empty = false;
                    
                    for (int k = 0; k < pattern_length; ++k) {
                        int val = board[i][j + k].item<int>();
                        if (val == player) count++;
                        else if (val == 0) has_empty = true;
                    }
                    
                    if (count == pattern_length) {
                        bool left_empty = (j > 0 && board[i][j-1].item<int>() == 0);
                        bool right_empty = (j + pattern_length < board_size && 
                                          board[i][j + pattern_length].item<int>() == 0);
                        
                        if (left_empty && right_empty) {
                            pattern_count += 2.0f;  // Open pattern
                        } else if (left_empty || right_empty) {
                            pattern_count += 1.0f;  // Semi-open pattern
                        } else {
                            pattern_count += 0.5f;  // Blocked pattern
                        }
                    } else if (has_empty && count == pattern_length - 1) {
                        bool left_empty = (j > 0 && board[i][j-1].item<int>() == 0);
                        bool right_empty = (j + pattern_length < board_size && 
                                          board[i][j + pattern_length].item<int>() == 0);
                        
                        if (left_empty || right_empty) {
                            pattern_count += 0.3f;  // Broken pattern
                        }
                    }
                }
            }
            
            // Vertical patterns
            for (int i = 0; i <= board_size - pattern_length; ++i) {
                for (int j = 0; j < board_size; ++j) {
                    int count = 0;
                    bool has_empty = false;
                    
                    for (int k = 0; k < pattern_length; ++k) {
                        int val = board[i + k][j].item<int>();
                        if (val == player) count++;
                        else if (val == 0) has_empty = true;
                    }
                    
                    if (count == pattern_length) {
                        bool top_empty = (i > 0 && board[i-1][j].item<int>() == 0);
                        bool bottom_empty = (i + pattern_length < board_size && 
                                           board[i + pattern_length][j].item<int>() == 0);
                        
                        if (top_empty && bottom_empty) {
                            pattern_count += 2.0f;  // Open pattern
                        } else if (top_empty || bottom_empty) {
                            pattern_count += 1.0f;  // Semi-open pattern
                        } else {
                            pattern_count += 0.5f;  // Blocked pattern
                        }
                    } else if (has_empty && count == pattern_length - 1) {
                        bool top_empty = (i > 0 && board[i-1][j].item<int>() == 0);
                        bool bottom_empty = (i + pattern_length < board_size && 
                                           board[i + pattern_length][j].item<int>() == 0);
                        
                        if (top_empty || bottom_empty) {
                            pattern_count += 0.3f;  // Broken pattern
                        }
                    }
                }
            }
            
            // Diagonal patterns
            for (int i = 0; i <= board_size - pattern_length; ++i) {
                for (int j = 0; j <= board_size - pattern_length; ++j) {
                    int count = 0;
                    bool has_empty = false;
                    
                    for (int k = 0; k < pattern_length; ++k) {
                        int val = board[i + k][j + k].item<int>();
                        if (val == player) count++;
                        else if (val == 0) has_empty = true;
                    }
                    
                    if (count == pattern_length) {
                        bool tl_empty = (i > 0 && j > 0 && board[i-1][j-1].item<int>() == 0);
                        bool br_empty = (i + pattern_length < board_size && 
                                       j + pattern_length < board_size && 
                                       board[i + pattern_length][j + pattern_length].item<int>() == 0);
                        
                        if (tl_empty && br_empty) {
                            pattern_count += 2.0f;  // Open pattern
                        } else if (tl_empty || br_empty) {
                            pattern_count += 1.0f;  // Semi-open pattern
                        } else {
                            pattern_count += 0.5f;  // Blocked pattern
                        }
                    } else if (has_empty && count == pattern_length - 1) {
                        bool tl_empty = (i > 0 && j > 0 && board[i-1][j-1].item<int>() == 0);
                        bool br_empty = (i + pattern_length < board_size && 
                                       j + pattern_length < board_size && 
                                       board[i + pattern_length][j + pattern_length].item<int>() == 0);
                        
                        if (tl_empty || br_empty) {
                            pattern_count += 0.3f;  // Broken pattern
                        }
                    }
                }
            }
            
            // Anti-diagonal patterns
            for (int i = 0; i <= board_size - pattern_length; ++i) {
                for (int j = pattern_length - 1; j < board_size; ++j) {
                    int count = 0;
                    bool has_empty = false;
                    
                    for (int k = 0; k < pattern_length; ++k) {
                        int val = board[i + k][j - k].item<int>();
                        if (val == player) count++;
                        else if (val == 0) has_empty = true;
                    }
                    
                    if (count == pattern_length) {
                        bool tr_empty = (i > 0 && j < board_size-1 && board[i-1][j+1].item<int>() == 0);
                        bool bl_empty = (i + pattern_length < board_size && j >= pattern_length && 
                                       board[i + pattern_length][j - pattern_length].item<int>() == 0);
                        
                        if (tr_empty && bl_empty) {
                            pattern_count += 2.0f;  // Open pattern
                        } else if (tr_empty || bl_empty) {
                            pattern_count += 1.0f;  // Semi-open pattern
                        } else {
                            pattern_count += 0.5f;  // Blocked pattern
                        }
                    } else if (has_empty && count == pattern_length - 1) {
                        bool tr_empty = (i > 0 && j < board_size-1 && board[i-1][j+1].item<int>() == 0);
                        bool bl_empty = (i + pattern_length < board_size && j >= pattern_length && 
                                       board[i + pattern_length][j - pattern_length].item<int>() == 0);
                        
                        if (tr_empty || bl_empty) {
                            pattern_count += 0.3f;  // Broken pattern
                        }
                    }
                }
            }
            
            return pattern_count;
        };
        
        // Count patterns of different lengths
        float fours = count_patterns(4);
        float threes = count_patterns(3);
        float twos = count_patterns(2);
        
        // Calculate total threat (matching CPU weights)
        total_threat = wins * 100.0f + fours * 20.0f + threes * 5.0f + twos * 1.0f;
        
        threat_counts[b] = total_threat;
    }
    
    // Transfer back to GPU
    return threat_counts.to(device_);
}

// Factory function implementation
std::unique_ptr<GPUAttackDefenseModule> createGPUAttackDefenseModule(
    core::GameType game_type, int board_size, torch::Device device) {
    switch (game_type) {
        case core::GameType::GOMOKU:
            return std::make_unique<GomokuGPUAttackDefense>(board_size, device);
        case core::GameType::CHESS:
            return std::make_unique<ChessGPUAttackDefense>(device);
        case core::GameType::GO:
            return std::make_unique<GoGPUAttackDefense>(board_size, device);
        default:
            throw std::runtime_error("Unsupported game type for GPU attack/defense");
    }
}

} // namespace alphazero