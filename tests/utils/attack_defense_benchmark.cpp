#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <random>
#include "utils/attack_defense_module.h"
#include "utils/gpu_attack_defense_module.h"
#include "games/gomoku/gomoku_state.h"
#include "games/chess/chess_state.h"
#include "games/go/go_state.h"

using namespace alphazero;

#ifdef WITH_TORCH

TEST(AttackDefenseTest, CPUBenchmark) {
    const int board_size = 15;
    const int batch_size = 8;  // Reduced batch size for faster testing
    const int num_iterations = 10;  // Reduced iterations to avoid stalling
    
    // Create random board positions
    std::vector<std::vector<std::vector<int>>> board_batch(batch_size, 
        std::vector<std::vector<int>>(board_size, std::vector<int>(board_size, 0)));
    
    // Fill with some random stones
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> player_dist(0, 2);
    std::uniform_int_distribution<int> pos_dist(0, board_size - 1);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < 30; ++i) {  // Place ~30 stones per board
            int row = pos_dist(rng);
            int col = pos_dist(rng);
            board_batch[b][row][col] = player_dist(rng);
        }
    }
    
    // Create moves to evaluate
    std::vector<int> chosen_moves(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        // Find an empty position
        for (int pos = 0; pos < board_size * board_size; ++pos) {
            int row = pos / board_size;
            int col = pos % board_size;
            if (board_batch[b][row][col] == 0) {
                chosen_moves[b] = pos;
                break;
            }
        }
    }
    
    std::vector<int> player_batch(batch_size, 1);
    
    // Benchmark CPU version
    auto cpu_module = std::make_unique<GomokuAttackDefenseModule>(board_size);
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        auto [attack, defense] = cpu_module->compute_bonuses(board_batch, chosen_moves, player_batch);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    
    std::cout << "CPU Attack/Defense computation:" << std::endl;
    std::cout << "  Total time: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "  Time per batch: " << cpu_duration.count() / (float)num_iterations << " ms" << std::endl;
    std::cout << "  Throughput: " << (batch_size * num_iterations * 1000.0f) / cpu_duration.count() 
              << " boards/sec" << std::endl;
    
    // GPU benchmarking
    if (torch::cuda::is_available()) {
        std::cout << "\nGPU Attack/Defense computation:" << std::endl;
        
        auto device = torch::kCUDA;
        auto gpu_module = createGPUAttackDefenseModule(core::GameType::GOMOKU, board_size, device);
        
        // Convert boards to GPU tensors
        auto board_tensor = torch::zeros({batch_size, board_size, board_size}, 
                                       torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto moves_tensor = torch::zeros({batch_size}, 
                                       torch::TensorOptions().dtype(torch::kInt32).device(device));
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < board_size; ++i) {
                for (int j = 0; j < board_size; ++j) {
                    board_tensor[b][i][j] = board_batch[b][i][j];
                }
            }
            moves_tensor[b] = chosen_moves[b];
        }
        
        // Warm up GPU
        for (int i = 0; i < 10; ++i) {
            auto [attack, defense] = gpu_module->compute_bonuses_gpu(board_tensor, moves_tensor, 1);
        }
        
        // Benchmark GPU version
        auto gpu_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            auto [attack, defense] = gpu_module->compute_bonuses_gpu(board_tensor, moves_tensor, 1);
            torch::cuda::synchronize(); // Ensure GPU computation completes
        }
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
        
        std::cout << "  Total time: " << gpu_duration.count() << " ms" << std::endl;
        std::cout << "  Time per batch: " << gpu_duration.count() / (float)num_iterations << " ms" << std::endl;
        std::cout << "  Throughput: " << (batch_size * num_iterations * 1000.0f) / gpu_duration.count() 
                  << " boards/sec" << std::endl;
        std::cout << "  Speedup vs CPU: " << (float)cpu_duration.count() / gpu_duration.count() << "x" << std::endl;
    } else {
        std::cout << "\nGPU not available for benchmarking" << std::endl;
    }
}

// GPU vs CPU Validation Test
TEST(AttackDefenseTest, GPUvsCPUValidation) {
    const int board_size = 15;
    const int batch_size = 8;
    
    std::cout << "Starting GPU vs CPU validation test..." << std::endl;
    
    // Skip test if CUDA is not available
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping GPU vs CPU validation test";
    }
    
    std::cout << "CUDA is available, proceeding with test..." << std::endl;
    
    // Create test boards
    std::vector<std::vector<std::vector<int>>> board_batch(batch_size, 
        std::vector<std::vector<int>>(board_size, std::vector<int>(board_size, 0)));
    
    // Create some specific patterns to test
    // Board 0: Horizontal three
    board_batch[0][7][6] = 1;
    board_batch[0][7][7] = 1;
    board_batch[0][7][8] = 1;
    
    // Board 1: Vertical three
    board_batch[1][6][7] = 1;
    board_batch[1][7][7] = 1;
    board_batch[1][8][7] = 1;
    
    // Board 2: Diagonal three
    board_batch[2][6][6] = 1;
    board_batch[2][7][7] = 1;
    board_batch[2][8][8] = 1;
    
    // Board 3: Opponent's four (defense scenario)
    board_batch[3][5][5] = 2;
    board_batch[3][5][6] = 2;
    board_batch[3][5][7] = 2;
    board_batch[3][5][8] = 2;
    
    // Create moves and players - need one move per board
    std::vector<int> chosen_moves(batch_size);
    chosen_moves[0] = 7 * board_size + 9;   // Extend horizontal
    chosen_moves[1] = 5 * board_size + 7;   // Extend vertical
    chosen_moves[2] = 5 * board_size + 5;   // Extend diagonal
    chosen_moves[3] = 5 * board_size + 9;   // Block opponent
    // For remaining boards, use simple moves
    for (int i = 4; i < batch_size; ++i) {
        chosen_moves[i] = 7 * board_size + (10 + i);  // Different positions
    }
    std::vector<int> player_batch(batch_size, 1);
    
    // CPU computation
    std::cout << "Creating CPU module..." << std::endl;
    auto cpu_module = std::make_unique<GomokuAttackDefenseModule>(board_size);
    
    // Place moves on boards for CPU version
    std::vector<std::vector<std::vector<int>>> cpu_boards = board_batch;
    for (int b = 0; b < batch_size; ++b) {
        int row = chosen_moves[b] / board_size;
        int col = chosen_moves[b] % board_size;
        cpu_boards[b][row][col] = player_batch[b];
    }
    
    std::cout << "Computing CPU bonuses..." << std::endl;
    auto [cpu_attack, cpu_defense] = cpu_module->compute_bonuses(
        cpu_boards, chosen_moves, player_batch);
    std::cout << "CPU computation done." << std::endl;
    
    // GPU computation
    std::cout << "Creating GPU module..." << std::endl;
    auto device = torch::kCUDA;
    auto gpu_module = createGPUAttackDefenseModule(core::GameType::GOMOKU, board_size, device);
    std::cout << "GPU module created." << std::endl;
    
    // Convert to torch tensors (use integer type to match implementation)
    auto board_tensor = torch::zeros({batch_size, board_size, board_size}, 
                                   torch::TensorOptions().dtype(torch::kInt32).device(device));
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < board_size; ++i) {
            for (int j = 0; j < board_size; ++j) {
                board_tensor[b][i][j] = cpu_boards[b][i][j];
            }
        }
    }
    
    auto moves_tensor = torch::tensor(chosen_moves, 
                                    torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    std::cout << "Board tensor created. Calling GPU compute_bonuses..." << std::endl;
    std::cout << "Board tensor shape: " << board_tensor.sizes() << std::endl;
    std::cout << "Moves tensor shape: " << moves_tensor.sizes() << std::endl;
    
    auto [gpu_attack_tensor, gpu_defense_tensor] = gpu_module->compute_bonuses_gpu(
        board_tensor, moves_tensor, 1);
    
    std::cout << "GPU computation done." << std::endl;
    
    // Convert GPU results back to CPU
    auto gpu_attack_cpu = gpu_attack_tensor.to(torch::kCPU);
    auto gpu_defense_cpu = gpu_defense_tensor.to(torch::kCPU);
    
    // Compare results
    const float tolerance = 1e-3f;
    for (int b = 0; b < batch_size; ++b) {
        EXPECT_NEAR(cpu_attack[b], gpu_attack_cpu[b].item<float>(), tolerance)
            << "Attack mismatch at batch " << b;
        EXPECT_NEAR(cpu_defense[b], gpu_defense_cpu[b].item<float>(), tolerance)
            << "Defense mismatch at batch " << b;
    }
}

// Test for compute_planes (tensor outputs used by neural network)
TEST(AttackDefenseTest, GPUvsCPUPlanesValidation) {
    const int board_size = 15;
    const int batch_size = 4;
    
    std::cout << "Starting GPU vs CPU planes validation test..." << std::endl;
    
    // Skip test if CUDA is not available
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping GPU vs CPU planes validation test";
    }
    
    // Create test game states with various scenarios
    std::vector<std::unique_ptr<core::IGameState>> states;
    
    // State 1: Empty board
    auto state1 = std::make_unique<games::gomoku::GomokuState>(board_size);
    states.push_back(std::move(state1));
    
    // State 2: Horizontal pattern
    auto state2 = std::make_unique<games::gomoku::GomokuState>(board_size);
    state2->makeMove(7 * board_size + 6);  // Player 1
    state2->makeMove(8 * board_size + 6);  // Player 2
    state2->makeMove(7 * board_size + 7);  // Player 1
    state2->makeMove(8 * board_size + 7);  // Player 2
    state2->makeMove(7 * board_size + 8);  // Player 1
    states.push_back(std::move(state2));
    
    // State 3: Vertical pattern
    auto state3 = std::make_unique<games::gomoku::GomokuState>(board_size);
    state3->makeMove(6 * board_size + 7);  // Player 1
    state3->makeMove(6 * board_size + 8);  // Player 2
    state3->makeMove(7 * board_size + 7);  // Player 1
    state3->makeMove(7 * board_size + 8);  // Player 2
    state3->makeMove(8 * board_size + 7);  // Player 1
    states.push_back(std::move(state3));
    
    // State 4: Complex pattern with multiple threats
    auto state4 = std::make_unique<games::gomoku::GomokuState>(board_size);
    // Create a complex board position
    state4->makeMove(7 * board_size + 7);   // Center
    state4->makeMove(7 * board_size + 8);
    state4->makeMove(8 * board_size + 7);
    state4->makeMove(8 * board_size + 8);
    state4->makeMove(6 * board_size + 7);
    state4->makeMove(9 * board_size + 7);
    state4->makeMove(7 * board_size + 6);
    state4->makeMove(7 * board_size + 9);
    states.push_back(std::move(state4));
    
    // CPU computation
    std::cout << "Computing CPU planes..." << std::endl;
    auto cpu_module = std::make_unique<GomokuAttackDefenseModule>(board_size);
    auto [cpu_attack_planes, cpu_defense_planes] = cpu_module->compute_planes(states);
    std::cout << "CPU planes computed. Shape: [" << cpu_attack_planes.size() 
              << ", " << cpu_attack_planes[0].size() 
              << ", " << cpu_attack_planes[0][0].size() << "]" << std::endl;
    
    // GPU computation
    std::cout << "Preparing GPU computation..." << std::endl;
    auto device = torch::kCUDA;
    auto gpu_module = createGPUAttackDefenseModule(core::GameType::GOMOKU, board_size, device);
    
    // Convert states to board tensors
    auto board_tensor = torch::zeros({batch_size, board_size, board_size}, 
                                   torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    for (size_t b = 0; b < states.size(); ++b) {
        auto board_repr = states[b]->getTensorRepresentation();
        for (int i = 0; i < board_size; ++i) {
            for (int j = 0; j < board_size; ++j) {
                if (board_repr[0][i][j] > 0.5f) {
                    board_tensor[b][i][j] = states[b]->getCurrentPlayer();
                } else if (board_repr[1][i][j] > 0.5f) {
                    board_tensor[b][i][j] = 3 - states[b]->getCurrentPlayer();
                }
            }
        }
    }
    
    std::cout << "Computing GPU planes..." << std::endl;
    auto [gpu_attack_tensor, gpu_defense_tensor] = gpu_module->compute_planes_gpu(
        board_tensor, 1);  // Assuming player 1's perspective
    
    // Convert GPU results to CPU
    auto gpu_attack_cpu = gpu_attack_tensor.to(torch::kCPU);
    auto gpu_defense_cpu = gpu_defense_tensor.to(torch::kCPU);
    
    std::cout << "GPU planes computed. Shape: " << gpu_attack_cpu.sizes() << std::endl;
    
    // Compare results for each position on each board
    const float tolerance = 1e-3f;
    int mismatches = 0;
    
    for (int b = 0; b < batch_size; ++b) {
        std::cout << "\nValidating board " << b << "..." << std::endl;
        
        for (int i = 0; i < board_size; ++i) {
            for (int j = 0; j < board_size; ++j) {
                float cpu_attack_val = cpu_attack_planes[b][i][j];
                float gpu_attack_val = gpu_attack_cpu[b][i][j].item<float>();
                float cpu_defense_val = cpu_defense_planes[b][i][j];
                float gpu_defense_val = gpu_defense_cpu[b][i][j].item<float>();
                
                if (std::abs(cpu_attack_val - gpu_attack_val) > tolerance) {
                    if (mismatches < 5) {  // Only print first few mismatches
                        std::cout << "Attack mismatch at [" << b << "][" << i << "][" << j << "]: "
                                  << "CPU=" << cpu_attack_val << ", GPU=" << gpu_attack_val << std::endl;
                    }
                    mismatches++;
                }
                
                if (std::abs(cpu_defense_val - gpu_defense_val) > tolerance) {
                    if (mismatches < 5) {
                        std::cout << "Defense mismatch at [" << b << "][" << i << "][" << j << "]: "
                                  << "CPU=" << cpu_defense_val << ", GPU=" << gpu_defense_val << std::endl;
                    }
                    mismatches++;
                }
            }
        }
    }
    
    if (mismatches > 0) {
        std::cout << "Total mismatches: " << mismatches << std::endl;
    }
    
    EXPECT_EQ(mismatches, 0) << "Found " << mismatches << " mismatches between CPU and GPU planes";
}

// Test GPU vs CPU for Chess
TEST(AttackDefenseTest, ChessGPUvsCPUValidation) {
    const int batch_size = 4;
    
    std::cout << "Starting Chess GPU vs CPU validation test..." << std::endl;
    
    // Skip test if CUDA is not available
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping Chess GPU vs CPU validation test";
    }
    
    // Create test boards
    std::vector<std::vector<std::vector<int>>> board_batch(batch_size, 
        std::vector<std::vector<int>>(8, std::vector<int>(8, 0)));
    
    // Board 0: Knight can attack multiple squares
    board_batch[0][4][4] = 2;  // White knight at e4
    
    // Board 1: Bishop diagonal attacks
    board_batch[1][3][3] = 3;  // White bishop at d4
    
    // Board 2: Rook horizontal/vertical attacks
    board_batch[2][4][4] = 4;  // White rook at e4
    
    // Board 3: Queen attacks all directions
    board_batch[3][3][3] = 5;  // White queen at d4
    
    // Create moves - test attack calculation
    std::vector<int> chosen_moves(batch_size);
    chosen_moves[0] = 4 * 8 + 4;  // Knight position
    chosen_moves[1] = 3 * 8 + 3;  // Bishop position  
    chosen_moves[2] = 4 * 8 + 4;  // Rook position
    chosen_moves[3] = 3 * 8 + 3;  // Queen position
    
    std::vector<int> player_batch(batch_size, 1);  // All white pieces
    
    // CPU computation
    auto cpu_module = std::make_unique<ChessAttackDefenseModule>();
    auto [cpu_attack, cpu_defense] = cpu_module->compute_bonuses(
        board_batch, chosen_moves, player_batch);
    
    // GPU computation
    auto device = torch::kCUDA;
    auto gpu_module = createGPUAttackDefenseModule(core::GameType::CHESS, 8, device);
    
    // Convert to torch tensors
    auto board_tensor = torch::zeros({batch_size, 8, 8}, 
                                   torch::TensorOptions().dtype(torch::kInt32).device(device));
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                board_tensor[b][i][j] = board_batch[b][i][j];
            }
        }
    }
    
    auto moves_tensor = torch::tensor(chosen_moves, 
                                    torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    auto [gpu_attack_tensor, gpu_defense_tensor] = gpu_module->compute_bonuses_gpu(
        board_tensor, moves_tensor, 1);
    
    // Convert GPU results back to CPU
    auto gpu_attack_cpu = gpu_attack_tensor.to(torch::kCPU);
    auto gpu_defense_cpu = gpu_defense_tensor.to(torch::kCPU);
    
    // Compare results
    const float tolerance = 1e-3f;
    for (int b = 0; b < batch_size; ++b) {
        EXPECT_NEAR(cpu_attack[b], gpu_attack_cpu[b].item<float>(), tolerance)
            << "Chess Attack mismatch at batch " << b;
        EXPECT_NEAR(cpu_defense[b], gpu_defense_cpu[b].item<float>(), tolerance)
            << "Chess Defense mismatch at batch " << b;
    }
}

// Test GPU vs CPU for Go
TEST(AttackDefenseTest, GoGPUvsCPUValidation) {
    const int board_size = 9;  // Use smaller board for testing
    const int batch_size = 4;
    
    std::cout << "Starting Go GPU vs CPU validation test..." << std::endl;
    
    // Skip test if CUDA is not available
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available, skipping Go GPU vs CPU validation test";
    }
    
    // Create test boards
    std::vector<std::vector<std::vector<int>>> board_batch(batch_size, 
        std::vector<std::vector<int>>(board_size, std::vector<int>(board_size, 0)));
    
    // Board 0: Capture scenario
    board_batch[0][4][4] = 1;  // Black stone
    board_batch[0][4][3] = 2;  // White stones surrounding
    board_batch[0][4][5] = 2;
    board_batch[0][3][4] = 2;
    // Leave 5,4 empty for capture
    
    // Board 1: Atari (one liberty)
    board_batch[1][2][2] = 1;  // Black group
    board_batch[1][2][3] = 1;
    board_batch[1][1][2] = 2;  // White stones
    board_batch[1][1][3] = 2;
    board_batch[1][2][1] = 2;
    board_batch[1][3][2] = 2;
    board_batch[1][3][3] = 2;
    // 2,4 would complete the capture
    
    // Board 2: Eye creation
    board_batch[2][1][1] = 1;  // Black stones forming eye shape
    board_batch[2][1][2] = 1;
    board_batch[2][1][3] = 1;
    board_batch[2][2][1] = 1;
    board_batch[2][2][3] = 1;
    board_batch[2][3][1] = 1;
    board_batch[2][3][2] = 1;
    board_batch[2][3][3] = 1;
    // 2,2 is the eye point
    
    // Board 3: Liberty gain
    board_batch[3][4][4] = 1;
    board_batch[3][4][5] = 1;
    board_batch[3][5][4] = 1;
    
    // Create moves
    std::vector<int> chosen_moves(batch_size);
    chosen_moves[0] = 5 * board_size + 4;  // Capture move
    chosen_moves[1] = 2 * board_size + 4;  // Complete capture
    chosen_moves[2] = 2 * board_size + 2;  // Eye point
    chosen_moves[3] = 5 * board_size + 5;  // Extend group
    
    std::vector<int> player_batch(batch_size);
    player_batch[0] = 2;  // White's turn
    player_batch[1] = 2;  // White's turn
    player_batch[2] = 1;  // Black's turn
    player_batch[3] = 1;  // Black's turn
    
    // Place the moves
    for (int b = 0; b < batch_size; ++b) {
        int row = chosen_moves[b] / board_size;
        int col = chosen_moves[b] % board_size;
        board_batch[b][row][col] = player_batch[b];
    }
    
    // CPU computation
    auto cpu_module = std::make_unique<GoAttackDefenseModule>(board_size);
    auto [cpu_attack, cpu_defense] = cpu_module->compute_bonuses(
        board_batch, chosen_moves, player_batch);
    
    // GPU computation
    auto device = torch::kCUDA;
    auto gpu_module = createGPUAttackDefenseModule(core::GameType::GO, board_size, device);
    
    // Convert to torch tensors
    auto board_tensor = torch::zeros({batch_size, board_size, board_size}, 
                                   torch::TensorOptions().dtype(torch::kInt32).device(device));
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < board_size; ++i) {
            for (int j = 0; j < board_size; ++j) {
                board_tensor[b][i][j] = board_batch[b][i][j];
            }
        }
    }
    
    auto moves_tensor = torch::tensor(chosen_moves, 
                                    torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    // Need to process each board with its correct player
    auto gpu_attack_tensor = torch::zeros({batch_size}, torch::kFloat32).to(device);
    auto gpu_defense_tensor = torch::zeros({batch_size}, torch::kFloat32).to(device);
    
    for (int b = 0; b < batch_size; ++b) {
        auto single_board = board_tensor[b].unsqueeze(0);
        auto single_move = moves_tensor[b].unsqueeze(0);
        auto [single_attack, single_defense] = gpu_module->compute_bonuses_gpu(
            single_board, single_move, player_batch[b]);
        gpu_attack_tensor[b] = single_attack[0];
        gpu_defense_tensor[b] = single_defense[0];
    }
    
    // Convert GPU results back to CPU
    auto gpu_attack_cpu = gpu_attack_tensor.to(torch::kCPU);
    auto gpu_defense_cpu = gpu_defense_tensor.to(torch::kCPU);
    
    // Compare results
    const float tolerance = 1e-3f;
    for (int b = 0; b < batch_size; ++b) {
        EXPECT_NEAR(cpu_attack[b], gpu_attack_cpu[b].item<float>(), tolerance)
            << "Go Attack mismatch at batch " << b;
        EXPECT_NEAR(cpu_defense[b], gpu_defense_cpu[b].item<float>(), tolerance)
            << "Go Defense mismatch at batch " << b;
    }
}

#else
// If WITH_TORCH is not defined, create a dummy test
TEST(AttackDefenseTest, NoGPUSupport) {
    GTEST_SKIP() << "GPU support not enabled (WITH_TORCH not defined)";
}
#endif // WITH_TORCH