#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <tuple>
#include "utils/attack_defense_module.h"

using namespace alphazero;

class GomokuPatternTest : public ::testing::Test {
protected:
    std::unique_ptr<GomokuAttackDefenseModule> module;
    const int board_size = 15;
    
    void SetUp() override {
        module = std::make_unique<GomokuAttackDefenseModule>(board_size);
    }
    
    // Helper to create empty board
    std::vector<std::vector<int>> createEmptyBoard() {
        return std::vector<std::vector<int>>(board_size, std::vector<int>(board_size, 0));
    }
    
    // Helper to place stones on board
    void placeStones(std::vector<std::vector<int>>& board, 
                     const std::vector<std::tuple<int, int, int>>& stones) {
        for (const auto& [row, col, player] : stones) {
            board[row][col] = player;
        }
    }
    
    // Helper to run attack/defense calculation
    std::pair<float, float> calculateBonus(const std::vector<std::vector<int>>& board,
                                           int move_row, int move_col, int player) {
        // Create a copy of the board and place the move
        auto board_with_move = board;
        board_with_move[move_row][move_col] = player;
        
        std::vector<std::vector<std::vector<int>>> board_batch = {board_with_move};
        std::vector<int> chosen_moves = {move_row * board_size + move_col};
        std::vector<int> player_batch = {player};
        
        auto [attack, defense] = module->compute_bonuses(board_batch, chosen_moves, player_batch);
        return {attack[0], defense[0]};
    }
};

// ========== Open Three Pattern Tests ==========

TEST_F(GomokuPatternTest, OpenThreeHorizontalPerfect) {
    // Test perfect open three: _XXX_
    auto board = createEmptyBoard();
    placeStones(board, {
        {7, 6, 1}, {7, 7, 1}, {7, 8, 1}
    });
    
    // Place stone to create open four
    auto [attack, defense] = calculateBonus(board, 7, 5, 1);
    EXPECT_GT(attack, 0.0f) << "Creating open four should increase attack";
    
    // Alternative position
    auto [attack2, defense2] = calculateBonus(board, 7, 9, 1);
    EXPECT_GT(attack2, 0.0f) << "Creating open four from other side should increase attack";
}

TEST_F(GomokuPatternTest, OpenThreeVerticalPerfect) {
    // Test vertical open three
    auto board = createEmptyBoard();
    placeStones(board, {
        {6, 7, 1}, {7, 7, 1}, {8, 7, 1}
    });
    
    auto [attack, defense] = calculateBonus(board, 5, 7, 1);
    EXPECT_GT(attack, 0.0f) << "Vertical open four creation should increase attack";
}

TEST_F(GomokuPatternTest, OpenThreeDiagonalMain) {
    // Test main diagonal (top-left to bottom-right)
    auto board = createEmptyBoard();
    placeStones(board, {
        {6, 6, 1}, {7, 7, 1}, {8, 8, 1}
    });
    
    auto [attack, defense] = calculateBonus(board, 5, 5, 1);
    EXPECT_GT(attack, 0.0f) << "Diagonal open four creation should increase attack";
}

TEST_F(GomokuPatternTest, OpenThreeDiagonalAnti) {
    // Test anti-diagonal (top-right to bottom-left)
    auto board = createEmptyBoard();
    placeStones(board, {
        {6, 8, 1}, {7, 7, 1}, {8, 6, 1}
    });
    
    auto [attack, defense] = calculateBonus(board, 5, 9, 1);
    EXPECT_GT(attack, 0.0f) << "Anti-diagonal open four creation should increase attack";
}

TEST_F(GomokuPatternTest, BrokenThreePattern) {
    // Test broken three: XX_X
    auto board = createEmptyBoard();
    placeStones(board, {
        {7, 5, 1}, {7, 6, 1}, {7, 8, 1}
    });
    
    // Fill the gap
    auto [attack, defense] = calculateBonus(board, 7, 7, 1);
    EXPECT_GT(attack, 0.0f) << "Completing broken pattern should increase attack";
}

// ========== Open Four Pattern Tests ==========

TEST_F(GomokuPatternTest, OpenFourCreation) {
    // Test creating open four: _XXXX_
    auto board = createEmptyBoard();
    placeStones(board, {
        {7, 6, 1}, {7, 7, 1}, {7, 8, 1}, {7, 9, 1}
    });
    
    auto [attack, defense] = calculateBonus(board, 7, 5, 1);
    EXPECT_GT(attack, 50.0f) << "Creating open four should have very high attack value";
}

// ========== Winning Move Tests ==========

TEST_F(GomokuPatternTest, WinningMoveHorizontal) {
    // Test completing five in a row
    auto board = createEmptyBoard();
    placeStones(board, {
        {7, 5, 1}, {7, 6, 1}, {7, 7, 1}, {7, 8, 1}
    });
    
    auto [attack, defense] = calculateBonus(board, 7, 9, 1);
    EXPECT_GT(attack, 100.0f) << "Winning move should have maximum attack value";
}

TEST_F(GomokuPatternTest, WinningMoveVertical) {
    auto board = createEmptyBoard();
    placeStones(board, {
        {3, 7, 1}, {4, 7, 1}, {5, 7, 1}, {6, 7, 1}
    });
    
    auto [attack, defense] = calculateBonus(board, 7, 7, 1);
    EXPECT_GT(attack, 100.0f) << "Vertical winning move should have maximum attack value";
}

// ========== Defense Pattern Tests ==========

TEST_F(GomokuPatternTest, BlockOpponentOpenThree) {
    // Test blocking opponent's open three
    auto board = createEmptyBoard();
    placeStones(board, {
        {7, 6, 2}, {7, 7, 2}, {7, 8, 2}  // Opponent's three
    });
    
    auto [attack, defense] = calculateBonus(board, 7, 5, 1);
    EXPECT_GT(defense, 0.0f) << "Blocking open three should have defense value";
}

TEST_F(GomokuPatternTest, BlockOpponentOpenFour) {
    // Must block opponent's open four
    auto board = createEmptyBoard();
    placeStones(board, {
        {7, 5, 2}, {7, 6, 2}, {7, 7, 2}, {7, 8, 2}  // Opponent's four
    });
    
    auto [attack, defense] = calculateBonus(board, 7, 9, 1);
    EXPECT_GT(defense, 20.0f) << "Blocking open four should have very high defense value";
}

TEST_F(GomokuPatternTest, BlockWinningMove) {
    // Must block opponent's winning threat
    auto board = createEmptyBoard();
    placeStones(board, {
        {7, 5, 2}, {7, 6, 2}, {7, 7, 2}, {7, 8, 2}  // Four in a row
    });
    
    auto [attack, defense] = calculateBonus(board, 7, 4, 1);
    EXPECT_GT(defense, 20.0f) << "Blocking winning move should have maximum defense value";
    
    // Also check blocking from other side
    auto [attack2, defense2] = calculateBonus(board, 7, 9, 1);
    EXPECT_GT(defense2, 20.0f) << "Blocking from other side should also have maximum defense";
}

// ========== Complex Pattern Tests ==========

TEST_F(GomokuPatternTest, DoubleThreats) {
    // Create position with double threats
    auto board = createEmptyBoard();
    placeStones(board, {
        // Horizontal threat
        {7, 6, 1}, {7, 7, 1}, {7, 8, 1},
        // Vertical threat
        {6, 7, 1}, {8, 7, 1}
    });
    
    // This move creates threats in two directions
    auto [attack, defense] = calculateBonus(board, 7, 5, 1);
    EXPECT_GT(attack, 0.0f) << "Creating multiple threats should have high attack value";
}

TEST_F(GomokuPatternTest, ForkCreation) {
    // Create a fork (multiple winning threats)
    auto board = createEmptyBoard();
    placeStones(board, {
        // Set up a position where one move creates two open threes
        {7, 6, 1}, {7, 7, 1},  // Horizontal
        {6, 8, 1}, {8, 8, 1}   // Vertical
    });
    
    auto [attack, defense] = calculateBonus(board, 7, 8, 1);
    EXPECT_GT(attack, 0.0f) << "Fork creation should have high attack value";
}

// ========== Edge Case Tests ==========

TEST_F(GomokuPatternTest, BoardEdgePatterns) {
    // Test patterns near board edges
    auto board = createEmptyBoard();
    
    // Top edge
    placeStones(board, {
        {0, 5, 1}, {0, 6, 1}, {0, 7, 1}
    });
    auto [attack1, defense1] = calculateBonus(board, 0, 4, 1);
    EXPECT_GE(attack1, 0.0f) << "Edge patterns should be handled correctly";
    
    // Corner
    board = createEmptyBoard();
    placeStones(board, {
        {0, 1, 1}, {0, 2, 1}
    });
    auto [attack2, defense2] = calculateBonus(board, 0, 0, 1);
    EXPECT_GE(attack2, 0.0f) << "Corner patterns should be handled correctly";
}

TEST_F(GomokuPatternTest, InvalidMoves) {
    // Test handling of invalid moves (occupied squares)
    auto board = createEmptyBoard();
    placeStones(board, {
        {7, 7, 1}  // Occupied square
    });
    
    // Try to place on occupied square
    auto [attack, defense] = calculateBonus(board, 7, 7, 2);
    // Should handle gracefully - implementation specific behavior
    EXPECT_GE(attack, 0.0f);
    EXPECT_GE(defense, 0.0f);
}

// ========== Batch Processing Tests ==========

TEST_F(GomokuPatternTest, BatchConsistency) {
    // Test that batch processing gives same results as individual processing
    auto board1 = createEmptyBoard();
    placeStones(board1, {{7, 7, 1}, {7, 8, 1}});
    
    auto board2 = createEmptyBoard();
    placeStones(board2, {{5, 5, 2}, {5, 6, 2}});
    
    // Process individually
    auto [attack1_single, defense1_single] = calculateBonus(board1, 7, 9, 1);
    auto [attack2_single, defense2_single] = calculateBonus(board2, 5, 7, 1);
    
    // Process as batch - need to place moves on boards first
    auto board1_with_move = board1;
    board1_with_move[7][9] = 1;
    auto board2_with_move = board2;
    board2_with_move[5][7] = 1;
    
    std::vector<std::vector<std::vector<int>>> board_batch = {board1_with_move, board2_with_move};
    std::vector<int> chosen_moves = {7 * board_size + 9, 5 * board_size + 7};
    std::vector<int> player_batch = {1, 1};
    
    auto [attack_batch, defense_batch] = module->compute_bonuses(
        board_batch, chosen_moves, player_batch);
    
    EXPECT_FLOAT_EQ(attack1_single, attack_batch[0]) << "Batch and single processing should match";
    EXPECT_FLOAT_EQ(defense1_single, defense_batch[0]);
    EXPECT_FLOAT_EQ(attack2_single, attack_batch[1]);
    EXPECT_FLOAT_EQ(defense2_single, defense_batch[1]);
}

// ========== Pattern Counting Tests ==========

TEST_F(GomokuPatternTest, CountMultipleOpenThrees) {
    // Test counting when there are multiple open threes
    auto board = createEmptyBoard();
    placeStones(board, {
        // First open three
        {5, 5, 2}, {5, 6, 2}, {5, 7, 2},
        // Second open three
        {8, 5, 2}, {8, 6, 2}, {8, 7, 2}
    });
    
    // Place a stone that doesn't directly interact with threats
    auto [attack, defense] = calculateBonus(board, 10, 10, 1);
    
    // The defense value should reflect multiple threats exist
    // (exact value depends on implementation)
    EXPECT_GE(defense, 0.0f);
}

// ========== Performance Test ==========

TEST_F(GomokuPatternTest, LargeBatchPerformance) {
    // Test with larger batch to ensure performance
    const int batch_size = 100;
    std::vector<std::vector<std::vector<int>>> board_batch;
    std::vector<int> chosen_moves;
    std::vector<int> player_batch;
    
    // Create varied positions
    for (int i = 0; i < batch_size; ++i) {
        auto board = createEmptyBoard();
        // Add some random stones
        placeStones(board, {
            {7, 7, 1}, {7, 8, (i % 2) + 1}, {8, 7, 2}
        });
        board_batch.push_back(board);
        chosen_moves.push_back(6 * board_size + 6);
        player_batch.push_back(1);
    }
    
    // Should complete without issues
    auto [attack, defense] = module->compute_bonuses(board_batch, chosen_moves, player_batch);
    
    EXPECT_EQ(attack.size(), batch_size);
    EXPECT_EQ(defense.size(), batch_size);
}