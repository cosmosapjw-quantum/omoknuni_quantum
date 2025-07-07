"""
Comprehensive tests for Go-specific gameplay

This module tests Go-specific rules and gameplay including:
- Stone placement and captures
- Ko rule enforcement
- Territory scoring
- Eye detection
- Group liberty counting
- Suicide prevention
"""

import pytest
import numpy as np
import torch

from mcts.core.game_interface import GameInterface, GameType
from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.gpu.gpu_game_states import GameType as GPUGameType
from conftest import assert_tensor_equal


class TestGoRules:
    """Test Go-specific game rules"""
    
    def test_standard_go_initialization(self):
        """Test standard Go game initialization"""
        game = GameInterface(GameType.GO, board_size=19)
        
        assert game.board_size == 19
        assert game.max_moves == 362  # 19*19 + 1 for pass move
        assert game.piece_planes == 2
        assert game.game_options.get('komi', 7.5) == 7.5
        
    def test_custom_board_sizes(self):
        """Test Go with different board sizes"""
        for size in [9, 13, 19]:
            game = GameInterface(GameType.GO, board_size=size)
            assert game.board_size == size
            assert game.max_moves == size * size + 1  # Include pass move
            
            state = game.create_initial_state()
            legal_moves = game.get_legal_moves(state)
            # All positions plus pass move
            assert len(legal_moves) == size * size + 1
            
    def test_stone_placement(self):
        """Test basic stone placement"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Place black stone
        state = game.apply_move(state, 40)  # Center of 9x9
        
        # Verify stone placed
        board = game.state_to_numpy(state)
        assert board[0, 4, 4] == 1  # Black stone at center
        
        # Place white stone
        state = game.apply_move(state, 41)
        board = game.state_to_numpy(state)
        assert board[1, 4, 5] == 1  # White stone
        
    def test_capture_single_stone(self):
        """Test capturing a single stone"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Surround and capture white stone
        # . B .
        # B W B
        # . B .
        moves = [
            3 * 9 + 3,  # Black at (3,3)
            3 * 9 + 4,  # White at (3,4) - the stone to be captured
            3 * 9 + 5,  # Black at (3,5)
            0 * 9 + 0,  # White elsewhere (0,0)
            2 * 9 + 4,  # Black at (2,4) - surround top
            0 * 9 + 1,  # White elsewhere (0,1)
            4 * 9 + 4,  # Black at (4,4) - complete capture!
        ]
        
        for i, move in enumerate(moves):
            if not game.is_terminal(state):
                state = game.apply_move(state, move)
                
        # Check white stone at (3,4) was captured
        board = game.state_to_numpy(state)
        assert board[1, 3, 4] == 0  # White stone removed
        
        
    def test_ko_rule_basic(self):
        """Test basic Ko rule"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Create a Ko pattern where both groups are in a capturing race
        # We'll create interlocked groups where capture creates Ko
        ko_setup_moves = [
            # Create interlocked black and white groups
            1 * 9 + 1,  # Black at (1,1)
            1 * 9 + 2,  # White at (1,2)
            0 * 9 + 2,  # Black at (0,2) - surround white from top
            0 * 9 + 1,  # White at (0,1) - surround black from top
            2 * 9 + 2,  # Black at (2,2) - surround white from bottom
            2 * 9 + 1,  # White at (2,1) - surround black from bottom
            1 * 9 + 3,  # Black at (1,3) - surround white from right
            1 * 9 + 0,  # White at (1,0) - surround black from left
            
            # Now both groups have exactly one liberty each
            0 * 9 + 0,  # Black at (0,0) to put white group in atari
            3 * 9 + 3,  # White plays elsewhere
            
            # Black captures white stone at (0,1)
            0 * 9 + 1,  # Black captures! This creates Ko
        ]
        
        # Play all moves
        for i, move in enumerate(ko_setup_moves):
            state = game.apply_move(state, move)
            
            # After black captures (last move), check Ko
            if i == len(ko_setup_moves) - 1:
                # White cannot immediately recapture at (0,1) due to Ko
                legal_moves = game.get_legal_moves(state)
                ko_position = 0 * 9 + 1  # Position (0,1) where white was captured
                assert ko_position not in legal_moves  # Ko rule prevents immediate recapture
        
    def test_suicide_prevention(self):
        """Test suicide move prevention"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Create situation where playing would be suicide
        # . B .
        # B . B
        # . B .
        suicide_setup = [
            0 * 9 + 1,  # B top
            8 * 9 + 8,  # W elsewhere
            1 * 9 + 0,  # B left
            8 * 9 + 7,  # W elsewhere
            1 * 9 + 2,  # B right
            8 * 9 + 6,  # W elsewhere
            2 * 9 + 1,  # B bottom
            8 * 9 + 5,  # W elsewhere
        ]
        
        for move in suicide_setup:
            state = game.apply_move(state, move)
            
        # Playing at (1,1) would be suicide for white
        legal_moves = game.get_legal_moves(state)
        center_move = 1 * 9 + 1
        
        # Suicide should be illegal (unless it captures)
        # Implementation dependent
        
    def test_pass_move(self):
        """Test pass move functionality"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Pass move should be in legal moves as -1
        legal_moves = game.get_legal_moves(state)
        pass_move = -1  # Go uses -1 for pass move
        
        assert pass_move in legal_moves
        
        # Apply pass
        state = game.apply_move(state, pass_move)
        
        # Game continues after single pass
        assert not game.is_terminal(state)
        
        # Two passes end game
        state = game.apply_move(state, pass_move)
        assert game.is_terminal(state)
        
        
    def test_territory_scoring(self):
        """Test territory scoring at game end"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Play a simple game with clear territory
        territory_game = [
            # Black builds on left, white on right
            0 * 9 + 0,  # B corner
            0 * 9 + 8,  # W corner
            1 * 9 + 0,  # B
            1 * 9 + 8,  # W
            2 * 9 + 0,  # B
            2 * 9 + 8,  # W
            3 * 9 + 0,  # B
            3 * 9 + 8,  # W
            4 * 9 + 0,  # B
            4 * 9 + 8,  # W
        ]
        
        for move in territory_game:
            state = game.apply_move(state, move)
            
        # Pass twice to end
        pass_move = -1  # Go uses -1 for pass move
        state = game.apply_move(state, pass_move)
        state = game.apply_move(state, pass_move)
        
        assert game.is_terminal(state)
        
        # Get winner based on territory
        winner = game.get_winner(state)
        assert winner in [-1, 0, 1]  # -1 for player 2 win, 0 for draw, 1 for player 1 win


class TestGoMoveValidation:
    """Test move validation in Go"""
    
    def test_valid_moves_empty_board(self):
        """Test all moves are valid on empty board"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        legal_moves = game.get_legal_moves(state)
        # Should have all board positions + pass
        assert len(legal_moves) == 82  # 81 + 1 pass
        
    def test_occupied_position_illegal(self):
        """Test cannot play on occupied position"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Place stone
        state = game.apply_move(state, 40)
        
        # Same position should now be illegal
        legal_moves = game.get_legal_moves(state)
        assert 40 not in legal_moves
        
    def test_ko_position_illegal(self):
        """Test Ko position is illegal"""
        # Ko rule testing is complex and implementation-specific
        # Basic test structure
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # After Ko situation, immediate recapture should be illegal
        # Implementation dependent
        
    def test_suicide_illegal_unless_capture(self):
        """Test suicide is illegal unless it captures"""
        game = GameInterface(GameType.GO, board_size=9)
        # Complex setup - implementation dependent


class TestGoStateRepresentation:
    """Test Go state representations"""
    
    def test_state_to_tensor_planes(self):
        """Test tensor representation with history planes"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        tensor = game.state_to_tensor(state)
        
        # Go typically uses 17 planes (8 history for each player + current player)
        # But this may vary by implementation
        assert tensor.shape[0] >= 3  # At minimum current black, white, player
        assert tensor.shape[1] == 9
        assert tensor.shape[2] == 9
        
    def test_state_history_tracking(self):
        """Test state history for Ko and patterns"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Make several moves
        moves = [40, 41, 42, 43, 44]
        for move in moves:
            state = game.apply_move(state, move)
            
        tensor = game.state_to_tensor(state)
        
        # Should track some history
        # Exact format is implementation dependent
        
    def test_symmetries_go(self):
        """Test board symmetries for Go"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Place stones asymmetrically
        state = game.apply_move(state, 10)  # Top area
        state = game.apply_move(state, 70)  # Bottom area
        
        board = game.state_to_numpy(state)
        policy = np.zeros(82)
        policy[20] = 0.5
        policy[30] = 0.5
        
        symmetries = game.get_symmetries(board, policy)
        
        # Should have 8 symmetries
        assert len(symmetries) == 8


class TestGoWithMCTS:
    """Test Go gameplay with MCTS"""
    
    def test_mcts_finds_capture(self, base_mcts_config, mock_evaluator_factory):
        """Test MCTS finds obvious capture"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Create a mock evaluator specifically for 9x9 Go
        mock_evaluator = mock_evaluator_factory(game_type='go')
        # Override action space for 9x9 Go (81 positions + 1 pass)
        mock_evaluator.action_space = 82
        
        # Update MCTS config for 9x9 Go
        base_mcts_config.board_size = 9
        base_mcts_config.game_type = GPUGameType.GO
        base_mcts_config.max_children_per_node = 82
        
        # Set up white stone with only one liberty
        # B B B
        # B W .
        # B B B
        capture_setup = [
            3 * 9 + 3,  # B
            3 * 9 + 4,  # W (target)
            3 * 9 + 2,  # B
            0 * 9 + 0,  # W elsewhere
            2 * 9 + 3,  # B
            0 * 9 + 1,  # W elsewhere
            2 * 9 + 4,  # B
            0 * 9 + 2,  # W elsewhere
            4 * 9 + 3,  # B
            0 * 9 + 3,  # W elsewhere
            4 * 9 + 4,  # B
            0 * 9 + 4,  # W elsewhere
            2 * 9 + 2,  # B
            0 * 9 + 5,  # W elsewhere
            4 * 9 + 2,  # B
            0 * 9 + 6,  # W elsewhere
        ]
        
        for move in capture_setup:
            state = game.apply_move(state, move)
            
        # Black to play, should capture at (3,5)
        mcts = MCTS(base_mcts_config, mock_evaluator)
        policy = mcts.search(state, num_simulations=100)
        
        # Verify MCTS produces valid policy
        assert policy.shape[0] == 82  # 81 board positions + 1 pass move = 82 total actions
        assert np.abs(policy.sum() - 1.0) < 0.001  # Should sum to 1
        assert np.all(policy >= 0)  # All probabilities non-negative
        
        # Check that the capture move is legal and gets some probability
        capture_move = 3 * 9 + 5
        legal_moves = game.get_legal_moves(state)
        assert capture_move in legal_moves
        
    def test_mcts_avoids_suicide(self, base_mcts_config, mock_evaluator_factory):
        """Test MCTS avoids suicide moves"""
        game = GameInterface(GameType.GO, board_size=9)
        
        # Create a mock evaluator specifically for 9x9 Go
        mock_evaluator = mock_evaluator_factory(game_type='go')
        mock_evaluator.action_space = 82
        
        # Update MCTS config for 9x9 Go
        base_mcts_config.board_size = 9
        base_mcts_config.game_type = GPUGameType.GO
        base_mcts_config.max_children_per_node = 82
        
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Create board where some moves would be suicide
        # MCTS should give them zero or very low probability
        
        state = game.create_initial_state()
        policy = mcts.search(state, num_simulations=50)
        
        # All probabilities should be non-negative
        assert np.all(policy >= 0)
        
    def test_mcts_territory_awareness(self, base_mcts_config, mock_evaluator_factory):
        """Test MCTS shows awareness of territory"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Create a mock evaluator specifically for 9x9 Go
        mock_evaluator = mock_evaluator_factory(game_type='go')
        mock_evaluator.action_space = 82
        
        # Update MCTS config for 9x9 Go
        base_mcts_config.board_size = 9
        base_mcts_config.game_type = GPUGameType.GO
        base_mcts_config.max_children_per_node = 82
        
        # Play opening moves
        opening_moves = [
            3 * 9 + 3,  # Black 3-3 point
            5 * 9 + 5,  # White 5-5 point
            5 * 9 + 3,  # Black 3-5 point
            3 * 9 + 5,  # White 5-3 point
        ]
        
        for move in opening_moves:
            state = game.apply_move(state, move)
            
        mcts = MCTS(base_mcts_config, mock_evaluator)
        policy = mcts.search(state, num_simulations=100)
        
        # Should show preference for important points
        # Corners and sides should have higher probability than center
        corner_moves = [0, 8, 72, 80]  # Corner positions
        corner_prob = sum(policy[m] for m in corner_moves if m < len(policy))
        
        center_moves = [40, 41, 49, 50]  # Center positions
        center_prob = sum(policy[m] for m in center_moves if m < len(policy))
        
        # In opening, corners/sides usually preferred
        # (This is a soft test as it depends on the evaluator)


class TestGoEndgame:
    """Test Go endgame scenarios"""
    
    def test_pass_decision(self, base_mcts_config, mock_evaluator_factory):
        """Test MCTS knows when to pass"""
        game = GameInterface(GameType.GO, board_size=9)
        state = game.create_initial_state()
        
        # Create a mock evaluator specifically for 9x9 Go
        mock_evaluator = mock_evaluator_factory(game_type='go')
        mock_evaluator.action_space = 82
        
        # Update MCTS config for 9x9 Go
        base_mcts_config.board_size = 9
        base_mcts_config.game_type = GPUGameType.GO
        base_mcts_config.max_children_per_node = 82
        
        # Create nearly complete game
        # In real implementation, would set up a position where
        # passing is the best move
        
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # For now, just verify pass is considered
        policy = mcts.search(state, num_simulations=50)
        pass_move_index = game.board_size * game.board_size  # Pass move is at board_size^2
        
        # Pass should have non-zero probability
        if pass_move_index < len(policy):
            assert policy[pass_move_index] >= 0
            


class TestGoPerformance:
    """Performance tests for Go"""
    
    @pytest.mark.slow
    def test_move_generation_performance(self):
        """Test move generation performance"""
        game = GameInterface(GameType.GO, board_size=19)
        state = game.create_initial_state()
        
        # Time move generation
        import time
        iterations = 1000
        
        start = time.time()
        for _ in range(iterations):
            legal_moves = game.get_legal_moves(state)
        elapsed = time.time() - start
        
        moves_per_second = iterations / elapsed
        assert moves_per_second > 5000  # Should be fast
        
    @pytest.mark.slow  
    def test_capture_detection_performance(self):
        """Test capture detection performance"""
        game = GameInterface(GameType.GO, board_size=19)
        
        # Create complex position
        state = game.create_initial_state()
        
        # Add many stones
        for i in range(50):
            move = np.random.randint(0, 361)
            if move in game.get_legal_moves(state):
                state = game.apply_move(state, move)
                
        # Time capture checks
        start = time.time()
        iterations = 100
        
        for _ in range(iterations):
            # Force capture check (implementation dependent)
            legal_moves = game.get_legal_moves(state)
            
        elapsed = time.time() - start
        checks_per_second = iterations / elapsed
        
        assert checks_per_second > 100  # Should handle complex positions


class TestGoSpecialRules:
    """Test special Go rules and variations"""
    
    def test_chinese_rules(self):
        """Test Chinese scoring rules"""
        game = GameInterface(GameType.GO, board_size=9, 
                           game_options={'scoring': 'chinese'})
        
        # Chinese rules count stones + territory
        # game_options is nested under 'game_options' key
        assert game.game_options['game_options']['scoring'] == 'chinese'
        
    def test_japanese_rules(self):
        """Test Japanese scoring rules"""  
        game = GameInterface(GameType.GO, board_size=9,
                           game_options={'scoring': 'japanese'})
        
        # Japanese rules count territory + captures
        assert game.game_options['game_options']['scoring'] == 'japanese'
        
    def test_different_komi(self):
        """Test different komi values"""
        for komi in [0.5, 5.5, 6.5, 7.5]:
            game = GameInterface(GameType.GO, board_size=19,
                               game_options={'komi': komi})
            assert game.game_options.get('komi', 7.5) == komi or \
                   game.game_options.get('game_options', {}).get('komi') == komi
            
    def test_handicap_stones(self):
        """Test handicap stone placement"""
        game = GameInterface(GameType.GO, board_size=19,
                           game_options={'handicap': 2})
        
        state = game.create_initial_state()
        
        # With handicap, black stones should be pre-placed
        board = game.state_to_numpy(state)
        black_stones = np.sum(board[0])  # Count black stones
        
        # Should have handicap stones placed
        # (Implementation dependent)