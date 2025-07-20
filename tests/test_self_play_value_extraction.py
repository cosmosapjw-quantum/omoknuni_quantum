"""Test for value extraction from MCTS in self-play"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from mcts.neural_networks.self_play_manager import SelfPlayGame, SelfPlayConfig
from mcts.core.mcts import MCTS
from mcts.core.game_interface import GameInterface


class TestSelfPlayValueExtraction:
    """Test that self-play correctly extracts values from MCTS"""
    
    def test_value_extraction_from_mcts(self):
        """Test that value is correctly extracted from MCTS get_root_value"""
        # Arrange
        config = SelfPlayConfig(
            enable_resign=True,
            resign_threshold=-0.9
        )
        
        # Create mock game
        mock_game = Mock()
        mock_game.get_state.return_value = "test_state"
        mock_game.get_current_player.return_value = 1
        mock_game.is_terminal.return_value = False
        mock_game.get_valid_actions.return_value = [0, 1, 2, 3, 4]
        
        # Create mock MCTS with get_root_value
        mock_mcts = Mock()
        mock_mcts.search.return_value = np.array([0.2, 0.3, 0.1, 0.3, 0.1])
        mock_mcts.get_root_value.return_value = 0.75  # Positive value
        
        # Create game interface mock
        mock_game_interface = Mock()
        mock_game_interface.state_to_numpy.return_value = np.zeros((15, 15))
        
        # Create self-play game
        self_play_game = SelfPlayGame(mock_game, mock_mcts, config, "test_game")
        self_play_game.game_interface = mock_game_interface
        
        # Act
        resigned = self_play_game.play_single_move()
        
        # Assert
        assert not resigned  # Should not resign with positive value
        assert mock_mcts.get_root_value.called
        assert len(self_play_game.value_predictions) == 1
        assert self_play_game.value_predictions[0] == (0.75, 1)
        
    def test_resignation_with_negative_value(self):
        """Test that resignation happens when value is below threshold"""
        # Arrange
        config = SelfPlayConfig(
            enable_resign=True,
            resign_threshold=-0.9
        )
        
        # Create mock game
        mock_game = Mock()
        mock_game.get_state.return_value = "test_state"
        mock_game.get_current_player.return_value = 1
        mock_game.is_terminal.return_value = False
        mock_game.get_valid_actions.return_value = [0, 1, 2, 3, 4]
        
        # Create mock MCTS with very negative value
        mock_mcts = Mock()
        mock_mcts.search.return_value = np.array([0.2, 0.3, 0.1, 0.3, 0.1])
        mock_mcts.get_root_value.return_value = -0.95  # Below threshold
        
        # Create game interface mock
        mock_game_interface = Mock()
        mock_game_interface.state_to_numpy.return_value = np.zeros((15, 15))
        
        # Create self-play game
        self_play_game = SelfPlayGame(mock_game, mock_mcts, config, "test_game")
        self_play_game.game_interface = mock_game_interface
        self_play_game.move_count = 15  # Past minimum moves for resignation
        
        # Act
        resigned = self_play_game.play_single_move()
        
        # Assert
        assert resigned  # Should resign with value below threshold
        assert mock_mcts.get_root_value.called
        
    def test_no_resignation_without_get_root_value(self):
        """Test that missing get_root_value doesn't cause resignation"""
        # Arrange
        config = SelfPlayConfig(
            enable_resign=True,
            resign_threshold=-0.9
        )
        
        # Create mock game
        mock_game = Mock()
        mock_game.get_state.return_value = "test_state"
        mock_game.get_current_player.return_value = 1
        mock_game.is_terminal.return_value = False
        mock_game.get_valid_actions.return_value = [0, 1, 2, 3, 4]
        
        # Create mock MCTS WITHOUT get_root_value
        mock_mcts = Mock()
        mock_mcts.search.return_value = np.array([0.2, 0.3, 0.1, 0.3, 0.1])
        # Remove get_root_value attribute
        del mock_mcts.get_root_value
        
        # Create game interface mock
        mock_game_interface = Mock()
        mock_game_interface.state_to_numpy.return_value = np.zeros((15, 15))
        
        # Create self-play game
        self_play_game = SelfPlayGame(mock_game, mock_mcts, config, "test_game")
        self_play_game.game_interface = mock_game_interface
        self_play_game.move_count = 15
        
        # Act
        resigned = self_play_game.play_single_move()
        
        # Assert
        assert not resigned  # Should not resign with default value 0.0
        assert len(self_play_game.value_predictions) == 0  # No value recorded


if __name__ == "__main__":
    pytest.main([__file__, "-v"])