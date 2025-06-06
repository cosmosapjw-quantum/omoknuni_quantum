"""Simple tests for concurrent MCTS to improve coverage"""

import pytest
from unittest.mock import Mock, patch
import time
import queue

from mcts.core.concurrent_mcts import ConcurrentWaveProcessor, ConcurrentMCTS
from mcts.core.mcts import MCTSConfig
from mcts.core.tree_arena import MemoryConfig


class TestConcurrentCoverage:
    """Tests for coverage improvement"""
    
    def test_concurrent_processor_stats(self):
        """Test statistics calculation edge cases"""
        mock_mcts = Mock()
        processor = ConcurrentWaveProcessor(mock_mcts, num_workers=2)
        
        # Test when avg_wave_time_ms is 0
        stats = processor.get_statistics()
        assert stats['simulations_per_second'] == 0
        
        # Test with some data
        processor.stats['avg_wave_time_ms'] = 10.0
        mock_mcts.wave_engine = Mock()
        mock_mcts.wave_engine.config = Mock()
        mock_mcts.wave_engine.config.initial_wave_size = 256
        
        stats = processor.get_statistics()
        assert stats['simulations_per_second'] == 256 * 1000.0 / 10.0
        
    def test_worker_error_handling(self):
        """Test error handling in workers"""
        mock_mcts = Mock()
        mock_mcts.wave_engine = Mock()
        processor = ConcurrentWaveProcessor(mock_mcts, num_workers=1)
        
        # Test queue.Empty handling
        processor.running = True
        processor._selection_worker()  # Should handle empty queue
        
        # Test exception handling
        processor.selection_queue.put(("invalid", time.time()))
        processor._selection_worker()  # Should handle exception
        
    def test_concurrent_mcts_search_edge_cases(self):
        """Test ConcurrentMCTS search edge cases"""
        game = Mock()
        evaluator = Mock()
        config = MCTSConfig(num_simulations=10, wave_size=10)
        
        # Mock methods
        game.get_legal_moves = Mock(return_value=[0, 1, 2])
        game.apply_move = Mock(return_value="new_state")
        
        mcts = ConcurrentMCTS(game, evaluator, config)
        
        # Mock internal methods
        mcts._get_or_create_root = Mock()
        root = Mock()
        root.is_expanded = False
        root.is_terminal = False
        mcts._get_or_create_root.return_value = root
        
        mcts.arena = Mock()
        mcts.arena.add_node = Mock(return_value="root_id")
        
        # Test with noise at root
        config.add_noise_at_root = True
        root.is_expanded = True
        mcts._add_dirichlet_noise = Mock()
        
        # Mock concurrent processor
        mcts.concurrent_processor = Mock()
        
        # Run search
        result = mcts.search("state")
        
        # Verify calls
        assert mcts._get_or_create_root.called
        assert root.expand.called
        assert mcts._add_dirichlet_noise.called
        assert mcts.concurrent_processor.start.called
        assert mcts.concurrent_processor.stop.called