"""
Tests for MCTS dynamics data extractor.

Following TDD principles - these tests are written before implementation.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

# Import to-be-implemented modules
import sys
import os

# Set library path for C++ extensions
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ['LD_LIBRARY_PATH'] = f"{os.path.join(project_root, 'python')}:{os.environ.get('LD_LIBRARY_PATH', '')}"

try:
    from python.mcts.quantum.analysis.dynamics_extractor import (
        MCTSDynamicsExtractor, ExtractionConfig, DynamicsData
    )
except ImportError:
    # Expected to fail before implementation
    MCTSDynamicsExtractor = None
    ExtractionConfig = None
    DynamicsData = None


class TestMCTSDynamicsExtractor:
    """Test suite for MCTS dynamics data extraction"""
    
    def test_extractor_exists(self):
        """MCTSDynamicsExtractor class should exist"""
        assert MCTSDynamicsExtractor is not None, "MCTSDynamicsExtractor not implemented"
    
    def test_extraction_config(self):
        """Should create extraction configuration"""
        if ExtractionConfig is None:
            pytest.skip("ExtractionConfig not yet implemented")
            
        config = ExtractionConfig(
            extract_q_values=True,
            extract_visits=True,
            extract_policy=True,
            extract_value_landscape=True,
            extract_search_depth=True,
            time_window=100,
            sampling_rate=1
        )
        
        assert config.extract_q_values
        assert config.time_window == 100
    
    def test_extract_from_game(self):
        """Should extract dynamics data from a game"""
        if MCTSDynamicsExtractor is None:
            pytest.skip("MCTSDynamicsExtractor not yet implemented")
            
        extractor = MCTSDynamicsExtractor()
        
        # Mock game with MCTS data
        mock_game = Mock()
        mock_game.get_current_position = Mock(return_value=0)
        mock_game.get_action_count = Mock(return_value=9)
        
        # Mock MCTS tree data
        mock_tree = Mock()
        mock_tree.get_q_values = Mock(return_value=torch.tensor([0.5, 0.3, 0.2, 0.1, 0.0]))
        mock_tree.get_visit_counts = Mock(return_value=torch.tensor([400, 300, 200, 80, 20]))
        mock_tree.get_policy = Mock(return_value=torch.tensor([0.4, 0.3, 0.2, 0.08, 0.02]))
        mock_tree.get_search_depth = Mock(return_value=10)
        
        # Extract data
        dynamics_data = extractor.extract_from_position(mock_game, mock_tree)
        
        assert dynamics_data is not None
        assert 'q_values' in dynamics_data
        assert 'visits' in dynamics_data
        assert 'policy' in dynamics_data
        assert dynamics_data['q_values'].shape == (5,)
    
    def test_extract_trajectory(self):
        """Should extract full trajectory from self-play"""
        if MCTSDynamicsExtractor is None:
            pytest.skip("MCTSDynamicsExtractor not yet implemented")
            
        extractor = MCTSDynamicsExtractor()
        
        # Mock self-play trajectory
        trajectory = []
        for i in range(50):
            position_data = {
                'position_id': i,
                'q_values': torch.randn(9),
                'visits': torch.randint(1, 100, (9,)).float(),
                'policy': torch.softmax(torch.randn(9), dim=0),
                'value': torch.rand(1).item(),
                'depth': i % 10 + 5
            }
            trajectory.append(position_data)
        
        # Extract dynamics
        dynamics = extractor.extract_trajectory_dynamics(trajectory)
        
        assert len(dynamics.snapshots) == 50
        assert dynamics.metadata['total_positions'] == 50
    
    def test_save_load_dynamics(self):
        """Should save and load dynamics data"""
        if DynamicsData is None:
            pytest.skip("DynamicsData not yet implemented")
            
        # Create sample dynamics data
        snapshots = []
        for i in range(10):
            snapshot = {
                'timestamp': i,
                'q_values': torch.randn(5).tolist(),
                'visits': torch.randint(1, 100, (5,)).tolist(),
                'temperature': 1.0 / np.sqrt(i + 1)
            }
            snapshots.append(snapshot)
        
        dynamics = DynamicsData(
            snapshots=snapshots,
            metadata={'game': 'test', 'player': 'test_agent'}
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            dynamics.save(f.name)
            temp_path = f.name
        
        # Load back
        loaded = DynamicsData.load(temp_path)
        
        assert len(loaded.snapshots) == 10
        assert loaded.metadata['game'] == 'test'
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_batch_extraction(self):
        """Should extract from multiple games in batch"""
        if MCTSDynamicsExtractor is None:
            pytest.skip("MCTSDynamicsExtractor not yet implemented")
            
        extractor = MCTSDynamicsExtractor()
        
        # Mock multiple games
        games = []
        for game_id in range(5):
            mock_game = Mock()
            mock_game.game_id = game_id
            mock_game.get_trajectory = Mock(return_value=[
                {'position': i, 'q_values': torch.randn(9)}
                for i in range(20)
            ])
            games.append(mock_game)
        
        # Batch extraction
        all_dynamics = extractor.extract_batch(games)
        
        assert len(all_dynamics) == 5
        assert all(len(d.snapshots) == 20 for d in all_dynamics)
    
    def test_filter_critical_positions(self):
        """Should filter for critical positions only"""
        if MCTSDynamicsExtractor is None:
            pytest.skip("MCTSDynamicsExtractor not yet implemented")
            
        extractor = MCTSDynamicsExtractor()
        
        # Create trajectory with mix of critical and non-critical
        trajectory = []
        for i in range(20):
            if i % 5 == 0:
                # Critical position: close Q-values
                q_values = torch.tensor([0.5, 0.49, 0.2, 0.1, 0.0])
            else:
                # Non-critical: clear best move
                q_values = torch.tensor([0.8, 0.3, 0.2, 0.1, 0.0])
            
            trajectory.append({
                'position_id': i,
                'q_values': q_values,
                'visits': torch.ones(5) * 100
            })
        
        # Extract with critical filter
        config = ExtractionConfig(
            extract_q_values=True,
            filter_critical=True,
            critical_threshold=0.05
        )
        
        dynamics = extractor.extract_trajectory_dynamics(
            trajectory, config=config
        )
        
        # Should only have critical positions
        assert len(dynamics.snapshots) == 4  # positions 0, 5, 10, 15
    
    def test_compute_derived_quantities(self):
        """Should compute temperature, entropy etc."""
        if MCTSDynamicsExtractor is None:
            pytest.skip("MCTSDynamicsExtractor not yet implemented")
            
        extractor = MCTSDynamicsExtractor()
        
        # Sample position data
        position_data = {
            'q_values': torch.tensor([0.5, 0.3, 0.2, 0.1, 0.0]),
            'visits': torch.tensor([400.0, 300.0, 200.0, 80.0, 20.0]),
            'total_visits': 1000
        }
        
        # Compute derived quantities
        derived = extractor.compute_derived_quantities(position_data)
        
        assert 'temperature' in derived
        assert 'entropy' in derived
        assert 'energy' in derived
        assert derived['temperature'] > 0
        assert derived['entropy'] > 0
    
    def test_parallel_extraction(self):
        """Should support parallel extraction for efficiency"""
        if MCTSDynamicsExtractor is None:
            pytest.skip("MCTSDynamicsExtractor not yet implemented")
            
        extractor = MCTSDynamicsExtractor(n_workers=4)
        
        # Mock many games
        games = [Mock() for _ in range(20)]
        for i, game in enumerate(games):
            game.game_id = i
            game.get_trajectory = Mock(return_value=[
                {'position': j} for j in range(10)
            ])
        
        # Extract in parallel
        import time
        start = time.time()
        results = extractor.extract_batch(games)
        duration = time.time() - start
        
        assert len(results) == 20
        # Parallel should be reasonably fast
        assert duration < 5.0  # seconds


class TestExtractionConfig:
    """Test extraction configuration"""
    
    def test_config_validation(self):
        """Should validate configuration parameters"""
        if ExtractionConfig is None:
            pytest.skip("ExtractionConfig not yet implemented")
            
        # Valid config
        config = ExtractionConfig(
            extract_q_values=True,
            time_window=100,
            sampling_rate=1
        )
        assert config.is_valid()
        
        # Invalid sampling rate
        with pytest.raises(ValueError):
            ExtractionConfig(sampling_rate=0)
        
        # Invalid time window
        with pytest.raises(ValueError):
            ExtractionConfig(time_window=-1)
    
    def test_config_serialization(self):
        """Should serialize/deserialize config"""
        if ExtractionConfig is None:
            pytest.skip("ExtractionConfig not yet implemented")
            
        config = ExtractionConfig(
            extract_q_values=True,
            extract_visits=True,
            filter_critical=True,
            critical_threshold=0.05
        )
        
        # To dict
        config_dict = config.to_dict()
        assert config_dict['filter_critical'] == True
        
        # From dict
        loaded = ExtractionConfig.from_dict(config_dict)
        assert loaded.critical_threshold == 0.05


class TestDynamicsData:
    """Test dynamics data structure"""
    
    def test_data_structure(self):
        """DynamicsData should store trajectory snapshots"""
        if DynamicsData is None:
            pytest.skip("DynamicsData not yet implemented")
            
        snapshots = [
            {
                'timestamp': 0,
                'q_values': [0.5, 0.3, 0.2],
                'visits': [400, 300, 200]
            }
        ]
        
        data = DynamicsData(
            snapshots=snapshots,
            metadata={'game': 'gomoku'}
        )
        
        assert len(data.snapshots) == 1
        assert data.metadata['game'] == 'gomoku'
    
    def test_data_compression(self):
        """Should support compressed storage"""
        if DynamicsData is None:
            pytest.skip("DynamicsData not yet implemented")
            
        # Large dataset
        snapshots = []
        for i in range(1000):
            snapshots.append({
                'timestamp': i,
                'q_values': np.random.randn(100).tolist(),
                'visits': np.random.randint(1, 1000, 100).tolist()
            })
        
        data = DynamicsData(snapshots=snapshots)
        
        # Save compressed
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            data.save_compressed(f.name)
            temp_path = f.name
        
        # Check file size is reasonable
        file_size = Path(temp_path).stat().st_size
        assert file_size < 1_000_000  # Less than 1MB
        
        # Load back
        loaded = DynamicsData.load_compressed(temp_path)
        assert len(loaded.snapshots) == 1000
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_data_streaming(self):
        """Should support streaming for large datasets"""
        if DynamicsData is None:
            pytest.skip("DynamicsData not yet implemented")
            
        # Create streaming data handler
        with tempfile.TemporaryDirectory() as tmpdir:
            stream = DynamicsData.create_stream(
                Path(tmpdir) / "stream.jsonl"
            )
            
            # Write snapshots incrementally
            for i in range(100):
                snapshot = {
                    'timestamp': i,
                    'q_values': torch.randn(5).tolist()
                }
                stream.write_snapshot(snapshot)
            
            stream.close()
            
            # Read back in chunks
            reader = DynamicsData.read_stream(
                Path(tmpdir) / "stream.jsonl",
                chunk_size=10
            )
            
            chunks = list(reader)
            assert len(chunks) == 10
            assert all(len(chunk) == 10 for chunk in chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])