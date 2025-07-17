"""
Tests for quantum tunneling detection in MCTS.

Following TDD principles - these tests are written before implementation.
"""
import pytest
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Import to-be-implemented modules
import sys
import os

# Set library path for C++ extensions
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.environ['LD_LIBRARY_PATH'] = f"{os.path.join(project_root, 'python')}:{os.environ.get('LD_LIBRARY_PATH', '')}"

try:
    from python.mcts.quantum.phenomena.tunneling import (
        TunnelingDetector, TunnelingEvent, ValueBarrier
    )
except ImportError:
    # Expected to fail before implementation
    TunnelingDetector = None
    TunnelingEvent = None
    ValueBarrier = None


@dataclass
class MockGameHistory:
    """Mock game history for testing"""
    id: int
    moves: List[int]
    value_trajectory: List[float]
    final_outcome: float
    selected_path: List[int]


def create_tunneling_game() -> MockGameHistory:
    """Create game with clear tunneling event"""
    # Value goes down (barrier) then up (tunneling success)
    value_trajectory = [
        0.5,   # Initial position
        0.4,   # Slightly worse
        0.2,   # Enter barrier (bad position)
        0.1,   # Bottom of barrier
        0.3,   # Starting to recover
        0.6,   # Escaped barrier
        0.8,   # Better than initial
    ]
    
    return MockGameHistory(
        id=1,
        moves=list(range(len(value_trajectory))),
        value_trajectory=value_trajectory,
        final_outcome=1.0,  # Won the game
        selected_path=list(range(len(value_trajectory)))
    )


def create_normal_game() -> MockGameHistory:
    """Create game without tunneling (monotonic improvement)"""
    value_trajectory = [
        0.5,   # Initial
        0.55,  # Better
        0.6,   # Better
        0.65,  # Better
        0.7,   # Better
        0.75,  # Better
    ]
    
    return MockGameHistory(
        id=2,
        moves=list(range(len(value_trajectory))),
        value_trajectory=value_trajectory,
        final_outcome=1.0,
        selected_path=list(range(len(value_trajectory)))
    )


class TestTunnelingDetector:
    """Test suite for tunneling detection"""
    
    def test_detector_exists(self):
        """TunnelingDetector class should exist"""
        assert TunnelingDetector is not None, "TunnelingDetector not implemented"
    
    def test_basic_detection(self):
        """Should detect basic tunneling events"""
        if TunnelingDetector is None:
            pytest.skip("TunnelingDetector not yet implemented")
            
        detector = TunnelingDetector()
        game = create_tunneling_game()
        
        events = detector.detect_tunneling_events([game])
        
        assert len(events) > 0, "Should detect tunneling in test game"
        assert isinstance(events[0], TunnelingEvent)
    
    def test_no_false_positives(self):
        """Should not detect tunneling in monotonic games"""
        if TunnelingDetector is None:
            pytest.skip("TunnelingDetector not yet implemented")
            
        detector = TunnelingDetector()
        game = create_normal_game()
        
        events = detector.detect_tunneling_events([game])
        
        assert len(events) == 0, "Should not detect tunneling in monotonic game"
    
    def test_barrier_identification(self):
        """Should correctly identify value barriers"""
        if TunnelingDetector is None:
            pytest.skip("TunnelingDetector not yet implemented")
            
        detector = TunnelingDetector()
        
        # U-shaped trajectory
        trajectory = [0.5, 0.4, 0.2, 0.1, 0.3, 0.5, 0.6]
        barriers = detector._find_value_barriers(trajectory)
        
        assert len(barriers) > 0
        barrier = barriers[0]
        
        assert barrier.entry_index < barrier.bottom_index
        assert barrier.bottom_index < barrier.exit_index
        assert barrier.height > 0
    
    def test_tunneling_validation(self):
        """Should validate tunneling events properly"""
        if TunnelingDetector is None:
            pytest.skip("TunnelingDetector not yet implemented")
            
        detector = TunnelingDetector()
        game = create_tunneling_game()
        
        trajectory = game.value_trajectory
        barriers = detector._find_value_barriers(trajectory)
        
        # Should validate as true tunneling (good final outcome)
        assert len(barriers) > 0
        is_valid = detector._validate_tunneling_event(barriers[0], game)
        assert is_valid
    
    def test_tunneling_metrics(self):
        """Should compute correct tunneling metrics"""
        if TunnelingDetector is None:
            pytest.skip("TunnelingDetector not yet implemented")
            
        detector = TunnelingDetector()
        game = create_tunneling_game()
        
        events = detector.detect_tunneling_events([game])
        assert len(events) > 0
        
        event = events[0]
        
        # Check required fields
        assert hasattr(event, 'game_id')
        assert hasattr(event, 'barrier_height')
        assert hasattr(event, 'tunnel_duration')
        assert hasattr(event, 'initial_disadvantage')
        assert hasattr(event, 'final_advantage')
        
        # Metrics should make sense
        assert event.barrier_height > 0
        assert event.tunnel_duration > 0
    
    def test_batch_detection(self):
        """Should handle multiple games efficiently"""
        if TunnelingDetector is None:
            pytest.skip("TunnelingDetector not yet implemented")
            
        detector = TunnelingDetector()
        
        games = [
            create_tunneling_game(),
            create_normal_game(),
            create_tunneling_game(),
        ]
        
        events = detector.detect_tunneling_events(games)
        
        # Should find events in games 0 and 2
        game_ids = [e.game_id for e in events]
        assert 1 in game_ids
        assert 2 not in game_ids
        assert 3 in game_ids  # Adjusted ID in third game
    
    def test_threshold_sensitivity(self):
        """Detection should be configurable via thresholds"""
        if TunnelingDetector is None:
            pytest.skip("TunnelingDetector not yet implemented")
            
        # Strict detector
        strict_detector = TunnelingDetector(min_barrier_height=0.4)
        
        # Lenient detector  
        lenient_detector = TunnelingDetector(min_barrier_height=0.1)
        
        game = create_tunneling_game()
        
        strict_events = strict_detector.detect_tunneling_events([game])
        lenient_events = lenient_detector.detect_tunneling_events([game])
        
        # Lenient should find more or equal events
        assert len(lenient_events) >= len(strict_events)


class TestTunnelingEvent:
    """Test TunnelingEvent data structure"""
    
    def test_event_structure(self):
        """TunnelingEvent should store event information"""
        if TunnelingEvent is None:
            pytest.skip("TunnelingEvent not yet implemented")
            
        event = TunnelingEvent(
            game_id=1,
            barrier_height=0.4,
            tunnel_duration=3,
            initial_disadvantage=-0.3,
            final_advantage=0.5,
            entry_move=2,
            exit_move=5,
            path=[2, 3, 4, 5]
        )
        
        assert event.game_id == 1
        assert event.barrier_height == 0.4
        assert event.tunnel_duration == 3
    
    def test_event_export(self):
        """Should export to dictionary"""
        if TunnelingEvent is None:
            pytest.skip("TunnelingEvent not yet implemented")
            
        event = TunnelingEvent(
            game_id=1,
            barrier_height=0.4,
            tunnel_duration=3,
            initial_disadvantage=-0.3,
            final_advantage=0.5,
            entry_move=2,
            exit_move=5,
            path=[2, 3, 4, 5]
        )
        
        data = event.to_dict()
        
        assert isinstance(data, dict)
        assert data['game_id'] == 1
        assert 'barrier_height' in data


class TestValueBarrier:
    """Test ValueBarrier detection"""
    
    def test_barrier_structure(self):
        """ValueBarrier should represent U-shaped value trajectory"""
        if ValueBarrier is None:
            pytest.skip("ValueBarrier not yet implemented")
            
        barrier = ValueBarrier(
            entry_index=1,
            bottom_index=3,
            exit_index=5,
            entry_value=0.5,
            bottom_value=0.1,
            exit_value=0.6,
            height=0.4,
            duration=4
        )
        
        assert barrier.height == 0.4
        assert barrier.duration == 4
        assert barrier.bottom_value < barrier.entry_value
        assert barrier.exit_value > barrier.bottom_value


class TestTunnelingStatistics:
    """Test tunneling statistics computation"""
    
    def test_tunneling_rate(self):
        """Should compute tunneling rate vs temperature"""
        if TunnelingDetector is None:
            pytest.skip("TunnelingDetector not yet implemented")
            
        detector = TunnelingDetector()
        
        # Create games at different temperatures
        games_low_temp = [create_tunneling_game() for _ in range(10)]
        games_high_temp = [create_tunneling_game() for _ in range(10)]
        
        # Tag with temperature
        for g in games_low_temp:
            g.temperature = 0.1
        for g in games_high_temp:
            g.temperature = 2.0
            
        stats = detector.compute_tunneling_statistics(
            games_low_temp + games_high_temp
        )
        
        assert 'tunneling_rate_vs_temperature' in stats
        assert len(stats['tunneling_rate_vs_temperature']) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])