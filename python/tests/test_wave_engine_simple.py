"""Simple tests for wave engine to improve coverage"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import torch

from mcts.wave_engine import WaveEngine, WaveConfig, Wave
from mcts.node import Node


class TestWaveEngineCoverage:
    """Tests for wave engine coverage improvement"""
    
    def test_wave_config_defaults(self):
        """Test wave config default values"""
        config = WaveConfig()
        assert config.initial_wave_size == 512
        assert config.max_wave_size == 2048
        assert config.min_wave_size == 64
        
    def test_wave_creation(self):
        """Test wave creation"""
        wave = Wave(root_id="root", size=256)
        assert wave.root_id == "root"
        assert wave.size == 256
        assert wave.phase == 0
        assert len(wave.paths) == 0
        
    def test_wave_engine_init(self):
        """Test wave engine initialization"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig()
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        assert engine.game == game
        assert engine.evaluator == evaluator
        assert engine.arena == arena
        assert engine.config == config
        
    def test_create_wave_basic(self):
        """Test basic wave creation"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig(initial_wave_size=128)
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        wave = engine.create_wave("root_id", size=256)
        
        assert isinstance(wave, Wave)
        assert wave.root_id == "root_id"
        assert wave.size == 256
        
    def test_run_selection_phase_empty(self):
        """Test selection phase with empty wave"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig()
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Create empty wave
        wave = Wave("root", 10)
        
        # Mock arena
        root_node = Mock()
        root_node.is_terminal = False
        root_node.is_expanded = True
        root_node.select_child = Mock(return_value=(0, Mock()))
        arena.get_node = Mock(return_value=root_node)
        
        # Run selection
        engine._run_selection_phase(wave)
        
        # Should have paths now
        assert len(wave.paths) == 10
        
    def test_run_expansion_phase_no_expansion_needed(self):
        """Test expansion when all nodes are expanded"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig()
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Create wave with expanded nodes
        wave = Wave("root", 2)
        expanded_node = Mock()
        expanded_node.is_expanded = True
        expanded_node.is_terminal = False
        wave.leaf_nodes = [expanded_node, expanded_node]
        
        # Run expansion
        engine._run_expansion_phase(wave)
        
        # No expansion should happen
        assert not expanded_node.expand.called
        
    def test_run_evaluation_phase_terminal_nodes(self):
        """Test evaluation with terminal nodes"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig()
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Create wave with terminal nodes
        wave = Wave("root", 2)
        terminal_node = Mock()
        terminal_node.is_terminal = True
        terminal_node.terminal_value = 1.0
        wave.leaf_nodes = [terminal_node, terminal_node]
        
        # Run evaluation
        values = engine._run_evaluation_phase(wave)
        
        assert values == [1.0, 1.0]
        
    def test_run_backup_phase(self):
        """Test backup phase"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig()
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Create wave with paths
        wave = Wave("root", 2)
        node1 = Mock()
        node2 = Mock()
        wave.paths = [[node1], [node2]]
        
        # Run backup
        values = [0.5, -0.5]
        engine._run_backup_phase(wave, values)
        
        # Check updates
        assert node1.update.called
        assert node2.update.called
        
    def test_process_wave_complete(self):
        """Test complete wave processing"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig()
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Mock all phases
        engine._run_selection_phase = Mock()
        engine._run_expansion_phase = Mock()
        engine._run_evaluation_phase = Mock(return_value=[0.5] * 10)
        engine._run_backup_phase = Mock()
        
        # Process wave
        wave = Wave("root", 10)
        engine.process_wave(wave)
        
        # Check all phases called
        assert engine._run_selection_phase.called
        assert engine._run_expansion_phase.called
        assert engine._run_evaluation_phase.called
        assert engine._run_backup_phase.called
        
    def test_compute_wave_diversity(self):
        """Test wave diversity computation"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig()
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Create wave with diverse paths
        wave = Wave("root", 3)
        wave.paths = [
            [Mock(), Mock(action=0)],
            [Mock(), Mock(action=1)],
            [Mock(), Mock(action=0)]
        ]
        
        diversity = engine._compute_wave_diversity(wave)
        assert 0 <= diversity <= 1
        
    def test_get_statistics(self):
        """Test statistics collection"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig()
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Set some stats
        engine.stats = {
            'waves_processed': 10,
            'total_paths': 1000,
            'avg_wave_size': 100.0
        }
        
        stats = engine.get_statistics()
        
        assert stats['waves_processed'] == 10
        assert stats['total_paths'] == 1000
        assert stats['avg_wave_size'] == 100.0
        
    def test_reset_statistics(self):
        """Test statistics reset"""
        game = Mock()
        evaluator = Mock()
        arena = Mock()
        config = WaveConfig()
        
        engine = WaveEngine(game, evaluator, arena, config)
        
        # Set some stats
        engine.stats['waves_processed'] = 10
        
        # Reset
        engine.reset_statistics()
        
        assert engine.stats['waves_processed'] == 0