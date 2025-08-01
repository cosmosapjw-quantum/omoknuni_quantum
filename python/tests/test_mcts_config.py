"""Tests for MCTS configuration with backend support

Following TDD principles, we start with tests for the backend parameter.
"""

import pytest
from mcts.core.mcts_config import MCTSConfig


class TestMCTSConfigBackend:
    """Test backend parameter in MCTSConfig"""
    
    def test_default_backend_is_gpu(self):
        """Test that default backend is GPU for backward compatibility"""
        config = MCTSConfig()
        assert config.backend == 'gpu'
    
    def test_can_set_backend_to_cpu(self):
        """Test that backend can be set to CPU"""
        config = MCTSConfig(backend='cpu')
        assert config.backend == 'cpu'
    
    def test_can_set_backend_to_gpu(self):
        """Test that backend can be explicitly set to GPU"""
        config = MCTSConfig(backend='gpu')
        assert config.backend == 'gpu'
    
    def test_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError"""
        with pytest.raises(ValueError, match="backend must be 'gpu' or 'cpu'"):
            MCTSConfig(backend='invalid')
    
    def test_backend_parameter_preserved_in_dict(self):
        """Test that backend parameter is preserved in to_dict()"""
        config = MCTSConfig(backend='cpu')
        config_dict = config.to_dict()
        assert 'backend' in config_dict
        assert config_dict['backend'] == 'cpu'