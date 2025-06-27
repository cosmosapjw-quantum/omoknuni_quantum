"""Comprehensive tests for ResNet neural network evaluator

This module tests the ResNet evaluator implementation including:
- Model initialization and configuration
- Single and batch evaluation
- Tensor/numpy mode switching
- Mixed precision support
- Checkpoint saving/loading
- Game-specific model creation
- Performance tracking
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import json

from mcts.neural_networks.resnet_evaluator import (
    ResNetEvaluator,
    create_evaluator_for_game,
    create_chess_evaluator,
    create_go_evaluator,
    create_gomoku_evaluator
)
from mcts.core.evaluator import EvaluatorConfig
# Use the simple mock classes defined in resnet_evaluator.py instead
from mcts.neural_networks.resnet_evaluator import BaseGameModel, ModelMetadata


class MockGameModel(BaseGameModel):
    """Mock game model for testing"""
    def __init__(self, num_actions=225, input_channels=20):
        super().__init__()
        self.metadata = ModelMetadata(
            game_type='gomoku',
            board_size=15,
            num_actions=num_actions,
            input_channels=input_channels,
            num_blocks=5,
            num_filters=64,
            version='1.0',
            training_steps=0
        )
        
        # Simple mock network
        self.conv = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.policy_head = nn.Linear(32 * 15 * 15, num_actions)
        self.value_head = nn.Linear(32 * 15 * 15, 1)
    
    def forward(self, x):
        """Simple forward pass"""
        batch_size = x.shape[0]
        x = torch.relu(self.conv(x))
        x = x.view(batch_size, -1)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
    def get_config(self):
        """Get model configuration"""
        return {
            'num_actions': self.metadata.num_actions,
            'input_channels': self.metadata.input_channels,
            'architecture': 'mock'
        }
    
    def count_parameters(self):
        """Count model parameters"""
        return sum(p.numel() for p in self.parameters())


class TestResNetEvaluatorInitialization:
    """Test ResNet evaluator initialization"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model"""
        return MockGameModel()
    
    @pytest.fixture
    def config(self):
        """Create evaluator config"""
        return EvaluatorConfig(device='cpu', use_fp16=False)
    
    def test_initialization_with_model(self, mock_model, config):
        """Test initialization with pre-loaded model"""
        evaluator = ResNetEvaluator(model=mock_model, config=config)
        
        assert evaluator.model == mock_model
        assert evaluator.device.type == 'cpu'
        assert evaluator.action_size == 225
        assert evaluator.eval_count == 0
        assert evaluator.total_time == 0.0
    
    def test_initialization_with_game_type(self):
        """Test initialization with game type"""
        with patch('mcts.neural_networks.resnet_evaluator.create_resnet_for_game') as mock_create:
            mock_create.return_value = MockGameModel()
            
            evaluator = ResNetEvaluator(game_type='gomoku', device='cpu')
            
            mock_create.assert_called_once_with('gomoku', input_channels=20)
            assert evaluator.model is not None
    
    def test_auto_device_detection(self):
        """Test automatic device detection"""
        with patch('torch.cuda.is_available', return_value=True):
            evaluator = ResNetEvaluator(model=MockGameModel(), device=None)
            assert evaluator.device.type == 'cuda'
        
        with patch('torch.cuda.is_available', return_value=False):
            evaluator = ResNetEvaluator(model=MockGameModel(), device=None)
            assert evaluator.device.type == 'cpu'
    
    def test_model_eval_mode(self, mock_model):
        """Test that model is set to eval mode"""
        mock_model.training = True
        evaluator = ResNetEvaluator(model=mock_model, device='cpu')
        
        assert not evaluator.model.training  # Should be in eval mode
    
    def test_mixed_precision_setup(self, mock_model):
        """Test mixed precision setup"""
        config = EvaluatorConfig(device='cuda', use_fp16=True)
        
        with patch('torch.cuda.is_available', return_value=True):
            evaluator = ResNetEvaluator(model=mock_model, config=config)
            
            assert evaluator.use_amp == True
            assert hasattr(evaluator, 'scaler')
    
    def test_checkpoint_loading(self):
        """Test loading from checkpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock checkpoint
            checkpoint_path = Path(tmpdir) / 'model.pt'
            mock_model = MockGameModel()
            checkpoint = {
                'model_state_dict': mock_model.state_dict(),
                'model_config': mock_model.get_config()
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Create metadata
            metadata_path = Path(tmpdir) / 'metadata.json'
            mock_model.metadata.save(str(metadata_path))
            
            # Mock ModelLoader
            with patch('mcts.neural_networks.resnet_evaluator.ModelLoader.load_checkpoint') as mock_load:
                mock_load.return_value = (mock_model, mock_model.metadata)
                
                evaluator = ResNetEvaluator(checkpoint_path=str(checkpoint_path))
                
                # Check that load_checkpoint was called with the checkpoint path
                # Device may be 'cuda' or 'cpu' depending on availability
                assert mock_load.call_count == 1
                assert mock_load.call_args[0][0] == str(checkpoint_path)
                assert mock_load.call_args[0][1] in ['cpu', 'cuda']
                assert evaluator.model is not None


class TestResNetEvaluation:
    """Test evaluation functionality"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing"""
        model = MockGameModel()
        return ResNetEvaluator(model=model, device='cpu')
    
    def test_single_evaluation(self, evaluator):
        """Test single position evaluation"""
        # Create test state
        state = np.random.rand(20, 15, 15).astype(np.float32)
        legal_mask = np.ones(225, dtype=bool)
        legal_mask[0] = False  # One illegal move
        
        policy, value = evaluator.evaluate(state, legal_mask, temperature=1.0)
        
        assert isinstance(policy, np.ndarray)
        assert isinstance(value, float)
        assert policy.shape == (225,)
        assert -1 <= value <= 1
        assert np.allclose(policy.sum(), 1.0)
        assert policy[0] == 0  # Illegal move should have 0 probability
        
        # Check eval count
        assert evaluator.eval_count == 1
    
    def test_batch_evaluation(self, evaluator):
        """Test batch evaluation"""
        batch_size = 5
        states = np.random.rand(batch_size, 20, 15, 15).astype(np.float32)
        legal_masks = np.ones((batch_size, 225), dtype=bool)
        
        policies, values = evaluator.evaluate_batch(states, legal_masks, temperature=1.0)
        
        assert isinstance(policies, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert policies.shape == (batch_size, 225)
        assert values.shape == (batch_size,)
        assert np.allclose(policies.sum(axis=1), 1.0)
        assert np.all(values >= -1) and np.all(values <= 1)
        
        assert evaluator.eval_count == batch_size
    
    def test_tensor_input(self, evaluator):
        """Test evaluation with torch tensor input"""
        states = torch.rand(3, 20, 15, 15)
        legal_masks = torch.ones(3, 225, dtype=torch.bool)
        
        policies, values = evaluator.evaluate_batch(states, legal_masks)
        
        assert isinstance(policies, np.ndarray)
        assert isinstance(values, np.ndarray)
        assert policies.shape == (3, 225)
        assert values.shape == (3,)
    
    def test_temperature_scaling(self, evaluator):
        """Test temperature scaling effect"""
        state = np.random.rand(20, 15, 15).astype(np.float32)
        
        # Test with different temperatures
        policy_t1, _ = evaluator.evaluate(state, temperature=1.0)
        policy_t0, _ = evaluator.evaluate(state, temperature=0.1)  # Near deterministic
        
        # Lower temperature should be more peaked
        entropy_t1 = -np.sum(policy_t1 * np.log(policy_t1 + 1e-8))
        entropy_t0 = -np.sum(policy_t0 * np.log(policy_t0 + 1e-8))
        
        assert entropy_t0 < entropy_t1  # Lower temperature = lower entropy
    
    def test_legal_move_masking(self, evaluator):
        """Test legal move masking"""
        state = np.random.rand(20, 15, 15).astype(np.float32)
        
        # Only a few moves are legal
        legal_mask = np.zeros(225, dtype=bool)
        legal_moves = [10, 50, 100, 150]
        legal_mask[legal_moves] = True
        
        policy, _ = evaluator.evaluate(state, legal_mask)
        
        # Only legal moves should have non-zero probability
        assert np.all(policy[~legal_mask] == 0)
        assert np.allclose(policy[legal_mask].sum(), 1.0)
    
    def test_3d_state_handling(self, evaluator):
        """Test handling of 3D state (without batch dimension)"""
        state = np.random.rand(20, 15, 15).astype(np.float32)
        
        policy, value = evaluator.evaluate(state)
        
        assert policy.shape == (225,)
        assert isinstance(value, float)
    
    def test_tensor_return_mode(self):
        """Test returning torch tensors instead of numpy"""
        model = MockGameModel()
        evaluator = ResNetEvaluator(model=model, device='cpu')
        
        # Enable tensor return mode
        evaluator._return_torch_tensors = True
        
        states = torch.rand(2, 20, 15, 15)
        policies, values = evaluator.evaluate_batch(states)
        
        assert isinstance(policies, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        assert policies.device.type == 'cpu'
        assert values.device.type == 'cpu'


class TestCheckpointOperations:
    """Test checkpoint saving and loading"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing"""
        model = MockGameModel()
        return ResNetEvaluator(model=model, device='cpu')
    
    def test_save_checkpoint(self, evaluator):
        """Test saving checkpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
            
            # Save checkpoint
            evaluator.save_checkpoint(str(checkpoint_path), {'epoch': 10})
            
            # Check files created
            assert checkpoint_path.exists()
            assert (Path(tmpdir) / 'metadata.json').exists()
            
            # Load and verify checkpoint
            checkpoint = torch.load(checkpoint_path)
            assert 'model_state_dict' in checkpoint
            assert 'model_config' in checkpoint
            assert 'evaluator_config' in checkpoint
            assert checkpoint['epoch'] == 10
    
    def test_load_from_checkpoint(self):
        """Test loading from checkpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a checkpoint
            model = MockGameModel()
            evaluator1 = ResNetEvaluator(model=model, device='cpu')
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
            evaluator1.save_checkpoint(str(checkpoint_path))
            
            # Load from checkpoint
            with patch('mcts.neural_networks.resnet_evaluator.ModelLoader.load_checkpoint') as mock_load:
                mock_load.return_value = (model, model.metadata)
                
                evaluator2 = ResNetEvaluator.from_checkpoint(str(checkpoint_path))
                
                assert evaluator2 is not None
                assert isinstance(evaluator2, ResNetEvaluator)
    
    def test_checkpoint_directory_creation(self, evaluator):
        """Test that directories are created for checkpoint"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested directory that doesn't exist
            checkpoint_path = Path(tmpdir) / 'models' / 'iteration_5' / 'model.pt'
            
            evaluator.save_checkpoint(str(checkpoint_path))
            
            assert checkpoint_path.exists()
            assert checkpoint_path.parent.exists()


class TestGameSpecificCreation:
    """Test game-specific evaluator creation"""
    
    @patch('mcts.neural_networks.resnet_evaluator.create_resnet_for_game')
    def test_create_evaluator_for_game(self, mock_create_model):
        """Test creating evaluator for specific game"""
        mock_create_model.return_value = MockGameModel()
        
        evaluator = create_evaluator_for_game(
            'chess',
            num_blocks=10,
            num_filters=128,
            device='cpu'
        )
        
        mock_create_model.assert_called_once_with(
            'chess',
            input_channels=20,
            num_blocks=10,
            num_filters=128
        )
        assert isinstance(evaluator, ResNetEvaluator)
    
    @patch('mcts.neural_networks.resnet_evaluator.create_resnet_for_game')
    def test_create_chess_evaluator(self, mock_create_model):
        """Test chess evaluator creation"""
        mock_create_model.return_value = MockGameModel()
        
        evaluator = create_chess_evaluator(num_blocks=20)
        
        mock_create_model.assert_called_with('chess', input_channels=20, num_blocks=20, num_filters=256)
    
    @patch('mcts.neural_networks.resnet_evaluator.create_resnet_for_game')
    def test_create_go_evaluator(self, mock_create_model):
        """Test Go evaluator creation"""
        mock_create_model.return_value = MockGameModel()
        
        evaluator = create_go_evaluator(num_filters=192)
        
        mock_create_model.assert_called_with('go', input_channels=20, num_blocks=20, num_filters=192)
    
    @patch('mcts.neural_networks.resnet_evaluator.create_resnet_for_game')
    def test_create_gomoku_evaluator(self, mock_create_model):
        """Test Gomoku evaluator creation"""
        mock_create_model.return_value = MockGameModel()
        
        evaluator = create_gomoku_evaluator()
        
        mock_create_model.assert_called_with('gomoku', input_channels=20, num_blocks=20, num_filters=256)
    
    def test_checkpoint_loading_in_creation(self):
        """Test loading from checkpoint in creation functions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'model.pt'
            
            # Create mock checkpoint
            model = MockGameModel()
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model.get_config()
            }, checkpoint_path)
            model.metadata.save(str(Path(tmpdir) / 'metadata.json'))
            
            with patch('mcts.neural_networks.resnet_evaluator.ModelLoader.load_checkpoint') as mock_load:
                mock_load.return_value = (model, model.metadata)
                
                evaluator = create_evaluator_for_game(
                    'gomoku',
                    checkpoint_path=str(checkpoint_path)
                )
                
                assert evaluator is not None


class TestPerformanceTracking:
    """Test performance statistics"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing"""
        model = MockGameModel()
        return ResNetEvaluator(model=model, device='cpu')
    
    def test_evaluation_counting(self, evaluator):
        """Test evaluation counting"""
        # Single evaluation
        state = np.random.rand(20, 15, 15)
        evaluator.evaluate(state)
        assert evaluator.eval_count == 1
        
        # Batch evaluation
        states = np.random.rand(5, 20, 15, 15)
        evaluator.evaluate_batch(states)
        assert evaluator.eval_count == 6  # 1 + 5
    
    def test_get_stats(self, evaluator):
        """Test statistics retrieval"""
        # Run some evaluations
        states = np.random.rand(10, 20, 15, 15)
        evaluator.evaluate_batch(states)
        
        stats = evaluator.get_stats()
        
        assert 'eval_count' in stats
        assert 'total_time' in stats
        assert 'avg_time_per_eval' in stats
        assert 'model_params' in stats
        assert 'device' in stats
        assert 'use_amp' in stats
        
        assert stats['eval_count'] == 10
        assert stats['model_params'] > 0
        assert stats['device'] == 'cpu'
    
    def test_parameter_counting(self, evaluator):
        """Test model parameter counting"""
        stats = evaluator.get_stats()
        
        # Should match actual parameter count
        expected_params = evaluator.model.count_parameters()
        assert stats['model_params'] == expected_params


class TestMixedPrecision:
    """Test mixed precision support"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_amp_evaluation(self):
        """Test evaluation with automatic mixed precision"""
        model = MockGameModel()
        config = EvaluatorConfig(device='cuda', use_fp16=True)
        evaluator = ResNetEvaluator(model=model, config=config)
        
        states = torch.rand(4, 20, 15, 15, device='cuda')
        
        # Should work with AMP
        policies, values = evaluator.evaluate_batch(states)
        
        assert policies.shape == (4, 225)
        assert values.shape == (4,)
    
    def test_amp_disabled_on_cpu(self):
        """Test that AMP is disabled on CPU"""
        model = MockGameModel()
        config = EvaluatorConfig(device='cpu', use_fp16=True)
        evaluator = ResNetEvaluator(model=model, config=config)
        
        assert evaluator.use_amp == False  # Should be disabled on CPU


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator for testing"""
        model = MockGameModel()
        return ResNetEvaluator(model=model, device='cpu')
    
    def test_empty_batch(self, evaluator):
        """Test handling of empty batch"""
        states = np.array([]).reshape(0, 20, 15, 15)
        
        policies, values = evaluator.evaluate_batch(states)
        
        assert policies.shape == (0, 225)
        assert values.shape == (0,)
    
    def test_invalid_state_shape(self, evaluator):
        """Test handling of invalid state shape"""
        # Wrong number of channels
        state = np.random.rand(10, 15, 15)  # Should be 20 channels
        
        with pytest.raises(RuntimeError):
            evaluator.evaluate(state)
    
    def test_mismatched_legal_mask(self, evaluator):
        """Test mismatched legal mask shape"""
        state = np.random.rand(20, 15, 15)
        legal_mask = np.ones(100, dtype=bool)  # Wrong size
        
        with pytest.raises(RuntimeError):
            evaluator.evaluate(state, legal_mask)


class TestIntegration:
    """Integration tests"""
    
    def test_with_mcts_integration(self):
        """Test integration with MCTS"""
        model = MockGameModel()
        evaluator = ResNetEvaluator(model=model, device='cpu')
        
        # Enable tensor mode for MCTS
        evaluator._return_torch_tensors = True
        
        # Simulate MCTS batch evaluation
        batch_size = 32
        states = torch.rand(batch_size, 20, 15, 15)
        legal_masks = torch.ones(batch_size, 225, dtype=torch.bool)
        
        # Mask some random moves as illegal
        for i in range(batch_size):
            illegal_moves = torch.randperm(225)[:10]
            legal_masks[i, illegal_moves] = False
        
        policies, values = evaluator.evaluate_batch(states, legal_masks)
        
        # Check tensor properties for MCTS
        assert isinstance(policies, torch.Tensor)
        assert isinstance(values, torch.Tensor)
        assert policies.shape == (batch_size, 225)
        assert values.shape == (batch_size,)
        
        # Check masking worked
        assert torch.all(policies[~legal_masks] == 0)
        
        # Check normalization
        policy_sums = policies.sum(dim=1)
        assert torch.allclose(policy_sums, torch.ones(batch_size))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_cpu_consistency(self):
        """Test consistency between GPU and CPU evaluation"""
        # Create separate models with same weights
        torch.manual_seed(42)
        model_cpu = MockGameModel()
        torch.manual_seed(42)
        model_gpu = MockGameModel()
        
        # Create evaluators
        eval_cpu = ResNetEvaluator(model=model_cpu, device='cpu')
        eval_gpu = ResNetEvaluator(model=model_gpu, device='cuda')
        
        # Same input
        state = np.random.rand(20, 15, 15).astype(np.float32)
        
        policy_cpu, value_cpu = eval_cpu.evaluate(state)
        policy_gpu, value_gpu = eval_gpu.evaluate(state)
        
        # Results should be very close
        assert np.allclose(policy_cpu, policy_gpu, rtol=1e-5, atol=1e-5)
        assert np.allclose(value_cpu, value_gpu, rtol=1e-5, atol=1e-5)