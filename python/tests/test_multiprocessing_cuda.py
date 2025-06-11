"""Test multiprocessing with CUDA tensors to ensure proper handling"""

import multiprocessing
import pytest
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import unittest

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)


def worker_with_cuda_tensors(model_state_dict):
    """Worker that tries to use CUDA tensors from parent process"""
    # This should fail if model_state_dict contains CUDA tensors
    model = SimpleModel()
    model.load_state_dict(model_state_dict)
    
    # Try to use the model
    x = torch.randn(1, 10)
    output = model(x)
    return output.sum().item()


def worker_with_cpu_tensors(model_state_dict):
    """Worker that properly handles CPU tensors and moves to CUDA if needed"""
    model = SimpleModel()
    model.load_state_dict(model_state_dict)
    
    # Move to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Try to use the model
    x = torch.randn(1, 10)
    if torch.cuda.is_available():
        x = x.cuda()
    
    output = model(x)
    return output.sum().item()


class TestMultiprocessingCUDA(unittest.TestCase):
    """Test suite for multiprocessing with CUDA"""
    
    def test_cuda_tensor_pickling_fails(self):
        """Test that passing CUDA tensors to workers fails"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create model on CUDA
        model = SimpleModel().cuda()
        
        # This should fail when trying to pickle CUDA tensors
        with pytest.raises((RuntimeError, Exception)):
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(worker_with_cuda_tensors, model.state_dict())
                future.result()
    
    def test_cpu_tensor_pickling_succeeds(self):
        """Test that passing CPU tensors to workers succeeds"""
        # Create model (on CPU by default)
        model = SimpleModel()
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Move state dict to CPU
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # This should succeed
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for _ in range(4):
                future = executor.submit(worker_with_cpu_tensors, cpu_state_dict)
                futures.append(future)
            
            # All should complete successfully
            results = [f.result() for f in futures]
            assert len(results) == 4
            assert all(isinstance(r, float) for r in results)
    
    def test_self_play_module_integration(self):
        """Test that self-play module properly handles CUDA multiprocessing"""
        # Add parent directory to path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from mcts.neural_networks.self_play_module import SelfPlayManager
        from mcts.utils.config_system import AlphaZeroConfig, create_default_config
        
        # Create config
        config = create_default_config(game_type="gomoku")
        config.training.num_workers = 2
        config.training.num_games_per_iteration = 2
        config.mcts.num_simulations = 10  # Very small for testing
        config.mcts.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create self-play manager
        manager = SelfPlayManager(config)
        
        # Create a simple model with correct architecture for gomoku
        from mcts.neural_networks.nn_model import create_model
        model = create_model(
            game_type="gomoku",
            input_height=15,
            input_width=15,
            num_actions=225,
            input_channels=config.network.input_channels,
            num_res_blocks=config.network.num_res_blocks,
            num_filters=config.network.num_filters
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # This should not raise any CUDA errors
        examples = manager.generate_games(model, iteration=1, num_games=2, num_workers=2)
        
        # Verify we got examples
        assert len(examples) > 0
        assert all(hasattr(ex, 'state') for ex in examples)
        assert all(hasattr(ex, 'policy') for ex in examples)
        assert all(hasattr(ex, 'value') for ex in examples)


if __name__ == "__main__":
    # Run tests
    import unittest
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMultiprocessingCUDA)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)