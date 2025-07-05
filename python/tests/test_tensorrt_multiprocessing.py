"""Test TensorRT pre-compilation with multiprocessing

This test verifies that TensorRT engines can be pre-compiled in the main process
and safely used by multiple worker processes without conflicts.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import multiprocessing as mp
import time
import tempfile
from pathlib import Path
from typing import Tuple, Optional

# Import necessary components
from mcts.utils.tensorrt_manager import get_tensorrt_manager, TensorRTManager
from mcts.neural_networks.tensorrt_converter import HAS_TENSORRT
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from mcts.neural_networks.resnet_model import ResNetModel
from mcts.utils.gpu_evaluator_service import GPUEvaluatorService
from mcts.neural_networks.self_play_module import SelfPlayManager
from mcts.utils.config_system import AlphaZeroConfig, MCTSFullConfig


# Skip all tests if TensorRT is not available or not working properly
def tensorrt_available():
    """Check if TensorRT is available and working"""
    if not HAS_TENSORRT:
        return False
    try:
        # Try a simple test to see if TensorRT actually works
        import torch
        dummy_model = torch.nn.Linear(10, 10).cuda()
        dummy_model.eval()
        dummy_input = torch.randn(1, 10).cuda()
        
        # Try to convert
        from mcts.neural_networks.tensorrt_converter import optimize_for_hardware
        trt_model = optimize_for_hardware(dummy_model, (10,))
        return trt_model is not None
    except Exception:
        return False

TENSORRT_WORKING = tensorrt_available()

@pytest.mark.skipif(not TENSORRT_WORKING, reason="TensorRT not available or not working")
class TestTensorRTMultiprocessing:
    """Test suite for TensorRT multiprocessing integration"""
    
    @pytest.fixture
    def test_model(self):
        """Create a simple test model"""
        from mcts.neural_networks.resnet_model import ResNetConfig
        
        config = ResNetConfig(
            num_blocks=2,  # Small for testing
            num_filters=32,
            input_channels=18,
            fc_value_hidden=128,
            fc_policy_hidden=128
        )
        
        model = ResNetModel(
            config=config,
            board_size=15,
            num_actions=225,
            game_type='gomoku'
        )
        model.eval()
        return model
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for TensorRT engines"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_tensorrt_precompilation(self, test_model, temp_cache_dir):
        """Test that TensorRT engine can be pre-compiled in main process"""
        # Get TensorRT manager with custom cache directory
        manager = get_tensorrt_manager(cache_dir=temp_cache_dir)
        
        # Define input shape for Gomoku
        input_shape = (18, 15, 15)
        
        # Pre-compile engine
        start_time = time.time()
        trt_model = manager.get_or_convert_model(
            pytorch_model=test_model,
            input_shape=input_shape,
            batch_size=128,
            fp16=True,
            workspace_size=1 << 30,  # 1GB
            worker_id=0
        )
        conversion_time = time.time() - start_time
        
        assert trt_model is not None, "TensorRT conversion failed"
        print(f"✅ TensorRT engine compiled in {conversion_time:.2f}s")
        
        # Verify engine file exists
        model_hash = manager.get_model_hash(test_model, input_shape)
        engine_path = manager.get_cached_engine_path(model_hash)
        assert engine_path.exists(), f"Engine file not found at {engine_path}"
        
        # Test inference
        dummy_input = torch.randn(1, *input_shape).cuda()
        with torch.no_grad():
            policy, value = trt_model(dummy_input)
        
        assert policy.shape == (1, 225), f"Invalid policy shape: {policy.shape}"
        assert value.shape == (1, 1), f"Invalid value shape: {value.shape}"
        print("✅ TensorRT inference successful")
    
    def test_multiprocess_engine_loading(self, test_model, temp_cache_dir):
        """Test that multiple processes can load the same pre-compiled engine"""
        # Pre-compile engine in main process
        manager = get_tensorrt_manager(cache_dir=temp_cache_dir)
        input_shape = (18, 15, 15)
        
        trt_model = manager.get_or_convert_model(
            pytorch_model=test_model,
            input_shape=input_shape,
            batch_size=128,
            fp16=True,
            workspace_size=1 << 30,
            worker_id=0
        )
        assert trt_model is not None
        
        # Get engine path for workers
        model_hash = manager.get_model_hash(test_model, input_shape)
        engine_path = manager.get_cached_engine_path(model_hash)
        
        # Function for worker processes
        def worker_test(worker_id: int, engine_path: Path, result_queue: mp.Queue):
            """Worker function to test engine loading"""
            try:
                # Each worker gets its own manager instance
                worker_manager = TensorRTManager(cache_dir=engine_path.parent)
                
                # Workers should load the pre-compiled engine
                start_time = time.time()
                trt_model = worker_manager.get_or_convert_model(
                    pytorch_model=test_model,
                    input_shape=input_shape,
                    batch_size=128,
                    fp16=True,
                    workspace_size=1 << 30,
                    worker_id=worker_id
                )
                load_time = time.time() - start_time
                
                assert trt_model is not None
                
                # Test inference
                dummy_input = torch.randn(4, *input_shape).cuda()
                with torch.no_grad():
                    policy, value = trt_model(dummy_input)
                
                result = {
                    'worker_id': worker_id,
                    'success': True,
                    'load_time': load_time,
                    'policy_shape': policy.shape,
                    'value_shape': value.shape
                }
                result_queue.put(result)
                
            except Exception as e:
                result_queue.put({
                    'worker_id': worker_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Start multiple worker processes
        num_workers = 4
        processes = []
        result_queue = mp.Queue()
        
        for i in range(num_workers):
            p = mp.Process(
                target=worker_test,
                args=(i, engine_path, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        results = []
        for _ in range(num_workers):
            result = result_queue.get(timeout=30)
            results.append(result)
        
        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        # Verify all workers succeeded
        for result in results:
            assert result['success'], f"Worker {result['worker_id']} failed: {result.get('error')}"
            assert result['load_time'] < 5.0, f"Worker {result['worker_id']} took too long to load: {result['load_time']:.2f}s"
            assert result['policy_shape'] == (4, 225)
            assert result['value_shape'] == (4, 1)
        
        print(f"✅ All {num_workers} workers successfully loaded and used the pre-compiled engine")
        avg_load_time = sum(r['load_time'] for r in results) / num_workers
        print(f"   Average load time: {avg_load_time:.3f}s")
    
    def test_gpu_service_with_tensorrt(self, test_model, temp_cache_dir):
        """Test GPU evaluator service with pre-compiled TensorRT engine"""
        # Pre-compile engine
        manager = get_tensorrt_manager(cache_dir=temp_cache_dir)
        input_shape = (18, 15, 15)
        
        trt_model = manager.get_or_convert_model(
            pytorch_model=test_model,
            input_shape=input_shape,
            batch_size=256,
            fp16=True,
            workspace_size=2 << 30,
            worker_id=0
        )
        
        # Get engine path
        model_hash = manager.get_model_hash(test_model, input_shape)
        engine_path = manager.get_cached_engine_path(model_hash)
        
        # Create GPU service with pre-compiled engine
        gpu_service = GPUEvaluatorService(
            model=test_model,
            device='cuda',
            batch_size=256,
            use_tensorrt=True,
            tensorrt_fp16=True,
            tensorrt_engine_path=str(engine_path)
        )
        
        # Verify TensorRT is being used
        assert gpu_service.use_tensorrt, "GPU service should use TensorRT"
        assert gpu_service.tensorrt_model is not None, "TensorRT model not loaded"
        
        print("✅ GPU evaluator service successfully initialized with pre-compiled TensorRT engine")
    
    def test_race_condition_prevention(self, test_model, temp_cache_dir):
        """Test that multiple processes trying to compile simultaneously don't conflict"""
        # Don't pre-compile - let workers race
        
        def worker_compile(worker_id: int, cache_dir: str, result_queue: mp.Queue):
            """Worker that tries to compile TensorRT engine"""
            try:
                manager = TensorRTManager(cache_dir=cache_dir)
                input_shape = (18, 15, 15)
                
                start_time = time.time()
                trt_model = manager.get_or_convert_model(
                    pytorch_model=test_model,
                    input_shape=input_shape,
                    batch_size=128,
                    fp16=True,
                    workspace_size=1 << 30,
                    worker_id=worker_id
                )
                compile_time = time.time() - start_time
                
                result_queue.put({
                    'worker_id': worker_id,
                    'success': trt_model is not None,
                    'compile_time': compile_time
                })
                
            except Exception as e:
                result_queue.put({
                    'worker_id': worker_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Start multiple workers simultaneously
        num_workers = 3
        processes = []
        result_queue = mp.Queue()
        
        for i in range(num_workers):
            p = mp.Process(
                target=worker_compile,
                args=(i, temp_cache_dir, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        results = []
        for _ in range(num_workers):
            result = result_queue.get(timeout=60)
            results.append(result)
        
        # Wait for processes
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        
        # Verify all succeeded
        successful_workers = [r for r in results if r['success']]
        assert len(successful_workers) == num_workers, f"Not all workers succeeded: {results}"
        
        # Check that only one worker actually compiled (others should have loaded)
        compile_times = [r['compile_time'] for r in successful_workers]
        long_compiles = [t for t in compile_times if t > 5.0]
        assert len(long_compiles) <= 1, f"Multiple workers compiled: {compile_times}"
        
        print("✅ Race condition prevention working correctly")
        print(f"   Compile times: {[f'{t:.2f}s' for t in compile_times]}")
    
    def test_full_training_pipeline_integration(self, temp_cache_dir):
        """Test full integration with training pipeline"""
        # Create minimal config for testing
        config = AlphaZeroConfig()
        config.game.game_type = 'gomoku'
        config.game.board_size = 15
        config.mcts.num_simulations = 10  # Very small for testing
        config.mcts.use_tensorrt = True
        config.mcts.tensorrt_fp16 = True
        config.mcts.tensorrt_engine_cache_dir = temp_cache_dir
        config.training.num_games_per_iteration = 2
        config.training.num_workers = 2
        config.training.batch_size = 32
        config.network.num_res_blocks = 2  # Small network
        config.network.num_filters = 32
        
        # Import here to avoid circular imports
        from mcts.neural_networks.unified_training_pipeline import UnifiedTrainingPipeline
        
        # Create pipeline
        pipeline = UnifiedTrainingPipeline(config)
        
        # Verify TensorRT pre-compilation happens
        assert hasattr(pipeline, 'tensorrt_engine_path'), "TensorRT engine path not set"
        assert pipeline.tensorrt_engine_path.exists(), "TensorRT engine not pre-compiled"
        
        print("✅ Training pipeline successfully pre-compiled TensorRT engine")
        print(f"   Engine path: {pipeline.tensorrt_engine_path}")


if __name__ == "__main__":
    # Run tests
    test = TestTensorRTMultiprocessing()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test model
        from mcts.neural_networks.resnet_model import ResNetConfig
        
        config = ResNetConfig(
            num_blocks=2,
            num_filters=32,
            input_channels=18,
            fc_value_hidden=128,
            fc_policy_hidden=128
        )
        
        model = ResNetModel(
            config=config,
            board_size=15,
            num_actions=225,
            game_type='gomoku'
        )
        model.eval()
        
        print("Running TensorRT multiprocessing tests...\n")
        
        # Run each test
        try:
            print("1. Testing TensorRT pre-compilation...")
            test.test_tensorrt_precompilation(model, tmpdir)
            print()
            
            print("2. Testing multiprocess engine loading...")
            test.test_multiprocess_engine_loading(model, tmpdir)
            print()
            
            print("3. Testing GPU service integration...")
            test.test_gpu_service_with_tensorrt(model, tmpdir)
            print()
            
            print("4. Testing race condition prevention...")
            test.test_race_condition_prevention(model, tmpdir)
            print()
            
            print("5. Testing full pipeline integration...")
            test.test_full_training_pipeline_integration(tmpdir)
            print()
            
            print("\n✅ All TensorRT multiprocessing tests passed!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()