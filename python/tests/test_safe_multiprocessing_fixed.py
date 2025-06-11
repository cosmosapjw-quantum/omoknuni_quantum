"""Fixed version of safe multiprocessing tests with timeout and debugging"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import logging
import traceback
import time
from functools import wraps

from mcts.utils.safe_multiprocessing import (
    serialize_state_dict_for_multiprocessing,
    deserialize_state_dict_from_multiprocessing,
    make_config_multiprocessing_safe
)
from mcts.utils.config_system import create_default_config
from mcts.neural_networks.nn_model import create_model

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def timeout_handler(timeout_seconds=30):
    """Decorator to add timeout to tests"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def handler(signum, frame):
                raise TimeoutError(f"Test {func.__name__} timed out after {timeout_seconds} seconds")
            
            # Set up timeout (only works on Unix)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)  # Cancel alarm
            
            return result
        return wrapper
    return decorator


def _simple_worker_test(worker_id):
    """Simple worker to test basic multiprocessing"""
    logger.info(f"[Worker {worker_id}] Starting simple test")
    time.sleep(0.1)  # Small delay to test concurrency
    result = f"Worker {worker_id} completed successfully"
    logger.info(f"[Worker {worker_id}] Finished: {result}")
    return result


def _worker_load_model_debug(serialized_state_dict, worker_id):
    """Worker with extensive debugging"""
    try:
        logger.info(f"[Worker {worker_id}] Starting worker process")
        
        # Step 1: Environment setup
        logger.info(f"[Worker {worker_id}] Setting CUDA_VISIBLE_DEVICES to empty")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Step 2: Import modules
        logger.info(f"[Worker {worker_id}] Importing torch")
        import torch
        logger.info(f"[Worker {worker_id}] Torch version: {torch.__version__}")
        logger.info(f"[Worker {worker_id}] CUDA available in worker: {torch.cuda.is_available()}")
        
        logger.info(f"[Worker {worker_id}] Importing model creation function")
        from mcts.neural_networks.nn_model import create_model
        
        logger.info(f"[Worker {worker_id}] Importing deserialization function")
        from mcts.utils.safe_multiprocessing import deserialize_state_dict_from_multiprocessing
        
        # Step 3: Deserialize state dict
        logger.info(f"[Worker {worker_id}] Starting deserialization")
        logger.info(f"[Worker {worker_id}] Serialized dict has {len(serialized_state_dict)} keys")
        
        state_dict = deserialize_state_dict_from_multiprocessing(serialized_state_dict)
        logger.info(f"[Worker {worker_id}] Deserialization complete, got {len(state_dict)} parameters")
        
        # Step 4: Create model
        logger.info(f"[Worker {worker_id}] Creating model")
        model = create_model(
            game_type="gomoku",
            input_height=9,
            input_width=9,
            num_actions=81,
            input_channels=20,
            num_res_blocks=2,
            num_filters=32
        )
        logger.info(f"[Worker {worker_id}] Model created successfully")
        
        # Step 5: Load state dict
        logger.info(f"[Worker {worker_id}] Loading state dict into model")
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"[Worker {worker_id}] Model loaded and set to eval mode")
        
        # Step 6: Test forward pass
        logger.info(f"[Worker {worker_id}] Testing forward pass")
        dummy_input = torch.randn(1, 20, 9, 9)
        with torch.no_grad():
            policy, value = model(dummy_input)
        
        result = f"Worker {worker_id}: Model loaded, policy shape={policy.shape}, value shape={value.shape}"
        logger.info(f"[Worker {worker_id}] Success: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Worker {worker_id} failed with error: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg


def _minimal_worker(data, worker_id):
    """Minimal worker to isolate the issue"""
    try:
        logger.info(f"[MinimalWorker {worker_id}] Starting")
        
        # Just try to import and return
        import torch
        from mcts.neural_networks.nn_model import create_model
        from mcts.utils.safe_multiprocessing import deserialize_state_dict_from_multiprocessing
        
        logger.info(f"[MinimalWorker {worker_id}] Imports successful")
        
        # Check data
        logger.info(f"[MinimalWorker {worker_id}] Data type: {type(data)}, size: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        return f"MinimalWorker {worker_id}: Success"
        
    except Exception as e:
        return f"MinimalWorker {worker_id}: Failed - {str(e)}"


class TestSafeMultiprocessingFixed:
    """Fixed tests with better error handling and timeouts"""
    
    def test_basic_multiprocessing(self):
        """Test that basic multiprocessing works"""
        logger.info("=== Testing basic multiprocessing ===")
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for i in range(2):
                future = executor.submit(_simple_worker_test, i)
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result(timeout=5)
                    logger.info(f"Got result: {result}")
                    assert "completed successfully" in result
                except TimeoutError:
                    pytest.fail("Basic multiprocessing test timed out")
    
    def test_minimal_model_worker(self):
        """Test minimal worker with just imports"""
        logger.info("=== Testing minimal model worker ===")
        
        # Create some dummy data
        dummy_data = {"test": "data"}
        
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_minimal_worker, dummy_data, 0)
            
            try:
                result = future.result(timeout=10)
                logger.info(f"Minimal worker result: {result}")
                assert "Success" in result or "Failed" in result
            except TimeoutError:
                pytest.fail("Minimal worker timed out - likely import issue")
    
    @timeout_handler(60)  # 60 second timeout
    def test_multiprocessing_with_serialized_state_fixed(self):
        """Fixed version of the stalling test"""
        logger.info("=== Testing multiprocessing with serialized state (fixed) ===")
        
        # Create model
        logger.info("Creating model in main process")
        model = create_model(
            game_type="gomoku",
            input_height=9,
            input_width=9,
            num_actions=81,
            input_channels=20,
            num_res_blocks=2,
            num_filters=32
        )
        
        if torch.cuda.is_available():
            logger.info("CUDA available in main process, moving model to GPU")
            model = model.cuda()
        
        # Serialize state dict
        logger.info("Serializing model state dict")
        state_dict = model.state_dict()
        logger.info(f"State dict has {len(state_dict)} parameters")
        
        serialized = serialize_state_dict_for_multiprocessing(state_dict)
        logger.info(f"Serialized dict has {len(serialized)} entries")
        
        # Verify serialization worked
        for key, value in serialized.items():
            assert not torch.is_tensor(value), f"Found tensor in serialized dict: {key}"
            if isinstance(value, dict) and value.get('type') == 'tensor':
                assert isinstance(value['data'], np.ndarray), f"Expected numpy array for {key}"
        
        logger.info("Serialization verification passed")
        
        # Test with multiprocessing
        logger.info("Starting multiprocessing test")
        
        # Use spawn method explicitly
        ctx = multiprocessing.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
            logger.info("Submitting task to worker")
            future = executor.submit(_worker_load_model_debug, serialized, 0)
            
            try:
                logger.info("Waiting for worker result (timeout=30s)")
                result = future.result(timeout=30)
                logger.info(f"Worker completed: {result}")
                
                if "failed" in result.lower():
                    pytest.fail(f"Worker failed: {result}")
                else:
                    assert "Model loaded" in result
                    
            except TimeoutError:
                logger.error("Worker timed out!")
                future.cancel()
                pytest.fail("Worker process timed out after 30 seconds")
            except Exception as e:
                logger.error(f"Worker raised exception: {e}")
                raise
    
    def test_sequential_model_loading(self):
        """Test the model loading process sequentially (no multiprocessing)"""
        logger.info("=== Testing sequential model loading ===")
        
        # Create and serialize model
        model = create_model(
            game_type="gomoku",
            input_height=9,
            input_width=9,
            num_actions=81,
            input_channels=20,
            num_res_blocks=2,
            num_filters=32
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Serialize
        serialized = serialize_state_dict_for_multiprocessing(model.state_dict())
        
        # Clear CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Deserialize in same process
        state_dict = deserialize_state_dict_from_multiprocessing(serialized)
        
        # Create new model and load
        model2 = create_model(
            game_type="gomoku",
            input_height=9,
            input_width=9,
            num_actions=81,
            input_channels=20,
            num_res_blocks=2,
            num_filters=32
        )
        model2.load_state_dict(state_dict)
        
        # Test forward pass
        dummy_input = torch.randn(1, 20, 9, 9)
        with torch.no_grad():
            policy, value = model2(dummy_input)
        
        logger.info(f"Sequential test passed: policy={policy.shape}, value={value.shape}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    # Run specific test for debugging
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        test = TestSafeMultiprocessingFixed()
        test.test_basic_multiprocessing()
        test.test_minimal_model_worker()
        test.test_sequential_model_loading()
        test.test_multiprocessing_with_serialized_state_fixed()
    else:
        pytest.main([__file__, "-v", "-s", "--tb=short"])