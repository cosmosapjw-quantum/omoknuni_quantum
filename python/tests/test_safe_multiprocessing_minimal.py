"""Minimal safe multiprocessing test that avoids model creation in workers"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import logging

from mcts.utils.safe_multiprocessing import (
    serialize_state_dict_for_multiprocessing,
    deserialize_state_dict_from_multiprocessing,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _worker_validate_serialized_data(serialized_state_dict, worker_id):
    """Worker that only validates serialized data without creating model"""
    try:
        logger.info(f"[Worker {worker_id}] Starting validation")
        
        # Verify all entries are properly serialized
        num_tensors = 0
        num_other = 0
        total_elements = 0
        
        for key, value in serialized_state_dict.items():
            if isinstance(value, dict) and value.get('type') == 'tensor':
                num_tensors += 1
                # Verify it's numpy array
                assert isinstance(value['data'], np.ndarray), f"Expected numpy array for {key}"
                total_elements += value['data'].size
            else:
                num_other += 1
        
        logger.info(f"[Worker {worker_id}] Found {num_tensors} tensors, {num_other} other values")
        logger.info(f"[Worker {worker_id}] Total elements: {total_elements}")
        
        # Test deserialization without creating model
        logger.info(f"[Worker {worker_id}] Testing deserialization")
        from mcts.utils.safe_multiprocessing import deserialize_state_dict_from_multiprocessing
        
        state_dict = deserialize_state_dict_from_multiprocessing(serialized_state_dict)
        
        # Verify deserialization worked
        for key, value in state_dict.items():
            assert torch.is_tensor(value), f"Expected tensor for {key}, got {type(value)}"
        
        result = f"Worker {worker_id}: Validated {len(state_dict)} parameters successfully"
        logger.info(f"[Worker {worker_id}] Success: {result}")
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Worker {worker_id} failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return error_msg


class TestSafeMultiprocessingMinimal:
    """Minimal tests for safe multiprocessing"""
    
    def test_serialization_only(self):
        """Test serialization without model creation in workers"""
        logger.info("=== Testing serialization only ===")
        
        # Create model in main process
        from mcts.neural_networks.nn_model import create_model
        
        model = create_model(
            game_type="gomoku",
            input_height=9,
            input_width=9,
            num_actions=81,
            input_channels=20,
            num_res_blocks=2,
            num_filters=32
        )
        
        # Get state dict and serialize
        state_dict = model.state_dict()
        logger.info(f"Model has {len(state_dict)} parameters")
        
        serialized = serialize_state_dict_for_multiprocessing(state_dict)
        logger.info(f"Serialized to {len(serialized)} entries")
        
        # Verify no tensors in serialized data
        for key, value in serialized.items():
            assert not torch.is_tensor(value), f"Found tensor {key} in serialized data"
        
        # Test with multiprocessing - only validate, don't create model
        ctx = multiprocessing.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as executor:
            futures = []
            
            for i in range(2):
                logger.info(f"Submitting validation worker {i}")
                future = executor.submit(_worker_validate_serialized_data, serialized, i)
                futures.append(future)
            
            # Get results
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=10)
                    logger.info(f"Worker {i} result: {result}")
                    assert "Validated" in result and "successfully" in result
                except Exception as e:
                    logger.error(f"Worker {i} failed: {e}")
                    raise
    
    def test_roundtrip_same_process(self):
        """Test serialization roundtrip in same process"""
        from mcts.neural_networks.nn_model import create_model
        
        # Create two models
        model1 = create_model(
            game_type="gomoku",
            input_height=9,
            input_width=9,
            num_actions=81,
            input_channels=20,
            num_res_blocks=2,
            num_filters=32
        )
        
        model2 = create_model(
            game_type="gomoku",
            input_height=9,
            input_width=9,
            num_actions=81,
            input_channels=20,
            num_res_blocks=2,
            num_filters=32
        )
        
        # Get state dict from model1
        state_dict1 = model1.state_dict()
        
        # Serialize and deserialize
        serialized = serialize_state_dict_for_multiprocessing(state_dict1)
        restored = deserialize_state_dict_from_multiprocessing(serialized)
        
        # Load into model2
        model2.load_state_dict(restored)
        
        # Verify parameters match
        for key in state_dict1:
            param1 = model1.state_dict()[key]
            param2 = model2.state_dict()[key]
            assert torch.allclose(param1, param2), f"Parameter {key} mismatch"
        
        logger.info("Roundtrip test passed")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    pytest.main([__file__, "-v", "-s"])