#!/usr/bin/env python3
"""
Neural Network Evaluation Validation for CPU Backend

This test suite validates that neural network evaluation works correctly
in the CPU backend, comparing against GPU backend when available and
testing edge cases specific to CPU tensor operations.
"""

import pytest
import torch
import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcts.cpu import cpu_game_states
from mcts.neural_networks.resnet_model import create_resnet_for_game
from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
from mcts.core.game_interface import GameInterface, GameType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralNetworkEvaluationValidator:
    """Comprehensive neural network evaluation validation framework"""
    
    def __init__(self, game_type: str = 'gomoku', board_size: int = 15):
        self.game_type = game_type
        self.board_size = board_size
        self.device_cpu = 'cpu'
        self.device_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create models for both devices
        self.model_cpu = self._create_model('cpu')
        self.model_gpu = self._create_model(self.device_gpu) if torch.cuda.is_available() else None
        
        # Create evaluators
        self.evaluator_cpu = ResNetEvaluator(
            model=self.model_cpu,
            game_type=self.game_type,
            device='cpu'
        )
        
        self.evaluator_gpu = None
        if self.model_gpu is not None:
            self.evaluator_gpu = ResNetEvaluator(
                model=self.model_gpu,
                game_type=self.game_type,
                device=self.device_gpu
            )
        
        # Create game interface
        self.game_interface = GameInterface(GameType.GOMOKU, board_size=self.board_size)
        
        # Create CPU game states for testing
        self.cpu_game_states = cpu_game_states.CPUGameStates(
            capacity=100,
            game_type=self.game_type,
            board_size=self.board_size
        )
    
    def _create_model(self, device: str):
        """Create and initialize model"""
        model = create_resnet_for_game(
            game_type=self.game_type,
            input_channels=19,
            num_blocks=2,
            num_filters=32
        )
        model.eval()
        
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    def create_test_states(self, count: int, with_moves: bool = True) -> Tuple[List[int], List]:
        """Create test states with known board configurations"""
        state_indices = self.cpu_game_states.allocate_states(count)
        state_list = state_indices.tolist() if hasattr(state_indices, 'tolist') else [state_indices.item()]
        
        game_states = []
        
        for i, state_idx in enumerate(state_list):
            # Create board with different patterns
            board = torch.zeros((15, 15), dtype=torch.int32)
            
            if with_moves:
                # Create different board patterns for testing
                if i == 0:
                    # Empty board
                    pass
                elif i == 1:
                    # Single move in center
                    board[7, 7] = 1
                elif i == 2:
                    # Two moves
                    board[7, 7] = 1
                    board[7, 8] = 2
                elif i == 3:
                    # Line of three
                    board[7, 6] = 1
                    board[7, 7] = 1
                    board[7, 8] = 1
                else:
                    # Random pattern
                    for j in range(min(i, 10)):
                        row = (7 + j) % 15
                        col = (7 + j * 2) % 15
                        board[row, col] = (j % 2) + 1
            
            # Set board in CPU game states
            self.cpu_game_states.set_board_from_tensor(state_idx, board)
            
            # Create corresponding game state for evaluator
            game_state = self.game_interface.create_initial_state()
            
            # Apply moves to match the board
            moves_applied = []
            for row in range(15):
                for col in range(15):
                    if board[row, col] != 0:
                        action = row * 15 + col
                        try:
                            if self.game_interface.is_legal_move(game_state, action):
                                game_state = self.game_interface.apply_move(game_state, action)
                                moves_applied.append(action)
                        except:
                            pass  # Skip if move not legal
            
            game_states.append(game_state)
        
        return state_list, game_states
    
    def test_single_state_evaluation(self) -> Dict:
        """Test neural network evaluation on single states"""
        results = {
            'cpu_evaluations': [],
            'gpu_evaluations': [],
            'cpu_errors': [],
            'gpu_errors': [],
            'evaluation_times': {'cpu': [], 'gpu': []},
            'consistency_scores': []
        }
        
        # Create test states
        state_indices, game_states = self.create_test_states(5)
        
        for i, (state_idx, game_state) in enumerate(zip(state_indices, game_states)):
            logger.info(f"Evaluating state {i} (index {state_idx})")
            
            # CPU Evaluation
            try:
                start_time = time.time()
                # Get basic tensor representation for evaluation (19 channels)
                state_tensor = self.game_interface.get_basic_tensor_representation(game_state)
                state_tensor = torch.from_numpy(state_tensor).float()
                cpu_policy, cpu_value = self.evaluator_cpu.evaluate(state_tensor)
                cpu_time = time.time() - start_time
                
                results['cpu_evaluations'].append({
                    'state_index': i,
                    'policy': cpu_policy.copy(),
                    'value': cpu_value,
                    'policy_entropy': -np.sum(cpu_policy * np.log(cpu_policy + 1e-10)),
                    'max_policy_prob': np.max(cpu_policy),
                    'nonzero_policy_count': np.sum(cpu_policy > 1e-8)
                })
                results['evaluation_times']['cpu'].append(cpu_time)
                
            except Exception as e:
                logger.error(f"CPU evaluation failed for state {i}: {e}")
                results['cpu_errors'].append({'state_index': i, 'error': str(e)})
            
            # GPU Evaluation (if available)
            if self.evaluator_gpu is not None:
                try:
                    start_time = time.time()
                    # Get basic tensor representation for evaluation (19 channels)
                    state_tensor = self.game_interface.get_basic_tensor_representation(game_state)
                    state_tensor = torch.from_numpy(state_tensor).float()
                    if self.device_gpu == 'cuda':
                        state_tensor = state_tensor.cuda()
                    gpu_policy, gpu_value = self.evaluator_gpu.evaluate(state_tensor)
                    gpu_time = time.time() - start_time
                    
                    results['gpu_evaluations'].append({
                        'state_index': i,
                        'policy': gpu_policy.copy(),
                        'value': gpu_value,
                        'policy_entropy': -np.sum(gpu_policy * np.log(gpu_policy + 1e-10)),
                        'max_policy_prob': np.max(gpu_policy),
                        'nonzero_policy_count': np.sum(gpu_policy > 1e-8)
                    })
                    results['evaluation_times']['gpu'].append(gpu_time)
                    
                    # Compare CPU vs GPU
                    if len(results['cpu_evaluations']) > i:
                        cpu_eval = results['cpu_evaluations'][i]
                        policy_diff = np.mean(np.abs(cpu_eval['policy'] - gpu_policy))
                        value_diff = abs(cpu_eval['value'] - gpu_value)
                        
                        consistency_score = {
                            'state_index': i,
                            'policy_l1_diff': policy_diff,
                            'value_diff': value_diff,
                            'policy_correlation': np.corrcoef(cpu_eval['policy'], gpu_policy)[0, 1]
                        }
                        results['consistency_scores'].append(consistency_score)
                    
                except Exception as e:
                    logger.error(f"GPU evaluation failed for state {i}: {e}")
                    results['gpu_errors'].append({'state_index': i, 'error': str(e)})
        
        return results
    
    def test_batch_evaluation(self, batch_size: int = 8) -> Dict:
        """Test batch neural network evaluation"""
        results = {
            'batch_cpu_success': False,
            'batch_gpu_success': False,
            'batch_sizes_tested': [],
            'batch_evaluation_times': {'cpu': [], 'gpu': []},
            'batch_consistency_scores': [],
            'batch_errors': {'cpu': [], 'gpu': []}
        }
        
        # Create test states
        state_indices, game_states = self.create_test_states(batch_size)
        
        # Test different batch sizes
        for test_batch_size in [1, 2, 4, min(batch_size, 8)]:
            logger.info(f"Testing batch size {test_batch_size}")
            
            test_states = game_states[:test_batch_size]
            results['batch_sizes_tested'].append(test_batch_size)
            
            # CPU Batch Evaluation
            try:
                start_time = time.time()
                # Get tensor representations for batch evaluation
                state_tensors = [torch.from_numpy(self.game_interface.get_basic_tensor_representation(state)).float() for state in test_states]
                cpu_policies, cpu_values = self.evaluator_cpu.evaluate_batch(state_tensors)
                cpu_time = time.time() - start_time
                
                results['batch_evaluation_times']['cpu'].append({
                    'batch_size': test_batch_size,
                    'time': cpu_time,
                    'time_per_state': cpu_time / test_batch_size
                })
                
                # Validate batch results
                assert len(cpu_policies) == test_batch_size
                assert len(cpu_values) == test_batch_size
                
                for policy in cpu_policies:
                    assert len(policy) == 225  # 15x15 board
                    assert abs(np.sum(policy) - 1.0) < 1e-5  # Probabilities sum to 1
                
                results['batch_cpu_success'] = True
                
            except Exception as e:
                logger.error(f"CPU batch evaluation failed for batch size {test_batch_size}: {e}")
                results['batch_errors']['cpu'].append({
                    'batch_size': test_batch_size,
                    'error': str(e)
                })
            
            # GPU Batch Evaluation (if available)
            if self.evaluator_gpu is not None:
                try:
                    start_time = time.time()
                    # Get tensor representations for batch evaluation
                    state_tensors = [torch.from_numpy(self.game_interface.get_basic_tensor_representation(state)).float() for state in test_states]
                    if self.device_gpu == 'cuda':
                        state_tensors = [t.cuda() for t in state_tensors]
                    gpu_policies, gpu_values = self.evaluator_gpu.evaluate_batch(state_tensors)
                    gpu_time = time.time() - start_time
                    
                    results['batch_evaluation_times']['gpu'].append({
                        'batch_size': test_batch_size,
                        'time': gpu_time,
                        'time_per_state': gpu_time / test_batch_size
                    })
                    
                    # Compare CPU vs GPU batch results
                    if results['batch_cpu_success']:
                        state_tensors_cpu = [torch.from_numpy(self.game_interface.get_basic_tensor_representation(state)).float() for state in test_states]
                        cpu_policies_batch, cpu_values_batch = self.evaluator_cpu.evaluate_batch(state_tensors_cpu)
                        
                        batch_policy_diffs = []
                        batch_value_diffs = []
                        
                        for cpu_pol, gpu_pol, cpu_val, gpu_val in zip(
                            cpu_policies_batch, gpu_policies, cpu_values_batch, gpu_values
                        ):
                            batch_policy_diffs.append(np.mean(np.abs(cpu_pol - gpu_pol)))
                            batch_value_diffs.append(abs(cpu_val - gpu_val))
                        
                        results['batch_consistency_scores'].append({
                            'batch_size': test_batch_size,
                            'avg_policy_diff': np.mean(batch_policy_diffs),
                            'avg_value_diff': np.mean(batch_value_diffs),
                            'max_policy_diff': np.max(batch_policy_diffs),
                            'max_value_diff': np.max(batch_value_diffs)
                        })
                    
                    results['batch_gpu_success'] = True
                    
                except Exception as e:
                    logger.error(f"GPU batch evaluation failed for batch size {test_batch_size}: {e}")
                    results['batch_errors']['gpu'].append({
                        'batch_size': test_batch_size,
                        'error': str(e)
                    })
        
        return results
    
    def test_cpu_tensor_operations(self) -> Dict:
        """Test CPU-specific tensor operations in neural network evaluation"""
        results = {
            'tensor_conversion_tests': [],
            'memory_usage_tests': [],
            'cpu_specific_errors': []
        }
        
        # Create test states
        state_indices, game_states = self.create_test_states(5)
        
        for i, (state_idx, game_state) in enumerate(zip(state_indices, game_states)):
            try:
                # Test tensor representation extraction
                start_time = time.time()
                
                # Get basic tensor representation from game state
                tensor_repr = self.game_interface.get_basic_tensor_representation(game_state)
                tensor_repr = torch.from_numpy(tensor_repr).float()
                
                # Ensure it's on CPU
                if hasattr(tensor_repr, 'device'):
                    assert tensor_repr.device.type == 'cpu'
                
                # Test conversion to different dtypes
                float_tensor = tensor_repr.float()
                double_tensor = tensor_repr.double()
                
                conversion_time = time.time() - start_time
                
                results['tensor_conversion_tests'].append({
                    'state_index': i,
                    'conversion_time': conversion_time,
                    'tensor_shape': list(tensor_repr.shape),
                    'tensor_dtype': str(tensor_repr.dtype),
                    'memory_usage_mb': tensor_repr.element_size() * tensor_repr.nelement() / 1024 / 1024
                })
                
                # Test memory usage during evaluation
                start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                state_tensor = torch.from_numpy(self.game_interface.get_basic_tensor_representation(game_state)).float()
                policy, value = self.evaluator_cpu.evaluate(state_tensor)
                
                end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                results['memory_usage_tests'].append({
                    'state_index': i,
                    'memory_increase_bytes': end_memory - start_memory,
                    'policy_memory_mb': policy.nbytes / 1024 / 1024,
                    'successful_evaluation': True
                })
                
            except Exception as e:
                logger.error(f"CPU tensor operation failed for state {i}: {e}")
                results['cpu_specific_errors'].append({
                    'state_index': i,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        
        return results
    
    def test_edge_cases(self) -> Dict:
        """Test edge cases specific to CPU neural network evaluation"""
        results = {
            'empty_board_evaluation': None,
            'full_board_evaluation': None,
            'near_terminal_evaluation': None,
            'invalid_input_handling': [],
            'edge_case_errors': []
        }
        
        try:
            # Test empty board
            empty_state = self.game_interface.create_initial_state()
            state_tensor = torch.from_numpy(self.game_interface.get_basic_tensor_representation(empty_state)).float()
            policy, value = self.evaluator_cpu.evaluate(state_tensor)
            results['empty_board_evaluation'] = {
                'policy_entropy': -np.sum(policy * np.log(policy + 1e-10)),
                'value': value,
                'policy_max': np.max(policy),
                'policy_nonzero_count': np.sum(policy > 1e-8)
            }
            
        except Exception as e:
            results['edge_case_errors'].append({
                'test': 'empty_board',
                'error': str(e)
            })
        
        try:
            # Test near-terminal state (create a state close to winning)
            near_terminal_state = self.game_interface.create_initial_state()
            
            # Create a line of 4 stones (one move away from winning)
            moves = [112, 127, 113, 128, 114, 129, 115]  # 4 in a row for player 1, some blocks by player 2
            for move in moves:
                if self.game_interface.is_legal_move(near_terminal_state, move):
                    near_terminal_state = self.game_interface.apply_move(near_terminal_state, move)
            
            state_tensor = torch.from_numpy(self.game_interface.get_basic_tensor_representation(near_terminal_state)).float()
            policy, value = self.evaluator_cpu.evaluate(state_tensor)
            results['near_terminal_evaluation'] = {
                'policy_entropy': -np.sum(policy * np.log(policy + 1e-10)),
                'value': value,
                'policy_max': np.max(policy),
                'winning_move_prob': policy[116] if 116 < len(policy) else 0  # Move to complete the line
            }
            
        except Exception as e:
            results['edge_case_errors'].append({
                'test': 'near_terminal',
                'error': str(e)
            })
        
        return results


class TestNeuralNetworkEvaluationValidation:
    """Test class for neural network evaluation validation"""
    
    @pytest.fixture
    def validator(self):
        return NeuralNetworkEvaluationValidator()
    
    def test_single_state_evaluation_cpu(self, validator):
        """Test single state evaluation on CPU"""
        results = validator.test_single_state_evaluation()
        
        # Ensure CPU evaluations worked
        assert len(results['cpu_evaluations']) > 0, "No CPU evaluations completed"
        assert len(results['cpu_errors']) == 0, f"CPU evaluation errors: {results['cpu_errors']}"
        
        # Validate each CPU evaluation
        for eval_result in results['cpu_evaluations']:
            policy = eval_result['policy']
            value = eval_result['value']
            
            # Basic validation
            assert len(policy) == 225, f"Policy should have 225 elements, got {len(policy)}"
            assert abs(np.sum(policy) - 1.0) < 1e-5, f"Policy should sum to 1, got {np.sum(policy)}"
            assert -1.0 <= value <= 1.0, f"Value should be in [-1, 1], got {value}"
            assert eval_result['nonzero_policy_count'] > 0, "Policy should have some non-zero probabilities"
            
        # Check evaluation times are reasonable
        avg_cpu_time = np.mean(results['evaluation_times']['cpu'])
        assert avg_cpu_time < 5.0, f"CPU evaluation too slow: {avg_cpu_time}s per state"
        
        logger.info(f"✓ CPU evaluations successful. Avg time: {avg_cpu_time:.3f}s")
        
        # Compare with GPU if available
        if len(results['gpu_evaluations']) > 0:
            for consistency in results['consistency_scores']:
                # Policy differences should be small (allowing for numerical precision)
                assert consistency['policy_l1_diff'] < 0.1, f"Large policy difference: {consistency['policy_l1_diff']}"
                assert consistency['value_diff'] < 0.2, f"Large value difference: {consistency['value_diff']}"
                
                # Correlation should be high
                if not np.isnan(consistency['policy_correlation']):
                    assert consistency['policy_correlation'] > 0.7, f"Low policy correlation: {consistency['policy_correlation']}"
            
            logger.info("✓ CPU vs GPU consistency validated")
    
    def test_batch_evaluation_cpu(self, validator):
        """Test batch evaluation on CPU"""
        results = validator.test_batch_evaluation(8)
        
        # Ensure CPU batch evaluation worked
        assert results['batch_cpu_success'], f"CPU batch evaluation failed: {results['batch_errors']['cpu']}"
        
        # Check batch evaluation times
        cpu_times = results['batch_evaluation_times']['cpu']
        assert len(cpu_times) > 0, "No CPU batch timing data"
        
        # Batch evaluation should be more efficient than individual evaluations
        for timing in cpu_times:
            assert timing['time_per_state'] < 5.0, f"Batch evaluation too slow: {timing['time_per_state']}s per state"
        
        # Compare batch sizes - larger batches should be more efficient per state
        if len(cpu_times) > 1:
            small_batch_time = next(t['time_per_state'] for t in cpu_times if t['batch_size'] == 1)
            large_batch_time = next((t['time_per_state'] for t in cpu_times if t['batch_size'] > 1), None)
            
            if large_batch_time is not None:
                efficiency_ratio = small_batch_time / large_batch_time
                logger.info(f"Batch efficiency ratio: {efficiency_ratio:.2f}")
        
        logger.info("✓ CPU batch evaluation successful")
        
        # Compare with GPU batch if available
        if results['batch_gpu_success'] and results['batch_consistency_scores']:
            for consistency in results['batch_consistency_scores']:
                assert consistency['avg_policy_diff'] < 0.1, f"Large batch policy difference: {consistency['avg_policy_diff']}"
                assert consistency['avg_value_diff'] < 0.2, f"Large batch value difference: {consistency['avg_value_diff']}"
            
            logger.info("✓ CPU vs GPU batch consistency validated")
    
    def test_cpu_tensor_operations(self, validator):
        """Test CPU-specific tensor operations"""
        results = validator.test_cpu_tensor_operations()
        
        # Ensure tensor operations worked
        assert len(results['tensor_conversion_tests']) > 0, "No tensor conversion tests completed"
        assert len(results['cpu_specific_errors']) == 0, f"CPU tensor errors: {results['cpu_specific_errors']}"
        
        # Validate tensor operations
        for test_result in results['tensor_conversion_tests']:
            assert test_result['conversion_time'] < 1.0, f"Tensor conversion too slow: {test_result['conversion_time']}s"
            assert test_result['tensor_shape'] == [19, 15, 15], f"Unexpected tensor shape: {test_result['tensor_shape']}"
            assert test_result['memory_usage_mb'] < 50, f"High memory usage: {test_result['memory_usage_mb']}MB"
        
        # Check memory usage during evaluation
        for memory_test in results['memory_usage_tests']:
            assert memory_test['successful_evaluation'], "Evaluation should succeed"
            assert memory_test['policy_memory_mb'] < 1, f"Policy uses too much memory: {memory_test['policy_memory_mb']}MB"
        
        logger.info("✓ CPU tensor operations validated")
    
    def test_edge_cases_cpu(self, validator):
        """Test edge cases for CPU evaluation"""
        results = validator.test_edge_cases()
        
        # Check that edge case evaluations worked
        assert len(results['edge_case_errors']) == 0, f"Edge case errors: {results['edge_case_errors']}"
        
        # Validate empty board evaluation
        if results['empty_board_evaluation']:
            empty_eval = results['empty_board_evaluation']
            assert empty_eval['policy_nonzero_count'] > 0, "Empty board should have valid moves"
            assert empty_eval['policy_entropy'] > 0, "Empty board policy should have some entropy"
            assert -1.0 <= empty_eval['value'] <= 1.0, f"Empty board value out of range: {empty_eval['value']}"
        
        # Validate near-terminal evaluation
        if results['near_terminal_evaluation']:
            terminal_eval = results['near_terminal_evaluation']
            assert -1.0 <= terminal_eval['value'] <= 1.0, f"Terminal value out of range: {terminal_eval['value']}"
            # Terminal positions should have more focused policies (lower entropy)
            if results['empty_board_evaluation']:
                assert terminal_eval['policy_entropy'] <= results['empty_board_evaluation']['policy_entropy'], \
                    "Terminal state should have more focused policy"
        
        logger.info("✓ CPU edge cases validated")


def test_neural_network_evaluation_integration():
    """Integration test for neural network evaluation validation"""
    validator = NeuralNetworkEvaluationValidator()
    
    # Run a quick integration test
    results = validator.test_single_state_evaluation()
    
    # Should have at least CPU results
    assert len(results['cpu_evaluations']) > 0
    assert len(results['cpu_errors']) == 0
    
    # Basic validation
    eval_result = results['cpu_evaluations'][0]
    assert len(eval_result['policy']) == 225
    assert abs(np.sum(eval_result['policy']) - 1.0) < 1e-5
    
    logger.info("✓ Neural network evaluation integration test passed")


if __name__ == "__main__":
    # Run tests manually if not using pytest
    pytest.main([__file__, "-v", "--tb=short"])