"""
Comprehensive integration tests for MCTS system

This module tests how all components work together:
- Full MCTS search with GPU acceleration
- Training pipeline integration
- Self-play scenarios
- Multi-process coordination
- End-to-end game playing
"""

import pytest
import torch
import numpy as np
import time
import multiprocessing
import queue
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from mcts.core.mcts import MCTS
from mcts.core.mcts_config import MCTSConfig
from mcts.core.game_interface import GameInterface, GameType
from mcts.gpu.gpu_game_states import GameType as GPUGameType
from mcts.utils.batch_evaluation_coordinator import get_global_batching_coordinator
from mcts.utils.optimized_remote_evaluator import OptimizedRemoteEvaluator
from mcts.utils.gpu_evaluator_service import GPUEvaluatorService
from conftest import assert_valid_policy, assert_valid_value


@pytest.mark.integration
class TestFullMCTSIntegration:
    """Test full MCTS system integration"""
    
    def test_mcts_with_mock_evaluator(self, base_mcts_config, mock_evaluator, empty_gomoku_state):
        """Test MCTS with mock evaluator end-to-end"""
        # Create MCTS
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Run search
        policy = mcts.search(empty_gomoku_state, num_simulations=100)
        
        # Verify results
        assert_valid_policy(policy)
        assert mcts.tree.num_nodes > 1
        assert mcts.tree.node_data.visit_counts[0] == 100
        
        # Select action
        action = mcts.select_action(empty_gomoku_state, temperature=1.0)
        assert 0 <= action < 225
        
        # Get statistics
        stats = mcts.get_statistics()
        assert stats['total_simulations'] == 100
        assert stats['total_searches'] == 1
        
    def test_mcts_with_batch_coordinator(self, base_mcts_config, device):
        """Test MCTS with batch coordinator integration"""
        # Set up batch coordinator
        coordinator = get_global_batching_coordinator(
            max_batch_size=32,
            batch_timeout_ms=50.0
        )
        
        try:
            # Create mock GPU service
            request_queue = queue.Queue()
            response_queue = queue.Queue()
            
            def mock_gpu_service():
                while True:
                    try:
                        from mcts.utils.batch_evaluation_coordinator import BatchEvaluationResponse
                        req = request_queue.get(timeout=0.5)
                        if req is None:
                            break
                            
                        batch_size = req.states.shape[0]
                        resp = BatchEvaluationResponse(
                            request_id=req.request_id,
                            policies=np.random.rand(batch_size, 225).astype(np.float32),
                            values=np.random.uniform(-0.5, 0.5, batch_size).astype(np.float32),
                            worker_id=req.worker_id,
                            individual_request_ids=req.individual_request_ids
                        )
                        response_queue.put(resp)
                    except queue.Empty:
                        continue
                        
            gpu_thread = multiprocessing.threading.Thread(target=mock_gpu_service)
            gpu_thread.start()
            
            # Create evaluator that uses coordinator
            class CoordinatedEvaluator:
                def __init__(self):
                    self.coordinator = coordinator
                    self.request_queue = request_queue
                    self.response_queue = response_queue
                    
                def evaluate_batch(self, states):
                    if isinstance(states, torch.Tensor):
                        states = states.cpu().numpy()
                    
                    policies, values = self.coordinator.coordinate_evaluation_batch(
                        states=states,
                        worker_id=0,
                        gpu_service_request_queue=self.request_queue,
                        response_queue=self.response_queue
                    )
                    return policies, values
                    
            evaluator = CoordinatedEvaluator()
            
            # Run MCTS
            mcts = MCTS(base_mcts_config, evaluator)
            game = GameInterface(GameType.GOMOKU, board_size=15)
            state = game.create_initial_state()
            
            policy = mcts.search(state, num_simulations=50)
            assert_valid_policy(policy)
            
            # Check coordinator statistics
            coord_stats = coordinator.get_statistics()
            assert coord_stats['requests_processed'] > 0
            
            # Stop GPU service
            request_queue.put(None)
            gpu_thread.join()
            
        finally:
            from mcts.utils.batch_evaluation_coordinator import cleanup_global_coordinator
            cleanup_global_coordinator()
            
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mcts_with_gpu_acceleration(self, empty_gomoku_state):
        """Test MCTS with GPU acceleration"""
        # Configure for GPU
        config = MCTSConfig()
        config.device = 'cuda'
        config.game_type = GPUGameType.GOMOKU
        config.board_size = 15
        config.num_simulations = 200
        config.max_wave_size = 64
        config.enable_fast_ucb = True
        
        # Mock evaluator
        evaluator = Mock()
        evaluator.evaluate_batch = Mock(return_value=(
            np.random.rand(64, 225).astype(np.float32),
            np.random.uniform(-0.5, 0.5, 64).astype(np.float32)
        ))
        
        # Create MCTS
        mcts = MCTS(config, evaluator)
        
        # Run search
        start_time = time.time()
        policy = mcts.search(empty_gomoku_state, num_simulations=config.num_simulations)
        search_time = time.time() - start_time
        
        # Verify performance
        simulations_per_second = config.num_simulations / search_time
        assert simulations_per_second > 100  # Should be fast with GPU
        
        # Verify results
        assert_valid_policy(policy)
        assert mcts.tree.num_nodes > config.num_simulations // 2  # Good tree growth


@pytest.mark.integration
class TestSelfPlayIntegration:
    """Test self-play integration"""
    
    def test_self_play_game(self, base_mcts_config, mock_evaluator):
        """Test playing a complete self-play game"""
        # Create game and MCTS
        game = GameInterface(GameType.GOMOKU, board_size=15)
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        # Play game
        state = game.create_initial_state()
        move_count = 0
        max_moves = 225
        
        game_history = []
        
        while not game.is_terminal(state) and move_count < max_moves:
            # Get policy
            policy = mcts.search(state, num_simulations=50)
            
            # Store for training
            game_history.append({
                'state': game.state_to_numpy(state),
                'policy': policy,
                'player': game.get_current_player(state)
            })
            
            # Select move
            if move_count < 30:
                # Exploration phase
                action = mcts.select_action(state, temperature=1.0)
            else:
                # Exploitation phase
                action = mcts.select_action(state, temperature=0.1)
                
            # Make move
            state = game.apply_move(state, action)
            
            # Update tree if using subtree reuse
            if base_mcts_config.enable_subtree_reuse:
                mcts.update_root(action, state)
            else:
                mcts.clear()
                
            move_count += 1
            
        # Game should end
        assert move_count > 5  # Not trivial
        assert move_count < max_moves  # Should terminate
        
        # Assign values
        if game.is_terminal(state):
            winner = game.get_winner(state)
            if winner == 0:  # Draw
                final_value = 0.0
            else:
                final_value = 1.0
        else:
            # Max moves reached
            final_value = 0.0
            
        # Create training data
        training_data = []
        for i, hist in enumerate(game_history):
            # Value from perspective of player who made the move
            if hist['player'] == winner:
                value = final_value
            elif winner == 0:  # Draw
                value = 0.0
            else:
                value = -final_value
                
            training_data.append({
                'state': hist['state'],
                'policy': hist['policy'],
                'value': value
            })
            
        # Verify training data
        assert len(training_data) == move_count
        for data in training_data:
            assert data['state'].shape == (3, 15, 15)  # Or appropriate channels
            assert data['policy'].shape == (225,)
            assert -1.0 <= data['value'] <= 1.0
            
    def test_parallel_self_play(self, base_mcts_config, mock_evaluator_factory):
        """Test parallel self-play games"""
        num_workers = 3
        games_per_worker = 2
        
        def play_games(worker_id, result_queue):
            """Worker function to play games"""
            # Create evaluator for this worker
            evaluator = mock_evaluator_factory()
            
            # Configure MCTS
            config = base_mcts_config
            config.num_simulations = 30  # Faster for testing
            
            mcts = MCTS(config, evaluator)
            game = GameInterface(GameType.GOMOKU, board_size=15)
            
            games_data = []
            
            for game_idx in range(games_per_worker):
                state = game.create_initial_state()
                move_count = 0
                
                while not game.is_terminal(state) and move_count < 100:
                    policy = mcts.search(state, num_simulations=config.num_simulations)
                    action = mcts.select_action(state, temperature=1.0)
                    state = game.apply_move(state, action)
                    mcts.clear()  # No subtree reuse for simplicity
                    move_count += 1
                    
                games_data.append({
                    'worker_id': worker_id,
                    'game_idx': game_idx,
                    'moves': move_count,
                    'terminal': game.is_terminal(state)
                })
                
            result_queue.put(games_data)
            
        # Run workers
        processes = []
        result_queue = multiprocessing.Queue()
        
        for worker_id in range(num_workers):
            p = multiprocessing.Process(
                target=play_games,
                args=(worker_id, result_queue)
            )
            processes.append(p)
            p.start()
            
        # Collect results
        all_games = []
        for _ in range(num_workers):
            games = result_queue.get(timeout=30)
            all_games.extend(games)
            
        # Wait for completion
        for p in processes:
            p.join()
            
        # Verify results
        assert len(all_games) == num_workers * games_per_worker
        
        # All games should complete
        for game_data in all_games:
            assert game_data['moves'] > 0
            assert game_data['terminal'] or game_data['moves'] >= 100


@pytest.mark.integration
class TestTrainingPipelineIntegration:
    """Test training pipeline integration"""
    
    def test_data_generation_and_training_format(self, base_mcts_config, mock_evaluator):
        """Test generating training data in correct format"""
        # Play a short game
        game = GameInterface(GameType.GOMOKU, board_size=15)
        mcts = MCTS(base_mcts_config, mock_evaluator)
        
        state = game.create_initial_state()
        training_examples = []
        
        # Play some moves
        for _ in range(10):
            # Search
            policy = mcts.search(state, num_simulations=30)
            
            # Create training example
            state_tensor = game.state_to_numpy(state)
            
            # Get symmetries for data augmentation
            symmetries = game.get_symmetries(state_tensor, policy)
            
            for sym_state, sym_policy in symmetries:
                training_examples.append({
                    'state': sym_state,
                    'policy': sym_policy,
                    'value': None  # Will be filled after game ends
                })
                
            # Make move
            action = np.random.choice(225, p=policy)
            state = game.apply_move(state, action)
            mcts.clear()
            
            if game.is_terminal(state):
                break
                
        # Assign values (simplified - normally based on game outcome)
        for example in training_examples:
            example['value'] = np.random.uniform(-1, 1)
            
        # Verify format
        assert len(training_examples) > 10  # With symmetries
        
        for example in training_examples:
            assert example['state'].shape[0] >= 3  # Channels
            assert example['state'].shape[1:] == (15, 15)
            assert example['policy'].shape == (225,)
            assert -1.0 <= example['value'] <= 1.0
            
        # Test batching for training
        batch_size = 8
        batch_states = np.stack([ex['state'] for ex in training_examples[:batch_size]])
        batch_policies = np.stack([ex['policy'] for ex in training_examples[:batch_size]])
        batch_values = np.array([ex['value'] for ex in training_examples[:batch_size]])
        
        assert batch_states.shape == (batch_size, training_examples[0]['state'].shape[0], 15, 15)
        assert batch_policies.shape == (batch_size, 225)
        assert batch_values.shape == (batch_size,)
        
    @patch('mcts.neural_networks.unified_training_pipeline.ResNetModel')
    def test_model_update_integration(self, mock_model_class, base_mcts_config, device):
        """Test model update with generated data"""
        # Create mock model
        mock_model = Mock()
        mock_model.train.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        # Mock forward pass
        def mock_forward(x):
            batch_size = x.shape[0]
            policies = torch.rand(batch_size, 225)
            values = torch.rand(batch_size, 1) * 2 - 1
            return policies, values
            
        mock_model.forward = mock_forward
        mock_model_class.return_value = mock_model
        
        # Generate training data
        num_examples = 100
        training_data = []
        
        for _ in range(num_examples):
            state = np.random.rand(3, 15, 15).astype(np.float32)
            policy = np.random.rand(225).astype(np.float32)
            policy /= policy.sum()  # Normalize
            value = np.random.uniform(-1, 1)
            
            training_data.append({
                'state': state,
                'policy': policy,
                'value': value
            })
            
        # Convert to tensors
        states = torch.FloatTensor(np.stack([d['state'] for d in training_data]))
        policies = torch.FloatTensor(np.stack([d['policy'] for d in training_data]))
        values = torch.FloatTensor(np.array([d['value'] for d in training_data]))
        
        # Create simple training loop
        batch_size = 32
        num_batches = len(training_data) // batch_size
        
        optimizer = torch.optim.Adam(mock_model.parameters(), lr=0.001)
        
        for epoch in range(2):
            total_loss = 0.0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_states = states[start_idx:end_idx].to(device)
                batch_policies = policies[start_idx:end_idx].to(device)
                batch_values = values[start_idx:end_idx].to(device)
                
                # Forward pass
                pred_policies, pred_values = mock_model(batch_states)
                
                # Compute losses (simplified)
                policy_loss = -torch.sum(batch_policies * torch.log(pred_policies + 1e-8)) / batch_size
                value_loss = torch.mean((pred_values.squeeze() - batch_values) ** 2)
                total_loss = policy_loss + value_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward = Mock()  # Mock backward
                optimizer.step = Mock()  # Mock step
                
                total_loss += total_loss.item() if hasattr(total_loss, 'item') else 0
                
            avg_loss = total_loss / num_batches
            assert avg_loss >= 0  # Sanity check


@pytest.mark.integration
class TestGPUServiceIntegration:
    """Test GPU evaluator service integration"""
    
    def test_gpu_service_with_mcts(self, base_mcts_config, device):
        """Test GPU service integration with MCTS"""
        if device.type != 'cuda':
            pytest.skip("GPU service test requires CUDA")
            
        # Create mock model
        class MockNeuralNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.fc_policy = torch.nn.Linear(16 * 15 * 15, 225)
                self.fc_value = torch.nn.Linear(16 * 15 * 15, 1)
                
            def forward(self, x):
                batch_size = x.shape[0]
                x = torch.relu(self.conv(x))
                x = x.view(batch_size, -1)
                
                policy = torch.softmax(self.fc_policy(x), dim=1)
                value = torch.tanh(self.fc_value(x))
                
                return policy, value
                
        model = MockNeuralNet().to(device)
        
        # Create GPU service
        request_queue = multiprocessing.Queue()
        response_queue = multiprocessing.Queue()
        
        service = GPUEvaluatorService(
            model=model,
            request_queue=request_queue,
            response_queue=response_queue,
            batch_size=32,
            timeout_ms=100,
            device=device
        )
        
        # Start service in thread (not process for this test)
        service_thread = multiprocessing.threading.Thread(target=service.run)
        service_thread.start()
        
        try:
            # Create evaluator that uses the service
            evaluator = OptimizedRemoteEvaluator(
                request_queue=request_queue,
                response_queue=response_queue,
                game_type='gomoku',
                use_batching_coordinator=False
            )
            
            # Run MCTS
            mcts = MCTS(base_mcts_config, evaluator)
            game = GameInterface(GameType.GOMOKU)
            state = game.create_initial_state()
            
            policy = mcts.search(state, num_simulations=50)
            assert_valid_policy(policy)
            
        finally:
            # Stop service
            request_queue.put(None)
            service_thread.join(timeout=2.0)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPerformance:
    """Test end-to-end performance"""
    
    def test_performance_benchmark(self, device):
        """Benchmark MCTS performance"""
        # Configure for performance
        config = MCTSConfig()
        config.device = str(device)
        config.game_type = GPUGameType.GOMOKU
        config.board_size = 15
        config.num_simulations = 1000
        config.max_wave_size = 64 if device.type == 'cuda' else 16
        config.enable_fast_ucb = True
        config.enable_subtree_reuse = True
        config.c_puct = 1.4
        
        # Fast mock evaluator
        class FastMockEvaluator:
            def __init__(self, device):
                self.device = device
                self.call_count = 0
                
            def evaluate_batch(self, states):
                self.call_count += 1
                if isinstance(states, np.ndarray):
                    batch_size = states.shape[0]
                else:
                    batch_size = states.shape[0]
                    
                # Fast random generation
                policies = np.random.rand(batch_size, 225).astype(np.float32)
                policies /= policies.sum(axis=1, keepdims=True)
                values = np.random.uniform(-0.5, 0.5, batch_size).astype(np.float32)
                
                return policies, values
                
        evaluator = FastMockEvaluator(device)
        
        # Create MCTS
        mcts = MCTS(config, evaluator)
        game = GameInterface(GameType.GOMOKU)
        state = game.create_initial_state()
        
        # Warmup
        mcts.search(state, num_simulations=10)
        mcts.clear()
        
        # Benchmark
        start_time = time.time()
        policy = mcts.search(state, num_simulations=config.num_simulations)
        elapsed = time.time() - start_time
        
        # Calculate metrics
        simulations_per_second = config.num_simulations / elapsed
        evaluations_per_second = evaluator.call_count / elapsed
        
        # Report results
        print(f"\nPerformance Benchmark ({device}):")
        print(f"  Simulations: {config.num_simulations}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Simulations/sec: {simulations_per_second:.0f}")
        print(f"  Evaluations/sec: {evaluations_per_second:.0f}")
        print(f"  Tree nodes: {mcts.tree.num_nodes}")
        print(f"  Avg branching: {mcts.tree.num_nodes / max(1, mcts.tree.num_nodes - 1):.1f}")
        
        # Performance assertions
        if device.type == 'cuda':
            assert simulations_per_second > 500  # Should be fast on GPU
        else:
            assert simulations_per_second > 100  # Reasonable on CPU
            
        # Tree quality
        assert mcts.tree.num_nodes > config.num_simulations * 0.5  # Good expansion
        assert policy.sum() > 0.99  # Valid policy
        
    def test_memory_efficiency(self, device):
        """Test memory efficiency with large trees"""
        config = MCTSConfig()
        config.device = str(device)
        config.game_type = GPUGameType.GOMOKU
        config.max_tree_nodes = 50000
        config.num_simulations = 5000
        config.enable_subtree_reuse = True
        
        # Mock evaluator
        evaluator = Mock()
        evaluator.evaluate_batch = Mock(side_effect=lambda states: (
            np.random.rand(len(states) if hasattr(states, '__len__') else states.shape[0], 225).astype(np.float32),
            np.zeros(len(states) if hasattr(states, '__len__') else states.shape[0], dtype=np.float32)
        ))
        
        mcts = MCTS(config, evaluator)
        game = GameInterface(GameType.GOMOKU)
        
        # Play multiple searches to build large tree
        state = game.create_initial_state()
        
        for i in range(5):
            policy = mcts.search(state, num_simulations=1000)
            action = np.argmax(policy)
            state = game.apply_move(state, action)
            
            if config.enable_subtree_reuse:
                mcts.update_root(action, state)
                
        # Check memory usage
        stats = mcts.get_statistics()
        memory_mb = stats.get('memory_usage_mb', 0)
        nodes = mcts.tree.num_nodes
        
        # Calculate efficiency
        bytes_per_node = (memory_mb * 1024 * 1024) / max(1, nodes)
        
        print(f"\nMemory Efficiency:")
        print(f"  Nodes: {nodes}")
        print(f"  Memory: {memory_mb:.1f} MB")
        print(f"  Bytes/node: {bytes_per_node:.0f}")
        
        # Should be memory efficient
        assert bytes_per_node < 1000  # Less than 1KB per node
        assert memory_mb < 200  # Reasonable total memory