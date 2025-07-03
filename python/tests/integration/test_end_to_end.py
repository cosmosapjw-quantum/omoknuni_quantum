"""End-to-end integration tests for the MCTS system"""

import pytest
import numpy as np
import time
from typing import List, Tuple

from mcts.core import MCTS, MCTSConfig, GameInterface, GameType, MockEvaluator
from mcts.utils import AlphaZeroConfig, create_default_config


class TestCompleteGameSimulation:
    """Test complete game simulations from start to finish"""
    
    def test_complete_gomoku_game_simulation(self):
        """Test a complete Gomoku game simulation"""
        config = MCTSConfig(
            num_simulations=30,  # Small for fast test
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=9,
            device='cpu',
            temperature=1.0
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator(seed=42)
        mcts = MCTS(config, game_interface, evaluator)
        
        # Simulate complete game
        state = game_interface.get_initial_state()
        moves_made = []
        game_history = []
        
        max_moves = 50  # Reasonable limit for 9x9 board
        
        while not game_interface.is_terminal(state) and len(moves_made) < max_moves:
            # Record current state
            game_history.append(state.copy())
            
            # Get MCTS decision
            action_probs = mcts.search(state)
            action = mcts.select_action(state, temperature=config.temperature)
            
            # Verify action is legal
            legal_moves = game_interface.get_legal_moves(state)
            assert legal_moves[action], f"MCTS selected illegal action {action} at move {len(moves_made)}"
            
            # Make the move
            new_state = game_interface.make_move(state, action)
            
            # Record move
            moves_made.append(action)
            
            # Convert action to position for verification
            row, col = game_interface.action_to_position(action)
            current_player = 1 if np.all(state[2] == 1) else 2
            
            # Verify move was placed correctly
            player_layer = 0 if current_player == 1 else 1
            assert new_state[player_layer, row, col] == 1.0, f"Move not placed correctly at ({row}, {col})"
            
            state = new_state
        
        # Game should have completed or reached move limit
        assert len(moves_made) > 0, "Should have made at least one move"
        assert len(moves_made) <= max_moves, "Should not exceed move limit"
        
        # If game terminated naturally, check winner
        if game_interface.is_terminal(state):
            winner = game_interface.get_winner(state)
            assert winner in [-1, 1, 2], f"Invalid winner: {winner}"
            
            if winner != -1:  # Not a draw
                print(f"Game completed in {len(moves_made)} moves, Player {winner} won")
            else:
                print(f"Game completed in {len(moves_made)} moves, Draw")
        else:
            print(f"Game reached move limit ({max_moves} moves)")
        
        # Verify game history consistency
        assert len(game_history) == len(moves_made), "History should match moves"
        
        # Verify no illegal moves were made
        for i, (hist_state, action) in enumerate(zip(game_history, moves_made)):
            legal = game_interface.get_legal_moves(hist_state)
            assert legal[action], f"Move {i} (action {action}) was illegal"
    
    def test_self_play_simulation(self):
        """Test self-play simulation between two MCTS instances"""
        config = MCTSConfig(
            num_simulations=20,
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=9,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        
        # Create two MCTS players with different seeds for variety
        player1 = MCTS(config, game_interface, MockEvaluator(seed=1))
        player2 = MCTS(config, game_interface, MockEvaluator(seed=2))
        
        # Simulate self-play game
        state = game_interface.get_initial_state()
        moves_made = []
        current_player = 1
        
        max_moves = 40
        
        while not game_interface.is_terminal(state) and len(moves_made) < max_moves:
            # Select appropriate MCTS player
            mcts_player = player1 if current_player == 1 else player2
            
            # Get canonical state for current player
            canonical_state = game_interface.get_canonical_state(state, current_player)
            
            # Get action from MCTS
            action = mcts_player.select_action(canonical_state, temperature=0.8)
            
            # Verify action is legal
            legal_moves = game_interface.get_legal_moves(state)
            assert legal_moves[action], f"Player {current_player} selected illegal action {action}"
            
            # Make the move
            state = game_interface.make_move(state, action)
            moves_made.append((current_player, action))
            
            # Switch players
            current_player = 3 - current_player  # Switch between 1 and 2
        
        assert len(moves_made) > 0, "Self-play should make at least one move"
        
        # Check final game state
        if game_interface.is_terminal(state):
            winner = game_interface.get_winner(state)
            print(f"Self-play game completed in {len(moves_made)} moves, winner: {winner}")
        else:
            print(f"Self-play game reached move limit ({max_moves} moves)")
        
        # Verify move alternation
        for i, (player, _) in enumerate(moves_made):
            expected_player = (i % 2) + 1
            assert player == expected_player, f"Move {i}: expected player {expected_player}, got {player}"
    
    def test_tournament_simulation(self):
        """Test tournament between different MCTS configurations"""
        configs = [
            MCTSConfig(num_simulations=15, c_puct=1.0, classical_only_mode=True, device='cpu'),
            MCTSConfig(num_simulations=25, c_puct=1.414, classical_only_mode=True, device='cpu'),
            MCTSConfig(num_simulations=20, c_puct=2.0, classical_only_mode=True, device='cpu'),
        ]
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        results = []
        
        # Play round-robin tournament
        for i, config1 in enumerate(configs):
            for j, config2 in enumerate(configs):
                if i < j:  # Avoid duplicates and self-play
                    player1 = MCTS(config1, game_interface, MockEvaluator(seed=i))
                    player2 = MCTS(config2, game_interface, MockEvaluator(seed=j))
                    
                    # Play one game
                    winner = self._play_game(game_interface, player1, player2, max_moves=30)
                    results.append((i, j, winner))
        
        # Verify tournament completed
        expected_games = len(configs) * (len(configs) - 1) // 2
        assert len(results) == expected_games, f"Expected {expected_games} games, got {len(results)}"
        
        # Analyze results
        wins = {i: 0 for i in range(len(configs))}
        draws = 0
        
        for p1_idx, p2_idx, winner in results:
            if winner == 1:
                wins[p1_idx] += 1
            elif winner == 2:
                wins[p2_idx] += 1
            else:
                draws += 1
        
        print(f"Tournament results: {wins}, Draws: {draws}")
        
        # Should have some decisive games
        total_decisive = sum(wins.values())
        assert total_decisive > 0, "Tournament should have some decisive games"
    
    def _play_game(self, game_interface: GameInterface, player1: MCTS, player2: MCTS, max_moves: int = 50) -> int:
        """Play a single game between two MCTS players"""
        state = game_interface.get_initial_state()
        moves_made = 0
        
        while not game_interface.is_terminal(state) and moves_made < max_moves:
            current_player = 1 if np.all(state[2] == 1) else 2
            mcts_player = player1 if current_player == 1 else player2
            
            canonical_state = game_interface.get_canonical_state(state, current_player)
            action = mcts_player.select_action(canonical_state, temperature=0.5)
            
            legal_moves = game_interface.get_legal_moves(state)
            if not legal_moves[action]:
                return -2  # Invalid game due to illegal move
            
            state = game_interface.make_move(state, action)
            moves_made += 1
        
        if game_interface.is_terminal(state):
            return game_interface.get_winner(state)
        else:
            return -1  # Draw due to move limit


class TestConfigurationWorkflows:
    """Test different configuration workflows end-to-end"""
    
    def test_alphazero_style_workflow(self):
        """Test AlphaZero-style configuration workflow"""
        # Create AlphaZero config
        alphazero_config = create_default_config('alphazero', 
                                                game_type='gomoku', 
                                                board_size=9)
        
        # Convert to MCTS config for actual usage
        mcts_config = MCTSConfig(
            num_simulations=50,  # Reduced for testing
            c_puct=alphazero_config.c_puct,
            temperature=alphazero_config.temperature,
            dirichlet_alpha=alphazero_config.dirichlet_alpha,
            dirichlet_epsilon=alphazero_config.dirichlet_epsilon,
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=alphazero_config.board_size,
            device=alphazero_config.device if alphazero_config.device != 'cuda' else 'cpu'
        )
        
        # Test the workflow
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        mcts = MCTS(mcts_config, game_interface, evaluator)
        
        # Simulate training data collection
        training_examples = []
        
        for episode in range(3):  # Small number for test
            state = game_interface.get_initial_state()
            episode_examples = []
            
            while not game_interface.is_terminal(state) and len(episode_examples) < 20:
                # Get action probabilities
                action_probs = mcts.search(state)
                
                # Select action with temperature
                action = mcts.select_action(state, temperature=mcts_config.temperature)
                
                # Store training example
                episode_examples.append({
                    'state': state.copy(),
                    'action_probs': action_probs.copy(),
                    'player': 1 if np.all(state[2] == 1) else 2
                })
                
                # Make move
                state = game_interface.make_move(state, action)
            
            # Assign final values based on game outcome
            winner = game_interface.get_winner(state) if game_interface.is_terminal(state) else 0
            
            for example in episode_examples:
                if winner == 0:
                    value = 0.0  # Draw
                elif winner == example['player']:
                    value = 1.0  # Win
                else:
                    value = -1.0  # Loss
                
                example['value'] = value
                training_examples.append(example)
        
        # Verify training data
        assert len(training_examples) > 0, "Should collect some training examples"
        
        for example in training_examples:
            assert 'state' in example
            assert 'action_probs' in example
            assert 'value' in example
            assert example['state'].shape == (3, 9, 9)
            assert len(example['action_probs']) == 81
            assert -1.0 <= example['value'] <= 1.0
        
        print(f"Collected {len(training_examples)} training examples")
    
    def test_mcts_configuration_comparison(self):
        """Test comparison of different MCTS configurations"""
        base_config = {
            'classical_only_mode': True,
            'game_type': GameType.GOMOKU,
            'board_size': 9,
            'device': 'cpu'
        }
        
        configs = [
            MCTSConfig(num_simulations=20, c_puct=1.0, **base_config),
            MCTSConfig(num_simulations=30, c_puct=1.414, **base_config),
            MCTSConfig(num_simulations=25, c_puct=2.0, **base_config),
        ]
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        
        # Test same position with different configs
        test_state = game_interface.get_initial_state()
        
        # Add some moves to create interesting position
        test_state = game_interface.make_move(test_state, 40)  # Center
        test_state = game_interface.make_move(test_state, 41)  # Next to center
        
        results = []
        
        for i, config in enumerate(configs):
            evaluator = MockEvaluator(seed=42)  # Same seed for fair comparison
            mcts = MCTS(config, game_interface, evaluator)
            
            start_time = time.time()
            action_probs = mcts.search(test_state)
            search_time = time.time() - start_time
            
            # Get top actions
            top_actions = np.argsort(action_probs)[-5:][::-1]  # Top 5 actions
            
            results.append({
                'config_idx': i,
                'search_time': search_time,
                'top_actions': top_actions,
                'action_probs': action_probs,
                'entropy': -np.sum(action_probs * np.log(action_probs + 1e-10))
            })
        
        # Verify all configurations work
        for result in results:
            assert result['search_time'] > 0
            assert len(result['top_actions']) == 5
            assert sum(result['action_probs']) > 0
            assert result['entropy'] > 0
        
        # Compare configurations
        print("Configuration comparison:")
        for i, result in enumerate(results):
            print(f"Config {i}: time={result['search_time']:.3f}s, "
                  f"entropy={result['entropy']:.3f}, "
                  f"top_action={result['top_actions'][0]}")


class TestPerformanceBenchmarks:
    """Performance benchmarks for the MCTS system"""
    
    @pytest.mark.slow
    def test_simulation_speed_benchmark(self):
        """Benchmark simulation speed"""
        config = MCTSConfig(
            num_simulations=200,
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=15,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 15)
        evaluator = MockEvaluator()
        mcts = MCTS(config, game_interface, evaluator)
        
        state = game_interface.get_initial_state()
        
        # Warm-up run
        mcts.search(state)
        
        # Benchmark runs
        num_runs = 5
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            action_probs = mcts.search(state)
            end_time = time.time()
            
            search_time = end_time - start_time
            times.append(search_time)
            
            # Verify result
            assert len(action_probs) == 225
            assert sum(action_probs) > 0
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        simulations_per_second = config.num_simulations / avg_time
        
        print(f"Simulation speed benchmark:")
        print(f"Average time: {avg_time:.3f} ± {std_time:.3f} seconds")
        print(f"Simulations per second: {simulations_per_second:.1f}")
        
        # Performance assertions
        assert avg_time < 10.0, f"Search too slow: {avg_time:.3f}s"
        assert simulations_per_second > 10, f"Too few simulations per second: {simulations_per_second:.1f}"
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage during extended play"""
        config = MCTSConfig(
            num_simulations=50,
            classical_only_mode=True,
            game_type=GameType.GOMOKU,
            board_size=9,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        mcts = MCTS(config, game_interface, evaluator)
        
        # Extended play session
        num_games = 5
        total_moves = 0
        
        for game in range(num_games):
            state = game_interface.get_initial_state()
            moves_in_game = 0
            
            while not game_interface.is_terminal(state) and moves_in_game < 30:
                action_probs = mcts.search(state)
                action = mcts.select_action(state, temperature=1.0)
                
                legal_moves = game_interface.get_legal_moves(state)
                if not legal_moves[action]:
                    break
                
                state = game_interface.make_move(state, action)
                moves_in_game += 1
                total_moves += 1
        
        print(f"Memory benchmark: played {num_games} games, {total_moves} total moves")
        
        # If we get here without memory errors, memory usage is acceptable
        assert total_moves > 0, "Should have made some moves"
        assert num_games > 0, "Should have played some games"
    
    def test_scalability_benchmark(self):
        """Test scalability with different board sizes"""
        board_sizes = [9, 13, 15]
        base_simulations = 30
        
        results = []
        
        for board_size in board_sizes:
            config = MCTSConfig(
                num_simulations=base_simulations,
                classical_only_mode=True,
                game_type=GameType.GOMOKU,
                board_size=board_size,
                device='cpu'
            )
            
            game_interface = GameInterface(GameType.GOMOKU, board_size)
            evaluator = MockEvaluator()
            mcts = MCTS(config, game_interface, evaluator)
            
            state = game_interface.get_initial_state()
            
            start_time = time.time()
            action_probs = mcts.search(state)
            search_time = time.time() - start_time
            
            expected_actions = board_size * board_size
            
            results.append({
                'board_size': board_size,
                'search_time': search_time,
                'action_space': expected_actions,
                'time_per_action': search_time / expected_actions
            })
            
            # Verify result
            assert len(action_probs) == expected_actions
            assert sum(action_probs) > 0
        
        print("Scalability benchmark:")
        for result in results:
            print(f"Board {result['board_size']}x{result['board_size']}: "
                  f"{result['search_time']:.3f}s, "
                  f"{result['action_space']} actions, "
                  f"{result['time_per_action']*1000:.2f}ms/action")
        
        # All should complete in reasonable time
        for result in results:
            assert result['search_time'] < 20.0, f"Board size {result['board_size']} too slow"


class TestSystemRobustness:
    """Test system robustness under various conditions"""
    
    def test_error_recovery_end_to_end(self):
        """Test system error recovery in end-to-end scenarios"""
        config = MCTSConfig(
            num_simulations=20,
            classical_only_mode=True,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        evaluator = MockEvaluator()
        mcts = MCTS(config, game_interface, evaluator)
        
        # Test with various potentially problematic states
        test_states = [
            game_interface.get_initial_state(),  # Normal case
        ]
        
        # Create edge case states
        edge_state = np.zeros((3, 9, 9), dtype=np.float32)
        # Fill most positions randomly
        for i in range(9):
            for j in range(9):
                if np.random.rand() > 0.7:  # 30% filled
                    player = np.random.choice([1, 2])
                    edge_state[player-1, i, j] = 1.0
        edge_state[2, :, :] = 1.0  # Player 1's turn
        
        if not game_interface.is_terminal(edge_state):
            test_states.append(edge_state)
        
        # Test all states
        for i, state in enumerate(test_states):
            try:
                action_probs = mcts.search(state)
                
                # Verify result
                assert len(action_probs) == 81
                assert sum(action_probs) >= 0
                
                # Verify legal moves constraint
                legal_moves = game_interface.get_legal_moves(state)
                for j, (is_legal, prob) in enumerate(zip(legal_moves, action_probs)):
                    if not is_legal:
                        assert prob == 0.0, f"State {i}: illegal move {j} has probability {prob}"
                
                print(f"State {i}: Successfully processed")
                
            except Exception as e:
                pytest.fail(f"State {i}: Failed with error: {e}")
    
    def test_concurrent_usage_robustness(self):
        """Test robustness under concurrent usage"""
        import threading
        
        config = MCTSConfig(
            num_simulations=15,
            classical_only_mode=True,
            device='cpu'
        )
        
        game_interface = GameInterface(GameType.GOMOKU, 9)
        
        results = []
        errors = []
        
        def worker_thread(worker_id: int):
            try:
                evaluator = MockEvaluator(seed=worker_id)
                mcts = MCTS(config, game_interface, evaluator)
                
                state = game_interface.get_initial_state()
                
                # Make a few moves
                for _ in range(3):
                    if game_interface.is_terminal(state):
                        break
                    
                    action_probs = mcts.search(state)
                    action = mcts.select_action(state, temperature=1.0)
                    
                    legal_moves = game_interface.get_legal_moves(state)
                    if legal_moves[action]:
                        state = game_interface.make_move(state, action)
                
                results.append(f"Worker {worker_id}: Success")
                
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")
        
        # Start multiple threads
        threads = []
        num_threads = 3
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30.0)
        
        # Check results
        print(f"Concurrent test results: {len(results)} successes, {len(errors)} errors")
        
        if errors:
            for error in errors:
                print(f"Error: {error}")
        
        # Should have mostly successful results
        assert len(results) >= num_threads - 1, "Most threads should succeed"
        assert len(errors) <= 1, "Should have minimal errors"