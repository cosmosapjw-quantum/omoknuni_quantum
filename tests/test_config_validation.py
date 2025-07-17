"""Test configuration parameter validation

This test ensures critical configuration parameters are set correctly
to prevent training corruption.
"""

import pytest
import yaml
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfigValidation:
    """Test cases for configuration validation"""
    
    def test_csr_max_actions_for_gomoku(self):
        """Test that csr_max_actions is properly sized for Gomoku"""
        # For 15x15 Gomoku, we need at least 225 actions
        board_size = 15
        required_actions = board_size * board_size
        
        # Test various configs
        test_configs = [
            {'csr_max_actions': 10, 'should_fail': True},
            {'csr_max_actions': 100, 'should_fail': True},
            {'csr_max_actions': 225, 'should_fail': False},
            {'csr_max_actions': 256, 'should_fail': False},
            {'csr_max_actions': 512, 'should_fail': False},
        ]
        
        for config in test_configs:
            max_actions = config['csr_max_actions']
            should_fail = config['should_fail']
            
            if should_fail:
                assert max_actions < required_actions, f"csr_max_actions={max_actions} should be insufficient"
            else:
                assert max_actions >= required_actions, f"csr_max_actions={max_actions} should be sufficient"
    
    def test_dirichlet_epsilon_reasonable(self):
        """Test that dirichlet_epsilon is in reasonable range"""
        # Dirichlet epsilon should be small but not too large
        test_values = [
            {'epsilon': 0.0, 'valid': False},  # No noise is bad
            {'epsilon': 0.1, 'valid': True},
            {'epsilon': 0.25, 'valid': True},
            {'epsilon': 0.5, 'valid': False},  # Too much noise
            {'epsilon': 0.75, 'valid': False},
            {'epsilon': 1.0, 'valid': False},
        ]
        
        for test in test_values:
            epsilon = test['epsilon']
            expected_valid = test['valid']
            
            # Reasonable range is (0, 0.3]
            is_valid = 0 < epsilon <= 0.3
            
            assert is_valid == expected_valid, f"epsilon={epsilon} validation mismatch"
    
    def test_arena_simulation_match(self):
        """Test that arena simulations match training simulations"""
        # Arena should use same or more simulations than training
        test_cases = [
            {'training': 500, 'arena': 100, 'valid': False},
            {'training': 500, 'arena': 500, 'valid': True},
            {'training': 500, 'arena': 1000, 'valid': True},
            {'training': 100, 'arena': 50, 'valid': False},
        ]
        
        for case in test_cases:
            training_sims = case['training']
            arena_sims = case['arena']
            expected_valid = case['valid']
            
            # Arena should have at least as many simulations
            is_valid = arena_sims >= training_sims
            
            assert is_valid == expected_valid, f"training={training_sims}, arena={arena_sims} validation mismatch"
    
    def test_gomoku_config_file_validation(self):
        """Test actual gomoku config files"""
        config_files = [
            'configs/gomoku_classical.yaml',
            'configs/gomoku_improved_training.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Check game type
                if config.get('game', {}).get('game_type') == 'gomoku':
                    board_size = config.get('game', {}).get('board_size', 15)
                    required_actions = board_size * board_size
                    
                    # Check csr_max_actions
                    csr_max_actions = config.get('mcts', {}).get('csr_max_actions', 0)
                    if csr_max_actions > 0:
                        assert csr_max_actions >= required_actions, \
                            f"{config_file}: csr_max_actions={csr_max_actions} < {required_actions} needed for {board_size}x{board_size} board"
                    
                    # Check dirichlet_epsilon
                    dirichlet_epsilon = config.get('mcts', {}).get('dirichlet_epsilon', 0.25)
                    assert 0 < dirichlet_epsilon <= 0.3, \
                        f"{config_file}: dirichlet_epsilon={dirichlet_epsilon} outside reasonable range (0, 0.3]"
                    
                    # Check arena simulations
                    training_sims = config.get('mcts', {}).get('num_simulations', 100)
                    arena_sims = config.get('arena', {}).get('mcts_simulations', 100)
                    assert arena_sims >= training_sims * 0.2, \
                        f"{config_file}: arena simulations ({arena_sims}) too low compared to training ({training_sims})"
    
    def test_default_csr_max_actions_minimum(self):
        """Test that default csr_max_actions is at least 225"""
        from python.mcts.core.mcts_config import MCTSConfig
        from python.mcts.utils.config_system import MCTSFullConfig
        
        # Check MCTSConfig default
        config = MCTSConfig()
        assert config.csr_max_actions >= 225, f"MCTSConfig default csr_max_actions={config.csr_max_actions} < 225"
        
        # Check MCTSFullConfig default
        full_config = MCTSFullConfig()
        assert full_config.csr_max_actions >= 225, f"MCTSFullConfig default csr_max_actions={full_config.csr_max_actions} < 225"
    
    def test_validate_training_parameters(self):
        """Test other important training parameters"""
        # Test learning rate
        lr_tests = [
            {'lr': 0.0, 'valid': False},
            {'lr': 0.00001, 'valid': True},
            {'lr': 0.002, 'valid': True},
            {'lr': 0.1, 'valid': False},  # Too high
            {'lr': 1.0, 'valid': False},
        ]
        
        for test in lr_tests:
            lr = test['lr']
            expected_valid = test['valid']
            
            # Reasonable range for AlphaZero training
            is_valid = 0 < lr <= 0.01
            
            assert is_valid == expected_valid, f"lr={lr} validation mismatch"
        
        # Test temperature threshold
        temp_threshold_tests = [
            {'threshold': 0, 'valid': False},  # No exploration
            {'threshold': 10, 'valid': True},
            {'threshold': 30, 'valid': True},
            {'threshold': 100, 'valid': False},  # Too much exploration
        ]
        
        for test in temp_threshold_tests:
            threshold = test['threshold']
            expected_valid = test['valid']
            
            # Reasonable range
            is_valid = 5 <= threshold <= 50
            
            assert is_valid == expected_valid, f"temperature_threshold={threshold} validation mismatch"


def validate_config_dict(config: dict) -> list:
    """Validate a configuration dictionary and return list of issues"""
    issues = []
    
    # Check game-specific parameters
    game_type = config.get('game', {}).get('game_type', '')
    if game_type == 'gomoku':
        board_size = config.get('game', {}).get('board_size', 15)
        required_actions = board_size * board_size
        
        # Check csr_max_actions
        csr_max_actions = config.get('mcts', {}).get('csr_max_actions', 0)
        if csr_max_actions > 0 and csr_max_actions < required_actions:
            issues.append(f"csr_max_actions={csr_max_actions} insufficient for {board_size}x{board_size} Gomoku (need {required_actions})")
    
    # Check dirichlet_epsilon
    dirichlet_epsilon = config.get('mcts', {}).get('dirichlet_epsilon', 0.25)
    if dirichlet_epsilon <= 0 or dirichlet_epsilon > 0.3:
        issues.append(f"dirichlet_epsilon={dirichlet_epsilon} outside reasonable range (0, 0.3]")
    
    # Check arena/training simulation match
    training_sims = config.get('mcts', {}).get('num_simulations', 100)
    arena_sims = config.get('arena', {}).get('mcts_simulations', 100)
    if arena_sims < training_sims * 0.2:
        issues.append(f"arena simulations ({arena_sims}) too low compared to training ({training_sims})")
    
    # Check learning rate
    lr = config.get('training', {}).get('learning_rate', 0.001)
    if lr <= 0 or lr > 0.01:
        issues.append(f"learning_rate={lr} outside reasonable range (0, 0.01]")
    
    return issues


if __name__ == "__main__":
    # Run tests
    test = TestConfigValidation()
    
    print("Running test: csr_max_actions for Gomoku...")
    try:
        test.test_csr_max_actions_for_gomoku()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nRunning test: dirichlet_epsilon reasonable...")
    try:
        test.test_dirichlet_epsilon_reasonable()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nRunning test: arena simulation match...")
    try:
        test.test_arena_simulation_match()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nRunning test: default csr_max_actions minimum...")
    try:
        test.test_default_csr_max_actions_minimum()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nRunning test: validate training parameters...")
    try:
        test.test_validate_training_parameters()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nRunning test: gomoku config file validation...")
    try:
        test.test_gomoku_config_file_validation()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")