"""Test all configuration files for validity"""

# import pytest
import yaml
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_config_validation import validate_config_dict


class TestAllConfigs:
    """Test all config files in configs directory"""
    
    def test_all_yaml_configs(self):
        """Test all YAML config files"""
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')
        
        # Game-specific requirements
        game_requirements = {
            'chess': {
                'min_csr_max_actions': 4096,
                'expected_dirichlet_alpha': 0.3,
                'board_size': 8
            },
            'go': {
                'min_csr_max_actions': 361,  # 19x19
                'expected_dirichlet_alpha': 0.03,  # Lower for large board
                'board_size': 19
            },
            'gomoku': {
                'min_csr_max_actions': 225,  # 15x15
                'expected_dirichlet_alpha': 0.3,
                'board_size': 15
            }
        }
        
        errors = []
        
        for filename in os.listdir(config_dir):
            if filename.endswith('.yaml'):
                filepath = os.path.join(config_dir, filename)
                print(f"\nTesting {filename}...")
                
                try:
                    with open(filepath, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    # Skip special configs without game type
                    special_configs = ['optimized_physics_analysis.yaml', 
                                      'optimized_selfplay.yaml',
                                      'optuna_optimization_base.yaml']
                    if filename in special_configs:
                        print(f"  Skipping {filename} (special optimization/analysis config)")
                        continue
                    
                    # Get game type
                    game_type = config.get('game', {}).get('game_type', '')
                    if not game_type:
                        errors.append(f"{filename}: Missing game_type")
                        continue
                    
                    # Check game-specific requirements
                    if game_type in game_requirements:
                        reqs = game_requirements[game_type]
                        
                        # Check csr_max_actions
                        csr_max_actions = config.get('mcts', {}).get('csr_max_actions', 0)
                        if csr_max_actions < reqs['min_csr_max_actions']:
                            errors.append(f"{filename}: csr_max_actions={csr_max_actions} < {reqs['min_csr_max_actions']} required for {game_type}")
                        
                        # Check dirichlet_alpha (with tolerance)
                        dirichlet_alpha = config.get('mcts', {}).get('dirichlet_alpha', 0)
                        expected_alpha = reqs['expected_dirichlet_alpha']
                        if abs(dirichlet_alpha - expected_alpha) > 0.1 and dirichlet_alpha > 0:
                            print(f"  Warning: {filename} has dirichlet_alpha={dirichlet_alpha}, expected ~{expected_alpha} for {game_type}")
                    
                    # Check arena simulations
                    training_sims = config.get('mcts', {}).get('num_simulations', 0)
                    arena_sims = config.get('arena', {}).get('mcts_simulations', 0)
                    if arena_sims > 0 and training_sims > 0:
                        ratio = arena_sims / training_sims
                        if ratio < 0.5:
                            print(f"  Warning: Arena simulations ({arena_sims}) < 50% of training ({training_sims})")
                    
                    # Run general validation
                    issues = validate_config_dict(config)
                    if issues:
                        for issue in issues:
                            errors.append(f"{filename}: {issue}")
                    else:
                        print(f"  ✓ {filename} passed all checks")
                        
                except Exception as e:
                    errors.append(f"{filename}: Error loading config - {e}")
        
        # Report results
        if errors:
            print("\n=== Configuration Errors ===")
            for error in errors:
                print(f"✗ {error}")
            assert False, f"Found {len(errors)} configuration errors"
        else:
            print("\n=== All Configurations Valid ===")
            print("✓ All config files passed validation")


if __name__ == "__main__":
    test = TestAllConfigs()
    test.test_all_yaml_configs()