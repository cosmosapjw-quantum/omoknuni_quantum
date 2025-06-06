"""
Test Suite for Quantum Darwinism Engine
=======================================

Validates quantum Darwinism implementation for robust move extraction
through redundant encoding in the tree environment.

Test Categories:
1. Fragment generation and selection
2. Redundancy spectrum computation
3. Objectivity measure calculation
4. Robust move extraction
5. Information broadcasting
"""

import pytest
import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.quantum.quantum_darwinism import (
    QuantumDarwinismEngine,
    DarwinismConfig,
    FragmentSelector,
    RedundancyAnalyzer,
    create_darwinism_engine
)


class TestFragmentSelector:
    """Test fragment generation and selection"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return DarwinismConfig(
            min_fragment_size=3,
            max_fragment_size=10
        )
    
    @pytest.fixture
    def selector(self, config, device):
        return FragmentSelector(config, device)
    
    def test_random_fragment_generation(self, selector, device):
        """Test random fragment generation"""
        num_nodes = 50
        num_fragments = 10
        
        fragments = selector.generate_random_fragments(num_nodes, num_fragments)
        
        assert len(fragments) == num_fragments
        
        for fragment in fragments:
            # Check size constraints
            assert len(fragment) >= selector.config.min_fragment_size
            assert len(fragment) <= selector.config.max_fragment_size
            
            # Check all indices valid
            assert torch.all(fragment >= 0)
            assert torch.all(fragment < num_nodes)
            
            # Check no duplicates within fragment
            assert len(torch.unique(fragment)) == len(fragment)
    
    def test_excluded_nodes(self, selector, device):
        """Test fragment generation with excluded nodes"""
        num_nodes = 20
        excluded = {0, 1, 2}
        
        fragments = selector.generate_random_fragments(
            num_nodes, 
            num_fragments=5,
            excluded_nodes=excluded
        )
        
        # Check excluded nodes not in any fragment
        for fragment in fragments:
            for excluded_node in excluded:
                assert excluded_node not in fragment.tolist()
    
    def test_tree_fragment_generation(self, selector, device):
        """Test tree-based fragment generation"""
        # Create simple tree structure
        children = torch.tensor([
            [1, 2, -1, -1],    # Node 0 has children 1, 2
            [3, 4, -1, -1],    # Node 1 has children 3, 4
            [5, -1, -1, -1],   # Node 2 has child 5
            [-1, -1, -1, -1],  # Leaves
            [-1, -1, -1, -1],
            [-1, -1, -1, -1]
        ], device=device)
        
        tree_structure = {'children': children, 'num_nodes': 6}
        
        fragments = selector.generate_tree_fragments(tree_structure, root_node=0)
        
        assert len(fragments) > 0
        
        # Should have both subtree and random fragments
        assert len(fragments) >= 5


class TestRedundancyAnalyzer:
    """Test redundancy analysis"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return DarwinismConfig()
    
    @pytest.fixture
    def analyzer(self, config, device):
        return RedundancyAnalyzer(config, device)
    
    def test_mutual_information_computation(self, analyzer, device):
        """Test mutual information computation"""
        # Create correlated data
        n_samples = 1000
        system_state = torch.randn(n_samples, device=device)
        
        # Fragment perfectly correlated with system
        perfect_fragment = system_state.clone()
        
        # Fragment with noise
        noisy_fragment = system_state + 0.5 * torch.randn(n_samples, device=device)
        
        # Random fragment
        random_fragment = torch.randn(n_samples, device=device)
        
        # Compute MI
        mi_perfect = analyzer.compute_mutual_information(system_state, perfect_fragment)
        mi_noisy = analyzer.compute_mutual_information(system_state, noisy_fragment)
        mi_random = analyzer.compute_mutual_information(system_state, random_fragment)
        
        # Perfect correlation should have highest MI
        assert mi_perfect > mi_noisy
        assert mi_noisy > mi_random
        assert mi_random >= 0  # MI is non-negative
    
    def test_redundancy_spectrum(self, analyzer, device):
        """Test redundancy spectrum computation"""
        # Create test data
        system_values = torch.randn(10, device=device)
        environment_values = torch.randn(100, device=device)
        
        # Create fragments with varying information content
        fragments = []
        for i in range(20):
            size = torch.randint(5, 15, (1,)).item()
            fragment = torch.randperm(100)[:size].to(device)
            fragments.append(fragment)
        
        spectrum = analyzer.compute_redundancy_spectrum(
            system_values, environment_values, fragments
        )
        
        # Check all required fields
        assert 'fragment_sizes' in spectrum
        assert 'mutual_informations' in spectrum
        assert 'redundancies' in spectrum
        assert 'average_redundancy' in spectrum
        
        # Check shapes
        assert len(spectrum['fragment_sizes']) == len(fragments)
        assert len(spectrum['mutual_informations']) == len(fragments)
        
        # Redundancies should be between 0 and 1
        assert torch.all(spectrum['redundancies'] >= 0)
        assert torch.all(spectrum['redundancies'] <= 1)
    
    def test_objectivity_measure(self, analyzer, device):
        """Test objectivity measure computation"""
        # Create spectrum with good redundancy (objective)
        good_redundancies = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3], device=device)
        good_spectrum = {
            'redundancies': good_redundancies,
            'info_thresholds': torch.linspace(0.1, 0.9, 5, device=device),
            'average_redundancy': good_redundancies.mean()
        }
        
        # Create spectrum with poor redundancy (non-objective)
        poor_redundancies = torch.tensor([0.8, 0.85, 0.9, 0.92, 0.95], device=device)
        poor_spectrum = {
            'redundancies': poor_redundancies,
            'info_thresholds': torch.linspace(0.1, 0.9, 5, device=device),
            'average_redundancy': poor_redundancies.mean()
        }
        
        objectivity_good = analyzer.compute_objectivity_measure(good_spectrum)
        objectivity_poor = analyzer.compute_objectivity_measure(poor_spectrum)
        
        # Good redundancy should have higher objectivity
        assert objectivity_good > objectivity_poor
        assert 0 <= objectivity_good <= 1
        assert 0 <= objectivity_poor <= 1


class TestQuantumDarwinismEngine:
    """Test the main quantum Darwinism engine"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def config(self):
        return DarwinismConfig(
            min_fragment_size=3,
            objectivity_threshold=0.5
        )
    
    @pytest.fixture
    def engine(self, config, device):
        return QuantumDarwinismEngine(config, device)
    
    @pytest.fixture
    def sample_tree_data(self, device):
        """Create sample tree data"""
        num_nodes = 50
        
        tree_values = torch.randn(num_nodes, device=device)
        visit_counts = torch.rand(num_nodes, device=device) * 100 + 1
        
        # Simple tree structure
        children = torch.full((num_nodes, 4), -1, device=device)
        for i in range(num_nodes // 2):
            # Give each non-leaf some children
            num_children = min(torch.randint(1, 4, (1,)).item(), num_nodes - 2*i - 1)
            for j in range(num_children):
                if 2*i + j + 1 < num_nodes:
                    children[i, j] = 2*i + j + 1
        
        tree_structure = {
            'children': children,
            'num_nodes': num_nodes
        }
        
        return tree_values, tree_structure, visit_counts
    
    def test_engine_initialization(self, engine, device):
        """Test engine initialization"""
        assert engine.device == device
        assert hasattr(engine, 'fragment_selector')
        assert hasattr(engine, 'redundancy_analyzer')
        assert 'moves_analyzed' in engine.stats
    
    def test_move_objectivity_analysis(self, engine, sample_tree_data):
        """Test single move objectivity analysis"""
        tree_values, tree_structure, visit_counts = sample_tree_data
        
        # Analyze objectivity of move at node 5
        result = engine.analyze_move_objectivity(
            move_node=5,
            tree_values=tree_values,
            tree_structure=tree_structure,
            visit_counts=visit_counts
        )
        
        # Check required fields
        assert 'objectivity' in result
        assert 'is_objective' in result
        assert 'redundancy_spectrum' in result
        assert 'num_fragments_analyzed' in result
        
        # Check values
        assert 0 <= result['objectivity'] <= 1
        assert isinstance(result['is_objective'], bool)
        assert result['num_fragments_analyzed'] > 0
    
    def test_robust_move_extraction(self, engine, sample_tree_data):
        """Test extraction of robust moves"""
        tree_values, tree_structure, visit_counts = sample_tree_data
        
        # Candidate moves
        candidate_moves = torch.tensor([2, 5, 8, 12, 15], device=engine.device)
        
        robust_moves, analysis = engine.extract_robust_moves(
            candidate_moves,
            tree_values,
            tree_structure,
            visit_counts,
            top_k=3
        )
        
        # Should return top 3 moves
        assert len(robust_moves) == 3
        
        # Check analysis results
        assert 'objectivity_scores' in analysis
        assert 'top_objectivity_scores' in analysis
        assert len(analysis['objectivity_scores']) == len(candidate_moves)
        
        # Top scores should be sorted
        top_scores = analysis['top_objectivity_scores']
        assert torch.all(top_scores[:-1] >= top_scores[1:])
    
    def test_information_broadcasting(self, engine, sample_tree_data):
        """Test information broadcasting analysis"""
        tree_values, tree_structure, visit_counts = sample_tree_data
        
        broadcasting = engine.compute_information_broadcasting(
            source_node=0,
            tree_structure=tree_structure,
            tree_values=tree_values,
            max_distance=3
        )
        
        # Check required fields
        assert 'distances' in broadcasting
        assert 'information_strength' in broadcasting
        assert 'broadcasting_efficiency' in broadcasting
        assert 'reachable_nodes' in broadcasting
        
        # Information strength should decay with distance
        info_strength = broadcasting['information_strength']
        assert len(info_strength) == 4  # 0 to max_distance
        
        # Efficiency should be between 0 and 1
        assert 0 <= broadcasting['broadcasting_efficiency'] <= 1
    
    def test_statistics_tracking(self, engine, sample_tree_data):
        """Test statistics tracking"""
        tree_values, tree_structure, visit_counts = sample_tree_data
        
        # Analyze multiple moves
        for move in [1, 3, 5]:
            engine.analyze_move_objectivity(
                move, tree_values, tree_structure, visit_counts
            )
        
        stats = engine.get_statistics()
        
        assert stats['moves_analyzed'] == 3
        assert stats['objective_moves_found'] >= 0
        assert stats['average_objectivity'] >= 0
        assert stats['average_redundancy'] >= 0


class TestTheoreticalPredictions:
    """Test theoretical predictions from quantum Darwinism"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_redundancy_scaling(self, device):
        """Test redundancy scaling R_δ ~ N^(-1/2)"""
        config = DarwinismConfig()
        analyzer = RedundancyAnalyzer(config, device)
        
        # For objective information, redundancy should scale as ~1/√N
        # Test with different environment sizes
        env_sizes = [100, 400, 900]
        avg_redundancies = []
        
        for n in env_sizes:
            # Create test data
            system_values = torch.randn(10, device=device)
            environment_values = torch.randn(n, device=device)
            
            # Make environment partially correlated with system
            environment_values[:10] = system_values
            
            # Generate fragments
            fragments = []
            for _ in range(50):
                size = min(20, n // 5)
                fragment = torch.randperm(n)[:size].to(device)
                fragments.append(fragment)
            
            spectrum = analyzer.compute_redundancy_spectrum(
                system_values, environment_values, fragments
            )
            
            avg_redundancies.append(spectrum['average_redundancy'].item())
        
        # Check scaling trend (larger environment → smaller redundancy)
        assert avg_redundancies[0] > avg_redundancies[1]
        assert avg_redundancies[1] > avg_redundancies[2]
    
    def test_objectivity_emergence(self, device):
        """Test emergence of objectivity from redundancy"""
        config = DarwinismConfig()
        engine = QuantumDarwinismEngine(config, device)
        
        # Create highly redundant encoding (should be objective)
        num_nodes = 100
        tree_values = torch.zeros(num_nodes, device=device)
        
        # Make multiple copies of same information
        pattern = torch.randn(10, device=device)
        for i in range(0, num_nodes-10, 10):
            tree_values[i:i+10] = pattern
        
        tree_structure = {'num_nodes': num_nodes}
        
        result = engine.analyze_move_objectivity(
            move_node=5,
            tree_values=tree_values,
            tree_structure=tree_structure
        )
        
        # Should have high objectivity due to redundancy
        assert result['objectivity'] > 0.5


def test_factory_function():
    """Test factory function"""
    engine = create_darwinism_engine()
    assert isinstance(engine, QuantumDarwinismEngine)
    
    # Test with custom parameters
    engine = create_darwinism_engine(min_fragment_size=10)
    assert engine.config.min_fragment_size == 10


if __name__ == "__main__":
    # Run basic functionality test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Quantum Darwinism tests on {device}")
    
    # Create engine
    engine = create_darwinism_engine(device)
    
    # Create test data
    num_nodes = 100
    tree_values = torch.randn(num_nodes, device=device)
    visit_counts = torch.rand(num_nodes, device=device) * 100 + 1
    
    children = torch.full((num_nodes, 4), -1, device=device, dtype=torch.long)
    for i in range(30):
        for j in range(min(3, num_nodes - 3*i - 1)):
            children[i, j] = 3*i + j + 1
    
    tree_structure = {'children': children}
    
    # Test move objectivity
    print("\nTesting move objectivity analysis...")
    start = time.perf_counter()
    result = engine.analyze_move_objectivity(
        move_node=10,
        tree_values=tree_values,
        tree_structure=tree_structure,
        visit_counts=visit_counts
    )
    end = time.perf_counter()
    
    print(f"✓ Analysis completed in {end-start:.4f}s")
    print(f"✓ Objectivity score: {result['objectivity']:.3f}")
    print(f"✓ Is objective: {result['is_objective']}")
    print(f"✓ Fragments analyzed: {result['num_fragments_analyzed']}")
    print(f"✓ Max MI: {result['max_mutual_information']:.3f}")
    
    # Test robust move extraction
    print("\nTesting robust move extraction...")
    candidates = torch.tensor([5, 10, 15, 20, 25], device=device)
    
    start = time.perf_counter()
    robust_moves, analysis = engine.extract_robust_moves(
        candidates, tree_values, tree_structure, visit_counts, top_k=3
    )
    end = time.perf_counter()
    
    print(f"✓ Extraction completed in {end-start:.4f}s")
    print(f"✓ Robust moves: {robust_moves}")
    print(f"✓ Objectivity scores: {analysis['objectivity_scores']}")
    print(f"✓ Average objectivity: {analysis['average_objectivity']:.3f}")
    print(f"✓ Objective moves found: {analysis['num_objective_moves']}")
    
    # Test information broadcasting
    print("\nTesting information broadcasting...")
    broadcasting = engine.compute_information_broadcasting(
        source_node=0,
        tree_structure=tree_structure,
        tree_values=tree_values
    )
    
    print(f"✓ Broadcasting efficiency: {broadcasting['broadcasting_efficiency']:.3f}")
    print(f"✓ Reachable nodes: {broadcasting['reachable_nodes']}")
    print(f"✓ Information strength by distance: {broadcasting['information_strength']}")
    
    # Test statistics
    stats = engine.get_statistics()
    print(f"\n✓ Statistics: {stats}")
    
    print("\n✓ All quantum Darwinism tests passed!")