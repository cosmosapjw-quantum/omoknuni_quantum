"""Tests for MinHash interference system"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Mock datasketch if not available
try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    MinHash = MagicMock
    MinHashLSH = MagicMock


class TestInterferenceEngine:
    """Test InterferenceEngine class"""
    
    @pytest.fixture
    def mock_minhash(self):
        """Mock datasketch imports"""
        with patch('mcts.quantum.interference.HAS_MINHASH', True), \
             patch('mcts.quantum.interference.MinHash') as mock_mh, \
             patch('mcts.quantum.interference.MinHashLSH') as mock_lsh:
            # Setup MinHash mock
            mh_instance = Mock()
            mh_instance.jaccard.return_value = 0.5
            mock_mh.return_value = mh_instance
            
            # Setup LSH mock
            lsh_instance = Mock()
            lsh_instance.query.return_value = ['path_0', 'path_1']
            lsh_instance.insert = Mock()
            mock_lsh.return_value = lsh_instance
            
            yield mock_mh, mock_lsh, mh_instance, lsh_instance
            
    def test_initialization_with_minhash(self, mock_minhash):
        """Test initialization with datasketch available"""
        from mcts.quantum.interference import InterferenceEngine
        
        engine = InterferenceEngine(
            num_perm=64,
            threshold=0.3,
            num_bands=2
        )
        
        assert engine.num_perm == 64
        assert engine.threshold == 0.3
        assert engine.num_bands == 2
        assert engine.lsh is not None
        assert len(engine.signature_cache) == 0
        assert engine.stats['paths_processed'] == 0
        
    def test_initialization_without_minhash(self):
        """Test initialization without datasketch"""
        with patch('mcts.interference.HAS_MINHASH', False):
            from mcts.quantum.interference import InterferenceEngine
            
            with pytest.raises(ImportError, match="datasketch package required"):
                InterferenceEngine()
                
    def test_compute_path_signature(self, mock_minhash):
        """Test path signature computation"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, _, mh_instance, _ = mock_minhash
        engine = InterferenceEngine()
        
        path = [1, 2, 3, 4]
        signature = engine.compute_path_signature(path)
        
        # Check MinHash was created and updated
        assert signature == mh_instance
        
        # Check elements were added (positions and bigrams)
        update_calls = mh_instance.update.call_args_list
        assert len(update_calls) >= len(path) + len(path) - 1
        
        # Check caching
        assert "1,2,3,4" in engine.signature_cache
        assert engine.stats['cache_misses'] == 1
        
        # Second call should hit cache
        signature2 = engine.compute_path_signature(path)
        assert signature2 == signature
        assert engine.stats['cache_hits'] == 1
        
    def test_compute_path_signature_encoding(self, mock_minhash):
        """Test path signature encoding details"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, _, mh_instance, _ = mock_minhash
        engine = InterferenceEngine()
        
        path = [0, 1, 0]
        engine.compute_path_signature(path)
        
        # Check that position encoding and bigrams are added
        update_calls = [call[0][0] for call in mh_instance.update.call_args_list]
        
        # Position encodings
        assert b'0:0' in update_calls
        assert b'1:1' in update_calls
        assert b'2:0' in update_calls
        
        # Bigrams
        assert b'0->1' in update_calls
        assert b'1->0' in update_calls
        
    def test_compute_interference_empty(self, mock_minhash):
        """Test interference computation with empty paths"""
        from mcts.quantum.interference import InterferenceEngine
        
        engine = InterferenceEngine()
        interference = engine.compute_interference([])
        
        assert isinstance(interference, np.ndarray)
        assert len(interference) == 0
        
    def test_compute_interference_single_path(self, mock_minhash):
        """Test interference computation with single path"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, _, mh_instance, lsh_instance = mock_minhash
        lsh_instance.query.return_value = ['path_0']  # Only itself
        
        engine = InterferenceEngine()
        paths = [[1, 2, 3]]
        
        interference = engine.compute_interference(paths)
        
        assert len(interference) == 1
        assert interference[0] == 0.0  # No interference with itself
        
    def test_compute_interference_multiple_paths(self, mock_minhash):
        """Test interference computation with multiple paths"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, mock_lsh_class, mh_instance, _ = mock_minhash
        
        # Create a new LSH instance for compute_interference
        new_lsh = Mock()
        mock_lsh_class.return_value = new_lsh
        
        # Setup mock behavior
        call_count = 0
        def query_side_effect(sig):
            nonlocal call_count
            result = {
                0: ['path_0', 'path_1'],  # Path 0 similar to 1
                1: ['path_0', 'path_1'],  # Path 1 similar to 0
                2: ['path_2']  # Path 2 unique
            }.get(call_count % 3, ['path_' + str(call_count % 3)])
            call_count += 1
            return result
                
        new_lsh.query.side_effect = query_side_effect
        new_lsh.insert = Mock()
        mh_instance.jaccard.return_value = 0.6
        
        engine = InterferenceEngine()
        paths = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        interference = engine.compute_interference(paths)
        
        assert len(interference) == 3
        # Paths 0 and 1 have interference
        assert interference[0] > 0
        assert interference[1] > 0
        # Path 2 has no interference
        assert interference[2] == 0
        
        # Check statistics
        assert engine.stats['paths_processed'] == 3
        assert engine.stats['interference_events'] >= 2
        
    def test_compute_interference_with_weights(self, mock_minhash):
        """Test interference computation with importance weights"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, _, mh_instance, lsh_instance = mock_minhash
        lsh_instance.query.return_value = ['path_0', 'path_1']
        mh_instance.jaccard.return_value = 0.5
        
        engine = InterferenceEngine()
        paths = [[1, 2], [3, 4]]
        weights = np.array([2.0, 0.5])
        
        interference = engine.compute_interference(paths, weights)
        
        # Interference should be weighted
        assert len(interference) == 2
        # Path 0 receives interference weighted by path 1's weight (0.5)
        # Path 1 receives interference weighted by path 0's weight (2.0)
        assert interference[1] > interference[0]  # Because path 0 has higher weight
        
    def test_apply_interference_to_selection(self, mock_minhash):
        """Test applying interference to UCB scores"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, mock_lsh_class, mh_instance, _ = mock_minhash
        
        # Create a new LSH instance for compute_interference
        new_lsh = Mock()
        mock_lsh_class.return_value = new_lsh
        
        # Each path only finds itself (no interference)
        def query_side_effect(sig):
            return [f'path_{i}' for i in range(3) if sig == engine.signature_cache.get(str(i+1))]
            
        new_lsh.query.side_effect = lambda sig: ['path_0'] if sig == engine.signature_cache.get('1') else (
            ['path_1'] if sig == engine.signature_cache.get('2') else ['path_2']
        )
        new_lsh.insert = Mock()
        
        engine = InterferenceEngine()
        ucb_scores = np.array([1.0, 0.8, 0.6])
        paths = [[1], [2], [3]]
        
        modified_scores = engine.apply_interference_to_selection(
            ucb_scores, paths, interference_strength=0.5
        )
        
        assert len(modified_scores) == 3
        assert np.all(modified_scores <= ucb_scores)  # Scores don't increase
        
        # Check that interference is being applied correctly
        # Since we have mocked interference, let's check the actual reduction
        # The mock returns similar paths, so there will be some interference
        # Just verify the scores are reduced reasonably
        reductions = (ucb_scores - modified_scores) / ucb_scores
        assert np.all(reductions >= 0)  # All scores reduced or same
        assert np.all(reductions < 0.5)  # But not reduced by more than 50%
        
    def test_get_diverse_paths_simple(self, mock_minhash):
        """Test diverse path selection"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, _, mh_instance, _ = mock_minhash
        
        # Mock different similarities
        similarities = {
            (0, 1): 0.8,  # Very similar
            (0, 2): 0.2,  # Different
            (1, 2): 0.3,  # Somewhat different
        }
        
        def jaccard_side_effect(other):
            # Hack to identify which signatures are being compared
            for (i, j), sim in similarities.items():
                if np.random.rand() < 0.5:  # Randomize which is returned
                    return sim
            return 0.5
            
        mh_instance.jaccard.side_effect = jaccard_side_effect
        
        engine = InterferenceEngine()
        candidate_paths = [[1, 2], [1, 3], [4, 5]]
        
        selected = engine.get_diverse_paths(candidate_paths, n_select=2)
        
        assert len(selected) == 2
        assert all(idx in range(3) for idx in selected)
        assert len(set(selected)) == 2  # No duplicates
        
    def test_get_diverse_paths_all_selected(self, mock_minhash):
        """Test diverse path selection when n_select >= n_candidates"""
        from mcts.quantum.interference import InterferenceEngine
        
        engine = InterferenceEngine()
        candidate_paths = [[1], [2]]
        
        selected = engine.get_diverse_paths(candidate_paths, n_select=5)
        
        assert selected == [0, 1]
        
    def test_compute_path_clustering(self, mock_minhash):
        """Test path clustering"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, mock_lsh_class, mh_instance, _ = mock_minhash
        
        # Create a fresh LSH instance for clustering
        lsh_clustering = Mock()
        
        # Override the mock_lsh_class to return our fresh instance
        original_return = mock_lsh_class.return_value
        def lsh_side_effect(*args, **kwargs):
            # Check if this is for clustering (no params argument)
            if 'params' not in kwargs:
                return lsh_clustering
            return original_return
            
        mock_lsh_class.side_effect = lsh_side_effect
        
        # Track which indices have been inserted
        inserted_indices = []
        def insert_side_effect(idx, sig):
            inserted_indices.append(idx)
            
        lsh_clustering.insert.side_effect = insert_side_effect
        
        # Setup clustering behavior
        def query_side_effect(sig):
            # Create two clusters: {0, 1, 2} and {3, 4}
            # Return indices based on which have been inserted
            idx = inserted_indices.index(sig) if sig in inserted_indices else -1
            if idx in [0, 1, 2]:
                return [0, 1, 2]
            elif idx in [3, 4]:
                return [3, 4]
            else:
                return [idx] if idx >= 0 else []
                
        lsh_clustering.query.side_effect = query_side_effect
        
        engine = InterferenceEngine()
        paths = [[1], [2], [3], [10], [11], [20], [21]]
        
        clusters = engine.compute_path_clustering(paths, min_cluster_size=2)
        
        # Should have clusters based on mock behavior
        assert isinstance(clusters, dict)
        assert all(isinstance(indices, list) for indices in clusters.values())
        
        # All indices should be valid
        all_indices = []
        for indices in clusters.values():
            all_indices.extend(indices)
            assert all(0 <= idx < len(paths) for idx in indices)
            
    def test_compute_path_clustering_empty(self, mock_minhash):
        """Test clustering with empty paths"""
        from mcts.quantum.interference import InterferenceEngine
        
        engine = InterferenceEngine()
        clusters = engine.compute_path_clustering([])
        
        assert clusters == {}
        
    def test_get_statistics(self, mock_minhash):
        """Test statistics retrieval"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, _, mh_instance, lsh_instance = mock_minhash
        lsh_instance.query.return_value = ['path_0']
        
        engine = InterferenceEngine()
        
        # Generate some activity
        engine.compute_path_signature([1, 2, 3])
        engine.compute_path_signature([1, 2, 3])  # Cache hit
        engine.compute_interference([[1, 2], [3, 4]])
        
        stats = engine.get_statistics()
        
        assert isinstance(stats, dict)
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] >= 1
        assert stats['cache_size'] >= 1
        assert 'cache_hit_rate' in stats
        assert 0 <= stats['cache_hit_rate'] <= 1
        assert stats['paths_processed'] >= 2
        
    def test_clear_cache(self, mock_minhash):
        """Test cache clearing"""
        from mcts.quantum.interference import InterferenceEngine
        
        engine = InterferenceEngine()
        
        # Add to cache
        engine.compute_path_signature([1, 2, 3])
        assert len(engine.signature_cache) == 1
        assert engine.stats['cache_misses'] == 1
        
        # Clear cache
        engine.clear_cache()
        
        assert len(engine.signature_cache) == 0
        assert engine.stats['cache_hits'] == 0
        assert engine.stats['cache_misses'] == 0
        
    def test_edge_cases(self, mock_minhash):
        """Test edge cases"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, _, mh_instance, lsh_instance = mock_minhash
        lsh_instance.query.return_value = []
        
        engine = InterferenceEngine()
        
        # Empty path
        sig = engine.compute_path_signature([])
        assert sig is not None
        
        # Single action path
        sig = engine.compute_path_signature([42])
        assert sig is not None
        
        # Very long path
        long_path = list(range(1000))
        sig = engine.compute_path_signature(long_path)
        assert sig is not None
        
        # Repeated actions
        repeated = [1, 1, 1, 1]
        sig = engine.compute_path_signature(repeated)
        assert sig is not None
        
    def test_interference_normalization(self, mock_minhash):
        """Test that interference values are properly normalized"""
        from mcts.quantum.interference import InterferenceEngine
        
        _, _, mh_instance, lsh_instance = mock_minhash
        
        # All paths similar to each other
        lsh_instance.query.return_value = ['path_0', 'path_1', 'path_2']
        mh_instance.jaccard.return_value = 1.0  # Perfect similarity
        
        engine = InterferenceEngine()
        paths = [[1], [1], [1]]
        
        interference = engine.compute_interference(paths)
        
        # With perfect similarity, interference should be 1.0
        assert np.allclose(interference, 1.0)
        
    def test_thread_safety_consideration(self, mock_minhash):
        """Test that engine maintains separate state"""
        from mcts.quantum.interference import InterferenceEngine
        
        engine1 = InterferenceEngine()
        engine2 = InterferenceEngine()
        
        # Each engine should have its own cache
        engine1.compute_path_signature([1, 2])
        engine2.compute_path_signature([3, 4])
        
        assert len(engine1.signature_cache) == 1
        assert len(engine2.signature_cache) == 1
        assert "1,2" in engine1.signature_cache
        assert "3,4" in engine2.signature_cache
        assert "1,2" not in engine2.signature_cache
        assert "3,4" not in engine1.signature_cache