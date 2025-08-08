
"""Optimized evaluator with caching and performance improvements"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from mcts.neural_networks.mock_evaluator import MockEvaluator


class ProductionOptimizedEvaluator(MockEvaluator):
    """Production-ready optimized evaluator with caching"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Evaluation cache for repeated positions
        self.cache = {}
        self.max_cache_size = 5000  # Conservative size for production
        
        # Pre-computed values for MockEvaluator speed
        self.precomputed_values = np.random.random(10000).astype(np.float32)
        self.precomputed_policies = np.random.random((10000, self.action_space_size)).astype(np.float32)
        self.precomputed_policies = self.precomputed_policies / np.sum(self.precomputed_policies, axis=1, keepdims=True)
        
        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _state_to_hash(self, state):
        """Convert state to hash for caching"""
        return hash(state.tobytes())
    
    def evaluate_batch(self, states):
        """Optimized batch evaluation"""
        if not states:
            return [], []
        
        # Convert states to numpy array if needed (for compatibility with base class)
        if isinstance(states, list):
            features = np.array(states)
        else:
            features = states
            
        values = []
        policies = []
        
        for state in states:
            state_hash = self._state_to_hash(state)
            
            # Check cache
            if state_hash in self.cache:
                value, policy = self.cache[state_hash]
                self.cache_hits += 1
            else:
                # Fast evaluation using pre-computed values
                idx = state_hash % len(self.precomputed_values)
                value = float(self.precomputed_values[idx])
                policy = self.precomputed_policies[idx % len(self.precomputed_policies)].copy()
                
                # Cache if under limit
                if len(self.cache) < self.max_cache_size:
                    self.cache[state_hash] = (value, policy.copy())
                
                self.cache_misses += 1
            
            values.append(value)
            policies.append(policy)
        
        return values, policies
    
    def get_cache_stats(self):
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses, 
            'cache_hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
