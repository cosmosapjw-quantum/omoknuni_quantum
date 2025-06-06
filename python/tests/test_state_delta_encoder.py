"""Tests for state delta encoding system

This module tests the state delta encoder which is critical for reducing
memory bandwidth and improving cache efficiency.
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple
import time

from mcts.utils.state_delta_encoder import StateDeltaEncoder, DeltaCache, StateCheckpoint


class TestStateDeltaEncoder:
    """Test state delta encoding functionality"""
    
    @pytest.fixture
    def sample_states(self):
        """Create sample game states for testing"""
        # Gomoku-like states (15x15 board, 2 channels for players)
        states = []
        
        # Empty board
        state0 = torch.zeros(2, 15, 15)
        states.append(state0)
        
        # Single move
        state1 = state0.clone()
        state1[0, 7, 7] = 1  # Player 1 center
        states.append(state1)
        
        # Two moves
        state2 = state1.clone()
        state2[1, 7, 8] = 1  # Player 2 adjacent
        states.append(state2)
        
        # Three moves
        state3 = state2.clone()
        state3[0, 6, 7] = 1  # Player 1 forms line
        states.append(state3)
        
        return states
        
    def test_basic_delta_encoding(self, sample_states):
        """Test basic delta encoding and decoding"""
        encoder = StateDeltaEncoder(state_shape=(2, 15, 15))
        
        # Encode sequence
        deltas = []
        for i in range(1, len(sample_states)):
            delta = encoder.encode_delta(sample_states[i-1], sample_states[i])
            deltas.append(delta)
            
            # Verify delta is sparse
            assert delta['positions'].numel() < 10  # Should only have a few changes
            
        # Decode and verify
        reconstructed = sample_states[0].clone()
        for i, delta in enumerate(deltas):
            reconstructed = encoder.apply_delta(reconstructed, delta)
            assert torch.allclose(reconstructed, sample_states[i+1])
            
    def test_delta_compression_ratio(self, sample_states):
        """Test compression ratio of delta encoding"""
        encoder = StateDeltaEncoder(state_shape=(2, 15, 15))
        
        total_full_size = 0
        total_delta_size = 0
        
        for i in range(1, len(sample_states)):
            # Full state size
            full_size = sample_states[i].numel() * 4  # 4 bytes per float32
            total_full_size += full_size
            
            # Delta size
            delta = encoder.encode_delta(sample_states[i-1], sample_states[i])
            delta_size = (delta['positions'].numel() + 
                         delta['values'].numel()) * 4
            total_delta_size += delta_size
            
        compression_ratio = total_full_size / total_delta_size
        print(f"\nCompression ratio: {compression_ratio:.2f}x")
        
        # Should achieve significant compression for sparse updates
        assert compression_ratio > 10.0
        
    def test_checkpoint_system(self, sample_states):
        """Test checkpoint creation and restoration"""
        encoder = StateDeltaEncoder(
            state_shape=(2, 15, 15),
            checkpoint_interval=2
        )
        
        # Process states and create checkpoints
        for i, state in enumerate(sample_states):
            should_checkpoint = encoder.should_checkpoint(i)
            if should_checkpoint:
                checkpoint = encoder.create_checkpoint(i, state)
                assert checkpoint.state_id == i
                assert torch.allclose(checkpoint.full_state, state)
                
    def test_delta_cache(self):
        """Test delta caching system"""
        cache = DeltaCache(max_size=100)
        
        # Add some deltas
        for i in range(10):
            delta = {
                'positions': torch.tensor([[0, i, i]]),
                'values': torch.tensor([float(i)])
            }
            cache.add(i, i+1, delta)
            
        # Retrieve deltas
        delta = cache.get(0, 1)
        assert delta is not None
        assert delta['values'][0] == 0.0
        
        # Test cache miss
        delta = cache.get(100, 101)
        assert delta is None
        
        # Test LRU eviction
        for i in range(100):
            cache.add(i+10, i+11, {
                'positions': torch.tensor([[0, 0, 0]]),
                'values': torch.tensor([1.0])
            })
            
        # Original entries should be evicted
        assert cache.get(0, 1) is None
        
    def test_batch_delta_encoding(self):
        """Test encoding deltas for multiple state pairs"""
        encoder = StateDeltaEncoder(state_shape=(2, 8, 8))
        
        # Create batch of state pairs
        batch_size = 32
        prev_states = torch.rand(batch_size, 2, 8, 8)
        next_states = prev_states.clone()
        
        # Make sparse changes
        for i in range(batch_size):
            # Random position
            c, x, y = np.random.randint(0, 2), np.random.randint(0, 8), np.random.randint(0, 8)
            next_states[i, c, x, y] = 1 - next_states[i, c, x, y]
            
        # Encode batch
        batch_deltas = encoder.encode_batch_deltas(prev_states, next_states)
        
        assert len(batch_deltas) == batch_size
        
        # Verify each delta
        for i, delta in enumerate(batch_deltas):
            reconstructed = encoder.apply_delta(prev_states[i], delta)
            assert torch.allclose(reconstructed, next_states[i])
            
    def test_gpu_delta_encoding(self):
        """Test delta encoding on GPU if available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        encoder = StateDeltaEncoder(
            state_shape=(2, 15, 15),
            device='cuda'
        )
        
        # Create states on GPU
        state1 = torch.zeros(2, 15, 15, device='cuda')
        state2 = state1.clone()
        state2[0, 7, 7] = 1.0
        
        # Encode on GPU
        delta = encoder.encode_delta(state1, state2)
        
        assert delta['positions'].device.type == 'cuda'
        assert delta['values'].device.type == 'cuda'
        
        # Decode on GPU
        reconstructed = encoder.apply_delta(state1, delta)
        assert torch.allclose(reconstructed, state2)
        
    def test_incremental_path_encoding(self):
        """Test encoding a full path incrementally"""
        encoder = StateDeltaEncoder(state_shape=(2, 15, 15))
        
        # Simulate a game path
        path_length = 20
        states = [torch.zeros(2, 15, 15)]
        
        for i in range(path_length):
            new_state = states[-1].clone()
            # Alternate players
            player = i % 2
            # Place in a line
            new_state[player, i // 2, i // 2] = 1.0
            states.append(new_state)
            
        # Encode path as deltas
        encoded_path = encoder.encode_path(states)
        
        assert len(encoded_path['deltas']) == path_length
        assert 'checkpoints' in encoded_path
        
        # Reconstruct any state efficiently
        for target_idx in [5, 10, 15, 20]:
            reconstructed = encoder.reconstruct_state(encoded_path, target_idx)
            assert torch.allclose(reconstructed, states[target_idx])
            
    def test_memory_efficiency(self):
        """Test memory usage of delta encoding vs full states"""
        n_states = 100
        state_shape = (2, 19, 19)  # Go board
        
        # Full state storage
        full_states = []
        for i in range(n_states):
            state = torch.zeros(state_shape)
            # Add i random stones
            for _ in range(i):
                x, y = np.random.randint(0, 19, size=2)
                player = np.random.randint(0, 2)
                state[player, x, y] = 1.0
            full_states.append(state)
            
        # Calculate full storage size
        full_size = sum(s.numel() * 4 for s in full_states)  # 4 bytes per float32
        
        # Delta encoding
        encoder = StateDeltaEncoder(state_shape)
        encoded = encoder.encode_path(full_states)
        
        # Calculate delta storage size
        delta_size = 0
        for delta in encoded['deltas']:
            if delta is not None:
                delta_size += (delta['positions'].numel() + 
                             delta['values'].numel()) * 4
                             
        # Add checkpoint sizes
        for checkpoint in encoded['checkpoints']:
            delta_size += checkpoint.full_state.numel() * 4
            
        compression_ratio = full_size / delta_size
        print(f"\nMemory compression: {compression_ratio:.2f}x")
        print(f"Full size: {full_size / 1024:.2f} KB")
        print(f"Delta size: {delta_size / 1024:.2f} KB")
        
        assert compression_ratio > 1.5  # Should achieve some compression
        

class TestStateDeltaPerformance:
    """Performance tests for state delta encoding"""
    
    @pytest.mark.benchmark
    def test_encoding_speed(self):
        """Benchmark delta encoding speed"""
        encoder = StateDeltaEncoder(state_shape=(2, 15, 15))
        
        # Create random state changes
        n_iterations = 1000
        state1 = torch.rand(2, 15, 15)
        state2 = state1.clone()
        
        # Make sparse changes
        for _ in range(5):
            x, y = np.random.randint(0, 15, size=2)
            state2[0, x, y] = 1 - state2[0, x, y]
            
        # Benchmark encoding
        start = time.time()
        for _ in range(n_iterations):
            delta = encoder.encode_delta(state1, state2)
        encoding_time = time.time() - start
        
        # Benchmark decoding
        start = time.time()
        for _ in range(n_iterations):
            reconstructed = encoder.apply_delta(state1, delta)
        decoding_time = time.time() - start
        
        print(f"\nEncoding speed: {n_iterations / encoding_time:.0f} states/sec")
        print(f"Decoding speed: {n_iterations / decoding_time:.0f} states/sec")
        
        # Should be very fast
        assert n_iterations / encoding_time > 10000  # >10k encodings/sec
        assert n_iterations / decoding_time > 20000  # >20k decodings/sec
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_encoding_speed(self):
        """Benchmark GPU delta encoding speed"""
        encoder = StateDeltaEncoder(state_shape=(2, 15, 15), device='cuda')
        
        # Create batch of states
        batch_size = 256
        states1 = torch.rand(batch_size, 2, 15, 15, device='cuda')
        states2 = states1.clone()
        
        # Make random changes
        for i in range(batch_size):
            for _ in range(3):
                x, y = np.random.randint(0, 15, size=2)
                states2[i, 0, x, y] = 1 - states2[i, 0, x, y]
                
        # Warmup
        _ = encoder.encode_batch_deltas(states1, states2)
        torch.cuda.synchronize()
        
        # Benchmark
        n_iterations = 100
        start = time.time()
        for _ in range(n_iterations):
            deltas = encoder.encode_batch_deltas(states1, states2)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        states_per_sec = (n_iterations * batch_size) / gpu_time
        print(f"\nGPU encoding speed: {states_per_sec:.0f} states/sec")
        
        # GPU should be much faster for batches
        assert states_per_sec > 500  # Reasonable for batch processing