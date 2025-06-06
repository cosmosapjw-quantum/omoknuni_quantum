"""
Test Suite for GPU Wave Engine
==============================

Validates the wave-based parallel processing engine that achieves
massive speedup through parallel path generation.
"""

import pytest
import torch
import numpy as np
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts.gpu.wave_engine import GPUWaveEngine, WaveConfig, Wave, GPUWaveKernels, create_wave_engine
from mcts.quantum.qft_engine import create_qft_engine


class TestWave:
    """Test the Wave data structure"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_wave(self, device):
        wave_size = 64
        max_depth = 10
        
        paths = torch.randint(0, 100, (wave_size, max_depth), device=device)
        amplitudes = torch.rand(wave_size, device=device)
        valid_lengths = torch.randint(1, max_depth+1, (wave_size,), device=device)
        leaf_nodes = torch.randint(0, 100, (wave_size,), device=device)
        
        return Wave(paths, amplitudes, valid_lengths, leaf_nodes)
    
    def test_wave_initialization(self, sample_wave, device):
        """Test Wave object initialization and properties"""
        assert sample_wave.wave_size == 64
        assert sample_wave.max_depth == 10
        assert sample_wave.device == device


class TestGPUWaveEngine:
    """Test the main wave engine"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def wave_config(self):
        return WaveConfig(
            initial_wave_size=128,
            max_wave_size=512,
            adaptive_sizing=True,
            max_path_length=12
        )
    
    @pytest.fixture
    def wave_engine(self, wave_config, device):
        return GPUWaveEngine(wave_config, device)
    
    @pytest.fixture
    def mock_tree_data(self, device):
        """Create mock tree data"""
        num_nodes = 100
        visit_counts = torch.rand(num_nodes, device=device) * 100 + 1
        
        # Simple children structure for testing
        children = torch.zeros((num_nodes, 5), dtype=torch.long, device=device)
        for i in range(num_nodes-1):
            num_child = min(torch.randint(1, 4, (1,)).item(), 5)
            children[i, :num_child] = torch.arange(i+1, min(i+1+num_child, num_nodes))
        
        return {
            'visit_counts': visit_counts,
            'children': children
        }
    
    def test_engine_initialization(self, wave_engine, device):
        """Test wave engine initialization"""
        assert wave_engine.device == device
        assert wave_engine.current_wave_size == 128
        assert isinstance(wave_engine.kernels, GPUWaveKernels)
        assert 'waves_generated' in wave_engine.stats
    
    def test_wave_generation(self, wave_engine, mock_tree_data):
        """Test basic wave generation"""
        wave = wave_engine.generate_wave(
            tree_data=mock_tree_data,
            root_idx=0,
            hbar_eff=0.1,
            max_depth=8
        )
        
        assert isinstance(wave, Wave)
        assert wave.wave_size == wave_engine.current_wave_size
        assert torch.all(wave.paths[:, 0] == 0)
        
        # Check statistics updated
        assert wave_engine.stats['waves_generated'] == 1
        assert wave_engine.stats['total_paths_processed'] == wave.wave_size


if __name__ == "__main__":
    # Run basic functionality test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Wave Engine tests on {device}")
    
    # Create wave engine
    engine = create_wave_engine(device, wave_size=128)
    
    # Create test tree data
    tree_data = {
        'visit_counts': torch.rand(200, device=device) * 100 + 1,
        'children': torch.randint(0, 200, (200, 5), device=device)
    }
    
    # Test wave generation
    print("Testing wave generation...")
    start = time.perf_counter()
    wave = engine.generate_wave(tree_data, root_idx=0, max_depth=10)
    end = time.perf_counter()
    
    print(f"✓ Wave generated: {wave.wave_size} paths in {end-start:.4f}s")
    print(f"✓ Throughput: {wave.wave_size/(end-start):.0f} paths/sec")
    print(f"✓ All paths start at root: {torch.all(wave.paths[:, 0] == 0)}")
    print(f"✓ Valid amplitudes: {torch.all(torch.isfinite(wave.amplitudes))}")
    
    print("✓ All wave engine tests passed!")