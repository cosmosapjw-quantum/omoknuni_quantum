"""Tests for configuration manager with hardware auto-detection"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from mcts.utils.config_manager import ConfigManager, HardwareInfo, OptimizedConfig, get_config_manager


class TestHardwareDetection:
    """Test hardware detection functionality"""
    
    @patch('mcts.utils.config_manager.psutil')
    @patch('mcts.utils.config_manager.multiprocessing')
    @patch('mcts.utils.config_manager.platform')
    def test_cpu_detection(self, mock_platform, mock_mp, mock_psutil):
        """Test CPU and memory detection"""
        # Mock CPU info
        mock_mp.cpu_count.return_value = 8
        mock_psutil.cpu_freq.return_value = Mock(current=3600)
        
        # Mock memory info
        mock_psutil.virtual_memory.return_value = Mock(
            total=16 * 1024**3,  # 16GB
            available=8 * 1024**3  # 8GB
        )
        
        # Mock platform info
        mock_platform.system.return_value = "Linux"
        mock_platform.python_version.return_value = "3.9.0"
        
        # Disable GPU for this test
        with patch('mcts.utils.config_manager.HAS_TORCH', False):
            manager = ConfigManager()
            
        hw = manager.hardware_info
        assert hw.cpu_count == 8
        assert hw.cpu_freq_mhz == 3600
        assert hw.total_memory_gb == pytest.approx(16.0, rel=0.1)
        assert hw.available_memory_gb == pytest.approx(8.0, rel=0.1)
        assert hw.os_name == "Linux"
        assert not hw.has_gpu
        
    @patch('mcts.utils.config_manager.torch')
    @patch('mcts.utils.config_manager.HAS_TORCH', True)
    @patch('mcts.utils.config_manager.psutil')
    @patch('mcts.utils.config_manager.multiprocessing')
    def test_gpu_detection(self, mock_mp, mock_psutil, mock_torch):
        """Test GPU detection"""
        # Mock basic system info
        mock_mp.cpu_count.return_value = 8
        mock_psutil.cpu_freq.return_value = Mock(current=3600)
        mock_psutil.virtual_memory.return_value = Mock(
            total=32 * 1024**3,
            available=16 * 1024**3
        )
        
        # Mock GPU info
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3080"
        mock_torch.cuda.get_device_properties.return_value = Mock(
            total_memory=10 * 1024**3  # 10GB
        )
        mock_torch.cuda.get_device_capability.return_value = (8, 6)
        mock_torch.version.cuda = "11.7"
        
        manager = ConfigManager()
        hw = manager.hardware_info
        
        assert hw.has_gpu
        assert hw.gpu_name == "NVIDIA RTX 3080"
        assert hw.gpu_memory_gb == pytest.approx(10.0, rel=0.1)
        assert hw.gpu_compute_capability == (8, 6)
        assert hw.cuda_version == "11.7"
        

class TestConfigOptimization:
    """Test configuration optimization"""
    
    def test_cpu_only_optimization(self):
        """Test optimization for CPU-only system"""
        hw_info = HardwareInfo(
            cpu_count=4,
            cpu_freq_mhz=2400,
            total_memory_gb=8.0,
            available_memory_gb=4.0,
            has_gpu=False,
            os_name="Windows",
            python_version="3.8.0"
        )
        
        with patch.object(ConfigManager, '_detect_hardware', return_value=hw_info):
            manager = ConfigManager()
            
        cfg = manager.optimized_config
        
        # Should use 75% of CPUs
        assert cfg.num_threads == 3
        
        # No GPU settings
        assert not cfg.use_gpu
        assert cfg.gpu_memory_limit_gb == 0
        
        # Conservative memory settings
        assert cfg.cpu_memory_limit_gb <= 2.0  # 50% of available
        
        # Smaller batch sizes for CPU
        assert cfg.batch_size == 64
        assert cfg.wave_size == 128
        assert cfg.nn_batch_size == 32
        
    def test_gpu_optimization_high_end(self):
        """Test optimization for high-end GPU system"""
        hw_info = HardwareInfo(
            cpu_count=16,
            cpu_freq_mhz=4500,
            total_memory_gb=64.0,
            available_memory_gb=48.0,
            has_gpu=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_memory_gb=24.0,
            gpu_compute_capability=(8, 9),
            cuda_version="12.0"
        )
        
        with patch.object(ConfigManager, '_detect_hardware', return_value=hw_info):
            manager = ConfigManager()
            
        cfg = manager.optimized_config
        
        # Should use GPU
        assert cfg.use_gpu
        assert cfg.gpu_memory_limit_gb == pytest.approx(24.0 * 0.8, rel=0.1)
        
        # Should enable mixed precision for modern GPU
        assert cfg.enable_mixed_precision
        
        # Large batch sizes for powerful GPU
        assert cfg.batch_size == 2048
        assert cfg.wave_size == 4096
        assert cfg.nn_batch_size == 2048
        
        # High simulation count
        assert cfg.num_simulations == 1600
        assert cfg.target_simulations_per_second == 150000
        
    def test_gpu_optimization_mid_range(self):
        """Test optimization for mid-range GPU"""
        hw_info = HardwareInfo(
            cpu_count=8,
            cpu_freq_mhz=3600,
            total_memory_gb=16.0,
            available_memory_gb=10.0,
            has_gpu=True,
            gpu_name="NVIDIA RTX 3060",
            gpu_memory_gb=6.0,
            gpu_compute_capability=(8, 6),
            cuda_version="11.7"
        )
        
        with patch.object(ConfigManager, '_detect_hardware', return_value=hw_info):
            manager = ConfigManager()
            
        cfg = manager.optimized_config
        
        # Should use GPU with moderate settings
        assert cfg.use_gpu
        assert cfg.batch_size == 1024
        assert cfg.wave_size == 2048
        assert cfg.num_simulations == 800
        
    def test_gpu_optimization_low_memory(self):
        """Test optimization for low memory GPU"""
        hw_info = HardwareInfo(
            cpu_count=4,
            cpu_freq_mhz=3000,
            total_memory_gb=8.0,
            available_memory_gb=4.0,
            has_gpu=True,
            gpu_name="NVIDIA GTX 1650",
            gpu_memory_gb=2.0,
            gpu_compute_capability=(7, 5),
            cuda_version="11.0"
        )
        
        with patch.object(ConfigManager, '_detect_hardware', return_value=hw_info):
            manager = ConfigManager()
            
        cfg = manager.optimized_config
        
        # Should still use GPU but with small batches
        assert cfg.use_gpu
        assert cfg.batch_size == 256
        assert cfg.wave_size == 512
        

class TestConfigGeneration:
    """Test configuration dict generation"""
    
    @patch.object(ConfigManager, '_detect_hardware')
    def test_mcts_config_generation(self, mock_detect):
        """Test MCTS config generation"""
        mock_detect.return_value = HardwareInfo(
            cpu_count=8, cpu_freq_mhz=3600,
            total_memory_gb=16, available_memory_gb=8,
            has_gpu=True, gpu_memory_gb=8
        )
        
        manager = ConfigManager()
        mcts_config = manager.get_mcts_config()
        
        assert 'num_simulations' in mcts_config
        assert 'batch_size' in mcts_config
        assert 'num_threads' in mcts_config
        assert 'use_wave_engine' in mcts_config
        assert mcts_config['use_wave_engine'] == True  # Has GPU
        
    @patch.object(ConfigManager, '_detect_hardware')
    def test_arena_config_generation(self, mock_detect):
        """Test arena config generation"""
        mock_detect.return_value = HardwareInfo(
            cpu_count=8, cpu_freq_mhz=3600,
            total_memory_gb=16, available_memory_gb=8,
            has_gpu=True, gpu_memory_gb=8
        )
        
        manager = ConfigManager()
        arena_config = manager.get_arena_config()
        
        assert 'gpu_memory_limit' in arena_config
        assert 'cpu_memory_limit' in arena_config
        assert 'page_size' in arena_config
        assert arena_config['gpu_memory_limit'] > 0
        
    @patch.object(ConfigManager, '_detect_hardware')
    def test_wave_config_generation(self, mock_detect):
        """Test wave config generation"""
        mock_detect.return_value = HardwareInfo(
            cpu_count=8, cpu_freq_mhz=3600,
            total_memory_gb=16, available_memory_gb=8
        )
        
        manager = ConfigManager()
        wave_config = manager.get_wave_config()
        
        assert 'initial_wave_size' in wave_config
        assert 'max_wave_size' in wave_config
        assert 'enable_interference' in wave_config
        assert wave_config['enable_adaptive_sizing'] == True
        
    @patch.object(ConfigManager, '_detect_hardware')
    def test_evaluator_config_generation(self, mock_detect):
        """Test evaluator config generation"""
        mock_detect.return_value = HardwareInfo(
            cpu_count=8, cpu_freq_mhz=3600,
            total_memory_gb=16, available_memory_gb=8,
            has_gpu=False
        )
        
        manager = ConfigManager()
        eval_config = manager.get_evaluator_config()
        
        assert 'batch_size' in eval_config
        assert 'cache_size' in eval_config
        assert 'device' in eval_config
        assert eval_config['device'] == 'cpu'  # No GPU
        

class TestSummaryGeneration:
    """Test summary string generation"""
    
    @patch.object(ConfigManager, '_detect_hardware')
    def test_hardware_summary(self, mock_detect):
        """Test hardware summary generation"""
        hw_info = HardwareInfo(
            cpu_count=8,
            cpu_freq_mhz=3600,
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            has_gpu=True,
            gpu_name="RTX 3080",
            gpu_memory_gb=10.0,
            gpu_compute_capability=(8, 6),
            cuda_version="11.7",
            os_name="Linux",
            python_version="3.9.0"
        )
        mock_detect.return_value = hw_info
        
        manager = ConfigManager()
        summary = manager.get_hardware_summary()
        
        assert "8 cores" in summary
        assert "3600MHz" in summary
        assert "16.0GB total" in summary
        assert "RTX 3080" in summary
        assert "10.0GB" in summary
        assert "8.6" in summary
        
    @patch.object(ConfigManager, '_detect_hardware')
    def test_optimization_summary(self, mock_detect):
        """Test optimization summary generation"""
        mock_detect.return_value = HardwareInfo(
            cpu_count=8, cpu_freq_mhz=3600,
            total_memory_gb=16, available_memory_gb=8,
            has_gpu=True, gpu_memory_gb=8
        )
        
        manager = ConfigManager()
        summary = manager.get_optimization_summary()
        
        assert "Threads:" in summary
        assert "Simulations:" in summary
        assert "Wave Size:" in summary
        assert "GPU: Enabled" in summary
        

class TestSingleton:
    """Test singleton pattern"""
    
    def test_singleton_instance(self):
        """Test that get_config_manager returns same instance"""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2