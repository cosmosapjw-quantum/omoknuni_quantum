"""Example of integrating adaptive parameter tuning into MCTS training

This script demonstrates how to use the adaptive parameter tuning system
to automatically balance CPU-GPU utilization across different simulation loads.
"""

import time
import logging
from typing import Optional

from .adaptive_parameter_tuner import get_global_parameter_tuner, cleanup_global_parameter_tuner
from .batch_evaluation_coordinator import get_global_batching_coordinator, cleanup_global_coordinator
from .gpu_evaluator_service import GPUEvaluatorService

logger = logging.getLogger(__name__)


class AdaptiveTrainingManager:
    """Manager that integrates adaptive parameter tuning into training"""
    
    def __init__(self, model, device='cuda'):
        """Initialize adaptive training manager
        
        Args:
            model: Neural network model
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.gpu_service = None
        self.parameter_tuner = None
        self.current_simulation_count = 100
        
    def start_adaptive_training(self):
        """Start adaptive training with parameter tuning"""
        logger.info("Starting adaptive training with intelligent parameter tuning")
        
        # Initialize GPU service with adaptive parameters
        self.gpu_service = GPUEvaluatorService(
            model=self.model,
            device=self.device,
            auto_optimize=True,
            workload_type="balanced"
        )
        self.gpu_service.start()
        
        # Get the global parameter tuner (automatically starts monitoring)
        self.parameter_tuner = get_global_parameter_tuner()
        
        logger.info("Adaptive training system initialized")
    
    def update_simulation_count(self, new_simulation_count: int):
        """Update simulation count and trigger parameter optimization
        
        Args:
            new_simulation_count: New simulation count per move
        """
        if new_simulation_count == self.current_simulation_count:
            return
        
        old_count = self.current_simulation_count
        self.current_simulation_count = new_simulation_count
        
        logger.info(f"Simulation count changed: {old_count} -> {new_simulation_count}")
        
        # Update components with new simulation count
        if self.gpu_service:
            self.gpu_service.update_simulation_count(new_simulation_count)
        
        # Update batch coordinator
        coordinator = get_global_batching_coordinator()
        coordinator.update_simulation_count(new_simulation_count)
        
        # Force immediate parameter optimization
        if self.parameter_tuner:
            self.parameter_tuner.force_parameter_update(new_simulation_count)
    
    def get_performance_report(self) -> dict:
        """Get current performance report from adaptive tuner"""
        if self.parameter_tuner:
            return self.parameter_tuner.get_performance_report()
        return {"error": "Parameter tuner not initialized"}
    
    def log_performance_summary(self):
        """Log a performance summary"""
        report = self.get_performance_report()
        
        if "error" in report:
            logger.warning(f"Performance report error: {report['error']}")
            return
        
        logger.info(f"Adaptive Training Performance Summary:")
        logger.info(f"  Current simulation count: {report.get('current_simulation_count', 'Unknown')}")
        logger.info(f"  Performance score: {report.get('performance_score', 0.0):.3f}")
        logger.info(f"  Adjustments made: {report.get('adjustment_count', 0)}")
        
        # Log profile-specific metrics
        for profile_name, metrics in report.get('profiles', {}).items():
            if metrics['sample_count'] > 0:
                logger.info(f"  {profile_name.title()} simulation profile:")
                logger.info(f"    CPU utilization: {metrics['avg_cpu_utilization']:.1%}")
                logger.info(f"    GPU utilization: {metrics['avg_gpu_utilization']:.1%}")
                logger.info(f"    Simulations/sec: {metrics['avg_simulations_per_second']:.0f}")
    
    def cleanup(self):
        """Clean up adaptive training resources"""
        logger.info("Cleaning up adaptive training system")
        
        if self.gpu_service:
            self.gpu_service.stop()
            self.gpu_service = None
        
        # Clean up global components
        cleanup_global_parameter_tuner()
        cleanup_global_coordinator()
        
        logger.info("Adaptive training cleanup completed")


def example_training_loop():
    """Example of how to use adaptive training in a training loop"""
    import torch
    
    # Create a dummy model for demonstration
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(18, 256, 3, padding=1)
            self.fc_policy = torch.nn.Linear(256, 225)
            self.fc_value = torch.nn.Linear(256, 1)
        
        def forward(self, x):
            x = self.conv(x).mean(dim=(2, 3))
            policy = self.fc_policy(x)
            value = self.fc_value(x)
            return policy, value
    
    model = DummyModel()
    
    # Initialize adaptive training manager
    training_manager = AdaptiveTrainingManager(model, device='cuda')
    training_manager.start_adaptive_training()
    
    try:
        # Simulate training with different simulation counts
        simulation_counts = [100, 300, 500, 800, 200]
        
        for iteration, sim_count in enumerate(simulation_counts):
            logger.info(f"\\nIteration {iteration + 1}: Using {sim_count} simulations per move")
            
            # Update simulation count (triggers adaptive parameter adjustment)
            training_manager.update_simulation_count(sim_count)
            
            # Simulate training work (replace with actual training code)
            time.sleep(5)  # Simulate 5 seconds of training
            
            # Log performance summary
            training_manager.log_performance_summary()
        
        # Final performance report
        logger.info("\\nFinal adaptive training performance report:")
        final_report = training_manager.get_performance_report()
        logger.info(f"Total adjustments: {final_report.get('adjustment_count', 0)}")
        logger.info(f"Final performance score: {final_report.get('performance_score', 0.0):.3f}")
        
    finally:
        # Always clean up
        training_manager.cleanup()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    example_training_loop()