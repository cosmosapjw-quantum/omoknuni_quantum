"""Hybrid CPU-GPU execution for MCTS

This module implements efficient hybrid execution to leverage both CPU and GPU
resources, particularly beneficial for systems with powerful CPUs like Ryzen 5900X.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
from collections import deque
from mcts.neural_networks.lightweight_evaluator import create_cpu_evaluator

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """Configuration for hybrid CPU-GPU execution"""
    # CPU configuration
    num_cpu_threads: int = 8  # Good for Ryzen 5900X (12 cores, leave some for OS)
    cpu_wave_size: int = 64  # Smaller waves for CPU
    cpu_batch_size: int = 16  # NN evaluation batch for CPU
    
    # GPU configuration  
    gpu_wave_size: int = 1024  # Large waves for GPU
    gpu_batch_size: int = 512  # NN evaluation batch for GPU
    
    # Work distribution
    gpu_allocation: float = 0.6  # 60% of work to GPU
    cpu_allocation: float = 0.4  # 40% of work to CPU
    
    # Queue management
    work_queue_size: int = 100
    result_queue_size: int = 100
    
    # Performance tuning
    enable_work_stealing: bool = True  # CPU threads can steal work
    enable_dynamic_allocation: bool = True  # Adjust CPU/GPU split based on performance
    profile_interval: int = 100  # Profile every N waves
    
    # Memory management
    cpu_memory_limit_mb: int = 4096  # Limit CPU memory usage
    shared_memory: bool = True  # Use shared memory for CPU-GPU transfer


class CPUWorker:
    """CPU worker for MCTS simulations"""
    
    def __init__(self, worker_id: int, config: HybridConfig, 
                 game_interface: Any, evaluator: Any):
        self.worker_id = worker_id
        self.config = config
        self.game_interface = game_interface
        # Use lightweight evaluator for CPU
        self.evaluator = create_cpu_evaluator('lightweight', device='cpu')
        self.device = torch.device('cpu')
        
        # Create lightweight tree for CPU
        from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig
        tree_config = CSRTreeConfig(
            max_nodes=100000 // config.num_cpu_threads,
            max_edges=(100000 // config.num_cpu_threads) * 5,
            device=str(self.device)
        )
        self.tree = CSRTree(tree_config)
        
        # Statistics
        self.stats = {
            'waves_processed': 0,
            'simulations_completed': 0,
            'evaluation_time': 0.0,
            'selection_time': 0.0
        }
        
    def process_wave(self, root_state: Any, wave_size: int) -> Dict[str, Any]:
        """Process a single wave on CPU
        
        CPU focuses on breadth-first exploration with simpler policies.
        """
        start_time = time.perf_counter()
        
        # Simplified wave processing for CPU
        paths = []
        leaf_states = []
        
        # Selection phase - simpler than GPU version
        selection_start = time.perf_counter()
        for _ in range(wave_size):
            path, leaf_state = self._select_path_cpu(root_state)
            paths.append(path)
            leaf_states.append(leaf_state)
        selection_time = time.perf_counter() - selection_start
        
        # Batch evaluation
        eval_start = time.perf_counter()
        if leaf_states:
            values = self._evaluate_batch_cpu(leaf_states)
        else:
            values = []
        eval_time = time.perf_counter() - eval_start
        
        # Backup values
        backup_start = time.perf_counter()
        for path, value in zip(paths, values):
            self._backup_value_cpu(path, value)
        backup_time = time.perf_counter() - backup_start
        
        # Update stats
        self.stats['waves_processed'] += 1
        self.stats['simulations_completed'] += wave_size
        self.stats['selection_time'] += selection_time
        self.stats['evaluation_time'] += eval_time
        
        total_time = time.perf_counter() - start_time
        
        return {
            'worker_id': self.worker_id,
            'wave_size': wave_size,
            'paths': paths,
            'values': values,
            'timing': {
                'total': total_time,
                'selection': selection_time,
                'evaluation': eval_time,
                'backup': backup_time
            }
        }
    
    def _select_path_cpu(self, root_state: Any) -> Tuple[List[int], Any]:
        """Simple path selection for CPU"""
        path = [0]  # Start at root
        current_state = root_state
        current_idx = 0
        
        # Traverse down using simple UCB
        while True:
            # Get children
            children = self.tree.children[current_idx]
            valid_children = children[children >= 0]
            
            if len(valid_children) == 0:
                # Leaf node - expand if needed
                if self.tree.visit_counts[current_idx] > 0:
                    # Expand node
                    legal_moves = self.game_interface.get_legal_moves(current_state)
                    for i, move in enumerate(legal_moves[:10]):  # Limit expansion for CPU
                        self.tree.add_child(current_idx, move, prior=1.0/len(legal_moves))
                break
            
            # Simple UCB selection
            best_child = self._select_best_child_cpu(current_idx, valid_children)
            
            # Apply move
            action = self.tree.parent_actions[best_child].item()
            current_state = self.game_interface.apply_move(current_state, action)
            
            path.append(best_child)
            current_idx = best_child
            
            # Depth limit for CPU
            if len(path) > 20:
                break
        
        return path, current_state
    
    def _select_best_child_cpu(self, parent_idx: int, children: torch.Tensor) -> int:
        """Simple UCB selection for CPU"""
        parent_visits = self.tree.visit_counts[parent_idx].float()
        
        if parent_visits == 0:
            # Random selection for first visit
            return children[torch.randint(len(children), (1,))].item()
        
        # Simple UCB calculation
        child_visits = self.tree.visit_counts[children].float()
        child_values = torch.where(
            child_visits > 0,
            self.tree.value_sums[children] / child_visits,
            torch.zeros_like(child_visits)
        )
        
        # Exploration term
        c_puct = 1.0
        exploration = c_puct * torch.sqrt(parent_visits) / (1 + child_visits)
        
        ucb = child_values + exploration
        best_idx = torch.argmax(ucb)
        
        return children[best_idx].item()
    
    def _evaluate_batch_cpu(self, states: List[Any]) -> List[float]:
        """Lightweight neural network evaluation for CPU"""
        if not states:
            return []
        
        # Convert states to numpy arrays
        state_arrays = []
        for state in states:
            # Convert game state to numpy array
            if hasattr(self.game_interface, 'state_to_numpy'):
                state_array = self.game_interface.state_to_numpy(state)
            else:
                # Fallback - create dummy array
                state_array = np.zeros((20, 15, 15), dtype=np.float32)
            state_arrays.append(state_array)
        
        # Batch evaluation
        batch = np.array(state_arrays)
        policies, values = self.evaluator.evaluate_batch(batch)
        
        return values.tolist()
    
    def _backup_value_cpu(self, path: List[int], value: float):
        """Backup value along path"""
        current_value = value
        
        for node_idx in reversed(path):
            self.tree.visit_counts[node_idx] += 1
            self.tree.value_sums[node_idx] += current_value
            current_value = -current_value  # Flip for two-player games


class HybridExecutor:
    """Orchestrates hybrid CPU-GPU execution"""
    
    def __init__(self, config: HybridConfig, gpu_wave_engine: Any,
                 game_interface: Any, evaluator: Any):
        self.config = config
        self.gpu_wave_engine = gpu_wave_engine
        self.game_interface = game_interface
        self.evaluator = evaluator
        
        # Work queues
        self.cpu_work_queue = queue.Queue(maxsize=config.work_queue_size)
        self.gpu_work_queue = queue.Queue(maxsize=config.work_queue_size)
        self.result_queue = queue.Queue(maxsize=config.result_queue_size)
        
        # CPU workers
        self.cpu_workers = []
        self.cpu_executor = ThreadPoolExecutor(max_workers=config.num_cpu_threads)
        
        # Initialize CPU workers
        for i in range(config.num_cpu_threads):
            worker = CPUWorker(i, config, game_interface, evaluator)
            self.cpu_workers.append(worker)
        
        # Performance tracking
        self.performance_stats = {
            'cpu_waves': 0,
            'gpu_waves': 0,
            'cpu_time': 0.0,
            'gpu_time': 0.0,
            'total_simulations': 0
        }
        
        # Dynamic allocation state
        self.allocation_history = deque(maxlen=100)
        self.current_gpu_allocation = config.gpu_allocation
        
        # Synchronization
        self.stop_event = threading.Event()
        
    def run_hybrid_search(self, root_state: Any, num_simulations: int) -> Dict[str, Any]:
        """Run hybrid CPU-GPU search
        
        Distributes work between CPU and GPU for optimal performance.
        """
        start_time = time.perf_counter()
        
        # Calculate work distribution
        gpu_sims = int(num_simulations * self.current_gpu_allocation)
        cpu_sims = num_simulations - gpu_sims
        
        gpu_waves = gpu_sims // self.config.gpu_wave_size
        cpu_waves = cpu_sims // self.config.cpu_wave_size
        
        logger.info(f"Hybrid execution: {gpu_waves} GPU waves, {cpu_waves} CPU waves")
        
        # Start CPU workers
        cpu_futures = []
        waves_per_worker = cpu_waves // self.config.num_cpu_threads
        
        for i, worker in enumerate(self.cpu_workers):
            # Distribute remaining waves to first workers
            worker_waves = waves_per_worker
            if i < cpu_waves % self.config.num_cpu_threads:
                worker_waves += 1
            
            if worker_waves > 0:
                future = self.cpu_executor.submit(
                    self._run_cpu_worker, worker, root_state, worker_waves
                )
                cpu_futures.append(future)
        
        # Run GPU waves concurrently
        gpu_future = self.cpu_executor.submit(
            self._run_gpu_waves, root_state, gpu_waves
        )
        
        # Collect results
        cpu_results = []
        gpu_result = None
        
        # Wait for CPU results
        for future in as_completed(cpu_futures):
            try:
                result = future.result()
                cpu_results.append(result)
            except Exception as e:
                logger.error(f"CPU worker failed: {e}")
        
        # Wait for GPU result
        try:
            gpu_result = gpu_future.result()
        except Exception as e:
            logger.error(f"GPU execution failed: {e}")
        
        # Merge results
        total_time = time.perf_counter() - start_time
        
        # Update performance stats
        self._update_performance_stats(cpu_results, gpu_result)
        
        # Adjust allocation if enabled
        if self.config.enable_dynamic_allocation:
            self._adjust_allocation()
        
        return {
            'total_time': total_time,
            'cpu_results': cpu_results,
            'gpu_result': gpu_result,
            'performance_stats': dict(self.performance_stats),
            'current_allocation': {
                'gpu': self.current_gpu_allocation,
                'cpu': 1.0 - self.current_gpu_allocation
            }
        }
    
    def _run_cpu_worker(self, worker: CPUWorker, root_state: Any, 
                       num_waves: int) -> Dict[str, Any]:
        """Run waves on a CPU worker"""
        results = []
        
        for _ in range(num_waves):
            wave_result = worker.process_wave(root_state, self.config.cpu_wave_size)
            results.append(wave_result)
        
        return {
            'worker_id': worker.worker_id,
            'num_waves': num_waves,
            'total_simulations': num_waves * self.config.cpu_wave_size,
            'results': results,
            'stats': dict(worker.stats)
        }
    
    def _run_gpu_waves(self, root_state: Any, num_waves: int) -> Dict[str, Any]:
        """Run waves on GPU"""
        results = []
        start_time = time.perf_counter()
        
        for _ in range(num_waves):
            wave_result = self.gpu_wave_engine.run_wave(
                root_state, self.config.gpu_wave_size
            )
            results.append(wave_result)
        
        total_time = time.perf_counter() - start_time
        
        return {
            'device': 'gpu',
            'num_waves': num_waves,
            'total_simulations': num_waves * self.config.gpu_wave_size,
            'total_time': total_time,
            'results': results
        }
    
    def _update_performance_stats(self, cpu_results: List[Dict], 
                                 gpu_result: Optional[Dict]):
        """Update performance statistics"""
        # CPU stats
        total_cpu_time = 0.0
        total_cpu_sims = 0
        
        for result in cpu_results:
            total_cpu_sims += result['total_simulations']
            for wave_result in result['results']:
                total_cpu_time += wave_result['timing']['total']
        
        # GPU stats
        if gpu_result:
            self.performance_stats['gpu_waves'] += gpu_result['num_waves']
            self.performance_stats['gpu_time'] += gpu_result['total_time']
            self.performance_stats['total_simulations'] += gpu_result['total_simulations']
        
        self.performance_stats['cpu_waves'] += sum(r['num_waves'] for r in cpu_results)
        self.performance_stats['cpu_time'] += total_cpu_time
        self.performance_stats['total_simulations'] += total_cpu_sims
    
    def _adjust_allocation(self):
        """Dynamically adjust CPU/GPU allocation based on performance"""
        if self.performance_stats['cpu_waves'] == 0 or self.performance_stats['gpu_waves'] == 0:
            return
        
        # Calculate throughput
        cpu_throughput = (self.performance_stats['cpu_waves'] * self.config.cpu_wave_size / 
                         self.performance_stats['cpu_time'])
        gpu_throughput = (self.performance_stats['gpu_waves'] * self.config.gpu_wave_size / 
                         self.performance_stats['gpu_time'])
        
        # Store in history
        self.allocation_history.append({
            'cpu_throughput': cpu_throughput,
            'gpu_throughput': gpu_throughput,
            'gpu_allocation': self.current_gpu_allocation
        })
        
        # Adjust allocation based on relative throughput
        total_throughput = cpu_throughput + gpu_throughput
        optimal_gpu_allocation = gpu_throughput / total_throughput
        
        # Smooth adjustment
        adjustment_rate = 0.1
        self.current_gpu_allocation = (
            (1 - adjustment_rate) * self.current_gpu_allocation +
            adjustment_rate * optimal_gpu_allocation
        )
        
        # Clamp to reasonable range
        self.current_gpu_allocation = max(0.3, min(0.8, self.current_gpu_allocation))
        
        logger.debug(f"Adjusted GPU allocation to {self.current_gpu_allocation:.2f}")
    
    def merge_trees(self, gpu_tree: Any, cpu_trees: List[Any]) -> None:
        """Merge CPU trees back into GPU tree
        
        This allows CPU discoveries to benefit GPU search.
        """
        # For each CPU tree, transfer high-value nodes to GPU tree
        for cpu_tree in cpu_trees:
            # Find nodes with high visit counts or values
            high_value_nodes = torch.where(
                cpu_tree.visit_counts > 10  # Threshold
            )[0]
            
            # Transfer these nodes to GPU tree
            # This is a simplified version - actual implementation would
            # need proper tree merging logic
            pass
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        if self.performance_stats['total_simulations'] == 0:
            return {}
        
        total_time = self.performance_stats['cpu_time'] + self.performance_stats['gpu_time']
        
        return {
            'total_simulations': self.performance_stats['total_simulations'],
            'total_time': total_time,
            'simulations_per_second': self.performance_stats['total_simulations'] / total_time,
            'cpu_statistics': {
                'waves': self.performance_stats['cpu_waves'],
                'time': self.performance_stats['cpu_time'],
                'throughput': (self.performance_stats['cpu_waves'] * self.config.cpu_wave_size / 
                              self.performance_stats['cpu_time']) if self.performance_stats['cpu_time'] > 0 else 0
            },
            'gpu_statistics': {
                'waves': self.performance_stats['gpu_waves'],
                'time': self.performance_stats['gpu_time'],
                'throughput': (self.performance_stats['gpu_waves'] * self.config.gpu_wave_size / 
                              self.performance_stats['gpu_time']) if self.performance_stats['gpu_time'] > 0 else 0
            },
            'allocation': {
                'current_gpu': self.current_gpu_allocation,
                'current_cpu': 1.0 - self.current_gpu_allocation
            }
        }