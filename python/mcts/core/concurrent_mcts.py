"""Concurrent MCTS implementation for maximum throughput

This module implements concurrent wave processing to achieve
80k-200k simulations per second.
"""

import threading
import queue
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging

from .high_performance_mcts import HighPerformanceMCTS as MCTS
from .mcts_config import MCTSConfig
from .node import Node
from .wave_engine import Wave
from .tree_arena import MemoryConfig

logger = logging.getLogger(__name__)


class ConcurrentWaveProcessor:
    """Processes multiple waves concurrently for high throughput"""
    
    def __init__(
        self,
        mcts: MCTS,
        num_workers: int = 4,
        prefetch_waves: int = 2
    ):
        """Initialize concurrent processor
        
        Args:
            mcts: MCTS instance to use
            num_workers: Number of worker threads
            prefetch_waves: Number of waves to prefetch
        """
        self.mcts = mcts
        self.num_workers = num_workers
        self.prefetch_waves = prefetch_waves
        
        # Queues for wave pipeline
        self.selection_queue = queue.Queue(maxsize=prefetch_waves * 2)
        self.expansion_queue = queue.Queue(maxsize=prefetch_waves * 2)
        self.evaluation_queue = queue.Queue(maxsize=prefetch_waves * 2)
        self.backup_queue = queue.Queue(maxsize=prefetch_waves * 2)
        
        # Worker threads
        self.workers = []
        self.running = False
        
        # Statistics with thread safety
        self.stats = {
            'waves_processed': 0,
            'total_simulations': 0,
            'pipeline_stalls': 0,
            'avg_wave_time_ms': 0,
        }
        self.stats_lock = threading.Lock()
        
    def start(self):
        """Start worker threads"""
        self.running = True
        
        # Selection workers
        for i in range(max(1, self.num_workers // 4)):
            t = threading.Thread(
                target=self._selection_worker,
                name=f"selection-{i}"
            )
            t.start()
            self.workers.append(t)
            
        # Expansion workers
        for i in range(max(1, self.num_workers // 4)):
            t = threading.Thread(
                target=self._expansion_worker,
                name=f"expansion-{i}"
            )
            t.start()
            self.workers.append(t)
            
        # Evaluation workers (more for NN bottleneck)
        for i in range(max(2, self.num_workers // 2)):
            t = threading.Thread(
                target=self._evaluation_worker,
                name=f"evaluation-{i}"
            )
            t.start()
            self.workers.append(t)
            
        # Backup worker (single to avoid conflicts)
        t = threading.Thread(
            target=self._backup_worker,
            name="backup-0"
        )
        t.start()
        self.workers.append(t)
        
        logger.info(f"Started {len(self.workers)} worker threads")
        
    def stop(self):
        """Stop worker threads"""
        self.running = False
        
        # Send stop signals - one for each worker
        for _ in range(max(1, self.num_workers // 4)):
            self.selection_queue.put(None)
        for _ in range(max(1, self.num_workers // 4)):
            self.expansion_queue.put(None)
        for _ in range(max(2, self.num_workers // 2)):
            self.evaluation_queue.put(None)
        self.backup_queue.put(None)
            
        # Wait for workers
        for t in self.workers:
            t.join(timeout=1.0)
            if t.is_alive():
                logger.warning(f"Worker {t.name} did not stop cleanly")
            
        self.workers.clear()
        logger.info("Stopped all worker threads")
        
    def process_waves(self, root_id: str, num_waves: int, wave_size: int):
        """Process multiple waves concurrently
        
        Args:
            root_id: Root node ID
            num_waves: Number of waves to process
            wave_size: Size of each wave
        """
        # Start pipeline with initial waves
        for _ in range(min(num_waves, self.prefetch_waves)):
            wave = self.mcts.wave_engine.create_wave(root_id, wave_size)
            self.selection_queue.put((wave, time.time()))
            
        waves_submitted = self.prefetch_waves
        waves_completed = 0
        
        # Submit all waves
        while waves_submitted < num_waves:
            wave = self.mcts.wave_engine.create_wave(root_id, wave_size)
            try:
                self.selection_queue.put((wave, time.time()), timeout=1.0)
                waves_submitted += 1
            except queue.Full:
                logger.warning("Selection queue full, skipping wave")
            
        # Wait for completion
        start_wait = time.time()
        while True:
            with self.stats_lock:
                processed = self.stats['waves_processed']
            
            if processed >= num_waves:
                break
                
            if time.time() - start_wait > num_waves * 0.5:  # Timeout after 0.5s per wave
                logger.warning(f"Timeout waiting for waves: {processed}/{num_waves} completed")
                break
            time.sleep(0.01)
                
    def _selection_worker(self):
        """Worker for selection phase"""
        while self.running:
            try:
                item = self.selection_queue.get(timeout=0.1)
                if item is None:
                    break
                    
                wave, start_time = item
                
                # Run selection
                self.mcts.wave_engine._run_selection_phase(wave)
                
                # Pass to expansion
                self.expansion_queue.put((wave, start_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Selection worker error: {e}")
                
    def _expansion_worker(self):
        """Worker for expansion phase"""
        while self.running:
            try:
                item = self.expansion_queue.get(timeout=0.1)
                if item is None:
                    break
                    
                wave, start_time = item
                
                # Run expansion
                self.mcts.wave_engine._run_expansion_phase(wave)
                
                # Pass to evaluation
                self.evaluation_queue.put((wave, start_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Expansion worker error: {e}")
                
    def _evaluation_worker(self):
        """Worker for evaluation phase"""
        while self.running:
            try:
                item = self.evaluation_queue.get(timeout=0.1)
                if item is None:
                    break
                    
                wave, start_time = item
                
                # Run evaluation
                values = self.mcts.wave_engine._run_evaluation_phase(wave)
                
                # Store values in wave
                wave.values = values
                
                # Pass to backup
                self.backup_queue.put((wave, start_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Evaluation worker error: {e}")
                
    def _backup_worker(self):
        """Worker for backup phase"""
        while self.running:
            try:
                item = self.backup_queue.get(timeout=0.1)
                if item is None:
                    break
                    
                wave, start_time = item
                
                # Run backup
                self.mcts.wave_engine._run_backup_phase(wave, wave.values)
                
                # Update statistics with thread safety
                with self.stats_lock:
                    self.stats['waves_processed'] += 1
                    self.stats['total_simulations'] += wave.size
                    
                    # Update timing
                    wave_time = (time.time() - start_time) * 1000
                    self.stats['avg_wave_time_ms'] = (
                        0.9 * self.stats['avg_wave_time_ms'] + 0.1 * wave_time
                    )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Backup worker error: {e}")
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self.stats_lock:
            stats = dict(self.stats)
        
        # Add throughput
        if stats['avg_wave_time_ms'] > 0:
            waves_per_second = 1000.0 / stats['avg_wave_time_ms']
            stats['simulations_per_second'] = (
                waves_per_second * self.mcts.wave_engine.config.initial_wave_size
            )
        else:
            stats['simulations_per_second'] = 0
            
        return stats


class ConcurrentMCTS(MCTS):
    """MCTS with concurrent wave processing"""
    
    def __init__(
        self,
        game,
        evaluator,
        config: MCTSConfig,
        memory_config: Optional[MemoryConfig] = None,
        num_workers: int = 4
    ):
        """Initialize concurrent MCTS
        
        Args:
            game: Game interface
            evaluator: Neural network evaluator
            config: MCTS configuration
            memory_config: Memory configuration
            num_workers: Number of worker threads
        """
        super().__init__(game, evaluator, config, memory_config)
        
        self.concurrent_processor = ConcurrentWaveProcessor(
            self, num_workers=num_workers
        )
        
    def search(self, state: Any, parent_action: Optional[int] = None) -> Node:
        """Run concurrent MCTS search
        
        Args:
            state: Current game state
            parent_action: Action that led to this state
            
        Returns:
            Root node after search
        """
        start_time = time.time()
        
        # Get or create root
        root = self._get_or_create_root(state, parent_action)
        root_id = self.arena.add_node(root)
        
        # Expand root if needed
        if not root.is_expanded and not root.is_terminal:
            legal_moves = self.game.get_legal_moves(state)
            if legal_moves:
                # Simple uniform prior for root
                action_probs = {move: 1.0/len(legal_moves) for move in legal_moves}
                child_states = {
                    move: self.game.apply_move(state, move)
                    for move in legal_moves
                }
                root.expand(action_probs, child_states)
                
        self.current_root = root
        self.current_root_id = root_id
        
        # Add noise at root if configured
        if self.config.add_noise_at_root and root.is_expanded:
            self._add_dirichlet_noise(root)
            
        # Start concurrent processor
        self.concurrent_processor.start()
        
        try:
            # Calculate number of waves
            wave_size = self.wave_engine.config.initial_wave_size
            num_waves = self.config.num_simulations // wave_size
            
            # Process waves concurrently
            self.concurrent_processor.process_waves(root_id, num_waves, wave_size)
            
        finally:
            # Stop processor
            self.concurrent_processor.stop()
            
        # Update statistics
        elapsed = time.time() - start_time
        self.stats['search_time'] += elapsed
        self.stats['searches_performed'] += 1
        
        # Get concurrent stats
        concurrent_stats = self.concurrent_processor.get_statistics()
        
        logger.info(
            f"Concurrent search complete: {root.visit_count} simulations in {elapsed:.3f}s "
            f"({concurrent_stats.get('simulations_per_second', 0):.0f} sims/s)"
        )
        
        return root
        
    def get_statistics(self) -> Dict[str, float]:
        """Get search statistics including concurrent processing"""
        stats = super().get_statistics()
        
        # Add concurrent statistics
        concurrent_stats = self.concurrent_processor.get_statistics()
        stats.update({
            f'concurrent_{k}': v for k, v in concurrent_stats.items()
        })
        
        return stats