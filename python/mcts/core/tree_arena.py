"""TreeArena: Resource-aware memory management for MCTS trees

This module provides efficient storage and retrieval of MCTS nodes with
automatic CPU/GPU memory paging, mixed precision support, and garbage collection.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import OrderedDict, deque
import numpy as np
import time
import pickle
import warnings
import threading

# Lazy torch import to avoid double import issues
torch = None
HAS_TORCH = None

def _get_torch():
    global torch, HAS_TORCH
    if HAS_TORCH is None:
        try:
            import torch as _torch
            torch = _torch
            HAS_TORCH = True
        except ImportError:
            torch = None
            HAS_TORCH = False
    return torch, HAS_TORCH

from .node import Node


@dataclass
class MemoryConfig:
    """Configuration for memory management
    
    Attributes:
        gpu_memory_limit: Maximum GPU memory in bytes
        cpu_memory_limit: Maximum CPU memory in bytes
        page_size: Number of nodes per memory page
        node_size_bytes: Estimated size of each node in bytes
        enable_mixed_precision: Whether to use FP16 for high-visit nodes
        fp16_visit_threshold: Visit count threshold for FP16 storage
        gc_threshold: Memory usage fraction that triggers garbage collection
    """
    gpu_memory_limit: int
    cpu_memory_limit: int
    page_size: int = 1024
    node_size_bytes: int = 512  # Estimated
    enable_mixed_precision: bool = True
    fp16_visit_threshold: int = 1000
    gc_threshold: float = 0.9
    
    @classmethod
    def desktop_preset(cls) -> 'MemoryConfig':
        """Desktop configuration: RTX 3060 Ti with 64GB RAM"""
        return cls(
            gpu_memory_limit=8 * 1024**3,  # 8GB
            cpu_memory_limit=32 * 1024**3,  # 32GB available for MCTS
            page_size=1024,
            enable_mixed_precision=True
        )
    
    @classmethod
    def laptop_preset(cls) -> 'MemoryConfig':
        """Laptop configuration: RTX 3050/4050 with 16GB RAM"""
        return cls(
            gpu_memory_limit=4 * 1024**3,  # 4GB
            cpu_memory_limit=8 * 1024**3,  # 8GB available
            page_size=512,
            enable_mixed_precision=True
        )
    
    @classmethod
    def cloud_preset(cls) -> 'MemoryConfig':
        """Cloud configuration: A10 GPU with high memory"""
        return cls(
            gpu_memory_limit=20 * 1024**3,  # 20GB
            cpu_memory_limit=64 * 1024**3,  # 64GB
            page_size=2048,
            enable_mixed_precision=True
        )


class NodePage:
    """A page of nodes stored together in memory"""
    
    def __init__(self, page_id: int, capacity: int):
        self.page_id = page_id
        self.capacity = capacity
        self.nodes: Dict[str, Node] = {}
        self.last_access_time = time.time()
        self.is_on_gpu = False
        self.gpu_tensors: Optional[Dict[str, Any]] = None
        
    def add_node(self, node_id: str, node: Node) -> None:
        """Add a node to this page"""
        self.nodes[node_id] = node
        self.last_access_time = time.time()
        
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node from this page"""
        self.last_access_time = time.time()
        return self.nodes.get(node_id)
        
    def is_full(self) -> bool:
        """Check if page is at capacity"""
        return len(self.nodes) >= self.capacity
        
    def memory_usage(self, node_size_bytes: int) -> int:
        """Estimate memory usage of this page"""
        return len(self.nodes) * node_size_bytes


class TreeArena:
    """Memory arena for efficient MCTS tree storage
    
    Manages automatic paging between GPU and CPU memory with LRU eviction.
    """
    
    def __init__(self, config: MemoryConfig, use_gpu: bool = True, enable_transpositions: bool = True):
        self.config = config
        torch, has_torch = _get_torch()
        self.use_gpu = use_gpu and has_torch and torch.cuda.is_available() if torch else False
        self.enable_transpositions = enable_transpositions
        
        # Node tracking
        self.node_registry: Dict[str, Tuple[int, str]] = {}  # node_id -> (page_id, location)
        self.node_pages: Dict[int, NodePage] = {}
        self.next_page_id = 0
        self.next_node_id = 0
        
        # Transposition table: state_hash -> node_id
        self.transposition_table: Dict[int, str] = {}
        # Track multiple parents for DAG structure
        self.node_parents: Dict[str, List[str]] = {}  # node_id -> [parent_ids]
        
        # Memory tracking
        self.gpu_pages: OrderedDict[int, NodePage] = OrderedDict()  # LRU order
        self.cpu_pages: OrderedDict[int, NodePage] = OrderedDict()
        self.gpu_memory_used = 0
        self.cpu_memory_used = 0
        
        # Statistics
        self.total_nodes = 0
        self.gpu_nodes = 0
        self.cpu_nodes = 0
        self.transposition_hits = 0
        self.transposition_misses = 0
        
        # GC tracking
        self.gc_history: deque = deque(maxlen=100)
        
        # Thread safety for concurrent access
        self.lock = threading.RLock()  # Reentrant lock for nested calls
        
    def add_node(self, node: Node, state_hash: Optional[int] = None) -> str:
        """Add a node to the arena
        
        Args:
            node: The node to add
            state_hash: Optional state hash for transposition table
            
        Returns:
            Unique node ID (may be existing if transposition found)
        """
        with self.lock:
            # Check transposition table if enabled
            if self.enable_transpositions and state_hash is not None:
                if state_hash in self.transposition_table:
                    # Found transposition - return existing node
                    existing_id = self.transposition_table[state_hash]
                    self.transposition_hits += 1
                    
                    # Track additional parent if node has parent
                    if node.parent is not None:
                        parent_id = self._get_node_id(node.parent)
                        if parent_id and existing_id in self.node_parents:
                            if parent_id not in self.node_parents[existing_id]:
                                self.node_parents[existing_id].append(parent_id)
                    
                    return existing_id
                else:
                    self.transposition_misses += 1
            
            # Create new node
            node_id = f"node_{self.next_node_id}"
            self.next_node_id += 1
            
            # Find or create a page with space
            page = self._find_available_page() or self._create_new_page()
            
            # Add node to page
            page.add_node(node_id, node)
            self.node_registry[node_id] = (page.page_id, 'gpu' if page.is_on_gpu else 'cpu')
            
            # Update transposition table
            if self.enable_transpositions and state_hash is not None:
                self.transposition_table[state_hash] = node_id
            
            # Track parent relationship
            if node.parent is not None:
                parent_id = self._get_node_id(node.parent)
                if parent_id:
                    self.node_parents[node_id] = [parent_id]
        
        # Update memory tracking
        if page.is_on_gpu:
            self.gpu_memory_used = sum(
                p.memory_usage(self.config.node_size_bytes) for p in self.gpu_pages.values()
            )
        else:
            self.cpu_memory_used = sum(
                p.memory_usage(self.config.node_size_bytes) for p in self.cpu_pages.values()
            )
            
        # Update statistics
        self.total_nodes += 1
        if page.is_on_gpu:
            self.gpu_nodes += 1
        else:
            self.cpu_nodes += 1
            
        # Check memory pressure
        self._check_memory_pressure()
        
        return node_id
        
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a single node by ID
        
        Args:
            node_id: The node ID
            
        Returns:
            The node or None if not found
        """
        with self.lock:
            if node_id not in self.node_registry:
                return None
            
            page_id, location = self.node_registry[node_id]
            page = self.node_pages[page_id]
            
            # Update LRU order for GPU pages
            if page.is_on_gpu and page.page_id in self.gpu_pages:
                # Move to end of OrderedDict (most recently used)
                self.gpu_pages.move_to_end(page.page_id)
                
            # Move to GPU if needed and possible
            if location == 'cpu' and self.use_gpu:
                self._maybe_move_page_to_gpu(page)
                
            return page.get_node(node_id)
        
    def get_nodes_batch(self, node_ids: List[str]) -> List[Node]:
        """Get multiple nodes efficiently
        
        Args:
            node_ids: List of node IDs
            
        Returns:
            List of nodes (None for missing nodes)
        """
        nodes = []
        pages_to_access = {}
        
        # Group by page
        for node_id in node_ids:
            if node_id in self.node_registry:
                page_id, _ = self.node_registry[node_id]
                if page_id not in pages_to_access:
                    pages_to_access[page_id] = []
                pages_to_access[page_id].append(node_id)
                
        # Access pages
        for page_id, page_node_ids in pages_to_access.items():
            page = self.node_pages[page_id]
            for node_id in page_node_ids:
                nodes.append(page.get_node(node_id))
                
        return nodes
        
    def get_gpu_node_ids(self) -> Set[str]:
        """Get IDs of all nodes currently on GPU"""
        gpu_ids = set()
        for node_id, (page_id, location) in self.node_registry.items():
            if location == 'gpu':
                gpu_ids.add(node_id)
        return gpu_ids
        
    def get_storage_info(self, node_id: str) -> Dict[str, Any]:
        """Get storage information for a node"""
        if node_id not in self.node_registry:
            return {}
            
        page_id, location = self.node_registry[node_id]
        page = self.node_pages[page_id]
        node = page.get_node(node_id)
        
        # Determine precision
        precision = 'fp32'
        if (self.config.enable_mixed_precision and 
            node.visit_count >= self.config.fp16_visit_threshold):
            precision = 'fp16'
            
        return {
            'location': location,
            'page_id': page_id,
            'precision': precision,
            'visit_count': node.visit_count
        }
        
    def get_gpu_tensors(self) -> Dict[str, Any]:
        """Get GPU tensors for all nodes on GPU"""
        torch, has_torch = _get_torch()
        if not self.use_gpu or not has_torch:
            return {}
            
        # Collect data from GPU pages
        states = []
        priors = []
        values = []
        visit_counts = []
        
        for page in self.gpu_pages.values():
            for node in page.nodes.values():
                # Convert state to tensor if needed
                if isinstance(node.state, np.ndarray):
                    states.append(torch.from_numpy(node.state))
                elif node.state is not None:
                    states.append(torch.tensor(node.state))
                    
                priors.append(node.prior)
                values.append(node.value())
                visit_counts.append(node.visit_count)
                
        # Stack into tensors
        if states:
            return {
                'states': torch.stack(states).cuda(),
                'priors': torch.tensor(priors).cuda(),
                'values': torch.tensor(values).cuda(),
                'visit_counts': torch.tensor(visit_counts).cuda()
            }
        return {}
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get tree statistics"""
        # Calculate tree depth and branching factor
        depths = []
        branching_factors = []
        
        for page in self.node_pages.values():
            for node in page.nodes.values():
                # Calculate depth
                depth = 0
                current = node
                while current.parent is not None:
                    depth += 1
                    current = current.parent
                depths.append(depth)
                
                # Count children
                if hasattr(node, 'children') and node.children:
                    branching_factors.append(len(node.children))
                    
        return {
            'total_nodes': self.total_nodes,
            'gpu_nodes': self.gpu_nodes,
            'cpu_nodes': self.cpu_nodes,
            'depth': max(depths) if depths else 0,
            'branching_factor': np.mean(branching_factors) if branching_factors else 0,
            'gpu_memory_used': self.gpu_memory_used,
            'cpu_memory_used': self.cpu_memory_used,
            'gpu_pages': len(self.gpu_pages),
            'cpu_pages': len(self.cpu_pages),
            'gc_count': len(self.gc_history)
        }
        
    def save(self, path: str) -> None:
        """Save tree state to disk"""
        state = {
            'config': self.config,
            'node_registry': self.node_registry,
            'total_nodes': self.total_nodes,
            'nodes': {}
        }
        
        # Collect all nodes
        for page in self.node_pages.values():
            for node_id, node in page.nodes.items():
                state['nodes'][node_id] = {
                    'state': node.state,
                    'parent': node.parent,
                    'action': node.action,
                    'prior': node.prior,
                    'visit_count': node.visit_count,
                    'value_sum': node.value_sum
                }
                
        # Save as numpy archive
        np.savez_compressed(path, **state)
        
    def load(self, path: str) -> None:
        """Load tree state from disk"""
        data = np.load(path, allow_pickle=True)
        
        # Restore configuration
        self.config = data['config'].item()
        self.node_registry = data['node_registry'].item()
        self.total_nodes = data['total_nodes'].item()
        
        # Recreate nodes
        nodes_data = data['nodes'].item()
        for node_id, node_data in nodes_data.items():
            node = Node(
                state=node_data['state'],
                parent=node_data['parent'],
                action=node_data['action'],
                prior=node_data['prior']
            )
            node.visit_count = node_data['visit_count']
            node.value_sum = node_data['value_sum']
            
            # Add to arena (simplified - just creates new pages)
            page = self._find_available_page() or self._create_new_page()
            page.add_node(node_id, node)
            
    def _find_available_page(self) -> Optional[NodePage]:
        """Find a page with available space"""
        # Check GPU pages first
        for page in self.gpu_pages.values():
            if not page.is_full():
                return page
                
        # Then CPU pages
        for page in self.cpu_pages.values():
            if not page.is_full():
                return page
                
        return None
        
    def _create_new_page(self) -> NodePage:
        """Create a new page"""
        page = NodePage(self.next_page_id, self.config.page_size)
        self.next_page_id += 1
        
        # Check if we need to evict pages first
        page_size = self.config.page_size * self.config.node_size_bytes
        
        # Try to place on GPU first
        if self.use_gpu:
            # Evict pages if necessary to make room
            while self.gpu_memory_used + page_size > self.config.gpu_memory_limit and self.gpu_pages:
                if not self._evict_lru_gpu_page():
                    break
                    
            if self.gpu_memory_used + page_size <= self.config.gpu_memory_limit:
                page.is_on_gpu = True
                self.gpu_pages[page.page_id] = page
                self.gpu_memory_used += page_size
                self.node_pages[page.page_id] = page
                return page
        
        # Place on CPU
        page.is_on_gpu = False
        self.cpu_pages[page.page_id] = page
        self.cpu_memory_used += page_size
        self.node_pages[page.page_id] = page
        return page
        
    def _can_fit_on_gpu(self, page: NodePage) -> bool:
        """Check if a page can fit on GPU"""
        if not self.use_gpu:
            return False
            
        page_size = page.memory_usage(self.config.node_size_bytes)
        return self.gpu_memory_used + page_size <= self.config.gpu_memory_limit
        
    def _maybe_move_page_to_gpu(self, page: NodePage) -> None:
        """Try to move a page to GPU"""
        if page.is_on_gpu or not self.use_gpu:
            return
            
        # Check if we need to evict something
        page_size = page.memory_usage(self.config.node_size_bytes)
        while self.gpu_memory_used + page_size > self.config.gpu_memory_limit:
            if not self._evict_lru_gpu_page():
                return  # Can't make space
                
        # Move page
        self._move_page(page, to_gpu=True)
        
    def _evict_lru_gpu_page(self) -> bool:
        """Evict least recently used GPU page"""
        if not self.gpu_pages:
            return False
            
        # Get LRU page
        lru_page_id = next(iter(self.gpu_pages))
        lru_page = self.gpu_pages[lru_page_id]
        
        # Move to CPU
        self._move_page(lru_page, to_gpu=False)
        return True
        
    def _move_page(self, page: NodePage, to_gpu: bool) -> None:
        """Move a page between GPU and CPU"""
        if to_gpu == page.is_on_gpu:
            return
            
        page_size = page.memory_usage(self.config.node_size_bytes)
        
        if to_gpu:
            # CPU -> GPU
            del self.cpu_pages[page.page_id]
            self.gpu_pages[page.page_id] = page
            self.cpu_memory_used -= page_size
            self.gpu_memory_used += page_size
            self.cpu_nodes -= len(page.nodes)
            self.gpu_nodes += len(page.nodes)
            page.is_on_gpu = True
            
            # Update registry
            for node_id in page.nodes:
                self.node_registry[node_id] = (page.page_id, 'gpu')
        else:
            # GPU -> CPU
            del self.gpu_pages[page.page_id]
            self.cpu_pages[page.page_id] = page
            self.gpu_memory_used -= page_size
            self.cpu_memory_used += page_size
            self.gpu_nodes -= len(page.nodes)
            self.cpu_nodes += len(page.nodes)
            page.is_on_gpu = False
            
            # Update registry
            for node_id in page.nodes:
                self.node_registry[node_id] = (page.page_id, 'cpu')
                
    def _check_memory_pressure(self) -> None:
        """Check if we need to run garbage collection"""
        total_memory = self.gpu_memory_used + self.cpu_memory_used
        total_limit = self.config.gpu_memory_limit + self.config.cpu_memory_limit
        
        if total_memory > total_limit * self.config.gc_threshold:
            self._run_garbage_collection()
            
    def _run_garbage_collection(self) -> None:
        """Run garbage collection to free memory"""
        gc_start = time.time()
        initial_nodes = self.total_nodes
        
        # Find nodes with low importance (low visit count, not recently accessed)
        candidates = []
        for page in list(self.node_pages.values()):
            for node_id, node in list(page.nodes.items()):
                importance = node.visit_count * (1.0 / (time.time() - page.last_access_time + 1))
                candidates.append((importance, node_id, page))
                
        # Sort by importance and remove bottom 10%
        candidates.sort(key=lambda x: x[0])
        to_remove = max(1, int(len(candidates) * 0.1))  # Remove at least 1
        
        for _, node_id, page in candidates[:to_remove]:
            # Remove from transposition table if present
            if self.enable_transpositions:
                # Find and remove hash entry
                for hash_val, nid in list(self.transposition_table.items()):
                    if nid == node_id:
                        del self.transposition_table[hash_val]
                        break
                # Remove parent tracking
                if node_id in self.node_parents:
                    del self.node_parents[node_id]
            
            del page.nodes[node_id]
            del self.node_registry[node_id]
            self.total_nodes -= 1
            if page.is_on_gpu:
                self.gpu_nodes -= 1
            else:
                self.cpu_nodes -= 1
                
        # Remove empty pages
        for page_id in list(self.node_pages.keys()):
            page = self.node_pages[page_id]
            if len(page.nodes) == 0:
                del self.node_pages[page_id]
                if page.is_on_gpu:
                    del self.gpu_pages[page_id]
                else:
                    del self.cpu_pages[page_id]
                    
        # Record GC event
        self.gc_history.append({
            'time': gc_start,
            'duration': time.time() - gc_start,
            'removed': initial_nodes - self.total_nodes
        })
        
        # Recalculate memory usage
        self._recalculate_memory_usage()
        
    def _recalculate_memory_usage(self) -> None:
        """Recalculate memory usage after GC"""
        self.gpu_memory_used = sum(
            page.memory_usage(self.config.node_size_bytes)
            for page in self.gpu_pages.values()
        )
        self.cpu_memory_used = sum(
            page.memory_usage(self.config.node_size_bytes)
            for page in self.cpu_pages.values()
        )
    
    def _get_node_id(self, node: Node) -> Optional[str]:
        """Get node ID from node object (for parent tracking)"""
        # In a real implementation, nodes would store their ID
        # For now, search through pages
        for page in self.node_pages.values():
            for node_id, n in page.nodes.items():
                if n is node:
                    return node_id
        return None
    
    def get_node_by_hash(self, state_hash: int) -> Optional[Node]:
        """Get node by state hash"""
        with self.lock:
            if state_hash in self.transposition_table:
                node_id = self.transposition_table[state_hash]
                return self.get_node(node_id)
            return None
    
    def get_transposition_stats(self) -> Dict[str, Any]:
        """Get transposition table statistics"""
        total_lookups = self.transposition_hits + self.transposition_misses
        hit_rate = self.transposition_hits / total_lookups if total_lookups > 0 else 0.0
        
        return {
            'enabled': self.enable_transpositions,
            'table_size': len(self.transposition_table),
            'hits': self.transposition_hits,
            'misses': self.transposition_misses,
            'hit_rate': hit_rate,
            'dag_nodes': len([nid for nid, parents in self.node_parents.items() if len(parents) > 1])
        }