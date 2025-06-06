"""MinHash interference system for path diversity

This module implements the quantum-inspired interference mechanism using
MinHash for O(n log n) diversity computation. It automatically uses GPU
acceleration when available.
"""

from typing import List, Dict, Tuple, Optional, Set, Union
import numpy as np
from collections import defaultdict
import logging
import torch

# Try to import GPU implementation first
try:
    from .interference_gpu import MinHashInterference as GPUMinHashInterference
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    GPUMinHashInterference = None
    HAS_GPU = False

# Fallback to datasketch if needed
try:
    from datasketch import MinHash, MinHashLSH
    HAS_MINHASH = True
except ImportError:
    MinHash = None
    MinHashLSH = None
    HAS_MINHASH = False


logger = logging.getLogger(__name__)


class MinHashInterference:
    """Unified MinHash interference with GPU acceleration
    
    This class automatically uses GPU implementation when available,
    falling back to CPU implementation otherwise.
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = 'cuda',
        strength: float = 0.15,
        num_perm: int = 128,
        threshold: float = 0.5
    ):
        """Initialize MinHash interference
        
        Args:
            device: Device for computation ('cuda' or 'cpu')
            strength: Interference strength
            num_perm: Number of permutations/hashes
            threshold: Similarity threshold
        """
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.strength = strength
        
        # Use GPU implementation if available
        if HAS_GPU and device.type == 'cuda':
            logger.info("Using GPU-accelerated MinHash interference")
            self.gpu_engine = GPUMinHashInterference(
                device=device,
                strength=strength
            )
            self.use_gpu = True
        else:
            logger.info("Using CPU MinHash interference")
            self.gpu_engine = None
            self.use_gpu = False
            
            if not HAS_MINHASH:
                raise ImportError(
                    "Neither GPU support nor datasketch package available. "
                    "Install datasketch or enable CUDA."
                )
            
            # Initialize CPU engine
            self.num_perm = num_perm
            self.threshold = threshold
            self.cpu_engine = InterferenceEngine(
                num_perm=num_perm,
                threshold=threshold
            )
    
    def compute_path_diversity_batch(
        self,
        paths: torch.Tensor,
        num_hashes: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute path diversity for batch
        
        Args:
            paths: Path tensor (batch_size, max_depth)
            num_hashes: Number of hash functions
            
        Returns:
            Signatures and similarity matrix
        """
        if self.use_gpu:
            return self.gpu_engine.compute_path_diversity_batch(paths, num_hashes)
        else:
            # Convert to CPU and use datasketch
            paths_cpu = paths.cpu().numpy()
            paths_list = []
            
            for i in range(paths_cpu.shape[0]):
                path = paths_cpu[i]
                valid_path = path[path >= 0].tolist()
                paths_list.append(valid_path)
            
            # Use CPU engine
            interference = self.cpu_engine.compute_interference(paths_list)
            
            # Create dummy signatures and similarities for compatibility
            batch_size = len(paths_list)
            signatures = torch.zeros(batch_size, num_hashes, device=self.device)
            
            # Convert interference to similarity matrix
            similarities = torch.eye(batch_size, device=self.device)
            for i in range(batch_size):
                similarities[i, :] += torch.tensor(
                    interference[i], device=self.device
                )
            
            return signatures, similarities
    
    def apply_interference(
        self,
        scores: torch.Tensor,
        similarities: torch.Tensor,
        interference_strength: Optional[float] = None
    ) -> torch.Tensor:
        """Apply interference to scores
        
        Args:
            scores: Original scores
            similarities: Similarity matrix
            interference_strength: Strength of interference
            
        Returns:
            Modified scores
        """
        if self.use_gpu:
            return self.gpu_engine.apply_interference(
                scores, similarities, interference_strength
            )
        else:
            # Simple CPU implementation
            if interference_strength is None:
                interference_strength = self.strength
                
            # Apply interference (ensure device consistency and dtype)
            eye_matrix = torch.eye(similarities.shape[0], device=similarities.device, dtype=similarities.dtype)
            # Ensure scores has same dtype as similarities
            scores_typed = scores.to(dtype=similarities.dtype)
            interference = torch.matmul(similarities - eye_matrix, scores_typed)
            
            modified_scores = scores_typed - interference_strength * interference
            return torch.clamp(modified_scores, min=0)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics"""
        if self.use_gpu:
            return self.gpu_engine.get_statistics()
        else:
            return self.cpu_engine.get_statistics()


class InterferenceEngine:
    """Engine for computing path interference using MinHash
    
    This implements the quantum-inspired interference mechanism that
    reduces redundant exploration by detecting similar paths efficiently.
    """
    
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.5,
        num_bands: int = 4
    ):
        """Initialize interference engine
        
        Args:
            num_perm: Number of permutations for MinHash
            threshold: Jaccard similarity threshold for interference
            num_bands: Number of bands for LSH (affects precision/recall tradeoff)
        """
        if not HAS_MINHASH:
            raise ImportError("datasketch package required for interference")
            
        self.num_perm = num_perm
        self.threshold = threshold
        self.num_bands = num_bands
        
        # LSH for fast similarity search
        self.lsh = MinHashLSH(
            threshold=threshold,
            num_perm=num_perm,
            params=(num_bands, num_perm // num_bands)
        )
        
        # Cache for MinHash signatures
        self.signature_cache: Dict[str, MinHash] = {}
        
        # Statistics
        self.stats = {
            'paths_processed': 0,
            'interference_events': 0,
            'average_similarity': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def compute_path_signature(self, path: List[int]) -> MinHash:
        """Compute MinHash signature for a path
        
        Args:
            path: List of actions representing the path
            
        Returns:
            MinHash signature
        """
        # Convert path to string key for caching
        path_key = ",".join(map(str, path))
        
        # Check cache
        if path_key in self.signature_cache:
            self.stats['cache_hits'] += 1
            return self.signature_cache[path_key]
            
        self.stats['cache_misses'] += 1
        
        # Create MinHash
        mh = MinHash(num_perm=self.num_perm)
        
        # Add path elements
        for i, action in enumerate(path):
            # Include position to maintain order information
            element = f"{i}:{action}"
            mh.update(element.encode('utf-8'))
            
        # Add n-grams for better similarity detection
        for i in range(len(path) - 1):
            bigram = f"{path[i]}->{path[i+1]}"
            mh.update(bigram.encode('utf-8'))
            
        # Cache signature
        self.signature_cache[path_key] = mh
        
        return mh
        
    def compute_interference(
        self,
        paths: List[List[int]],
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute interference values for a set of paths
        
        Args:
            paths: List of paths (each path is a list of actions)
            weights: Optional importance weights for paths
            
        Returns:
            Interference values for each path
        """
        n_paths = len(paths)
        if n_paths == 0:
            return np.array([])
            
        if weights is None:
            weights = np.ones(n_paths)
            
        # Compute signatures
        signatures = []
        for path in paths:
            sig = self.compute_path_signature(path)
            signatures.append(sig)
            
        # Compute pairwise similarities efficiently using LSH
        interference = np.zeros(n_paths)
        similarity_sum = 0.0
        
        # Clear LSH for fresh computation
        self.lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm,
            params=(self.num_bands, self.num_perm // self.num_bands)
        )
        
        # Insert all signatures into LSH
        for i, sig in enumerate(signatures):
            self.lsh.insert(f"path_{i}", sig)
            
        # Query for similar paths
        for i, sig in enumerate(signatures):
            # Find similar paths
            similar_keys = self.lsh.query(sig)
            
            # Compute interference from similar paths
            for key in similar_keys:
                if key != f"path_{i}":
                    j = int(key.split('_')[1])
                    
                    # Compute exact similarity
                    similarity = sig.jaccard(signatures[j])
                    
                    # Weight by importance
                    interference[i] += similarity * weights[j]
                    similarity_sum += similarity
                    
        # Normalize interference
        if n_paths > 1:
            interference /= (n_paths - 1)
            
        # Update statistics
        self.stats['paths_processed'] += n_paths
        self.stats['interference_events'] += np.sum(interference > 0)
        if n_paths > 1:
            self.stats['average_similarity'] = similarity_sum / (n_paths * (n_paths - 1))
            
        return interference
        
    def apply_interference_to_selection(
        self,
        ucb_scores: np.ndarray,
        paths: List[List[int]],
        interference_strength: float = 0.5
    ) -> np.ndarray:
        """Apply interference to UCB scores for path selection
        
        Args:
            ucb_scores: Original UCB scores for paths
            paths: Corresponding paths
            interference_strength: How much to reduce scores of similar paths
            
        Returns:
            Modified UCB scores
        """
        # Compute interference
        interference = self.compute_interference(paths)
        
        # Apply interference to reduce scores of similar paths
        modified_scores = ucb_scores * (1.0 - interference_strength * interference)
        
        logger.debug(
            f"Interference applied: avg reduction {np.mean(1 - modified_scores/ucb_scores):.3f}"
        )
        
        return modified_scores
        
    def get_diverse_paths(
        self,
        candidate_paths: List[List[int]],
        n_select: int,
        diversity_weight: float = 0.5
    ) -> List[int]:
        """Select diverse paths from candidates
        
        Args:
            candidate_paths: List of candidate paths
            n_select: Number of paths to select
            diversity_weight: Weight for diversity vs quality
            
        Returns:
            Indices of selected paths
        """
        n_candidates = len(candidate_paths)
        if n_candidates <= n_select:
            return list(range(n_candidates))
            
        # Compute all signatures
        signatures = [self.compute_path_signature(path) for path in candidate_paths]
        
        # Greedy selection for diversity
        selected = []
        remaining = set(range(n_candidates))
        
        # Select first path (could be based on quality score)
        first = np.random.choice(list(remaining))
        selected.append(first)
        remaining.remove(first)
        
        # Select remaining paths to maximize diversity
        while len(selected) < n_select and remaining:
            best_idx = None
            best_score = -float('inf')
            
            for idx in remaining:
                # Compute minimum similarity to already selected
                min_similarity = 1.0
                for sel_idx in selected:
                    sim = signatures[idx].jaccard(signatures[sel_idx])
                    min_similarity = min(min_similarity, sim)
                    
                # Score combines diversity and randomness
                diversity_score = 1.0 - min_similarity
                score = diversity_weight * diversity_score + (1 - diversity_weight) * np.random.rand()
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
                    
            selected.append(best_idx)
            remaining.remove(best_idx)
            
        return selected
        
    def compute_path_clustering(
        self,
        paths: List[List[int]],
        min_cluster_size: int = 5
    ) -> Dict[int, List[int]]:
        """Cluster similar paths together
        
        Args:
            paths: List of paths
            min_cluster_size: Minimum size for a cluster
            
        Returns:
            Dictionary mapping cluster ID to path indices
        """
        n_paths = len(paths)
        if n_paths == 0:
            return {}
            
        # Compute signatures
        signatures = [self.compute_path_signature(path) for path in paths]
        
        # Build similarity graph using LSH
        graph = defaultdict(set)
        
        # Create fresh LSH
        lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm
        )
        
        # Insert all signatures
        for i, sig in enumerate(signatures):
            lsh.insert(i, sig)
            
        # Find connections
        for i, sig in enumerate(signatures):
            similar = lsh.query(sig)
            for j in similar:
                if i != j:
                    graph[i].add(j)
                    graph[j].add(i)
                    
        # Find connected components (clusters)
        clusters = {}
        visited = set()
        cluster_id = 0
        
        for start in range(n_paths):
            if start in visited:
                continue
                
            # BFS to find component
            cluster = []
            queue = [start]
            
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                    
                visited.add(node)
                cluster.append(node)
                
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        
            # Add cluster if large enough
            if len(cluster) >= min_cluster_size:
                clusters[cluster_id] = cluster
                cluster_id += 1
                
        return clusters
        
    def get_statistics(self) -> Dict[str, float]:
        """Get interference statistics
        
        Returns:
            Dictionary of statistics
        """
        stats = dict(self.stats)
        
        # Add cache statistics
        stats['cache_size'] = len(self.signature_cache)
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            )
        else:
            stats['cache_hit_rate'] = 0.0
            
        return stats
        
    def clear_cache(self) -> None:
        """Clear signature cache"""
        self.signature_cache.clear()
        self.stats['cache_hits'] = 0
        self.stats['cache_misses'] = 0