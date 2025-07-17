"""
MCTS dynamics data extractor for quantum phenomena analysis.

Extracts raw MCTS data during self-play for subsequent analysis
by statistical mechanics and quantum phenomena modules.
"""
import torch
import numpy as np
import json
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

# Import quantum phenomena analyzers if available
try:
    from ..phenomena import (
        DecoherenceAnalyzer, DecoherenceResult,
        TunnelingDetector, TunnelingEvent, ValueBarrier,
        EntanglementAnalyzer, EntanglementResult
    )
    QUANTUM_ANALYZERS_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from python.mcts.quantum.phenomena import (
            DecoherenceAnalyzer, DecoherenceResult,
            TunnelingDetector, TunnelingEvent, ValueBarrier,
            EntanglementAnalyzer, EntanglementResult
        )
        QUANTUM_ANALYZERS_AVAILABLE = True
    except ImportError:
        logger.warning("Quantum phenomena analyzers not available")
        QUANTUM_ANALYZERS_AVAILABLE = False


@dataclass
class ExtractionConfig:
    """Configuration for dynamics extraction"""
    extract_q_values: bool = True
    extract_visits: bool = True
    extract_policy: bool = True
    extract_value_landscape: bool = True
    extract_search_depth: bool = True
    extract_quantum_phenomena: bool = True
    
    # Filtering options
    filter_critical: bool = False
    critical_threshold: float = 0.05
    
    # Sampling options
    time_window: int = 100
    sampling_rate: int = 1  # Extract every nth position
    
    # Performance options
    batch_size: int = 32
    compress_data: bool = True
    
    # Quantum phenomena options
    extract_quantum_phenomena: bool = True
    include_decoherence: bool = True
    include_tunneling: bool = True
    include_entanglement: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if self.time_window < 0:
            raise ValueError("time_window must be non-negative")
        if self.critical_threshold < 0:
            raise ValueError("critical_threshold must be non-negative")
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return (
            self.sampling_rate > 0 and
            self.time_window >= 0 and
            self.critical_threshold >= 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class DynamicsData:
    """Container for extracted MCTS dynamics data"""
    snapshots: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save to JSON file"""
        path = Path(path)
        data = {
            'snapshots': self.snapshots,
            'metadata': self.metadata
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DynamicsData':
        """Load from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            snapshots=data['snapshots'],
            metadata=data.get('metadata', {})
        )
    
    def save_compressed(self, path: Union[str, Path]) -> None:
        """Save compressed using numpy"""
        path = Path(path)
        
        # Convert to numpy arrays for compression
        arrays = {}
        metadata = self.metadata.copy()
        
        # Extract common fields into arrays
        if self.snapshots:
            timestamps = [s['timestamp'] for s in self.snapshots]
            arrays['timestamps'] = np.array(timestamps)
            
            # Handle variable-size data
            if 'q_values' in self.snapshots[0]:
                q_values = [s['q_values'] for s in self.snapshots]
                arrays['q_values'] = np.array(q_values)
            
            if 'visits' in self.snapshots[0]:
                visits = [s['visits'] for s in self.snapshots]
                arrays['visits'] = np.array(visits)
        
        # Save as compressed numpy
        np.savez_compressed(path, metadata=metadata, **arrays)
    
    @classmethod
    def load_compressed(cls, path: Union[str, Path]) -> 'DynamicsData':
        """Load from compressed numpy file"""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        
        metadata = data['metadata'].item() if 'metadata' in data else {}
        
        # Reconstruct snapshots
        snapshots = []
        n_snapshots = len(data['timestamps']) if 'timestamps' in data else 0
        
        for i in range(n_snapshots):
            snapshot = {'timestamp': int(data['timestamps'][i])}
            
            if 'q_values' in data:
                snapshot['q_values'] = data['q_values'][i].tolist()
            
            if 'visits' in data:
                snapshot['visits'] = data['visits'][i].tolist()
            
            snapshots.append(snapshot)
        
        return cls(snapshots=snapshots, metadata=metadata)
    
    @classmethod
    def create_stream(cls, path: Union[str, Path]) -> 'StreamWriter':
        """Create a streaming writer"""
        return StreamWriter(path)
    
    @classmethod
    def read_stream(cls, path: Union[str, Path], 
                   chunk_size: int = 100) -> Iterator[List[Dict[str, Any]]]:
        """Read snapshots in chunks"""
        path = Path(path)
        chunk = []
        
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    snapshot = json.loads(line)
                    chunk.append(snapshot)
                    
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            
            # Yield remaining
            if chunk:
                yield chunk


class StreamWriter:
    """Streaming writer for large datasets"""
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.file = open(self.path, 'w')
    
    def write_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Write a single snapshot"""
        json.dump(snapshot, self.file)
        self.file.write('\n')
        self.file.flush()
    
    def close(self) -> None:
        """Close the stream"""
        self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MCTSDynamicsExtractor:
    """
    Extracts MCTS dynamics data for quantum phenomena analysis.
    
    Captures:
    - Q-value landscapes
    - Visit distributions
    - Policy evolution
    - Search statistics
    - Derived thermodynamic quantities
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None,
                 n_workers: int = 1):
        """
        Initialize extractor.
        
        Args:
            config: Extraction configuration
            n_workers: Number of parallel workers
        """
        self.config = config or ExtractionConfig()
        self.n_workers = n_workers
    
    def extract_from_position(self, state: Any, mcts: Any) -> Dict[str, Any]:
        """
        Extract dynamics data from a single position.
        
        Args:
            state: Game state
            mcts: MCTS object (not just tree)
            
        Returns:
            Position dynamics data
        """
        position_data = {}
        
        # Get root node from MCTS
        root = mcts.get_root()
        
        # Extract Q-values and visits for all actions
        if self.config.extract_q_values or self.config.extract_visits:
            # Get action space size from state
            action_space_size = len(mcts.game_interface.get_legal_moves(state))
            q_values = []
            visits = []
            
            for action in range(action_space_size):
                child = mcts.get_child(root, action)
                if child:
                    if self.config.extract_q_values:
                        q_val = mcts.get_q_value(child)
                        q_values.append(q_val)
                    if self.config.extract_visits:
                        visit_count = mcts.get_visit_count(child)
                        visits.append(visit_count)
                else:
                    if self.config.extract_q_values:
                        q_values.append(0.0)
                    if self.config.extract_visits:
                        visits.append(0)
            
            if self.config.extract_q_values:
                position_data['q_values'] = q_values
            if self.config.extract_visits:
                position_data['visits'] = visits
                position_data['total_visits'] = sum(visits)
        
        # Extract policy (from last search result)
        if self.config.extract_policy:
            # Policy was already computed during search
            position_data['policy'] = mcts.get_last_policy() if hasattr(mcts, 'get_last_policy') else []
        
        # Extract search depth
        if self.config.extract_search_depth:
            position_data['depth'] = mcts.get_tree_depth() if hasattr(mcts, 'get_tree_depth') else 0
        
        # Compute derived quantities
        derived = self.compute_derived_quantities(position_data)
        position_data.update(derived)
        
        # Extract quantum phenomena if enabled
        if self.config.extract_quantum_phenomena and QUANTUM_ANALYZERS_AVAILABLE:
            quantum_data = self._extract_quantum_phenomena(mcts, position_data)
            position_data.update(quantum_data)
        
        return position_data
    
    def extract_trajectory_dynamics(self, trajectory: List[Dict[str, Any]],
                                  config: Optional[ExtractionConfig] = None) -> DynamicsData:
        """
        Extract dynamics from a game trajectory.
        
        Args:
            trajectory: List of position data (may contain tree references)
            config: Optional config override
            
        Returns:
            Extracted dynamics data
        """
        config = config or self.config
        snapshots = []
        
        for i, position in enumerate(trajectory):
            # Apply sampling rate
            if i % config.sampling_rate != 0:
                continue
            
            # Check if data is already extracted (new format)
            if 'q_values' in position and 'visits' in position:
                # Use pre-extracted data directly
                position_data = position
            elif 'tree' in position and 'state' in position:
                # Try to extract from tree (legacy format)
                try:
                    extracted_data = self.extract_from_position(position['state'], position['tree'])
                    extracted_data['position_id'] = position.get('position_id', i)
                    extracted_data['timestamp'] = position.get('move_number', i)
                    extracted_data['temperature'] = position.get('temperature', 1.0)
                    extracted_data['player'] = position.get('player', 1)
                    extracted_data['action'] = position.get('action')
                    position_data = extracted_data
                except Exception as e:
                    logger.debug(f"Failed to extract from tree: {e}, using position data as-is")
                    position_data = position
            else:
                # Use position data as-is
                position_data = position
            
            # Apply critical filter if enabled
            if config.filter_critical:
                if not self._is_critical_position(position_data, config.critical_threshold):
                    continue
            
            # Create snapshot
            snapshot = {
                'timestamp': position_data.get('timestamp', i),
                'position_id': position_data.get('position_id', i)
            }
            
            # Extract requested data
            if config.extract_q_values and 'q_values' in position_data:
                snapshot['q_values'] = self._tensor_to_list(position_data['q_values'])
            
            if config.extract_visits and 'visits' in position_data:
                snapshot['visits'] = self._tensor_to_list(position_data['visits'])
                if 'total_visits' in position_data:
                    snapshot['total_visits'] = position_data['total_visits']
            
            if config.extract_policy and 'policy' in position_data:
                snapshot['policy'] = self._tensor_to_list(position_data['policy'])
            
            if 'depth' in position_data:
                snapshot['depth'] = position_data['depth']
            
            # Add derived quantities
            if 'temperature' in position_data:
                snapshot['temperature'] = position_data['temperature']
            if 'energy' in position_data:
                snapshot['energy'] = position_data['energy']
            if 'entropy' in position_data:
                snapshot['entropy'] = position_data['entropy']
            
            # Add quantum phenomena data if present
            for key in ['decoherence_result', 'tunneling_events', 'entanglement_result']:
                if key in position_data:
                    snapshot[key] = position_data[key]
            
            snapshots.append(snapshot)
        
        # Create metadata
        metadata = {
            'total_positions': len(trajectory),
            'extracted_positions': len(snapshots),
            'config': config.to_dict()
        }
        
        return DynamicsData(snapshots=snapshots, metadata=metadata)
    
    def extract_batch(self, games: List[Any]) -> List[DynamicsData]:
        """
        Extract from multiple games in parallel.
        
        Args:
            games: List of game objects
            
        Returns:
            List of dynamics data
        """
        if self.n_workers <= 1:
            # Sequential processing
            results = []
            for game in games:
                trajectory = game.get_trajectory()
                dynamics = self.extract_trajectory_dynamics(trajectory)
                results.append(dynamics)
            return results
        
        # Parallel processing - use ThreadPoolExecutor to avoid file descriptor issues
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for game in games:
                trajectory = game.get_trajectory()
                future = executor.submit(
                    self.extract_trajectory_dynamics, trajectory
                )
                futures.append(future)
            
            results = [f.result() for f in futures]
        
        return results
    
    def compute_derived_quantities(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute thermodynamic and other derived quantities.
        
        Args:
            position_data: Raw position data
            
        Returns:
            Dictionary of derived quantities
        """
        derived = {}
        
        # Compute temperature from visit distribution
        if 'visits' in position_data and 'q_values' in position_data:
            visits = position_data['visits']
            q_values = position_data['q_values']
            
            if isinstance(visits, list):
                visits = torch.tensor(visits)
            if isinstance(q_values, list):
                q_values = torch.tensor(q_values)
            
            # Temperature from visit distribution
            if isinstance(visits, torch.Tensor):
                total_visits = position_data.get('total_visits', visits.sum().item())
            else:
                total_visits = position_data.get('total_visits', sum(visits))
            if total_visits > 0:
                # β ∝ √N from theory
                temperature = 1.0 / np.sqrt(total_visits)
                derived['temperature'] = temperature
                
                # Entropy S = -Σ π log π
                probs = visits / total_visits
                probs = probs + 1e-10  # Avoid log(0)
                entropy = -torch.sum(probs * torch.log(probs)).item()
                derived['entropy'] = entropy
                
                # Energy E = -<Q>
                energy = -torch.sum(probs * q_values).item()
                derived['energy'] = energy
        
        return derived
    
    def _extract_quantum_phenomena(self, tree: Any, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract quantum phenomena measurements.
        
        Args:
            tree: MCTS tree
            position_data: Basic position data
            
        Returns:
            Dictionary of quantum phenomena results
        """
        quantum_data = {}
        
        # Initialize analyzers if not done
        if not hasattr(self, '_quantum_analyzers'):
            self._quantum_analyzers = {}
            if self.config.include_decoherence:
                self._quantum_analyzers['decoherence'] = DecoherenceAnalyzer()
            if self.config.include_tunneling:
                self._quantum_analyzers['tunneling'] = TunnelingDetector()
            if self.config.include_entanglement:
                self._quantum_analyzers['entanglement'] = EntanglementAnalyzer()
        
        # Decoherence analysis
        if self.config.include_decoherence and 'decoherence' in self._quantum_analyzers:
            try:
                result = self._quantum_analyzers['decoherence'].analyze(tree)
                quantum_data['decoherence_result'] = asdict(result) if hasattr(result, '__dict__') else result
            except Exception as e:
                logger.debug(f"Decoherence analysis failed: {e}")
        
        # Tunneling detection
        if self.config.include_tunneling and 'tunneling' in self._quantum_analyzers:
            try:
                events = self._quantum_analyzers['tunneling'].detect_tunneling(tree)
                if events:
                    quantum_data['tunneling_events'] = [
                        asdict(event) if hasattr(event, '__dict__') else event 
                        for event in events
                    ]
                
                # Barrier analysis
                barriers = self._quantum_analyzers['tunneling'].analyze_barriers(tree)
                if barriers:
                    quantum_data['barrier_analysis'] = barriers
            except Exception as e:
                logger.debug(f"Tunneling analysis failed: {e}")
        
        # Entanglement analysis
        if self.config.include_entanglement and 'entanglement' in self._quantum_analyzers:
            try:
                result = self._quantum_analyzers['entanglement'].analyze(tree)
                quantum_data['entanglement_result'] = asdict(result) if hasattr(result, '__dict__') else result
            except Exception as e:
                logger.debug(f"Entanglement analysis failed: {e}")
        
        return quantum_data
    
    def _is_critical_position(self, position: Dict[str, Any],
                            threshold: float) -> bool:
        """Check if position is critical"""
        if 'q_values' not in position:
            return False
        
        q_values = position['q_values']
        if isinstance(q_values, list):
            q_values = torch.tensor(q_values)
        
        if len(q_values) < 2:
            return False
        
        # Sort Q-values
        sorted_q, _ = torch.sort(q_values, descending=True)
        
        # Check if top two are close
        delta_q = sorted_q[0] - sorted_q[1]
        
        return delta_q < threshold
    
    def _tensor_to_list(self, tensor: Union[torch.Tensor, List, np.ndarray]) -> List:
        """Convert tensor to list for JSON serialization"""
        if isinstance(tensor, torch.Tensor):
            return tensor.tolist()
        elif isinstance(tensor, np.ndarray):
            return tensor.tolist()
        else:
            return list(tensor)