#!/usr/bin/env python3
"""
Real MCTS Data Collector for Quantum Research

This module collects authentic data from actual MCTS self-play runs for physics analysis.
It captures real tree expansion dynamics, quantum state evolution, and performance metrics
during actual game simulations using the optimized MCTS implementation.

Usage:
    collector = RealMCTSDataCollector()
    collector.run_data_collection_session(n_games=10, num_simulations=1000)
    data = collector.export_data('quantum_mcts_data.json')
"""

import sys
import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle
import threading
from datetime import datetime
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Set up logging - reduce verbosity by default
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # Keep our module at WARNING level

# Suppress verbose CUDA and other logging completely
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('mcts').setLevel(logging.ERROR)  # Reduced back to ERROR level
logging.getLogger('alphazero_py').setLevel(logging.ERROR)
logging.getLogger('authentic_mcts_physics_extractor').setLevel(logging.ERROR)

# Suppress all other potential verbose loggers
for name in ['matplotlib', 'PIL', 'numba', 'cuda', 'gpu']:
    logging.getLogger(name).setLevel(logging.ERROR)

# Add the MCTS modules to path
mcts_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(mcts_root))

# Import MCTS modules
try:
    from mcts.core.mcts import MCTS, MCTSConfig
    from mcts.core.game_interface import GameInterface, GameType
    from mcts.gpu.gpu_game_states import GameType as GPUGameType  
    from mcts.neural_networks.resnet_evaluator import ResNetEvaluator
    from mcts.neural_networks.mock_evaluator import MockEvaluator
    import alphazero_py
    
    MCTS_AVAILABLE = True
    # Successfully imported MCTS modules
except ImportError as e:
    logger.error(f"Could not import MCTS modules: {e}")
    logger.error("This collector requires the full MCTS implementation")
    sys.exit(1)

@dataclass
class RealTreeSnapshot:
    """Snapshot of real MCTS tree at a specific point in time"""
    timestamp: float
    move_number: int
    player: int
    
    # Tree statistics
    total_nodes: int
    total_visits: int
    tree_depth: int
    root_visits: int
    root_value: float
    
    # Node-level data (extracted from CSRTree)
    visit_counts: List[int]
    value_sums: List[float]  
    q_values: List[float]
    node_priors: List[float]
    parent_indices: List[int]
    parent_actions: List[int]
    node_phases: List[float]  # Quantum phases
    
    # Tree structure analysis
    branching_factor: float
    max_depth: int
    leaf_nodes: List[int]
    
    # Policy and search data
    policy_distribution: List[float]
    search_time: float
    simulations_per_second: float
    
    # Quantum-specific data
    quantum_mode_active: bool
    quantum_corrections: List[float]
    decoherence_detected: bool

@dataclass 
class RealGameSession:
    """Complete real game session data"""
    game_id: str
    start_time: float
    end_time: float
    game_type: str
    board_size: int
    winner: int
    final_score: float
    move_count: int
    
    # MCTS configuration used
    mcts_config: Dict[str, Any]
    
    # Tree snapshots during game
    tree_snapshots: List[RealTreeSnapshot]
    
    # Move sequence with full context
    moves: List[Dict[str, Any]]
    
    # Performance metrics
    total_simulations: int
    avg_simulations_per_move: float
    total_search_time: float
    peak_sims_per_second: float
    
    # Game progression analysis
    game_phases: List[str]  # opening, midgame, endgame
    complexity_evolution: List[float]
    decision_confidence: List[float]


class RealMCTSDataCollector:
    """Collects comprehensive data from real MCTS self-play runs"""
    
    def __init__(self, evaluator_type: str = "mock", device: str = "cpu", verbose: bool = False):
        self.evaluator_type = evaluator_type
        self.device = device
        self.verbose = verbose
        self.game_sessions: List[RealGameSession] = []
        
        # Set logging level based on verbose flag
        if not verbose:
            logger.setLevel(logging.ERROR)
        
        # Initialize MCTS system
        self._initialize_mcts_system()
        
        # Collection metadata
        self.collection_metadata = {
            'collection_start': None,
            'collection_end': None,
            'total_games': 0,
            'total_tree_snapshots': 0,
            'evaluator_type': evaluator_type,
            'device': device,
            'data_version': '4.0.0-real'
        }
        
        # Real MCTS Data Collector initialized
    
    def _initialize_mcts_system(self):
        """Initialize real MCTS components"""
        
        # Create evaluator
        if self.evaluator_type == "mock":
            self.evaluator = MockEvaluator()
            pass  # Using mock evaluator
        else:
            # Would use real ResNet evaluator here
            self.evaluator = MockEvaluator()
            logger.warning("ResNet evaluator not implemented, using mock evaluator")
        
        # MCTS configuration optimized for data collection - CLASSICAL MODE
        self.mcts_config = MCTSConfig(
            num_simulations=800,  # Reasonable number for data collection
            c_puct=1.414,
            temperature=1.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            device=self.device,
            game_type=GPUGameType.GOMOKU,
            board_size=15,
            # QUANTUM FEATURES DISABLED for pure classical MCTS data
            enable_quantum=False,
            # Wave parallelization settings - appropriate for performance
            wave_size=3072,
            min_wave_size=3072,
            max_wave_size=3072,
            # Memory settings  
            max_tree_nodes=50000,
            use_mixed_precision=False,  # Simpler for data collection
            # Enable debug logging to diagnose tree expansion issues
            enable_debug_logging=True,
        )
        
        # Create MCTS instance
        self.mcts = MCTS(self.mcts_config, self.evaluator)
        
        # Optimize for current hardware
        self.mcts.optimize_for_hardware()
        
        # MCTS initialized
    
    def _extract_tree_snapshot(self, move_number: int, player: int, 
                             search_time: float, policy: np.ndarray) -> RealTreeSnapshot:
        """Extract comprehensive tree snapshot from current MCTS state"""
        
        timestamp = time.time()
        
        # Access the CSRTree directly
        tree = self.mcts.tree
        
        # Extract basic tree statistics
        total_nodes = tree.num_nodes
        root_visits = tree.visit_counts[0].item() if total_nodes > 0 else 0
        root_value = tree.value_sums[0].item() / max(root_visits, 1) if root_visits > 0 else 0.0
        
        # Calculate tree depth
        max_depth = 0
        if total_nodes > 0:
            # Simple depth calculation by following longest path
            current_depth = 0
            for node_idx in range(min(total_nodes, 1000)):  # Limit for performance
                depth = 0
                current = node_idx
                while current != -1 and depth < 100:  # Avoid infinite loops
                    parent = tree.parent_indices[current].item()
                    if parent == -1:
                        break
                    current = parent
                    depth += 1
                max_depth = max(max_depth, depth)
        
        # Extract node-level data (limit to reasonable size)
        extract_limit = min(total_nodes, 1000)  # Don't extract massive trees
        
        visit_counts = []
        value_sums = []
        q_values = []
        node_priors = []
        parent_indices = []
        parent_actions = []
        node_phases = []
        
        if total_nodes > 0:
            # Convert tensors to lists with proper type conversion for JSON
            visit_counts = [int(x) for x in tree.visit_counts[:extract_limit].cpu().numpy()]
            value_sums = [float(x) for x in tree.value_sums[:extract_limit].cpu().numpy()]
            node_priors = [float(x) for x in tree.node_priors[:extract_limit].cpu().numpy()]
            parent_indices = [int(x) for x in tree.parent_indices[:extract_limit].cpu().numpy()]
            parent_actions = [int(x) for x in tree.parent_actions[:extract_limit].cpu().numpy()]
            node_phases = [float(x) for x in tree.phases[:extract_limit].cpu().numpy()]
            
            # Calculate Q-values
            q_values = []
            for i in range(extract_limit):
                visits = visit_counts[i]
                if visits > 0:
                    q_val = value_sums[i] / visits
                else:
                    q_val = 0.0
                q_values.append(q_val)
        
        # Find leaf nodes
        leaf_nodes = []
        for i in range(min(extract_limit, 100)):  # Limit leaf search
            if i < total_nodes:
                # Check if node has children by looking at row_ptr
                if hasattr(tree, 'row_ptr') and i + 1 < len(tree.row_ptr):
                    has_children = tree.row_ptr[i + 1] > tree.row_ptr[i]
                    if not has_children:
                        leaf_nodes.append(i)
        
        # Calculate branching factor
        branching_factor = 0.0
        if total_nodes > 1:
            total_children = sum(1 for i in range(min(total_nodes, 100)) 
                               if parent_indices[i] != -1) if parent_indices else 0
            internal_nodes = max(total_nodes - len(leaf_nodes), 1)
            branching_factor = total_children / internal_nodes
        
        # Search performance
        sims_per_second = self.mcts_config.num_simulations / max(search_time, 0.001)
        
        # Quantum analysis - DISABLED for classical MCTS data collection
        quantum_mode_active = False  # Explicitly disabled
        quantum_corrections = []     # No quantum corrections in classical mode
        decoherence_detected = False # No decoherence in classical mode
        
        # Note: node_phases may still exist in CSRTree structure but will be zeros
        # This provides baseline data for quantum vs classical comparison
        
        return RealTreeSnapshot(
            timestamp=timestamp,
            move_number=move_number,
            player=player,
            total_nodes=total_nodes,
            total_visits=sum(visit_counts) if visit_counts else 0,
            tree_depth=max_depth,
            root_visits=root_visits,
            root_value=root_value,
            visit_counts=visit_counts,
            value_sums=value_sums,
            q_values=q_values,
            node_priors=node_priors,
            parent_indices=parent_indices,
            parent_actions=parent_actions,
            node_phases=node_phases,
            branching_factor=branching_factor,
            max_depth=max_depth,
            leaf_nodes=leaf_nodes,
            policy_distribution=policy.tolist(),
            search_time=search_time,
            simulations_per_second=sims_per_second,
            quantum_mode_active=quantum_mode_active,
            quantum_corrections=quantum_corrections,
            decoherence_detected=decoherence_detected
        )
    
    def play_real_game_with_data_collection(self, game_type: str = "gomoku",
                                          board_size: int = 15,
                                          snapshot_frequency: int = 5) -> RealGameSession:
        """Play a real MCTS self-play game while collecting comprehensive data"""
        
        game_id = f"{game_type}_{int(time.time())}_{np.random.randint(1000, 9999)}"
        start_time = time.time()
        
        # Starting real MCTS game
        
        # Initialize game state
        if game_type.lower() == "gomoku":
            state = alphazero_py.GomokuState()
        else:
            raise ValueError(f"Unsupported game type: {game_type}")
        
        # Initialize game session
        game_session = RealGameSession(
            game_id=game_id,
            start_time=start_time,
            end_time=0.0,
            game_type=game_type,
            board_size=board_size,
            winner=-1,
            final_score=0.0,
            move_count=0,
            mcts_config=asdict(self.mcts_config),
            tree_snapshots=[],
            moves=[],
            total_simulations=0,
            avg_simulations_per_move=0.0,
            total_search_time=0.0,
            peak_sims_per_second=0.0,
            game_phases=[],
            complexity_evolution=[],
            decision_confidence=[]
        )
        
        move_count = 0
        total_search_time = 0.0
        
        # Play game until terminal
        while not state.is_terminal() and move_count < 500:  # Limit game length
            move_count += 1
            current_player = state.get_current_player()
            
            # Move progress (suppressed for clean output)
            
            # Determine temperature (exploration vs exploitation)
            if move_count <= 30:
                temperature = 1.0  # Explore early game
            else:
                temperature = 0.1  # More deterministic late game
            
            # Reset tree for new position
            self.mcts.reset_tree()
            
            # Run MCTS search
            search_start = time.perf_counter()
            policy = self.mcts.search(state, self.mcts_config.num_simulations)
            search_time = time.perf_counter() - search_start
            
            # DEBUG: Check tree state after search
            if move_count <= 2:  # Only debug first few moves
                root_children, _, _ = self.mcts.tree.get_children(0)
                total_visits = self.mcts.tree.visit_counts[0].item()
                logger.warning(f"Move {move_count}: Root has {len(root_children)} children, {total_visits} total visits")
                if len(root_children) > 0:
                    child_visits = [self.mcts.tree.visit_counts[child].item() for child in root_children[:5]]
                    logger.warning(f"  Child visits: {child_visits}")
            
            total_search_time += search_time
            game_session.total_simulations += self.mcts_config.num_simulations
            
            # Update peak performance
            sims_per_sec = self.mcts_config.num_simulations / search_time
            game_session.peak_sims_per_second = max(game_session.peak_sims_per_second, sims_per_sec)
            
            # Extract tree snapshot
            if move_count % snapshot_frequency == 0 or move_count <= 10:
                snapshot = self._extract_tree_snapshot(move_count, current_player, search_time, policy)
                game_session.tree_snapshots.append(snapshot)
            
            # Normalize policy to ensure probabilities sum to 1
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                # Fallback to uniform distribution if policy is all zeros
                policy = np.ones(len(policy)) / len(policy)
            
            # Select action based on policy and temperature
            if temperature > 0.5:
                # Sample from policy
                action = np.random.choice(len(policy), p=policy)
            else:
                # Select best action
                action = np.argmax(policy)
            
            # Record move
            move_data = {
                'move_number': move_count,
                'player': current_player,
                'action': action,
                'policy': policy.tolist(),
                'search_time': search_time,
                'temperature': temperature,
                'tree_size': self.mcts.tree.num_nodes,
                'confidence': float(np.max(policy))
            }
            game_session.moves.append(move_data)
            game_session.decision_confidence.append(float(np.max(policy)))
            
            # Calculate game complexity (entropy of policy)
            policy_entropy = -np.sum(policy * np.log(policy + 1e-10))
            game_session.complexity_evolution.append(float(policy_entropy))
            
            # Determine game phase
            if move_count <= 15:
                phase = "opening"
            elif move_count <= 30:
                phase = "midgame"
            else:
                phase = "endgame"
            game_session.game_phases.append(phase)
            
            # Apply action to game state
            state.make_move(action)
            
            # Action details (suppressed for clean output)
        
        # Game finished
        game_session.end_time = time.time()
        game_session.move_count = move_count
        game_session.avg_simulations_per_move = game_session.total_simulations / max(move_count, 1)
        game_session.total_search_time = total_search_time
        
        # Determine winner
        if state.is_terminal():
            result = state.get_game_result()
            if result == alphazero_py.GameResult.WIN_PLAYER1:
                game_session.winner = 0  # Player 1 (0-indexed)
                game_session.final_score = 1.0
            elif result == alphazero_py.GameResult.WIN_PLAYER2:
                game_session.winner = 1  # Player 2 (0-indexed)
                game_session.final_score = 1.0
            elif result == alphazero_py.GameResult.DRAW:
                game_session.winner = -1  # Draw
                game_session.final_score = 0.5
            else:
                game_session.winner = -1  # Unknown
                game_session.final_score = 0.0
        
        game_duration = game_session.end_time - game_session.start_time
        # Game completed (details suppressed for clean output)
        
        return game_session
    
    def run_data_collection_session(self, n_games: int = 10,
                                  game_type: str = "gomoku",
                                  board_size: int = 15,
                                  snapshot_frequency: int = 5) -> Dict[str, Any]:
        """Run a complete real MCTS data collection session"""
        
        print(f"🎯 Starting MCTS data collection: {n_games} games")  # Use print instead of logger
        self.collection_metadata['collection_start'] = time.time()
        
        # Use tqdm for progress tracking
        for game_idx in tqdm(range(n_games), desc="Collecting MCTS data", unit="games"):
            
            try:
                game_session = self.play_real_game_with_data_collection(
                    game_type=game_type,
                    board_size=board_size,
                    snapshot_frequency=snapshot_frequency
                )
                self.game_sessions.append(game_session)
                
            except Exception as e:
                import traceback
                logger.error(f"Error in game {game_idx + 1}: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue
        
        self.collection_metadata['collection_end'] = time.time()
        self.collection_metadata['total_games'] = len(self.game_sessions)
        self.collection_metadata['total_tree_snapshots'] = sum(
            len(session.tree_snapshots) for session in self.game_sessions
        )
        
        # Generate summary
        total_duration = self.collection_metadata['collection_end'] - self.collection_metadata['collection_start']
        total_moves = sum(session.move_count for session in self.game_sessions)
        total_simulations = sum(session.total_simulations for session in self.game_sessions)
        
        summary = {
            'collection_summary': {
                'total_duration': total_duration,
                'games_completed': len(self.game_sessions),
                'games_requested': n_games,
                'success_rate': len(self.game_sessions) / n_games if n_games > 0 else 0,
            },
            'game_statistics': {
                'total_moves': total_moves,
                'avg_moves_per_game': total_moves / max(len(self.game_sessions), 1),
                'total_simulations': total_simulations,
                'avg_simulations_per_game': total_simulations / max(len(self.game_sessions), 1)
            },
            'tree_statistics': {
                'total_snapshots': self.collection_metadata['total_tree_snapshots'],
                'avg_snapshots_per_game': self.collection_metadata['total_tree_snapshots'] / max(len(self.game_sessions), 1)
            },
            'performance_statistics': {
                'avg_sims_per_second': sum(session.peak_sims_per_second for session in self.game_sessions) / max(len(self.game_sessions), 1),
                'total_search_time': sum(session.total_search_time for session in self.game_sessions)
            }
        }
        
        print(f"✅ MCTS data collection completed: {summary}")  # Use print instead of logger
        return summary
    
    def export_data(self, output_path: str) -> str:
        """Export collected real MCTS data to JSON file"""
        
        # Convert game sessions to dictionary format
        game_data = []
        tree_snapshots = []
        
        for session in self.game_sessions:
            # Convert session to dict
            session_dict = asdict(session)
            game_data.append(session_dict)
            
            # Extract tree snapshots for physics analysis
            for snapshot in session.tree_snapshots:
                snapshot_dict = asdict(snapshot)
                tree_snapshots.append(snapshot_dict)
        
        # Create final data structure
        export_data = {
            'metadata': self.collection_metadata,
            'game_sessions': game_data,
            'tree_expansion_data': tree_snapshots,  # For compatibility with visualization tools
            'performance_metrics': [
                {
                    'game_id': session.game_id,
                    'move_count': session.move_count,
                    'total_simulations': session.total_simulations,
                    'avg_search_time': session.total_search_time / max(session.move_count, 1),
                    'peak_sims_per_second': session.peak_sims_per_second,
                    'winner': session.winner
                }
                for session in self.game_sessions
            ]
        }
        
        # Save to file with custom encoder
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Real MCTS data exported to {output_path}")
        logger.info(f"Export contains {len(tree_snapshots)} tree snapshots from {len(self.game_sessions)} games")
        
        return output_path


def main():
    """Main function for real MCTS data collection"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect REAL MCTS data for quantum research")
    parser.add_argument('--n-games', type=int, default=5, help='Number of games to collect')
    parser.add_argument('--game-type', type=str, default='gomoku', help='Type of game')
    parser.add_argument('--board-size', type=int, default=15, help='Board size')
    parser.add_argument('--output', type=str, default='real_mcts_data.json', help='Output file')
    parser.add_argument('--snapshot-freq', type=int, default=5, help='Snapshot frequency')
    parser.add_argument('--evaluator', type=str, default='mock', help='Evaluator type (mock/resnet)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--simulations', type=int, default=800, help='MCTS simulations per move')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("REAL MCTS DATA COLLECTOR FOR QUANTUM RESEARCH")
    print(f"{'='*70}")
    print(f"Collecting data from {args.n_games} real MCTS self-play games")
    print(f"MCTS simulations per move: {args.simulations}")
    print(f"Device: {args.device}")
    print(f"Evaluator: {args.evaluator}")
    print(f"{'='*70}")
    
    # Create collector
    collector = RealMCTSDataCollector(evaluator_type=args.evaluator, device=args.device)
    
    # Override simulation count if specified
    collector.mcts_config.num_simulations = args.simulations
    collector.mcts = MCTS(collector.mcts_config, collector.evaluator)
    collector.mcts.optimize_for_hardware()
    
    # Run collection
    summary = collector.run_data_collection_session(
        n_games=args.n_games,
        game_type=args.game_type,
        board_size=args.board_size,
        snapshot_frequency=args.snapshot_freq
    )
    
    # Export data
    output_path = collector.export_data(args.output)
    
    print(f"\n{'='*70}")
    print("REAL MCTS DATA COLLECTION COMPLETED")
    print(f"{'='*70}")
    print(f"Games collected: {summary['game_statistics']['total_moves']}")
    print(f"Tree snapshots: {summary['tree_statistics']['total_snapshots']}")
    print(f"Total simulations: {summary['game_statistics']['total_simulations']:,}")
    print(f"Average sims/second: {summary['performance_statistics']['avg_sims_per_second']:.0f}")
    print(f"Data exported to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()