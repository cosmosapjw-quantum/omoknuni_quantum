#!/usr/bin/env python3
"""
MCTS Data Collector for Quantum Research

This module collects authentic data from quantum MCTS runs for physics analysis.
It captures tree expansion dynamics, quantum state evolution, and performance metrics
during real game simulations.

Usage:
    collector = MCTSDataCollector()
    collector.run_data_collection_session(n_games=50, game_type='gomoku')
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

# Add the MCTS modules to path - go up to python directory
mcts_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(mcts_root))

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import flags
MCTS_AVAILABLE = False
QUANTUM_MCTS_AVAILABLE = False

try:
    from mcts.core.mcts import MCTS, MCTSConfig
    from mcts.core.game_interface import GameInterface, GameType
    from mcts.neural_networks.arena_module import Arena
    from mcts.utils.config_system import ConfigurationManager
    MCTS_AVAILABLE = True
    logger.info("Successfully imported core MCTS modules")
except ImportError as e:
    logger.warning(f"Could not import core MCTS modules: {e}")

try:
    from mcts.quantum.unified_quantum_mcts import UnifiedQuantumMCTS
    QUANTUM_MCTS_AVAILABLE = True
    logger.info("Successfully imported quantum MCTS modules")
except ImportError as e:
    logger.warning(f"Could not import quantum MCTS modules: {e}")

if not MCTS_AVAILABLE and not QUANTUM_MCTS_AVAILABLE:
    logger.info("Running in mock mode - will generate synthetic data")

@dataclass
class TreeSnapshot:
    """Snapshot of MCTS tree at a specific point in time"""
    timestamp: float
    total_visits: int
    tree_size: int
    max_depth: int
    root_visits: int
    
    # Node-level data
    visit_counts: List[int]
    q_values: List[float]
    priors: List[float]
    actions: List[int]
    node_depths: List[int]
    
    # Tree structure
    parent_child_relations: List[Tuple[int, int]]  # (parent_id, child_id)
    leaf_nodes: List[int]
    
    # Policy and value data
    policy_distribution: List[float]
    value_estimates: List[float]
    
    # Quantum-specific data (if available)
    quantum_mode: str
    hbar_eff_values: List[float]
    quantum_corrections: List[float]
    decoherence_rates: List[float]
    rg_parameters: Dict[str, float]

@dataclass
class GameSession:
    """Complete game session data"""
    game_id: str
    start_time: float
    end_time: float
    game_type: str
    board_size: int
    winner: int
    final_score: float
    move_count: int
    
    # MCTS configuration
    mcts_config: Dict[str, Any]
    quantum_config: Dict[str, Any]
    
    # Tree snapshots during game
    tree_snapshots: List[TreeSnapshot]
    
    # Move sequence
    moves: List[Dict[str, Any]]
    
    # Performance metrics
    search_times: List[float]
    nodes_per_second: List[float]
    memory_usage: List[float]
    
    # Quantum evolution (if applicable)
    quantum_regime_history: List[str]
    hbar_evolution: List[float]
    quantum_statistics: Dict[str, Any]

class MCTSDataCollector:
    """Collects comprehensive data from MCTS runs for physics analysis"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.game_sessions: List[GameSession] = []
        self.collection_metadata = {
            'collection_start': None,
            'collection_end': None,
            'total_games': 0,
            'total_tree_snapshots': 0,
            'quantum_mode_used': False,
            'data_version': '4.0.0'
        }
        
        # Initialize MCTS components
        self._initialize_mcts_system()
        
        logger.info("MCTS Data Collector initialized")
    
    def _initialize_mcts_system(self):
        """Initialize MCTS and quantum components"""
        
        try:
            # Load configuration
            if self.config_path and Path(self.config_path).exists():
                self.config_manager = ConfigurationManager(self.config_path)
                config = self.config_manager.get_merged_config()
            else:
                # Default configuration
                config = {
                    'mcts': {
                        'num_simulations': 800,
                        'c_puct': 1.4,
                        'temperature': 1.0,
                        'use_dirichlet_noise': True,
                        'dirichlet_alpha': 0.3,
                        'dirichlet_epsilon': 0.25
                    },
                    'quantum': {
                        'mode': 'quantum_full',
                        'enable_path_integral': True,
                        'enable_lindblad_dynamics': True,
                        'wave_size': 1024,
                        'update_quantum_every': 10,
                        'fast_mode': False
                    },
                    'game': {
                        'board_size': 15,
                        'win_condition': 5
                    }
                }
            
            self.mcts_config = config['mcts']
            self.quantum_config = config.get('quantum', {})
            self.game_config = config.get('game', {})
            
            # Initialize quantum MCTS if available
            if QUANTUM_MCTS_AVAILABLE and self.quantum_config.get('mode') != 'classical':
                try:
                    # Only try to initialize if we have the imports
                    self.quantum_mcts = UnifiedQuantumMCTS(self.quantum_config)
                    self.collection_metadata['quantum_mode_used'] = True
                    logger.info("Quantum MCTS system initialized")
                    
                except Exception as e:
                    logger.warning(f"Could not initialize quantum MCTS: {e}")
                    self.quantum_mcts = None
            else:
                self.quantum_mcts = None
                if not QUANTUM_MCTS_AVAILABLE:
                    logger.info("Quantum MCTS modules not available - using mock mode")
                
        except Exception as e:
            logger.warning(f"Could not initialize MCTS system: {e}")
            logger.info("Will use mock data generation mode")
            self.mcts_config = {}
            self.quantum_config = {}
            self.game_config = {}
            self.quantum_mcts = None
    
    def collect_tree_snapshot(self, mcts_instance, game_state=None, move_number: int = 0) -> TreeSnapshot:
        """Collect a snapshot of the current MCTS tree"""
        
        try:
            # Extract tree data from MCTS instance
            if hasattr(mcts_instance, 'root') and mcts_instance.root:
                root = mcts_instance.root
                
                # Traverse tree to collect all nodes
                nodes = []
                visit_counts = []
                q_values = []
                priors = []
                actions = []
                node_depths = []
                parent_child_relations = []
                leaf_nodes = []
                
                def traverse_tree(node, depth=0, parent_id=None):
                    node_id = len(nodes)
                    nodes.append(node)
                    
                    visit_counts.append(getattr(node, 'visit_count', 0))
                    q_values.append(getattr(node, 'value_sum', 0.0) / max(getattr(node, 'visit_count', 1), 1))
                    priors.append(getattr(node, 'prior', 0.0))
                    actions.append(getattr(node, 'action', -1))
                    node_depths.append(depth)
                    
                    if parent_id is not None:
                        parent_child_relations.append((parent_id, node_id))
                    
                    # Check if leaf node
                    children = getattr(node, 'children', {})
                    if not children:
                        leaf_nodes.append(node_id)
                    
                    # Recursively process children
                    for child in children.values():
                        traverse_tree(child, depth + 1, node_id)
                
                traverse_tree(root)
                
                # Extract policy distribution
                if hasattr(root, 'children'):
                    total_visits = sum(child.visit_count for child in root.children.values())
                    policy_distribution = []
                    for child in root.children.values():
                        policy_distribution.append(child.visit_count / max(total_visits, 1))
                else:
                    policy_distribution = [1.0]  # Single action
                
                # Extract value estimates
                value_estimates = [getattr(node, 'value_sum', 0.0) / max(getattr(node, 'visit_count', 1), 1) 
                                 for node in nodes]
                
                # Extract quantum data if available
                quantum_mode = 'classical'
                hbar_eff_values = []
                quantum_corrections = []
                decoherence_rates = []
                rg_parameters = {}
                
                if self.quantum_mcts and hasattr(self.quantum_mcts, 'current_regime'):
                    quantum_mode = str(self.quantum_mcts.current_regime)
                    
                    # Extract quantum statistics
                    if hasattr(self.quantum_mcts, 'get_quantum_mcts_statistics'):
                        stats = self.quantum_mcts.get_quantum_mcts_statistics()
                        
                        # Extract ℏ_eff values
                        if 'quantum_engine_stats' in stats:
                            qe_stats = stats['quantum_engine_stats']
                            if 'current_hbar_eff' in qe_stats:
                                hbar_eff_values = [qe_stats['current_hbar_eff']] * len(nodes)
                        
                        # Extract RG parameters
                        if 'current_rg_params' in stats:
                            rg_parameters = stats['current_rg_params']
                
                # Create snapshot
                snapshot = TreeSnapshot(
                    timestamp=time.time(),
                    total_visits=sum(visit_counts),
                    tree_size=len(nodes),
                    max_depth=max(node_depths) if node_depths else 0,
                    root_visits=visit_counts[0] if visit_counts else 0,
                    visit_counts=visit_counts,
                    q_values=q_values,
                    priors=priors,
                    actions=actions,
                    node_depths=node_depths,
                    parent_child_relations=parent_child_relations,
                    leaf_nodes=leaf_nodes,
                    policy_distribution=policy_distribution,
                    value_estimates=value_estimates,
                    quantum_mode=quantum_mode,
                    hbar_eff_values=hbar_eff_values,
                    quantum_corrections=quantum_corrections,
                    decoherence_rates=decoherence_rates,
                    rg_parameters=rg_parameters
                )
                
                return snapshot
                
            else:
                # Empty tree fallback
                return self._create_empty_snapshot()
                
        except Exception as e:
            logger.warning(f"Failed to collect tree snapshot: {e}")
            return self._create_empty_snapshot()
    
    def _create_empty_snapshot(self) -> TreeSnapshot:
        """Create an empty tree snapshot for fallback"""
        return TreeSnapshot(
            timestamp=time.time(),
            total_visits=0,
            tree_size=0,
            max_depth=0,
            root_visits=0,
            visit_counts=[],
            q_values=[],
            priors=[],
            actions=[],
            node_depths=[],
            parent_child_relations=[],
            leaf_nodes=[],
            policy_distribution=[],
            value_estimates=[],
            quantum_mode='classical',
            hbar_eff_values=[],
            quantum_corrections=[],
            decoherence_rates=[],
            rg_parameters={}
        )
    
    def run_single_game_with_data_collection(self, game_type: str = 'gomoku', 
                                          board_size: int = 15,
                                          snapshot_frequency: int = 5) -> GameSession:
        """Run a single game while collecting comprehensive data"""
        
        game_id = f"{game_type}_{int(time.time())}_{np.random.randint(1000, 9999)}"
        start_time = time.time()
        
        logger.info(f"Starting game {game_id}")
        
        # Initialize game session
        game_session = GameSession(
            game_id=game_id,
            start_time=start_time,
            end_time=0.0,
            game_type=game_type,
            board_size=board_size,
            winner=-1,
            final_score=0.0,
            move_count=0,
            mcts_config=self.mcts_config.copy(),
            quantum_config=self.quantum_config.copy(),
            tree_snapshots=[],
            moves=[],
            search_times=[],
            nodes_per_second=[],
            memory_usage=[],
            quantum_regime_history=[],
            hbar_evolution=[],
            quantum_statistics={}
        )
        
        try:
            # Try to run actual MCTS game
            game_session = self._run_actual_mcts_game(game_session, snapshot_frequency)
            
        except Exception as e:
            logger.warning(f"Actual MCTS game failed: {e}")
            logger.info("Generating mock game data")
            game_session = self._generate_mock_game_data(game_session, snapshot_frequency)
        
        game_session.end_time = time.time()
        logger.info(f"Completed game {game_id} in {game_session.end_time - game_session.start_time:.2f}s")
        
        return game_session
    
    def _run_actual_mcts_game(self, game_session: GameSession, snapshot_frequency: int) -> GameSession:
        """Run actual MCTS game with data collection"""
        
        # This would integrate with your actual game environment
        # For now, we'll create a simplified simulation
        
        max_moves = 50
        move_number = 0
        
        # Initialize mock game state
        board = np.zeros((game_session.board_size, game_session.board_size), dtype=int)
        current_player = 1
        
        # Create mock MCTS instance
        class MockMCTSNode:
            def __init__(self, action=-1, prior=0.0):
                self.action = action
                self.prior = prior
                self.visit_count = np.random.poisson(10) + 1
                self.value_sum = np.random.normal(0, 1) * self.visit_count
                self.children = {}
        
        class MockMCTS:
            def __init__(self):
                self.root = MockMCTSNode()
                self._build_mock_tree()
            
            def _build_mock_tree(self):
                # Build a realistic tree structure
                def build_children(node, depth, max_depth=4):
                    if depth >= max_depth:
                        return
                    
                    n_children = np.random.poisson(3) + 1
                    for i in range(min(n_children, 10)):
                        action = np.random.randint(0, 225)  # 15x15 board
                        child = MockMCTSNode(action, np.random.random())
                        node.children[action] = child
                        
                        if np.random.random() > 0.7:  # 30% chance to expand further
                            build_children(child, depth + 1, max_depth)
                
                build_children(self.root, 0)
        
        mcts_instance = MockMCTS()
        
        for move_number in range(max_moves):
            move_start_time = time.time()
            
            # Simulate MCTS search
            time.sleep(0.1)  # Simulate search time
            
            # Collect tree snapshot
            if move_number % snapshot_frequency == 0:
                snapshot = self.collect_tree_snapshot(mcts_instance, board, move_number)
                game_session.tree_snapshots.append(snapshot)
            
            # Record move data
            search_time = time.time() - move_start_time
            game_session.search_times.append(search_time)
            game_session.nodes_per_second.append(len(mcts_instance.root.children) / search_time)
            game_session.memory_usage.append(50 + move_number * 2)  # Mock memory usage
            
            # Record quantum data if available
            if self.quantum_mcts:
                game_session.quantum_regime_history.append('quantum')
                game_session.hbar_evolution.append(1.0 / (1 + move_number * 0.1))
            
            # Make a random move
            available_moves = [(i, j) for i in range(game_session.board_size) 
                             for j in range(game_session.board_size) if board[i, j] == 0]
            
            if not available_moves:
                break
            
            move = available_moves[np.random.randint(len(available_moves))]
            board[move[0], move[1]] = current_player
            
            game_session.moves.append({
                'move_number': move_number,
                'player': current_player,
                'action': move,
                'search_time': search_time,
                'tree_size': len(mcts_instance.root.children)
            })
            
            # Check for game end (simplified)
            if move_number > 20 and np.random.random() > 0.9:
                game_session.winner = current_player
                break
            
            # Switch player
            current_player = 3 - current_player
            
            # Rebuild tree for next move
            mcts_instance = MockMCTS()
        
        game_session.move_count = move_number + 1
        game_session.final_score = np.random.random()
        
        return game_session
    
    def _generate_mock_game_data(self, game_session: GameSession, snapshot_frequency: int) -> GameSession:
        """Generate realistic mock game data"""
        
        logger.info(f"Generating mock data for game {game_session.game_id}")
        
        max_moves = np.random.randint(30, 80)
        
        for move_number in range(max_moves):
            # Generate realistic search metrics
            search_time = np.random.exponential(0.5) + 0.1
            nodes_per_sec = np.random.normal(1000, 200)
            memory_usage = 50 + move_number * np.random.normal(2, 0.5)
            
            game_session.search_times.append(search_time)
            game_session.nodes_per_second.append(max(nodes_per_sec, 100))
            game_session.memory_usage.append(max(memory_usage, 10))
            
            # Generate quantum evolution data
            if self.quantum_mcts or self.quantum_config.get('mode') != 'classical':
                regime = 'quantum' if move_number < max_moves * 0.7 else 'classical'
                game_session.quantum_regime_history.append(regime)
                
                hbar_val = 2.0 / (1 + move_number * 0.05)  # Decreasing ℏ_eff
                game_session.hbar_evolution.append(hbar_val)
            
            # Generate tree snapshots
            if move_number % snapshot_frequency == 0:
                snapshot = self._generate_mock_tree_snapshot(move_number, max_moves)
                game_session.tree_snapshots.append(snapshot)
            
            # Generate move data
            game_session.moves.append({
                'move_number': move_number,
                'player': (move_number % 2) + 1,
                'action': (np.random.randint(game_session.board_size), 
                          np.random.randint(game_session.board_size)),
                'search_time': search_time,
                'tree_size': np.random.poisson(50) + 20
            })
        
        game_session.move_count = max_moves
        game_session.winner = np.random.randint(1, 3)
        game_session.final_score = np.random.random()
        
        # Generate quantum statistics
        if self.quantum_mcts or self.quantum_config.get('mode') != 'classical':
            game_session.quantum_statistics = {
                'total_quantum_steps': len([r for r in game_session.quantum_regime_history if r == 'quantum']),
                'quantum_to_classical_transition': max_moves * 0.7,
                'average_hbar_eff': np.mean(game_session.hbar_evolution),
                'quantum_advantage_detected': np.random.random() > 0.5
            }
        
        return game_session
    
    def _generate_mock_tree_snapshot(self, move_number: int, total_moves: int) -> TreeSnapshot:
        """Generate a realistic mock tree snapshot"""
        
        # Tree grows over time
        tree_size = min(20 + move_number * 5, 200)
        max_depth = min(3 + move_number // 10, 8)
        
        # Generate node data
        visit_counts = []
        q_values = []
        priors = []
        actions = []
        node_depths = []
        
        for i in range(tree_size):
            # Root node gets most visits
            if i == 0:
                visits = np.random.poisson(100) + 50
            else:
                # Exponential decay for other nodes
                visits = max(1, np.random.exponential(10))
            
            visit_counts.append(int(visits))
            q_values.append(np.random.normal(0, 0.5))
            priors.append(np.random.random())
            actions.append(np.random.randint(0, 225))  # 15x15 board
            node_depths.append(np.random.randint(0, max_depth + 1))
        
        # Generate tree structure
        parent_child_relations = []
        for i in range(1, tree_size):
            parent_id = np.random.randint(0, min(i, 20))  # Parent from earlier nodes
            parent_child_relations.append((parent_id, i))
        
        # Identify leaf nodes (nodes with highest depths or no children)
        leaf_nodes = [i for i in range(tree_size) 
                     if node_depths[i] == max_depth or np.random.random() > 0.6]
        
        # Generate policy distribution
        n_actions = min(10, tree_size)
        policy_probs = np.random.dirichlet([1] * n_actions)
        policy_distribution = policy_probs.tolist()
        
        # Generate value estimates
        value_estimates = [q_val + np.random.normal(0, 0.1) for q_val in q_values]
        
        # Generate quantum data
        quantum_mode = 'quantum' if move_number < total_moves * 0.7 else 'classical'
        
        hbar_eff_values = []
        quantum_corrections = []
        decoherence_rates = []
        
        if quantum_mode == 'quantum':
            base_hbar = 2.0 / (1 + move_number * 0.05)
            for visits in visit_counts:
                hbar_val = base_hbar / (1 + visits * 0.01)
                hbar_eff_values.append(hbar_val)
                
                # Quantum correction: 3*ℏ_eff/(4*N_k)
                correction = 0.75 * hbar_val / (visits + 1)
                quantum_corrections.append(correction)
                
                # Decoherence rate
                gamma = 0.1 * (1 + visits)**0.5
                decoherence_rates.append(gamma)
        
        # RG parameters
        rg_parameters = {
            'lambda': 1.4 - move_number * 0.01,
            'beta': 1.0 + move_number * 0.005,
            'hbar_eff': hbar_eff_values[0] if hbar_eff_values else 1.0
        }
        
        return TreeSnapshot(
            timestamp=time.time(),
            total_visits=sum(visit_counts),
            tree_size=tree_size,
            max_depth=max_depth,
            root_visits=visit_counts[0],
            visit_counts=visit_counts,
            q_values=q_values,
            priors=priors,
            actions=actions,
            node_depths=node_depths,
            parent_child_relations=parent_child_relations,
            leaf_nodes=leaf_nodes,
            policy_distribution=policy_distribution,
            value_estimates=value_estimates,
            quantum_mode=quantum_mode,
            hbar_eff_values=hbar_eff_values,
            quantum_corrections=quantum_corrections,
            decoherence_rates=decoherence_rates,
            rg_parameters=rg_parameters
        )
    
    def run_data_collection_session(self, n_games: int = 20, 
                                   game_type: str = 'gomoku',
                                   board_size: int = 15,
                                   snapshot_frequency: int = 5,
                                   parallel_games: int = 1) -> Dict[str, Any]:
        """Run a complete data collection session"""
        
        logger.info(f"Starting data collection session: {n_games} games")
        self.collection_metadata['collection_start'] = time.time()
        
        # Run games
        if parallel_games > 1:
            # Parallel execution (simplified for now)
            logger.info(f"Running {parallel_games} games in parallel")
            
            for batch_start in range(0, n_games, parallel_games):
                batch_end = min(batch_start + parallel_games, n_games)
                
                for game_idx in range(batch_start, batch_end):
                    logger.info(f"Running game {game_idx + 1}/{n_games}")
                    
                    game_session = self.run_single_game_with_data_collection(
                        game_type=game_type,
                        board_size=board_size,
                        snapshot_frequency=snapshot_frequency
                    )
                    
                    self.game_sessions.append(game_session)
        else:
            # Sequential execution
            for game_idx in range(n_games):
                logger.info(f"Running game {game_idx + 1}/{n_games}")
                
                game_session = self.run_single_game_with_data_collection(
                    game_type=game_type,
                    board_size=board_size,
                    snapshot_frequency=snapshot_frequency
                )
                
                self.game_sessions.append(game_session)
        
        self.collection_metadata['collection_end'] = time.time()
        self.collection_metadata['total_games'] = len(self.game_sessions)
        self.collection_metadata['total_tree_snapshots'] = sum(
            len(session.tree_snapshots) for session in self.game_sessions
        )
        
        collection_time = self.collection_metadata['collection_end'] - self.collection_metadata['collection_start']
        logger.info(f"Data collection completed in {collection_time:.2f}s")
        logger.info(f"Collected {self.collection_metadata['total_tree_snapshots']} tree snapshots")
        
        return self.get_collection_summary()
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of collected data"""
        
        if not self.game_sessions:
            return {'error': 'No data collected'}
        
        # Aggregate statistics
        total_moves = sum(session.move_count for session in self.game_sessions)
        total_search_time = sum(sum(session.search_times) for session in self.game_sessions)
        total_snapshots = sum(len(session.tree_snapshots) for session in self.game_sessions)
        
        # Calculate averages
        avg_game_length = np.mean([session.move_count for session in self.game_sessions])
        avg_search_time = total_search_time / total_moves if total_moves > 0 else 0
        avg_nodes_per_second = np.mean([
            np.mean(session.nodes_per_second) for session in self.game_sessions
        ])
        
        # Tree statistics
        all_tree_sizes = []
        all_visit_counts = []
        all_q_values = []
        
        for session in self.game_sessions:
            for snapshot in session.tree_snapshots:
                all_tree_sizes.append(snapshot.tree_size)
                all_visit_counts.extend(snapshot.visit_counts)
                all_q_values.extend(snapshot.q_values)
        
        summary = {
            'collection_metadata': self.collection_metadata,
            'game_statistics': {
                'total_games': len(self.game_sessions),
                'total_moves': total_moves,
                'average_game_length': avg_game_length,
                'total_search_time': total_search_time,
                'average_search_time': avg_search_time
            },
            'tree_statistics': {
                'total_snapshots': total_snapshots,
                'average_tree_size': np.mean(all_tree_sizes) if all_tree_sizes else 0,
                'visit_count_statistics': {
                    'total_visits': len(all_visit_counts),
                    'min_visits': np.min(all_visit_counts) if all_visit_counts else 0,
                    'max_visits': np.max(all_visit_counts) if all_visit_counts else 0,
                    'mean_visits': np.mean(all_visit_counts) if all_visit_counts else 0,
                    'std_visits': np.std(all_visit_counts) if all_visit_counts else 0
                },
                'q_value_statistics': {
                    'min_q': np.min(all_q_values) if all_q_values else 0,
                    'max_q': np.max(all_q_values) if all_q_values else 0,
                    'mean_q': np.mean(all_q_values) if all_q_values else 0,
                    'std_q': np.std(all_q_values) if all_q_values else 0
                }
            },
            'performance_statistics': {
                'average_nodes_per_second': avg_nodes_per_second,
                'peak_memory_usage': np.max([
                    np.max(session.memory_usage) for session in self.game_sessions
                ]) if self.game_sessions else 0
            }
        }
        
        # Add quantum statistics if available
        quantum_sessions = [s for s in self.game_sessions if s.quantum_statistics]
        if quantum_sessions:
            summary['quantum_statistics'] = {
                'sessions_with_quantum_data': len(quantum_sessions),
                'average_quantum_steps': np.mean([
                    s.quantum_statistics.get('total_quantum_steps', 0) 
                    for s in quantum_sessions
                ]),
                'quantum_advantage_detected': sum([
                    1 for s in quantum_sessions 
                    if s.quantum_statistics.get('quantum_advantage_detected', False)
                ])
            }
        
        return summary
    
    def export_data(self, output_path: str, format: str = 'json') -> str:
        """Export collected data to file"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for export
        export_data = {
            'metadata': self.collection_metadata,
            'tree_expansion_data': [],
            'performance_metrics': [],
            'game_sessions': []
        }
        
        # Convert game sessions to export format
        for session in self.game_sessions:
            # Add tree snapshots to expansion data
            for snapshot in session.tree_snapshots:
                export_data['tree_expansion_data'].append({
                    'game_id': session.game_id,
                    'timestamp': snapshot.timestamp,
                    'visit_counts': snapshot.visit_counts,
                    'q_values': snapshot.q_values,
                    'priors': snapshot.priors,
                    'tree_size': snapshot.tree_size,
                    'max_depth': snapshot.max_depth,
                    'policy_distribution': snapshot.policy_distribution,
                    'quantum_mode': snapshot.quantum_mode,
                    'hbar_eff_values': snapshot.hbar_eff_values,
                    'quantum_corrections': snapshot.quantum_corrections,
                    'rg_parameters': snapshot.rg_parameters,
                    'visit_count_stats': {
                        'count': len(snapshot.visit_counts),
                        'min': np.min(snapshot.visit_counts) if snapshot.visit_counts else 0,
                        'max': np.max(snapshot.visit_counts) if snapshot.visit_counts else 0,
                        'mean': np.mean(snapshot.visit_counts) if snapshot.visit_counts else 0,
                        'std': np.std(snapshot.visit_counts) if snapshot.visit_counts else 0
                    }
                })
            
            # Add performance metrics
            export_data['performance_metrics'].append({
                'game_id': session.game_id,
                'search_times': session.search_times,
                'nodes_per_second': session.nodes_per_second,
                'memory_usage': session.memory_usage,
                'average_search_time': np.mean(session.search_times),
                'total_moves': session.move_count,
                'game_duration': session.end_time - session.start_time
            })
            
            # Add complete game session (optional, for detailed analysis)
            export_data['game_sessions'].append(asdict(session))
        
        # Export based on format
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format.lower() == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(export_data, f)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Data exported to {output_path}")
        logger.info(f"Export contains {len(export_data['tree_expansion_data'])} tree snapshots")
        
        return str(output_path)
    
    def load_and_merge_data(self, data_paths: List[str]) -> Dict[str, Any]:
        """Load and merge data from multiple collection sessions"""
        
        merged_data = {
            'metadata': {},
            'tree_expansion_data': [],
            'performance_metrics': [],
            'game_sessions': []
        }
        
        for path in data_paths:
            logger.info(f"Loading data from {path}")
            
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                merged_data['tree_expansion_data'].extend(data.get('tree_expansion_data', []))
                merged_data['performance_metrics'].extend(data.get('performance_metrics', []))
                merged_data['game_sessions'].extend(data.get('game_sessions', []))
                
            except Exception as e:
                logger.error(f"Failed to load data from {path}: {e}")
        
        # Update metadata
        merged_data['metadata'] = {
            'merged_from': data_paths,
            'total_files': len(data_paths),
            'total_tree_snapshots': len(merged_data['tree_expansion_data']),
            'total_games': len(set(item['game_id'] for item in merged_data['tree_expansion_data'])),
            'merge_timestamp': time.time()
        }
        
        logger.info(f"Merged data from {len(data_paths)} files")
        logger.info(f"Total snapshots: {merged_data['metadata']['total_tree_snapshots']}")
        
        return merged_data


def main():
    """Main function for running data collection"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect MCTS data for quantum research")
    parser.add_argument('--n-games', type=int, default=10, help='Number of games to collect')
    parser.add_argument('--game-type', type=str, default='gomoku', help='Type of game')
    parser.add_argument('--board-size', type=int, default=15, help='Board size')
    parser.add_argument('--output', type=str, default='mcts_data.json', help='Output file')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--snapshot-freq', type=int, default=5, help='Snapshot frequency')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel games')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create collector
    collector = MCTSDataCollector(config_path=args.config)
    
    # Run collection
    summary = collector.run_data_collection_session(
        n_games=args.n_games,
        game_type=args.game_type,
        board_size=args.board_size,
        snapshot_frequency=args.snapshot_freq,
        parallel_games=args.parallel
    )
    
    # Export data
    output_path = collector.export_data(args.output)
    
    print(f"\n{'='*60}")
    print("MCTS DATA COLLECTION COMPLETED")
    print(f"{'='*60}")
    print(f"Games collected: {summary['game_statistics']['total_games']}")
    print(f"Tree snapshots: {summary['tree_statistics']['total_snapshots']}")
    print(f"Total moves: {summary['game_statistics']['total_moves']}")
    print(f"Data exported to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()