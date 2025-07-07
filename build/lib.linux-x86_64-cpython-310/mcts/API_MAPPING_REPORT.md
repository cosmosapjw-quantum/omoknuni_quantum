# MCTS Python Module API Mapping Report

This report provides a comprehensive mapping of all classes, functions, and dataclasses defined in the `/home/cosmosapjw/omoknuni_quantum/python/mcts` directory.

## Directory Structure
```
mcts/
├── core/
├── gpu/
├── neural_networks/
└── utils/
```

## Module: mcts/core/

### mcts/core/__init__.py
**Module Docstring:** "Core MCTS components"

**Exported Components:**
- `GameInterface` (from game_interface)
- `GameType` (from game_interface)
- `Evaluator` (from evaluator - lazy import)
- `EvaluatorConfig` (from evaluator - lazy import)
- `AlphaZeroEvaluator` (from evaluator - lazy import)
- `MCTS` (from mcts - lazy import)
- `MCTSConfig` (from mcts_config - lazy import)
- `WaveSearch` (from wave_search - lazy import)
- `TreeOperations` (from tree_operations - lazy import)

### mcts/core/game_interface.py
**Module Docstring:** "GameInterface wrapper for C++ game states"

**Classes:**
1. **GameType** (Enum)
   - Values: CHESS, GO, GOMOKU

2. **GameInterface**
   - `__init__(self, game_type: GameType, board_size: Optional[int] = None, input_representation: str = 'basic', **kwargs)`
   - Key methods:
     - `create_initial_state() -> Any`
     - `state_to_numpy(state: Any, representation_type: str = None) -> np.ndarray`
     - `state_to_tensor(state: Any, representation_type: str = None) -> torch.Tensor`
     - `tensor_to_state(tensor: torch.Tensor) -> Any`
     - `get_state_shape() -> tuple`
     - `batch_state_to_numpy(states: List[Any]) -> np.ndarray`
     - `get_legal_moves(state: Any, shuffle: bool = True) -> List[int]`
     - `batch_get_legal_moves(states: List[Any], shuffle: bool = True) -> List[List[int]]`
     - `apply_move(state: Any, move: int) -> Any`
     - `is_terminal(state: Any) -> bool`
     - `get_winner(state: Any) -> int`
     - `get_current_player(state: Any) -> int`
     - `move_to_action_index(move: int) -> int`
     - `action_index_to_move(action_index: int) -> int`
     - `get_action_probabilities_mask(state: Any) -> np.ndarray`
     - `get_symmetries(board: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]`
     - `encode_for_nn(state: Any, history: List[Any]) -> np.ndarray`
     - `get_hash(state: Any) -> int`
     - `reset() -> None`
     - `get_nn_input() -> np.ndarray`
     - `get_state() -> Any`
     - `make_move(action: int) -> None`
     - `get_reward() -> float`
     - `is_legal_move(state: Any, move: int) -> bool`
     - `undo_move(state: Any) -> None`
     - `get_game_result(state: Any) -> str`
     - `get_board_size(state: Any) -> int`
     - `get_action_space_size(state: Any) -> int`
     - `get_tensor_representation(state: Any) -> np.ndarray`
     - `get_enhanced_tensor_representation(state: Any) -> np.ndarray`
     - `action_to_string(state: Any, action: int) -> str`
     - `string_to_action(state: Any, move_str: str) -> int`
     - `to_string(state: Any) -> str`
     - `get_move_history(state: Any) -> List[int]`
     - `clone_state(state: Any) -> Any`
     - `get_canonical_form(state: Any) -> np.ndarray`
     - `get_next_state(state: Any, action: int) -> Any`
     - `get_value(state: Any) -> float`

### mcts/core/attack_defense.py
**Module Docstring:** "Attack/Defense scoring module for board games"

**Functions:**
1. **compute_attack_defense_scores**(game_type: str, board: np.ndarray, current_player: int) -> Tuple[np.ndarray, np.ndarray]
2. **compute_gomoku_attack_defense**(board: np.ndarray, current_player: int) -> Tuple[np.ndarray, np.ndarray]
3. **evaluate_line_potential**(board: np.ndarray, row: int, col: int, di: int, dj: int, stone_type: int) -> float
4. **compute_chess_attack_defense**(board: np.ndarray, current_player: int) -> Tuple[np.ndarray, np.ndarray]
5. **compute_go_attack_defense**(board: np.ndarray, current_player: int) -> Tuple[np.ndarray, np.ndarray]
6. **count_liberties**(board: np.ndarray, row: int, col: int) -> int
7. **batch_compute_attack_defense_scores**(game_type: str, boards: np.ndarray, current_players: np.ndarray) -> Tuple[np.ndarray, np.ndarray]

### mcts/core/evaluator.py
**Module Docstring:** "Neural network evaluator interface for MCTS"

**Dataclasses:**
1. **EvaluatorConfig**
   - Fields:
     - `batch_size: int = 64`
     - `device: str = 'cuda' if torch.cuda.is_available() else 'cpu'`
     - `timeout: float = 1.0`
     - `enable_caching: bool = False`
     - `cache_size: int = 10000`

**Classes:**
1. **Evaluator** (ABC)
   - `__init__(self, config: EvaluatorConfig, action_size: int)`
   - Abstract methods:
     - `evaluate(state: np.ndarray, legal_mask: Optional[np.ndarray] = None, temperature: float = 1.0) -> Tuple[np.ndarray, float]`
     - `evaluate_batch(states: np.ndarray, legal_masks: Optional[np.ndarray] = None, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]`
   - Methods:
     - `get_stats() -> Dict`
     - `reset_stats()`
     - `warmup(dummy_state: np.ndarray, num_iterations: int = 10)`

2. **AlphaZeroEvaluator** (Evaluator)
   - `__init__(self, model, config: Optional[EvaluatorConfig] = None, device: Optional[str] = None, action_size: Optional[int] = None)`
   - Implements abstract methods from Evaluator

3. **RandomEvaluator** (Evaluator)
   - `__init__(self, action_size: int = 225, config: Optional[EvaluatorConfig] = None)`
   - Implements abstract methods from Evaluator

### mcts/core/mcts_config.py
**Module Docstring:** "MCTS Configuration classes and utilities"

**Dataclasses:**
1. **MCTSConfig**
   - Fields:
     - `num_simulations: int = 10000`
     - `c_puct: float = 1.414`
     - `temperature: float = 1.0`
     - `dirichlet_alpha: float = 0.3`
     - `dirichlet_epsilon: float = 0.25`
     - `classical_only_mode: bool = False`
     - `enable_fast_ucb: bool = True`
     - `wave_size: Optional[int] = None`
     - `min_wave_size: int = 3072`
     - `max_wave_size: int = 3072`
     - `device: str = 'cuda'`
     - `game_type: Union[GameType, LegacyGameType] = GameType.GOMOKU`
     - `board_size: int = 15`
     - `enable_quantum: bool = False`
     - `enable_virtual_loss: bool = True`
     - `virtual_loss: float = 1.0`
     - `memory_pool_size_mb: int = 2048`
     - `max_tree_nodes: int = 500000`
     - `use_mixed_precision: bool = True`
     - `use_cuda_graphs: bool = True`
     - `use_tensor_cores: bool = True`
     - `initial_children_per_expansion: int = 8`
     - `max_children_per_node: int = 50`
     - `progressive_expansion_threshold: int = 5`
     - `progressive_widening_alpha: float = 0.5`
     - `progressive_widening_base: float = 10.0`
     - `target_sims_per_second: int = 100000`
     - `cache_legal_moves: bool = True`
     - `cache_features: bool = True`
     - `use_zobrist_hashing: bool = True`
     - `tree_batch_size: int = 1024`
     - `enable_subtree_reuse: bool = True`
     - `subtree_reuse_min_visits: int = 10`
     - `enable_debug_logging: bool = False`
     - `enable_state_pool_debug: bool = False`
     - `profile_gpu_kernels: bool = False`
   - Methods:
     - `get_or_create_quantum_config()`
     - `_estimate_branching_factor() -> int`
     - `_estimate_game_length() -> int`

**Functions:**
1. **create_optimized_config**(game_type, num_simulations=10000, device='cuda', enable_quantum=False, **kwargs) -> MCTSConfig
2. **create_performance_config**(wave_size=3072, memory_pool_mb=2048, **kwargs) -> MCTSConfig

### mcts/core/mcts.py
**Module Docstring:** "High-performance MCTS implementation"

**Classes:**
1. **MCTS**
   - `__init__(self, config: MCTSConfig, evaluator: Any, game_interface: Optional[GameInterface] = None)`
   - Key methods:
     - `search(state: Any, num_simulations: Optional[int] = None) -> np.ndarray`
     - `get_action_probs(state: Any, temperature: float = 1.0) -> np.ndarray`
     - `update_with_move(move: int)`
     - `reset_tree()`
     - `get_stats() -> Dict[str, Any]`
     - `reset_stats()`
     - `extract_training_data() -> List[Tuple[np.ndarray, np.ndarray, float]]`
     - `get_root_values() -> Dict[int, float]`
     - `set_root(state: Any)`
     - `get_pv() -> List[int]`
     - `ponder(time_limit: float = 10.0) -> int`
     - `get_mcts_probs(state: Any, temp: float = 1.0) -> np.ndarray`
     - `_get_evaluator_input_channels() -> int`

### mcts/core/wave_search.py
**Module Docstring:** "Wave-based search module for high-performance MCTS"

**Classes:**
1. **WaveSearch**
   - `__init__(self, tree: CSRTree, game_states: GPUGameStates, evaluator: Any, config: MCTSConfig, device: torch.device, gpu_ops: Optional[Any] = None)`
   - Key methods:
     - `run_wave(wave_size: int, node_to_state: torch.Tensor, state_pool_free_list: List[int]) -> int`

### mcts/core/tree_operations.py
**Module Docstring:** "Tree operations module for MCTS"

**Classes:**
1. **TreeOperations**
   - `__init__(self, tree: CSRTree, config: MCTSConfig, device: torch.device)`
   - Key methods:
     - `get_root_children_info() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
     - `get_best_child(node_idx: int) -> int`
     - `extract_principal_variation(max_depth: int = 10) -> List[int]`
     - `get_node_info(node_idx: int) -> Dict[str, Any]`
     - `validate_tree_consistency() -> bool`

## Module: mcts/gpu/

### mcts/gpu/__init__.py
**Module Docstring:** "GPU acceleration components for MCTS"

**Exported Components:**
- `CSRTree`
- `CSRTreeConfig`
- `GPUGameStates`
- `GPUGameStatesConfig`
- `GameType`
- `get_mcts_gpu_accelerator`
- `MCTSGPUAccelerator`
- `CSRGPUKernels`
- `OptimizedCSRKernels`
- `StateMemoryPool`
- `UCBSelector`

### mcts/gpu/csr_tree.py
**Module Docstring:** "CSR (Compressed Sparse Row) tree implementation for GPU-accelerated MCTS"

**Dataclasses:**
1. **CSRTreeConfig**
   - Fields:
     - `max_nodes: int = 100000`
     - `max_edges: int = 5000000`
     - `max_actions: int = 512`
     - `device: str = 'cuda'`
     - `enable_virtual_loss: bool = True`
     - `virtual_loss_value: float = -1.0`
     - `batch_size: int = 3072`
     - `enable_batched_ops: bool = True`

**Classes:**
1. **CSRTree**
   - `__init__(self, config: CSRTreeConfig)`
   - Key attributes:
     - `node_data: NodeData`
     - `edge_data: EdgeData`
   - Key methods:
     - `add_node(parent_idx: int = -1) -> int`
     - `add_child(parent_idx: int, action: int, prior: float) -> int`
     - `batch_add_children(parent_indices: torch.Tensor, actions: torch.Tensor, priors: torch.Tensor) -> torch.Tensor`
     - `get_children(node_idx: int) -> Tuple[torch.Tensor, torch.Tensor]`
     - `get_child_by_action(node_idx: int, action: int) -> int`
     - `batch_get_children(node_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
     - `update_node(node_idx: int, value: float)`
     - `batch_update_nodes(node_indices: torch.Tensor, values: torch.Tensor)`
     - `apply_virtual_loss(node_indices: torch.Tensor)`
     - `revert_virtual_loss(node_indices: torch.Tensor, values: torch.Tensor)`
     - `reset()`
     - `validate() -> bool`
     - `get_statistics() -> Dict[str, Any]`

### mcts/gpu/gpu_game_states.py
**Module Docstring:** "GPU-accelerated game state management"

**Enums:**
1. **GameType**
   - Values: CHESS = 0, GO = 1, GOMOKU = 2

**Dataclasses:**
1. **GPUGameStatesConfig**
   - Fields:
     - `capacity: int = 100000`
     - `game_type: GameType = GameType.GOMOKU`
     - `board_size: int = 15`
     - `device: str = 'cuda'`
     - `use_pinned_memory: bool = True`
     - `num_history_planes: int = 8`
     - `enable_enhanced_features: bool = True`

**Classes:**
1. **GPUGameStates**
   - `__init__(self, config: GPUGameStatesConfig)`
   - Key methods:
     - `initialize_root(state: Any) -> int`
     - `apply_moves_batch(state_indices: torch.Tensor, actions: torch.Tensor) -> torch.Tensor`
     - `get_legal_moves_mask(state_indices: torch.Tensor) -> torch.Tensor`
     - `is_terminal_batch(state_indices: torch.Tensor) -> torch.Tensor`
     - `get_winner_batch(state_indices: torch.Tensor) -> torch.Tensor`
     - `get_nn_input_batch(state_indices: torch.Tensor) -> torch.Tensor`
     - `clone_states(src_indices: torch.Tensor, dst_indices: torch.Tensor)`
     - `enable_enhanced_features()`
     - `set_enhanced_channels(num_channels: int)`
     - `update_history_batch(state_indices: torch.Tensor, parent_indices: torch.Tensor, actions: torch.Tensor)`
     - `get_game_specific_features(state_indices: torch.Tensor) -> torch.Tensor`
     - `compute_zobrist_hash_batch(state_indices: torch.Tensor) -> torch.Tensor`
     - `undo_moves_batch(state_indices: torch.Tensor)`
     - `get_symmetries_batch(state_indices: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]`
     - `reset()`
     - `get_statistics() -> Dict[str, Any]`

### mcts/gpu/mcts_gpu_accelerator.py
**Module Docstring:** "Unified GPU acceleration for MCTS operations"

**Classes:**
1. **MCTSGPUAccelerator**
   - `__init__(self, device: torch.device)`
   - Key methods:
     - `select_batch_puct(visit_counts: torch.Tensor, value_sums: torch.Tensor, priors: torch.Tensor, parent_visits: torch.Tensor, edges_from: torch.Tensor, edge_counts: torch.Tensor, c_puct: float, max_batch_size: int) -> torch.Tensor`
     - `backup_batch(node_indices: torch.Tensor, values: torch.Tensor, visit_counts: torch.Tensor, value_sums: torch.Tensor)`
     - `expand_batch(parent_indices: torch.Tensor, legal_moves_masks: torch.Tensor, priors: torch.Tensor, edges_from: torch.Tensor, edges_to: torch.Tensor, edges_action: torch.Tensor, edges_prior: torch.Tensor, edge_counts: torch.Tensor, num_nodes: int, num_edges: int, max_children_per_node: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]`
     - `compute_root_policy(edges_from: torch.Tensor, edges_action: torch.Tensor, edge_counts: torch.Tensor, visit_counts: torch.Tensor, temperature: float, num_actions: int) -> torch.Tensor`
     - `find_leaf_paths_batch(node_indices: torch.Tensor, edges_from: torch.Tensor, edges_to: torch.Tensor, edge_counts: torch.Tensor, visit_counts: torch.Tensor, value_sums: torch.Tensor, priors: torch.Tensor, c_puct: float, max_depth: int) -> Tuple[torch.Tensor, torch.Tensor]`
     - `apply_dirichlet_noise_batch(edges_from: torch.Tensor, edges_prior: torch.Tensor, edge_counts: torch.Tensor, parent_indices: torch.Tensor, alpha: float, epsilon: float)`

**Functions:**
1. **get_mcts_gpu_accelerator**(device: torch.device) -> MCTSGPUAccelerator

### mcts/gpu/csr_gpu_kernels.py
**Module Docstring:** "GPU kernels for CSR tree operations"

**Classes:**
1. **CSRGPUKernels**
   - `__init__(self, device: torch.device = None)`
   - Key methods: (Similar to MCTSGPUAccelerator methods)

### mcts/gpu/csr_storage.py
**Module Docstring:** "CSR storage structures for GPU tree"

**Dataclasses:**
1. **NodeData**
   - Fields:
     - `visit_counts: torch.Tensor`
     - `value_sums: torch.Tensor`
     - `virtual_losses: torch.Tensor`
     - `is_expanded: torch.Tensor`

2. **EdgeData**
   - Fields:
     - `edges_from: torch.Tensor`
     - `edges_to: torch.Tensor`
     - `edges_action: torch.Tensor`
     - `edges_prior: torch.Tensor`
     - `edge_counts: torch.Tensor`

### mcts/gpu/memory_pool.py
**Module Docstring:** "Memory pool management for GPU states"

**Classes:**
1. **StateMemoryPool**
   - `__init__(self, capacity: int, state_shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype = torch.float32)`
   - Key methods:
     - `allocate() -> int`
     - `allocate_batch(count: int) -> torch.Tensor`
     - `deallocate(idx: int)`
     - `deallocate_batch(indices: torch.Tensor)`
     - `get_state(idx: int) -> torch.Tensor`
     - `set_state(idx: int, state: torch.Tensor)`
     - `batch_get_states(indices: torch.Tensor) -> torch.Tensor`
     - `batch_set_states(indices: torch.Tensor, states: torch.Tensor)`
     - `reset()`
     - `get_statistics() -> Dict[str, Any]`

### mcts/gpu/node_data_manager.py
**Module Docstring:** "Node data management for GPU tree"

**Classes:**
1. **NodeDataManager**
   - `__init__(self, max_nodes: int, device: torch.device)`
   - Key methods:
     - `allocate_node() -> int`
     - `allocate_nodes_batch(count: int) -> torch.Tensor`
     - `deallocate_node(idx: int)`
     - `update_visit_count(idx: int, increment: int = 1)`
     - `update_value_sum(idx: int, value: float)`
     - `batch_update_stats(indices: torch.Tensor, values: torch.Tensor)`
     - `get_node_stats(idx: int) -> Dict[str, float]`
     - `reset()`

### mcts/gpu/ucb_selector.py
**Module Docstring:** "UCB selection for MCTS on GPU"

**Classes:**
1. **UCBSelector**
   - `__init__(self, c_puct: float = 1.414, device: torch.device = None)`
   - Key methods:
     - `select_child(parent_idx: int, edges_from: torch.Tensor, edges_to: torch.Tensor, edge_counts: torch.Tensor, visit_counts: torch.Tensor, value_sums: torch.Tensor, priors: torch.Tensor) -> int`
     - `batch_select_children(parent_indices: torch.Tensor, edges_from: torch.Tensor, edges_to: torch.Tensor, edge_counts: torch.Tensor, visit_counts: torch.Tensor, value_sums: torch.Tensor, priors: torch.Tensor) -> torch.Tensor`
     - `compute_ucb_scores(parent_visits: torch.Tensor, child_visits: torch.Tensor, child_values: torch.Tensor, priors: torch.Tensor) -> torch.Tensor`

### mcts/gpu/gpu_attack_defense.py
**Module Docstring:** "GPU-accelerated attack/defense computation"

**Classes:**
1. **GPUAttackDefenseComputer**
   - `__init__(self, board_size: int, device: torch.device, max_batch_size: int = 1024)`
   - Key methods:
     - `compute_batch(board_batch: torch.Tensor, current_players: torch.Tensor, game_type: str) -> Tuple[torch.Tensor, torch.Tensor]`
     - `compute_gomoku_scores(board_batch: torch.Tensor, current_players: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
     - `compute_go_scores(board_batch: torch.Tensor, current_players: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
     - `compute_chess_scores(board_batch: torch.Tensor, current_players: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`

## Module: mcts/neural_networks/

### mcts/neural_networks/__init__.py
**Module Docstring:** "Neural network components for MCTS"

**Exported Components:**
- `BaseNeuralEvaluator`
- `ResNetEvaluator`
- `ResNetModel`
- `ResNetConfig`
- `AlphaZeroNetwork`
- `BaseGameModel`
- `ModelMetadata`
- `create_model`
- `load_model`
- `save_model`
- `convert_to_onnx`
- `TensorRTConverter`
- `TensorRTEvaluator`
- `ArenaModule`
- `SelfPlayModule`
- `UnifiedTrainingPipeline`

### mcts/neural_networks/base_neural_evaluator.py
**Module Docstring:** "Base class for neural network evaluators"

**Classes:**
1. **BaseNeuralEvaluator** (ABC)
   - `__init__(self)`
   - Abstract methods:
     - `evaluate(state: np.ndarray) -> Tuple[np.ndarray, float]`
     - `evaluate_batch(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
     - `get_model_info() -> Dict[str, Any]`
     - `save_checkpoint(path: str)`
     - `load_checkpoint(path: str)`

### mcts/neural_networks/resnet_model.py
**Module Docstring:** "ResNet Model for AlphaZero"

**Dataclasses:**
1. **ResNetConfig**
   - Fields:
     - `num_blocks: int = 19`
     - `num_filters: int = 256`
     - `input_channels: int = 20`
     - `fc_value_hidden: int = 256`
     - `fc_policy_hidden: int = 256`

**Classes:**
1. **ResidualBlock** (nn.Module)
   - `__init__(self, num_filters: int)`
   - `forward(x: torch.Tensor) -> torch.Tensor`

2. **PolicyHead** (nn.Module)
   - `__init__(self, num_filters: int, board_size: int, num_actions: int, fc_hidden: int = 256)`
   - `forward(x: torch.Tensor) -> torch.Tensor`

3. **ValueHead** (nn.Module)
   - `__init__(self, num_filters: int, board_size: int, fc_hidden: int = 256)`
   - `forward(x: torch.Tensor) -> torch.Tensor`

4. **ResNetModel** (BaseGameModel)
   - `__init__(self, config: ResNetConfig, board_size: int, num_actions: int, game_type: str)`
   - `forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
   - `get_config() -> ResNetConfig`

5. **AlphaZeroNetwork** (ResNetModel) - Alias for backward compatibility

### mcts/neural_networks/resnet_evaluator.py
**Module Docstring:** "ResNet-based neural network evaluator for MCTS"

**Classes:**
1. **ResNetEvaluator** (BaseNeuralEvaluator)
   - `__init__(self, model_path: Optional[str] = None, config: Optional[ResNetConfig] = None, device: str = 'cuda', board_size: int = 15, num_actions: int = 225, game_type: str = 'gomoku')`
   - Implements abstract methods from BaseNeuralEvaluator
   - Additional methods:
     - `enable_tensorrt(batch_sizes: List[int] = [1, 8, 16, 32, 64])`
     - `set_eval_mode()`
     - `set_train_mode()`

### mcts/neural_networks/nn_framework.py
**Module Docstring:** "Neural network framework utilities"

**Dataclasses:**
1. **ModelMetadata**
   - Fields:
     - `game_type: str`
     - `board_size: int`
     - `num_actions: int`
     - `input_channels: int`
     - `model_version: str = "1.0"`
     - `training_steps: int = 0`
     - `creation_date: str = ""`
     - `additional_info: Dict[str, Any] = None`

**Classes:**
1. **BaseGameModel** (nn.Module, ABC)
   - `__init__(self)`
   - Abstract methods:
     - `forward(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`
   - Methods:
     - `get_metadata() -> ModelMetadata`
     - `set_metadata(metadata: ModelMetadata)`

**Functions:**
1. **create_model**(model_type: str, **kwargs) -> BaseGameModel
2. **load_model**(path: str, device: str = 'cuda') -> BaseGameModel
3. **save_model**(model: BaseGameModel, path: str, metadata: Optional[Dict] = None)
4. **convert_to_onnx**(model: BaseGameModel, output_path: str, batch_size: int = 1)
5. **autocast_forward**(model: nn.Module, x: torch.Tensor, enabled: bool = True) -> Tuple[torch.Tensor, torch.Tensor]

### mcts/neural_networks/tensorrt_converter.py
**Module Docstring:** "TensorRT conversion utilities"

**Classes:**
1. **TensorRTConverter**
   - `__init__(self, workspace_size_mb: int = 2048, fp16_mode: bool = True, int8_mode: bool = False)`
   - Key methods:
     - `convert_model(model: nn.Module, input_shape: Tuple[int, ...], output_path: str, batch_sizes: List[int] = [1, 8, 16, 32, 64])`
     - `optimize_onnx(onnx_path: str, optimized_path: str)`
     - `build_engine(onnx_path: str, engine_path: str, batch_sizes: List[int])`
     - `profile_engine(engine_path: str, input_shape: Tuple[int, ...])`

### mcts/neural_networks/tensorrt_evaluator.py
**Module Docstring:** "TensorRT-accelerated evaluator"

**Classes:**
1. **TensorRTEvaluator** (BaseNeuralEvaluator)
   - `__init__(self, engine_path: str, device: str = 'cuda', board_size: int = 15, num_actions: int = 225)`
   - Implements abstract methods from BaseNeuralEvaluator
   - Additional methods:
     - `warmup(num_iterations: int = 10)`
     - `profile_performance(batch_sizes: List[int] = [1, 8, 16, 32, 64])`

### mcts/neural_networks/arena_module.py
**Module Docstring:** "Arena module for model comparison"

**Classes:**
1. **ArenaModule**
   - `__init__(self, game_interface: GameInterface, num_games: int = 100, simulations_per_move: int = 800, temperature: float = 0.1)`
   - Key methods:
     - `play_match(player1_evaluator: BaseNeuralEvaluator, player2_evaluator: BaseNeuralEvaluator) -> Dict[str, Any]`
     - `run_tournament(evaluators: List[Tuple[str, BaseNeuralEvaluator]]) -> pd.DataFrame`
     - `compare_models(model1_path: str, model2_path: str) -> Dict[str, Any]`

### mcts/neural_networks/self_play_module.py
**Module Docstring:** "Self-play module for training data generation"

**Classes:**
1. **SelfPlayModule**
   - `__init__(self, game_interface: GameInterface, evaluator: BaseNeuralEvaluator, config: Dict[str, Any])`
   - Key methods:
     - `run_self_play_game() -> List[Tuple[np.ndarray, np.ndarray, float]]`
     - `generate_training_data(num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]`
     - `save_training_data(data: List, path: str)`
     - `load_training_data(path: str) -> List`

### mcts/neural_networks/unified_training_pipeline.py
**Module Docstring:** "Unified training pipeline for AlphaZero"

**Classes:**
1. **UnifiedTrainingPipeline**
   - `__init__(self, config: Dict[str, Any], game_interface: GameInterface)`
   - Key methods:
     - `train(num_iterations: int)`
     - `self_play_phase(num_games: int) -> List[Tuple]`
     - `training_phase(training_data: List[Tuple])`
     - `evaluation_phase() -> bool`
     - `save_checkpoint(iteration: int)`
     - `load_checkpoint(iteration: int)`
     - `get_latest_model() -> BaseGameModel`

## Module: mcts/utils/

### mcts/utils/__init__.py
**Module Docstring:** "Utility modules for MCTS"

**Exported Components:**
- `BatchEvaluationCoordinator`
- `RequestBatchingCoordinator`
- `OptimizedRemoteEvaluator`
- `GPUEvaluatorService`
- `ConfigSystem`
- `load_config`
- `save_config`
- `get_default_config`
- `TensorRTManager`
- `SafeProcessManager`
- `setup_cuda_visible_devices`
- `cleanup_processes`
- `worker_init`
- `validate_config`
- `validate_model`
- `validate_tree_consistency`

### mcts/utils/batch_evaluation_coordinator.py
**Module Docstring:** "Batch Evaluation Coordinator for Phase 1 Optimization"

**Dataclasses:**
1. **BatchEvaluationRequest**
   - Fields:
     - `request_id: int`
     - `states: np.ndarray`
     - `legal_masks: Optional[np.ndarray]`
     - `temperatures: np.ndarray`
     - `worker_id: int`
     - `timestamp: float`
     - `individual_request_ids: List[int]`

2. **BatchEvaluationResponse**
   - Fields:
     - `request_id: int`
     - `policies: np.ndarray`
     - `values: np.ndarray`
     - `worker_id: int`
     - `individual_request_ids: List[int]`

**Classes:**
1. **RequestBatchingCoordinator**
   - `__init__(self, max_batch_size: int = 64, batch_timeout_ms: float = 100.0, enable_cross_worker_batching: bool = True)`
   - Key methods:
     - `start()`
     - `stop()`
     - `submit_request(states: np.ndarray, legal_masks: Optional[np.ndarray], temperatures: np.ndarray, worker_id: int) -> Tuple[np.ndarray, np.ndarray]`
     - `process_batch(batch_request: BatchEvaluationRequest) -> BatchEvaluationResponse`
     - `get_stats() -> Dict[str, Any]`

2. **BatchEvaluationCoordinator** - Alias for RequestBatchingCoordinator

### mcts/utils/optimized_remote_evaluator.py
**Module Docstring:** "Optimized Remote Evaluator - Phase 1 Performance Fix"

**Classes:**
1. **OptimizedRemoteEvaluator**
   - `__init__(self, gpu_service, coordinator: Optional[RequestBatchingCoordinator] = None, worker_id: int = 0, device: str = 'cuda')`
   - Key methods:
     - `evaluate(state: np.ndarray, legal_mask: Optional[np.ndarray] = None, temperature: float = 1.0) -> Tuple[np.ndarray, float]`
     - `evaluate_batch(states: np.ndarray, legal_masks: Optional[np.ndarray] = None, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]`
     - `close()`
     - `get_stats() -> Dict[str, Any]`

### mcts/utils/gpu_evaluator_service.py
**Module Docstring:** "GPU Evaluator Service - Centralized GPU evaluation"

**Classes:**
1. **GPUEvaluatorService**
   - `__init__(self, model_path: str, device: str = 'cuda', batch_timeout_ms: float = 5.0, max_batch_size: int = 512)`
   - Key methods:
     - `start()`
     - `stop()`
     - `evaluate_batch(states: np.ndarray, legal_masks: Optional[np.ndarray] = None, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]`
     - `health_check() -> bool`
     - `get_stats() -> Dict[str, Any]`

### mcts/utils/config_system.py
**Module Docstring:** "Unified configuration system"

**Classes:**
1. **ConfigSystem**
   - `__init__(self, base_config_path: Optional[str] = None)`
   - Key methods:
     - `load_config(path: str) -> Dict[str, Any]`
     - `save_config(config: Dict[str, Any], path: str)`
     - `merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]`
     - `validate_config(config: Dict[str, Any]) -> bool`
     - `get_default_config(game_type: str) -> Dict[str, Any]`

**Functions:**
1. **load_config**(path: str) -> Dict[str, Any]
2. **save_config**(config: Dict[str, Any], path: str)
3. **get_default_config**(game_type: str = 'gomoku') -> Dict[str, Any]

### mcts/utils/tensorrt_manager.py
**Module Docstring:** "TensorRT model management"

**Classes:**
1. **TensorRTManager**
   - `__init__(self, cache_dir: str = './tensorrt_cache')`
   - Key methods:
     - `get_or_create_engine(model_path: str, batch_sizes: List[int] = [1, 8, 16, 32, 64]) -> str`
     - `clear_cache()`
     - `profile_all_engines()`
     - `optimize_model(model_path: str, optimization_level: str = 'fp16') -> str`

### mcts/utils/safe_multiprocessing.py
**Module Docstring:** "Safe multiprocessing utilities"

**Classes:**
1. **SafeProcessManager**
   - `__init__(self, max_workers: int = 4)`
   - Key methods:
     - `start_worker(target: callable, args: tuple = (), kwargs: dict = None) -> int`
     - `stop_worker(worker_id: int)`
     - `stop_all_workers()`
     - `restart_worker(worker_id: int)`
     - `get_worker_status(worker_id: int) -> str`

**Functions:**
1. **setup_cuda_visible_devices**(num_workers: int, gpus_per_worker: int = 1)
2. **cleanup_processes**()
3. **cuda_multiprocessing_fix**()

### mcts/utils/worker_init.py
**Module Docstring:** "Worker initialization utilities"

**Functions:**
1. **worker_init**(worker_id: int, config: Dict[str, Any])
2. **setup_worker_logging**(worker_id: int, log_level: str = 'INFO')
3. **initialize_cuda_context**(device_id: int)
4. **cleanup_worker**()

### mcts/utils/validation.py
**Module Docstring:** "Validation utilities"

**Functions:**
1. **validate_config**(config: Dict[str, Any]) -> Tuple[bool, List[str]]
2. **validate_model**(model_path: str) -> Tuple[bool, Dict[str, Any]]
3. **validate_tree_consistency**(tree: Any) -> Tuple[bool, List[str]]
4. **validate_game_state**(state: Any, game_interface: Any) -> Tuple[bool, str]
5. **validate_training_data**(data_path: str) -> Tuple[bool, Dict[str, Any]]

### mcts/utils/validation_helpers.py
**Module Docstring:** "Helper functions for validation"

**Functions:**
1. **check_tensor_validity**(tensor: torch.Tensor, name: str = 'tensor') -> Tuple[bool, str]
2. **check_probability_distribution**(probs: np.ndarray) -> Tuple[bool, str]
3. **check_value_range**(value: float, min_val: float = -1.0, max_val: float = 1.0) -> Tuple[bool, str]
4. **check_board_consistency**(board: np.ndarray, game_type: str) -> Tuple[bool, str]
5. **check_move_legality**(move: int, legal_moves: List[int]) -> Tuple[bool, str]

## Summary

This comprehensive mapping covers all Python modules in the MCTS directory with:
- **Core modules**: 8 files with game interface, MCTS algorithm, evaluators, and configuration
- **GPU modules**: 10 files with GPU-accelerated tree operations and game state management
- **Neural network modules**: 9 files with ResNet implementation, TensorRT support, and training pipeline
- **Utility modules**: 9 files with coordination, configuration, and validation utilities

Key architectural patterns:
1. Heavy use of dataclasses for configuration
2. Abstract base classes for extensibility
3. GPU acceleration through PyTorch tensors
4. Batch processing for efficiency
5. Comprehensive validation and error handling

The codebase follows a modular design with clear separation of concerns between game logic, tree search, neural network evaluation, and GPU acceleration.