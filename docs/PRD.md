# Omoknuni: Production Requirements Document
## AlphaZero-Style Game AI with Vectorized Quantum-Inspired MCTS

**Version**: 1.0  
**Date**: June 2025  
**Status**: Initial Draft

---

## Executive Summary

Omoknuni is an AlphaZero-style game AI engine that implements massively parallel vectorized Monte Carlo Tree Search (MCTS) with quantum-inspired diversity mechanisms. The system achieves 50-200k simulations per second through wave-based GPU processing, enabling real-time strong AI on consumer hardware.

### Key Features
- **Wave-based vectorized MCTS** processing thousands of paths simultaneously
- **Quantum-inspired exploration** using path interference and phase-kicked priors
- **Adaptive resource management** scaling from laptops to cloud GPUs
- **Self-play training pipeline** with distributed architecture
- **C++ game engine integration** with Python AI components

### Performance Targets
- **Throughput**: 80k-200k simulations/second (desktop GPU)
- **Strength**: Professional level (>2800 Elo equivalent)
- **Latency**: <100ms move generation at 10k simulations
- **Training**: Convergence in 72 hours on 8x A10 GPUs

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Omoknuni AI Engine                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   C++ Game      │  │  Python MCTS    │  │  Training   │  │
│  │   Interface     │←→│  Engine         │  │  Pipeline   │  │
│  │                 │  │                 │  │             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
│           ↑                    ↓                    ↓         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              GPU Acceleration Layer                      │ │
│  │  (CUDA Kernels, PyTorch JIT, Mixed Precision)          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Neural Network Architecture

**Input Representation (20 channels)**:
1. Current board state (1 channel)
2. Current player indicator (1 channel)
3. Player 1 previous moves (8 channels - last 8 moves)
4. Player 2 previous moves (8 channels - last 8 moves)
5. Attack score plane (1 channel)
6. Defense score plane (1 channel)

**Network Structure**:
- ResNet backbone (20 blocks)
- Policy head: 256 filters → FC layer → move probabilities
- Value head: 256 filters → FC layer → tanh activation
- Optional: Auxiliary heads for move prediction and board ownership

---

## File Structure and Components

### Core Engine Files

#### 1. `game_interface.cpp` / `game_interface.h`
**Purpose**: Bridge between C++ game logic and Python AI engine
```
Functions:
- BoardState getCurrentState()
- vector<Move> getLegalMoves(BoardState)
- BoardState applyMove(BoardState, Move)
- bool isTerminal(BoardState)
- float getReward(BoardState, Player)
- Tensor boardToTensor(BoardState)  // Convert to NN input format
```

#### 2. `vectorized_mcts.py`
**Purpose**: Main MCTS engine with wave-based processing
```python
class VectorizedMCTS:
    def __init__(self, config):
        # Initialize components
        
    def search(self, root_state, num_simulations):
        # Main search loop
        
    def process_wave(self, wave_size):
        # Parallel path processing
        
    def apply_quantum_effects(self, paths):
        # Interference and phase kicks
```

#### 3. `tree_arena.py`
**Purpose**: GPU-optimized tree storage with automatic paging
```python
class TreeArena:
    def __init__(self, preset):
        # Allocate GPU/CPU memory
        
    def allocate_nodes(self, count):
        # Dynamic node allocation
        
    def get_node_batch(self, indices):
        # Batched node access
        
    def page_to_cpu(self, gpu_indices):
        # Memory overflow handling
```

#### 4. `wave_engine.py`
**Purpose**: Orchestrates wave-based MCTS iterations
```python
class WaveEngine:
    def __init__(self, tree_arena, evaluator):
        # Setup wave processing
        
    def select_paths(self, wave_size):
        # Batch path selection
        
    def evaluate_leaves(self, leaf_states):
        # Neural network evaluation
        
    def backup_values(self, paths, values):
        # Parallel backup
```

#### 5. `quantum_diversity.py`
**Purpose**: Implements quantum-inspired exploration mechanisms
```python
class QuantumDiversity:
    def __init__(self):
        # Initialize quantum parameters
        
    def compute_interference(self, paths):
        # MinHash-based O(n log n) interference
        
    def apply_phase_kicks(self, priors, uncertainties):
        # Complex-valued policy enhancement
        
    def calculate_path_integral(self, tree):
        # Path integral formulation
```

#### 6. `neural_evaluator.py`
**Purpose**: Neural network evaluation with ensemble support
```python
class NeuralEvaluator:
    def __init__(self, model_paths):
        # Load models
        
    def evaluate_batch(self, states):
        # Batch evaluation
        
    def compute_envariance(self, states):
        # Robustness metric
```

#### 7. `cuda_kernels.cu` / `cuda_kernels.h`
**Purpose**: Custom CUDA kernels for performance-critical operations
```cuda
__global__ void batched_ucb_kernel(...)
__global__ void minhash_sketch_kernel(...)
__global__ void mixed_precision_backup_kernel(...)
__global__ void phase_kick_kernel(...)
```

#### 8. `training_pipeline.py`
**Purpose**: Self-play training orchestration
```python
class TrainingPipeline:
    def __init__(self, config):
        # Setup distributed training
        
    def generate_self_play_games(self):
        # Parallel game generation
        
    def train_network(self, games):
        # Network training loop
        
    def evaluate_checkpoint(self):
        # Model evaluation
```

#### 9. `config_manager.py`
**Purpose**: Hardware-aware configuration management
```python
class ConfigManager:
    def detect_hardware(self):
        # Auto-detect GPU/RAM
        
    def load_preset(self, name):
        # Load optimized presets
        
    def validate_config(self, config):
        # Ensure valid parameters
```

#### 10. `benchmark_suite.py`
**Purpose**: Performance and quality benchmarking
```python
class BenchmarkSuite:
    def __init__(self):
        # Load test positions
        
    def measure_throughput(self, engine):
        # Simulations per second
        
    def measure_strength(self, engine):
        # Elo estimation
        
    def ablation_study(self):
        # Component analysis
```

---

## Detailed Algorithms

### Algorithm 1: Wave-Based Selection
```
function SELECT_WAVE(tree_arena, wave_size):
    # Initialize batch of paths
    paths = zeros(wave_size, max_depth)
    current_nodes = get_root_nodes(wave_size)
    
    for depth in 0 to max_depth:
        # Batch compute UCB scores
        ucb_scores = COMPUTE_UCB_BATCH(current_nodes)
        
        # Apply quantum interference
        if use_interference:
            sketches = COMPUTE_MINHASH(current_nodes)
            similarities = ESTIMATE_SIMILARITIES(sketches)
            ucb_scores = APPLY_INTERFERENCE(ucb_scores, similarities)
        
        # Select actions
        actions = argmax(ucb_scores, axis=1)
        
        # Get children
        children = GET_CHILDREN_BATCH(current_nodes, actions)
        
        # Update paths
        paths[:, depth] = current_nodes
        
        # Check terminal conditions
        is_leaf = (children < 0)
        if all(is_leaf):
            break
            
        current_nodes = where(is_leaf, current_nodes, children)
    
    return paths
```

### Algorithm 2: MinHash Interference
```
function MINHASH_INTERFERENCE(paths, num_hashes=4):
    # Step 1: Compute MinHash sketches - O(n)
    sketches = zeros(len(paths), num_hashes)
    
    for i, path in enumerate(paths):
        for j in range(num_hashes):
            sketch_value = infinity
            for node in path:
                hash_value = hash_function[j](node)
                sketch_value = min(sketch_value, hash_value)
            sketches[i, j] = sketch_value
    
    # Step 2: Build LSH buckets - O(n log n)
    buckets = defaultdict(list)
    for i, sketch in enumerate(sketches):
        bucket_id = hash(tuple(sketch))
        buckets[bucket_id].append(i)
    
    # Step 3: Apply interference within buckets - O(n)
    interference_weights = ones(len(paths))
    
    for bucket in buckets.values():
        if len(bucket) > 1:
            # Paths in same bucket have high overlap
            for i in bucket:
                # Reduce selection probability
                interference_weights[i] *= (1.0 - interference_strength)
    
    return interference_weights
```

### Algorithm 3: Phase-Kicked Prior Policy
```
function PHASE_KICKED_POLICY(logits, value_uncertainties, temperature=1.0):
    # Compute phase based on uncertainty
    phase_strength = 0.1  # Hyperparameter
    phases = phase_strength * value_uncertainties / temperature
    
    # Apply complex exponential
    complex_logits = logits * exp(1j * phases)
    
    # Interference pattern emerges from magnitude
    magnitudes = abs(complex_logits)
    
    # Temperature-scaled softmax
    exp_magnitudes = exp(magnitudes / temperature)
    probabilities = exp_magnitudes / sum(exp_magnitudes)
    
    return probabilities
```

### Algorithm 4: Envariance-Filtered Backup
```
function ENVARIANT_BACKUP(paths, evaluator_ensemble):
    # Evaluate paths with multiple evaluators
    evaluations = []
    for evaluator in evaluator_ensemble:
        values = evaluator.evaluate_batch(paths)
        evaluations.append(values)
    
    # Compute envariance (inverse of variance)
    means = mean(evaluations, axis=0)
    stds = std(evaluations, axis=0)
    envariances = exp(-stds / (abs(means) + epsilon))
    
    # Weight backup by envariance
    for i, path in enumerate(paths):
        backup_weight = envariances[i]
        
        # Traverse path and update values
        for depth, node in enumerate(path):
            if node < 0:
                break
                
            # Update with envariance weighting
            tree_arena.values[node] += means[i] * backup_weight
            tree_arena.visits[node] += backup_weight
            tree_arena.envariance[node] = moving_average(
                tree_arena.envariance[node], 
                envariances[i]
            )
```

---

## Application Flow Diagram

```
┌─────────────────┐
│   Game State    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ C++ Interface   │────▶│ State Encoding  │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ VectorizedMCTS  │
                        └────────┬────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                                 ▼
       ┌─────────────────┐              ┌─────────────────┐
       │   WaveEngine    │              │  TreeArena      │
       └────────┬────────┘              └────────┬────────┘
                │                                 │
                ▼                                 │
       ┌─────────────────┐                       │
       │ Select Paths    │◀──────────────────────┘
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐     ┌─────────────────┐
       │QuantumDiversity │────▶│ Apply Effects   │
       └─────────────────┘     └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │NeuralEvaluator  │
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │ Backup Values   │
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │ Move Selection  │
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │  Return Move    │
                               └─────────────────┘
```

---

## Technology Stack

### Core Technologies
- **Language**: Python 3.10+ (AI engine), C++ 17 (game logic)
- **Deep Learning**: PyTorch 2.0+ with CUDA 11.8+
- **GPU Acceleration**: CUDA custom kernels, Triton for JIT compilation
- **Distributed Training**: PyTorch DDP, Ray for orchestration
- **Configuration**: YAML with Pydantic validation
- **Testing**: pytest, C++ Google Test
- **Profiling**: NVIDIA Nsight, PyTorch Profiler

### Key Dependencies
```yaml
python:
  - pytorch >= 2.0.0
  - numpy >= 1.24.0
  - numba >= 0.57.0  # CPU JIT compilation
  - ray >= 2.5.0     # Distributed computing
  - pydantic >= 2.0  # Configuration validation
  - tensorboard      # Training monitoring
  
cuda:
  - cuda-toolkit >= 11.8
  - cudnn >= 8.6
  - nccl >= 2.14     # Multi-GPU communication
  
cpp:
  - cmake >= 3.20
  - pybind11         # Python bindings
  - eigen3           # Linear algebra
```

---

## Claude Code Guidance

### Prompt Template for Implementation

```
You are implementing the Omoknuni game AI engine, which uses vectorized MCTS with quantum-inspired enhancements. The system processes thousands of MCTS paths in parallel waves on GPU.

Key architectural principles:
1. **Wave-based processing**: Process batches of 256-2048 paths simultaneously
2. **No virtual loss**: Use MinHash interference for diversity instead
3. **Mixed precision**: FP16 for high visit counts, FP32 for low counts
4. **Quantum effects**: Phase-kicked priors and path interference
5. **Resource-aware**: Automatic CPU/GPU memory management

When implementing:
- Prioritize GPU parallelism over sequential optimizations
- Use PyTorch operations for automatic differentiation
- Implement custom CUDA kernels only for proven bottlenecks
- Maintain strict separation between game logic (C++) and AI (Python)
- Use type hints and docstrings for all public APIs

Performance targets:
- Single wave processing: <5ms for 1024 paths
- GPU utilization: >85% during search
- Memory efficiency: <4GB for 1M node tree

Current file: [FILENAME]
Task: [SPECIFIC IMPLEMENTATION TASK]
```

### Example Implementation Request

```
Implement the WaveEngine.select_paths method that:
1. Takes a wave_size parameter (typically 256-2048)
2. Performs batched UCB calculation across all paths
3. Applies MinHash-based interference to similar paths
4. Returns selected paths as a (wave_size, max_depth) tensor
5. Ensures coalesced memory access for GPU efficiency

The method should handle both leaf and internal nodes gracefully, 
stopping path extension when reaching terminal states.
```

---

## Development Phases

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Basic vectorized MCTS without quantum features

- [ ] Week 1: Setup and Infrastructure
  - [ ] Project structure and build system
  - [ ] C++ game interface bindings
  - [ ] Basic configuration management
  - [ ] Unit test framework

- [ ] Week 2: Core MCTS Components
  - [ ] TreeArena with GPU allocation
  - [ ] Basic WaveEngine (no interference)
  - [ ] Simple neural evaluator wrapper
  - [ ] Batch UCB calculation

- [ ] Week 3: Integration
  - [ ] Connect C++ game to Python MCTS
  - [ ] Implement basic search loop
  - [ ] Add move selection logic
  - [ ] Initial benchmarking

- [ ] Week 4: Optimization
  - [ ] Profile and identify bottlenecks
  - [ ] Implement first CUDA kernels
  - [ ] Add mixed precision support
  - [ ] Target: 20k sims/s

### Phase 2: Quantum Enhancements (Weeks 5-8)
**Goal**: Add interference and phase-kicked exploration

- [ ] Week 5: MinHash Infrastructure
  - [ ] Implement MinHash sketching
  - [ ] Add LSH bucketing
  - [ ] Create interference mechanism
  - [ ] Unit tests for diversity

- [ ] Week 6: Phase-Kicked Priors
  - [ ] Complex-valued policy implementation
  - [ ] Uncertainty estimation
  - [ ] Phase kick kernel
  - [ ] Ablation testing

- [ ] Week 7: Path Integral Framework
  - [ ] Implement path action calculation
  - [ ] Add decoherence model
  - [ ] Optimize batch sizes
  - [ ] Theoretical validation

- [ ] Week 8: Integration and Testing
  - [ ] Combine all quantum features
  - [ ] Comprehensive benchmarking
  - [ ] Tune hyperparameters
  - [ ] Target: 50k sims/s

### Phase 3: Training Pipeline (Weeks 9-12)
**Goal**: Self-play training system

- [ ] Week 9: Data Generation
  - [ ] Self-play game generator
  - [ ] Position augmentation
  - [ ] Data serialization
  - [ ] Distributed generation

- [ ] Week 10: Network Training
  - [ ] Training loop implementation
  - [ ] Loss functions
  - [ ] Learning rate scheduling
  - [ ] Checkpoint management

- [ ] Week 11: Evaluation System
  - [ ] Elo rating system
  - [ ] Automated tournaments
  - [ ] Model comparison
  - [ ] Regression testing

- [ ] Week 12: Optimization
  - [ ] Multi-GPU training
  - [ ] Pipeline optimization
  - [ ] Resource monitoring
  - [ ] Target: 24h to 2000 Elo

### Phase 4: Production (Weeks 13-16)
**Goal**: Production-ready system

- [ ] Week 13: Robustness
  - [ ] Error handling
  - [ ] Fallback mechanisms
  - [ ] Resource limits
  - [ ] Stress testing

- [ ] Week 14: Performance
  - [ ] Final optimizations
  - [ ] Hardware-specific tuning
  - [ ] Latency optimization
  - [ ] Target: 100k sims/s

- [ ] Week 15: Polish
  - [ ] API finalization
  - [ ] Documentation
  - [ ] Example applications
  - [ ] Deployment guides

- [ ] Week 16: Release
  - [ ] Final benchmarks
  - [ ] Performance validation
  - [ ] Release packaging
  - [ ] Community setup

---

## In-Scope Items

### Core Features
- Vectorized MCTS with wave-based processing
- Quantum-inspired diversity mechanisms (interference, phase kicks)
- GPU-accelerated tree operations
- Neural network integration with ensemble support
- Self-play training pipeline
- Hardware-aware configuration system
- Comprehensive benchmarking suite
- C++ game engine integration

### Performance Targets
- 80k-200k simulations/second on consumer GPUs
- <100ms latency for strong moves
- Support for 100k+ node trees
- Distributed training on 8+ GPUs

### Platforms
- Linux (primary)
- Windows with WSL2
- Cloud deployment (AWS, GCP)

---

## Out-of-Scope Items

### Not Included in V1.0
- Mobile device support
- Web browser deployment
- Real-time online play infrastructure
- Graphical user interface
- Game-specific heuristics
- Opening book integration
- Endgame tablebase support
- Multi-game support (single game only)

### Future Considerations
- Quantization for edge deployment
- ONNX export for inference
- Rust implementation for systems programming
- WebGPU support for browser deployment
- Multi-agent MCTS variants
- Continuous action spaces

---

## Success Metrics

### Performance Metrics
- **Throughput**: ≥80k sims/s (RTX 3060 Ti)
- **Latency**: ≤100ms for 10k simulations
- **Memory**: ≤4GB for 1M nodes
- **GPU Utilization**: ≥85% during search

### Quality Metrics
- **Strength**: ≥2800 Elo equivalent
- **Consistency**: <50 Elo variation across runs
- **Convergence**: Professional level in 72 hours
- **Robustness**: No crashes in 1M games

### Engineering Metrics
- **Test Coverage**: ≥90% for core components
- **Documentation**: All public APIs documented
- **Build Time**: <5 minutes full rebuild
- **Deployment**: One-command installation

---

## Risk Mitigation

### Technical Risks
1. **Quantum features hurt performance**
   - Mitigation: Ablation studies, optional features
   
2. **GPU memory limitations**
   - Mitigation: Automatic CPU paging, adaptive batch sizes

3. **Training instability**
   - Mitigation: Gradient clipping, careful initialization

### Schedule Risks
1. **CUDA kernel optimization time**
   - Mitigation: Start with PyTorch, optimize incrementally

2. **Integration complexity**
   - Mitigation: Clear interfaces, extensive testing

3. **Performance targets not met**
   - Mitigation: Multiple optimization paths identified

---

## Appendices

### A. Configuration Schema
```yaml
engine:
  wave_size: 512
  max_simulations: 10000
  cpuct: 1.0
  
quantum:
  use_interference: true
  phase_strength: 0.1
  num_minhash: 4
  
hardware:
  preset: "auto"  # auto, laptop-16GB, desktop-64GB, cloud-A10
  gpu_memory_fraction: 0.9
  
training:
  batch_size: 1024
  learning_rate: 0.001
  checkpoint_interval: 1000
```

### B. Benchmark Positions
- Standard test suite of 1000 positions
- Tactical puzzles for move accuracy
- Endgame positions for precision
- Opening positions for strategic play

### C. Development Tools
- NVIDIA Nsight for GPU profiling
- Tensorboard for training monitoring
- pytest-benchmark for performance regression
- Memory profilers for leak detection