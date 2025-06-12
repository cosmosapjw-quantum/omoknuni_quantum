# MCTS Comprehensive Profiler Usage Guide

## Overview

The MCTS Comprehensive Profiler is a sophisticated benchmarking and profiling tool designed to provide deep performance analysis of the MCTS implementation. It monitors GPU/CPU/RAM/VRAM usage, profiles individual MCTS phases, detects bottlenecks, and provides actionable optimization recommendations.

## Features

### 🔍 **Deep Performance Analysis**
- **GPU Timing**: Precise CUDA event-based timing for all GPU operations
- **CPU Monitoring**: Real-time CPU usage and timing measurement
- **Memory Tracking**: Continuous monitoring of GPU VRAM and system RAM
- **Phase Breakdown**: Individual profiling of selection, expansion, evaluation, and backup phases

### 📊 **Comprehensive Metrics**
- **Throughput**: Simulations per second at different scales
- **Efficiency**: Performance per GPU memory usage (sims/s/GB)
- **Resource Utilization**: Peak and average CPU/GPU/memory usage
- **Tree Statistics**: Node count, edge count, memory consumption

### 🎯 **Bottleneck Detection**
- **Phase Analysis**: Identifies which MCTS phase is the performance bottleneck
- **Memory Pressure**: Detects GPU memory limitations
- **Kernel Overhead**: Monitors CUDA kernel launch frequency
- **Transfer Detection**: Identifies CPU-GPU memory transfers

### 💡 **Optimization Recommendations**
- **Configuration Tuning**: Suggests optimal wave_size and batch_size
- **Memory Optimization**: Recommends memory usage improvements
- **Algorithm Tuning**: Phase-specific optimization suggestions
- **Hardware Utilization**: GPU/CPU utilization improvement tips

## Quick Start

### Basic Usage
```bash
# Quick benchmark with default settings
python mcts_comprehensive_profiler.py --quick

# Full comprehensive benchmark
python mcts_comprehensive_profiler.py

# Custom output directory
python mcts_comprehensive_profiler.py --output-dir my_results
```

### Programmatic Usage
```python
from mcts_comprehensive_profiler import MCTSBenchmarkSuite, ProfilingConfig

# Create custom configuration
config = ProfilingConfig(
    simulation_counts=[1000, 5000, 10000],
    wave_sizes=[512, 1024, 2048, 3072],
    measurement_iterations=5,
    generate_plots=True,
    save_detailed_logs=True
)

# Run benchmark
benchmark = MCTSBenchmarkSuite(config)
benchmark.run_comprehensive_benchmark()
```

## Configuration Options

### ProfilingConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `simulation_counts` | [1000, 5000, 10000, 25000, 50000] | List of simulation counts to test |
| `wave_sizes` | [256, 512, 1024, 2048, 3072, 4096] | List of wave sizes to test |
| `warmup_iterations` | 3 | Number of warmup runs |
| `measurement_iterations` | 5 | Number of measurement runs |
| `memory_sample_interval` | 0.01 | Memory sampling interval (seconds) |
| `gpu_event_precision` | True | Use CUDA events for precise timing |
| `trace_cuda_calls` | True | Trace CUDA kernel calls |
| `profile_individual_phases` | True | Profile each MCTS phase separately |
| `save_detailed_logs` | True | Save detailed JSON logs |
| `generate_plots` | True | Generate performance plots |
| `output_dir` | "mcts_profiling_results" | Output directory |

### Example Configurations

#### Quick Development Testing
```python
config = ProfilingConfig(
    simulation_counts=[1000],
    wave_sizes=[1024],
    measurement_iterations=2,
    generate_plots=False
)
```

#### Production Benchmarking
```python
config = ProfilingConfig(
    simulation_counts=[10000, 50000, 100000],
    wave_sizes=[1024, 2048, 3072, 4096],
    measurement_iterations=10,
    memory_sample_interval=0.005  # 5ms sampling
)
```

#### Memory-Focused Analysis
```python
config = ProfilingConfig(
    simulation_counts=[25000],
    wave_sizes=[512, 1024, 2048, 3072, 4096, 8192],
    memory_sample_interval=0.001,  # 1ms sampling
    trace_cuda_calls=True
)
```

## Output Formats

### Console Output
The profiler provides real-time console output with:
- Configuration being tested
- Performance metrics (sims/s, memory usage)
- Bottleneck identification
- Progress indicators

### Generated Files

#### 1. Performance Analysis Plot (`performance_analysis.png`)
- **Performance vs Wave Size**: Scatter plot showing optimal wave sizes
- **Memory vs Performance**: Memory efficiency analysis
- **Phase Time Breakdown**: Stacked bar chart of phase timings
- **Efficiency Distribution**: Histogram of efficiency scores

#### 2. Summary Results (`summary_results.csv`)
CSV file with key metrics for each configuration:
```csv
wave_size,num_simulations,total_time_ms,simulations_per_second,peak_gpu_memory_mb,efficiency_score,bottleneck_phase,tree_nodes,tree_memory_mb
1024,10000,303.5,32981,1247.3,26.4,selection,501,12.4
```

#### 3. Detailed Results (`detailed_results.json`)
Complete profiling data including:
- Full configuration details
- Phase-by-phase timing breakdowns
- Memory usage snapshots
- Optimization recommendations

## Interpreting Results

### Performance Metrics

#### Simulations per Second
- **Target**: 100k+ sims/s for high performance
- **Good**: 50k-100k sims/s for training
- **Needs optimization**: <50k sims/s

#### Efficiency Score (sims/s/GB)
- **Excellent**: >50 sims/s/GB
- **Good**: 20-50 sims/s/GB  
- **Poor**: <20 sims/s/GB

#### Memory Utilization
- **Optimal**: 70-85% GPU memory usage
- **Underutilized**: <50% GPU memory
- **Overloaded**: >90% GPU memory

### Bottleneck Analysis

#### Selection Bottleneck
**Symptoms**: High time in selection phase
**Causes**: 
- Inefficient UCB computation
- Poor memory access patterns
- Too many tree traversals

**Solutions**:
- Optimize UCB kernels
- Reduce tree depth
- Improve cache locality

#### Expansion Bottleneck  
**Symptoms**: High time in expansion phase
**Causes**:
- Slow legal move generation
- Inefficient node creation
- Memory allocation overhead

**Solutions**:
- Optimize legal move kernels
- Batch node allocation
- Reduce expansion breadth

#### Evaluation Bottleneck
**Symptoms**: High time in evaluation phase
**Causes**:
- Neural network bottleneck
- Small batch sizes
- CPU-GPU transfers

**Solutions**:
- Increase batch size
- Optimize neural network
- Use GPU-resident features

#### Backup Bottleneck
**Symptoms**: High time in backup phase
**Causes**:
- Inefficient parallel backup
- Memory contention
- Atomic operation overhead

**Solutions**:
- Optimize backup kernel
- Reduce path lengths
- Use coalesced updates

### Common Optimization Patterns

#### Wave Size Tuning
```python
# Too small - underutilizes GPU
wave_size = 256  # Poor parallelization

# Optimal - balances memory and parallelization  
wave_size = 3072  # Good for RTX 3060 Ti

# Too large - exceeds memory
wave_size = 8192  # May cause OOM
```

#### Memory Management
```python
# Monitor peak memory usage
if peak_gpu_memory_mb > 0.9 * total_gpu_memory:
    # Reduce wave_size or batch_size
    
if peak_gpu_memory_mb < 0.5 * total_gpu_memory:
    # Increase wave_size for better utilization
```

## Advanced Usage

### Custom Instrumentation
```python
from mcts_comprehensive_profiler import InstrumentedMCTS
from mcts.core.mcts import MCTSConfig

# Create instrumented MCTS for custom profiling
config = MCTSConfig(wave_size=3072, device='cuda')
evaluator = MyEvaluator()
profiler_config = ProfilingConfig()

instrumented_mcts = InstrumentedMCTS(config, evaluator, profiler_config)

# Run custom profiling
game_state = MyGameState()
profile = instrumented_mcts.profile_search(game_state, 10000)

# Access detailed phase data
for phase in profile.phases:
    print(f"{phase.name}: {phase.gpu_time_ms:.1f}ms, batch_size={phase.batch_size}")
```

### Memory Monitoring
```python
from mcts_comprehensive_profiler import MemoryMonitor

monitor = MemoryMonitor(sample_interval=0.005)  # 5ms sampling
monitor.start_monitoring()

# Run your code here

monitor.stop_monitoring()
peak_usage = monitor.get_peak_usage()
print(f"Peak GPU: {peak_usage.gpu_allocated_mb:.1f}MB")
```

### GPU Profiling
```python
from mcts_comprehensive_profiler import GPUProfiler

profiler = GPUProfiler(torch.device('cuda'))

with profiler.timer('my_operation'):
    # GPU operations here
    pass

stats = profiler.get_stats()
print(f"Operation time: {stats['my_operation']['avg_ms']:.2f}ms")
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
**Symptoms**: RuntimeError: CUDA out of memory
**Solution**: Reduce wave_size or simulation count

#### Poor Performance
**Symptoms**: <10k sims/s performance
**Solution**: Check GPU utilization, reduce CPU-GPU transfers

#### Profiler Crashes
**Symptoms**: Profiler stops unexpectedly
**Solution**: Check CUDA version compatibility, reduce monitoring frequency

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with error catching
try:
    benchmark.run_comprehensive_benchmark()
except Exception as e:
    print(f"Profiling error: {e}")
    import traceback
    traceback.print_exc()
```

## Performance Optimization Workflow

### 1. Baseline Measurement
```bash
python mcts_comprehensive_profiler.py --quick
```

### 2. Identify Bottleneck
Check console output for bottleneck phase and recommendations.

### 3. Targeted Optimization
Focus on the identified bottleneck phase:
- **Selection**: Optimize UCB computation
- **Expansion**: Optimize legal move generation  
- **Evaluation**: Optimize neural network
- **Backup**: Optimize parallel updates

### 4. Validation
```bash
python mcts_comprehensive_profiler.py
```

### 5. Iterate
Repeat until target performance is achieved.

## Example Results Interpretation

```
🏆 Best Performance: 168,234 sims/s
   Configuration: wave_size=3072, 50000 sims
   GPU Memory: 2,847.3MB

🎯 Most Efficient: 59.1 sims/s/GB
   Configuration: wave_size=3072

🔍 Common Bottlenecks:
   selection: 8/12 configurations
   evaluation: 3/12 configurations
   expansion: 1/12 configurations

💡 Top Recommendations:
   GPU memory underutilized - consider increasing wave_size (6 times)
   Selection is bottleneck - optimize UCB kernel (8 times)
```

This indicates:
- Optimal performance at wave_size=3072
- Selection phase needs optimization
- GPU memory can handle larger batches
- Focus optimization efforts on UCB computation

## Integration with Development Workflow

### CI/CD Integration
```bash
# Add to build pipeline
python mcts_comprehensive_profiler.py --quick --output-dir ci_results
if [ performance < threshold ]; then exit 1; fi
```

### Performance Regression Detection
```bash
# Compare with baseline
python mcts_comprehensive_profiler.py --output-dir current_results
python compare_results.py baseline_results current_results
```

### Continuous Optimization
```bash
# Weekly performance monitoring
crontab -e
0 2 * * 1 cd /path/to/mcts && python mcts_comprehensive_profiler.py --output-dir weekly_$(date +%Y%m%d)
```