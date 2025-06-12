#!/usr/bin/env python3
"""Analyze MCTS profiling results to identify bottlenecks and performance patterns."""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_profiling_results():
    """Analyze MCTS profiling results to identify bottlenecks."""
    
    # Load summary results
    summary_df = pd.read_csv('mcts_profiling_results/summary_results.csv')
    
    # Load detailed results
    with open('mcts_profiling_results/detailed_results.json', 'r') as f:
        detailed_results = json.load(f)
    
    # Load function analysis
    with open('mcts_profiling_results/function_analysis.json', 'r') as f:
        function_analysis = json.load(f)
    
    print("=== PERFORMANCE ANALYSIS ===\n")
    
    # 1. Performance by wave size
    print("1. PERFORMANCE BY WAVE SIZE:")
    wave_perf = summary_df.groupby('wave_size').agg({
        'simulations_per_second': ['mean', 'std', 'min', 'max'],
        'efficiency_score': 'mean'
    }).round(2)
    print(wave_perf)
    print()
    
    # 2. Performance degradation with simulation count
    print("2. PERFORMANCE DEGRADATION WITH SIMULATION COUNT:")
    for wave_size in summary_df['wave_size'].unique():
        wave_data = summary_df[summary_df['wave_size'] == wave_size]
        print(f"\nWave size {wave_size}:")
        for _, row in wave_data.iterrows():
            print(f"  {row['num_simulations']:>6} sims: {row['simulations_per_second']:>8.0f} sims/s")
    
    # 3. Bottleneck analysis
    print("\n3. BOTTLENECK PHASES:")
    bottleneck_counts = summary_df['bottleneck_phase'].value_counts()
    print(bottleneck_counts)
    
    # 4. Top time-consuming functions
    print("\n4. TOP TIME-CONSUMING FUNCTIONS:")
    func_times = []
    for func_name, data in function_analysis.items():
        if isinstance(data, dict) and 'total_time' in data:
            ncalls = data.get('ncalls', 0)
            func_times.append((data['total_time'], ncalls, func_name))
    
    func_times.sort(reverse=True)
    print("\nTotal Time (s) | Calls | Function")
    print("-" * 60)
    for total_time, ncalls, func_name in func_times[:20]:
        # Skip built-in sleep and other utility functions
        if 'sleep' in func_name or 'psutil' in func_name:
            continue
        print(f"{total_time:>13.3f} | {ncalls:>5} | {func_name}")
    
    # 5. Phase-level analysis from detailed results
    print("\n5. PHASE-LEVEL PERFORMANCE ANALYSIS:")
    phase_times = {'selection': [], 'expansion': [], 'evaluation': [], 'backup': []}
    
    for result in detailed_results:
        if 'phases' in result:
            for phase in result['phases']:
                phase_name = phase['name']
                if phase_name in phase_times:
                    phase_times[phase_name].append({
                        'gpu_time': phase['gpu_time_ms'],
                        'cpu_time': phase['cpu_time_ms'],
                        'wave_size': result['wave_size'],
                        'num_sims': result['num_simulations']
                    })
    
    # Calculate phase time percentages
    print("\nAverage phase times as percentage of total:")
    for phase_name, times in phase_times.items():
        if times:
            df = pd.DataFrame(times)
            avg_gpu = df['gpu_time'].mean()
            avg_cpu = df['cpu_time'].mean()
            print(f"  {phase_name:>10}: GPU {avg_gpu:>6.1f}ms, CPU {avg_cpu:>6.1f}ms")
    
    # 6. Identify specific bottleneck functions
    print("\n6. CRITICAL BOTTLENECK FUNCTIONS:")
    critical_funcs = [
        '_expand_batch',
        '_select_batch', 
        'clone_states',
        'add_children_batch',
        'batch_action_to_child',
        'batch_select_ucb_optimized',
        'allocate_states',
        '_reset_states'
    ]
    
    for func in critical_funcs:
        for func_name, data in function_analysis.items():
            if func in func_name and isinstance(data, dict) and 'total_time' in data:
                ncalls = data.get('ncalls', 0)
                avg_time = data['total_time'] / ncalls if ncalls > 0 else 0
                print(f"{func_name}:")
                print(f"  Total time: {data['total_time']:.3f}s")
                print(f"  Calls: {ncalls}")
                print(f"  Avg time per call: {avg_time*1000:.3f}ms")
                break
    
    # 7. Memory patterns
    print("\n7. MEMORY USAGE PATTERNS:")
    mem_by_wave = summary_df.groupby('wave_size')['peak_gpu_memory_mb'].agg(['mean', 'std']).round(2)
    print(mem_by_wave)
    
    # 8. Scaling analysis
    print("\n8. SCALING ANALYSIS:")
    print("\nPerformance ratio (wave_size / base_1824):")
    base_perf = summary_df[summary_df['wave_size'] == 1824].groupby('num_simulations')['simulations_per_second'].mean()
    
    for wave_size in sorted(summary_df['wave_size'].unique()):
        if wave_size == 1824:
            continue
        wave_perf = summary_df[summary_df['wave_size'] == wave_size].groupby('num_simulations')['simulations_per_second'].mean()
        print(f"\nWave size {wave_size}:")
        for num_sims in sorted(base_perf.index):
            if num_sims in wave_perf.index:
                ratio = wave_perf[num_sims] / base_perf[num_sims]
                print(f"  {num_sims:>6} sims: {ratio:.2f}x")

if __name__ == "__main__":
    analyze_profiling_results()