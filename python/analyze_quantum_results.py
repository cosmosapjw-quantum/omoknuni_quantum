#!/usr/bin/env python3
# Parse quantum performance results
import json
with open('mcts_quantum_profiling_results/quantum_detailed_results.json', 'r') as f:
    data = json.load(f)

print('Detailed Performance Analysis')
print('='*80)
print(f'{"Config":<20} {"Sims":<10} {"Time (ms)":<15} {"Sims/sec":<15} {"Bottleneck":<20}')
print('-'*80)

# Group by simulation count for comparison
by_sims = {}
for run in data:
    sims = run['num_simulations']
    if sims not in by_sims:
        by_sims[sims] = []
    by_sims[sims].append(run)

# Show comparisons
for sims in sorted(by_sims.keys()):
    print(f'\n{sims} simulations:')
    runs = sorted(by_sims[sims], key=lambda x: x['quantum_level'])
    for run in runs:
        print(f"{run['quantum_level']:<20} {run['num_simulations']:<10} {run['total_time_ms']:<15.2f} {run['simulations_per_second']:<15.1f} {run['bottleneck_phase']:<20}")
    
    # Calculate speedup
    classical = next(r for r in runs if r['quantum_level'] == 'classical')
    for run in runs:
        if run['quantum_level'] != 'classical':
            speedup = run['simulations_per_second'] / classical['simulations_per_second']
            print(f"  -> {run['quantum_level']} speedup: {speedup:.3f}x")