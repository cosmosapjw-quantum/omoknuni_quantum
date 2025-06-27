"""
Analysis of Collected MCTS Data for Quantum Research
===================================================

Analyzes the real MCTS data collected from 200 games to validate
quantum-inspired enhancements and identify optimization opportunities.

Key analyses:
1. MCTS performance characteristics across game phases
2. Visit count distributions and entropy analysis
3. Search depth and branching factor patterns
4. Performance bottlenecks and optimization opportunities
5. Quantum enhancement applicability assessment
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class MCTSDataAnalyzer:
    """Analyzes collected MCTS data for quantum research insights"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.data = None
        self.games = []
        self.tree_snapshots = []
        
        self._load_data()
        self._preprocess_data()
        
    def _load_data(self):
        """Load MCTS data from JSON file"""
        print(f"📊 Loading MCTS data from {self.data_file}...")
        
        with open(self.data_file, 'r') as f:
            self.data = json.load(f)
        
        self.games = self.data.get('game_sessions', [])
        
        # Extract tree snapshots from all games
        for game in self.games:
            snapshots = game.get('tree_snapshots', [])
            for snapshot in snapshots:
                snapshot['game_id'] = game['game_id']
                snapshot['move_number'] = snapshot.get('move_number', 0)
                self.tree_snapshots.append(snapshot)
        
        print(f"✓ Loaded {len(self.games)} games with {len(self.tree_snapshots)} tree snapshots")
        
    def _preprocess_data(self):
        """Preprocess data for analysis"""
        print("🔧 Preprocessing data...")
        
        # Add derived metrics to tree snapshots
        for snapshot in self.tree_snapshots:
            if 'visit_counts' in snapshot and 'parent_visits' in snapshot:
                visit_counts = np.array(snapshot['visit_counts'])
                parent_visits = snapshot['parent_visits']
                
                # Visit distribution entropy
                if parent_visits > 0:
                    visit_probs = visit_counts / parent_visits
                    # Add small epsilon to avoid log(0)
                    visit_probs = visit_probs + 1e-10
                    entropy = -np.sum(visit_probs * np.log(visit_probs))
                    max_entropy = np.log(len(visit_counts))
                    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                else:
                    normalized_entropy = 0
                
                snapshot['visit_entropy'] = normalized_entropy
                snapshot['max_visit_count'] = np.max(visit_counts) if len(visit_counts) > 0 else 0
                snapshot['min_visit_count'] = np.min(visit_counts[visit_counts > 0]) if np.any(visit_counts > 0) else 0
                snapshot['visit_count_ratio'] = (snapshot['max_visit_count'] / 
                                               (snapshot['min_visit_count'] + 1e-8))
        
        print("✓ Preprocessing complete")
    
    def analyze_performance_characteristics(self) -> Dict[str, Any]:
        """Analyze MCTS performance characteristics"""
        print("\n📈 Analyzing Performance Characteristics...")
        
        results = {}
        
        # Game-level statistics
        game_stats = {
            'move_counts': [g['move_count'] for g in self.games],
            'total_simulations': [g.get('total_simulations', 0) for g in self.games],
            'avg_search_times': [g.get('avg_search_time', 0) for g in self.games],
            'peak_sims_per_second': [g.get('peak_sims_per_second', 0) for g in self.games]
        }
        
        results['game_statistics'] = {
            'num_games': len(self.games),
            'avg_moves_per_game': np.mean(game_stats['move_counts']),
            'avg_total_simulations': np.mean(game_stats['total_simulations']),
            'avg_search_time': np.mean(game_stats['avg_search_times']),
            'avg_peak_performance': np.mean(game_stats['peak_sims_per_second']),
            'performance_std': np.std(game_stats['peak_sims_per_second'])
        }
        
        print(f"✓ Games analyzed: {results['game_statistics']['num_games']}")
        print(f"✓ Average moves per game: {results['game_statistics']['avg_moves_per_game']:.1f}")
        print(f"✓ Average search time: {results['game_statistics']['avg_search_time']:.3f}s")
        print(f"✓ Average peak performance: {results['game_statistics']['avg_peak_performance']:.0f} sims/sec")
        
        return results
    
    def analyze_visit_distributions(self) -> Dict[str, Any]:
        """Analyze visit count distributions for quantum enhancement opportunities"""
        print("\n🎯 Analyzing Visit Count Distributions...")
        
        results = {}
        
        # Collect visit statistics from all snapshots
        all_visit_entropies = []
        all_parent_visits = []
        all_max_visits = []
        all_visit_ratios = []
        low_visit_fractions = []
        
        for snapshot in self.tree_snapshots:
            if 'visit_counts' in snapshot:
                visit_counts = np.array(snapshot['visit_counts'])
                parent_visits = snapshot.get('parent_visits', 0)
                
                all_visit_entropies.append(snapshot.get('visit_entropy', 0))
                all_parent_visits.append(parent_visits)
                all_max_visits.append(snapshot.get('max_visit_count', 0))
                all_visit_ratios.append(snapshot.get('visit_count_ratio', 1))
                
                # Fraction of low-visit nodes (< 50 visits, from quantum research threshold)
                if len(visit_counts) > 0:
                    low_visit_count = np.sum(visit_counts < 50)
                    low_visit_fraction = low_visit_count / len(visit_counts)
                    low_visit_fractions.append(low_visit_fraction)
        
        results['visit_analysis'] = {
            'num_snapshots': len(self.tree_snapshots),
            'avg_visit_entropy': np.mean(all_visit_entropies),
            'avg_parent_visits': np.mean(all_parent_visits),
            'avg_max_visits': np.mean(all_max_visits),
            'avg_visit_ratio': np.mean(all_visit_ratios),
            'avg_low_visit_fraction': np.mean(low_visit_fractions),
            'entropy_std': np.std(all_visit_entropies)
        }
        
        # Categorize snapshots by exploration/exploitation phase
        exploration_snapshots = [s for s in self.tree_snapshots if s.get('visit_entropy', 0) > 0.7]
        exploitation_snapshots = [s for s in self.tree_snapshots if s.get('visit_entropy', 0) < 0.3]
        critical_snapshots = [s for s in self.tree_snapshots if 0.3 <= s.get('visit_entropy', 0) <= 0.7]
        
        results['phase_analysis'] = {
            'exploration_snapshots': len(exploration_snapshots),
            'exploitation_snapshots': len(exploitation_snapshots),
            'critical_snapshots': len(critical_snapshots),
            'exploration_fraction': len(exploration_snapshots) / len(self.tree_snapshots),
            'exploitation_fraction': len(exploitation_snapshots) / len(self.tree_snapshots),
            'critical_fraction': len(critical_snapshots) / len(self.tree_snapshots)
        }
        
        print(f"✓ Tree snapshots analyzed: {results['visit_analysis']['num_snapshots']}")
        print(f"✓ Average visit entropy: {results['visit_analysis']['avg_visit_entropy']:.3f}")
        print(f"✓ Average low-visit fraction: {results['visit_analysis']['avg_low_visit_fraction']:.3f}")
        print(f"✓ Exploration phase: {results['phase_analysis']['exploration_fraction']:.1%}")
        print(f"✓ Exploitation phase: {results['phase_analysis']['exploitation_fraction']:.1%}")
        print(f"✓ Critical phase: {results['phase_analysis']['critical_fraction']:.1%}")
        
        return results
    
    def analyze_quantum_enhancement_opportunities(self) -> Dict[str, Any]:
        """Identify opportunities for quantum enhancements"""
        print("\n⚛️  Analyzing Quantum Enhancement Opportunities...")
        
        results = {}
        
        # Analyze nodes that would benefit from quantum corrections
        quantum_beneficial_snapshots = 0
        total_quantum_eligible_nodes = 0
        total_nodes = 0
        
        move_phase_analysis = defaultdict(list)
        
        for snapshot in self.tree_snapshots:
            if 'visit_counts' in snapshot:
                visit_counts = np.array(snapshot['visit_counts'])
                move_number = snapshot.get('move_number', 0)
                parent_visits = snapshot.get('parent_visits', 0)
                
                total_nodes += len(visit_counts)
                
                # Quantum-eligible nodes (< 50 visits from research threshold)
                quantum_eligible = np.sum(visit_counts < 50)
                total_quantum_eligible_nodes += quantum_eligible
                
                if quantum_eligible > 0:
                    quantum_beneficial_snapshots += 1
                
                # Analyze by game phase (early/mid/late)
                if move_number < 10:
                    phase = 'early'
                elif move_number < 30:
                    phase = 'mid'
                else:
                    phase = 'late'
                
                move_phase_analysis[phase].append({
                    'quantum_eligible_fraction': quantum_eligible / len(visit_counts),
                    'visit_entropy': snapshot.get('visit_entropy', 0),
                    'parent_visits': parent_visits
                })
        
        # Compute phase-specific statistics
        phase_stats = {}
        for phase, snapshots in move_phase_analysis.items():
            if snapshots:
                phase_stats[phase] = {
                    'count': len(snapshots),
                    'avg_quantum_eligible_fraction': np.mean([s['quantum_eligible_fraction'] for s in snapshots]),
                    'avg_visit_entropy': np.mean([s['visit_entropy'] for s in snapshots]),
                    'avg_parent_visits': np.mean([s['parent_visits'] for s in snapshots])
                }
        
        results['quantum_opportunities'] = {
            'beneficial_snapshots': quantum_beneficial_snapshots,
            'total_snapshots': len(self.tree_snapshots),
            'beneficial_fraction': quantum_beneficial_snapshots / len(self.tree_snapshots),
            'quantum_eligible_nodes': total_quantum_eligible_nodes,
            'total_nodes': total_nodes,
            'quantum_eligible_fraction': total_quantum_eligible_nodes / total_nodes,
            'phase_statistics': phase_stats
        }
        
        print(f"✓ Snapshots benefiting from quantum: {quantum_beneficial_snapshots}/{len(self.tree_snapshots)} ({results['quantum_opportunities']['beneficial_fraction']:.1%})")
        print(f"✓ Quantum-eligible nodes: {total_quantum_eligible_nodes}/{total_nodes} ({results['quantum_opportunities']['quantum_eligible_fraction']:.1%})")
        
        for phase, stats in phase_stats.items():
            print(f"✓ {phase.title()} game phase: {stats['avg_quantum_eligible_fraction']:.1%} quantum-eligible")
        
        return results
    
    def analyze_search_efficiency(self) -> Dict[str, Any]:
        """Analyze search efficiency and identify bottlenecks"""
        print("\n🔍 Analyzing Search Efficiency...")
        
        results = {}
        
        # Performance metrics from snapshots
        search_times = []
        simulations_per_move = []
        nodes_visited = []
        
        for snapshot in self.tree_snapshots:
            if 'search_time' in snapshot:
                search_times.append(snapshot['search_time'])
            if 'simulations_count' in snapshot:
                simulations_per_move.append(snapshot['simulations_count'])
            if 'visit_counts' in snapshot:
                nodes_visited.append(len(snapshot['visit_counts']))
        
        # Game-level performance analysis
        game_performance = []
        for game in self.games:
            perf_data = {
                'total_simulations': game.get('total_simulations', 0),
                'move_count': game.get('move_count', 0),
                'avg_search_time': game.get('avg_search_time', 0),
                'peak_sims_per_second': game.get('peak_sims_per_second', 0)
            }
            
            # Compute efficiency metrics
            if perf_data['move_count'] > 0:
                perf_data['simulations_per_move'] = perf_data['total_simulations'] / perf_data['move_count']
            else:
                perf_data['simulations_per_move'] = 0
            
            game_performance.append(perf_data)
        
        results['efficiency_analysis'] = {
            'avg_search_time': np.mean([g['avg_search_time'] for g in game_performance]),
            'avg_sims_per_move': np.mean([g['simulations_per_move'] for g in game_performance]),
            'avg_peak_performance': np.mean([g['peak_sims_per_second'] for g in game_performance]),
            'performance_variance': np.var([g['peak_sims_per_second'] for g in game_performance]),
            'avg_nodes_per_snapshot': np.mean(nodes_visited) if nodes_visited else 0
        }
        
        # Identify performance bottlenecks
        slow_games = [g for g in game_performance if g['avg_search_time'] > 0.5]  # > 0.5s per move
        fast_games = [g for g in game_performance if g['avg_search_time'] < 0.1]  # < 0.1s per move
        
        results['bottleneck_analysis'] = {
            'slow_games_count': len(slow_games),
            'fast_games_count': len(fast_games),
            'slow_game_fraction': len(slow_games) / len(game_performance),
            'performance_improvement_potential': results['efficiency_analysis']['avg_peak_performance'] > 1000  # If > 1k sims/sec, good performance
        }
        
        print(f"✓ Average search time: {results['efficiency_analysis']['avg_search_time']:.3f}s")
        print(f"✓ Average simulations per move: {results['efficiency_analysis']['avg_sims_per_move']:.0f}")
        print(f"✓ Average peak performance: {results['efficiency_analysis']['avg_peak_performance']:.0f} sims/sec")
        print(f"✓ Slow games (>0.5s/move): {results['bottleneck_analysis']['slow_game_fraction']:.1%}")
        
        return results
    
    def generate_quantum_enhancement_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations for quantum enhancements based on analysis"""
        print("\n💡 Generating Quantum Enhancement Recommendations...")
        
        recommendations = {}
        
        # Extract key metrics
        beneficial_fraction = analysis_results['quantum_opportunities']['beneficial_fraction']
        quantum_eligible_fraction = analysis_results['quantum_opportunities']['quantum_eligible_fraction']
        avg_low_visit_fraction = analysis_results['visit_analysis']['avg_low_visit_fraction']
        exploration_fraction = analysis_results['phase_analysis']['exploration_fraction']
        
        # Recommendation 1: Quantum bonus applicability
        if quantum_eligible_fraction > 0.3:  # > 30% of nodes eligible
            recommendations['quantum_bonus'] = {
                'recommended': True,
                'priority': 'high',
                'reason': f'{quantum_eligible_fraction:.1%} of nodes have <50 visits and would benefit from quantum exploration bonuses',
                'expected_impact': 'significant'
            }
        elif quantum_eligible_fraction > 0.1:  # > 10% eligible
            recommendations['quantum_bonus'] = {
                'recommended': True,
                'priority': 'medium',
                'reason': f'{quantum_eligible_fraction:.1%} of nodes would benefit from quantum bonuses',
                'expected_impact': 'moderate'
            }
        else:
            recommendations['quantum_bonus'] = {
                'recommended': False,
                'priority': 'low',
                'reason': f'Only {quantum_eligible_fraction:.1%} of nodes would benefit',
                'expected_impact': 'minimal'
            }
        
        # Recommendation 2: Phase-adaptive policy
        if exploration_fraction > 0.2:  # > 20% in exploration phase
            recommendations['phase_adaptive_policy'] = {
                'recommended': True,
                'priority': 'high',
                'reason': f'{exploration_fraction:.1%} of snapshots in exploration phase would benefit from adaptive c_puct',
                'expected_impact': 'significant'
            }
        else:
            recommendations['phase_adaptive_policy'] = {
                'recommended': True,
                'priority': 'medium',
                'reason': 'Phase adaptation would help balance exploration/exploitation',
                'expected_impact': 'moderate'
            }
        
        # Recommendation 3: Power-law annealing
        avg_search_time = analysis_results['efficiency_analysis']['avg_search_time']
        if avg_search_time > 0.2:  # Slower searches benefit more from annealing
            recommendations['power_law_annealing'] = {
                'recommended': True,
                'priority': 'medium',
                'reason': f'Average search time of {avg_search_time:.3f}s suggests benefit from improved exploration schedule',
                'expected_impact': 'moderate'
            }
        else:
            recommendations['power_law_annealing'] = {
                'recommended': True,
                'priority': 'low',
                'reason': 'Would provide marginal improvement to exploration schedule',
                'expected_impact': 'minimal'
            }
        
        # Recommendation 4: Wave vectorization
        avg_peak_performance = analysis_results['efficiency_analysis']['avg_peak_performance']
        if avg_peak_performance < 5000:  # < 5k sims/sec suggests vectorization opportunity
            recommendations['wave_vectorization'] = {
                'recommended': True,
                'priority': 'high',
                'reason': f'Current peak performance of {avg_peak_performance:.0f} sims/sec has room for vectorization improvement',
                'expected_impact': 'significant'
            }
        else:
            recommendations['wave_vectorization'] = {
                'recommended': True,
                'priority': 'medium',
                'reason': 'Wave vectorization would provide additional performance gains',
                'expected_impact': 'moderate'
            }
        
        print("✓ Quantum Enhancement Recommendations:")
        for enhancement, rec in recommendations.items():
            priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            status = "✅ RECOMMENDED" if rec['recommended'] else "❌ NOT RECOMMENDED"
            print(f"  {priority_emoji} {enhancement}: {status}")
            print(f"    Reason: {rec['reason']}")
            print(f"    Expected impact: {rec['expected_impact']}")
        
        return recommendations
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis of MCTS data"""
        print("MCTS Data Analysis for Quantum Research")
        print("=" * 50)
        
        results = {}
        
        # Run individual analyses
        results['performance'] = self.analyze_performance_characteristics()
        results['visit_distributions'] = self.analyze_visit_distributions()
        results['quantum_opportunities'] = self.analyze_quantum_enhancement_opportunities()
        results['efficiency'] = self.analyze_search_efficiency()
        
        # Generate recommendations
        all_results = {}
        for category, data in results.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        all_results.update(value)
                    else:
                        all_results[key] = value
        results['recommendations'] = self.generate_quantum_enhancement_recommendations(all_results)
        
        # Summary
        print(f"\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        
        print(f"📊 Dataset: {len(self.games)} games, {len(self.tree_snapshots)} tree snapshots")
        print(f"⚛️  Quantum opportunity: {all_results['quantum_eligible_fraction']:.1%} of nodes eligible")
        print(f"🎯 Performance: {all_results['avg_peak_performance']:.0f} sims/sec average")
        print(f"🔍 Search phases: {all_results['exploration_fraction']:.1%} exploration, {all_results['exploitation_fraction']:.1%} exploitation")
        
        high_priority_recs = [name for name, rec in results['recommendations'].items() 
                             if rec['recommended'] and rec['priority'] == 'high']
        print(f"💡 High-priority recommendations: {len(high_priority_recs)}")
        for rec_name in high_priority_recs:
            print(f"   • {rec_name}")
        
        return results

def main():
    """Main analysis function"""
    data_file = "/home/cosmosapjw/omoknuni_quantum/python/mcts/quantum/research/data_collection/real_mcts_data.json"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please run the MCTS data collector first.")
        return
    
    # Run analysis
    analyzer = MCTSDataAnalyzer(data_file)
    results = analyzer.run_comprehensive_analysis()
    
    # Save analysis results
    output_file = data_file.replace('.json', '_analysis.json')
    with open(output_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        import json
        json.dump(results, f, indent=2, default=convert_numpy)
    
    print(f"\n📁 Analysis results saved to: {output_file}")
    print("✅ MCTS data analysis complete!")
    
    return results

if __name__ == "__main__":
    main()