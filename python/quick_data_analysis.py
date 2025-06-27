"""
Quick Analysis of Collected MCTS Data
====================================

Simple analysis of the 200 games collected to extract key insights
for quantum MCTS optimization.
"""

import json
import numpy as np
from collections import defaultdict

def analyze_collected_data():
    """Analyze the collected MCTS data"""
    
    data_file = "/home/cosmosapjw/omoknuni_quantum/python/mcts/quantum/research/data_collection/real_mcts_data.json"
    
    print("MCTS Data Analysis Summary")
    print("=" * 40)
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        games = data.get('game_sessions', [])
        print(f"📊 Total games collected: {len(games)}")
        
        if not games:
            print("❌ No game data found")
            return
        
        # Basic game statistics
        move_counts = [g.get('move_count', 0) for g in games]
        total_sims = [g.get('total_simulations', 0) for g in games]
        search_times = [g.get('avg_search_time', 0) for g in games]
        peak_performance = [g.get('peak_sims_per_second', 0) for g in games]
        
        print(f"🎯 Average moves per game: {np.mean(move_counts):.1f}")
        print(f"🔄 Average simulations per game: {np.mean(total_sims):.0f}")
        print(f"⏱️  Average search time: {np.mean(search_times):.3f}s")
        print(f"🚀 Average peak performance: {np.mean(peak_performance):.0f} sims/sec")
        
        # Extract tree snapshots for deeper analysis
        all_snapshots = []
        for game in games:
            snapshots = game.get('tree_snapshots', [])
            for snapshot in snapshots:
                snapshot['game_id'] = game['game_id']
                all_snapshots.append(snapshot)
        
        print(f"🌳 Total tree snapshots: {len(all_snapshots)}")
        
        if all_snapshots:
            # Analyze visit patterns
            low_visit_nodes = 0
            total_nodes = 0
            entropy_values = []
            
            for snapshot in all_snapshots:
                visit_counts = snapshot.get('visit_counts', [])
                if visit_counts:
                    visit_array = np.array(visit_counts)
                    total_nodes += len(visit_array)
                    low_visit_nodes += np.sum(visit_array < 50)  # Quantum threshold
                    
                    # Calculate entropy
                    parent_visits = snapshot.get('parent_visits', 0)
                    if parent_visits > 0:
                        probs = visit_array / parent_visits
                        probs = probs[probs > 0]  # Remove zeros
                        if len(probs) > 0:
                            entropy = -np.sum(probs * np.log(probs))
                            max_entropy = np.log(len(visit_array))
                            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                            entropy_values.append(normalized_entropy)
            
            if total_nodes > 0:
                quantum_eligible_fraction = low_visit_nodes / total_nodes
                print(f"⚛️  Quantum-eligible nodes (< 50 visits): {quantum_eligible_fraction:.1%}")
                
            if entropy_values:
                avg_entropy = np.mean(entropy_values)
                print(f"📈 Average visit entropy: {avg_entropy:.3f}")
                
                # Phase classification
                exploration_count = sum(1 for e in entropy_values if e > 0.7)
                exploitation_count = sum(1 for e in entropy_values if e < 0.3)
                critical_count = len(entropy_values) - exploration_count - exploitation_count
                
                total_snapshots = len(entropy_values)
                print(f"🔍 Exploration phase: {exploration_count/total_snapshots:.1%}")
                print(f"🎯 Exploitation phase: {exploitation_count/total_snapshots:.1%}")
                print(f"⚖️  Critical phase: {critical_count/total_snapshots:.1%}")
        
        # Quantum enhancement recommendations
        print(f"\n💡 QUANTUM ENHANCEMENT RECOMMENDATIONS")
        print("-" * 40)
        
        if 'quantum_eligible_fraction' in locals() and quantum_eligible_fraction > 0.5:
            print("✅ HIGH PRIORITY: Quantum exploration bonuses")
            print(f"   {quantum_eligible_fraction:.1%} of nodes would benefit from quantum corrections")
        
        if np.mean(peak_performance) < 1000:
            print("✅ HIGH PRIORITY: Wave-based vectorization")
            print(f"   Current performance of {np.mean(peak_performance):.0f} sims/sec has room for improvement")
        
        if 'exploration_count' in locals() and exploration_count > 0:
            print("✅ RECOMMENDED: Phase-adaptive policy")
            print(f"   {exploration_count} snapshots in exploration phase would benefit")
        
        print("✅ RECOMMENDED: Power-law temperature annealing")
        print("   Would provide more sophisticated exploration schedule")
        
        print(f"\n🎉 DATA COLLECTION SUCCESSFUL!")
        print(f"Collected comprehensive dataset from {len(games)} games")
        print("Ready for quantum MCTS optimization implementation!")
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_file}")
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

if __name__ == "__main__":
    analyze_collected_data()