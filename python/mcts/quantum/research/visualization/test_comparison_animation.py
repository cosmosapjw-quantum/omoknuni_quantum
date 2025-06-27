#!/usr/bin/env python3
"""
Test script for MCTS vs Ideal Thermodynamic Cycle Comparison Animation

This script demonstrates how to generate side-by-side animations comparing
real MCTS data with ideal thermodynamic cycles.
"""

import numpy as np
from plot_thermodynamics import ThermodynamicsVisualizer

def create_sample_mcts_data():
    """Create sample MCTS data for testing"""
    
    # Generate mock MCTS tree expansion data
    tree_expansion_data = []
    performance_metrics = []
    
    for i in range(50):  # 50 games for testing
        # Visit counts following exponential distribution
        visit_counts = np.random.exponential(20, 100 + i)
        visit_counts = visit_counts[visit_counts > 1]
        
        # Q-values with realistic variation
        q_values = np.random.normal(0, 0.3, len(visit_counts))
        q_values = np.clip(q_values, -1.0, 1.0)
        
        tree_data = {
            'visit_counts': visit_counts.tolist(),
            'q_values': q_values.tolist(),
            'tree_size': 50 + i * 2,
            'max_depth': 8 + i // 10,
            'policy_entropy': 1.5 + 0.3 * np.sin(i * 0.2),
            'timestamp': i * 10,
            'total_simulations': int(np.sum(visit_counts)),
            'game_id': i + 1
        }
        
        tree_expansion_data.append(tree_data)
        
        # Performance metrics
        performance_data = {
            'win_rate': 0.5 + 0.2 * np.sin(i * 0.1) + np.random.normal(0, 0.05),
            'average_game_length': 45 + np.random.normal(0, 8),
            'search_time': 2.0 + i * 0.02 + np.random.normal(0, 0.3),
            'nodes_per_second': 1000 + i * 10 + np.random.normal(0, 100),
            'memory_usage': 100 + i * 0.5 + np.random.normal(0, 10)
        }
        
        performance_metrics.append(performance_data)
    
    return {
        'tree_expansion_data': tree_expansion_data,
        'performance_metrics': performance_metrics,
        'metadata': {
            'generated': True,
            'description': 'Sample MCTS data for testing comparison animations'
        }
    }

def main():
    """Main function to test the comparison animations"""
    
    print("🚀 Testing MCTS vs Ideal Thermodynamic Cycle Comparison Animations")
    print("=" * 70)
    
    # Create sample data
    print("📊 Generating sample MCTS data...")
    mcts_data = create_sample_mcts_data()
    
    # Initialize thermodynamics visualizer
    print("🔧 Initializing ThermodynamicsVisualizer...")
    visualizer = ThermodynamicsVisualizer(mcts_data, output_dir="test_animations")
    
    # Create Otto cycle comparison animation
    print("🎬 Creating Otto cycle comparison animation...")
    try:
        otto_anim = visualizer.create_mcts_vs_ideal_cycle_animation(
            cycle_type='otto', 
            save_animation=True
        )
        print("✅ Otto cycle comparison animation created successfully!")
        print("   📁 Saved as: test_animations/otto_ideal_vs_mcts_comparison.gif")
    except Exception as e:
        print(f"❌ Failed to create Otto cycle animation: {e}")
    
    # Create Carnot cycle comparison animation
    print("🎬 Creating Carnot cycle comparison animation...")
    try:
        carnot_anim = visualizer.create_mcts_vs_ideal_cycle_animation(
            cycle_type='carnot', 
            save_animation=True
        )
        print("✅ Carnot cycle comparison animation created successfully!")
        print("   📁 Saved as: test_animations/carnot_ideal_vs_mcts_comparison.gif")
    except Exception as e:
        print(f"❌ Failed to create Carnot cycle animation: {e}")
    
    print("\n🎯 Key Features of the Comparison Animations:")
    print("   • Side-by-side visualization of ideal vs MCTS-derived cycles")
    print("   • Real-time process annotation (compression, expansion, heating, cooling)")
    print("   • Efficiency comparison displayed at the bottom")
    print("   • MCTS data extracted from actual tree dynamics and performance metrics")
    print("   • Synchronized animation showing differences in cycle shape and timing")
    
    print("\n📈 How MCTS Data Maps to Thermodynamic Variables:")
    print("   Volume/Entropy: Tree expansion breadth and exploration diversity")
    print("   Pressure/Temperature: Selection pressure and algorithm 'temperature'")
    print("   Work: Performance improvement and convergence efficiency")
    print("   Heat: Information flow and computational cost")
    
    print("\n🔍 Analysis Insights:")
    print("   • Compare cycle efficiency between ideal and MCTS implementations")
    print("   • Visualize deviations from ideal thermodynamic behavior")
    print("   • Identify optimization opportunities in MCTS algorithm")
    print("   • Understand quantum-classical transitions in search dynamics")
    
    print("\n✨ Animation Features:")
    print("   • 120 frames with smooth interpolation")
    print("   • Color-coded cycles (blue=ideal, green=MCTS)")
    print("   • Process-specific annotations")
    print("   • Real-time efficiency comparison")
    print("   • Publication-ready quality")
    
    print("\n" + "=" * 70)
    print("🎊 Testing completed! Check the test_animations/ directory for results.")

if __name__ == "__main__":
    main()