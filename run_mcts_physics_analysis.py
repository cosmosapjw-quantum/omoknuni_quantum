"""
Wrapper script to run complete MCTS physics analysis.
This can be executed from anywhere in the project.
"""
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add the analysis module to Python path
analysis_path = project_root / "python" / "mcts" / "quantum" / "analysis"
sys.path.insert(0, str(analysis_path))

# Import and run the main script
import run_complete_physics_analysis

if __name__ == '__main__':
    run_complete_physics_analysis.main()