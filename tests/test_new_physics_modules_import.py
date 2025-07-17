"""Test that the new physics modules can be imported without errors."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set library path for libalphazero.so
lib_path = project_root / "python"
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ['LD_LIBRARY_PATH'] = str(lib_path)

def test_imports():
    """Test that all three new physics modules can be imported."""
    print("Testing imports of new physics modules...")
    
    # Test gauge_policy import
    try:
        from python.mcts.quantum.phenomena.gauge_policy import (
            GaugeField, WilsonLoop, GaugeInvariantPolicy, GaugeInvariantPolicyLearner
        )
        print("✓ gauge_policy module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import gauge_policy: {e}")
        return False
    
    # Test quantum_error_correction import
    try:
        from python.mcts.quantum.phenomena.quantum_error_correction import (
            QuantumCode, ErrorSyndrome, LogicalQubit, QuantumErrorCorrector
        )
        print("✓ quantum_error_correction module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import quantum_error_correction: {e}")
        return False
    
    # Test topological_analysis import
    try:
        from python.mcts.quantum.phenomena.topological_analysis import (
            PersistentFeature, MorseCriticalPoint, TopologicalPhase, TopologicalAnalyzer
        )
        print("✓ topological_analysis module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import topological_analysis: {e}")
        return False
    
    print("\nAll modules imported successfully!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)