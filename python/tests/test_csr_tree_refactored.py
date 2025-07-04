"""Test suite for refactored CSRTree to ensure functionality is maintained"""

import pytest
import sys
import os

# Temporarily modify the import to use refactored version
original_csr_tree = sys.modules.get('mcts.gpu.csr_tree')

# Import the refactored version
from mcts.gpu import CSRTree, CSRTreeConfig

# Import test classes from original test file
from test_csr_tree import (
    TestCSRTreeBasics,
    TestCSRTreeAdvanced,
    TestCSRTreeBackup,
    TestCSRTreeRootShifting,
    TestCSRTreeEdgeCases,
    TestCSRTreeConsistency,
    test_integration_with_real_workload
)

# Override the imports in the test modules to use refactored version
import mcts.gpu.csr_tree
mcts.gpu.csr_tree.CSRTree = CSRTree
mcts.gpu.csr_tree.CSRTreeConfig = CSRTreeConfig


class TestRefactoredCSRTreeBasics(TestCSRTreeBasics):
    """Test basic operations with refactored CSRTree"""
    pass


class TestRefactoredCSRTreeAdvanced(TestCSRTreeAdvanced):
    """Test advanced operations with refactored CSRTree"""
    pass


class TestRefactoredCSRTreeBackup(TestCSRTreeBackup):
    """Test backup operations with refactored CSRTree"""
    pass


class TestRefactoredCSRTreeRootShifting(TestCSRTreeRootShifting):
    """Test root shifting with refactored CSRTree"""
    # shift_root is now fully implemented, tests will run from parent class


class TestRefactoredCSRTreeEdgeCases(TestCSRTreeEdgeCases):
    """Test edge cases with refactored CSRTree"""
    pass


class TestRefactoredCSRTreeConsistency(TestCSRTreeConsistency):
    """Test consistency with refactored CSRTree"""
    pass


def test_refactored_integration_with_real_workload():
    """Integration test with refactored CSRTree"""
    test_integration_with_real_workload()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])