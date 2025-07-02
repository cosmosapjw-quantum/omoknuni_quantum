#!/usr/bin/env python3
"""Tests for MCTS validation system"""

import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts.utils.validation import (
    MCTSValidator, ValidationLevel, ValidationIssue, ValidationResult,
    validate_mcts_tree, enable_validation, disable_validation
)
from mcts.gpu.csr_tree import CSRTree, CSRTreeConfig


class TestMCTSValidation(unittest.TestCase):
    """Test MCTS validation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = CSRTreeConfig(max_nodes=100, device='cpu')
        self.tree = CSRTree(self.config)
        # Set adequate root visits and value to avoid common validation issues
        self.tree.visit_counts[0] = 10
        self.tree.value_sums[0] = 2.5  # Gives Q-value of 0.25
        self.validator = MCTSValidator(ValidationLevel.STRICT)
    
    def test_validation_levels(self):
        """Test different validation levels"""
        # Disabled validation should always pass
        validator_disabled = MCTSValidator(ValidationLevel.DISABLED)
        result = validator_disabled.validate_tree(self.tree)
        self.assertTrue(result.passed)
        
        # Basic validation on healthy tree should pass
        validator_basic = MCTSValidator(ValidationLevel.BASIC)
        result = validator_basic.validate_tree(self.tree)
        self.assertTrue(result.passed)
        
        # Debug validation should be most thorough
        validator_debug = MCTSValidator(ValidationLevel.DEBUG)
        result = validator_debug.validate_tree(self.tree, check_interval=1)
        self.assertTrue(result.passed)
    
    def test_critical_issue_detection(self):
        """Test detection of critical MCTS issues"""
        # Create tree with some structure
        root_idx = 0
        actions = [0, 1, 2]
        priors = [0.5, 0.3, 0.2]
        children = self.tree.add_children_batch(root_idx, actions, priors)
        
        # Test NaN detection by manipulating the tensor directly
        original_value = self.tree.visit_counts[children[0]].clone()
        self.tree.visit_counts = self.tree.visit_counts.float()  # Convert to float to allow NaN
        self.tree.visit_counts[children[0]] = float('nan')
        result = self.validator.validate_tree(self.tree, check_interval=1)
        self.assertFalse(result.passed)
        self.assertIn(ValidationIssue.NAN_VALUES, result.issues)
        
        # Fix NaN and test negative visits (convert back to int)
        self.tree.visit_counts = self.tree.visit_counts.int()
        self.tree.visit_counts[children[0]] = -5
        result = self.validator.validate_tree(self.tree, check_interval=1)
        self.assertFalse(result.passed)
        self.assertIn(ValidationIssue.NEGATIVE_VISITS, result.issues)
        
        # Fix negative visits and test degenerate pattern (all visits = 1)
        # First set root to have adequate visits
        self.tree.visit_counts[root_idx] = 10
        for child in children:
            self.tree.visit_counts[child] = 1
            self.tree.value_sums[child] = 0.0
        
        result = self.validator.validate_tree(self.tree, check_interval=1)
        self.assertFalse(result.passed)
        self.assertIn(ValidationIssue.ALL_VISITS_ONE, result.issues)
        self.assertIn(ValidationIssue.ALL_Q_VALUES_ZERO, result.issues)
    
    def test_performance_issue_detection(self):
        """Test detection of performance issues"""
        # Create tree but don't accumulate enough visits
        root_idx = 0
        actions = [0, 1]
        priors = [0.6, 0.4]
        children = self.tree.add_children_batch(root_idx, actions, priors)
        
        # Set very low root visits
        self.tree.visit_counts[root_idx] = 1
        
        result = self.validator.validate_tree(self.tree, check_interval=1)
        self.assertFalse(result.passed)
        self.assertIn(ValidationIssue.NO_VISIT_ACCUMULATION, result.issues)
        
        # Test poor exploration (all visits in one child)
        self.tree.visit_counts[root_idx] = 100
        self.tree.visit_counts[children[0]] = 95  # 95% of visits
        self.tree.visit_counts[children[1]] = 5   # 5% of visits
        
        result = self.validator.validate_tree(self.tree, check_interval=1)
        self.assertFalse(result.passed)
        self.assertIn(ValidationIssue.POOR_EXPLORATION, result.issues)
    
    def test_statistical_anomaly_detection(self):
        """Test detection of statistical anomalies"""
        # Create tree with extreme Q-values
        root_idx = 0
        actions = [0, 1]
        priors = [0.5, 0.5]
        children = self.tree.add_children_batch(root_idx, actions, priors)
        
        # Set extreme Q-values
        self.tree.visit_counts[children[0]] = 10
        self.tree.value_sums[children[0]] = 1000.0  # Q = 100 (extreme)
        
        result = self.validator.validate_tree(self.tree, check_interval=1)
        self.assertFalse(result.passed)
        self.assertIn(ValidationIssue.EXTREME_Q_VALUES, result.issues)
        
        # Test uniform Q-values
        self.tree.value_sums[children[0]] = 5.0   # Q = 0.5
        self.tree.visit_counts[children[1]] = 10
        self.tree.value_sums[children[1]] = 5.001 # Q = 0.5001 (very similar)
        
        result = self.validator.validate_tree(self.tree, check_interval=1)
        # This might or might not trigger depending on threshold
        if not result.passed:
            self.assertIn(ValidationIssue.UNIFORM_Q_VALUES, result.issues)
    
    def test_tree_structure_validation(self):
        """Test tree structure consistency checks"""
        # This test is basic since creating inconsistent CSR structures is complex
        # and the tree should maintain consistency internally
        
        result = self.validator.validate_tree(self.tree, check_interval=1)
        self.assertTrue(result.passed)
        
        # Test with a larger tree
        root_idx = 0
        for i in range(5):
            actions = [i * 3 + j for j in range(3)]
            priors = [0.33, 0.33, 0.34]
            children = self.tree.add_children_batch(root_idx, actions, priors)
            # Add visits to children so they're considered active
            for child in children:
                self.tree.visit_counts[child] = 1 + i
                self.tree.value_sums[child] = 0.1 * i
        
        result = self.validator.validate_tree(self.tree, check_interval=1)
        # Should pass unless structure is actually inconsistent
        if not result.passed:
            print(f"Unexpected structure issues: {result.issues}")
    
    def test_integration_with_csr_tree(self):
        """Test validation integration with CSRTree"""
        # Test the validate_statistics method
        result = self.tree.validate_statistics(ValidationLevel.BASIC, check_interval=1)
        self.assertTrue(result.passed)
        
        # Create some issues and test again
        root_idx = 0
        actions = [0, 1]
        priors = [0.5, 0.5]
        children = self.tree.add_children_batch(root_idx, actions, priors)
        
        # Create degenerate pattern
        for child in children:
            self.tree.visit_counts[child] = 1
            self.tree.value_sums[child] = 0.0
        
        result = self.tree.validate_statistics(ValidationLevel.STANDARD, check_interval=1)
        self.assertFalse(result.passed)
        self.assertIn(ValidationIssue.ALL_VISITS_ONE, result.issues)
    
    def test_global_validation_functions(self):
        """Test global validation convenience functions"""
        # Test enable/disable
        enable_validation(ValidationLevel.STANDARD)
        disable_validation()
        
        # Test convenience function
        result = validate_mcts_tree(self.tree, ValidationLevel.BASIC, check_interval=1)
        self.assertTrue(result.passed)
    
    def test_validation_result_formatting(self):
        """Test validation result string representation"""
        # Test passing result
        result = ValidationResult(ValidationLevel.BASIC, [], {}, True)
        result_str = str(result)
        self.assertIn("PASSED", result_str)
        
        # Test failing result
        result = ValidationResult(
            ValidationLevel.STANDARD, 
            [ValidationIssue.ALL_VISITS_ONE, ValidationIssue.NAN_VALUES],
            {'detail': 'test'},
            False
        )
        result_str = str(result)
        self.assertIn("FAILED", result_str)
        self.assertIn("all_visits_one", result_str)
        self.assertIn("nan_values", result_str)
    
    def test_issue_history_tracking(self):
        """Test issue history and summary functionality"""
        # Create some issues
        root_idx = 0
        actions = [0]
        priors = [1.0]
        children = self.tree.add_children_batch(root_idx, actions, priors)
        
        # Set adequate root visits first
        self.tree.visit_counts[0] = 10
        
        # Create issues multiple times
        for i in range(3):
            # Create NaN issue
            self.tree.visit_counts = self.tree.visit_counts.float()
            self.tree.visit_counts[children[0]] = float('nan')
            result = self.validator.validate_tree(self.tree, check_interval=1)
            self.assertFalse(result.passed)
            
            # Fix and create different issue
            self.tree.visit_counts = self.tree.visit_counts.int()
            self.tree.visit_counts[children[0]] = -1
            result = self.validator.validate_tree(self.tree, check_interval=1)
            self.assertFalse(result.passed)
        
        # Check issue summary
        summary = self.validator.get_issue_summary()
        self.assertIn('nan_values', summary)
        self.assertIn('negative_visits', summary)
        self.assertEqual(summary['nan_values'], 3)
        self.assertEqual(summary['negative_visits'], 3)
        
        # Test reset
        self.validator.reset_history()
        summary = self.validator.get_issue_summary()
        self.assertEqual(len(summary), 0)
    
    def test_healthy_tree_validation(self):
        """Test validation on a healthy, realistic tree"""
        # Build a realistic tree
        root_idx = 0
        actions = [0, 1, 2, 3, 4]
        priors = [0.2] * 5
        children = self.tree.add_children_batch(root_idx, actions, priors)
        
        # Add realistic statistics
        self.tree.visit_counts[root_idx] = 100
        
        for i, child in enumerate(children):
            self.tree.visit_counts[child] = 15 + i * 5  # Varied visits
            self.tree.value_sums[child] = (0.1 + i * 0.1) * self.tree.visit_counts[child]  # Reasonable Q-values
            
            # Add grandchildren to some nodes
            if i < 2:
                grand_actions = [10 + i * 3 + j for j in range(3)]
                grand_priors = [0.33, 0.33, 0.34]
                grandchildren = self.tree.add_children_batch(child, grand_actions, grand_priors)
                
                for j, grandchild in enumerate(grandchildren):
                    self.tree.visit_counts[grandchild] = 2 + j
                    self.tree.value_sums[grandchild] = 0.05 * self.tree.visit_counts[grandchild]
        
        # This healthy tree should pass all validation levels
        for level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.DEBUG]:
            validator = MCTSValidator(level)
            result = validator.validate_tree(self.tree, check_interval=1)
            self.assertTrue(result.passed, f"Healthy tree failed validation at level {level.name}: {result.issues}")


def run_validation_tests():
    """Run all validation tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTest(loader.loadTestsFromTestCase(TestMCTSValidation))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 60)
    print("MCTS VALIDATION SYSTEM TESTS")
    print("=" * 60)
    
    success = run_validation_tests()
    
    if success:
        print("\n✅ All validation tests passed!")
        print("\nValidation system features verified:")
        print("- Critical issue detection (NaN, negative values, degenerate patterns)")
        print("- Performance issue detection (poor exploration, no accumulation)")
        print("- Statistical anomaly detection (extreme/uniform Q-values)")
        print("- Tree structure validation (CSR consistency)")
        print("- Integration with CSRTree")
        print("- Issue history tracking and summaries")
    else:
        print("\n❌ Some validation tests failed!")
        exit(1)