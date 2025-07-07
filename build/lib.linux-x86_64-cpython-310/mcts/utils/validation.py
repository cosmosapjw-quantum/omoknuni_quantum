"""Runtime validation system for MCTS to detect degenerate statistics early

This module provides validation functions to detect common MCTS issues during runtime,
allowing for early intervention and debugging.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    DISABLED = 0      # No validation
    BASIC = 1         # Only critical issues
    STANDARD = 2      # Common issues
    STRICT = 3        # All issues including performance warnings
    DEBUG = 4         # Maximum validation for debugging


class ValidationIssue(Enum):
    """Types of validation issues"""
    # Critical issues that break MCTS
    ALL_VISITS_ONE = "all_visits_one"
    ALL_Q_VALUES_ZERO = "all_q_values_zero"
    NEGATIVE_VISITS = "negative_visits"
    NAN_VALUES = "nan_values"
    INFINITE_VALUES = "infinite_values"
    
    # Performance issues
    NO_VISIT_ACCUMULATION = "no_visit_accumulation"
    POOR_EXPLORATION = "poor_exploration"
    EXCESSIVE_TREE_GROWTH = "excessive_tree_growth"
    
    # Statistical anomalies
    UNIFORM_Q_VALUES = "uniform_q_values"
    EXTREME_Q_VALUES = "extreme_q_values"
    VISIT_CONCENTRATION = "visit_concentration"
    
    # Tree structure issues
    ORPHANED_NODES = "orphaned_nodes"
    CIRCULAR_REFERENCES = "circular_references"
    INCONSISTENT_CSR = "inconsistent_csr"


@dataclass
class ValidationResult:
    """Result of validation check"""
    level: ValidationLevel
    issues: List[ValidationIssue]
    details: Dict[str, Any]
    passed: bool
    
    def __str__(self):
        if self.passed:
            return f"Validation PASSED (level: {self.level.name})"
        else:
            issue_names = [issue.value for issue in self.issues]
            return f"Validation FAILED (level: {self.level.name}, issues: {issue_names})"


class MCTSValidator:
    """Runtime validator for MCTS statistics and tree structure"""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STANDARD):
        self.level = level
        self.validation_count = 0
        self.issue_history = []
        
        # Thresholds for validation (configurable)
        self.thresholds = {
            'min_root_visits': 2,
            'max_q_value': 10.0,
            'min_q_value': -10.0,
            'max_tree_nodes': 100000,
            'visit_concentration_threshold': 0.9,  # 90% of visits in one child
            'uniform_q_threshold': 0.01,  # Q-values within this range are "uniform"
        }
    
    def validate_tree(self, tree, check_interval: int = 1000) -> ValidationResult:
        """Validate CSR tree statistics and structure
        
        Args:
            tree: CSRTree instance to validate
            check_interval: Only validate every N calls (for performance)
            
        Returns:
            ValidationResult with findings
        """
        self.validation_count += 1
        
        # Skip validation based on interval
        if self.validation_count % check_interval != 0 and self.level != ValidationLevel.DEBUG:
            return ValidationResult(self.level, [], {}, True)
        
        if self.level == ValidationLevel.DISABLED:
            return ValidationResult(self.level, [], {}, True)
        
        issues = []
        details = {}
        
        try:
            # Basic validation (all levels)
            if self.level.value >= ValidationLevel.BASIC.value:
                issues.extend(self._check_critical_issues(tree, details))
            
            # Standard validation
            if self.level.value >= ValidationLevel.STANDARD.value:
                issues.extend(self._check_performance_issues(tree, details))
                issues.extend(self._check_statistical_anomalies(tree, details))
            
            # Strict validation
            if self.level.value >= ValidationLevel.STRICT.value:
                issues.extend(self._check_tree_structure(tree, details))
            
            # Debug validation
            if self.level.value >= ValidationLevel.DEBUG.value:
                issues.extend(self._check_debug_issues(tree, details))
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            issues.append(ValidationIssue.INCONSISTENT_CSR)
            details['validation_error'] = str(e)
        
        passed = len(issues) == 0
        result = ValidationResult(self.level, issues, details, passed)
        
        if not passed:
            self.issue_history.append(result)
            self._log_issues(result)
        
        return result
    
    def _check_critical_issues(self, tree, details: Dict) -> List[ValidationIssue]:
        """Check for critical issues that break MCTS functionality"""
        issues = []
        
        if tree.num_nodes == 0:
            return issues  # Empty tree is valid
        
        # Check for NaN values
        if torch.any(torch.isnan(tree.node_data.visit_counts[:tree.num_nodes])):
            issues.append(ValidationIssue.NAN_VALUES)
            details['nan_visit_counts'] = torch.sum(torch.isnan(tree.node_data.visit_counts[:tree.num_nodes])).item()
        
        if torch.any(torch.isnan(tree.node_data.value_sums[:tree.num_nodes])):
            issues.append(ValidationIssue.NAN_VALUES)
            details['nan_value_sums'] = torch.sum(torch.isnan(tree.node_data.value_sums[:tree.num_nodes])).item()
        
        # Check for infinite values
        if torch.any(torch.isinf(tree.node_data.visit_counts[:tree.num_nodes])):
            issues.append(ValidationIssue.INFINITE_VALUES)
        
        if torch.any(torch.isinf(tree.node_data.value_sums[:tree.num_nodes])):
            issues.append(ValidationIssue.INFINITE_VALUES)
        
        # Check for negative visits
        if torch.any(tree.node_data.visit_counts[:tree.num_nodes] < 0):
            issues.append(ValidationIssue.NEGATIVE_VISITS)
            details['negative_visits'] = torch.sum(tree.node_data.visit_counts[:tree.num_nodes] < 0).item()
        
        # Check for degenerate pattern: all visits = 1
        active_nodes = tree.node_data.visit_counts[:tree.num_nodes] > 0
        if torch.any(active_nodes):
            active_visits = tree.node_data.visit_counts[:tree.num_nodes][active_nodes]
            if torch.all(active_visits == 1):
                issues.append(ValidationIssue.ALL_VISITS_ONE)
                details['active_nodes_with_visits'] = torch.sum(active_nodes).item()
        
        # Check for degenerate pattern: all Q-values = 0
        if torch.any(active_nodes):
            active_values = tree.node_data.value_sums[:tree.num_nodes][active_nodes]
            active_visits_vals = tree.node_data.visit_counts[:tree.num_nodes][active_nodes]
            q_values = active_values / torch.clamp(active_visits_vals, min=1)
            
            if torch.all(torch.abs(q_values) < 1e-10):
                issues.append(ValidationIssue.ALL_Q_VALUES_ZERO)
                details['zero_q_values'] = torch.sum(torch.abs(q_values) < 1e-10).item()
        
        return issues
    
    def _check_performance_issues(self, tree, details: Dict) -> List[ValidationIssue]:
        """Check for performance-related issues"""
        issues = []
        
        if tree.num_nodes == 0:
            return issues
        
        # Check root visit accumulation
        root_visits = tree.node_data.visit_counts[0].item()
        if root_visits < self.thresholds['min_root_visits']:
            issues.append(ValidationIssue.NO_VISIT_ACCUMULATION)
            details['root_visits'] = root_visits
        
        # Check for excessive tree growth
        if tree.num_nodes > self.thresholds['max_tree_nodes']:
            issues.append(ValidationIssue.EXCESSIVE_TREE_GROWTH)
            details['tree_nodes'] = tree.num_nodes
        
        # Check exploration pattern (if root has children)
        try:
            root_children, _, _ = tree.get_children(0)
            if len(root_children) > 1:
                child_visits = tree.node_data.visit_counts[root_children]
                total_child_visits = torch.sum(child_visits).item()
                
                if total_child_visits > 0:
                    max_child_visits = torch.max(child_visits).item()
                    concentration = max_child_visits / total_child_visits
                    
                    if concentration > self.thresholds['visit_concentration_threshold']:
                        issues.append(ValidationIssue.POOR_EXPLORATION)
                        details['visit_concentration'] = concentration
                        details['child_visits'] = child_visits.tolist()
        except Exception:
            # get_children might fail for various reasons
            pass
        
        return issues
    
    def _check_statistical_anomalies(self, tree, details: Dict) -> List[ValidationIssue]:
        """Check for statistical anomalies in MCTS data"""
        issues = []
        
        if tree.num_nodes <= 1:
            return issues
        
        # Check for extreme Q-values
        active_mask = tree.node_data.visit_counts[:tree.num_nodes] > 0
        if torch.any(active_mask):
            active_visits = tree.node_data.visit_counts[:tree.num_nodes][active_mask]
            active_values = tree.node_data.value_sums[:tree.num_nodes][active_mask]
            q_values = active_values / torch.clamp(active_visits, min=1)
            
            max_q = torch.max(q_values).item()
            min_q = torch.min(q_values).item()
            
            if max_q > self.thresholds['max_q_value'] or min_q < self.thresholds['min_q_value']:
                issues.append(ValidationIssue.EXTREME_Q_VALUES)
                details['q_value_range'] = (min_q, max_q)
            
            # Check for overly uniform Q-values
            if len(q_values) > 1:
                q_std = torch.std(q_values).item()
                if q_std < self.thresholds['uniform_q_threshold']:
                    issues.append(ValidationIssue.UNIFORM_Q_VALUES)
                    details['q_value_std'] = q_std
        
        return issues
    
    def _check_tree_structure(self, tree, details: Dict) -> List[ValidationIssue]:
        """Check tree structure consistency"""
        issues = []
        
        # Check CSR structure consistency
        try:
            if hasattr(tree, 'row_ptr') and hasattr(tree, 'col_indices'):
                # Basic CSR consistency
                if tree.row_ptr[tree.num_nodes].item() > len(tree.col_indices):
                    issues.append(ValidationIssue.INCONSISTENT_CSR)
                    details['csr_inconsistency'] = 'row_ptr exceeds col_indices length'
                
                # Check for self-references (node pointing to itself)
                for i in range(tree.num_nodes):
                    start = tree.row_ptr[i].item()
                    end = tree.row_ptr[i + 1].item()
                    children = tree.col_indices[start:end]
                    if i in children:
                        issues.append(ValidationIssue.CIRCULAR_REFERENCES)
                        details['circular_node'] = i
                        break
        
        except Exception as e:
            issues.append(ValidationIssue.INCONSISTENT_CSR)
            details['csr_error'] = str(e)
        
        return issues
    
    def _check_debug_issues(self, tree, details: Dict) -> List[ValidationIssue]:
        """Comprehensive debug-level checks"""
        issues = []
        
        # Check for orphaned nodes (nodes with visits but unreachable from root)
        if tree.num_nodes > 1:
            reachable = set([0])  # Start with root
            to_visit = [0]
            
            try:
                while to_visit:
                    node = to_visit.pop()
                    children, _, _ = tree.get_children(node)
                    for child in children:
                        child_idx = child.item() if hasattr(child, 'item') else child
                        if child_idx not in reachable and child_idx < tree.num_nodes:
                            reachable.add(child_idx)
                            to_visit.append(child_idx)
                
                # Check for nodes with visits that are not reachable
                for i in range(tree.num_nodes):
                    if i not in reachable and tree.node_data.visit_counts[i].item() > 0:
                        issues.append(ValidationIssue.ORPHANED_NODES)
                        details['orphaned_nodes'] = details.get('orphaned_nodes', []) + [i]
                        
            except Exception:
                # If tree traversal fails, structure might be inconsistent
                pass
        
        return issues
    
    def _log_issues(self, result: ValidationResult):
        """Log validation issues"""
        if result.issues:
            logger.warning(f"MCTS Validation Issues Detected: {[issue.value for issue in result.issues]}")
            for issue in result.issues:
                issue_details = result.details.get(issue.value, {})
                logger.warning(f"  - {issue.value}: {issue_details}")
    
    def get_issue_summary(self) -> Dict[str, int]:
        """Get summary of issues encountered"""
        issue_counts = {}
        for result in self.issue_history:
            for issue in result.issues:
                issue_counts[issue.value] = issue_counts.get(issue.value, 0) + 1
        return issue_counts
    
    def reset_history(self):
        """Reset issue history"""
        self.issue_history = []
        self.validation_count = 0


# Global validator instance
_global_validator = None


def get_validator(level: ValidationLevel = ValidationLevel.STANDARD) -> MCTSValidator:
    """Get global validator instance"""
    global _global_validator
    if _global_validator is None or _global_validator.level != level:
        _global_validator = MCTSValidator(level)
    return _global_validator


def validate_mcts_tree(tree, level: ValidationLevel = ValidationLevel.STANDARD, 
                      check_interval: int = 1000) -> ValidationResult:
    """Convenience function to validate MCTS tree"""
    validator = get_validator(level)
    return validator.validate_tree(tree, check_interval)


def enable_validation(level: ValidationLevel = ValidationLevel.STANDARD):
    """Enable global MCTS validation"""
    global _global_validator
    _global_validator = MCTSValidator(level)
    logger.info(f"MCTS validation enabled at level: {level.name}")


def disable_validation():
    """Disable global MCTS validation"""
    global _global_validator
    _global_validator = MCTSValidator(ValidationLevel.DISABLED)
    logger.info("MCTS validation disabled")