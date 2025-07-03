#!/usr/bin/env python3
"""Test runner script for MCTS project

This script provides convenient ways to run different test suites with appropriate configurations.
"""

import sys
import argparse
import subprocess
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run MCTS test suites")
    parser.add_argument(
        'suite', 
        choices=['unit', 'integration', 'all', 'core', 'utils', 'fast', 'slow', 'smoke'],
        help='Test suite to run'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-x', '--stop-on-fail', action='store_true', help='Stop on first failure')
    parser.add_argument('-k', '--keyword', help='Run tests matching keyword')
    parser.add_argument('--no-cov', action='store_true', help='Disable coverage reporting')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Base pytest command - use virtual environment python
    venv_python = os.path.expanduser('~/venv/bin/python')
    if os.path.exists(venv_python):
        base_cmd = [venv_python, '-m', 'pytest']
    else:
        base_cmd = ['python', '-m', 'pytest']
    
    # Add common flags
    if args.verbose:
        base_cmd.append('-v')
    else:
        base_cmd.append('-q')
    
    if args.stop_on_fail:
        base_cmd.append('-x')
    
    if args.keyword:
        base_cmd.extend(['-k', args.keyword])
    
    # Add coverage if not disabled
    if not args.no_cov:
        base_cmd.extend([
            '--cov=mcts',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov'
        ])
    
    # Add parallel execution if requested
    if args.parallel:
        base_cmd.extend(['-n', 'auto'])
    
    # Configure test paths based on suite
    success = True
    
    if args.suite == 'unit':
        # Run unit tests (core and utils)
        cmd = base_cmd + ['tests/core/', 'tests/utils/']
        success = run_command(cmd, "Unit Tests (Core + Utils)")
    
    elif args.suite == 'integration':
        # Run integration tests
        cmd = base_cmd + ['tests/integration/']
        success = run_command(cmd, "Integration Tests")
    
    elif args.suite == 'core':
        # Run core module tests only
        cmd = base_cmd + ['tests/core/']
        success = run_command(cmd, "Core Module Tests")
    
    elif args.suite == 'utils':
        # Run utils module tests only
        cmd = base_cmd + ['tests/utils/']
        success = run_command(cmd, "Utils Module Tests")
    
    elif args.suite == 'fast':
        # Run fast tests (exclude slow markers)
        cmd = base_cmd + ['-m', 'not slow', 'tests/']
        success = run_command(cmd, "Fast Tests")
    
    elif args.suite == 'slow':
        # Run only slow tests
        cmd = base_cmd + ['-m', 'slow', 'tests/']
        success = run_command(cmd, "Slow Tests")
    
    elif args.suite == 'smoke':
        # Run smoke tests (basic functionality)
        smoke_tests = [
            'tests/core/test_evaluator.py::TestMockEvaluator::test_mock_evaluator_creation',
            'tests/core/test_game_interface.py::TestGameInterface::test_game_interface_creation_gomoku',
            'tests/core/test_mcts.py::TestMCTSConfig::test_mcts_config_defaults',
            'tests/utils/test_config_system.py::TestAlphaZeroConfig::test_alphazero_config_creation_defaults'
        ]
        cmd = base_cmd + smoke_tests
        success = run_command(cmd, "Smoke Tests")
    
    elif args.suite == 'all':
        # Run all test suites in sequence
        test_suites = [
            (['tests/core/'], "Core Module Tests"),
            (['tests/utils/'], "Utils Module Tests"),
            (['tests/integration/'], "Integration Tests")
        ]
        
        for test_paths, description in test_suites:
            cmd = base_cmd + test_paths
            suite_success = run_command(cmd, description)
            if not suite_success:
                success = False
                if args.stop_on_fail:
                    break
    
    # Print summary
    print(f"\n{'='*60}")
    if success:
        print("ALL TESTS PASSED ✓")
        print("Test suite completed successfully!")
    else:
        print("SOME TESTS FAILED ✗")
        print("Check the output above for details.")
    print('='*60)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()