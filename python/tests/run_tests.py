#!/usr/bin/env python3
"""
Convenient test runner for MCTS test suite

This script provides easy commands to run different test configurations.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "python"))


def run_command(cmd, description=None):
    """Run a command and handle output"""
    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print('='*60)
    
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="MCTS Test Suite Runner")
    
    # Test selection
    parser.add_argument("target", nargs="?", default="all",
                        choices=["all", "core", "gpu", "utils", "integration", "quick"],
                        help="Which tests to run")
    
    # Options
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--coverage", action="store_true",
                        help="Generate coverage report")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Run on CPU only")
    parser.add_argument("--slow", action="store_true",
                        help="Include slow tests")
    parser.add_argument("--integration", action="store_true",
                        help="Include integration tests")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarks")
    parser.add_argument("-k", "--keyword", type=str,
                        help="Run tests matching keyword")
    parser.add_argument("--pdb", action="store_true",
                        help="Drop into debugger on failure")
    parser.add_argument("--profile", action="store_true",
                        help="Profile test execution")
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["pytest"]
    
    # Add test path based on target
    test_paths = {
        "all": ["python/tests/"],
        "core": ["python/tests/test_core/"],
        "gpu": ["python/tests/test_gpu/"],
        "utils": ["python/tests/test_utils/"],
        "integration": ["python/tests/test_integration/"],
        "quick": ["python/tests/", "-m", "not slow and not integration"]
    }
    
    cmd.extend(test_paths[args.target])
    
    # Add options
    if args.verbose:
        cmd.append("-v")
        
    if args.coverage:
        cmd.extend(["--cov=mcts", "--cov-report=html", "--cov-report=term-missing"])
        
    if args.cpu_only:
        cmd.append("--cpu-only")
        
    if args.slow:
        cmd.append("--slow")
        
    if args.integration and args.target != "integration":
        cmd.append("--integration")
        
    if args.benchmark:
        cmd.append("--benchmark-only")
        
    if args.keyword:
        cmd.extend(["-k", args.keyword])
        
    if args.pdb:
        cmd.append("--pdb")
        
    if args.profile:
        cmd.append("--profile")
    
    # Set environment variables
    env = os.environ.copy()
    if args.cpu_only:
        env["DISABLE_CUDA_KERNELS"] = "1"
    
    # Run tests
    print(f"Running MCTS Test Suite")
    print(f"Target: {args.target}")
    print(f"Project root: {project_root}")
    
    result = run_command(cmd, f"{args.target} tests")
    
    # Print summary
    print(f"\n{'='*60}")
    if result == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        
    if args.coverage:
        print(f"\nCoverage report generated at: {project_root}/htmlcov/index.html")
        
    return result


if __name__ == "__main__":
    sys.exit(main())