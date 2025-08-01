#!/usr/bin/env python3
"""Run comprehensive MCTS test suite

This script runs all MCTS tests and provides a detailed report of:
- Test coverage for each backend (CPU, GPU, hybrid)
- Performance benchmarks
- Any failures or issues discovered
- Recommendations for fixes
"""

import sys
import os
import time
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pytest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Store test execution results"""
    test_file: str
    backend: Optional[str]
    passed: int
    failed: int
    skipped: int
    errors: List[str]
    duration: float
    performance_metrics: Dict[str, float]


class ComprehensiveTestRunner:
    """Run and analyze comprehensive MCTS tests"""
    
    def __init__(self):
        self.test_files = [
            "test_mcts_comprehensive.py",
            "test_cpu_backend_detailed.py", 
            "test_gpu_backend_detailed.py",
            "test_hybrid_backend_detailed.py",
            "test_mcts_phases_detailed.py"
        ]
        self.results: List[TestResult] = []
        self.performance_data: Dict[str, Dict[str, float]] = {
            'cpu': {},
            'gpu': {},
            'hybrid': {}
        }
    
    def run_all_tests(self) -> bool:
        """Run all test files and collect results"""
        logger.info("Starting comprehensive MCTS test suite...")
        
        all_passed = True
        
        for test_file in self.test_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {test_file}...")
            logger.info('='*60)
            
            result = self.run_test_file(test_file)
            self.results.append(result)
            
            if result.failed > 0 or result.errors:
                all_passed = False
        
        return all_passed
    
    def run_test_file(self, test_file: str) -> TestResult:
        """Run a single test file and parse results"""
        start_time = time.time()
        
        # Run pytest with JSON output
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file=test_report_{test_file}.json",
            "-p", "no:warnings"
        ]
        
        # Add coverage if available
        try:
            import pytest_cov
            cmd.extend(["--cov=mcts", "--cov-report=term-missing"])
        except ImportError:
            pass
        
        # Run tests
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        duration = time.time() - start_time
        
        # Parse results
        passed = failed = skipped = 0
        errors = []
        
        # Try to parse JSON report
        json_report_file = Path(__file__).parent / f"test_report_{test_file}.json"
        if json_report_file.exists():
            try:
                with open(json_report_file, 'r') as f:
                    report = json.load(f)
                    
                passed = report['summary'].get('passed', 0)
                failed = report['summary'].get('failed', 0)
                skipped = report['summary'].get('skipped', 0)
                
                # Extract errors
                for test in report.get('tests', []):
                    if test['outcome'] == 'failed':
                        errors.append(f"{test['nodeid']}: {test.get('call', {}).get('longrepr', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Failed to parse JSON report: {e}")
        
        # Fallback to parsing stdout
        if not (passed or failed or skipped):
            output_lines = process.stdout.split('\n')
            for line in output_lines:
                if 'passed' in line and 'failed' in line:
                    # Parse pytest summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'passed' in part and i > 0:
                            passed = int(parts[i-1])
                        elif 'failed' in part and i > 0:
                            failed = int(parts[i-1])
                        elif 'skipped' in part and i > 0:
                            skipped = int(parts[i-1])
        
        # Extract performance metrics from output
        perf_metrics = self.extract_performance_metrics(process.stdout)
        
        # Detect backend from test file name
        backend = None
        if 'cpu' in test_file.lower():
            backend = 'cpu'
        elif 'gpu' in test_file.lower():
            backend = 'gpu'
        elif 'hybrid' in test_file.lower():
            backend = 'hybrid'
        
        return TestResult(
            test_file=test_file,
            backend=backend,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors[:10],  # Limit errors shown
            duration=duration,
            performance_metrics=perf_metrics
        )
    
    def extract_performance_metrics(self, output: str) -> Dict[str, float]:
        """Extract performance metrics from test output"""
        metrics = {}
        
        # Look for patterns like "1234 sims/s" or "simulations/second: 1234"
        import re
        
        # Pattern for simulations per second
        sims_pattern = r'(\d+(?:\.\d+)?)\s*(?:sims?/s|simulations?/second)'
        matches = re.findall(sims_pattern, output, re.IGNORECASE)
        if matches:
            metrics['simulations_per_second'] = float(matches[-1])  # Take last match
        
        # Pattern for throughput
        throughput_pattern = r'throughput:\s*(\d+(?:\.\d+)?)'
        matches = re.findall(throughput_pattern, output, re.IGNORECASE)
        if matches:
            metrics['throughput'] = float(matches[-1])
        
        # Pattern for batch time
        batch_pattern = r'batch\s+time:\s*(\d+(?:\.\d+)?)\s*s'
        matches = re.findall(batch_pattern, output, re.IGNORECASE)
        if matches:
            metrics['batch_time'] = float(matches[-1])
        
        return metrics
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("\n" + "="*80)
        report.append("MCTS COMPREHENSIVE TEST REPORT")
        report.append("="*80 + "\n")
        
        # Summary
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_duration = sum(r.duration for r in self.results)
        
        report.append("SUMMARY")
        report.append("-"*40)
        report.append(f"Total Tests Run: {total_passed + total_failed + total_skipped}")
        report.append(f"Passed: {total_passed}")
        report.append(f"Failed: {total_failed}")
        report.append(f"Skipped: {total_skipped}")
        report.append(f"Total Duration: {total_duration:.2f}s")
        report.append("")
        
        # Per-file results
        report.append("DETAILED RESULTS")
        report.append("-"*40)
        
        for result in self.results:
            report.append(f"\n{result.test_file}:")
            report.append(f"  Backend: {result.backend or 'all'}")
            report.append(f"  Passed: {result.passed}")
            report.append(f"  Failed: {result.failed}")
            report.append(f"  Skipped: {result.skipped}")
            report.append(f"  Duration: {result.duration:.2f}s")
            
            if result.performance_metrics:
                report.append("  Performance:")
                for metric, value in result.performance_metrics.items():
                    report.append(f"    {metric}: {value:.2f}")
            
            if result.errors:
                report.append("  Errors:")
                for error in result.errors[:3]:  # Show first 3 errors
                    report.append(f"    - {error[:100]}...")
        
        # Performance comparison
        report.append("\n\nPERFORMANCE COMPARISON")
        report.append("-"*40)
        
        # Collect performance by backend
        backend_perf = {'cpu': [], 'gpu': [], 'hybrid': []}
        for result in self.results:
            if result.backend and 'simulations_per_second' in result.performance_metrics:
                backend_perf[result.backend].append(
                    result.performance_metrics['simulations_per_second']
                )
        
        for backend, perfs in backend_perf.items():
            if perfs:
                avg_perf = sum(perfs) / len(perfs)
                report.append(f"{backend.upper()}: {avg_perf:.0f} sims/s (avg)")
        
        # Issues and recommendations
        report.append("\n\nISSUES AND RECOMMENDATIONS")
        report.append("-"*40)
        
        if total_failed > 0:
            report.append(f"⚠️  {total_failed} tests failed!")
            report.append("\nCommon failure patterns:")
            
            # Analyze errors
            error_categories = self.categorize_errors()
            for category, count in error_categories.items():
                report.append(f"  - {category}: {count} occurrences")
            
            report.append("\nRecommended actions:")
            report.append("  1. Check error logs for specific failure details")
            report.append("  2. Run failing tests individually with -vv flag")
            report.append("  3. Verify CUDA availability for GPU tests")
            report.append("  4. Check memory limits for large tree tests")
        else:
            report.append("✅ All tests passed!")
        
        # Performance recommendations
        report.append("\nPERFORMANCE NOTES:")
        
        # Check if GPU is being utilized well
        gpu_perf = backend_perf.get('gpu', [0])
        cpu_perf = backend_perf.get('cpu', [0])
        
        if gpu_perf and cpu_perf:
            speedup = max(gpu_perf) / max(cpu_perf) if max(cpu_perf) > 0 else 0
            if speedup < 3:
                report.append(f"  - GPU speedup is only {speedup:.1f}x - consider optimization")
                report.append("    - Check batch sizes")
                report.append("    - Verify CUDA kernels are loaded")
                report.append("    - Profile GPU utilization")
        
        return '\n'.join(report)
    
    def categorize_errors(self) -> Dict[str, int]:
        """Categorize errors by type"""
        categories = {
            'assertion': 0,
            'timeout': 0,
            'memory': 0,
            'cuda': 0,
            'import': 0,
            'other': 0
        }
        
        for result in self.results:
            for error in result.errors:
                error_lower = error.lower()
                if 'assert' in error_lower:
                    categories['assertion'] += 1
                elif 'timeout' in error_lower:
                    categories['timeout'] += 1
                elif 'memory' in error_lower or 'oom' in error_lower:
                    categories['memory'] += 1
                elif 'cuda' in error_lower or 'gpu' in error_lower:
                    categories['cuda'] += 1
                elif 'import' in error_lower or 'module' in error_lower:
                    categories['import'] += 1
                else:
                    categories['other'] += 1
        
        return {k: v for k, v in categories.items() if v > 0}


def main():
    """Main test runner entry point"""
    # Change to tests directory
    os.chdir(Path(__file__).parent)
    
    # Create test runner
    runner = ComprehensiveTestRunner()
    
    # Run all tests
    logger.info("Running comprehensive MCTS test suite...")
    logger.info("This may take several minutes...\n")
    
    all_passed = runner.run_all_tests()
    
    # Generate report
    report = runner.generate_report()
    print(report)
    
    # Save report
    report_file = Path("mcts_test_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"\nReport saved to: {report_file}")
    
    # Return appropriate exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())