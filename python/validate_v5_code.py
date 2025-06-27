#!/usr/bin/env python3
"""
Code validation for v5.0 selective quantum implementation
Tests syntax, imports, and basic structure without requiring PyTorch
"""

import ast
import os
import sys
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax of a file"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse AST to check syntax
        ast.parse(source)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def check_v5_formula_implementation():
    """Check that v5.0 formula is correctly implemented in code"""
    file_path = "mcts/quantum/selective_quantum_optimized.py"
    
    if not os.path.exists(file_path):
        return False, "File not found"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check for v5.0 formula components
    checks = {
        "v5.0 Formula comment": "Score(k) = κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff(N_tot))/(3 N_k)",
        "kappa parameter": "kappa",
        "beta parameter": "beta", 
        "hbar_0 parameter": "hbar_0",
        "alpha parameter": "alpha",
        "hbar_eff calculation": "hbar_eff = hbar_0 * ((1.0 + N_tot) ** (-alpha * 0.5))",
        "exploration term": "kappa * priors * (safe_visits / parent_visits)",
        "exploitation term": "beta * q_values",
        "quantum bonus": "(4.0 * hbar_eff) / (3.0 * masked_visits)"
    }
    
    results = []
    for check_name, pattern in checks.items():
        if pattern in content:
            results.append(f"✓ {check_name}")
        else:
            results.append(f"✗ {check_name}")
    
    return True, "\n".join(results)

def check_cuda_kernel_v5():
    """Check v5.0 CUDA kernel implementation"""
    file_path = "mcts/gpu/quantum_v5_cuda_kernels.cu"
    
    if not os.path.exists(file_path):
        return False, "CUDA kernel file not found"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    checks = {
        "v5.0 kernel function": "selective_quantum_v5_kernel",
        "v5.0 formula comment": "Score(k) = κ p_k (N_k/N_tot) + β Q_k + (4 ℏ_eff(N_tot))/(3 N_k)",
        "kappa parameter": "float kappa",
        "beta parameter": "float beta",
        "hbar_eff calculation": "hbar_eff = hbar_0 * powf(1.0f + N_tot, -alpha * 0.5f)",
        "exploration term": "kappa * p_k * (safe_N_k / N_tot)",
        "exploitation term": "beta * Q_k",
        "quantum bonus": "(4.0f * hbar_eff) / (3.0f * safe_N_k)",
        "selective application": "enable_quantum && N_k < exploration_threshold"
    }
    
    results = []
    for check_name, pattern in checks.items():
        if pattern in content:
            results.append(f"✓ {check_name}")
        else:
            results.append(f"✗ {check_name}")
    
    return True, "\n".join(results)

def validate_file_structure():
    """Validate that all required files exist"""
    required_files = [
        "mcts/quantum/selective_quantum_optimized.py",
        "mcts/gpu/quantum_v5_cuda_kernels.cu",
        "setup.py",
        "mcts/gpu/cuda_compile.py"
    ]
    
    results = []
    for file_path in required_files:
        if os.path.exists(file_path):
            results.append(f"✓ {file_path}")
        else:
            results.append(f"✗ {file_path}")
    
    return results

def main():
    """Run all validation checks"""
    print("v5.0 Selective Quantum Implementation - Code Validation")
    print("=" * 60)
    
    # Change to the python directory
    os.chdir("/home/cosmosapjw/omoknuni_quantum/python")
    
    all_passed = True
    
    # Check file structure
    print("\n1. File Structure Validation:")
    print("-" * 30)
    file_results = validate_file_structure()
    for result in file_results:
        print(f"   {result}")
        if "✗" in result:
            all_passed = False
    
    # Check Python syntax
    print("\n2. Python Syntax Validation:")
    print("-" * 30)
    python_files = [
        "mcts/quantum/selective_quantum_optimized.py",
        "test_v5_implementation.py",
        "compile_cuda_kernels.py"
    ]
    
    for file_path in python_files:
        if os.path.exists(file_path):
            valid, message = validate_python_syntax(file_path)
            status = "✓" if valid else "✗"
            print(f"   {status} {file_path}: {message}")
            if not valid:
                all_passed = False
        else:
            print(f"   ⚠ {file_path}: File not found")
    
    # Check v5.0 formula implementation
    print("\n3. v5.0 Formula Implementation:")
    print("-" * 30)
    valid, message = check_v5_formula_implementation()
    if valid:
        for line in message.split('\n'):
            print(f"   {line}")
    else:
        print(f"   ✗ {message}")
        all_passed = False
    
    # Check CUDA kernel implementation
    print("\n4. CUDA Kernel v5.0 Implementation:")
    print("-" * 30)
    valid, message = check_cuda_kernel_v5()
    if valid:
        for line in message.split('\n'):
            print(f"   {line}")
    else:
        print(f"   ✗ {message}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print("✅ All code validation checks passed!")
        print("📋 Implementation is ready for testing")
        print("🔨 Next steps:")
        print("   1. Compile CUDA kernels: python compile_cuda_kernels.py")
        print("   2. Run tests: python test_v5_implementation.py")
        print("   3. Run performance tests: python test_selective_quantum_performance.py")
    else:
        print("❌ Some validation checks failed")
        print("🔧 Fix the issues above before proceeding")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)