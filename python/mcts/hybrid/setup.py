from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension
import os

# Compiler flags for maximum optimization
extra_compile_args = [
    '-O3',                # Maximum optimization
    '-march=native',      # Use native CPU instructions
    '-ffast-math',        # Fast floating point (slight accuracy tradeoff)
    '-fopenmp',          # Enable OpenMP for parallel execution
    '-std=c99',          # C99 standard
    '-funroll-loops',    # Loop unrolling
    '-ftree-vectorize',  # Auto-vectorization
]

extra_link_args = [
    '-fopenmp',          # Link OpenMP library
]

# Define extension
extensions = [
    Extension(
        "cython_hybrid_backend",
        ["cython_hybrid_backend.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c",
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,      # Disable bounds checking
            'wraparound': False,        # Disable negative indexing
            'initializedcheck': False,  # Disable initialized checks
            'nonecheck': False,         # Disable None checks
            'cdivision': True,          # Use C division
            'profile': False,           # Disable profiling
            'embedsignature': True,     # Include function signatures
        },
    ),
    zip_safe=False,
)

print("\n✅ Cython hybrid backend module built successfully!")
print("   Optimizations enabled:")
print("   - OpenMP parallel execution")
print("   - nogil operations for critical paths")
print("   - Vectorized batch processing")
print("   - Native CPU instructions")
print("   - Aggressive compiler optimizations")