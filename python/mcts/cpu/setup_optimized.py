from setuptools import setup
from Cython.Build import cythonize
import numpy as np
from distutils.extension import Extension

# Compiler flags for optimization
extra_compile_args = [
    '-O3',
    '-march=native',
    '-ffast-math',
    '-fopenmp',
    '-std=c99',
]

extra_link_args = [
    '-fopenmp',
]

# Define extension
extensions = [
    Extension(
        "cython_tree_optimized",
        ["cython_tree_optimized.pyx"],
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
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True,
        },
    ),
    zip_safe=False,
)

print("\nOptimized Cython tree module built successfully!")