#!/usr/bin/env python3
"""
AlphaZero (Omoknuni) Quantum MCTS Setup Script

This script provides one-step installation for the complete AlphaZero pipeline:
1. Compiles C++ game engines (Chess, Go, Gomoku)
2. Compiles custom CUDA kernels for accelerated MCTS
3. Installs Python package dependencies
4. Sets up the development environment

Usage:
    python setup.py install         # Full installation
    python setup.py develop         # Development installation
    python setup.py build_ext       # Build extensions only
    python setup.py clean           # Clean build artifacts

Requirements:
    - CMake 3.12+
    - CUDA Toolkit 11.0+ (optional, for GPU acceleration)
    - GCC/Clang with C++17 support
    - Python 3.8+
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop

# Version information
__version__ = "1.0.0"
__author__ = "AlphaZero Omoknuni Team"
__email__ = "support@omoknuni.ai"

class CMakeExtension(Extension):
    """Extension that uses CMake to build"""
    
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    """Custom build command that uses CMake"""
    
    def build_extension(self, ext):
        """Build the C++ extensions using CMake"""
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we check this.
        # Can be set with Cmake -G
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            f"-DBUILD_PYTHON_BINDINGS=ON",
            f"-DBUILD_TESTS=ON",
        ]
        
        # Add optimization flags for release builds
        if cfg == "Release":
            cmake_args.extend([
                "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -march=native -mtune=native -flto -funroll-loops -ffast-math",
                "-DCMAKE_C_FLAGS_RELEASE=-O3 -march=native -mtune=native -flto -funroll-loops -ffast-math",
                "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",  # Enable LTO
            ])
            
            # Add CUDA-specific optimization flags if CUDA is available
            if self.check_cuda_availability():
                cmake_args.extend([
                    "-DCMAKE_CUDA_FLAGS=-O3 --use_fast_math --ptxas-options=-v --generate-line-info",
                    "-DCMAKE_CUDA_ARCHITECTURES=60;61;70;75;80;86;89;90",  # Support multiple GPU architectures
                ])
        
        # Check for CUDA availability
        cuda_available = self.check_cuda_availability()
        if cuda_available:
            print("✓ CUDA detected - enabling GPU acceleration")
            cmake_args.extend([
                "-DWITH_TORCH=ON",
                "-DWITH_CUDNN=ON",
            ])
        else:
            print("⚠ CUDA not found - building CPU-only version")
            cmake_args.extend([
                "-DWITH_TORCH=OFF",
                "-DWITH_CUDNN=OFF",
            ])

        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja
                    ninja_executable_path = os.path.join(ninja.BIN_DIR, "ninja")
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]
            else:
                # Use all available cores
                import multiprocessing
                build_args += [f"-j{multiprocessing.cpu_count()}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print(f"Building C++ extensions in {build_temp}")
        print(f"CMake args: {' '.join(cmake_args)}")
        
        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", "--target", "alphazero_py"] + build_args, 
            cwd=build_temp, check=True
        )

        # Copy the built library to the correct location
        built_lib = None
        for pattern in ["alphazero_py*.so", "alphazero_py*.pyd", "alphazero_py*.dll"]:
            matches = list(build_temp.glob(f"**/{pattern}"))
            if matches:
                built_lib = matches[0]
                break
        
        if built_lib:
            target = Path(extdir) / built_lib.name
            print(f"Copying {built_lib} to {target}")
            shutil.copy2(built_lib, target)
        else:
            print("Warning: Could not find built library")

    def check_cuda_availability(self):
        """Check if CUDA is available for compilation"""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return True
        except FileNotFoundError:
            pass
        return False

class InstallCommand(install):
    """Custom install command"""
    
    def run(self):
        """Run the installation process"""
        print("🚀 Starting AlphaZero (Omoknuni) installation...")
        
        # Check system requirements
        self.check_requirements()
        
        # Compile CUDA kernels if available
        self.compile_cuda_kernels()
        
        # Run standard installation
        super().run()
        
        # Post-installation setup
        self.post_install()
        
        print("✅ Installation completed successfully!")
        print("\nQuick start:")
        print("  python -c 'import alphazero_py; print(\"C++ bindings loaded successfully\")'")
        print("  python python/example_self_play.py")

    def check_requirements(self):
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8+ is required")
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
        
        # Check CMake
        try:
            result = subprocess.run(['cmake', '--version'], capture_output=True)
            if result.returncode == 0:
                print("✓ CMake available")
            else:
                raise RuntimeError("CMake is required but not found")
        except FileNotFoundError:
            raise RuntimeError("CMake is required but not found")
        
        # Check compiler
        if platform.system() == "Windows":
            # On Windows, check for MSVC
            print("⚠ Windows detected - ensure Visual Studio 2019+ is installed")
        else:
            # On Unix-like systems, check for GCC/Clang
            for compiler in ['g++', 'clang++']:
                try:
                    result = subprocess.run([compiler, '--version'], capture_output=True)
                    if result.returncode == 0:
                        print(f"✓ {compiler} available")
                        break
                except FileNotFoundError:
                    continue
            else:
                print("⚠ No C++ compiler detected - compilation may fail")

    def compile_cuda_kernels(self):
        """Compile CUDA kernels if CUDA is available"""
        try:
            # Check if compile_kernels.py exists
            kernel_script = Path("python/compile_kernels.py")
            if kernel_script.exists():
                print("🔧 Compiling CUDA kernels...")
                result = subprocess.run([sys.executable, str(kernel_script)], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("✓ CUDA kernels compiled successfully")
                else:
                    print(f"⚠ CUDA kernel compilation failed: {result.stderr}")
            else:
                print("⚠ CUDA kernel compilation script not found")
        except Exception as e:
            print(f"⚠ CUDA kernel compilation failed: {e}")

    def post_install(self):
        """Post-installation setup"""
        print("🔧 Running post-installation setup...")
        print("✓ Post-installation setup completed")

class DevelopCommand(develop):
    """Custom develop command for development installation"""
    
    def run(self):
        """Run development installation"""
        print("🚀 Setting up development environment...")
        super().run()
        print("✅ Development environment ready!")

# Platform-specific settings
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_file = Path("python/requirements.txt")
    if requirements_file.exists():
        with open(requirements_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        # Fallback requirements
        return [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "numpy>=1.21.0",
            "pyyaml>=6.0",
            "tqdm>=4.64.0",
            "tensorboard>=2.9.0",
            "scipy>=1.8.0",
            "matplotlib>=3.5.0",
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "psutil>=5.9.0",
        ]

def main():
    """Main setup function"""
    
    # Read long description from README
    readme_file = Path("README.md")
    long_description = ""
    if readme_file.exists():
        with open(readme_file, encoding="utf-8") as f:
            long_description = f.read()
    
    setup(
        name="alphazero-omoknuni",
        version=__version__,
        author=__author__,
        author_email=__email__,
        description="High-Performance AlphaZero with Quantum-Enhanced MCTS",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/your-org/omoknuni_quantum",
        
        # Package discovery
        packages=find_packages(where="python"),
        package_dir={"": "python"},
        
        # C++ Extensions
        ext_modules=[CMakeExtension("alphazero_py")],
        cmdclass={
            "build_ext": CMakeBuild,
            "install": InstallCommand,
            "develop": DevelopCommand,
        },
        
        # Requirements
        python_requires=">=3.8",
        install_requires=read_requirements(),
        
        # Optional dependencies
        extras_require={
            "dev": [
                "pytest>=7.0.0",
                "black>=22.0.0", 
                "flake8>=4.0.0",
                "mypy>=0.950",
                "pre-commit>=2.19.0",
            ],
            "gpu": [
                "cupy-cuda11x>=10.0.0",  # For additional GPU utilities
            ],
            "analysis": [
                "jupyter>=1.0.0",
                "seaborn>=0.11.0",
                "plotly>=5.0.0",
            ]
        },
        
        # Entry points
        entry_points={
            "console_scripts": [
                "alphazero-train=mcts.neural_networks.unified_training_pipeline:main",
                "alphazero-selfplay=example_self_play:main",
                "alphazero-benchmark=benchmark_mcts_profiler:main",
            ],
        },
        
        # Package data
        include_package_data=True,
        package_data={
            "": ["*.yaml", "*.yml", "*.md", "*.txt"],
        },
        
        # Metadata
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: C++",
            "Programming Language :: CUDA",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Games/Entertainment :: Board Games",
        ],
        
        keywords="alphazero mcts reinforcement-learning quantum chess go gomoku",
        
        # Zip safe
        zip_safe=False,
    )

if __name__ == "__main__":
    main()