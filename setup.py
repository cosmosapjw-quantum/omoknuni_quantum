#!/usr/bin/env python3
"""
Auto-adaptive setup script for AlphaZero Omoknuni
Automatically detects Python version and adapts compilation accordingly
"""

import os
import sys
import subprocess
import platform
import time
import shutil
import sysconfig
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools.command.develop import develop


def get_python_info() -> Dict[str, Any]:
    """Get comprehensive Python environment information"""
    info = {
        "version": sys.version_info,
        "version_string": f"{sys.version_info.major}.{sys.version_info.minor}",
        "executable": sys.executable,
        "prefix": sys.prefix,
        "include_dir": sysconfig.get_path("include"),
        "stdlib_dir": sysconfig.get_path("stdlib"),
        "platlib_dir": sysconfig.get_path("platlib"),
        "purelib_dir": sysconfig.get_path("purelib"),
        "get_python_inc": sysconfig.get_path("include"),
        "in_virtualenv": hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        ),
    }
    
    # Get include directories
    include_dirs = []
    
    # Primary include directory
    include_dirs.append(info["include_dir"])
    
    # Try to find Python.h
    python_h_locations = [
        Path(info["include_dir"]) / "Python.h",
        Path(sys.prefix) / "include" / f"python{info['version_string']}" / "Python.h",
        Path(sys.prefix) / "include" / f"python{info['version_string']}m" / "Python.h",
        Path("/usr/include") / f"python{info['version_string']}" / "Python.h",
        Path("/usr/local/include") / f"python{info['version_string']}" / "Python.h",
    ]
    
    for loc in python_h_locations:
        if loc.exists():
            info["python_h_dir"] = str(loc.parent)
            if str(loc.parent) not in include_dirs:
                include_dirs.append(str(loc.parent))
            break
    
    info["include_dirs"] = include_dirs
    
    # Get library information
    info["ldlibrary"] = sysconfig.get_config_var("LDLIBRARY")
    info["library_dir"] = sysconfig.get_config_var("LIBDIR")
    
    return info


class CMakeExtension(Extension):
    """Extension that uses CMake for building"""
    
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).absolute()


class SmartBuildExt(build_ext):
    """Smart build extension that auto-adapts to the environment"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.python_info = get_python_info()
        self.cuda_available = self._check_cuda()
        self.cmake_available = self._check_cmake()
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            result = subprocess.run(
                ["nvcc", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print("✓ CUDA detected")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  CUDA not found - GPU acceleration will be disabled")
            return False
    
    def _check_cmake(self) -> bool:
        """Check if CMake is available"""
        try:
            subprocess.run(["cmake", "--version"], capture_output=True, check=True)
            print("✓ CMake detected")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  CMake not found - C++ extensions will be skipped")
            return False
    
    def run(self):
        """Run the build process"""
        print(f"\n🐍 Python Environment:")
        print(f"   Version: {self.python_info['version_string']}")
        print(f"   Executable: {self.python_info['executable']}")
        print(f"   Include: {self.python_info['include_dir']}")
        print(f"   Virtual env: {'Yes' if self.python_info['in_virtualenv'] else 'No'}")
        
        # Build C++ modules first using build_cpp.py
        self._build_cpp_modules_with_script()
        
        # Build CUDA kernels if available
        if self.cuda_available:
            self._build_cuda_kernels_with_script()
    
    def build_extension(self, ext: CMakeExtension):
        """Build a single extension"""
        # Extensions are now built via build_cpp.py in run() method
        pass
    
    def _build_cpp_modules_with_script(self):
        """Build C++ modules using the build_cpp.py script"""
        print("\n🔧 Building C++ modules with build_cpp.py...")
        
        build_script = Path("build_cpp.py")
        if not build_script.exists():
            print("⚠️  build_cpp.py not found")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(build_script)],
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ C++ modules built successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  C++ module build failed: {e.stderr}")
            return False
    
    def _build_cuda_kernels_with_script(self):
        """Build CUDA kernels using PyTorch extension builder"""
        print("\n🔧 Building CUDA kernels with PyTorch...")
        
        # Try the new PyTorch-based builder first
        build_script = Path("build_cuda_torch.py")
        if not build_script.exists():
            # Fallback to old script
            build_script = Path("build_cuda.py")
            
        if not build_script.exists():
            print("⚠️  CUDA build script not found")
            return True  # Continue without CUDA
        
        try:
            # Check if PyTorch is available
            try:
                import torch
                if not torch.cuda.is_available():
                    print("⚠️  PyTorch CUDA not available - skipping CUDA kernel build")
                    return True
            except ImportError:
                print("⚠️  PyTorch not found - skipping CUDA kernel build")
                return True
            
            # Run build script
            print(f"   Using {build_script.name}...")
            result = subprocess.run(
                [sys.executable, str(build_script)],
                check=False  # Don't fail on error
            )
            if result.returncode == 0:
                print("✅ CUDA kernels built successfully")
                return True
            else:
                print("⚠️  CUDA kernel build failed - continuing without GPU acceleration")
                print("   (This is normal if you don't have CUDA or if using CPU-only)")
                return True  # Always return True to continue installation
        except Exception as e:
            print(f"⚠️  CUDA kernel build error: {e}")
            print("   Continuing without GPU acceleration")
            return True  # Always return True to continue installation
    
    def _build_cuda_kernels(self):
        """Build CUDA kernels with auto-detected configuration"""
        print("\n🔧 Building CUDA kernels...")
        
        # Find CUDA files
        cuda_files = []
        search_dirs = [
            Path("cuda"),
            Path("python/mcts/gpu"),
            Path("python/mcts/cuda"),
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                cuda_files.extend(search_dir.glob("**/*.cu"))
        
        if not cuda_files:
            print("⚠️  No CUDA files found")
            return
        
        print(f"📁 Found {len(cuda_files)} CUDA files")
        
        # Build output directory
        build_dir = Path(self.build_lib) / "mcts" / "cuda"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Get include directories
        include_dirs = self._get_cuda_include_dirs()
        
        # Compile each CUDA file
        for cuda_file in cuda_files:
            if not self._compile_cuda_file(cuda_file, build_dir, include_dirs):
                print(f"⚠️  Failed to compile {cuda_file.name}")
                return
        
        print("✅ CUDA kernels built successfully")
    
    def _get_cuda_include_dirs(self) -> List[str]:
        """Get include directories for CUDA compilation"""
        include_dirs = []
        
        # Python includes - use detected paths
        include_dirs.extend(self.python_info['include_dirs'])
        
        # PyTorch includes
        try:
            import torch
            torch_path = Path(torch.__path__[0])
            torch_includes = [
                torch_path / "include",
                torch_path / "include/torch/csrc/api/include", 
                torch_path / "include/TH",
                torch_path / "include/THC",
            ]
            include_dirs.extend([str(p) for p in torch_includes if p.exists()])
        except ImportError:
            print("⚠️  PyTorch not found - some features may be limited")
        
        # CUDA includes
        cuda_paths = [
            os.environ.get("CUDA_HOME", "/usr/local/cuda"),
            "/usr/local/cuda",
            "/opt/cuda",
        ]
        
        for cuda_path in cuda_paths:
            cuda_inc = Path(cuda_path) / "include"
            if cuda_inc.exists():
                include_dirs.append(str(cuda_inc))
                break
        
        # NumPy includes
        try:
            import numpy as np
            include_dirs.append(np.get_include())
        except ImportError:
            pass
        
        # Project includes
        project_includes = [
            "cuda/include",
            "cpp/include",
            "python/mcts/gpu",
        ]
        include_dirs.extend([
            str(Path(p).absolute()) 
            for p in project_includes 
            if Path(p).exists()
        ])
        
        return include_dirs
    
    def _compile_cuda_file(self, cuda_file: Path, output_dir: Path, 
                          include_dirs: List[str]) -> bool:
        """Compile a single CUDA file"""
        print(f"  Compiling {cuda_file.name}...")
        
        # Detect CUDA compute capability
        compute_caps = self._detect_cuda_arch()
        
        # Build nvcc command
        nvcc_cmd = ["nvcc", "-c", "-O3", "--use_fast_math"]
        
        # Add architectures
        for cap in compute_caps:
            nvcc_cmd.extend(["-gencode", f"arch=compute_{cap},code=sm_{cap}"])
        
        # Add includes
        for inc_dir in include_dirs:
            nvcc_cmd.extend(["-I", inc_dir])
        
        # Add flags
        nvcc_cmd.extend([
            "-std=c++17",
            "--compiler-options", "-fPIC",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ])
        
        # Source and output
        output_file = output_dir / f"{cuda_file.stem}.so"
        nvcc_cmd.extend([str(cuda_file), "-o", str(output_file)])
        
        # Add shared library flag
        nvcc_cmd.append("--shared")
        
        try:
            # Set environment to help nvcc find the right compiler
            env = os.environ.copy()
            if self.python_info['in_virtualenv']:
                # Help nvcc find system libraries when in virtualenv
                env['LD_LIBRARY_PATH'] = f"{self.python_info['library_dir']}:{env.get('LD_LIBRARY_PATH', '')}"
            
            result = subprocess.run(
                nvcc_cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env
            )
            
            # Copy to package directory
            package_cuda_dir = Path("python/mcts/cuda")
            if package_cuda_dir.exists():
                shutil.copy2(output_file, package_cuda_dir / output_file.name)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"    Error: {e.stderr}")
            # Try simpler compilation
            return self._simple_cuda_compile(cuda_file, output_dir)
    
    def _simple_cuda_compile(self, cuda_file: Path, output_dir: Path) -> bool:
        """Fallback simple CUDA compilation"""
        print("    Trying simple compilation...")
        
        output_file = output_dir / f"{cuda_file.stem}.so"
        
        # Use the build script approach
        build_script_path = Path("build_cuda.py")
        if build_script_path.exists():
            try:
                subprocess.run(
                    [sys.executable, str(build_script_path), 
                     "--source", str(cuda_file.parent),
                     "--output", str(output_dir)],
                    check=True
                )
                return True
            except subprocess.CalledProcessError:
                pass
        
        # Ultimate fallback - use legacy setup
        legacy_setup = Path("setup_legacy.py")
        if legacy_setup.exists():
            print("    Using legacy CUDA build...")
            try:
                subprocess.run(
                    [sys.executable, str(legacy_setup), "build_cuda"],
                    check=True,
                    capture_output=True
                )
                return True
            except subprocess.CalledProcessError:
                pass
        
        return False
    
    def _detect_cuda_arch(self) -> List[str]:
        """Detect CUDA architectures"""
        archs = []
        
        # Try to detect from PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                archs.append(f"{major}{minor}")
        except ImportError:
            pass
        
        # Add common architectures
        if not archs:
            archs = ["60", "70", "75", "80", "86"]
        
        return archs


class AutoInstallCommand(install):
    """Installation command with automatic configuration"""
    
    def run(self):
        """Run installation"""
        print("\n🚀 Starting AlphaZero (Omoknuni) installation...")
        print("🔍 Auto-detecting environment...")
        
        # Check system requirements
        self._check_requirements()
        
        # Run standard install
        super().run()
        
        print("\n✅ Installation completed successfully!")
        print("\nQuick test:")
        print("  python -c 'import mcts; print(\"MCTS imported successfully\")'")
    
    def _check_requirements(self):
        """Check and report system requirements"""
        requirements = {
            "Python": sys.version_info >= (3, 8),
            "pip": True,  # We're running, so pip exists
            "CMake": shutil.which("cmake") is not None,
            "CUDA": shutil.which("nvcc") is not None,
            "PyTorch": self._check_torch(),
        }
        
        print("\n📋 System Requirements:")
        for req, status in requirements.items():
            icon = "✓" if status else "✗"
            print(f"   {icon} {req}")
        
        if not requirements["Python"]:
            print(f"\n⚠️  Python 3.8+ required (found {sys.version_info.major}.{sys.version_info.minor})")
    
    def _check_torch(self) -> bool:
        """Check if PyTorch is available"""
        try:
            import torch
            return True
        except ImportError:
            return False


class AutoDevelopCommand(develop):
    """Development installation with automatic configuration"""
    
    def run(self):
        """Run development installation"""
        print("\n🔧 Starting development installation...")
        super().run()
        print("\n✅ Development installation completed!")


# Auto-detect version
def get_version() -> str:
    """Get version from pyproject.toml or default"""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return "1.0.0"
    
    try:
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version", "1.0.0")
    except:
        return "1.0.0"


# Main setup configuration
if __name__ == "__main__":
    # Detect if we should use legacy setup for specific commands
    if len(sys.argv) > 1 and sys.argv[1] in ["build_cuda", "legacy"]:
        legacy_setup = Path("setup_legacy.py")
        if legacy_setup.exists():
            print("Using legacy setup for CUDA build...")
            sys.argv[0] = str(legacy_setup)
            exec(open(legacy_setup).read())
            sys.exit(0)
    
    # Modern setup
    setup(
        name="alphazero-omoknuni",
        version=get_version(),
        packages=find_packages(where="python"),
        package_dir={"": "python"},
        ext_modules=[CMakeExtension("alphazero_py", sourcedir=".")] if os.path.exists("CMakeLists.txt") else [],
        cmdclass={
            "build_ext": SmartBuildExt,
            "install": AutoInstallCommand,
            "develop": AutoDevelopCommand,
        },
        python_requires=">=3.8",
        include_package_data=True,
    )