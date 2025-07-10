#!/usr/bin/env python3
"""
Standalone C++ module builder for AlphaZero Omoknuni
This handles the C++ compilation that's difficult to integrate with standard setuptools
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from typing import List, Optional, Tuple


class CPPBuilder:
    """Builder for C++ game modules"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.cmake_available = self._check_cmake()
        self.compiler_available = self._check_compiler()
        
    def _check_cmake(self) -> bool:
        """Check if CMake is available"""
        try:
            result = subprocess.run(
                ["cmake", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _check_compiler(self) -> bool:
        """Check if C++ compiler is available"""
        try:
            result = subprocess.run(
                ["g++", "--version"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                result = subprocess.run(
                    ["clang++", "--version"], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
    
    def _log(self, message: str, level: str = "INFO"):
        """Log a message"""
        if self.verbose:
            prefix = {
                "INFO": "ℹ️ ",
                "SUCCESS": "✅",
                "WARNING": "⚠️ ",
                "ERROR": "❌",
            }.get(level, "")
            print(f"{prefix} {message}")
    
    def build_all(self, source_dir: Path = Path("."), 
                  build_dir: Path = Path("build_cpp"),
                  install_dir: Optional[Path] = None) -> bool:
        """Build all C++ modules"""
        if not self.cmake_available:
            self._log("CMake not found, skipping C++ module compilation", "WARNING")
            return True
        
        if not self.compiler_available:
            self._log("C++ compiler not found, skipping module compilation", "WARNING")
            return True
        
        # Create build directory
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare CMake arguments
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DBUILD_PYTHON_BINDINGS=ON",
            f"-DBUILD_TESTS=OFF",
            f"-DBUILD_SHARED_LIBS=ON",
        ]
        
        # Add Python executable
        cmake_args.append(f"-DPYTHON_EXECUTABLE={sys.executable}")
        
        # Add install directory if specified
        if install_dir:
            cmake_args.append(f"-DCMAKE_INSTALL_PREFIX={install_dir.absolute()}")
            cmake_args.append(f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={install_dir.absolute()}")
        
        # Configure
        self._log("Configuring C++ build with CMake...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ["cmake", str(source_dir.absolute())] + cmake_args,
                cwd=build_dir,
                capture_output=True,
                text=True,
                check=True
            )
            elapsed = time.time() - start_time
            self._log(f"Configuration completed in {elapsed:.2f}s", "SUCCESS")
            
        except subprocess.CalledProcessError as e:
            self._log(f"CMake configuration failed: {e.stderr}", "ERROR")
            return False
        
        # Build
        self._log("Building C++ modules...")
        start_time = time.time()
        
        try:
            # Get number of cores for parallel build
            try:
                num_cores = os.cpu_count() or 1
            except:
                num_cores = 1
            
            result = subprocess.run(
                ["cmake", "--build", ".", "--config", "Release", "--", f"-j{num_cores}"],
                cwd=build_dir,
                capture_output=True,
                text=True,
                check=True
            )
            elapsed = time.time() - start_time
            self._log(f"Build completed in {elapsed:.2f}s", "SUCCESS")
            
        except subprocess.CalledProcessError as e:
            self._log(f"Build failed: {e.stderr}", "ERROR")
            return False
        
        # Find and copy built libraries
        self._log("Looking for built libraries...")
        lib_patterns = ["*.so", "*.dylib", "*.dll", "*.pyd"]
        found_libs = []
        
        for pattern in lib_patterns:
            found_libs.extend(build_dir.rglob(pattern))
        
        if not found_libs:
            self._log("No libraries found after build", "WARNING")
            return False
        
        # Copy libraries to Python package directory
        package_dir = Path("python")
        if package_dir.exists():
            for lib_file in found_libs:
                # Skip test libraries
                if "test" in lib_file.name.lower():
                    continue
                    
                dest = package_dir / lib_file.name
                shutil.copy2(lib_file, dest)
                self._log(f"Copied {lib_file.name} to {dest}", "SUCCESS")
                
                # Also copy libalphazero.so if this is alphazero_py
                if "alphazero_py" in lib_file.name:
                    # Look for libalphazero.so in the same directory
                    libalphazero = lib_file.parent / "libalphazero.so"
                    if libalphazero.exists():
                        dest_lib = package_dir / "libalphazero.so"
                        shutil.copy2(libalphazero, dest_lib)
                        self._log(f"Copied libalphazero.so to {dest_lib}", "SUCCESS")
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build C++ modules for AlphaZero Omoknuni")
    parser.add_argument("--source", type=Path, default=Path("."), 
                       help="Source directory containing CMakeLists.txt")
    parser.add_argument("--build", type=Path, default=Path("build_cpp"),
                       help="Build directory for compiled files")
    parser.add_argument("--install", type=Path, default=None,
                       help="Installation directory for libraries")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output")
    
    args = parser.parse_args()
    
    builder = CPPBuilder(verbose=not args.quiet)
    success = builder.build_all(args.source, args.build, args.install)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()