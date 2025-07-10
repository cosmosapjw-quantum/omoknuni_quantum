#!/usr/bin/env python3
"""
Standalone CUDA kernel builder for AlphaZero Omoknuni
Uses direct nvcc compilation for fast builds
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


class CUDABuilder:
    """Builder for custom CUDA kernels using direct nvcc"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.cuda_available = self._check_cuda()
        self.torch_available = self._check_torch()
        self.gpu_arch = self._detect_gpu_arch()
        
    def _check_cuda(self) -> Tuple[bool, Optional[str]]:
        """Check if CUDA and nvcc are available"""
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if not cuda_home:
            for path in ['/usr/local/cuda', '/opt/cuda']:
                if Path(path).exists():
                    cuda_home = path
                    os.environ['CUDA_HOME'] = cuda_home
                    break
        
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                return True, cuda_home
        except FileNotFoundError:
            pass
        
        return False, None
        
    def _check_torch(self) -> bool:
        """Check if PyTorch with CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _detect_gpu_arch(self) -> Optional[str]:
        """Detect GPU architecture for faster compilation"""
        if self.torch_available:
            try:
                import torch
                if torch.cuda.is_available():
                    major, minor = torch.cuda.get_device_capability()
                    arch = f"{major}{minor}"
                    self._log(f"Detected GPU architecture: sm_{arch}", "INFO")
                    return arch
            except:
                pass
        return None
    
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
    
    def _get_include_dirs(self) -> List[str]:
        """Get include directories for compilation"""
        include_dirs = []
        
        # Python includes
        import sysconfig
        python_include = sysconfig.get_path("include")
        if python_include:
            include_dirs.append(python_include)
        
        # PyTorch includes
        if self.torch_available:
            try:
                import torch
                from torch.utils import cpp_extension
                include_dirs.extend(cpp_extension.include_paths())
            except:
                pass
        
        # CUDA includes
        cuda_ok, cuda_home = self.cuda_available
        if cuda_home:
            cuda_include = Path(cuda_home) / "include"
            if cuda_include.exists():
                include_dirs.append(str(cuda_include))
        
        # Project includes
        project_includes = [
            Path.cwd() / "include",
            Path.cwd() / "cpp/include",
        ]
        include_dirs.extend([str(p) for p in project_includes if p.exists()])
        
        return include_dirs
    
    def build_all(self, source_dir: Path = Path("python/mcts/gpu"), 
                  output_dir: Path = Path("build_cuda_shared")) -> bool:
        """Build all CUDA kernels using direct nvcc compilation"""
        
        cuda_ok, cuda_home = self.cuda_available
        if not cuda_ok:
            self._log("CUDA/nvcc not found. Skipping GPU acceleration", "WARNING")
            self._log("The software will work without GPU acceleration", "INFO")
            return True
        
        if not self.torch_available:
            self._log("PyTorch with CUDA not found. Skipping GPU acceleration", "WARNING")
            return True
        
        # Find CUDA files
        cuda_files = list(source_dir.glob("*.cu"))
        if not cuda_files:
            self._log(f"No CUDA files found in {source_dir}", "WARNING")
            return True
        
        self._log(f"Found {len(cuda_files)} CUDA files to compile")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build each CUDA file
        for cuda_file in cuda_files:
            self._log(f"Compiling {cuda_file.name}...")
            
            # First compile to object file
            obj_file = output_dir / f"{cuda_file.stem}.o"
            
            # Build nvcc command for compilation
            cmd = ['nvcc', '-c']
            
            # Optimization flags
            cmd.extend(['-O3', '--use_fast_math'])
            
            # C++ standard
            cmd.extend(['-std=c++17'])
            
            # Compiler options
            cmd.extend(['--compiler-options', '-fPIC'])
            
            # Architecture - only compile for detected GPU
            if self.gpu_arch:
                cmd.extend(['-gencode', f'arch=compute_{self.gpu_arch},code=sm_{self.gpu_arch}'])
            else:
                # Fallback to common architecture
                cmd.extend(['-gencode', 'arch=compute_75,code=sm_75'])
            
            # Extended features
            cmd.extend(['--expt-extended-lambda', '--expt-relaxed-constexpr'])
            
            # Include directories
            for inc_dir in self._get_include_dirs():
                cmd.extend(['-I', inc_dir])
            
            # Defines
            cmd.extend([
                '-DTORCH_EXTENSION_NAME=mcts_cuda_kernels',
                '-DTORCH_API_INCLUDE_EXTENSION_H',
                '-D_GLIBCXX_USE_CXX11_ABI=1',
            ])
            
            # Disable specific warnings to reduce output
            cmd.extend([
                '-Xcompiler', '-Wno-unused-variable',
                '-diag-suppress', '177',  # Suppress "variable declared but never referenced"
            ])
            
            # Input and output
            cmd.extend([str(cuda_file), '-o', str(obj_file)])
            
            # Compile
            start_time = time.time()
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                elapsed = time.time() - start_time
                self._log(f"Compiled to object file in {elapsed:.1f}s", "SUCCESS")
                
                # Now link to shared library
                so_file = output_dir / f"{cuda_file.stem}.so"
                
                link_cmd = ['nvcc', '--shared']
                link_cmd.extend([str(obj_file), '-o', str(so_file)])
                
                # Add library paths
                if self.torch_available:
                    try:
                        import torch
                        torch_lib = Path(torch.__path__[0]) / "lib"
                        if torch_lib.exists():
                            link_cmd.extend(['-L', str(torch_lib)])
                    except:
                        pass
                
                # Link
                result = subprocess.run(
                    link_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                self._log(f"Linked shared library successfully", "SUCCESS")
                
                # Copy to package directory
                dest_dir = Path("python/mcts/gpu")
                dest_file = dest_dir / "mcts_cuda_kernels.so"
                
                if dest_dir.exists() and so_file.exists():
                    shutil.copy2(so_file, dest_file)
                    self._log(f"Installed to {dest_file}", "SUCCESS")
                    return True
                    
            except subprocess.CalledProcessError as e:
                self._log(f"Compilation failed: {e.stderr}", "ERROR")
                self._log("Continuing without GPU acceleration", "WARNING")
                return True  # Don't fail installation
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build CUDA kernels for AlphaZero Omoknuni")
    parser.add_argument("--source", type=Path, default=Path("python/mcts/gpu"), 
                       help="Source directory containing CUDA files")
    parser.add_argument("--output", type=Path, default=Path("build_cuda_shared"),
                       help="Output directory for compiled files")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output")
    
    args = parser.parse_args()
    
    builder = CUDABuilder(verbose=not args.quiet)
    success = builder.build_all(args.source, args.output)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()