#!/usr/bin/env python3
"""
PyTorch-based CUDA kernel builder for AlphaZero Omoknuni
Uses torch.utils.cpp_extension for proper symbol resolution
"""

import os
import sys
import time
import shutil
from pathlib import Path

class CUDABuilder:
    """Builder for CUDA kernels using PyTorch extension system"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.torch_available = self._check_torch()
        
    def _check_torch(self) -> bool:
        """Check if PyTorch with CUDA is available"""
        try:
            import torch
            if not torch.cuda.is_available():
                self._log("PyTorch is installed but CUDA is not available", "WARNING")
                return False
            return True
        except ImportError:
            self._log("PyTorch not found - CUDA kernels require PyTorch", "ERROR")
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
    
    def build_all(self, source_dir: Path = Path("python/mcts/gpu"), 
                  output_dir: Path = Path("build_torch_cuda")) -> bool:
        """Build CUDA kernels using PyTorch's extension system"""
        
        if not self.torch_available:
            self._log("PyTorch with CUDA not found. Skipping GPU acceleration", "WARNING")
            return True
            
        import torch
        from torch.utils.cpp_extension import load
        
        # GPU info
        device_name = torch.cuda.get_device_name()
        capability = torch.cuda.get_device_capability()
        self._log(f"Target GPU: {device_name} (capability {capability[0]}.{capability[1]})")
        
        # Find CUDA files
        cuda_files = list(source_dir.glob("*.cu"))
        if not cuda_files:
            self._log(f"No CUDA files found in {source_dir}", "WARNING")
            return True
        
        self._log(f"Found {len(cuda_files)} CUDA files to compile")
        
        # Set up compilation
        os.environ['TORCH_CUDA_ARCH_LIST'] = f"{capability[0]}.{capability[1]}"
        
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        try:
            self._log("Compiling CUDA kernels (this may take a minute)...")
            start_time = time.time()
            
            # Build the module
            module = load(
                name='mcts_cuda_kernels',
                sources=[str(cuda_files[0])],
                extra_cuda_cflags=[
                    '-O3',
                    '--use_fast_math',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-diag-suppress=20012',  # Suppress "function was declared but never referenced"
                    '-diag-suppress=177',    # Suppress "variable was declared but never referenced"
                ],
                verbose=False,
                build_directory=str(output_dir)
            )
            
            elapsed = time.time() - start_time
            self._log(f"Compilation successful in {elapsed:.1f}s", "SUCCESS")
            
            # Find the compiled .so file
            import glob
            
            # Look in PyTorch cache and build directory
            cache_dir = Path.home() / ".cache/torch_extensions"
            so_patterns = [
                str(cache_dir / "**" / "mcts_cuda_kernels*.so"),
                str(output_dir / "**" / "*.so"),
            ]
            
            so_files = []
            for pattern in so_patterns:
                so_files.extend(glob.glob(pattern, recursive=True))
                
            if not so_files:
                self._log("Compiled module not found in cache", "WARNING")
                return False
                
            # Get the most recent .so file
            so_files.sort(key=os.path.getmtime, reverse=True)
            source_so = Path(so_files[0])
            
            # Copy to the package directory
            dest_so = source_dir / "mcts_cuda_kernels.so"
            shutil.copy2(source_so, dest_so)
            self._log(f"Installed to {dest_so}", "SUCCESS")
            
            # Verify the module
            self._log("Verifying module...")
            try:
                # Import the module to check it works
                import importlib.util
                spec = importlib.util.spec_from_file_location("mcts_cuda_kernels", str(dest_so))
                test_module = importlib.util.module_from_spec(spec)
                
                # Load with torch first to resolve symbols
                torch.ops.load_library(str(dest_so))
                
                # Then execute the module
                spec.loader.exec_module(test_module)
                
                # Check for expected functions
                expected = ['batched_ucb_selection', 'vectorized_backup', 'find_expansion_nodes', 'batched_add_children']
                found = [f for f in expected if hasattr(test_module, f)]
                
                if found:
                    self._log(f"Verified {len(found)} kernel functions", "SUCCESS")
                else:
                    self._log("No expected functions found in module", "WARNING")
                    
            except Exception as e:
                self._log(f"Module verification warning: {e}", "WARNING")
                # Not critical - module may still work
                
            return True
                
        except Exception as e:
            self._log(f"Compilation failed: {str(e)}", "ERROR")
            if self.verbose:
                import traceback
                traceback.print_exc()
            self._log("Continuing without GPU acceleration", "WARNING")
            return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build CUDA kernels for AlphaZero Omoknuni")
    parser.add_argument("--source", type=Path, default=Path("python/mcts/gpu"), 
                       help="Source directory containing CUDA files")
    parser.add_argument("--output", type=Path, default=Path("build_torch_cuda"),
                       help="Output directory for build files")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress output")
    
    args = parser.parse_args()
    
    builder = CUDABuilder(verbose=not args.quiet)
    success = builder.build_all(args.source, args.output)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()