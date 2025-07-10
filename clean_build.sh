#!/bin/bash
# Clean build script for AlphaZero (Omoknuni) project

echo "🧹 Cleaning all build artifacts..."

# Remove Python build directories
echo "Removing Python build directories..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info
rm -rf python/*.egg-info
rm -rf python/build/
rm -rf python/dist/

# Remove compiled Python files
echo "Removing compiled Python files..."
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove compiled extensions
echo "Removing compiled extensions..."
find . -type f -name "*.so" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name "*.dll" -delete

# Remove CMake build directory
echo "Removing CMake build directory..."
rm -rf cmake-build-*/
rm -rf CMakeFiles/
rm -f CMakeCache.txt
rm -f cmake_install.cmake
rm -f Makefile

# Remove CUDA kernel builds
echo "Removing CUDA kernel builds..."
rm -rf python/mcts/gpu/build/
rm -rf python/build_cuda/
rm -rf build_cuda_shared/
rm -f python/mcts/gpu/*.so
rm -f python/mcts/gpu/*.pyd

# Remove C++ build artifacts
echo "Removing C++ build artifacts..."
rm -rf bin/
rm -rf lib/
rm -rf obj/

echo "✅ Clean complete!"
echo ""
echo "To rebuild everything:"
echo "  source ~/venv/bin/activate"
echo "  pip install -e ."
echo ""
echo "This will automatically:"
echo "  - Build C++ game backends (libalphazero.so and alphazero_py module)"
echo "  - Compile CUDA kernels if CUDA is available (using fast nvcc)"
echo ""
echo "For manual builds:"
echo "  - CUDA kernels: python build_cuda.py"
echo "  - C++ only: python build_cpp.py"
echo "  - Or: mkdir build && cd build && cmake .. && make -j$(nproc)"