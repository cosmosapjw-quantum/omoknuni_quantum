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
rm -f python/mcts/gpu/*.so
rm -f python/mcts/gpu/*.pyd

# Remove C++ build artifacts
echo "Removing C++ build artifacts..."
rm -rf bin/
rm -rf lib/
rm -rf obj/

echo "✅ Clean complete!"
echo ""
echo "To rebuild with CUDA support:"
echo "  source ~/venv/bin/activate"
echo "  python setup.py build_ext --inplace"
echo "  python setup.py install"
echo ""
echo "Or for development:"
echo "  source ~/venv/bin/activate" 
echo "  python setup.py develop"