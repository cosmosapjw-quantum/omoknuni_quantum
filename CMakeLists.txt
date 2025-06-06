cmake_minimum_required(VERSION 3.20)

# Check if CUDA is requested before project declaration
if(WITH_TORCH)
    project(AlphaZero VERSION 0.1.0 LANGUAGES CXX CUDA)
else()
    project(AlphaZero VERSION 0.1.0 LANGUAGES CXX)
endif()

# ────────────────────────────────── Standard settings ────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Threads
find_package(Threads REQUIRED)

# Set build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

# Compiler options
add_compile_options(-Wall -Wextra)
add_compile_options($<$<CONFIG:Release>:-O3>)

# ───────────────────────────────────── Options ─────────────────────────────────
option(BUILD_PYTHON_BINDINGS  "Build Python bindings"  ON)
option(BUILD_TESTS            "Build tests"            ON)
option(BUILD_SHARED_LIBS      "Build shared libs"      ON)
option(WITH_TORCH             "Build with PyTorch GPU support"     ON)
option(WITH_CUDNN             "Enable cuDNN support (requires WITH_TORCH)" ON)

include(GNUInstallDirs)

# ───────────────────────────────────── CUDA/Torch detection ────────────────────────────
if(WITH_TORCH)
    # Enable CUDA language
    enable_language(CUDA)
    
    # Find CUDA
    find_package(CUDA REQUIRED)
    message(STATUS "Found CUDA: ${CUDA_VERSION_STRING}")
    
    # Find cuDNN if requested
    if(WITH_CUDNN)
        find_path(CUDNN_INCLUDE_DIR cudnn.h
            HINTS ${CUDA_INCLUDE_DIRS}
            PATH_SUFFIXES cuda
        )
        
        find_library(CUDNN_LIBRARY cudnn
            HINTS ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib64 lib/x64 lib
        )
        
        if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
            message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
            set(CUDNN_FOUND TRUE)
        else()
            message(WARNING "cuDNN not found. Some features may be disabled.")
            set(CUDNN_FOUND FALSE)
        endif()
    else()
        set(CUDNN_FOUND FALSE)
    endif()
    
    # Set CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures")
    
    # Find PyTorch from Python environment
    if(Python_FOUND)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import torch; import os; print(os.path.join(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'share', 'cmake', 'Torch'))"
            OUTPUT_VARIABLE TORCH_CMAKE_PATH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE TORCH_CMAKE_RESULT
            ERROR_QUIET
        )
        
        if(TORCH_CMAKE_RESULT EQUAL 0 AND EXISTS "${TORCH_CMAKE_PATH}")
            set(Torch_DIR "${TORCH_CMAKE_PATH}" CACHE PATH "Directory containing TorchConfig.cmake" FORCE)
            message(STATUS "Found PyTorch CMake directory via Python: ${Torch_DIR}")
        endif()
    endif()
    
    # Fallback to manual path if Python detection fails
    if(NOT Torch_DIR OR NOT EXISTS "${Torch_DIR}")
        set(Torch_DIR "/opt/libtorch/share/cmake/Torch" CACHE PATH "Directory containing TorchConfig.cmake")
    endif()
    
    find_package(Torch REQUIRED)
    message(STATUS "Found PyTorch: ${Torch_DIR}")
    
    # Define WITH_TORCH for preprocessor
    add_compile_definitions(WITH_TORCH)
endif()

# ───────────────────────────────────── Tests setup ────────────────────────────
if(BUILD_TESTS)
    include(CTest)
    enable_testing()
    
    # Use find_package for system-installed GTest
    find_package(GTest REQUIRED)
    message(STATUS "Found GTest: ${GTEST_VERSION}")
endif()

# ───────────────────────────────────── Python bindings ────────────────────────
set(PYBIND11_FOUND FALSE)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development QUIET)
    if(Python_FOUND)
        message(STATUS "Found Python interpreter: ${Python_EXECUTABLE} (${Python_VERSION})")
        message(STATUS "Found Python libraries: ${Python_LIBRARIES}")
        message(STATUS "Found Python include dirs: ${Python_INCLUDE_DIRS}")
        
        # First try to get pybind11 paths from Python
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE pybind11_cmake_dir_from_python
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE pybind11_get_cmake_dir_result
            ERROR_QUIET
        )
        
        if(pybind11_get_cmake_dir_result EQUAL 0 AND EXISTS "${pybind11_cmake_dir_from_python}")
            message(STATUS "Found pybind11 CMake directory via Python: ${pybind11_cmake_dir_from_python}")
            list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir_from_python}")
            set(pybind11_DIR "${pybind11_cmake_dir_from_python}" CACHE PATH "pybind11 CMake directory from Python" FORCE)
        endif()
    endif()

    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "Building Python bindings with pybind11 ${pybind11_VERSION}")
        set(PYBIND11_FOUND TRUE)
    else()
        message(WARNING "pybind11 not found - Python bindings will be disabled.")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()

# ───────────────────────────────────── nlohmann_json ────────────────────────────
find_package(nlohmann_json CONFIG QUIET)
if(nlohmann_json_FOUND)
    message(STATUS "Found nlohmann_json: ${nlohmann_json_VERSION}")
else()
    # Try to find nlohmann_json headers directly
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS
            /usr/include
            /usr/local/include
            ${CMAKE_PREFIX_PATH}
        NO_DEFAULT_PATH
    )
    
    if(NLOHMANN_JSON_INCLUDE_DIR)
        message(STATUS "Found nlohmann_json include dir: ${NLOHMANN_JSON_INCLUDE_DIR}")
        include_directories(BEFORE SYSTEM ${NLOHMANN_JSON_INCLUDE_DIR})
        include_directories(BEFORE SYSTEM "/usr/include")
        
        # Create the target manually if find_package failed
        if(NOT TARGET nlohmann_json::nlohmann_json)
            add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
            set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${NLOHMANN_JSON_INCLUDE_DIR}")
        endif()
    else()
        message(FATAL_ERROR "Could not find nlohmann/json.hpp header file. Please install nlohmann_json library.")
    endif()
endif()

# ─────────────────────────────────── Source sets ───────────────────────────────
set(CORE_SOURCES      
    src/core/igamestate.cpp 
    src/core/game_export.cpp
)

set(CHESS_SOURCES     
    src/games/chess/chess_state.cpp 
    src/games/chess/chess_rules.cpp 
    src/games/chess/chess960.cpp
)

set(GO_SOURCES        
    src/games/go/go_state.cpp   
    src/games/go/go_rules.cpp
)

set(GOMOKU_SOURCES    
    src/games/gomoku/gomoku_state.cpp 
    src/games/gomoku/gomoku_rules.cpp
)

set(UTIL_SOURCES      
    src/utils/zobrist_hash.cpp 
    src/utils/hash_specializations.cpp 
    src/utils/logger.cpp
    src/utils/attack_defense_module.cpp
)

# GPU-specific sources (only included when WITH_TORCH is enabled)
set(GPU_UTIL_SOURCES "")
if(WITH_TORCH)
    set(GPU_UTIL_SOURCES
        src/utils/gpu_attack_defense_gomoku.cpp
        src/utils/gpu_attack_defense_chess.cpp
        src/utils/gpu_attack_defense_go.cpp
        src/utils/attack_defense_gpu_manager.cpp
    )
endif()

set(ALL_SOURCES
    ${CORE_SOURCES} 
    ${CHESS_SOURCES} 
    ${GO_SOURCES} 
    ${GOMOKU_SOURCES}
    ${UTIL_SOURCES} 
    ${GPU_UTIL_SOURCES}
)

# ───────────────────────────── alphazero library target ───────────────────────
add_library(alphazero ${ALL_SOURCES})

set_target_properties(alphazero PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(alphazero
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Define ALPHAZERO_EXPORTS when building the library to ensure symbols are exported
if(BUILD_SHARED_LIBS)
    target_compile_definitions(alphazero PRIVATE -DALPHAZERO_EXPORTS)
endif()

# Link required libraries
target_link_libraries(alphazero PUBLIC 
    Threads::Threads
    nlohmann_json::nlohmann_json
    fmt  # For logger
)

# Link PyTorch if enabled
if(WITH_TORCH)
    target_link_libraries(alphazero PUBLIC "${TORCH_LIBRARIES}")
    target_compile_definitions(alphazero PUBLIC -DWITH_TORCH)
    
    # Add CUDA libraries
    if(CUDA_FOUND)
        target_link_libraries(alphazero PUBLIC ${CUDA_CUDART_LIBRARY})
        
        # Add cuDNN if found
        if(CUDNN_FOUND)
            target_include_directories(alphazero PUBLIC ${CUDNN_INCLUDE_DIR})
            target_link_libraries(alphazero PUBLIC ${CUDNN_LIBRARY})
            target_compile_definitions(alphazero PUBLIC -DWITH_CUDNN)
        endif()
    endif()
endif()

# ───────────────────────────────── Python module ─────────────────────────────────
if(PYBIND11_FOUND AND BUILD_PYTHON_BINDINGS)
    message(STATUS "Building Python module...")
    pybind11_add_module(alphazero_py src/python/bindings.cpp)
    target_link_libraries(alphazero_py PRIVATE alphazero)
    
    if(Python_FOUND)
        # Set the Python module installation path
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(PYTHON_SITE_PACKAGES)
            message(STATUS "Python site-packages directory: ${PYTHON_SITE_PACKAGES}")
            install(TARGETS alphazero_py DESTINATION "${PYTHON_SITE_PACKAGES}")
        endif()
    endif()
else()
    message(STATUS "Skipping Python module creation")
endif()

# ───────────────────────────────────── Tests ──────────────────────────────────
if(BUILD_TESTS)
    set(CORE_TEST_SOURCES   
        tests/core/igamestate_test.cpp  
        tests/core/game_export_test.cpp
    )
    set(CHESS_TEST_SOURCES  tests/games/chess/chess_test.cpp)
    set(GO_TEST_SOURCES     tests/games/go/go_test.cpp)
    set(GOMOKU_TEST_SOURCES tests/games/gomoku/gomoku_test.cpp)
    set(ATTACK_DEFENSE_TEST_SOURCES 
        tests/utils/attack_defense_test.cpp
        tests/utils/gomoku_attack_defense_test.cpp
    )
    
    # GPU tests (only when WITH_TORCH is enabled)
    if(WITH_TORCH)
        set(GPU_TEST_SOURCES tests/utils/attack_defense_benchmark.cpp)
        # Also add GPU tests to attack_defense test suite
        list(APPEND ATTACK_DEFENSE_TEST_SOURCES ${GPU_TEST_SOURCES})
    endif()
    
    function(make_test name)
        add_executable(${name} ${ARGN})
        
        # Set RPATH to find shared libraries relative to executable location  
        set_target_properties(${name} PROPERTIES
            INSTALL_RPATH_USE_LINK_PATH TRUE
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN/../lib/Release:$ORIGIN/../lib:$ORIGIN/../../lib/Release:$ORIGIN/../../lib"
        )
        
        # Link libraries
        target_link_libraries(${name} PRIVATE alphazero)
        
        # Add GTest
        if (${name} STREQUAL "all_tests" OR ${name} STREQUAL "core_tests")
            # These tests provide their own main.cpp
            target_link_libraries(${name} PRIVATE GTest::gmock GTest::gtest)
        else()
            # These tests use the default main provided by gtest_main
            target_link_libraries(${name} PRIVATE GTest::gtest_main GTest::gtest GTest::gmock)
        endif()
        
        add_test(NAME ${name} COMMAND ${name})
    endfunction()
    
    make_test(core_tests    tests/core_tests_main.cpp ${CORE_TEST_SOURCES})
    make_test(chess_tests   ${CHESS_TEST_SOURCES})
    make_test(go_tests      ${GO_TEST_SOURCES})
    make_test(gomoku_tests  ${GOMOKU_TEST_SOURCES})
    make_test(attack_defense_tests ${ATTACK_DEFENSE_TEST_SOURCES})
    
    # Add GPU tests if WITH_TORCH is enabled
    if(WITH_TORCH)
        make_test(gpu_attack_defense_test ${GPU_TEST_SOURCES})
    endif()
    
    # Create all_tests executable with conditional GPU tests
    if(WITH_TORCH)
        make_test(all_tests     tests/all_tests_main.cpp
                                ${CORE_TEST_SOURCES} 
                                ${CHESS_TEST_SOURCES}
                                ${GO_TEST_SOURCES} 
                                ${GOMOKU_TEST_SOURCES}
                                ${ATTACK_DEFENSE_TEST_SOURCES}
                                ${GPU_TEST_SOURCES})
    else()
        make_test(all_tests     tests/all_tests_main.cpp
                                ${CORE_TEST_SOURCES} 
                                ${CHESS_TEST_SOURCES}
                                ${GO_TEST_SOURCES} 
                                ${GOMOKU_TEST_SOURCES}
                                ${ATTACK_DEFENSE_TEST_SOURCES})
    endif()
endif()

# ───────────────────────────────── Installation ───────────────────────────────
install(TARGETS alphazero
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION include)