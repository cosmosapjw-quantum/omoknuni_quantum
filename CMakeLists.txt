cmake_minimum_required(VERSION 3.20)

project(AlphaZero VERSION 0.1.0 LANGUAGES CXX)

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

# Advanced optimization flags for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # CPU-specific optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto -funroll-loops -ffast-math -fomit-frame-pointer -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto -funroll-loops -ffast-math -fomit-frame-pointer -DNDEBUG")
    
    # Enable interprocedural optimization (Link Time Optimization)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    
    # Additional vectorization hints
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(
            -ftree-vectorize
            -mfpmath=sse
            -mavx2
            -mfma
        )
    endif()
    
    # CUDA-specific optimizations
    if(CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --ptxas-options=-v")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --expt-relaxed-constexpr")
        
        # Set CUDA architectures for common GPUs
        set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
    endif()
endif()

# ───────────────────────────────────── Options ─────────────────────────────────
option(BUILD_PYTHON_BINDINGS  "Build Python bindings"  ON)
option(BUILD_TESTS            "Build tests"            OFF)
option(BUILD_SHARED_LIBS      "Build shared libs"      ON)
option(WITH_TORCH             "Build with PyTorch GPU support"     OFF)
option(WITH_CUDNN             "Enable cuDNN support (requires WITH_TORCH)" OFF)

include(GNUInstallDirs)

# PyTorch support disabled

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

# GPU-specific sources removed
set(GPU_UTIL_SOURCES "")

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

# PyTorch linking removed

# ───────────────────────────────── Python module ─────────────────────────────────
if(PYBIND11_FOUND AND BUILD_PYTHON_BINDINGS)
    message(STATUS "Building Python module...")
    pybind11_add_module(alphazero_py src/python/bindings.cpp)
    target_link_libraries(alphazero_py PRIVATE alphazero)
    
    # Also install the alphazero shared library alongside the Python module
    install(TARGETS alphazero
            LIBRARY DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            ARCHIVE DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    
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
    
    # GPU tests removed
    
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
    
    # GPU tests removed
    
    # Create all_tests executable
    make_test(all_tests     tests/all_tests_main.cpp
                            ${CORE_TEST_SOURCES} 
                            ${CHESS_TEST_SOURCES}
                            ${GO_TEST_SOURCES} 
                            ${GOMOKU_TEST_SOURCES}
                            ${ATTACK_DEFENSE_TEST_SOURCES})
endif()

# ───────────────────────────────── Installation ───────────────────────────────
install(TARGETS alphazero
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/ DESTINATION include)