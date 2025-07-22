cmake_minimum_required(VERSION 3.18)
project(gguf-shard VERSION 1.0.0 LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(GGUF_SHARD_CUDA "Enable CUDA support" ON)
option(GGUF_SHARD_TESTS "Build tests" OFF)
option(GGUF_SHARD_BENCHMARKS "Build benchmarks" OFF)
option(GGUF_SHARD_STATIC "Static linking" OFF)

# Find packages
if(GGUF_SHARD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3)
endif()

if(GGUF_SHARD_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/patches/cuda)

# CUDA Atlas Library
if(GGUF_SHARD_CUDA)
    add_library(gguf_shard_atlas
        patches/cuda/gguf_shard_atlas.cu
        patches/cuda/gguf_shard_atlas.h
    )
    
    target_link_libraries(gguf_shard_atlas
        CUDA::cudart
        CUDA::cuda_driver
    )
    
    set_target_properties(gguf_shard_atlas PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_ARCHITECTURES "70;75;80;86"
    )
endif()

# Forge tools (Python - just copy)
install(DIRECTORY forge/
    DESTINATION bin/forge
    FILES_MATCHING PATTERN "*.py"
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ)

# Trainer tools (Python - just copy)
install(DIRECTORY trainer/
    DESTINATION bin/trainer  
    FILES_MATCHING PATTERN "*.py"
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_EXECUTE GROUP_READ
                WORLD_EXECUTE WORLD_READ)

# Tests
if(GGUF_SHARD_TESTS)
    enable_testing()
    
    # C++ tests would go here
    add_subdirectory(tests/cpp)
    
    # Python tests via CTest
    add_test(NAME python_tests
        COMMAND python3 -m pytest ${CMAKE_SOURCE_DIR}/tests/ -v
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# Benchmarks
if(GGUF_SHARD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in 
                   ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(docs
        ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()

# Install targets
if(GGUF_SHARD_CUDA)
    install(TARGETS gguf_shard_atlas
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
        
    install(FILES patches/cuda/gguf_shard_atlas.h
        DESTINATION include)
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "gguf-shard")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Sharded GGUF Implementation")
set(CPACK_PACKAGE_VENDOR "GGUF-Shard Project")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

include(CPack)
