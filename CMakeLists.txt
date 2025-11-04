cmake_minimum_required(VERSION 3.21)
project(nano2 LANGUAGES C CUDA)

# Let caller pass -DCMAKE_CUDA_ARCHITECTURES=..., else default to H100.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

# Option: allow GCC 15 with CUDA 12.9
option(NANO2_ALLOW_UNSUPPORTED_GCC "Pass --allow-unsupported-compiler to NVCC" ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(nano2
  src/main.c
  src/utils.c
  src/cuda/dummy_kernels.cu
)

target_include_directories(nano2 PRIVATE include)

target_link_libraries(nano2 PRIVATE
  MPI::MPI_C
  CUDA::cudart
)

set_target_properties(nano2 PROPERTIES
  C_STANDARD 11
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Flags applied only to CUDA compilation (NVCC)
if(NANO2_ALLOW_UNSUPPORTED_GCC)
  target_compile_options(nano2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler -Wno-deprecated-gpu-targets>
  )
endif()

# Host compiler warnings for C sources
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(nano2 PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>)
endif()

# Optional: add your shared helper headers if present
if(EXISTS "/var/tmp/Common/include")
  target_include_directories(nano2 PRIVATE /var/tmp/Common/include)
endif()

