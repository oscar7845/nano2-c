cmake_minimum_required(VERSION 3.21)
project(nano2 LANGUAGES C CUDA)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

option(NANO2_ALLOW_UNSUPPORTED_GCC "Pass --allow-unsupported-compiler to NVCC" ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(nano2
  src/main.c
  src/config.c
  src/data.c
  src/cuda/dummy_kernels.cu
  src/tensor.c
  src/model.c
  src/cuda/gemm.cu
  src/cuda/layernorm.cu
  src/cuda/gelu.cu
  src/cuda/softmax.cu
  src/cuda/attention.cu
  src/cuda/loss.cu
  src/train_forward.cu
  src/nano2_model.h
  src/optim.cu
  src/cuda/attention_backward.cu
  src/train_backward.cu
  src/cuda/embed.cu
  src/main.c
  src/checkpoint.c
)

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

if(NANO2_ALLOW_UNSUPPORTED_GCC)
  target_compile_options(nano2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler -Wno-deprecated-gpu-targets>
  )
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(nano2 PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>)
endif()

# Optional shared headers path no longer needed; remove if you like
# if(EXISTS "/var/tmp/Common/include")
#   target_include_directories(nano2 PRIVATE /var/tmp/Common/include)
# endif()

# ---- Packer tool (no headers) ----
add_executable(nano2_pack
  tools/tokenizer.c
)
set_target_properties(nano2_pack PROPERTIES C_STANDARD 11)

