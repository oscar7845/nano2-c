cmake_minimum_required(VERSION 3.21)
project(nano2 LANGUAGES C CUDA)

# Default arch; can be overridden at configure time:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=50
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

option(NANO2_ALLOW_UNSUPPORTED_GCC "Pass --allow-unsupported-compiler to NVCC" ON)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ---- Main nano2 binary ----
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
  src/checkpoint.c
  src/chat.h
  src/chat.c
)

# ---- Forward-only test binary ----
add_executable(nano2_fw
  src/config.c
  test-fw/main.c
  src/data.c
  src/model.c
  src/tensor.c
  src/train_forward.cu
  src/cuda/attention.cu
  src/cuda/embed.cu
  src/cuda/layernorm.cu
  src/cuda/softmax.cu
  src/cuda/gemm.cu
  src/cuda/gelu.cu
  src/cuda/loss.cu
)

# ---- Forward + backward test binary ----
add_executable(nano2_fwbw
  src/config.c
  test-fwandbw/main.c
  src/data.c
  src/model.c
  src/tensor.c
  src/train_forward.cu
  src/train_backward.cu
  src/cuda/attention.cu
  src/cuda/attention_backward.cu
  src/cuda/embed.cu
  src/cuda/layernorm.cu
  src/cuda/softmax.cu
  src/cuda/gemm.cu
  src/cuda/gelu.cu
  src/cuda/loss.cu
  src/optim.cu
  src/checkpoint.c
  src/chat.c
)

# ---- Packer tool (no CUDA) ----
add_executable(nano2_pack
  tools/tokenizer.c
)
set_target_properties(nano2_pack PROPERTIES C_STANDARD 11)

# ---- Common settings for CUDA executables ----
set(NANO2_CUDA_TARGETS
  nano2
  nano2_fw
  nano2_fwbw
)

foreach(tgt IN LISTS NANO2_CUDA_TARGETS)
  target_include_directories(${tgt} PRIVATE ${PROJECT_SOURCE_DIR}/src)

  target_link_libraries(${tgt} PRIVATE
    MPI::MPI_C
    CUDA::cudart
  )

  set_target_properties(${tgt} PROPERTIES
    C_STANDARD 11
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )

  if(NANO2_ALLOW_UNSUPPORTED_GCC)
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler -Wno-deprecated-gpu-targets>
    )
  endif()

  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${tgt} PRIVATE
      $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic>
    )
  endif()
endforeach()

