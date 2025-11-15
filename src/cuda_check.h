#pragma once
#include <cuda_runtime_api.h>
#include <stdio.h>

static inline void nano2_cuda_check_(const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[cuda] ERROR at %s: %s\n", where, cudaGetErrorString(e));
    }
}

#define CUDA_CHECK(tag) nano2_cuda_check_(tag)
