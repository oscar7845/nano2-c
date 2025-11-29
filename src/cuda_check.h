#pragma once
#include <cuda_runtime.h>

//declare printf with C linkage so the host linker finds libc's symbol.
#ifdef __cplusplus
extern "C" {
#endif
int printf(const char*, ...);
#ifdef __cplusplus
}
#endif

static inline void nano2_cuda_check_(const char* where) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("[cuda] ERROR at %s: %s\n", where, cudaGetErrorString(e));
    }
}

#define CUDA_CHECK(tag) nano2_cuda_check_(tag)

