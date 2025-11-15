//just to see cuda errors
//TODO:
#pragma once
#include <cuda_runtime_api.h>
#include <stdio.h>

static inline void cuda_check_v1(const char *msg){
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess){
        printf("[cuda err] at %s: %s\n", msg, cudaGetErrorString(e));
    }
}

#define CUDA_CHECK(msg) cuda_check_v1(msg)
