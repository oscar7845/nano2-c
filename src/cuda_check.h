//add sync sometimes
//TODO:
#pragma once
#include <cuda_runtime_api.h>
#include <stdio.h>

static inline void nano2_cuda_check2(const char *where){
    // force runtime error to show up
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if(e != cudaSuccess){
        fprintf(stderr,"[cuda] error %s : %s\n", where, cudaGetErrorString(e));
    }
}

#define CUDA_CHECK(tag) nano2_cuda_check2(tag)
