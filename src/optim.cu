//grad norm clipping
//AdamW update
//TODO: clip by global norm 
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <math.h>

#include "cuda_check.h"

__global__ void sumsq_kernel(const float* __restrict__ g, size_t n, double* __restrict__ out){
    double acc = 0.0;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)blockDim.x * gridDim.x;
    for (; i < n; i += s){ 
      double v = (double)g[i]; acc += v*v; 
    }
    atomicAdd(out, acc);
}

extern "C" double nano2_grad_l2_norm(const float* g, size_t n){
    if (!n) return 0.0;
    int block = 256, grid = (int)((n + block - 1) / block); if (grid > 65535) grid = 65535;
    double* d_sum = NULL; cudaMalloc(&d_sum, sizeof(double)); cudaMemset(d_sum, 0, sizeof(double));
    sumsq_kernel<<<grid, block>>>(g, n, d_sum); CUDA_CHECK("sumsq_kernel");
    double h=0.0; 
    cudaMemcpy(&h, d_sum, sizeof(double), cudaMemcpyDeviceToHost); cudaFree(d_sum);
    return sqrt(h);
}

__global__ void scale_kernel(float* __restrict__ g, size_t n, float s){
    size_t i=(size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st=(size_t)blockDim.x * gridDim.x;
    for (; i<n; i+=st) g[i] *= s;
}
extern "C" void nano2_clip_grad_global_norm(float* g, size_t n, float max_norm){
    if (max_norm <= 0.0f || !n) return;
    double norm = nano2_grad_l2_norm(g, n);
    if (norm>(double)max_norm){
        float s=(float)(max_norm / ((double)norm + 1e-12));
        int block= 256, grid = (int)((n + block - 1) / block); if (grid > 65535) grid = 65535;
        scal_kernel<<<grid, block>>>(g, n, s); CUDA_CHECK("scale_kernel");
    }
}



