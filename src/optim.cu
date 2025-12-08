//grad-norm clipping 
//AdamW update 
//(fp32, no bias-correction).
#include "atomic_utils.cuh"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <math.h>

#include "cuda_check.h"

__global__ void sumsq_kernel(const float* __restrict__ g, size_t n, double* __restrict__ out){
    double acc = 0.0;
    size_t i=(size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s=(size_t)blockDim.x * gridDim.x;
    for (; i<n; i+= s) { double v = (double)g[i]; acc += v*v; }
    atomicAdd(out, acc);
}

extern "C" double nano2_grad_l2_norm(const float* g, size_t n){
    if (!n) return 0.0;
    int block= 256, grid = (int)((n + block - 1) / block); 
    if (grid > 65535) grid=65535;
    double* d_sum = NULL; cudaMalloc(&d_sum, sizeof(double)); cudaMemset(d_sum, 0, sizeof(double));
    sumsq_kernel<<<grid, block>>>(g, n, d_sum); CUDA_CHECK("sumsq_kernel");
    double h=0.0; cudaMemcpy(&h, d_sum, sizeof(double), cudaMemcpyDeviceToHost); 
    cudaFree(d_sum);
    return sqrt(h);
}

__global__ void scale_kernel(float* __restrict__ g, size_t n, float s){
    size_t i= (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st= (size_t)blockDim.x * gridDim.x;
    for (; i<n; i+=st) g[i] *= s;
}
extern "C" void nano2_clip_grad_global_norm(float* g, size_t n, float max_norm){
    if (max_norm <= 0.0f || !n) return;
    double norm= nano2_grad_l2_norm(g, n);
    if (norm > (double)max_norm){
        float s=(float)(max_norm / ((double)norm + 1e-12));
        int block=256, grid = (int)((n + block - 1) / block); 
	if (grid > 65535) grid=65535;
        scale_kernel<<<grid, block>>>(g, n, s); CUDA_CHECK("scale_kernel");
    }
}

//AdamW (weight decay to gradients)
//no bias correction to make simpler
//can add later
__global__ void adamw_kernel(float* __restrict__ p,
                             float* __restrict__ g,
                             float* __restrict__ m,
                             float* __restrict__ v,
                             size_t n,
                             float lr, float beta1, float beta2,
                             float eps, float weight_decay){
    size_t i= (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t st= (size_t)blockDim.x * gridDim.x;
    for (; i<n; i += st){
        float gi= g[i] + weight_decay * p[i];
        float mi= m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        float vi= v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        float upd = mi / (sqrtf(vi) + eps);
        p[i] -= lr * upd;
        g[i] = 0.0f; //clear grad for next step
    }
}
extern "C" void nano2_adamw_step(float* params, float* grads, float* m, float* v, size_t n,
                                 float lr, float beta1, float beta2, float eps, float weight_decay){
    if (!n) return;
    int block=256, grid = (int)((n + block - 1) / block); 
    if (grid>65535) grid=65535;
    adamw_kernel<<<grid, block>>>(params, grads, m, v, n, lr, beta1, beta2, eps, weight_decay);
    CUDA_CHECK("adamw_kernel");
}

