//grad norm clipping 
//AdamW 
//TODO: add bias correction
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <math.h>

#include "cuda_check.h"

//sum of squares for L2 norm
__global__ void sumsq_kernel(const float* __restrict__ g,
                             size_t n,
                             double* __restrict__ out){
    double acc=0.0;
    size_t i=(size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step=(size_t)blockDim.x * gridDim.x;
    for(; i<n; i+=step){
        double v= (double)g[i];
        acc += v*v;
    }
    atomicAdd(out, acc);
}

extern "C" double nano2_grad_l2_norm(const float* g, size_t n){
    if(!g || n == 0) return 0.0;

    int block=256;
    int grid=(int)((n + block - 1) / block);
    if(grid>65535) grid = 65535;

    double *d_sum = NULL;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));

    sumsq_kernel<<<grid, block>>>(g, n, d_sum);
    CUDA_CHECK("sumsq_kernel");

    double h=0.0;
    cudaMemcpy(&h, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);

    return sqrt(h);
}

//scale grads by constant
__global__ void scale_kernel(float* __restrict__ g,
                             size_t n,
                             float s){
    size_t i=(size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t str=(size_t)blockDim.x * gridDim.x;
    for(; i<n; i += str){
        g[i] *= s;
    }
}

//clip global grad norm
extern "C" void nano2_clip_grad_global_norm(float* g,
                                            size_t n,
                                            float max_norm){
    if(!g || n == 0) return;
    if(max_norm <= 0.0f) return;

    double norm=nano2_grad_l2_norm(g, n);
    if(norm>(double)max_norm){
        float s=(float)(max_norm / ((double)norm + 1e-12));
        int block= 256;
        int grid= (int)((n + block - 1) / block);
        if(grid>65535) grid = 65535;
        scale_kernel<<<grid, block>>>(g, n, s);
        CUDA_CHECK("scale_kernel");
    }
}

//AdamW kernel 
__global__ void adamw_kernel(float* __restrict__ p,
                             float* __restrict__ g,
                             float* __restrict__ m,
                             float* __restrict__ v,
                             size_t n,
                             float lr,
                             float beta1,
                             float beta2,
                             float eps,
                             float weight_decay){
    size_t i   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t str = (size_t)blockDim.x * gridDim.x;
    for(; i < n; i += str){
        float gi = g[i] + weight_decay * p[i];
        //update moments
        float mi = m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
        float vi = v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        //no bias correction here (simple v2)
        float upd = mi / (sqrtf(vi) + eps);
        p[i] -= lr * upd;
        //optional: clear grads
        g[i] = 0.0f;
    }
}

//launcher for AdamW
extern "C" void nano2_adamw_step(float* params,
                                 float* grads,
                                 float* m,
                                 float* v,
                                 size_t n,
                                 float lr,
                                 float beta1,
                                 float beta2,
                                 float eps,
                                 float weight_decay){
    if(!params || !grads || !m || !v || n == 0) return;

    int block = 256;
    int grid  = (int)((n + block - 1) / block);
    if(grid > 65535) grid = 65535;

    adamw_kernel<<<grid, block>>>(params, grads, m, v, n,
                                  lr, beta1, beta2, eps, weight_decay);
    CUDA_CHECK("adamw_kernel");
}

