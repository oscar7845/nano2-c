//GELU forward (fp32)
//just elementwise. not optimized.
//TODO:
//

#include <cuda_runtime.h>
#include <math_constants.h>

__device__ float gelu_f(float x){
    //exact erf version for now
    return 0.5f * x * (1.f + erff(x * 0.70710678f));
}

__global__ void gelu_k(const float *x, float *y, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) y[i] = gelu_f(x[i]);
}

extern "C" void nano2_gelu_forward(const float *x, float *y, int n){
    if(n <= 0) return;
    int B=256, G=(n+B-1)/B;
    gelu_k<<<G,B>>>(x,y,n);
}

