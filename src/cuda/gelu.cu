//clamp input a bit 
//TODO: add debug prints option
//TODO:
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ float gelu_fast(float x, int approx){
    //keep range a bit sane
    if(x > 12.f) x = 12.f;
    if(x < -12.f) x = -12.f;

    if(approx){
        float u = 0.79788456f*(x + 0.044715f*x*x*x);
        return 0.5f * x * (1.f + tanhf(u));
    } else {
        return 0.5f * x * (1.f + erff(x * 0.70710678f));
    }
}

__global__ void gelu_k(const float *x, float *y, int n, int approx){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n) y[i] = gelu_fast(x[i],approx);
}

extern "C" void nano2_gelu_forward(const float *x, float *y, int n, int approx){
    int B=256, G=(n+B-1)/B;
    gelu_k<<<G,B>>>(x,y,n,approx);
}

