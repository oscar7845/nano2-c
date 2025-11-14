//small approx option
//TODO:
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ float gelu_exact(float x){
    return 0.5f * x * (1.f + erff(x * 0.70710678f));
}

__device__ float gelu_tanh(float x){
    float u = 0.79788456f * (x + 0.044715f * x*x*x);
    return 0.5f * x * (1.f + tanhf(u));
}

__global__ void gelu_k(const float *x, float *y, int n, int approx){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        float v = x[i];
        y[i] = approx ? gelu_tanh(v) : gelu_exact(v);
    }
}

extern "C" void nano2_gelu_forward(const float *x, float *y, int n, int approx){
    int B=256, G=(n+B-1)/B;
    if(G>65535) G=65535;
    gelu_k<<<G,B>>>(x,y,n,approx);
}

