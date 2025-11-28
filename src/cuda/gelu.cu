//GELU forward
//exact or tanh approx
//y[i] = 0.5 * x[i] * (1 + erf(x / sqrt(2))) // exact
//y[i] = 0.5 * x[i] * (1 + tanh(√(2/π) * (x + 0.044715 x^3))) // tanh approx
//elementwise on a flat buffer

#include <cuda_runtime.h>
#include <math_constants.h>
#include "../cuda_check.h"

#ifndef M_SQRT1_2_F
#define M_SQRT1_2_F 0.70710678118654752440084436210485f
#endif
#ifndef SQRT_2_OVER_PI_F
#define SQRT_2_OVER_PI_F 0.79788456080286535587989211986876f // sqrt(2/pi)
#endif

__device__ __forceinline__ float gelu_exact_f(float x){
  return 0.5f * x * (1.0f + erff(x * M_SQRT1_2_F));
}

__device__ __forceinline__ float gelu_tanh_f(float x){
  float x3 = x * x * x;
  float t = SQRT_2_OVER_PI_F * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + tanhf(t));
}

__global__ void gelu_forward_kernel(const float* __restrict__ x,
  float* __restrict__ y,
  int n, int use_tanh){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = idx; i < n; i += stride){
    float v = x[i];
    y[i] = use_tanh ? gelu_tanh_f(v) : gelu_exact_f(v);
  }
}

extern "C" void nano2_gelu_forward(const float* x, float* y, int n, int approximate){
  if(n <= 0) return;
  int block = 256;
  int grid = (n + block - 1) / block;
  //cap grid to something reasonable 
  //oversubscription is fine for large n
  if(grid > 65535) grid = 65535;
  gelu_forward_kernel<<<grid, block>>>(x, y, n, approximate ? 1 : 0);
}


//Backward
//dx = dy * gelu'(x)
__device__ __forceinline__ float gelu_tanh_grad(float x){
    //y = 0.5*x*(1 + tanh(t)), t = a*(x + b*x^3)
    const float a = SQRT_2_OVER_PI_F; // sqrt(2/pi)
    const float b = 0.044715f;
    const float x2 = x*x;
    const float t  = a * (x + b*x2*x);
    const float th = tanhf(t);
    const float dt_dx = a * (1.0f + 3.0f*b*x2) * (1.0f - th*th); // sech^2(t) = 1 - tanh^2
    return 0.5f * (1.0f + th) + 0.5f * x * dt_dx;
}
__device__ __forceinline__ float gelu_exact_grad(float x){
    //d/dx 0.5 x (1 + erf(x/sqrt(2)))
    const float k = M_SQRT1_2_F;
    const float erf_part = erff(k * x);
    const float exp_part = expf(-0.5f * x * x);
    return 0.5f * (1.0f + erf_part) + 0.5f * x * (M_SQRT1_2_F * 2.0f / sqrtf(CUDART_PI_F)) * exp_part;
    //simplified version widely used:
    //return 0.5f * (1.0f + erff(k*x)) + (x * 0.5f) * (M_SQRT1_2_F * 2.0f / sqrtf(CUDART_PI_F)) * expf(-0.5f*x*x);
}

__global__ void gelu_backward_kernel(const float* __restrict__ x,  // pre-activation
                                     const float* __restrict__ dy,
                                     float* __restrict__ dx,
                                     int n, int use_tanh){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockDim.x * gridDim.x;
    for (; i < n; i += s){
        float g = use_tanh ? gelu_tanh_grad(x[i]) : gelu_exact_grad(x[i]);
        dx[i] = dy[i] * g;
    }
}

extern "C" void nano2_gelu_backward(const float* x, const float* dy, float* dx, int n, int approximate){
    if(n <= 0) return;
    int block = 256, grid = (n + block - 1)/block; if(grid>65535) grid=65535;
    gelu_backward_kernel<<<grid, block>>>(x, dy, dx, n, approximate?1:0);
    CUDA_CHECK("gelu_backward");
}


