//GELU fw, exact or tanh approx
//y[i]= 0.5 * x[i] * (1 + erf(x / sqrt(2))) // exact
//y[i]= 0.5 * x[i] * (1 + tanh(√(2/π) * (x + 0.044715 x^3))) // tanh approx
//elementwise on flat buff
//TODO: warning on last f 
#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef M_SQRT1_2_F
#define M_SQRT1_2_F 0.70710678118654752440084436210485f
#endif
#ifndef SQRT_2_OVER_PI_F
#define SQRT_2_OVER_PI_F 0.79788456080286535587989211986876f //sqrt(2/pi)
#endif

__device__ __forceinline__ float gelu_exact_f(float x){
  return 0.5f * x * (1.0f + erff(x * M_SQRT1_2_F));
}

__device__ __forceinline__ float gelu_tanh_f(float x){
  float x3= x * x * x;
  float t= SQRT_2_OVER_PI_F * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + tanhf(t));
}

__global__ void gelu_forward_kernel(const float* __restrict__ x,
  float* __restrict__ y,
  int n, int use_tanh){
  int idx= blockIdx.x * blockDim.x + threadIdx.x;
  int stride= blockDim.x * gridDim.x;
  for(int i=idx; i<n; i += stride){
    float v= x[i];
    y[i]=use_tanh ? gelu_tanh_f(v) : gelu_exact_f(v);
  }
}


extern "C" void nano2_gelu_forward(const float* x, float* y, int n, int approximate){
  if(n<=0) return;
  int block= 256;
  int grid=(n+block-1)/block;
  //cap grid to something reasonable; oversubscription is fine for large n
  if(grid > 65535) grid=65535;
  gelu_forward_kernel<<<grid, block>>>(x, y, n, approximate ? 1 : 0);
}

