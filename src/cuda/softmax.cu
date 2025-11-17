#include <cuda_runtime.h>
#include <math_constants.h>
#include "../cuda_check.h"

static inline __host__ __device__ int div_up(int a, int b){ 
  return (a+b-1) / b; 
}

__global__ void softmax_forward_kernel(const float* __restrict__ x, float* __restrict__ y, int rows, int cols){
  int row = blockIdx.x; //one blk per row
  if(row >= rows) return;

  extern __shared__ float smem[];
  float* smax = smem; //[blockDim.x]
  float* ssum = smem + blockDim.x; //[blockDim.x]

  const int tid= threadIdx.x;
  const int stride= blockDim.x;
  const size_t base= (size_t)row * (size_t)cols;


  //reduce max for num stability
  float m = -CUDART_INF_F;
  for(int j=tid; j<cols; j += stride){
    float v = x[base+j];
    m = fmaxf(m, v);
  }
  smax[tid] = m;
  __syncthreads();

  for(int s=blockDim.x >> 1; s>0; s >>= 1){
    if(tid<s){ 
      smax[tid] = fmaxf(smax[tid], smax[tid + s]); 
    }
    __syncthreads();
  }
  float row_max=smax[0];


  //do exp(x - max) and reduce sum
  float sum = 0.0f;
  for(int j=tid; j<cols; j += stride){
    float e= expf(x[base + j] - row_max);
    y[base + j] = e; //store numerator temporarily
    sum += e;
  }
  ssum[tid]=sum;
  __syncthreads();

  for(int s=blockDim.x >> 1; s>0; s >>= 1){
    if(tid<s){ 
      ssum[tid] += ssum[tid+s]; 
    }
    __syncthreads();
  }
  float row_sum = ssum[0] + 1e-20f; // avoid 0-div


  //normalize
  for(int j=tid; j<cols; j += stride){
    y[base+j] = y[base+j] / row_sum;
  }
}


extern "C" void nano2_softmax_forward(const float* x, float* y, int rows, int cols){
  if(rows <= 0 || cols <= 0) return;
  int threads = (cols >= 256) ? 256 : (cols >= 128 ? 128 : 64);
  dim3 block(threads, 1, 1);
  dim3 grid(rows, 1, 1);
  size_t shmem_bytes = (size_t)threads * 2 * sizeof(float);
  softmax_forward_kernel<<<grid, block, shmem_bytes>>>(x, y, rows, cols);
  CUDA_CHECK("softmax_forward");
}

