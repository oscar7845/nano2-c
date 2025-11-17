//LN fw: y = (x - mean) / sqrt(var + eps) * gamma + beta
//- ins: x[N, D], gamma[D], beta[D]
//- outs: y[N, D]; can save mean[N] and inv_std[N] for backward
//one CUDA blk per row. threads strided over D with reduction
//TODO: 
#include <cuda_runtime.h>
#include <stdint.h>
#include "../cuda_check.h"

//sum & sumsq reduction across a row, then normalize + affine
__global__ void layernorm_forward_kernel(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ gamma,
  				  	 const float* __restrict__ beta, int N, int D, float eps, 
  					 float* __restrict__ mean_out, float* __restrict__ invstd_out){ //could be nullptr
  int row = blockIdx.x; //one blk per row
  if(row >= N) return;

  //shared mem layout: [0..blockDim.x) partial sums, [blockDim.x..2*blockDim.x) partial sumsq
  extern __shared__ float smem[];
  float* sh_sum= smem;
  float* sh_sumsq= smem + blockDim.x;
  const int tid= threadIdx.x;
  const int stride= blockDim.x;

  //accumulate partials for this row
  size_t base= (size_t)row * (size_t)D;
  float sum= 0.0f;
  float sumsq= 0.0f;
  for(int j=tid; j<D; j += stride){
    float v = x[base+j];
    sum += v;
    sumsq += v*v;
  }
  sh_sum[tid]= sum;
  sh_sumsq[tid]= sumsq;
  __syncthreads();

  //parallel reduction
  for(int s=blockDim.x >> 1; s > 0; s >>= 1){
    if(tid < s){
      sh_sum[tid] += sh_sum[tid + s];
      sh_sumsq[tid] += sh_sumsq[tid + s];
    }
  __syncthreads();
  }

  //mean and inv_std; 
  //bcast via shared memory slot 0/1
  float mean= sh_sum[0] / (float)D;
  float var= sh_sumsq[0] / (float)D - mean * mean;
  float inv_std= rsqrtf(var + eps);

  if(tid==0){
    sh_sum[0]= mean; //reuse to bcast
    sh_sumsq[0]= inv_std; //reuse to bcast
    if(mean_out) mean_out[row]= mean;
    if(invstd_out) invstd_out[row]= inv_std;
  }
  __syncthreads();


  mean= sh_sum[0];
  inv_std= sh_sumsq[0];

  //normalize and affine transform
  for(int j=tid; j<D; j += stride){
    float v= (x[base + j] - mean) * inv_std;
    float out= v * gamma[j] + beta[j];
    y[base+j]=out;
  }
}




extern "C" void nano2_layernorm_forward(const float* x, float* y, const float* gamma, const float* beta,
 				        int N, int D, float eps, float* mean_out, float* invstd_out){
  //1 blk per row; 
  //256 threads; 
  //shared = 2*block*sizeof(float)
  int threads = (D >= 256) ? 256 : (D >= 128 ? 128 : 64);
  dim3 block(threads, 1, 1);
  dim3 grid(N, 1, 1);
  size_t shmem_bytes = (size_t)block.x * 2 * sizeof(float);
  layernorm_forward_kernel<<<grid, block, shmem_bytes>>>(x, y, gamma, beta, N, D, eps, mean_out, invstd_out);
  //no cudaDeviceSynchronize()
  CUDA_CHECK("layernorm_forward");
}

