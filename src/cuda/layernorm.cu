//LN fw: y = (x - mean) / sqrt(var + eps) * gamma + beta
//- ins: x[N, D], gamma[D], beta[D]
//- outs: y[N, D]; can save mean[N] and inv_std[N] for backward
//one CUDA blk per row. threads strided over D with reduction
//TODO: 
#include <cuda_runtime.h>
#include <stdint.h>
#include "../cuda_check.h"
#include "layernorm.h"

// sum & sumsq reduction across a row, then normalize + affine
__global__ void layernorm_forward_kernel(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ gamma,
  				  	 const float* __restrict__ beta, int N, int D, float eps, 
  					 float* __restrict__ mean_out, float* __restrict__ invstd_out){ // may be nullptr
  int row = blockIdx.x; // one block per row
  if(row >= N) return;

  //shared memory layout: [0..blockDim.x) partial sums, [blockDim.x..2*blockDim.x) partial sumsq
  extern __shared__ float smem[];
  float* sh_sum = smem;
  float* sh_sumsq = smem + blockDim.x;

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  //accumulate partials for this row
  size_t base = (size_t)row * (size_t)D;
  float sum = 0.0f;
  float sumsq = 0.0f;
  for(int j = tid; j < D; j += stride){
    float v = x[base + j];
    sum += v;
    sumsq += v * v;
  }
  sh_sum[tid] = sum;
  sh_sumsq[tid] = sumsq;
  __syncthreads();

  //parallel reduction
  for(int s = blockDim.x >> 1; s > 0; s >>= 1){
    if(tid < s){
      sh_sum[tid] += sh_sum[tid + s];
      sh_sumsq[tid] += sh_sumsq[tid + s];
    }
  __syncthreads();
  }

  //compute mean and inv_std; broadcast via shared memory slot 0/1
  float mean = sh_sum[0] / (float)D;
  float var = sh_sumsq[0] / (float)D - mean * mean;
  float inv_std = rsqrtf(var + eps);

  if(tid == 0){
    sh_sum[0] = mean; //reuse to broadcast
    sh_sumsq[0] = inv_std; //reuse to broadcast
    if(mean_out) mean_out[row] = mean;
    if(invstd_out) invstd_out[row] = inv_std;
  }
  __syncthreads();


  mean = sh_sum[0];
  inv_std = sh_sumsq[0];

  //normalize and affine transform
  for(int j = tid; j < D; j += stride){
    float v = (x[base + j] - mean) * inv_std;
    float out = v * gamma[j] + beta[j];
    y[base + j] = out;
  }
}


extern "C" void nano2_layernorm_forward(const float* x, float* y,
                                        const float* gamma, const float* beta,
                                        int N, int D, float eps) {
    static float* d_mean  = nullptr;
    static float* d_rstd  = nullptr;
    static int    cap_N   = 0;

    if (N > cap_N) {
        if (d_mean) cudaFree(d_mean);
        if (d_rstd) cudaFree(d_rstd);
        cudaMalloc(&d_mean, (size_t)N * sizeof(float));
        cudaMalloc(&d_rstd, (size_t)N * sizeof(float));
        cap_N = N;
    }

    int threads = (D >= 256) ? 256 : (D >= 128 ? 128 : 64);
    dim3 block(threads, 1, 1), grid(N, 1, 1);
    size_t shmem = (size_t)threads * 2 * sizeof(float);

    layernorm_forward_kernel<<<grid, block, shmem>>>(x, y, gamma, beta, N, D, eps, d_mean, d_rstd);
    CUDA_CHECK("layernorm_forward");
}

//backward for row-wise LayerNorm
// Inputs per row (length D): x, dy, gamma, saved mean+invstd
// Outputs: dx[row,D], plus dgamma[D], dbeta[D] (accumulated across rows).
__global__ void layernorm_backward_kernel(const float* __restrict__ x,
    					  const float* __restrict__ dy, const float* __restrict__ gamma,
   					  const float* __restrict__ mean,    // [N]
    					  const float* __restrict__ invstd,  // [N]
   					  float* __restrict__ dx,
    					  float* __restrict__ dgamma,        // [D], to be atomically accumulated
   					  float* __restrict__ dbeta,         // [D], to be atomically accumulated
    					  int N, int D)
    					  {
    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float smem[];
    float* s1 = smem;              // sum(dy*gamma)
    float* s2 = smem + blockDim.x; // sum(dy*gamma * x_hat)

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const size_t base = (size_t)row * (size_t)D;

    const float mu = mean[row];
    const float iv = invstd[row];

    //reductions for s1 and s2; also accumulate per-dim dgamma/dbeta
    float t1 = 0.f, t2 = 0.f;
    for (int j = tid; j < D; j += stride){
        float xhat = (x[base + j] - mu) * iv;
        float g    = gamma[j];
        float dyj  = dy[base + j];
        t1 += dyj * g;
        t2 += dyj * g * xhat;

        //dgamma += dy * xhat ; dbeta += dy
        atomicAdd(&dgamma[j], dyj * xhat);
        atomicAdd(&dbeta[j],  dyj);
    }
    s1[tid] = t1; s2[tid] = t2;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1){
        if (tid < s){ s1[tid] += s1[tid + s]; s2[tid] += s2[tid + s]; }
        __syncthreads();
    }
    const float sum1 = s1[0];
    const float sum2 = s2[0];
    const float invD = 1.0f / (float)D;

    for (int j = tid; j < D; j += stride){
        float xhat = (x[base + j] - mu) * iv;
        float g    = gamma[j];
        float dyj  = dy[base + j];

        float dxhat = dyj * g;
        //dx = (1/D) * invstd * (D*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
        float term = (float)D * dxhat - sum1 - xhat * sum2;
        dx[base + j] = invD * iv * term;
    }
}

extern "C" void nano2_layernorm_backward(
    const float* x, const float* dy,
    const float* gamma,
    const float* mean, const float* invstd,
    float* dx, float* dgamma, float* dbeta,
    int N, int D)
{
    //dgamma/dbeta are expected zeroed by caller (we atomicAdd into them)
    int threads = (D >= 256) ? 256 : (D >= 128 ? 128 : 64);
    dim3 block(threads,1,1), grid(N,1,1);
    size_t shmem = (size_t)threads * 2 * sizeof(float);
    layernorm_backward_kernel<<<grid, block, shmem>>>(
        x, dy, gamma, mean, invstd, dx, dgamma, dbeta, N, D);
    CUDA_CHECK("layernorm_backward");
}

