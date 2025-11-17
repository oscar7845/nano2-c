//cross-entropy with stability (and maybe/opt dlogits).
//given logits[N,V] and int targets[N], calculate:
//loss_i = -log softmax(logits_i)[target_i] and return the mean over N. 
//If dlogits != nullptr, write gradient of the
//MEAN loss w.r.t. logits (so each row's grad is scaled by 1/N):
//dlogits[i, j] = softmax(logits_i)[j] - 1{j == target_i}, then / N
//
//notes:
//- one blk per row (N). Threads stride across V.
//- two pass reduction per row: reduce max, then reduce sum(exp(x-max)).
//- use single atomicAdd into a device scalar for sum of losses; wrapper
//copies it back and divides by N.
//TODO:
#include <cuda_runtime.h>
#include <math_constants.h>
#include "../cuda_check.h"

static inline __host__ __device__ int div_up(int a, int b){ return (a + b - 1) / b; }

__global__ void xent_forward_kernel(const float* __restrict__ logits, //[N,V]
				    const int* __restrict__ targets,//[N]
				    int N, int V,
				    float* __restrict__ dlogits, //[N,V] or nullptr
				    float* __restrict__ loss_sum) //scalar on device (init to 0)
				    {
  int row= blockIdx.x; //one blk per row
  if(row >= N) return;

  extern __shared__ float smem[];
  float* smax= smem; //[blockDim.x]
  float* ssum= smem + blockDim.x; //[blockDim.x]

  const int tid= threadIdx.x;
  const int stride= blockDim.x;
  const size_t base= (size_t)row * (size_t)V;


  //row max for numerical stability
  float m= -CUDART_INF_F;
  for(int j=tid; j<V; j += stride){
    float v= logits[base + j];
    m=fmaxf(m,v);
  }
  smax[tid]=m;
  __syncthreads();
  for(int s=blockDim.x >> 1; s > 0; s >>= 1){
    if(tid<s){ 
      smax[tid] = fmaxf(smax[tid], smax[tid + s]); 
    }
    __syncthreads();
  }
  float row_max=smax[0];


  //sum of exp(logits - row_max)
  float sum = 0.0f;
  for(int j = tid; j < V; j += stride){
    sum += expf(logits[base + j] - row_max);
  }
  ssum[tid] = sum;
  __syncthreads();
  for(int s = blockDim.x >> 1; s > 0; s >>= 1){
    if(tid < s){ 
      ssum[tid] += ssum[tid + s]; 
    }
    __syncthreads();
  }
  float row_sum = ssum[0] + 1e-20f; // guard

  //per-row loss contribution (thread 0)
  if(tid==0){
    int t= targets[row];
    float zt= logits[base + t];
    float lse= row_max + logf(row_sum);
    float loss= lse - zt;
    atomicAdd(loss_sum, loss);
  }

  //optional gradient of MEAN loss
  if(dlogits){
    //i will scale by 1/N here so the caller can use it directly
    float invN = 1.0f / (float)N;
    int t=0; //bcast target index to all threads via shared memory slot 0
    if(tid==0){ 
      ((int*)smax)[0] = targets[row]; 
    }
    __syncthreads();
    t = ((int*)smax)[0];

    for(int j=tid; j<V; j += stride){
      float p= expf(logits[base + j] - row_max) / row_sum;
      float g= p-(j == t ? 1.0f : 0.0f);
      dlogits[base+j] = g*invN;
    }
  }
}


extern "C" float nano2_xent_forward_mean(const float* logits, const int* targets,
  					 int rows, int cols, float* dlogits /*nullable*/)
					 {
  if(rows<=0 || cols<=0) return 0.0f;
  //choose threads per row (something like pow of two)
  int threads= (cols >= 256) ? 256 : (cols >= 128 ? 128 : 64);
  dim3 block(threads,1,1);
  dim3 grid(rows,1,1);
  size_t shmem= (size_t)threads * 2 * sizeof(float);

  float* d_sum=nullptr;
  cudaMalloc(&d_sum, sizeof(float));
  cudaMemset(d_sum, 0, sizeof(float));

  xent_forward_kernel<<<grid, block, shmem>>>(logits, targets, rows, cols, dlogits, d_sum);
  CUDA_CHECK("xent_forward_kernel");
  cudaDeviceSynchronize(); //expose any runtime errors now
  CUDA_CHECK("xent_forward_sync");

  float h_sum=0.0f;
  cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_sum);

  return h_sum / (float)rows;
}

