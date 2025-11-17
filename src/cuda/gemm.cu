//typical call: nano2_gemm_f32(0,0, M,N,K, A, K, B, N, C, N, 1.0f, 0.0f);
//For QK^T (scores): transB = 1 (since K^T is needed)
//TODO: rm warns
#include <cuda_runtime.h>
#include <stdint.h>
#include "../cuda_check.h"

static inline __host__ __device__ int div_up(int a, int b){ 
  return (a+b-1)/b; 
}

//access hlps for row‑major buffs with optional transpose
//if trans==0: element(i,j) => base[i*ld + j], where ld = cols
//if trans==1: we view A as (cols x rows), so element(i,j) => base[j*ld + i]
__device__ __forceinline__ float ld_elem(const float* base, int ld, int i, int j, int trans){
  return trans ? base[j*ld + i] : base[i*ld + j];
}


//each thread computes one output element C(i,j)
__global__ void gemm_naive_kernel(int transA, int transB, int M, int N, int K, const float* __restrict__ A, int lda,  
		const float* __restrict__ B, int ldb, float* __restrict__ C, int ldc, float alpha, float beta){
  int j = blockIdx.x * blockDim.x + threadIdx.x; //col of C
  int i = blockIdx.y * blockDim.y + threadIdx.y; //row of C
  if(i >= M || j >= N) return;
    float acc = 0.0f;
    //sum over K
    for(int p=0; p<K; ++p){
      float a= ld_elem(A, lda, i, p, transA);
      float b= ld_elem(B, ldb, p, j, transB);
      acc += a*b;
    }

  //C(i,j) stored with ldc=N (if contiguous)
  float out = alpha*acc;
  if(beta != 0.0f){ 
    out += beta * C[i*ldc + j]; 
  }
  C[i*ldc + j] = out;
}


extern "C" void nano2_gemm_f32(int transA, int transB, int M, int N, int K, const float* A, int lda,
                               const float* B, int ldb, float* C, int ldc, float alpha, float beta){
  const dim3 block(16,16,1);
  const dim3 grid(div_up(N, block.x), div_up(M, block.y), 1);
  gemm_naive_kernel<<<grid, block>>>(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
  //no cudaDeviceSynchronize()
  //debug
  CUDA_CHECK("gemm_naive_kernel");
}

