//single head causal self attn fw (fp32)
//shapes (row-major):
// x_ln : [B*T, D] (LayerNorm'd tokens)
// Wq/Wk/Wv/Wo: [D, D]
// q/k/v: [B*T, D]
// scores, probs (per batch): [T, T]
// ctx, out: [B*T, D]
//for every batch b:
// Q= X_b @ Wq
// K= X_b @ Wk
// V= X_b @ Wv
// S= (Q @ K^T) * (1/sqrt(D)); apply causal mask (j>i -> -INF)
// P= softmax(S) row-wise
// C= P @ V
// O= C @ Wo
//using GEMM and softmax

//TODO: rm debugs
#include <cuda_runtime.h>
#include <math.h>
#include "../cuda_check.h"
//GEMM: C=alpha * op(A) * op(B) + beta * C
extern "C" void nano2_gemm_f32(
int transA, int transB,
int M, int N, int K,
const float* A, int lda,
const float* B, int ldb,
float* C, int ldc,
float alpha, float beta);

//stable row-wise softmax
extern "C" void nano2_softmax_forward(const float* x, float* y, int rows, int cols);

//hlpr
static inline __host__ __device__ int div_up(int a, int b){ return (a + b - 1) / b; }

//scaling and causal mask to a [T,T] score matrix
//scale = 1/sqrt(D); for j>i set to a large neg num
__global__ void attn_scale_and_causal_mask(float* __restrict__ S, int T, float scale){
  int j= blockIdx.x * blockDim.x + threadIdx.x; //col
  int i= blockIdx.y * blockDim.y + threadIdx.y; //row
  if(i >= T || j >= T) return;
  float v = S[(size_t)i * (size_t)T + j] * scale;
  if(j>i) v = -1e9f;
  S[(size_t)i * (size_t)T + j] = v;
}

//dot prod scores with scaling + causal mask (no GEMM)
//Q,K: [T,D] row-major for one batch item, S: [T,T]
__global__ void qk_scores_scale_mask_kernel(
    const float* __restrict__ Q, //[T,D]
    const float* __restrict__ K, //[T,D]
    float* __restrict__ S, //[T,T]
    int T, int D, float scale){
    int j= blockIdx.x * blockDim.x + threadIdx.x; //col
    int i= blockIdx.y * blockDim.y + threadIdx.y; //row
    if (i >= T || j >= T) return;

    // dot(Q[i], K[j])
    float acc=0.0f;
    const size_t qbase = (size_t)i * (size_t)D;
    const size_t kbase = (size_t)j * (size_t)D;
    for (int p=0; p<D; ++p){
        acc += Q[qbase + p] * K[kbase + p];
    }
    float v= acc * scale;
    if (j>i) v = -1e9f; // causal mask
    S[(size_t)i * (size_t)T + j] = v;
}


extern "C" void nano2_attention_forward(const float* x_ln, // [B*T, D]
                                        //inputs
					int B, int T, int D,
  					const float* Wq, const float* Wk, const float* Wv, const float* Wo, // [D,D]
  					//workspaces/outputs
  					float* q, float* k, float* v, // [B*T, D]
 					float* scores, float* probs, // [B*T*T] (treated as B slabs of [T,T])
  					float* ctx, // [B*T, D] temporary
  					float* out) // [B*T, D] result after Wo
  				        {
  const int BT= B * T;

  //project to Q,K,V (batched as a single GEMM per matrix)
  // Q = X @ Wq
  nano2_gemm_f32(/*transA=*/0, /*transB=*/0,
                 /*M=*/BT, /*N=*/D, /*K=*/D,
                 x_ln, /*lda=*/D,
                 Wq, /*ldb=*/D,
                 q, /*ldc=*/D,
                 /*alpha=*/1.0f, /*beta=*/0.0f);
  //K = X @ Wk
  nano2_gemm_f32(0,0, BT, D, D, x_ln, D, Wk, D, k, D, 1.0f, 0.0f);
  //V = X @ Wv
  nano2_gemm_f32(0,0, BT, D, D, x_ln, D, Wv, D, v, D, 1.0f, 0.0f);
  CUDA_CHECK("attn QKV gemms");


  //for each batch item:
  //compute scores
  //apply mask+softmax
  //then C = P@V and O = C@Wo
  const float scale = 1.0f / sqrtf((float)D);
  dim3 block(16,16,1);
  
  for (int b=0; b<B; ++b){
    const size_t off_td= (size_t)b * (size_t)T * (size_t)D;
    const size_t off_tt= (size_t)b * (size_t)T * (size_t)T;

    const float* Qb= q + off_td;
    const float* Kb= k + off_td;
    const float* Vb= v + off_td;
    float* Sb= scores + off_tt; // [T,T]
    float* Pb= probs  + off_tt; // [T,T]
    float* Cb= ctx    + off_td; // [T,D]
    float* Ob= out    + off_td; // [T,D]

    //S= Qb @ Kb^T
    //nano2_gemm_f32(/*transA=*/0, /*transB=*/1, T, T, D, Qb, D, Kb, D, Sb, T, 1.0f, 0.0f);
    //cudaDeviceSynchronize(); CUDA_CHECK("attn scores gemm");

    //S= Qb @ Kb^T  (via direct kernel)
    {
      dim3 block(16,16,1);
      dim3 grid((T + block.x - 1)/block.x, (T + block.y - 1)/block.y, 1);
      qk_scores_scale_mask_kernel<<<grid, block>>>(Qb, Kb, Sb, T, D, scale);
      CUDA_CHECK("attn qk kernel");
      cudaDeviceSynchronize();
      CUDA_CHECK("attn qk sync");
    }

    //scale + mask
    //dim3 block(16,16,1);
    //dim3 grid(div_up(T, block.x), div_up(T, block.y), 1);
    //attn_scale_and_causal_mask<<<grid, block>>>(Sb, T, scale);
    //cudaDeviceSynchronize(); CUDA_CHECK("attn scale+mask");

    //softmax(S) -> P
    nano2_softmax_forward(Sb, Pb, T, T);
    cudaDeviceSynchronize(); CUDA_CHECK("attn softmax");

    //C= P @ Vb
    nano2_gemm_f32(0,0, T, D, T, Pb, T, Vb, D, Cb, D, 1.0f, 0.0f);
    cudaDeviceSynchronize(); CUDA_CHECK("attn ctx gemm");

    //O= C @ Wo
    nano2_gemm_f32(0,0, T, D, D, Cb, D, Wo, D, Ob, D, 1.0f, 0.0f);
    cudaDeviceSynchronize(); CUDA_CHECK("attn proj gemm");
  }

  cudaDeviceSynchronize(); //sync?
  CUDA_CHECK("attn sync");
}

