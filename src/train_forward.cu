//fw pass:
//- embed + sinusoidal pos
//- pre-LN --> single-head causal attention --> residual
//- pre-LN --> FFN (GELU) --> residual
//- tied LM head (X @ E^T)
//- cross-entropy mean loss (returns host float)
//
//i assume shapes/dims match the Config/Model.
//allocate temp device buffers for input tokens and targets
//later, move those to Model so to avoid the per-step allocs
//TODO: rm debug prints
#include "nano2_model.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "cuda_check.h"

//external ops
extern "C" void nano2_layernorm_forward(const float* x, float* y, const float* gamma, const float* beta,
                                        int N, int D, float eps, float* mean_out, float* invstd_out);
extern "C" void nano2_attention_forward(const float* x_ln, int B, int T, int D,
					const float* Wq, const float* Wk, const float* Wv, const float* Wo,
					float* q, float* k, float* v, float* scores, float* probs,
					float* ctx, float* out);
extern "C" void nano2_gemm_f32(int transA, int transB, int M, int N, int K, const float* A, int lda,
			       const float* B, int ldb, float* C, int ldc, float alpha, float beta);
extern "C" void nano2_gelu_forward(const float* x, float* y, int n, int approximate);
extern "C" float nano2_xent_forward_mean(const float* logits, const int* targets,
int rows, int cols, float* dlogits /*nullable*/);

//convert uint8 targets to int32
__global__ void u8_to_i32_kernel(const uint8_t* __restrict__ in, int* __restrict__ out, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(; i < n; i += stride){ 
    out[i] = (int)in[i]; 
  }
}

//y=a+b (elementwise over n floats)
__global__ void add_inplace_kernel(float* __restrict__ y, const float* __restrict__ b, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(; i<n; i += stride){ 
    y[i] += b[i]; 
  }
}

//X[rows,cols] += bias[cols]
__global__ void add_bias_inplace_kernel(float* __restrict__ X, const float* __restrict__ bias,
  int rows, int cols){
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if(col >= cols) return;
  for(int r=0; r<rows; ++r){ 
    X[(size_t)r * (size_t)cols + col] += bias[col]; 
  }
}

//embed tokens and add sinusoidal positions: out[row, d] = E[token, d] + pos_{t}[d]
//i use half-dim sin/cos tables and mix them: even d -> sin, odd d -> cos.
/**__global__ void embed_add_pos_kernel(const uint8_t* __restrict__ tokens, const float* __restrict__ E,
		const float* __restrict__ pos_sin, const float* __restrict__ pos_cos, float* __restrict__ out,
		int B, int T, int D){
  int row = blockIdx.x; //which token in [0, B*T)
  if (row >= B*T) return;
  int tid=threadIdx.x;
  int stride=blockDim.x;
  const int H=D>>1; //D assumed even
  int tok=(int)tokens[row];
  int t=row%T; //position within sequence
  size_t ebase= (size_t)tok * (size_t)D;
  size_t obase= (size_t)row * (size_t)D;
  size_t pbase= (size_t)t * (size_t)H;
  for(int d=tid; d<D; d += stride){
    float v = E[ebase+d];
    int i=d>>1;
    v += ( (d&1) == 0 ? pos_sin[pbase + i] : pos_cos[pbase + i] );
    out[obase+d]=v;
  }
}
**/

//run one forward + mean loss
extern "C" float nano2_forward_loss(struct Model* M, 
		                    const uint8_t* h_tokens_x, // [B*T] host
				    const uint8_t* h_tokens_y) // [B*T] host
				    {
  const int B=M->B, T=M->T, D=M->D, V=M->V, F=M->F;
  const int BT= B*T;

  //device buffers for tokens/targets (temp for now)
  uint8_t *d_x = nullptr, *d_y_u8 = nullptr; int *d_y = nullptr;
  cudaMalloc(&d_x, (size_t)BT * sizeof(uint8_t));
  cudaMalloc(&d_y_u8, (size_t)BT * sizeof(uint8_t));
  cudaMalloc(&d_y, (size_t)BT * sizeof(int));
  cudaMemcpy(d_x, h_tokens_x, (size_t)BT, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_u8, h_tokens_y, (size_t)BT, cudaMemcpyHostToDevice);
  {
    int block=256; int grid=(BT+block-1) / block; if(grid>65535) grid=65535;
    u8_to_i32_kernel<<<grid, block>>>(d_y_u8, d_y, BT);
    CUDA_CHECK("u8_to_i32");
  }
  cudaFree(d_y_u8);


  //embedding + positions --> x[BT,D]
  {
    dim3 block(256,1,1);
    dim3 grid(BT,1,1);
    //embed_add_pos_kernel<<<grid, block>>>(d_x, M->p.E, M->pos_sin, M->pos_cos, M->buf.x, B, T, D);
    CUDA_CHECK("embed_add_pos");
  }


  //TokenMix: pre-LN --> attention --> residual
  nano2_layernorm_forward(M->buf.x, M->buf.x_ln1,
  			  M->p.ln1_g, M->p.ln1_b,
  			  /*N=*/BT, /*D=*/D, /*eps=*/1e-5f,
  			  /*mean_out=*/nullptr, /*invstd_out=*/nullptr);
  CUDA_CHECK("ln1");
  nano2_attention_forward(M->buf.x_ln1, B, T, D,
  			  M->p.Wq, M->p.Wk, M->p.Wv, M->p.Wo,
  			  M->buf.q, M->buf.k, M->buf.v,
  			  M->buf.scores, M->buf.probs,
  			  M->buf.x_res1, // ctx temp
  			  M->buf.attn_out); // out = O
  CUDA_CHECK("attention");
  {    
    //x_res1 = x+attn_out
    int n = BT * D; int block = 256; int grid = (n + block - 1) / block; if (grid < 1) grid = 1; if(grid > 65535) grid = 65535;
    //cpy x to x_res1, then add attn_out
    cudaMemcpy(M->buf.x_res1, M->buf.x, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice);
    add_inplace_kernel<<<grid, block>>>(M->buf.x_res1, M->buf.attn_out, n);
    CUDA_CHECK("residual1 add");
  }


  //FFN: pre-LN -->  W1+b1 --> GELU --> W2+b2 --> residual
  nano2_layernorm_forward(M->buf.x_res1, M->buf.x_ln2,
  			  M->p.ln2_g, M->p.ln2_b,
  			  /*N=*/BT, /*D=*/D, /*eps=*/1e-5f,
  			  /*mean_out=*/nullptr, /*invstd_out=*/nullptr);
  CUDA_CHECK("ln2");
  
  //ff1 = x_ln2 @ W1, shape [BT,F]
  nano2_gemm_f32(0,0, /*M=*/BT, /*N=*/F, /*K=*/D,
  M->buf.x_ln2, /*lda=*/D,
  M->p.W1, /*ldb=*/F,
  M->buf.ff1, /*ldc=*/F,
  1.0f, 0.0f);
  // + b1
  {
    int threads = (F >= 256) ? 256 : (F >= 128 ? 128 : 64);
    dim3 block(threads,1,1); dim3 grid( (F + threads - 1)/threads, 1, 1 );
    add_bias_inplace_kernel<<<grid, block>>>(M->buf.ff1, M->p.b1, BT, F);
    CUDA_CHECK("b1 add");
  }
  //GELU(ff1) --> ff1 (in-place ok: use separate output then swap; here write to ff1)
  nano2_gelu_forward(M->buf.ff1, M->buf.ff1, /*n=*/BT*F, /*approximate=*/1);
  CUDA_CHECK("gelu");

  //ff2 = ff1 @ W2, shape [BT,D]
  nano2_gemm_f32(0,0, /*M=*/BT, /*N=*/D, /*K=*/F,
  	         M->buf.ff1, /*lda=*/F,
  		 M->p.W2, /*ldb=*/D,
  		 M->buf.ff2, /*ldc=*/D,
  		 1.0f, 0.0f);
  //+b2
  {
    int threads= (D >= 256) ? 256 : (D >= 128 ? 128 : 64);
    dim3 block(threads,1,1); dim3 grid( (D + threads - 1)/threads, 1, 1 );
    add_bias_inplace_kernel<<<grid, block>>>(M->buf.ff2, M->p.b2, BT, D);
    CUDA_CHECK("b2 add");
  }

  //x_res2 = x_res1+ff2
  {
    int n = BT*D; int block = 256; int grid=(n+block-1) / block; if (grid < 1) grid = 1; if(grid > 65535) grid = 65535;
    cudaMemcpy(M->buf.x_res2, M->buf.x_res1, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice);
    add_inplace_kernel<<<grid, block>>>(M->buf.x_res2, M->buf.ff2, n);
    CUDA_CHECK("residual2 add");
  }


  //LM head (tied): logits = x_res2 @ E^T --> [BT,V]
  nano2_gemm_f32(/*transA=*/0, /*transB=*/1,
  		 /*M=*/BT, /*N=*/V, /*K=*/D,
  		 M->buf.x_res2, /*lda=*/D,
  		 M->p.E, /*ldb=*/D,
 		 M->buf.logits, /*ldc=*/V,
  		 1.0f, 0.0f);
  CUDA_CHECK("logits gemm");


  //Loss (mean over BT)
  float mean_loss = nano2_xent_forward_mean(M->buf.logits, d_y,
  /*rows=*/BT, /*cols=*/V,
  /*dlogits=*/nullptr);

  #ifdef NANO2_DEBUG_FIRSTROW
    //cpy logits[0] row (length V) and y[0] to host and compute CE on CPU
    float* h_logits0 = (float*)malloc((size_t)V * sizeof(float));
    int h_t0=0;
    cudaMemcpy(h_logits0, M->buf.logits, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_t0, d_y, sizeof(int), cudaMemcpyDeviceToHost);

    //stable softmax + CE
    float m=-1e30f, s=0.f;
    for (int j=0;j<V;++j) if (h_logits0[j] > m) m = h_logits0[j];
    for (int j=0;j<V;++j) s += expf(h_logits0[j] - m);
    float lse = m+logf(s);
    float cpu_loss0 = lse - h_logits0[h_t0];
    printf("[debug] first-row CE = %.6f, target=%d\n", cpu_loss0, h_t0);
    free(h_logits0);
  #endif


  cudaFree(d_x);
  cudaFree(d_y);
  cudaDeviceSynchronize();
  return mean_loss;
}

