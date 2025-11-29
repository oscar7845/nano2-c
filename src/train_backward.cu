//one training step: 
//forward (reusing fw kernels), 
//backward to fill grads,
//optional MPI allreduce, 
//grad-norm clip, 
//AdamW update
//TODO: check decay prints
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include "nano2_model.h"
#include "cuda_check.h"

#include "cuda/embed.h"
#include "cuda/layernorm.h"

#define TAG(x) CUDA_CHECK(x)

//externs
extern "C" void nano2_gemm_f32(int transA, int transB, int M, int N, int K,
                                const float* A, int lda, const float* B, int ldb,
                                float* C, int ldc, float alpha, float beta);

extern "C" void nano2_gelu_forward(const float* x, float* y, int n, int approximate);

extern "C" void nano2_softmax_forward(const float* x, float* y, int rows, int cols);

extern "C" void nano2_attention_forward(const float* x_ln, int B, int T, int D,
                                        const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                                        float* q, float* k, float* v,
                                        float* scores, float* probs,
                                        float* ctx_tmp, float* out);

//extern "C" void nano2_layernorm_forward(const float* x, float* y,
//                                        const float* gamma, const float* beta,
//                                        int N, int D, float eps);

extern "C" void nano2_clip_grad_global_norm(float* g, size_t n, float max_norm);
extern "C" void nano2_adamw_step(float* params, float* grads, float* m, float* v, size_t n,
		float lr, float beta1, float beta2, float eps, float weight_decay);

//(attention_backward.cu)
extern "C" void nano2_softmax_backward(const float* P, const float* dP, float* dS, int rows, int cols);

//Utilities reused from forward file
static __global__ void u8_to_i32_kernel(const uint8_t* __restrict__ in, int* __restrict__ out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x; int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) out[i] = (int)in[i];
}
static __global__ void add_inplace_kernel(float* __restrict__ y, const float* __restrict__ b, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x; int s = blockDim.x * gridDim.x;
    for (; i < n; i += s) y[i] += b[i];
}
static __global__ void add_bias_inplace_kernel(float* __restrict__ X, const float* __restrict__ bias,
                                        int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x; if (col >= cols) return;
    for (int r = 0; r < rows; ++r) X[(size_t)r * (size_t)cols + col] += bias[col];
}

//helper
//scale by 1/sqrt(D) and apply causal mask (j>i -> -inf)
static __global__ void scale_causal_mask_kernel(float* S, int T, float scale){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= T || j >= T) return;
    float v = S[(size_t)i * (size_t)T + j] * scale;
    if (j > i) v = -1e9f;
    S[(size_t)i * (size_t)T + j] = v;
}

static inline void gemm_checked(
    int transA, int transB, int M, int N, int K,
    const float* A, int lda, const float* B, int ldb,
    float* C, int ldc, float alpha, float beta,
    const char* label)
{
    nano2_gemm_f32(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, alpha, beta);
    // Hard-sync so the *right* call gets blamed; then print a descriptive label.
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr,
            "[GEMM ERROR] %s | transA=%d transB=%d M=%d N=%d K=%d lda=%d ldb=%d ldc=%d alpha=%.3g beta=%.3g : %s\n",
            label, transA, transB, M, N, K, lda, ldb, ldc, alpha, beta, cudaGetErrorString(err));
    }
    CUDA_CHECK(label);
}

//GELU backward (tanh approximation), in-place multiply on d
//d <- d * GELU'(z) in-place
static __global__ void gelu_bw_tanh_kernel(const float* __restrict__ z, // pre-activation
                                           float* __restrict__ d,       // upstream grad (dff1)
                                           int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockDim.x * gridDim.x;
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    for (; i < n; i += s){
        float x = z[i];
        float u = k0 * (x + k1 * x * x * x);
        float t = tanhf(u);
        float sech2 = 1.0f - t * t;
        float dgelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * k0 * (1.0f + 3.0f * k1 * x * x);
        d[i] *= dgelu;
    }
}

//softmax + Xent loss + dlogits (mean)
__global__ void xent_grad_mean_kernel(const float* __restrict__ logits, // [rows, cols]
                                      const uint8_t* __restrict__ targets, // [rows]
                                      int rows, int cols,
                                      float* __restrict__ dlogits, // [rows, cols]
                                      double* __restrict__ loss_sum) // scalar
				      {
    int row = blockIdx.x; if (row >= rows) return;
    extern __shared__ float smem[];
    float* smax = smem;
    float* ssum = smem + blockDim.x;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const size_t base = (size_t)row * (size_t)cols;

    //max
    float m = -1e30f;
    for (int j = tid; j < cols; j += stride) m = fmaxf(m, logits[base + j]);
    smax[tid] = m; __syncthreads();
    for (int t = blockDim.x >> 1; t > 0; t >>= 1){
        if (tid < t) smax[tid] = fmaxf(smax[tid], smax[tid + t]);
        __syncthreads();
    }
    m = smax[0];

    //sum exp
    float se = 0.0f;
    for (int j = tid; j < cols; j += stride) se += expf(logits[base + j] - m);
    ssum[tid] = se; __syncthreads();
    for (int t = blockDim.x >> 1; t > 0; t >>= 1){
        if (tid < t) ssum[tid] += ssum[tid + t];
        __syncthreads();
    }
    se = ssum[0];

    //loss contribution (once)
    int y = (int)targets[row];
    float zt = logits[base + y];
    float lse = m + logf(se);
    if (tid == 0){
        atomicAdd(loss_sum, (double)(lse - zt));
    }

    //gradient = (softmax - onehot) / rows
    float inv_rows = 1.0f / (float)rows;
    for (int j = tid; j < cols; j += stride){
        float p = expf(logits[base + j] - m) / se;
        float g = p - (j == y ? 1.0f : 0.0f);
        dlogits[base + j] = g * inv_rows;
    }
}

//LayerNorm backward (recompute stats)
__global__ void layernorm_bw_recompute_kernel(
    const float* __restrict__ x,   // [N,D]
    const float* __restrict__ dy,  // [N,D]
    const float* __restrict__ gamma, // [D]
    float eps,
    float* __restrict__ dx,        // [N,D]
    float* __restrict__ dgamma,    // [D] (atomic add)
    float* __restrict__ dbeta,     // [D] (atomic add)
    int N, int D){

    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float smem[];
    float* s1 = smem;             // reducers
    float* s2 = smem + blockDim.x;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const size_t base = (size_t)row * (size_t)D;

    //mean
    float sum = 0.f;
    for (int j = tid; j < D; j += stride) sum += x[base + j];
    s1[tid] = sum; __syncthreads();
    for (int t = blockDim.x >> 1; t > 0; t >>= 1){ if (tid < t) s1[tid] += s1[tid + t]; __syncthreads(); }
    float mu = s1[0] / (float)D;

    //variance
    float vpart = 0.f;
    for (int j = tid; j < D; j += stride){
        float d = x[base + j] - mu; vpart += d*d;
    }
    s1[tid] = vpart; __syncthreads();
    for (int t = blockDim.x >> 1; t > 0; t >>= 1){ if (tid < t) s1[tid] += s1[tid + t]; __syncthreads(); }
    float invstd = rsqrtf(s1[0] / (float)D + eps);

    //sums for dx formula; also dgamma/dbeta
    float t1 = 0.f, t2 = 0.f;
    for (int j = tid; j < D; j += stride){
        float xhat = (x[base + j] - mu) * invstd;
        float g    = gamma[j];
        float dyj  = dy[base + j];
        t1 += dyj * g;
        t2 += dyj * g * xhat;
        atomicAdd(&dgamma[j], dyj * xhat);
        atomicAdd(&dbeta[j],  dyj);
    }
    s1[tid] = t1; s2[tid] = t2; __syncthreads();
    for (int t = blockDim.x >> 1; t > 0; t >>= 1){
        if (tid < t){ s1[tid] += s1[tid + t]; s2[tid] += s2[tid + t]; }
        __syncthreads();
    }
    float sum1 = s1[0], sum2 = s2[0];
    float invD = 1.0f / (float)D;

    for (int j = tid; j < D; j += stride){
        float xhat = (x[base + j] - mu) * invstd;
        float g    = gamma[j];
        float dxhat = dy[base + j] * g;
        float term = (float)D * dxhat - sum1 - xhat * sum2;
        dx[base + j] = invD * invstd * term;
    }
}

//embedding scatter-add (tied)
__global__ void embed_scatter_add_kernel(const float* __restrict__ dx, // [rows,D]
                                         const uint8_t* __restrict__ tokens, // [rows]
                                         float* __restrict__ gE, // [V,D]
                                         int rows, int D){
    int row = blockIdx.x; if (row >= rows) return;
    int tid = threadIdx.x; int stride = blockDim.x;
    int tok = (int)tokens[row];
    size_t base = (size_t)row * (size_t)D;
    size_t ebase = (size_t)tok * (size_t)D;
    for (int d = tid; d < D; d += stride){
        atomicAdd(&gE[ebase + d], dx[base + d]);
    }
}

//rowwise sum over rows for bias grads
__global__ void rowwise_sum_cols_kernel(const float* __restrict__ Y, float* __restrict__ db,
                                        int rows, int cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x; if (col >= cols) return;
    float s = 0.f; for (int r = 0; r < rows; ++r) s += Y[(size_t)r * (size_t)cols + col];
    atomicAdd(&db[col], s);
}

//Train step
extern "C" float nano2_train_step(struct Model* M,
                                  const uint8_t* h_tokens_x,
                                  const uint8_t* h_tokens_y,
                                  const struct Config* cfg,
                                  int world_size, int rank)
				  {
    const int B = M->B, T = M->T, D = M->D, V = M->V, F = M->F, BT = B*T;

    //zero grads once per step
    cudaMemset(M->flat_grads, 0, M->n_params * sizeof(float));

    //Forward (same as forward loss) 
    //copy tokens
    uint8_t *d_x=nullptr, *d_y=nullptr;
    cudaMalloc(&d_x, (size_t)BT);
    cudaMalloc(&d_y, (size_t)BT);
    cudaMemcpy(d_x, h_tokens_x, (size_t)BT, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_tokens_y, (size_t)BT, cudaMemcpyHostToDevice);

    //embed + pos
    { 
      dim3 block(256,1,1), grid(BT,1,1);
      cudaMemset(M->buf.x, 0xCD, (size_t)BT * (size_t)D * sizeof(float));
      CUDA_CHECK("sentinel fill before embed");
      nano2_embed_add_pos(d_x, M->p.E, M->pos_sin, M->pos_cos, M->buf.x, B, T, D);
      CUDA_CHECK("embed_add_pos"); 
    }

    //LN1
    nano2_layernorm_forward(M->buf.x, M->buf.x_ln1, M->p.ln1_g, M->p.ln1_b, BT, D, 1e-5f);
    CUDA_CHECK("ln1 forward");
    
    //Attention (gemm-based, mirrors forward)
    nano2_gemm_f32(0,0, BT, D, D, M->buf.x_ln1, D, M->p.Wq, D, M->buf.q, D, 1.0f, 0.0f);
    CUDA_CHECK("gemm q = x_ln1 @ Wq");
    nano2_gemm_f32(0,0, BT, D, D, M->buf.x_ln1, D, M->p.Wk, D, M->buf.k, D, 1.0f, 0.0f);
    CUDA_CHECK("gemm k = x_ln1 @ Wk");
    nano2_gemm_f32(0,0, BT, D, D, M->buf.x_ln1, D, M->p.Wv, D, M->buf.v, D, 1.0f, 0.0f);
    CUDA_CHECK("gemm v = x_ln1 @ Wv");

    //scores, mask+softmax, context per batch
    const float scale = 1.0f / sqrtf((float)D);
    for (int b = 0; b < B; ++b){
      const size_t off_td = (size_t)b * (size_t)T * (size_t)D;
      const size_t off_tt = (size_t)b * (size_t)T * (size_t)T;

      const float* Qb = M->buf.q + off_td;
      const float* Kb = M->buf.k + off_td;
      float* Sb       = M->buf.scores + off_tt;
      float* Pb       = M->buf.probs  + off_tt;
      const float* Vb = M->buf.v      + off_td;
      float* Cb       = M->buf.x_res1 + off_td;  // tmp ctx

      //S = Q K^T
      nano2_gemm_f32(0,1, T, T, D, Qb, D, Kb, D, Sb, T, 1.0f, 0.0f);
      CUDA_CHECK("gemm S = Q @ K^T");

      //scale + causal mask
      dim3 block(16,16), grid((T+15)/16, (T+15)/16);
      scale_causal_mask_kernel<<<grid,block>>>(Sb, T, scale);
      CUDA_CHECK("attn scale+mask (train step)");

      //P = softmax(S)
      nano2_softmax_forward(Sb, Pb, T, T);
      CUDA_CHECK("attn softmax (train step)");

      //C = P V
      nano2_gemm_f32(0,0, T, D, T, Pb, T, Vb, D, Cb, D, 1.0f, 0.0f);
      CUDA_CHECK("gemm C = P @ V");
    }
    //O = C W_o
    nano2_gemm_f32(0,0, BT, D, D, M->buf.x_res1, D, M->p.Wo, D, M->buf.attn_out, D, 1.0f, 0.0f);
    CUDA_CHECK("gemm O = C @ Wo");

    //Residual1: x_res1 = x + attn_out
    { 
      int n=BT*D, block=256, grid=(n+block-1)/block; if(grid>65535) grid=65535;
      cudaMemcpy(M->buf.x_res1, M->buf.x, (size_t)n*sizeof(float), cudaMemcpyDeviceToDevice);
      add_inplace_kernel<<<grid,block>>>(M->buf.x_res1, M->buf.attn_out, n); CUDA_CHECK("res1 add"); }

    //LN2
    nano2_layernorm_forward(M->buf.x_res1, M->buf.x_ln2, M->p.ln2_g, M->p.ln2_b, BT, D, 1e-5f);

    //FFN: ff1 = GELU(x_ln2 @ W1 + b1); ff2 = ff1 @ W2 + b2
    nano2_gemm_f32(0,0, BT, F, D, M->buf.x_ln2, D, M->p.W1, F, M->buf.ff1, F, 1.0f, 0.0f);
    CUDA_CHECK("gemm ff1_pre = x_ln2 @ W1");
    { 
      int thr = (F>=256)?256:(F>=128?128:64); dim3 block(thr,1,1), grid((F+thr-1)/thr,1,1);
      add_bias_inplace_kernel<<<grid,block>>>(M->buf.ff1, M->p.b1, BT, F); CUDA_CHECK("b1 add"); 
    }
    nano2_gelu_forward(M->buf.ff1, M->buf.ff1, BT*F, 1);
    nano2_gemm_f32(0,0, BT, D, F, M->buf.ff1, F, M->p.W2, D, M->buf.ff2, D, 1.0f, 0.0f);
    CUDA_CHECK("gemm ff2 = ff1 @ W2");
    { 
      int thr = (D>=256)?256:(D>=128?128:64); dim3 block(thr,1,1), grid((D+thr-1)/thr,1,1);
      add_bias_inplace_kernel<<<grid,block>>>(M->buf.ff2, M->p.b2, BT, D); CUDA_CHECK("b2 add"); 
    }

    //residual2: x_res2 = x_res1 + ff2
    { 
      int n=BT*D, block=256, grid=(n+block-1)/block; if(grid>65535) grid=65535;
      cudaMemcpy(M->buf.x_res2, M->buf.x_res1, (size_t)n*sizeof(float), cudaMemcpyDeviceToDevice);
      add_inplace_kernel<<<grid,block>>>(M->buf.x_res2, M->buf.ff2, n); CUDA_CHECK("res2 add"); 
    }

    //logits = x_res2 @ E^T
    nano2_gemm_f32(0,1, BT, V, D, M->buf.x_res2, D, M->p.E, D, M->buf.logits, V, 1.0f, 0.0f);
    CUDA_CHECK("gemm logits = x_res2 @ E^T");

    //cross-entropy mean loss + dlogits
    float* dlogits=nullptr; cudaMalloc(&dlogits, (size_t)BT * (size_t)V * sizeof(float));
    double* d_loss_sum=nullptr; cudaMalloc(&d_loss_sum, sizeof(double)); cudaMemset(d_loss_sum, 0, sizeof(double));
    { 
      int threads = (V>=256)?256:(V>=128?128:64);
      dim3 block(threads,1,1), grid(BT,1,1);
      size_t shmem = (size_t)threads * 2 * sizeof(float);
      xent_grad_mean_kernel<<<grid, block, shmem>>>(M->buf.logits, d_y, BT, V, dlogits, d_loss_sum);
      CUDA_CHECK("xent_grad"); 
    }
    double h_sum=0.0; cudaMemcpy(&h_sum, d_loss_sum, sizeof(double), cudaMemcpyDeviceToHost);
    float mean_loss = (float)(h_sum / (double)BT);
    cudaFree(d_loss_sum);

    //Backward
    //carve grad views (same order as model.c carving)
    float* g = M->flat_grads;
    const size_t VD=(size_t)V*D, DD=(size_t)D*D, DF=(size_t)D*F, FD=(size_t)F*D;
    float *gE      = g;              g += VD;
    float *gln1_g  = g;              g += D;
    float *gln1_b  = g;              g += D;
    float *gWq     = g;              g += DD;
    float *gWk     = g;              g += DD;
    float *gWv     = g;              g += DD;
    float *gWo     = g;              g += DD;
    float *gln2_g  = g;              g += D;
    float *gln2_b  = g;              g += D;
    float *gW1     = g;              g += DF;
    float *gb1     = g;              g += F;
    float *gW2     = g;              g += FD;
    float *gb2     = g;              /*g+=D*/;

    //LM head
    float* dX2=nullptr; cudaMalloc(&dX2, (size_t)BT * D * sizeof(float));
    nano2_gemm_f32(0,0, BT, D, V, dlogits, V, M->p.E, D, dX2, D, 1.0f, 0.0f);       // dX2 = dlogits @ E
    CUDA_CHECK("gemm dX2 = dlogits @ E");
    TAG("dX2 = dlogits @ E");
    for (int b = 0; b < B; ++b) {
      size_t off_tv = (size_t)b * (size_t)T * (size_t)V;
      size_t off_td = (size_t)b * (size_t)T * (size_t)D;
      // dlogits_b: [T,V], x_res2_b: [T,D]
      nano2_gemm_f32(1,0, V, D, T,
                   dlogits       + off_tv, V,
                   M->buf.x_res2 + off_td, D,
                   gE, D, 1.0f, 1.0f);
      CUDA_CHECK("gemm gE += dlogits^T @ x_res2 (per-batch)");
    }

    //Residual2 split
    float* dRes1=nullptr; cudaMalloc(&dRes1, (size_t)BT * D * sizeof(float));
    cudaMemcpy(dRes1, dX2, (size_t)BT*D*sizeof(float), cudaMemcpyDeviceToDevice);

    //FFN backward
    float* dff2=nullptr; cudaMalloc(&dff2, (size_t)BT * D * sizeof(float));
    cudaMemcpy(dff2, dX2, (size_t)BT*D*sizeof(float), cudaMemcpyDeviceToDevice);

    //gW2 += ff1^T @ dff2 ; db2 += sum_r dff2 ; dff1 = dff2 @ W2^T
    for (int b = 0; b < B; ++b) {
      size_t off_td = (size_t)b * (size_t)T * (size_t)D;
      size_t off_tf = (size_t)b * (size_t)T * (size_t)F;
      // ff1_b: [T,F], dff2_b: [T,D]
      nano2_gemm_f32(1,0, F, D, T,
                   M->buf.ff1 + off_tf, F,
                   dff2        + off_td, D,
                   gW2, D, 1.0f, 1.0f);
      CUDA_CHECK("gemm gW2 += ff1^T @ dff2 (per-batch)");
    }
    { 
      int thr=(D>=256)?256:(D>=128?128:64); dim3 block(thr,1,1), grid((D+thr-1)/thr,1,1);
      rowwise_sum_cols_kernel<<<grid,block>>>(dff2, gb2, BT, D); CUDA_CHECK("db2"); 
    }
    float* dff1=nullptr; cudaMalloc(&dff1, (size_t)BT * F * sizeof(float));
    nano2_gemm_f32(0,1, BT, F, D, dff2, D, M->p.W2, D, dff1, F, 1.0f, 0.0f);
    CUDA_CHECK("gemm dff1 = dff2 @ W2^T");

   
    //recompute pre-activation z = x_ln2 @ W1 + b1, then apply GELU
    float* z = nullptr; 
    cudaMalloc(&z, (size_t)BT * (size_t)F * sizeof(float));
    //z = x_ln2 @ W1
    nano2_gemm_f32(0,0, BT, F, D, M->buf.x_ln2, D, M->p.W1, F, z, F, 1.0f, 0.0f);
    CUDA_CHECK("gelu bw: z = x_ln2 @ W1");
    //z += b1 rowwise
    {
      int thr = (F>=256)?256:(F>=128?128:64);
      dim3 block(thr,1,1), grid((F+thr-1)/thr,1,1);
      add_bias_inplace_kernel<<<grid,block>>>(z, M->p.b1, BT, F);
      CUDA_CHECK("gelu bw: add b1");
    }
    //apply GELU in-place to dff1
    {
      int n = BT * F;
      int blk = 256;
      int grd = (n + blk - 1) / blk; if (grd > 65535) grd = 65535;
      gelu_bw_tanh_kernel<<<grd, blk>>>(z, dff1, n);
      CUDA_CHECK("gelu bw: apply dgelu");
    }
    cudaFree(z);


    //gW1 += x_ln2^T @ dff1 ; db1 += sum_r dff1 ; dXln2 = dff1 @ W1^T
    for (int b = 0; b < B; ++b) {
      size_t off_td = (size_t)b * (size_t)T * (size_t)D;
      size_t off_tf = (size_t)b * (size_t)T * (size_t)F;
      // x_ln2_b: [T,D], dff1_b: [T,F]
      nano2_gemm_f32(1,0, D, F, T,
                   M->buf.x_ln2 + off_td, D,
                   dff1         + off_tf, F,
                   gW1, F, 1.0f, 1.0f);
      CUDA_CHECK("gemm gW1 += x_ln2^T @ dff1 (per-batch)");
    }

    { 
      int thr=(F>=256)?256:(F>=128?128:64); dim3 block(thr,1,1), grid((F+thr-1)/thr,1,1);
      rowwise_sum_cols_kernel<<<grid,block>>>(dff1, gb1, BT, F); CUDA_CHECK("db1"); 
    }
    float* dXln2=nullptr; cudaMalloc(&dXln2, (size_t)BT * D * sizeof(float));
    nano2_gemm_f32(0,1, BT, D, F, dff1, F, M->p.W1, F, dXln2, D, 1.0f, 0.0f);
    CUDA_CHECK("gemm dXln2 = dff1 @ W1^T");

    //LN2 backward (recompute stats)
    float* dXres1_from_ln2=nullptr; cudaMalloc(&dXres1_from_ln2, (size_t)BT*D*sizeof(float));
    {
      int threads = (D>=256)?256:(D>=128?128:64);
      dim3 block(threads,1,1), grid(BT,1,1);
      size_t shmem = (size_t)threads * 2 * sizeof(float);
      layernorm_bw_recompute_kernel<<<grid,block,shmem>>>(
          M->buf.x_res1, dXln2, M->p.ln2_g, 1e-5f,
          dXres1_from_ln2, gln2_g, gln2_b, BT, D);
      CUDA_CHECK("ln2 bw");
    }
    //Merge into residual1
    { 
      int n=BT*D, block=256, grid=(n+block-1)/block; if(grid>65535) grid=65535;
      add_inplace_kernel<<<grid,block>>>(dRes1, dXres1_from_ln2, n); CUDA_CHECK("res1 merge"); 
    }

    //Attention backward (naive; recompute parts)
    float* dAttnOut=nullptr; cudaMalloc(&dAttnOut, (size_t)BT * D * sizeof(float));
    cudaMemcpy(dAttnOut, dRes1, (size_t)BT*D*sizeof(float), cudaMemcpyDeviceToDevice);

    //Recompute C = P @ V into ctx_tmp (using buf.x as scratch) for dWo
    float* Ctmp = M->buf.x; // [BT,D] scratch
    for (int b = 0; b < B; ++b){
        const size_t off_td = (size_t)b * (size_t)T * (size_t)D;
        const size_t off_tt = (size_t)b * (size_t)T * (size_t)T;
        const float* Pb = M->buf.probs + off_tt;
        const float* Vb = M->buf.v + off_td;
        float* Cb = Ctmp + off_td;
        nano2_gemm_f32(0,0, T, D, T, Pb, T, Vb, D, Cb, D, 1.0f, 0.0f);
        CUDA_CHECK("gemm Ctmp = P @ V (recompute per-batch)");
    }
    //dWo = C^T @ dO
    for (int b = 0; b < B; ++b) {
      size_t off_td = (size_t)b * (size_t)T * (size_t)D;
      // Cb: [T,D], dOb: [T,D]
      nano2_gemm_f32(1,0, D, D, T, Ctmp+off_td, D, dAttnOut + off_td, D, gWo, D, 1.0f, 1.0f);
      CUDA_CHECK("gemm gWo += C^T @ dO (per-batch)");
    }
    //dC = dO @ Wo^T  (reuse buf.x_ln1 as scratch)
    float* dC = M->buf.x_ln1;
    nano2_gemm_f32(0,1, BT, D, D, dAttnOut, D, M->p.Wo, D, dC, D, 1.0f, 0.0f);
    CUDA_CHECK("gemm dC = dO @ Wo^T");

    //Allocate dQ,dK,dV
    float* dQ=nullptr; cudaMalloc(&dQ, (size_t)BT * D * sizeof(float)); cudaMemset(dQ, 0, (size_t)BT*D*sizeof(float));
    float* dK=nullptr; cudaMalloc(&dK, (size_t)BT * D * sizeof(float)); cudaMemset(dK, 0, (size_t)BT*D*sizeof(float));
    float* dV=nullptr; cudaMalloc(&dV, (size_t)BT * D * sizeof(float)); cudaMemset(dV, 0, (size_t)BT*D*sizeof(float));

    //dP = dC @ V^T; dV += P^T @ dC; dS = softmax_bw(P, dP); dQ,dK
    for (int b = 0; b < B; ++b){
        const size_t off_td = (size_t)b * (size_t)T * (size_t)D;
        const size_t off_tt = (size_t)b * (size_t)T * (size_t)T;
        const float* Pb = M->buf.probs + off_tt;
        const float* Vb = M->buf.v + off_td;
        const float* dCb = dC + off_td;
        float* dPb = M->buf.scores + off_tt; // reuse scores buf as dP
        float* dSb = M->buf.probs  + off_tt; // reuse probs  buf as dS (overwrites P, OK after this point)

        //dP
        nano2_gemm_f32(0,1, T, T, D, dCb, D, Vb, D, dPb, T, 1.0f, 0.0f);
        CUDA_CHECK("gemm dP = dC @ V^T (per-batch)");
        //dV
        nano2_gemm_f32(1,0, T, D, T, Pb, T, dCb, D, dV + off_td, D, 1.0f, 1.0f);
        CUDA_CHECK("gemm dV += P^T @ dC (per-batch)");
        // dS
        nano2_softmax_backward(Pb, dPb, dSb, T, T);
        //dQ += (scale*dS) @ K ; dK += (scale*dS)^T @ Q
        const float scale = 1.0f / sqrtf((float)D);
        nano2_gemm_f32(0,0, T, D, T, dSb, T, M->buf.k + off_td, D, dQ + off_td, D, scale, 1.0f);
        CUDA_CHECK("gemm dQ += (scale*dS) @ K (per-batch)");
        nano2_gemm_f32(1,0, T, D, T, dSb, T, M->buf.q + off_td, D, dK + off_td, D, scale, 1.0f);
        CUDA_CHECK("gemm dK += (scale*dS)^T @ Q (per-batch)");
    }

    //param grads for Q,K,V projections + dXln1
    float* dXln1=nullptr; cudaMalloc(&dXln1, (size_t)BT * D * sizeof(float)); cudaMemset(dXln1, 0, (size_t)BT*D*sizeof(float));
    for (int b = 0; b < B; ++b) {
      size_t off_td = (size_t)b * (size_t)T * (size_t)D;
      // x_ln1_b: [T,D], dQ_b: [T,D]
      nano2_gemm_f32(1,0, D, D, T,
                   M->buf.x_ln1 + off_td, D,
                   dQ           + off_td, D,
                   gWq, D, 1.0f, 1.0f);
      CUDA_CHECK("gemm gWq += x_ln1^T @ dQ (per-batch)");
    }
    for (int b = 0; b < B; ++b) {
      size_t off_td = (size_t)b * (size_t)T * (size_t)D;
      nano2_gemm_f32(1,0, D, D, T,
                   M->buf.x_ln1 + off_td, D,
                   dK           + off_td, D,
                   gWk, D, 1.0f, 1.0f);
      CUDA_CHECK("gemm gWk += x_ln1^T @ dK (per-batch)");
    }
    for (int b = 0; b < B; ++b) {
      size_t off_td = (size_t)b * (size_t)T * (size_t)D;
      nano2_gemm_f32(1,0, D, D, T,
                   M->buf.x_ln1 + off_td, D,
                   dV           + off_td, D,
                   gWv, D, 1.0f, 1.0f);
      CUDA_CHECK("gemm gWv += x_ln1^T @ dV (per-batch)");
    }
    nano2_gemm_f32(0,1, BT, D, D, dQ, D, M->p.Wq, D, dXln1, D, 1.0f, 1.0f);
    CUDA_CHECK("gemm dXln1 += dQ @ Wq^T");
    nano2_gemm_f32(0,1, BT, D, D, dK, D, M->p.Wk, D, dXln1, D, 1.0f, 1.0f);
    CUDA_CHECK("gemm dXln1 += dK @ Wk^T");
    nano2_gemm_f32(0,1, BT, D, D, dV, D, M->p.Wv, D, dXln1, D, 1.0f, 1.0f);
    CUDA_CHECK("gemm dXln1 += dV @ Wv^T");

    //LN1 backward (recompute stats)
    float* dX0=nullptr; cudaMalloc(&dX0, (size_t)BT * D * sizeof(float));
    cudaMemcpy(dX0, dRes1, (size_t)BT*D*sizeof(float), cudaMemcpyDeviceToDevice); // residual path
    {
      int threads = (D>=256)?256:(D>=128?128:64);
      dim3 block(threads,1,1), grid(BT,1,1);
      size_t shmem = (size_t)threads * 2 * sizeof(float);
      //reuse s-mem kernel; need temporary buffers gln1_g/gln1_b (already carved), use atomics
      layernorm_bw_recompute_kernel<<<grid,block,shmem>>>(
          M->buf.x, dXln1, M->p.ln1_g, 1e-5f,
          M->buf.x_ln1, gln1_g, gln1_b, BT, D); // write dx into buf.x_ln1 temporarily
      CUDA_CHECK("ln1 bw");
      //dX0 += dx_from_ln1
      int n=BT*D; int b=256, gr=(n+b-1)/b; if (gr>65535) gr=65535;
      add_inplace_kernel<<<gr,b>>>(dX0, M->buf.x_ln1, n); CUDA_CHECK("x0 merge");
    }

    //Embedding backward (tied): add contribution into gE
    { dim3 block(256,1,1), grid(BT,1,1);
      embed_scatter_add_kernel<<<grid,block>>>(dX0, d_x, gE, BT, D); CUDA_CHECK("embed scatter"); }

    //Allreduce
    if (world_size > 1){
        float* h = (float*)malloc(M->n_params * sizeof(float));
        float* h_accum = (float*)malloc(M->n_params * sizeof(float));
        cudaMemcpy(h, M->flat_grads, M->n_params * sizeof(float), cudaMemcpyDeviceToHost);
        MPI_Allreduce(h, h_accum, (int)M->n_params, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        for (size_t i=0;i<M->n_params;++i) h_accum[i] /= (float)world_size;
        cudaMemcpy(M->flat_grads, h_accum, M->n_params, cudaMemcpyHostToDevice);
        free(h); free(h_accum);
    }

    //Clip + AdamW
    if ((float)cfg->clip_grad_norm > 0.0f)
        nano2_clip_grad_global_norm(M->flat_grads, M->n_params, (float)cfg->clip_grad_norm);
    nano2_adamw_step(M->flat_params, M->flat_grads, M->opt.m, M->opt.v, M->n_params,
                     (float)cfg->lr, 0.9f, 0.999f, 1e-8f, (float)cfg->weight_decay);

    
    cudaFree(d_x); cudaFree(d_y);
    cudaFree(dlogits);
    cudaFree(dX2); cudaFree(dRes1); cudaFree(dff2); cudaFree(dff1);
    cudaFree(dXln2); cudaFree(dXres1_from_ln2);
    cudaFree(dAttnOut); cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dXln1); cudaFree(dX0);

    return mean_loss;
}

