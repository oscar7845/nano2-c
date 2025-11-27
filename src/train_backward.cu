#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "cuda_check.h"
#include "nano2_model.h"
#include "cuda/embed.h"
#include "cuda/layernorm.h"

extern "C" float nano2_train_step(struct Model* M,
                                  const uint8_t* h_tokens_x,
                                  const uint8_t* h_tokens_y,
                                  const struct Config* cfg,
                                  int world_size,int rank)
{
    int B=M->B, T=M->T, D=M->D, V=M->V, F=M->F;
    int BT=B*T;

    cudaMemset(M->flat_grads,0,M->n_params*sizeof(float));

    //copy tokens
    uint8_t* d_x=0;
    uint8_t* d_y=0;
    cudaMalloc(&d_x,(size_t)BT);
    cudaMalloc(&d_y,(size_t)BT);
    cudaMemcpy(d_x,h_tokens_x,(size_t)BT,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_tokens_y,(size_t)BT,cudaMemcpyHostToDevice);

    //embed+pos
    nano2_embed_add_pos(d_x,M->p.E,M->pos_sin,M->pos_cos,M->buf.x,B,T,D);
    CUDA_CHECK("embed");

    //LN1
    nano2_layernorm_forward(M->buf.x,M->buf.x_ln1,M->p.ln1_g,M->p.ln1_b,BT,D,1e-5f);

    //Q,K,V
    nano2_gemm_f32(0,0,BT,D,D,M->buf.x_ln1,D,M->p.Wq,D,M->buf.q,D,1.0f,0.0f);
    nano2_gemm_f32(0,0,BT,D,D,M->buf.x_ln1,D,M->p.Wk,D,M->buf.k,D,1.0f,0.0f);
    nano2_gemm_f32(0,0,BT,D,D,M->buf.x_ln1,D,M->p.Wv,D,M->buf.v,D,1.0f,0.0f);

    //scores = Q K^T (per batch)
    for(int b=0;b<B;++b){
        size_t off_td=(size_t)b*T*D;
        size_t off_tt=(size_t)b*T*T;
        nano2_gemm_f32(0,1,T,T,D,
                       M->buf.q+off_td,D,
                       M->buf.k+off_td,D,
                       M->buf.scores+off_tt,T,
                       1.0f,0.0f);
    }

    //softmax
    nano2_softmax_forward(M->buf.scores,M->buf.probs,T,T);

    //context C=P*V
    for(int b=0;b<B;++b){
        size_t off_td=(size_t)b*T*D;
        size_t off_tt=(size_t)b*T*T;
        nano2_gemm_f32(0,0,T,D,T,
                       M->buf.probs+off_tt,T,
                       M->buf.v+off_td,D,
                       M->buf.x_res1+off_td,D,
                       1.0f,0.0f);
    }

    //O=C*Wo
    nano2_gemm_f32(0,0,BT,D,D,M->buf.x_res1,D,M->p.Wo,D,M->buf.attn_out,D,1.0f,0.0f);

    //residual
    cudaMemcpy(M->buf.x_res1,M->buf.x,(size_t)BT*D*sizeof(float),cudaMemcpyDeviceToDevice);
    {
        int n=BT*D;
        int blk=256;
        int grd=(n+blk-1)/blk;
        add_inplace_kernel<<<grd,blk>>>(M->buf.x_res1,M->buf.attn_out,n);
    }

    //LN2
    nano2_layernorm_forward(M->buf.x_res1,M->buf.x_ln2,M->p.ln2_g,M->p.ln2_b,BT,D,1e-5f);

    //FF1=x_ln2 W1 + b1
    nano2_gemm_f32(0,0,BT,F,D,M->buf.x_ln2,D,M->p.W1,F,M->buf.ff1,F,1.0f,0.0f);
    {
        int thr=(F>=256)?256:(F>=128)?128:64;
        dim3 block(thr,1,1),grid((F+thr-1)/thr,1,1);
        add_bias_inplace_kernel<<<grid,block>>>(M->buf.ff1,M->p.b1,BT,F);
    }
    nano2_gelu_forward(M->buf.ff1,M->buf.ff1,BT*F,1);

    //FF2=ff1 W2 + b2
    nano2_gemm_f32(0,0,BT,D,F,M->buf.ff1,F,M->p.W2,D,M->buf.ff2,D,1.0f,0.0f);
    {
        int thr=(D>=256)?256:(D>=128)?128:64;
        dim3 block(thr,1,1),grid((D+thr-1)/thr,1,1);
        add_bias_inplace_kernel<<<grid,block>>>(M->buf.ff2,M->p.b2,BT,D);
    }

    //residual2
    cudaMemcpy(M->buf.x_res2,M->buf.x_res1,(size_t)BT*D*sizeof(float),cudaMemcpyDeviceToDevice);
    {
        int n=BT*D, blk=256, grd=(n+blk-1)/blk;
        add_inplace_kernel<<<grd,blk>>>(M->buf.x_res2,M->buf.ff2,n);
    }

    //logits
    nano2_gemm_f32(0,1,BT,V,D,M->buf.x_res2,D,M->p.E,D,M->buf.logits,V,1.0f,0.0f);

    //loss is fake for now
    float loss=0.0f;

    //a few placeholder grads
    cudaMemset(M->flat_grads,0,M->n_params*sizeof(float));

    //adamw
    nano2_adamw_step(M->flat_params,M->flat_grads,M->opt.m,M->opt.v,M->n_params,
                     (float)cfg->lr,0.9f,0.999f,1e-8f,(float)cfg->weight_decay);

    cudaFree(d_x);
    cudaFree(d_y);

    return loss;
}

