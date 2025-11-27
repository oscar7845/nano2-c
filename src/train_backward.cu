#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include "nano2_model.h"
#include "cuda_check.h"
#include "cuda/embed.h"

extern "C" float nano2_train_step(struct Model* M,
                                  const uint8_t* h_tokens_x,
                                  const uint8_t* h_tokens_y,
                                  const struct Config* cfg,
                                  int world_size,int rank)
{
    int B=M->B, T=M->T, D=M->D, V=M->V;
    int BT=B*T;

    cudaMemset(M->flat_grads,0,M->n_params*sizeof(float));

    uint8_t* d_x=0;
    uint8_t* d_y=0;
    cudaMalloc(&d_x,(size_t)BT);
    cudaMalloc(&d_y,(size_t)BT);
    cudaMemcpy(d_x,h_tokens_x,(size_t)BT,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_tokens_y,(size_t)BT,cudaMemcpyHostToDevice);

    //embed+pos
    nano2_embed_add_pos(d_x,M->p.E,M->pos_sin,M->pos_cos,M->buf.x,B,T,D);
    CUDA_CHECK("embedpos");

    //simple logits = x @ E^T
    nano2_gemm_f32(0,1,BT,V,D,M->buf.x,D,M->p.E,D,M->buf.logits,V,1.0f,0.0f);
    CUDA_CHECK("logits");

    //compute loss on cpu
    float loss=0.0f;

    //simple gradient = zeros
    cudaMemset(M->flat_grads,0,M->n_params*sizeof(float));

    //adamw step 
    nano2_adamw_step(M->flat_params,M->flat_grads,M->opt.m,M->opt.v,
                     M->n_params,(float)cfg->lr,0.9f,0.999f,1e-8f,
                     (float)cfg->weight_decay);

    cudaFree(d_x);
    cudaFree(d_y);

    return loss;
}

