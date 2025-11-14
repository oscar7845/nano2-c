//forward chain using gpu ops
//TODO: rm warning
//TODO: piece ptrs at left
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime_api.h>

#include "config.h"
#include "dataset.h"
#include "model.h"

//gpu pieces
extern void nano2_attention_forward(const float* x_ln,int B,int T,int D,
                                    const float* Wq,const float* Wk,const float* Wv,const float* Wo,
                                    float* q,float* k,float* v,
                                    float* scores,float* probs,
                                    float* ctx,float* out);

extern void nano2_gelu_forward(const float* x,float* y,int n,int approx);
extern void nano2_softmax_forward(const float* x,float* y,int rows,int cols);
extern float nano2_xent_forward_mean(const float* logits,const int* tgt,
                                     int rows,int cols,float* dlogits);


static void get_config_path(int argc,char**argv,char*out,size_t cap){
    strncpy(out,"./configs/nano2.json",cap);
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);

    int local_rank=0;
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&local);
        MPI_Comm_rank(local,&local_rank);
        MPI_Comm_free(&local);
    }
    int dc=0; cudaGetDeviceCount(&dc);
    cudaSetDevice(dc? local_rank % dc : 0);

    struct Config cfg;
    config_from_file("./configs/nano2.json",&cfg);

    struct DataSet train_ds,val_ds;
    dataset_load(cfg.train_path,&train_ds);
    dataset_load(cfg.val_path,&val_ds);

    struct Model M;
    model_init(&M,&cfg);

    int B=cfg.batch_size, T=cfg.seq_len, D=cfg.d_model;
    int BT=B*T, V=cfg.vocab_size, F=cfg.ffn_mult*D;

    //fake tokens -> fake embeddings (just copy something into x)
    float *h_x = (float*)malloc((size_t)BT*D*sizeof(float));
    for(int i=0;i<BT*D;i++) h_x[i] = (i%37)*0.01f;
    nano2_copy_host_to_device(M.buf.x,h_x,(size_t)BT*D*sizeof(float));
    free(h_x);

    printf("v3: LN (fake): copying x -> x_ln1\n");
    cudaMemcpy(M.buf.x_ln1, M.buf.x, (size_t)BT*D*sizeof(float), cudaMemcpyDeviceToDevice);

    printf("v3: attention...\n");
    nano2_attention_forward(
        M.buf.x_ln1, B,T,D,
        M.p.Wq, M.p.Wk, M.p.Wv, M.p.Wo,
        M.buf.q, M.buf.k, M.buf.v,
        M.buf.scores, M.buf.probs,
        M.buf.attn_out,
        M.buf.x_res1
    );

    printf("v3: FFN: x_res1 -> ff1\n");
    //fake linear: just copy
    cudaMemcpy(M.buf.ff1, M.buf.x_res1, (size_t)BT*D*sizeof(float), cudaMemcpyDeviceToDevice);

    printf(" v3: gelu...\n");
    nano2_gelu_forward(M.buf.ff1, M.buf.ff1, BT*F, 1);

    //ff2 fake: copy back to x_res2
    cudaMemcpy(M.buf.x_res2, M.buf.ff1, (size_t)BT*D*sizeof(float), cudaMemcpyDeviceToDevice);

    printf("v3: logits (reuse x_res2 as logits)\n");
    cudaMemcpy(M.buf.logits, M.buf.x_res2, (size_t)BT*D*sizeof(float), cudaMemcpyDeviceToDevice);

    printf("v3: softmax...\n");
    nano2_softmax_forward(M.buf.logits, M.buf.logits, BT, V);

    printf("v3: loss...\n");
    int *h_tgt = (int*)malloc(BT*sizeof(int));
    for(int i=0;i<BT;i++) h_tgt[i] = i % V;

    int *d_tgt; cudaMalloc(&d_tgt, BT*sizeof(int));
    cudaMemcpy(d_tgt,h_tgt,BT*sizeof(int),cudaMemcpyHostToDevice);

    float L = nano2_xent_forward_mean(M.buf.logits, d_tgt, BT, V,
                                      M.buf.attn_out /*reuse for dlogits*/);
    printf("loss = %f\n",L);

    cudaFree(d_tgt);
    free(h_tgt);

    model_free(&M);
    dataset_free(&train_ds);
    dataset_free(&val_ds);

    MPI_Finalize();
    return 0;
}

