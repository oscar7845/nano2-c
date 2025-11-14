//actually try the gpu ops with real model buffers from M
//TODO: real forward
//TODO: rm debug prints
//fix xent warnings

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include "config.h"
#include "dataset.h"
#include "model.h"
//gpu ops
extern void nano2_gelu_forward(const float *x,float *y,int n,int approx);
extern void nano2_attention_forward(const float *x_ln,int B,int T,int D,
                                    const float *Wq,const float *Wk,const float *Wv,const float *Wo,
                                    float *q,float *k,float *v,
                                    float *scores,float *probs,
                                    float *ctx,float *out);
extern void nano2_softmax_forward(const float* x,float* y,int rows,int cols);
extern float nano2_xent_forward_mean(const float* logits,const int* targets,
                                     int rows,int cols,float* dlogits);
static void get_config_path(int argc,char**argv,char*out,size_t cap){
    const char* def="./configs/nano2.json";
    strncpy(out,def,cap-1); out[cap-1]=0;
    for(int i=1;i<argc;i++){
        if(strncmp(argv[i],"--config=",9)==0){
            strncpy(out,argv[i]+9,cap-1); out[cap-1]=0;
        }
    }
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
    cudaSetDevice(dc? local_rank % dc:0);

    //load cfg
    char cfg_path[512];
    get_config_path(argc,argv,cfg_path,sizeof(cfg_path));
    struct Config cfg;
    config_from_file(cfg_path,&cfg);

    //load ds
    struct DataSet train_ds,val_ds;
    dataset_load(cfg.train_path,&train_ds);
    dataset_load(cfg.val_path,&val_ds);

    //model
    struct Model M;
    model_init(&M,&cfg);

    int B=cfg.batch_size;
    int T=cfg.seq_len;
    int D=cfg.d_model;
    int BT = B*T;

    //fake x (device)
    size_t nbytes = (size_t)BT * D * sizeof(float);
    float *h_x = (float*)malloc(nbytes);
    for(size_t i=0;i<BT*(size_t)D;i++) h_x[i] = (float)(i%13)*0.05f;
    nano2_copy_host_to_device(M.buf.x, h_x, nbytes);
    free(h_x);

    printf("v2: running attention...\n");
    nano2_attention_forward(
        M.buf.x_ln2, // just reusing;
	//say they normally LN first??
	//fix layer
	//rm warning vocab 
        B,T,D,
        M.p.Wq, M.p.Wk, M.p.Wv, M.p.Wo,
        M.buf.q, M.buf.k, M.buf.v,
        M.buf.scores, M.buf.probs,
        M.buf.attn_out,
        M.buf.x_res1
    );

    printf("v2: running gelu on ff1...\n");
    //fake treat ff1 as linear output
    int FFN = cfg.ffn_mult * D;
    int n_ff1 = BT * FFN;
    nano2_gelu_forward(M.buf.ff1, M.buf.ff1, n_ff1, 0);

    printf("v2: softmax on logits...\n");
    nano2_softmax_forward(M.buf.logits, M.buf.logits, BT, cfg.vocab_size);

    printf("v2: xent loss test...\n");
    int *h_tgt = (int*)malloc(BT*sizeof(int));
    for(int i=0;i<BT;i++) h_tgt[i] = i % cfg.vocab_size;

    int *d_tgt; cudaMalloc(&d_tgt, BT*sizeof(int));
    cudaMemcpy(d_tgt,h_tgt,BT*sizeof(int),cudaMemcpyHostToDevice);
    free(h_tgt);

    float L = nano2_xent_forward_mean(M.buf.logits,d_tgt, BT, cfg.vocab_size,
                                      M.buf.attn_out/*reusing as dlogits*/);
    printf("v2 loss = %f\n",L);

    cudaFree(d_tgt);

    model_free(&M);
    dataset_free(&train_ds);
    dataset_free(&val_ds);

    MPI_Finalize();
    return 0;
}

