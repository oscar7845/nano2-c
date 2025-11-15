//TODO:remove the cfg warns
// mpi boxes test
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime_api.h>

#include "config.h"
#include "dataset.h"
#include "model.h"

//gpu ops (still external)
extern void nano2_attention_forward(const float*,int,int,int,
                                    const float*,const float*,const float*,const float*,
                                    float*,float*,float*,float*,float*,float*,float*);
extern void nano2_gelu_forward(const float*,float*,int,int);
extern void nano2_softmax_forward(const float*,float*,int,int);
extern float nano2_xent_forward_mean(const float*,const int*,int,int,float*);

//just pick default cfg
static void get_config_path(int argc,char**argv,char*out,size_t cap){
    strncpy(out,"./configs/nano2.json",cap-1);
    out[cap-1]=0;
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);

    //local rank → gpu select
    int local_rank=0;
    {
        MPI_Comm loc;
        MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&loc);
        MPI_Comm_rank(loc,&local_rank);
        MPI_Comm_free(&loc);
    }

    int devc=0; cudaGetDeviceCount(&devc);
    int dev = devc ? (local_rank % devc) : 0;
    cudaSetDevice(dev);

    if(local_rank==0){
        printf("v1 main: local_rank=%d dev=%d\n",local_rank,dev);
    }

    //config load
    char cfg_path[512];
    get_config_path(argc,argv,cfg_path,sizeof(cfg_path));
    struct Config cfg;
    config_from_file(cfg_path,&cfg);

    if(local_rank==0){
        printf("v1 cfg: %s\n",cfg_path);
        config_log(&cfg);
    }

    //datasets
    struct DataSet train_ds,val_ds;
    dataset_load(cfg.train_path,&train_ds);
    dataset_load(cfg.val_path,&val_ds);

    if(local_rank==0){
        dataset_log(&train_ds,"train");
        dataset_log(&val_ds,"val");
    }

    //model
    struct Model M;
    model_init(&M,&cfg);

    //fake inputs
    int B=cfg.batch_size, T=cfg.seq_len, D=cfg.d_model;
    int BT=B*T;
    size_t bytes_x = (size_t)BT * D * sizeof(float);

    float* h_x = (float*)malloc(bytes_x);
    for(int i=0;i<BT*D;i++) h_x[i] = (i%23)*0.01f;
    nano2_copy_host_to_device(M.buf.x,h_x,bytes_x);
    free(h_x);

    //fake LN1: copy x->x_ln1
    cudaMemcpy(M.buf.x_ln1,M.buf.x,bytes_x,cudaMemcpyDeviceToDevice);

    //attention
    nano2_attention_forward(
        M.buf.x_ln1,B,T,D,
        M.p.Wq,M.p.Wk,M.p.Wv,M.p.Wo,
        M.buf.q,M.buf.k,M.buf.v,
        M.buf.scores,M.buf.probs,
        M.buf.attn_out,
        M.buf.x_res1
    );

    //fake FFN1
    cudaMemcpy(M.buf.ff1,M.buf.x_res1,bytes_x,cudaMemcpyDeviceToDevice);
    nano2_gelu_forward(M.buf.ff1,M.buf.ff1,BT*D,1);

    //fake output
    cudaMemcpy(M.buf.logits,M.buf.ff1,bytes_x,cudaMemcpyDeviceToDevice);

    //softmax over V (not correct dims but ok v1)
    nano2_softmax_forward(M.buf.logits,M.buf.logits,BT,cfg.vocab_size);

    //fake tgt
    int* h_tgt = (int*)malloc(BT*sizeof(int));
    for(int i=0;i<BT;i++) h_tgt[i] = i % cfg.vocab_size;

    int* d_tgt; cudaMalloc(&d_tgt,BT*sizeof(int));
    cudaMemcpy(d_tgt,h_tgt,BT*sizeof(int),cudaMemcpyHostToDevice);

    float L = nano2_xent_forward_mean(M.buf.logits,d_tgt,BT,cfg.vocab_size,
                                      M.buf.attn_out);
    if(local_rank==0){
        printf("v1 loss=%f\n",L);
    }

    cudaFree(d_tgt);
    free(h_tgt);
    model_free(&M);
    dataset_free(&train_ds);
    dataset_free(&val_ds);
    MPI_Finalize();
    return 0;
}

