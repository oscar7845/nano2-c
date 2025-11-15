//TODO:
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

//gpu ops
extern void nano2_attention_forward(...);
extern void nano2_gelu_forward(const float*,float*,int,int);
extern void nano2_softmax_forward(const float*,float*,int,int);
extern float nano2_xent_forward_mean(const float*,const int*,int,int,float*);

static void get_config_path(int a,char**v,char*out,size_t n){
    strncpy(out,"./configs/nano2.json",n-1); out[n-1]=0;
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);

    int local_rank=0;
    {
        MPI_Comm loc;
        MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&loc);
        MPI_Comm_rank(loc,&local_rank);
        MPI_Comm_free(&loc);
    }
    int devc=0; cudaGetDeviceCount(&devc);
    cudaSetDevice(devc? local_rank % devc : 0);

    //config
    struct Config cfg;
    char cfg_path[512];
    get_config_path(argc,argv,cfg_path,sizeof(cfg_path));
    config_from_file(cfg_path,&cfg);

    //data
    struct DataSet train_ds,val_ds;
    dataset_load(cfg.train_path,&train_ds);
    dataset_load(cfg.val_path,&val_ds);

    //model
    struct Model M;
    model_init(&M,&cfg);

    int B=cfg.batch_size, T=cfg.seq_len, D=cfg.d_model;
    int BT=B*T;

    //make real batch tokens (x,y)
    uint8_t* h_x_tok = malloc(BT);
    uint8_t* h_y_tok = malloc(BT);
    dataset_next_batch(&train_ds,B,T,h_x_tok,h_y_tok);

    //(student) fake embed: shove token values into x buffer
    float* h_x = malloc((size_t)BT*D*sizeof(float));
    for(int i=0;i<BT;i++){
        for(int j=0;j<D;j++){
            h_x[i*D+j] = (float)(h_x_tok[i])*0.01f; // fake embed by scaling
        }
    }
    nano2_copy_host_to_device(M.buf.x,h_x,(size_t)BT*D*sizeof(float));
    free(h_x);

    //pass x->x_ln1
    cudaMemcpy(M.buf.x_ln1,M.buf.x,(size_t)BT*D*sizeof(float),cudaMemcpyDeviceToDevice);

    //attn
    nano2_attention_forward(
        M.buf.x_ln1,B,T,D,
        M.p.Wq,M.p.Wk,M.p.Wv,M.p.Wo,
        M.buf.q,M.buf.k,M.buf.v,
        M.buf.scores,M.buf.probs,
        M.buf.attn_out,
        M.buf.x_res1
    );

    //FF fake
    cudaMemcpy(M.buf.ff1,M.buf.x_res1,(size_t)BT*D*sizeof(float),cudaMemcpyDeviceToDevice);
    nano2_gelu_forward(M.buf.ff1,M.buf.ff1,BT*D,1);
    cudaMemcpy(M.buf.ff2,M.buf.ff1,(size_t)BT*D*sizeof(float),cudaMemcpyDeviceToDevice);

    //out->logits
    cudaMemcpy(M.buf.logits,M.buf.ff2,(size_t)BT*D*sizeof(float),cudaMemcpyDeviceToDevice);

    //softmax on vocab (still mismatched dims → student cheat)
    nano2_softmax_forward(M.buf.logits,M.buf.logits,BT,cfg.vocab_size);

    //targets
    int* h_tgt = malloc(BT*sizeof(int));
    for(int i=0;i<BT;i++) h_tgt[i]= h_y_tok[i] % cfg.vocab_size;
    int* d_tgt; cudaMalloc(&d_tgt,BT*sizeof(int));
    cudaMemcpy(d_tgt,h_tgt,BT*sizeof(int),cudaMemcpyHostToDevice);

    float L = nano2_xent_forward_mean(M.buf.logits,d_tgt,BT,cfg.vocab_size,
                                      M.buf.attn_out);
    printf("v2 loss = %f\n",L);

    cudaFree(d_tgt);
    free(h_tgt);
    free(h_x_tok); free(h_y_tok);
    model_free(&M);
    dataset_free(&train_ds); dataset_free(&val_ds);
    MPI_Finalize();
    return 0;
}
