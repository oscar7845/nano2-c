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
extern void nano2_gelu_forward(const float *x, float *y, int n, int approx);
extern void nano2_attention_forward(const float *x_ln, int B,int T,int D,
                                    const float *Wq,const float *Wk,const float *Wv,const float *Wo,
                                    float *q,float *k,float *v,
                                    float *scores,float *probs,
                                    float *ctx,float *out);
extern float nano2_xent_forward_mean(const float *logits,const int *targets,
                                     int rows,int cols,float *dlogits);
extern void nano2_softmax_forward(const float *x,float *y,int rows,int cols);
static void get_config_path(int argc,char**argv,char*out,size_t cap){
    const char* def="./configs/nano2.json";
    strncpy(out,def,cap-1); out[cap-1]=0;
    for(int i=1;i<argc;i++){
        if(strncmp(argv[i],"--config=",9)==0){
            strncpy(out,argv[i]+9,cap-1); out[cap-1]=0;
        } else if(strcmp(argv[i],"--config")==0 && i+1<argc){
            strncpy(out,argv[i+1],cap-1); out[cap-1]=0;
        }
    }
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);

    //gpu select
    int local_rank=0;
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&local);
        MPI_Comm_rank(local,&local_rank);
        MPI_Comm_free(&local);
    }
    int dev_count=0; cudaGetDeviceCount(&dev_count);
    int dev=(dev_count? local_rank % dev_count:0);
    cudaSetDevice(dev);

    //load config
    char cfg_path[512];
    get_config_path(argc,argv,cfg_path,sizeof(cfg_path));
    struct Config cfg;
    config_from_file(cfg_path,&cfg);

    printf("cfg loaded\n");

    //quick load datasets
    struct DataSet train_ds, val_ds;
    dataset_load(cfg.train_path,&train_ds);
    dataset_load(cfg.val_path,&val_ds);

    //model alloc
    struct Model M;
    model_init(&M,&cfg);

    //GPU ops quick tests
    printf("gpu ops test...\n");

    //gelu test
    //fix
    {
        float h_in[8];
        for(int i=0;i<8;i++) h_in[i]=0.1f*i;
        float *d_in, *d_out;
        cudaMalloc(&d_in, 8*sizeof(float));
        cudaMalloc(&d_out,8*sizeof(float));
        cudaMemcpy(d_in,h_in,8*sizeof(float),cudaMemcpyHostToDevice);

        nano2_gelu_forward(d_in,d_out,8,0);
        cudaFree(d_in); cudaFree(d_out);
        printf(" gelu ok\n");
    }

    //softmax test
    {
        float h_x[6]={1,2,3,4,5,6};
        float *dx,*dy;
        cudaMalloc(&dx,6*sizeof(float));
        cudaMalloc(&dy,6*sizeof(float));
        cudaMemcpy(dx,h_x,6*sizeof(float),cudaMemcpyHostToDevice);
        nano2_softmax_forward(dx,dy,2,3);
        cudaFree(dx); cudaFree(dy);
        printf(" softmax ok\n");
    }

    //xent test
    {
        float h_logits[6]={1,2,3,4,5,6};
        int   h_tgt[2]={2,1};
        float *d_logits,*d_dlog;
        int  *d_tgt;
        cudaMalloc(&d_logits,6*sizeof(float));
        cudaMalloc(&d_dlog,6*sizeof(float));
        cudaMalloc(&d_tgt,2*sizeof(int));
        cudaMemcpy(d_logits,h_logits,6*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(d_tgt,h_tgt,2*sizeof(int),cudaMemcpyHostToDevice);

        float L = nano2_xent_forward_mean(d_logits,d_tgt,2,3,d_dlog);
        printf(" xent ok: L=%f\n",L);

        cudaFree(d_logits); cudaFree(d_dlog); cudaFree(d_tgt);
    }

    printf("attention test skip in v1 (needs model buffers)\n");

    model_free(&M);
    dataset_free(&train_ds);
    dataset_free(&val_ds);

    MPI_Finalize();
    return 0;
}

