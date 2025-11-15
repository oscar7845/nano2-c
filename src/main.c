//now i use nano2_forward_loss() instead of hand-wiring GPU ops
//TODO: rm debug prints
//
#include "nano2_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda_runtime_api.h>

#include "config.h"
#include "dataset.h"

static void get_config_path(int ac,char**av,char*out,size_t cap){
    const char* def="./configs/nano2.json";
    strncpy(out,def,cap-1); out[cap-1]=0;
    for(int i=1;i<ac;i++){
        if(strncmp(av[i],"--config=",9)==0){
            strncpy(out,av[i]+9,cap-1); out[cap-1]=0;
        }
    }
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);

    int rank=0, world=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world);

    int local_rank=0;
    {
        MPI_Comm loc;
        MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&loc);
        MPI_Comm_rank(loc,&local_rank);
        MPI_Comm_free(&loc);
    }

    int dc=0; cudaGetDeviceCount(&dc);
    int dev = dc? local_rank % dc : 0;
    cudaSetDevice(dev);

    if(rank==0){
        printf("v3: world=%d local_rank=%d dev=%d\n",world,local_rank,dev);
    }

    //config
    char cfg_path[512];
    get_config_path(argc,argv,cfg_path,sizeof(cfg_path));

    struct Config cfg;
    config_from_file(cfg_path,&cfg);

    if(rank==0){
        printf("config: %s\n",cfg_path);
        config_log(&cfg);
    }

    //dataset
    struct DataSet train_ds,val_ds;
    dataset_load(cfg.train_path,&train_ds);
    dataset_load(cfg.val_path,&val_ds);

    if(rank==0){
        dataset_log(&train_ds,"train");
        dataset_log(&val_ds,"val");
    }

    //model
    struct Model M;
    model_init(&M,&cfg);
    if(rank==0) model_log_summary(&M,&cfg);

    //real batch tokens
    int B=cfg.batch_size, T=cfg.seq_len, BT=B*T;
    uint8_t* x = malloc(BT);
    uint8_t* y = malloc(BT);
    dataset_next_batch(&train_ds,B,T,x,y);

    //preview
    if(rank==0){
        int pv = (T<16? T:16);
        printf("preview x[0]: ");
        for(int i=0;i<pv;i++) printf("%u ",(unsigned)x[i]);
        printf("\n");
    }

    //timed forward
    cudaEvent_t t0,t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    float loss = nano2_forward_loss(&M,x,y);
    cudaDeviceSynchronize();

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms=0;
    cudaEventElapsedTime(&ms,t0,t1);

    if(rank==0){
        double toks = (double)BT;
        printf("v3 mean loss: %.6f\n",loss);
        printf("time: %.3f ms | toks=%d | toks/sec=%.0f\n",
               ms, BT, toks/(ms*1e-3));
    }

    cudaEventDestroy(t0);
   cudaEventDestroy(t1);
    free(x); free(y);
    model_free(&M);
    dataset_free(&train_ds);
    dataset_free(&val_ds);
    MPI_Finalize();
    return 0;
}

