//load config and data
//build model, log summary
//dummy input test
//TODO: real forward
//TODO

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <mpi.h>
#include <cuda_runtime_api.h>
struct Config {
    char train_path[512];
    char val_path[512];
    int seq_len;
    int batch_size;
    int vocab_size;
    int d_model;
    int ffn_mult;
    double lr;
    double weight_decay;
    double clip_grad_norm;
    int seed;
    int top_k;
};
int  config_from_file(const char *path, struct Config *out);
void config_log(const struct Config *c);
struct DataSet {
    uint8_t* data;
    size_t n;
    size_t cursor;
    char path[512];
};
int dataset_load(const char* path, struct DataSet* ds);
void dataset_free(struct DataSet* ds);
void dataset_reset(struct DataSet* ds, size_t pos);
void dataset_next_batch(struct DataSet* ds, int B, int T, uint8_t* x, uint8_t* y);
void dataset_log(const struct DataSet* ds, const char* tag);
void* nano2_malloc_host(size_t bytes);
void  nano2_free_host(void* p);
void* nano2_malloc_pinned(size_t bytes);
void  nano2_free_pinned(void* p);
void* nano2_malloc_device(size_t bytes);
void  nano2_free_device(void* p);
void  nano2_copy_host_to_device(void* dst_dev, const void* src_host, size_t bytes);
void  nano2_copy_device_to_host(void* dst_host, const void* src_dev, size_t bytes);
void  nano2_memset_device(void* dst_dev, int value, size_t bytes);


//model.c 
struct Model;
void model_init(struct Model *M, const struct Config *c);
void model_free(struct Model *M);
void model_log_summary(const struct Model *M, const struct Config *c);


//arg parser for config path
//TODO:

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
    int rank=0,world=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world);

    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&local);
    int local_rank=0, local_size=1;
    MPI_Comm_rank(local,&local_rank);
    MPI_Comm_size(local,&local_size);
    MPI_Comm_free(&local);


    //CUDA device select
    int dev_count=0; cudaGetDeviceCount(&dev_count);
    int dev=(dev_count>0)? (local_rank % dev_count):0;
    cudaSetDevice(dev);

    if(rank==0){
        printf("nano2 start: world=%d  local=%d/%d  gpu=%d\n",
            world, local_rank, local_size, dev);
    }


    //load config
    char cfg_path[512];
    get_config_path(argc,argv,cfg_path,sizeof(cfg_path));

    struct Config cfg;
    config_from_file(cfg_path,&cfg);

    if(rank==0){
        printf("config: %s\n",cfg_path);
        config_log(&cfg);
    }


    //load datasets
    struct DataSet train_ds, val_ds;
    dataset_load(cfg.train_path, &train_ds);
    dataset_load(cfg.val_path,  &val_ds);

    if(rank==0){
        dataset_log(&train_ds,"train");
        dataset_log(&val_ds,"val");
    }


    //CPU batch preview test
    const int B = cfg.batch_size;
    const int T = cfg.seq_len;

    uint8_t *bx = (uint8_t*)malloc((size_t)B*T);
    uint8_t *by = (uint8_t*)malloc((size_t)B*T);
    dataset_next_batch(&train_ds,B,T,bx,by);

    if(rank==0){
        int pv = (T<16? T:16);
        printf("batch x[0,0:%d): ",pv);
        for(int i=0;i<pv;i++) printf("%u ", (unsigned)bx[i]);
        printf("\n");
    }

    free(bx); free(by);


    //model creation
    if(rank==0) printf("\nmodel init...\n");

    struct Model M;
    model_init(&M,&cfg);

    if(rank==0){
        model_log_summary(&M,&cfg);
    }


    //fake input test (host->device copy only)
    if(rank==0){
        printf("\ntest input alloc (no forward yet)\n");
    }

    size_t BTD_bytes = (size_t)cfg.batch_size * cfg.seq_len * cfg.d_model * sizeof(float);

    float *h_x = (float*)nano2_malloc_host(BTD_bytes);
    for(size_t i=0;i<(size_t)cfg.batch_size*cfg.seq_len*cfg.d_model;i++){
        h_x[i] = (float)(i%97) * 0.01f; // some garbage vals
    }
    // copy to model buffer M.buf.x
    nano2_copy_host_to_device(M.buf.x, h_x, BTD_bytes);
    nano2_free_host(h_x);

    if(rank==0){
        printf("initial device input copied\n");
        //printf("forward TODO\n");
    }


    model_free(&M);

    dataset_free(&train_ds);
    dataset_free(&val_ds);

    MPI_Finalize();
    return 0;
}
