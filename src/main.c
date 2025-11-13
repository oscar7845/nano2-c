//load cfg, data,model forward test

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <mpi.h>
#include <cuda_runtime_api.h>

//(config.c)
struct Config{
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

//(data.c)
struct DataSet{
    uint8_t *data;
    size_t n;
    size_t cursor;
    char path[512];
};

int dataset_load(const char *path, struct DataSet *ds);
void dataset_free(struct DataSet *ds);
void dataset_reset(struct DataSet *ds, size_t pos);
void dataset_next_batch(struct DataSet *ds, int B, int T, uint8_t *x, uint8_t *y);
void dataset_log(const struct DataSet *ds,const char *tag);

//tensor + model
struct Tensor{
    int rows;
    int cols;
    float *data;
};

struct Tensor* tensor_create(int r,int c);
void tensor_fill(struct Tensor *t,float v);
void tensor_fill_random(struct Tensor *t);
void tensor_show(struct Tensor *t);
void tensor_matmul(struct Tensor *A, struct Tensor *B, struct Tensor *C);
void tensor_free(struct Tensor *t);

struct Model{
    struct Tensor *W1;
    struct Tensor *b1;
    struct Tensor *W2;
    struct Tensor *b2;
    int d_model;
    int hidden;
};

struct Model* model_new(int d_model);
void model_forward(struct Model *m,
                   struct Tensor *x_in,
                   struct Tensor *x_tmp1,
                   struct Tensor *x_out);
void model_free(struct Model *m);

//dummy kernel test
void nano2_cuda_selftest(void);


// get config flag
static void get_config_path(int argc,char **argv,char *out,size_t cap){
    const char *def="./configs/nano2.json";
    strncpy(out,def,cap); out[cap-1]='\0';
    for(int i=1;i<argc;i++){
        const char *a=argv[i];
        if(strncmp(a,"--config=",9)==0){
            strncpy(out,a+9,cap); out[cap-1]='\0';
        } else if(strcmp(a,"--config")==0 && i+1<argc){
            strncpy(out,argv[i+1],cap); out[cap-1]='\0';
            i++;
        }
    }
}


int main(int argc,char **argv){
    MPI_Init(&argc,&argv);

    int rank=0,world=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&world);

    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);

    int local_rank=0,local_size=1;
    MPI_Comm_rank(local,&local_rank);
    MPI_Comm_size(local,&local_size);
    MPI_Comm_free(&local);

    int dev_count=0;
    cudaGetDeviceCount(&dev_count);
    int dev=(dev_count>0)? (local_rank % dev_count) : 0;
    cudaSetDevice(dev);

    if(rank==0){
        printf("nano2 main: world=%d local_rank=%d/%d dev=%d\n",
            world,local_rank,local_size,dev);
    }

    //load config
    char cfg_path[512];
    get_config_path(argc,argv,cfg_path,sizeof(cfg_path));

    struct Config cfg;
    config_from_file(cfg_path,&cfg);
    if(rank==0){
        printf("config: %s\n", cfg_path);
        config_log(&cfg);
    }

    //load datasets
    struct DataSet train_ds, val_ds;
    dataset_load(cfg.train_path,&train_ds);
    dataset_load(cfg.val_path,&val_ds);

    if(rank==0){
        dataset_log(&train_ds,"train");
        dataset_log(&val_ds,"val");
    }

    int B = cfg.batch_size;
    int T = cfg.seq_len;

    uint8_t *x = malloc(B*T);
    uint8_t *y = malloc(B*T);

    dataset_next_batch(&train_ds,B,T,x,y);

    if(rank==0){
        int prev=(T<16)?T:16;
        printf("batch preview x[0]: ");
        for(int i=0;i<prev;i++) printf("%u ",(unsigned)x[i]);
        printf("\n");
        printf("batch preview y[0]: ");
        for(int i=0;i<prev;i++) printf("%u ",(unsigned)y[i]);
        printf("\n");
    }

    free(x);
    free(y);

    //gpu check
    nano2_cuda_selftest();
    if(rank==0) printf("cuda test OK\n");

    //==============================
    //     MODEL TESTING
    //==============================

    if(rank==0){
        printf("\n=== model test ===\n");

        int dm = cfg.d_model;
        int batch_test = 2;

        struct Model *m = model_new(dm);

        struct Tensor *x_in   = tensor_create(batch_test, dm);
        struct Tensor *x_tmp1 = tensor_create(batch_test, dm * 4);
        struct Tensor *x_out  = tensor_create(batch_test, dm);

        //just fill with tiny random-ish
        tensor_fill_random(x_in);

        printf("input preview:\n");
        tensor_show(x_in);

        //forward
        model_forward(m, x_in, x_tmp1, x_out);

        printf("output preview:\n");
        tensor_show(x_out);

        //cleanup
        tensor_free(x_in);
        tensor_free(x_tmp1);
        tensor_free(x_out);
        model_free(m);
    }

    dataset_free(&train_ds);
    dataset_free(&val_ds);

    MPI_Finalize();
    return 0;
}

