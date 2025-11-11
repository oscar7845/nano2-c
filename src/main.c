//MPI init and pick GPU
//load config + data, and make one batch
//CUDA test

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
int config_from_file(const char* path, struct Config* out);
void config_log(const struct Config* c);

//(data.c)
struct DataSet {
    uint8_t* data;
    size_t n;
    size_t cursor;
    char path[512];
};
int dataset_load(const char* path, struct DataSet* ds);
void dataset_free(struct DataSet* ds);
void dataset_reset(struct DataSet* ds, size_t pos);
void dataset_next_batch(struct DataSet* ds, int batch_size, int seq_len, uint8_t* x, uint8_t* y);
void dataset_log(const struct DataSet* ds, const char* tag);

//(dummy_kernels.cu)
void nano2_cuda_selftest(void);

//arg parsing for --config
static void get_config_path(int argc, char** argv, char* out, size_t cap){
    const char* def = "./configs/nano2.json";
    size_t n = strlen(def); if (n >= cap) n = cap - 1;
    memcpy(out, def, n); out[n] = '\0';
    for (int i = 1; i < argc; ++i){
        const char* a = argv[i];
        if (strncmp(a, "--config=", 9) == 0){
            strncpy(out, a + 9, cap - 1); out[cap-1] = '\0';
        } else if (strcmp(a, "--config") == 0 && i + 1 < argc){
            strncpy(out, argv[i+1], cap - 1); out[cap-1] = '\0'; ++i;
        }
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    int rank=0, world=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int local_rank=0, local_size=1;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_size(local, &local_size);
    MPI_Comm_free(&local);

    int dev_count=0; cudaGetDeviceCount(&dev_count);
    int dev = (dev_count > 0) ? (local_rank % dev_count) : 0;
    cudaSetDevice(dev);

    if (rank==0){
        printf("nano2: world=%d local_rank=%d/%d device=%d\n", world, local_rank, local_size, dev);
    }

    char config_path[512];
    get_config_path(argc, argv, config_path, sizeof(config_path));

    struct Config cfg;
    config_from_file(config_path, &cfg);
    if (rank==0){
        printf("config: %s\n", config_path);
        config_log(&cfg);
    }

    struct DataSet train_ds, val_ds;
    dataset_load(cfg.train_path, &train_ds);
    dataset_load(cfg.val_path, &val_ds);
    if (rank==0){
        dataset_log(&train_ds, "train");
        dataset_log(&val_ds, "val");
    }

    const int B=cfg.batch_size;
    const int T=cfg.seq_len;
    uint8_t* x = (uint8_t*)malloc((size_t)B * (size_t)T);
    uint8_t* y = (uint8_t*)malloc((size_t)B * (size_t)T);
    dataset_next_batch(&train_ds, B, T, x, y);

    if (rank==0){
        int preview = (T<16) ? T : 16;
        printf("batch preview x[0,0:%d): ", preview);
        for (int t = 0; t < preview; ++t) 
		printf("%u ", (unsigned)x[t]);
                //printf("test %d", t);
	printf("\n");
        printf("batch preview y[0,0:%d): ", preview);
        for (int t = 0; t < preview; ++t) printf("%u ", (unsigned)y[t]);
        printf("\n");
    }

    free(x); free(y);

    nano2_cuda_selftest();
    if (rank==0) printf("cuda test: OK\n");

    //test model
    //2-layer MLP w/ ReLU
    if(rank==0){
     printf("\nmodel test:\n");
     //printf("check later")
     extern struct Model* model_new(int d_model);
     extern void model_forward(struct Model *m, struct Tensor *x_in, struct Tensor *x_tmp1,struct Tensor *x_out);
     extern void model_free(struct Model *m);
     extern struct Tensor* tensor_create(int r, int c);
     extern void tensor_fill(struct Tensor *t, float v);
     extern void tensor_show(struct Tensor *t);
     extern void tensor_free(struct Tensor *t);

     int d= cfg.d_model;//match config
     struct Model *m =model_new(d);

     //batch= 1for now
     struct Tensor *x_in=tensor_create(1, d);
     struct Tensor *x_tmp1=tensor_create(1, d * 4); // hidden size
     struct Tensor *x_out =tensor_create(1, d);

     tensor_fill(x_in,1.0f);//test input = ones
     model_forward(m, x_in,x_tmp1, x_out);
     tensor_show(x_out);//preview
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

