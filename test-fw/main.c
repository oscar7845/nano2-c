//bring-up executable: MPI + CUDA device selection, load config & data,
//run a single forward pass and print mean loss & timing.
//Minimal checks by design (inputs assumed valid).

#include "nano2_model.h"
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

//Model + Forward (model.c, train_forward.cu)
struct Model; // opaque here
void model_init(struct Model* M, const struct Config* c);
void model_log_summary(const struct Model* M, const struct Config* c);
void model_free(struct Model* M);
float nano2_forward_loss(struct Model* M, const uint8_t* h_x, const uint8_t* h_y);

// CLI: --config path
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
    int rank=0, world=1; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int local_rank=0, local_size=1; MPI_Comm_rank(local, &local_rank); MPI_Comm_size(local, &local_size);
    MPI_Comm_free(&local);

    //CUDA device selection (one GPU per local rank)
    int dev_count=0; cudaGetDeviceCount(&dev_count);
    int dev = (dev_count > 0) ? (local_rank % dev_count) : 0;
    cudaSetDevice(dev);

    if (rank==0){ 
	    printf("nano2: world=%d local_rank=%d/%d device=%d\n", world, local_rank, local_size, dev); 
    }

    //parse config
    char config_path[512]; get_config_path(argc, argv, config_path, sizeof(config_path));
    struct Config cfg; config_from_file(config_path, &cfg);
    if (rank==0){ 
	    printf("config: %s\n", config_path); config_log(&cfg); 
    }
    
    //load datasets
    struct DataSet train_ds, val_ds; dataset_load(cfg.train_path, &train_ds); dataset_load(cfg.val_path, &val_ds);
    if (rank==0){ 
	    dataset_log(&train_ds, "train"); dataset_log(&val_ds, "val"); 
    }

    //initialize model
    struct Model M; model_init(&M, &cfg);
    if (rank==0) model_log_summary(&M, &cfg);

    //make one batch from train set
    const int B = cfg.batch_size; const int T = cfg.seq_len; const int BT = B * T;
    uint8_t* x = (uint8_t*)malloc((size_t)BT);
    uint8_t* y = (uint8_t*)malloc((size_t)BT);
    dataset_next_batch(&train_ds, B, T, x, y);

    //preview first few tokens
    if (rank==0){
        int preview = (T<16) ? T : 16;
        printf("batch preview x[0,0:%d): ", preview);
        for (int t = 0; t < preview; ++t) printf("%u ", (unsigned)x[t]);
        printf("\n");
        printf("batch preview y[0,0:%d): ", preview);
        for (int t = 0; t < preview; ++t) printf("%u ", (unsigned)y[t]);
        printf("\n");
    }

    //GPU memory snapshot
    size_t mem_free=0, mem_total=0; cudaMemGetInfo(&mem_free, &mem_total);
    if(rank==0){
        double used_mib = (double)(mem_total - mem_free)/(1024.0*1024.0);
        double total_mib = (double)mem_total/(1024.0*1024.0);
        printf("[gpu] memory used: %.2f MiB / %.2f MiB\n", used_mib, total_mib);
    }

    //run forward once, measure time
    cudaEvent_t ev0, ev1; cudaEventCreate(&ev0); 
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, 0);
    
    float loss = nano2_forward_loss(&M, x, y);
    cudaDeviceSynchronize();

    cudaEventRecord(ev1, 0); 
    cudaEventSynchronize(ev1);
    float ms=0.0f; 
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0); 
    cudaEventDestroy(ev1);

    if (rank==0){
        double toks = (double)BT;
        double toks_per_s = toks / (ms * 1e-3);
        printf("forward mean loss: %.6f (expect ~ ln(256)=5.545)\n", loss);
        printf("step time: %.3f ms | tokens/step: %d | tokens/sec: %.0f\n", ms, BT, toks_per_s);
    }

    free(x); free(y);
    model_free(&M);
    dataset_free(&train_ds); dataset_free(&val_ds);

    MPI_Finalize();
    return 0;
}
