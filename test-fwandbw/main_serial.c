#include "nano2_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime_api.h>

// config.c
int  config_from_file(const char* path, struct Config* out);
void config_log(const struct Config* c);

// data.c
struct DataSet {
    uint8_t* data;
    size_t n;
    size_t cursor;
    char path[512];
};
int  dataset_load(const char* path, struct DataSet* ds);
void dataset_free(struct DataSet* ds);
void dataset_reset(struct DataSet* ds, size_t pos);
void dataset_next_batch(struct DataSet* ds, int batch_size, int seq_len,
                        uint8_t* x, uint8_t* y);
void dataset_log(const struct DataSet* ds, const char* tag);

// model + train step
struct Model; // opaque
void  model_init(struct Model* M, const struct Config* c);
void  model_log_summary(const struct Model* M, const struct Config* c);
void  model_free(struct Model* M);

// one training step: forward + backward + (optionally) Allreduce + AdamW
float nano2_train_step(struct Model* M,
                       const uint8_t* h_x,
                       const uint8_t* h_y,
                       const struct Config* cfg,
                       int world_size,
                       int rank);

// ------------------ CLI helpers ------------------

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

// parse number of forward+backward iterations: --fwbw-iters=N (default 1)
static int get_fwbw_iters(int argc, char** argv){
    int iters = 1;
    for (int i = 1; i < argc; ++i){
        if (strncmp(argv[i], "--fwbw-iters=", 13) == 0){
            iters = atoi(argv[i] + 13);
        } else if (strcmp(argv[i], "--fwbw-iters") == 0 && i + 1 < argc){
            iters = atoi(argv[i+1]);
            ++i;
        }
    }
    if (iters < 1) iters = 1;
    return iters;
}

// ------------------ main (no MPI) ------------------

int main(int argc, char** argv){
    // In the MPI version these come from MPI_Comm_rank/size.
    const int world      = 1;
    const int rank       = 0;
    const int local_rank = 0;
    const int local_size = 1;

    // CUDA device selection (one GPU; still respect device count)
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    int dev = (dev_count > 0) ? (local_rank % dev_count) : 0;
    cudaSetDevice(dev);

    printf("nano2-fwbw-serial: world=%d local_rank=%d/%d device=%d (no MPI)\n",
           world, local_rank, local_size, dev);

    // parse config + fwbw-iters
    char config_path[512];
    get_config_path(argc, argv, config_path, sizeof(config_path));
    int fwbw_iters = get_fwbw_iters(argc, argv);

    struct Config cfg;
    if (config_from_file(config_path, &cfg) != 0){
        fprintf(stderr, "failed to load config from %s\n", config_path);
        return 1;
    }

    printf("config: %s\n", config_path);
    config_log(&cfg);
    printf("[bench] forward+backward iters: %d\n", fwbw_iters);

    // load datasets
    struct DataSet train_ds, val_ds;
    if (dataset_load(cfg.train_path, &train_ds) != 0){
        fprintf(stderr, "failed to load train dataset '%s'\n", cfg.train_path);
        return 1;
    }
    if (dataset_load(cfg.val_path, &val_ds) != 0){
        fprintf(stderr, "failed to load val dataset '%s'\n", cfg.val_path);
        dataset_free(&train_ds);
        return 1;
    }

    dataset_log(&train_ds, "train");
    dataset_log(&val_ds,   "val");

    // initialize model
    struct Model M;
    model_init(&M, &cfg);
    model_log_summary(&M, &cfg);

    // make one batch from train set (will be overwritten each iter)
    const int B  = cfg.batch_size;
    const int T  = cfg.seq_len;
    const int BT = B * T;

    uint8_t* x = (uint8_t*)malloc((size_t)BT);
    uint8_t* y = (uint8_t*)malloc((size_t)BT);
    if (!x || !y){
        fprintf(stderr, "host batch malloc failed\n");
        free(x); free(y);
        model_free(&M);
        dataset_free(&train_ds);
        dataset_free(&val_ds);
        return 1;
    }

    dataset_next_batch(&train_ds, B, T, x, y);

    // preview first few tokens
    {
        int preview = (T < 16) ? T : 16;
        printf("batch preview x[0,0:%d): ", preview);
        for (int t = 0; t < preview; ++t) printf("%u ", (unsigned)x[t]);
        printf("\n");
        printf("batch preview y[0,0:%d): ", preview);
        for (int t = 0; t < preview; ++t) printf("%u ", (unsigned)y[t]);
        printf("\n");
    }

    // GPU memory snapshot
    {
        size_t mem_free = 0, mem_total = 0;
        cudaMemGetInfo(&mem_free, &mem_total);
        double used_mib  = (double)(mem_total - mem_free) / (1024.0 * 1024.0);
        double total_mib = (double)mem_total / (1024.0 * 1024.0);
        printf("[gpu] memory used: %.2f MiB / %.2f MiB\n", used_mib, total_mib);
    }

    // --------- forward+backward benchmark (no param updates) ---------

    // cfg_step is cfg but with lr=0, weight_decay=0 so AdamW doesn't change params.
    struct Config cfg_step = cfg;
    cfg_step.lr = 0.0;
    cfg_step.weight_decay = 0.0;

    // Make sure GPU is idle before timing
    cudaDeviceSynchronize();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0, 0);

    float loss = 0.0f;
    for (int i = 0; i < fwbw_iters; ++i){
        dataset_next_batch(&train_ds, B, T, x, y);
        loss = nano2_train_step(&M, x, y, &cfg_step, world, rank);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(ev1, 0);
    cudaEventSynchronize(ev1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    // Single process: "global" time is just our local time
    double max_ms = (double)ms;

    double toks       = (double)BT * (double)fwbw_iters * (double)world;
    double toks_per_s = toks / (max_ms * 1e-3);

    printf("train loss (last iter): %.6f\n", loss);
    printf("iters: %d | world size: %d\n", fwbw_iters, world);
    printf("total wall time (serial): %.3f ms\n", max_ms);
    printf("time/iter: %.3f ms\n", max_ms / (double)fwbw_iters);
    printf("tokens/iter: %d (B=%d * T=%d)\n", BT, B, T);
    printf("tokens/sec (serial, fw+bw): %.0f\n", toks_per_s);

    free(x);
    free(y);
    model_free(&M);
    dataset_free(&train_ds);
    dataset_free(&val_ds);

    return 0;
}
