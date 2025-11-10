#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <mpi.h>
#include <cuda_runtime_api.h>

//logging
static inline void log_ranked(int rank, const char* level, const char* fmt, ...){
    va_list args; va_start(args, fmt);
    fprintf(stderr, "[%s][rank=%d] ", level, rank);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}
#define LOGI(rank, ...) do { log_ranked((rank), "INFO", __VA_ARGS__); } while (0)
#define LOGW(rank, ...) do { log_ranked((rank), "WARN", __VA_ARGS__); } while (0)
#define LOGE(rank, ...) do { log_ranked((rank), "ERR",  __VA_ARGS__); } while (0)

//helpers
static void print_cuda_inventory(int rank){
    int ndev = 0; cudaGetDeviceCount(&ndev);
    LOGI(rank, "CUDA devices: %d", ndev);
    for (int i = 0; i < ndev; ++i) {
        struct cudaDeviceProp p; cudaGetDeviceProperties(&p, i);
        LOGI(rank, "  [%d] %s | cc %d.%d | %.1f GB",
             i, p.name, p.major, p.minor,
             (double)p.totalGlobalMem / (1024.0*1024.0*1024.0));
    }
}

static void print_cuda_env(int rank){
    int dev = -1; cudaGetDevice(&dev);
    if (dev >= 0) {
        struct cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
        LOGI(rank, "Using device %d: %s (cc %d.%d)", dev, p.name, p.major, p.minor);
    }
    print_cuda_inventory(rank);
}

static int ensure_shared_path(const char* path){
    struct stat st;
    if (stat(path, &st) == 0){
        if ((st.st_mode & S_IFMT) == S_IFDIR) return 0;
        LOGE(0, "Path exists but is not a directory: %s", path);
        return -1;
    }
    if (mkdir(path, 0775) != 0 && errno != EEXIST){
        LOGE(0, "Failed to create '%s': %s", path, strerror(errno));
        return -1;
    }
    return 0;
}

//config (config.c)
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

int config_from_file(const char* path, struct Config* out);
void config_log(const struct Config* c);

//CUDA test (src/cuda/dummy_kernels.cu)
void nano2_cuda_selftest(void);

//CLI arg parsing for --config
static void parse_args_config_path(int argc, char** argv, char* out_path, size_t cap){
    //default
    const char* def = "./configs/nano2.json";
    size_t n = strlen(def);
    if (n >= cap) n = cap - 1;
    memcpy(out_path, def, n);
    out_path[n] = '\0';

    for (int i = 1; i < argc; ++i){
        const char* a = argv[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0){
            fprintf(stderr, "Usage: %s [--config PATH]\n", argv[0]);
            //keep default path; just show help
        } else if (strcmp(a, "--config") == 0 && i+1 < argc){
            strncpy(out_path, argv[i+1], cap - 1);
            out_path[cap-1] = '\0';
            ++i;
        } else if (strncmp(a, "--config=", 9) == 0){
            strncpy(out_path, a + 9, cap - 1);
            out_path[cap-1] = '\0';
        }
    }
}

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank=0, world=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    //local rank within a node
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_size(local, &local_size);
    MPI_Comm_free(&local);

    //choose GPU by local rank
    int dev_count=0; cudaGetDeviceCount(&dev_count);
    if (dev_count <= 0){
        if (rank == 0) fprintf(stderr, "No CUDA device visible. Exiting.\n");
        MPI_Finalize(); return 1;
    }
    int dev=local_rank % dev_count;
    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess){
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
        MPI_Finalize(); return 1;
    }

    if (rank == 0) (void)ensure_shared_path("/var/tmp");
    MPI_Barrier(MPI_COMM_WORLD);

    LOGI(rank, "nano2 bootstrap: world=%d, local_rank=%d/%d, my_device=%d", world, local_rank, local_size, dev);
    print_cuda_env(rank);

    //load config (same file on all ranks; only rank 0 prints it)
    char config_path[512];
    parse_args_config_path(argc, argv, config_path, sizeof(config_path));

    struct Config cfg;
    if (config_from_file(config_path, &cfg) != 0){
        if (rank == 0) LOGE(rank, "Failed to load config: %s", config_path);
        MPI_Finalize(); return 1;
    }
    if (rank == 0){
        LOGI(rank, "Loaded config: %s", config_path);
        config_log(&cfg);
    }

    //CUDA test 
    nano2_cuda_selftest();
    LOGI(rank, "CUDA self-test OK.");

    if (rank == 0){
        LOGI(rank, "scaffold up");
    }

    MPI_Finalize();
    return 0;
}

