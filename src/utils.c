#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "utils.h"
#include "log.h"

#include <cuda_runtime_api.h>   // CUDA runtime declarations

// Helper: enumerate devices and print basic info
static void print_cuda_info(int rank) {
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    LOGI(rank, "CUDA devices: %d", ndev);
    for (int i = 0; i < ndev; ++i) {
        struct cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        LOGI(rank, "  [%d] %s | cc %d.%d | %.1f GB",
             i, p.name, p.major, p.minor,
             (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
}

// Public API (declared in utils.h)
void nano2_print_env(int mpi_rank) {
    // current device (best-effort)
    int dev = -1;
    cudaGetDevice(&dev);
    if (dev >= 0) {
        struct cudaDeviceProp p;
        cudaGetDeviceProperties(&p, dev);
        LOGI(mpi_rank, "Using device %d: %s (cc %d.%d)",
             dev, p.name, p.major, p.minor);
    }
    // full inventory
    print_cuda_info(mpi_rank);
}

int nano2_ensure_shared_path(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if ((st.st_mode & S_IFMT) == S_IFDIR) return 0;
        LOGE(0, "Path exists but is not a directory: %s", path);
        return -1;
    }
    if (mkdir(path, 0775) != 0 && errno != EEXIST) {
        LOGE(0, "Failed to create '%s': %s", path, strerror(errno));
        return -1;
    }
    return 0;
}

