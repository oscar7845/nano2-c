#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <cuda_runtime_api.h>

#include "log.h"
#include "utils.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    // Figure out local rank (per node)
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int local_rank = 0, local_size = 1;
    MPI_Comm_rank(local, &local_rank);
    MPI_Comm_size(local, &local_size);
    MPI_Comm_free(&local);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA device visible. Exiting.\n");
        MPI_Finalize();
        return 1;
    }
    int dev = local_rank % dev_count;
    cudaError_t err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", dev, cudaGetErrorString(err));
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        nano2_ensure_shared_path("/var/tmp"); // shared for your setup
    }
    MPI_Barrier(MPI_COMM_WORLD);

    LOGI(rank, "nano2 bootstrap: world=%d, local_rank=%d/%d, my_device=%d", world, local_rank, local_size, dev);
    nano2_print_env(rank);

    // CUDA self-test
    nano2_cuda_selftest();
    LOGI(rank, "CUDA self-test OK.");

    if (rank == 0) {
        LOGI(rank, "Scaffold up. Next steps: data loader, embeddings, LN, attention, FFN, AdamW.");
    }

    MPI_Finalize();
    return 0;
}

