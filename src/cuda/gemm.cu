// naive GEMM GPU version
// TODO: shared mem tiling
// TODO: stride/batch
// currently assumes row-major

#include <stdio.h>
#include <cuda_runtime.h>
//????

extern "C" void cuda_gemm(const float* A, const float* B, float* C, int M, int N, int K);

__global__ void kernel_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float acc = 0.0f;
        for(int k = 0; k < K; k++){
            acc += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = acc;
    }
}

extern "C" void cuda_gemm(const float* A, const float* B, float* C,
                          int M, int N, int K)
{
    dim3 threads(16, 16);
    dim3 blocks((N+15)/16, (M+15)/16);

    kernel_gemm<<<blocks, threads>>>(A, B, C, M, N, K);
    //cudaDeviceSynchronize(); // debug only
    //printf("cuda_gemm OK\n");
}

