#include <stdio.h>
#include <cuda_runtime.h>

extern "C" void cuda_gemm(const float* A, const float* B, float* C, int M, int N, int K);
extern "C" void cuda_layernorm(float* x, float* gamma, float* beta, int batch, int dim);
static float* dev_malloc_copy(const float* host, int n){
    float *d;
    cudaMalloc(&d, sizeof(float)*n);
    cudaMemcpy(d, host, sizeof(float)*n, cudaMemcpyHostToDevice);
    return d;
}
static float* dev_malloc_zero(int n){
    float *d;
    cudaMalloc(&d, sizeof(float)*n);
    cudaMemset(d, 0, sizeof(float)*n);
    return d;
}


//TODO:


extern "C" void train_forward_gpu(
    const float *h_x,
    const float *h_W1,
    const float *h_b1,
    const float *h_W2,
    const float *h_b2,
    float *h_out,
    int batch,
    int d_model){
    int hidden = d_model * 4;
    int size_x= batch * d_model;
    int size_h= batch * hidden;
    float *d_x= dev_malloc_copy(h_x, size_x);
    float *d_tmp= dev_malloc_zero(size_h);
    float *d_W1= dev_malloc_copy(h_W1, d_model*hidden);
    float *d_b1= dev_malloc_copy(h_b1, hidden);
    float *d_W2= dev_malloc_copy(h_W2, hidden*d_model);
    float *d_b2= dev_malloc_copy(h_b2, d_model);
    float *d_out = dev_malloc_zero(batch*d_model);

    //layer 1
    cuda_gemm(d_x, d_W1, d_tmp, batch, hidden, d_model);

    //bias add (uber hack)
    for(int b=0;b<batch;b++){
        cudaMemcpy(d_tmp+b*hidden, d_tmp+b*hidden, hidden*sizeof(float), cudaMemcpyDeviceToDevice);
    }

    //layernorm pretend to act as activation for now
    cuda_layernorm(d_tmp, d_b1, d_b1, batch, hidden);

    //layer 2
    cuda_gemm(d_tmp, d_W2, d_out, batch, d_model, hidden);

    //add b2  same lol
    for(int b=0;b<batch;b++){
        cudaMemcpy(d_out+b*d_model, d_out+b*d_model, d_model*sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_out, d_out, sizeof(float)*batch*d_model, cudaMemcpyDeviceToHost);

    cudaFree(d_x); cudaFree(d_tmp); cudaFree(d_out);
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
}

