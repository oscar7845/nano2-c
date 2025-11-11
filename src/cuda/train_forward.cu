#include <stdio.h>
#include <cuda_runtime.h>

//gemm.cu
extern "C" void cuda_gemm(const float* A, const float* B, float* C,
                          int M, int N, int K);

//layernorm.cu
extern "C" void cuda_layernorm(float* x, float* gamma, float* beta,
                               int batch, int dim);

//device alloc helpers
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

extern "C" void train_forward_gpu(
    const float *h_x,//nput batch  (batch x d_model)
    const float *h_W1,
    const float *h_b1,
    const float *h_W2,
    const float *h_b2,
    float *h_out,//output buffer (batch x d_model)
    int batch,
    int d_model
){
    int hidden = d_model * 4;

    int size_x = batch * d_model;
    int size_hid= batch * hidden;
    int size_w1= d_model * hidden;
    int size_w2= hidden * d_model;

    float *d_x = dev_malloc_copy(h_x, size_x);
    float *d_tmp= dev_malloc_zero(size_hid);
    float *d_out= dev_malloc_zero(batch * d_model);

    float *d_W1= dev_malloc_copy(h_W1, size_w1);
    float *d_b1= dev_malloc_copy(h_b1, hidden);

    float *d_W2 = dev_malloc_copy(h_W2, size_w2);
    float *d_b2= dev_malloc_copy(h_b2, d_model);

    cuda_gemm(d_x, d_W1, d_tmp, batch, hidden, d_model);

    for(int b=0; b<batch; b++){
        cudaMemcpy(d_tmp + b*hidden, d_tmp + b*hidden, hidden*sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cuda_layernorm(d_tmp, d_b1, d_b1, batch, hidden); // HACK
    cuda_gemm(d_tmp, d_W2, d_out, batch, d_model, hidden);
    for(int b=0; b<batch; b++){
        cudaMemcpy(d_out + b*d_model, d_out + b*d_model, d_model*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(h_out, d_out, sizeof(float)*batch*d_model, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_tmp);
    cudaFree(d_out);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);
}

