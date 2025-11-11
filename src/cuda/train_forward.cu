#include <stdio.h>
#include <cuda_runtime.h>
//need to debug the weights ??
//--
//not implementing anything   yet
extern "C" void cuda_gemm(const float* A, const float* B, float* C, int M, int N, int K);
extern "C" void cuda_layernorm(float* x, float* gamma, float* beta,int batch, int dim);

//device alloc help (proto stage)
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


//first draft forward 
//nly copy in + out
extern "C" void train_forward_gpu(
    const float *h_x,
    const float *h_W1,
    const float *h_b1,
    const float *h_W2,
    const float *h_b2,
    float *h_out,
    int batch,
    int d_model){
    int hidden =d_model * 4; 
    
    // ok   from model_new

    int size_x= batch * d_model;
    int size_hid= batch * hidden;

    float *d_x= dev_malloc_copy(h_x, size_x);
    float *d_tmp= dev_malloc_zero(size_hid);

    //literally nothing else yet
    //just copy x to tmp (for now)
    cudaMemcpy(d_tmp, d_x, sizeof(float)*size_x, cudaMemcpyDeviceToDevice);

    //and then back out (just shape mismatch but assume same for test)
    cudaMemcpy(h_out, d_tmp, sizeof(float)*size_x, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_tmp);
}

