//simple LayerNorm for (batch x dim)
//no epsilon config yet
//TODO: GPU reduction optimization

#include <stdio.h>
#include <cuda_runtime.h>

extern "C" void cuda_layernorm(float* x, float* gamma, float* beta, int batch, int dim);

__global__ void kernel_layernorm(float* x, float* gamma, float* beta,int batch, int dim){
    int row = blockIdx.x;
    if(row >= batch) return;
    float mean = 0.0f;
    for(int i=0;i<dim;i++){
        mean += x[row*dim + i];
    }
    mean /= dim;

    //fix later the second

    float var = 0.0f;
    for(int i=0;i<dim;i++){
        float d = x[row*dim + i] - mean;
        var += d*d;
    }
    var /=dim;
    float inv_std = rsqrtf(var + 1e-5f);

    for(int i=0;i<dim;i++){
        float v = (x[row*dim + i] - mean) * inv_std;
        v = v * gamma[i] + beta[i];
        x[row*dim + i] = v;
    }
}

extern "C" void cuda_layernorm(float* x, float* gamma, float* beta,int batch, int dim){
    kernel_layernorm<<<batch,1>>>(x, gamma, beta, batch, dim);
    //cudaDeviceSynchronize(); // debug only
    //printf("cuda_layernorm done\n");
}
