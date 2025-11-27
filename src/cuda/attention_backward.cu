//softmax backward,
//TODO:
#include <cuda_runtime.h>
#include "../cuda_check.h"

__global__ void softmax_backward_kernel(const float* __restrict__ P,
                                        const float* __restrict__ dP,
                                        float* __restrict__ dS,
                                        int rows,int cols){

    int row=blockIdx.x;
    if(row>=rows) return;

    extern __shared__ float sdata[];
    int tid=threadIdx.x;
    int base=row*cols;

    float acc=0.f;
    for(int j=tid; j<cols; j+=blockDim.x){
        float dp=dP[base+j];
        float p=P[base+j];
        acc+=dp*p;
    }

    sdata[tid]=acc;
    __syncthreads();

    for(int s=blockDim.x>>1; s>0; s>>=1){
        if(tid<s) sdata[tid]+=sdata[tid+s];
        __syncthreads();
    }

    float dot=sdata[0];

    for(int j=tid; j<cols; j+=blockDim.x){
        float p=P[base+j];
        float g=dP[base+j]-dot;
        dS[base+j]=p*g;
    }
}

extern "C" void nano2_softmax_backward(const float* P,const float* dP,float* dS,int rows,int cols){
    int threads=(cols>128)?256:128;
    dim3 block(threads), grid(rows);
    size_t sh=threads*sizeof(float);
    softmax_backward_kernel<<<grid,block,sh>>>(P,dP,dS,rows,cols);
    CUDA_CHECK("softmax_backward");
}

