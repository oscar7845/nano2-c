//softmax backward, 
//not sure about reduction yet
//TODO:
#include <cuda_runtime.h>
#include "../cuda_check.h"

__global__ void softmax_backward_kernel(const float* P,
                                        const float* dP,
                                        float* dS,
                                        int rows, int cols){
    int r=blockIdx.x;
    if(r>=rows) return;

    extern __shared__ float tmp[];
    int tid=threadIdx.x;
    int base=r*cols;

    float sum=0.f;
    for(int j=tid; j<cols; j+=blockDim.x){
        float a=dP[base+j];
        float b=P[base+j];
        sum+=a*b;
    }
    tmp[tid]=sum;
    __syncthreads();

    for(int s=blockDim.x/2; s>0; s>>=1){
        if(tid<s) tmp[tid]+=tmp[tid+s];
        __syncthreads();
    }

    float dot=tmp[0];

    for(int j=tid; j<cols; j+=blockDim.x){
        float p=P[base+j];
        float g=dP[base+j]-dot;
        dS[base+j]=g*p;
    }
}

extern "C" void nano2_softmax_backward(const float* P,const float* dP,float* dS,int rows,int cols){
    int t=128;
    dim3 block(t), grid(rows);
    size_t shm=t*sizeof(float);
    softmax_backward_kernel<<<grid,block,shm>>>(P,dP,dS,rows,cols);
    CUDA_CHECK("softmax_backward");
}

