//softmax backward used 
//in attention backward: dS=(dP-sum(dP*P))*P
#include <cuda_runtime.h>
#include "../cuda_check.h"

__global__ void softmax_backward_kernel(const float* __restrict__ P,
                                        const float* __restrict__ dP,
                                        float* __restrict__ dS,
                                        int rows,int cols){
    int row=blockIdx.x;
    if(row>=rows) return;

    extern __shared__ float smem[];
    int tid=threadIdx.x;
    int base=row*cols;

    float accum=0.f;
    for(int j=tid; j<cols; j+=blockDim.x){
        accum+=dP[base+j]*P[base+j];
    }
    smem[tid]=accum;
    __syncthreads();

    for(int t=blockDim.x>>1; t>0; t>>=1){
        if(tid<t) smem[tid]+=smem[tid+t];
        __syncthreads();
    }

    float dot=smem[0];

    for(int j=tid; j<cols; j+=blockDim.x){
        float p=P[base+j];
        float g=dP[base+j]-dot;
        dS[base+j]=g*p;
    }
}

extern "C" void nano2_softmax_backward(const float* P,const float* dP,float* dS,int rows,int cols){
    int threads=(cols>=256)?256:128;
    dim3 block(threads,1,1), grid(rows,1,1);
    size_t shmem=threads*sizeof(float);
    softmax_backward_kernel<<<grid,block,shmem>>>(P,dP,dS,rows,cols);
    CUDA_CHECK("softmax_backward");
}

