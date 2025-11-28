//softmax backward 
//used by attention backward: 
//dS = (dP - sum(dP*P))*P (row-wise)
#include <cuda_runtime.h>
#include "../cuda_check.h"

__global__ void softmax_backward_kernel(const float* __restrict__ P,
                                        const float* __restrict__ dP,
                                        float* __restrict__ dS,
                                        int rows,int cols){
    int row=blockIdx.x;
    if(row>=rows) return;

    extern __shared__ float smem[];
    float* ssum=smem; // [blockDim.x]

    const int tid=threadIdx.x;
    const int stride=blockDim.x;
    const size_t base=(size_t)row*(size_t)cols;

    float acc=0.0f;
    for(int j=tid; j<cols; j+=stride){
        acc+=dP[base+j]*P[base+j];
    }
    ssum[tid]=acc; 
    __syncthreads();

    for(int t=blockDim.x>>1; t>0; t>>=1){
        if(tid<t) ssum[tid]+=ssum[tid+t];
        __syncthreads();
    }
    float dot=ssum[0];

    for(int j=tid; j<cols; j+=stride){
        float p=P[base+j];
        float g=dP[base+j]-dot;
        dS[base+j]=g*p;
    }
}

extern "C" void nano2_softmax_backward(const float* P,const float* dP,float* dS,int rows,int cols){
    int threads=(cols>=256)?256:(cols>=128?128:64);
    dim3 block(threads,1,1), grid(rows,1,1);
    size_t shmem=(size_t)threads*sizeof(float);
    softmax_backward_kernel<<<grid,block,shmem>>>(P,dP,dS,rows,cols);
    CUDA_CHECK("softmax_backward");
}
