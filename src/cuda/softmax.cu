//row-softmax
//1 block/row
//reduce max then reduce sum
//TODO: remove debugs
//

#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void sm_row(const float* x,float* y,int R,int C){
    int r = blockIdx.x; if(r>=R) return;
    extern __shared__ float sm[];
    float* smax = sm;
    float* ssum = sm + blockDim.x;

    int tid = threadIdx.x;
    int step= blockDim.x;
    size_t base = (size_t)r*C;

    float m=-CUDART_INF_F;
    for(int j=tid;j<C;j+=step) m = fmaxf(m, x[base+j]);
    smax[tid]=m; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s && smax[tid+s]>smax[tid]) smax[tid]=smax[tid+s];
        __syncthreads();
    }
    float mx=smax[0];

    float smu=0;
    for(int j=tid;j<C;j+=step){
        float e=expf(x[base+j]-mx);
        y[base+j]=e;
        smu+=e;
    }
    ssum[tid]=smu; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s) ssum[tid]+=ssum[tid+s];
        __syncthreads();
    }
    float total = ssum[0] + 1e-20f;

    for(int j=tid;j<C;j+=step) y[base+j]/=total;
}

extern "C"
void nano2_softmax_forward(const float* x,float* y,int R,int C){
    if(R<=0||C<=0) return;
    int th = (C>=256?256:(C>=128?128:64));
    sm_row<<<R,th,th*2*sizeof(float)>>>(x,y,R,C);
}

