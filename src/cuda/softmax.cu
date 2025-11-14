//TODO: logic but ordered steps
//remove sync bugs
// add base
//TODO:
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void sm_kernel(const float* x,float* y,int R,int C){
    int r=blockIdx.x; if(r>=R) return;

    extern __shared__ float mem[];
    float* smax=mem;
    float* ssum=mem+blockDim.x;

    int tid=threadIdx.x, step=blockDim.x;
    size_t base=(size_t)r*C;

    //max
    float mx=-CUDART_INF_F;
    for(int j=tid;j<C;j+=step) mx=fmaxf(mx,x[base+j]);
    smax[tid]=mx; __syncthreads();

    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s) smax[tid]=fmaxf(smax[tid],smax[tid+s]);
        __syncthreads();
    }
    mx = smax[0];

    //sumexp
    float sm=0.f;
    for(int j=tid;j<C;j+=step){
        float e=expf(x[base+j]-mx);
        y[base+j]=e; sm+=e;
    }
    ssum[tid]=sm; __syncthreads();

    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s) ssum[tid]+=ssum[tid+s];
        __syncthreads();
    }
    float denom = ssum[0]+1e-20f;

    //norm
    for(int j=tid;j<C;j+=step)
        y[base+j] /= denom;
}

extern "C"
void nano2_softmax_forward(const float* x,float* y,int R,int C){
    if(R<=0||C<=0) return;
    int th=(C>=256?256:(C>=128?128:64));
    sm_kernel<<<R,th,th*2*sizeof(float)>>>(x,y,R,C);
}

