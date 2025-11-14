//TODO;
//blk and exp + sums warnings
//sms always off by 1
//re add sync
//TODO:
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void soft_fwd(const float* x,float* y,int R,int C){
    int r=blockIdx.x; if(r>=R) return;

    extern __shared__ float tmp[];
    float* mxs = tmp;
    float* sms = tmp + blockDim.x;

    int tid=threadIdx.x;
    int step=blockDim.x;
    size_t base=(size_t)r*C;

    //max
    float mx=-CUDART_INF_F;
    for(int j=tid;j<C;j+=step) mx=fmaxf(mx,x[base+j]);
    mxs[tid]=mx; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s) mxs[tid]=fmaxf(mxs[tid],mxs[tid+s]);
        __syncthreads();
    }
    mx=mxs[0];

    //exp + sum
    float sum=0.f;
    for(int j=tid;j<C;j+=step){
        float e=expf(x[base+j]-mx);
        y[base+j]=e;
        sum+=e;
    }
    sms[tid]=sum; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s) sms[tid]+=sms[tid+s];
        __syncthreads();
    }
    float denom=sms[0]+1e-20f;

    //normalize
    for(int j=tid;j<C;j+=step)
        y[base+j]/=denom;
}

extern "C"
void nano2_softmax_forward(const float* x,float* y,int R,int C){
    if(R<=0||C<=0) return;
    int th=(C>=256?256:(C>=128?128:64));
    soft_fwd<<<R,th,th*2*sizeof(float)>>>(x,y,R,C);
}

