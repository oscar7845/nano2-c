//still 1 block/row
//TODO: test blk rw 
//smax error index???
//TODO:
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void xent_fwd(const float* logit,const int* tgt,
                         int N,int V,float* dlog,float* sumloss)
{
    int r = blockIdx.x;
    if(r>=N) return;

    extern __shared__ float buf[];
    float* smax = buf;
    float* ssum = buf + blockDim.x;

    int tid = threadIdx.x;
    int step = blockDim.x;
    size_t base = (size_t)r * V;

    //max
    float m=-CUDART_INF_F;
    for(int j=tid;j<V;j+=step)
        m = fmaxf(m, logit[base+j]);
    smax[tid] = m; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s) smax[tid] = fmaxf(smax[tid], smax[tid+s]);
        __syncthreads();
    }
    float mx = smax[0];

    //sumexp
    float sm=0.f;
    for(int j=tid;j<V;j+=step)
        sm += expf(logit[base+j] - mx);
    ssum[tid] = sm; __syncthreads();
    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s) ssum[tid] += ssum[tid+s];
        __syncthreads();
    }
    float srow = ssum[0] + 1e-20f;

    //loss
    if(tid==0){
        int t = tgt[r];
        float z = logit[base+t];
        float l = (mx + logf(srow)) - z;
        atomicAdd(sumloss, l);
    }

    //grad
    if(dlog){
        float invN = 1.f/N;
        int t = tgt[r];
        for(int j=tid;j<V;j+=step){
            float p = expf(logit[base+j] - mx) / srow;
            dlog[base+j] = (p - (j==t)) * invN;
        }
    }
}

extern "C"
float nano2_xent_forward_mean(const float* L,const int* T,int N,int V,float* dL)
{
    if(N<=0||V<=0) return 0.f;
    int th = (V>256?256:(V>128?128:64));
    float* dev; cudaMalloc(&dev,4); cudaMemset(dev,0,4);
    xent_fwd<<<N,th,th*2*sizeof(float)>>>(L,T,N,V,dL,dev);

    float h=0; cudaMemcpy(&h,dev,4,cudaMemcpyDeviceToHost);
    cudaFree(dev);
    return h/N;
}

