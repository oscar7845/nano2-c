//TODO: remove debugs print
// TODO: smax fix
//
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void loss_kernel(const float* logits,const int* tgt,
                            int N,int V,float* dlogits,float* loss_sum)
{
    int r = blockIdx.x;
    if(r>=N) return;

    extern __shared__ float sm[];
    float* smax = sm;
    float* ssum = sm + blockDim.x;

    int tid = threadIdx.x;
    int step = blockDim.x;
    size_t base = (size_t)r * V;

    //max
    //fix that later
    float mx = -CUDART_INF_F;
    for(int j=tid;j<V;j+=step){
        float v = logits[base+j];
        if(v > mx) mx = v;
    }
    smax[tid] = mx;
    __syncthreads();

    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s && smax[tid+s] > smax[tid])
            smax[tid] = smax[tid+s];
        __syncthreads();
    }
    float rowmax = smax[0];

    //sumexp
    float smu = 0.f;
    for(int j=tid;j<V;j+=step)
        smu += expf(logits[base+j] - rowmax);
    ssum[tid] = smu;
    __syncthreads();

    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid<s) ssum[tid] += ssum[tid+s];
        __syncthreads();
    }
    float rowsum = ssum[0] + 1e-20f;

    //loss
    if(tid==0){
        int t = tgt[r];
        float zt = logits[base+t];
        float loss = (rowmax + logf(rowsum)) - zt;
        atomicAdd(loss_sum, loss);
    }

    //grad
    if(dlogits){
        float invN = 1.f/N;
        int t = tgt[r];
        for(int j=tid;j<V;j+=step){
            float p = expf(logits[base+j] - rowmax) / rowsum;
            dlogits[base+j] = (p - (j==t)) * invN;
        }
    }
}

extern "C"
float nano2_xent_forward_mean(const float* L,const int* T,int N,int V,float* dL)
{
    if(N<=0||V<=0) return 0;
    int th = (V>=256?256:(V>=128?128:64));
    float* dev; cudaMalloc(&dev,4); cudaMemset(dev,0,4);
    loss_kernel<<<N,th,th*2*sizeof(float)>>>(L,T,N,V,dL,dev);

    float h=0; cudaMemcpy(&h,dev,4,cudaMemcpyDeviceToHost);
    cudaFree(dev);
    return h / N;
}

