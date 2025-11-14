//i do 1 block/row, reduce max, reduce sumexp, get loss.
//dlogits??
//TODO:
//
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void loss_row_kernel(const float* logits,
                                const int* targets,
                                int N, int V,
                                float* dlogits,
                                float* loss_sum)
{
    int r = blockIdx.x;
    if(r >= N) return;

    extern __shared__ float sm[];
    float* smax = sm;
    float* ssum = sm + blockDim.x;

    int tid = threadIdx.x;
    int step = blockDim.x;
    size_t base = (size_t)r * V;

    //max
    float m = -CUDART_INF_F;
    for(int j=tid;j<V;j+=step){
        float v = logits[base+j];
        if(v > m) m = v;
    }
    smax[tid] = m;
    __syncthreads();

    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid < s && smax[tid+s] > smax[tid])
            smax[tid] = smax[tid+s];
        __syncthreads();
    }
    float rowmax = smax[0];

    //sum exp
    float sum = 0.f;
    for(int j=tid;j<V;j+=step){
        sum += expf(logits[base+j] - rowmax);
    }
    ssum[tid] = sum;
    __syncthreads();

    for(int s=blockDim.x>>1;s>0;s>>=1){
        if(tid < s) ssum[tid] += ssum[tid+s];
        __syncthreads();
    }
    float rowsum = ssum[0] + 1e-20f;

    //loss
    if(tid==0){
        int t = targets[r];
        float zt = logits[base+t];
        float l = (rowmax + logf(rowsum)) - zt;
        atomicAdd(loss_sum, l);
    }

    //grad 
    //fix that
    //
    if(dlogits){
        float invN = 1.f / N;
        int t = targets[r];
        for(int j=tid;j<V;j+=step){
            float p = expf(logits[base+j] - rowmax) / rowsum;
            dlogits[base+j] = (p - (j==t)) * invN;
        }
    }
}

extern "C" float nano2_xent_forward_mean(const float* logits,const int* targets,
                                         int N,int V,float* dlogits)
{
    if(N<=0||V<=0) return 0.f;
    int tb = (V>=256?256:(V>=128?128:64));
    float* dsum; cudaMalloc(&dsum,4); cudaMemset(dsum,0,4);
    loss_row_kernel<<<N,tb, tb*2*sizeof(float)>>>(logits,targets,N,V,dlogits,dsum);

    float h=0.f; cudaMemcpy(&h,dsum,4,cudaMemcpyDeviceToHost);
    cudaFree(dsum);
    return h / N;
}

