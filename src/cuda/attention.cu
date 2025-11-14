//single-head attention (very slow, to fix)
//Q,K,V computed by GEMM outside or simple loops
//do direct dot loops, change
//TODO:
//
#include <cuda_runtime.h>
#include <math.h>

__global__ void attn_scores(const float*Q,const float*K,float*S,int T,int D,float scale){
    int i = blockIdx.y*blockDim.y + threadIdx.y;
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<T && j<T){
        float s=0;
        for(int p=0;p<D;p++) s += Q[i*D+p]*K[j*D+p];
        float v = s*scale;
        if(j>i) v=-1e9f;
        S[i*T+j] = v;
    }
}

extern "C" void nano2_attention_forward(
    const float* Q, const float* K, const float* V,
    float* S, float* P, float* O,
    int T, int D)
{
    float scale = 1.f/sqrtf((float)D);

    dim3 b(16,16); 
    dim3 g((T+15)/16,(T+15)/16);
    attn_scores<<<g,b>>>(Q,K,S,T,D,scale);

    //softmax outside? placeholder:
    //just copy S->P for now (student version)
    cudaMemcpy(P,S,sizeof(float)*T*T,cudaMemcpyDeviceToDevice);

    //ctx = P@V ; O = ctx (for now)
    cudaMemcpy(O,V,sizeof(float)*T*D,cudaMemcpyDeviceToDevice);
}

