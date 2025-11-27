#include <cuda_runtime.h>
#include "../cuda_check.h"
#include "embed.h"

//embed+pos
extern "C" __global__
void embed_add_pos_kernel(const uint8_t* __restrict__ tokens,const float* __restrict__ E,
    const float* __restrict__ pos_sin,const float* __restrict__ pos_cos,float* __restrict__ out,
    int B,int T,int D);

extern "C" __global__ void embed_add_pos_kernel(const uint8_t* __restrict__ tokens,const float* __restrict__ E,
        const float* __restrict__ pos_sin,const float* __restrict__ pos_cos,float* __restrict__ out,
        int B,int T,int D){

    int row=blockIdx.x;
    if(row>=B*T) return;

    int tid=threadIdx.x;
    int stride=blockDim.x;
    int H=D>>1;

    int tok=(int)tokens[row];
    int t=row%T;

    size_t ebase=(size_t)tok*D;
    size_t obase=(size_t)row*D;
    size_t pbase=(size_t)t*H;

    for(int d=tid; d<D; d+=stride){
        float v=E[ebase+d];
        int i=d>>1;
        if((d&1)==0) v+=pos_sin[pbase+i];
        else v+=pos_cos[pbase+i];
        out[obase+d]=v;
    }
}

extern "C" void nano2_embed_add_pos(const uint8_t* x,const float* E,
    const float* pos_sin,const float* pos_cos,float* out,int B,int T,int D){

    int BT=B*T;
    int threads=(D>=256)?256:(D>=128)?128:64;
    dim3 block(threads,1,1);
    dim3 grid(BT,1,1);

    embed_add_pos_kernel<<<grid,block>>>(x,E,pos_sin,pos_cos,out,B,T,D);
    CUDA_CHECK("embed_add_pos");
}

