#include <cuda_runtime.h>
#include "../cuda_check.h"
#include "embed.h"

extern "C" __global__
void embed_add_pos_kernel(
    const uint8_t* __restrict__ tokens,
    const float*   __restrict__ E,
    const float*   __restrict__ pos_sin,
    const float*   __restrict__ pos_cos,
    float*         __restrict__ out,
    int B,int T,int D
);

extern "C" __global__ void embed_add_pos_kernel(const uint8_t* __restrict__ tokens,
    const float* __restrict__ E,const float* __restrict__ pos_sin,const float* __restrict__ pos_cos,
    float* __restrict__ out,int B,int T,int D){

    int row=blockIdx.x;
    if(row>=B*T) return;

    int tid=threadIdx.x;
    int stride=blockDim.x;
    int H=D>>1;

    int tok=(int)tokens[row];
    int t=row%T;

    size_t ebase=(size_t)tok*(size_t)D;
    size_t obase=(size_t)row*(size_t)D;
    size_t pbase=(size_t)t*(size_t)H;

    for(int d=tid; d<D; d+=stride){
        float v=E[ebase+d];
        int i=d>>1;
        float pos=( (d&1)==0 ? pos_sin[pbase+i] : pos_cos[pbase+i] );
        v+=pos;
        out[obase+d]=v;
    }
}
