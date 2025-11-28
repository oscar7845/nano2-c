#include <cuda_runtime.h>
#include "../cuda_check.h"
#include "embed.h"

//the actual kernel (moved here, single definition)
extern "C" __global__
void embed_add_pos_kernel(
    const uint8_t* __restrict__ tokens,
    const float*   __restrict__ E,
    const float*   __restrict__ pos_sin,
    const float*   __restrict__ pos_cos,
    float*         __restrict__ out,
    int B,int T,int D
);


//embed tokens and add sinusoidal positions: out[row, d] = E[token, d] + pos_{t}[d]
//we use half-dim sin/cos tables and interleave them: even d -> sin, odd d -> cos.
extern "C" __global__ void embed_add_pos_kernel(const uint8_t* __restrict__ tokens,const float* __restrict__ E,
                const float* __restrict__ pos_sin,const float* __restrict__ pos_cos,float* __restrict__ out,
                int B,int T,int D){
  int row=blockIdx.x; // which token in [0, B*T)
  if(row>=B*T) return;
  int tid=threadIdx.x;
  int stride=blockDim.x;
  const int H=D>>1; // D assumed even
  int tok=(int)tokens[row];
  int t=row%T; // position within sequence
  size_t ebase=(size_t)tok*(size_t)D;
  size_t obase=(size_t)row*(size_t)D;
  size_t pbase=(size_t)t*(size_t)H;
  for(int d=tid; d<D; d+=stride){
    float v=E[ebase+d];
    int i=d>>1;
    v+=( (d&1)==0 ? pos_sin[pbase+i] : pos_cos[pbase+i] );
    out[obase+d]=v;
  }
}


//wrapper: same launch config forward/backward
extern "C" void nano2_embed_add_pos(
    const uint8_t* x,
    const float*   E,
    const float*   pos_sin,
    const float*   pos_cos,
    float*         out,
    int B,int T,int D
){
    int BT=B*T;
    int threads=(D>=256)?256:(D>=128?128:64);
    dim3 block(threads,1,1);
    dim3 grid(BT,1,1);

    embed_add_pos_kernel<<<grid,block>>>(x,E,pos_sin,pos_cos,out,B,T,D);
    CUDA_CHECK("embed_add_pos");
}

