//TODO: add debug 
//scaling notes
//TODO:
//
#include <cuda_runtime.h>
#include <math.h>

static int ATTN_DEBUG=0;

__global__ void qk_kernel(const float*Q,const float*K,float*S,int T,int D,float scale){
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<T && j<T){
        float acc=0;
        for(int p=0;p<D;p++) acc += Q[i*D+p]*K[j*D+p];
        float v=acc*scale;
        if(j>i) v=-1e9f;
        S[i*T+j]=v;
    }
}

extern "C" void nano2_attention_forward(
    const float*x_ln,int B,int T,int D,
    const float*Wq,const float*Wk,const float*Wv,const float*Wo,
    float*q,float*k,float*v,
    float*S,float*P,float*ctx,float*O)
{
    float scale=1.f/sqrtf((float)D);

    //Q,K,V
    extern void nano2_gemm_f32(int,int,int,int,int,const float*,int,const float*,int,float*,int,float,float);
    nano2_gemm_f32(0,0,B*T,D,D,x_ln,D,Wq,D,q,D,1,0);
    nano2_gemm_f32(0,0,B*T,D,D,x_ln,D,Wk,D,k,D,1,0);
    nano2_gemm_f32(0,0,B*T,D,D,x_ln,D,Wv,D,v,D,1,0);

    dim3 b(16,16), g((T+15)/16,(T+15)/16);

    for(int bidx=0;bidx<B;bidx++){
        float*Qb=q+bidx*T*D;
        float*Kb=k+bidx*T*D;
        float*Sb=S+bidx*T*T;

        qk_kernel<<<g,b>>>(Qb,Kb,Sb,T,D,scale);

        extern void nano2_softmax_forward(const float*,float*,int,int);
        float*Pb=P+bidx*T*T;
        nano2_softmax_forward(Sb,Pb,T,T);

        float*Vb=v+bidx*T*D;
        float*Cb=ctx+bidx*T*D;
        nano2_gemm_f32(0,0,T,D,T,Pb,T,Vb,D,Cb,D,1,0);

        float*Ob=O+bidx*T*D;
        nano2_gemm_f32(0,0,T,D,D,Cb,D,Wo,D,Ob,D,1,0);

        if(ATTN_DEBUG && bidx==0) printf("attn batch0 ok\n");
    }
}

