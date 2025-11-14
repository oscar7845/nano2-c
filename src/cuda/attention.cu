//real softmax + gemm hooks
//TODO:
#include <cuda_runtime.h>
#include <math.h>

extern "C" void nano2_gemm_f32(int tA,int tB,int M,int N,int K,
                               const float*A,int lda,
                               const float*B,int ldb,
                               float*C,int ldc,
                               float alpha,float beta);

extern "C" void nano2_softmax_forward(const float*x,float*y,int R,int C);

__global__ void attn_skm(float*Q,float*K,float*S,int T,int D,float scale){
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<T && j<T){
        float v=0;
        for(int p=0;p<D;p++) v += Q[i*D+p]*K[j*D+p];
        v*=scale;
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
    int BT=B*T;

    //Q,K,V
    nano2_gemm_f32(0,0,BT,D,D,x_ln,D,Wq,D,q,D,1,0);
    nano2_gemm_f32(0,0,BT,D,D,x_ln,D,Wk,D,k,D,1,0);
    nano2_gemm_f32(0,0,BT,D,D,x_ln,D,Wv,D,v,D,1,0);

    float scale=1.f/sqrtf((float)D);
    dim3 b(16,16);
    dim3 g((T+15)/16,(T+15)/16);

    for(int bidx=0;bidx<B;bidx++){
        float*Qb=q + bidx*T*D;
        float*Kb=k + bidx*T*D;
        float*Vb=v + bidx*T*D;

        float*Sb=S + bidx*T*T;
        float*Pb=P + bidx*T*T;
        float*Cb=ctx + bidx*T*D;
        float*Ob=O + bidx*T*D;

        attn_skm<<<g,b>>>(Qb,Kb,Sb,T,D,scale);
        nano2_softmax_forward(Sb,Pb,T,T);
        nano2_gemm_f32(0,0,T,D,T,Pb,T,Vb,D,Cb,D,1,0);
        nano2_gemm_f32(0,0,T,D,D,Cb,D,Wo,D,Ob,D,1,0);
    }
}

