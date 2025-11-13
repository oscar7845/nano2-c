//CPU-only but struct/organization closer to final transformer model
//grouped param struct, 
//more logging, 
//seed passthrough, 
//clearer carving
//TODO:
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int MODEL_DEBUG = 1;

struct RNG{ unsigned s; };
static unsigned xr(unsigned x){ x^=x<<13; x^=x>>17; x^=x<<5; return x; }
static void rng_seed(struct RNG *r,unsigned s){ r->s=s?s:1337u; }
static float rng_norm(struct RNG *r){
    r->s=xr(r->s);
    float u1=(r->s&0xffff)/65536.f;
    r->s=xr(r->s);
    float u2=(r->s&0xffff)/65536.f;
    float m=sqrtf(-2.f*logf(u1+1e-9));
    return m*cosf(6.283185f*u2);
}

struct Params {
    float *W1,*b1;
    float *W2,*b2;
};

struct Model {
    int D,H;
    struct Params p;

    float *flat;
    size_t n;

    //scratch like future Buffers
    float *ff1;
    float *ff2;
};

static float* carve(float **b,size_t n){ float *p=*b; *b+=n; return p; }

static size_t count(int D){
    int H=4*D;
    return (size_t)D*H + H + (size_t)H*D + D;
}

struct Model* model_new(int D){
    int H=4*D;
    struct Model *m=malloc(sizeof(*m));
    memset(m,0,sizeof(*m));
    m->D=D; m->H=H;

    m->n = count(D);
    m->flat = malloc(m->n*sizeof(float));

    float *base=m->flat;
    m->p.W1 = carve(&base,(size_t)D*H);
    m->p.b1 = carve(&base,H);
    m->p.W2 = carve(&base,(size_t)H*D);
    m->p.b2 = carve(&base,D);

    struct RNG r; rng_seed(&r,12345);
    for(size_t i=0;i<m->n;i++) m->flat[i]=0.04f*rng_norm(&r);

    if(MODEL_DEBUG)
        printf("model_new v3.0: D=%d H=%d params=%zu\n",D,H,m->n);

    return m;
}

//matmul same as before
static void mm(float *A,float *B,float *C,int M,int K,int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float s=0;
            for(int k=0;k<K;k++) s+=A[i*K+k]*B[k*N+j];
            C[i*N+j]=s;
        }
    }
}

void model_forward(struct Model *m, float *x,int B){
    m->ff1 = malloc((size_t)B*m->H*sizeof(float));
    m->ff2 = malloc((size_t)B*m->D*sizeof(float));

    mm(x, m->p.W1, m->ff1, B, m->D, m->H);

    for(int b=0;b<B;b++)
        for(int j=0;j<m->H;j++)
            m->ff1[b*m->H+j] += m->p.b1[j];

    for(int i=0;i<B*m->H;i++) m->ff1[i] = m->ff1[i]>0?m->ff1[i]:0;

    mm(m->ff1, m->p.W2, m->ff2, B, m->H, m->D);

    for(int b=0;b<B;b++)
        for(int j=0;j<m->D;j++)
            m->ff2[b*m->D+j] += m->p.b2[j];

    if(MODEL_DEBUG)
        printf("forward v3.0 ok\n");

    free(m->ff1);
    free(m->ff2);
}

void model_free(struct Model *m){
    free(m->flat);
    free(m);
    if(MODEL_DEBUG) printf("model_free v3.0\n");
}
