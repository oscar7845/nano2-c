//CPU-only MLP but closer structure to the future GPU transformer
//adds: scratch buffers, activation flag, tiny timer stub

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int MODEL_DEBUG = 1;
static int MODEL_ACT = 0; //0=ReLU, 1=tanh

//RNG as before
struct RNG{ unsigned s; };
static unsigned xr(unsigned x){ x^=x<<13; x^=x>>17; x^=x<<5; return x; }
static void rng_seed(struct RNG *r,unsigned s){ r->s=s?s:99991; }
static float rng_unif(struct RNG *r){ r->s=xr(r->s); return (r->s&0xffffff)/16777216.f; }
static float rng_norm(struct RNG *r){
    float u1=rng_unif(r), u2=rng_unif(r);
    float m=sqrtf(-2.f*logf(u1+1e-9f));
    return m*cosf(6.283185f*u2);
}

//activation
static void act_inplace(float *x,int n){
    if(MODEL_ACT==0){
        for(int i=0;i<n;i++) x[i] = x[i]>0?x[i]:0;
    } else {
        for(int i=0;i<n;i++) x[i] = tanhf(x[i]);
    }
}

//timers (stub)
static void t0(){}
static void t1(const char *tag){
    if(MODEL_DEBUG) printf("[timer] %s\n",tag);
}

//model
struct Model{
    int D,H;
    float *W1,*b1,*W2,*b2;
    float *flat;
    size_t n;

    //forward scratch
    float *tmp1; // batch x hidden
    float *tmp2; // batch x D
};

static float* carve(float **b,size_t n){ float *p=*b; *b+=n; return p; }

static size_t count_params(int D){
    int H=4*D;
    return (size_t)D*H + H + (size_t)H*D + D;
}

struct Model* model_new(int D){
    int H=4*D;
    struct Model *m=malloc(sizeof(*m));
    memset(m,0,sizeof(*m));
    m->D=D; m->H=H;

    m->n = count_params(D);
    m->flat = malloc(m->n*sizeof(float));

    float *base=m->flat;
    m->W1 = carve(&base, (size_t)D*H);
    m->b1 = carve(&base, H);
    m->W2 = carve(&base, (size_t)H*D);
    m->b2 = carve(&base, D);

    struct RNG r; rng_seed(&r, 42);
    for(size_t i=0;i<m->n;i++) m->flat[i]=0.03f*rng_norm(&r);

    if(MODEL_DEBUG)
        printf("model_new v2.0: D=%d H=%d params=%zu\n",D,H,m->n);

    return m;
}

//simple CPU matmul: C=BxH or HxD shapes as needed
static void mm(float *A, float *B, float *C,int M,int K,int N){
    //A: MxK, B:KxN, C:MxN
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            float s=0;
            for(int k=0;k<K;k++) s += A[i*K+k]*B[k*N+j];
            C[i*N+j]=s;
        }
    }
}

void model_forward(struct Model *m, float *x,int batch){
    //allocate scratch each call (not ideal, but closer to your dev style)
    m->tmp1 = malloc((size_t)batch*m->H*sizeof(float));
    m->tmp2 = malloc((size_t)batch*m->D*sizeof(float));

    t0();
    mm(x, m->W1, m->tmp1, batch, m->D, m->H);
    t1("mm1");

    for(int b=0;b<batch;b++)
        for(int j=0;j<m->H;j++)
            m->tmp1[b*m->H+j] += m->b1[j];

    act_inplace(m->tmp1, batch*m->H);

    t0();
    mm(m->tmp1, m->W2, m->tmp2, batch, m->H, m->D);
    t1("mm2");

    for(int b=0;b<batch;b++)
        for(int j=0;j<m->D;j++)
            m->tmp2[b*m->D+j] += m->b2[j];

    if(MODEL_DEBUG) printf("forward v2.0 done\n");

    //return tmp2 as result? up to user
    free(m->tmp1);
    free(m->tmp2);
}

void model_free(struct Model *m){
    free(m->flat);
    free(m);
    if(MODEL_DEBUG) printf("model_free v2.0\n");
}
