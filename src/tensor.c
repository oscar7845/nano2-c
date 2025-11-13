// add sin/cos positional tables
// full RNG + fills + host/device helpers
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>

//memory helpers
void* nano2_malloc_host(size_t b){ return b? malloc(b):NULL; }
void nano2_free_host(void *p){ free(p); }

void* nano2_malloc_device(size_t b){ void *p=NULL; if(b) cudaMalloc(&p,b); return p; }
void nano2_free_device(void *p){ if(p) cudaFree(p); }

void nano2_copy_host_to_device(void*d,const void*s,size_t n){
    if(n) cudaMemcpy(d,s,n,cudaMemcpyHostToDevice);
}
void nano2_copy_device_to_host(void*d,const void*s,size_t n){
    if(n) cudaMemcpy(d,s,n,cudaMemcpyDeviceToHost);
}
void nano2_memset_device(void*d,int v,size_t n){
    if(n) cudaMemset(d,v,n);
}

//RNG
struct Nano2RNG{
    uint32_t s;
    int have_spare;
    float spare;
};

static inline uint32_t xorshift32(uint32_t x){
    x ^= x<<13; x ^= x>>17; x ^= x<<5;
    return x;
}

void nano2_rng_seed(struct Nano2RNG* r, uint32_t seed){
    r->s = seed? seed: 0x9e3779b9u;
    r->s = xorshift32(r->s);
    r->have_spare=0;
}

float nano2_rand_uniform(struct Nano2RNG* r){
    r->s = xorshift32(r->s);
    const float scale = 1.0f/4294967296.0f;
    float u = (r->s+1u)*scale;
    if(u>=1.0f) u=0.99999994f;
    return u;
}

float nano2_randn(struct Nano2RNG* r){
    if(r->have_spare){
        r->have_spare=0;
        return r->spare;
    }
    float u1=nano2_rand_uniform(r);
    float u2=nano2_rand_uniform(r);
    float m=sqrtf(-2.0f*logf(u1));
    float z0=m*cosf(2.0f*M_PI*u2);
    float z1=m*sinf(2.0f*M_PI*u2);
    r->spare=z1; r->have_spare=1;
    return z0;
}

//fill helpers
void nano2_fill_gaussian(float *dst,size_t n,float std,struct Nano2RNG *rng){
    for(size_t i=0;i<n;i++) dst[i] = std * nano2_randn(rng);
}
void nano2_fill_constant(float *dst,size_t n,float v){
    for(size_t i=0;i<n;i++) dst[i]=v;
}
void nano2_fill_zeros(float *dst,size_t n){ nano2_fill_constant(dst,n,0); }
void nano2_fill_ones (float *dst,size_t n){ nano2_fill_constant(dst,n,1); }

//sin/cos tables
void nano2_make_sincos_tables(int T,int D,float *sin_out,float *cos_out){
    int H=D/2;
    for(int t=0;t<T;t++){
        for(int i=0;i<H;i++){
            float exponent = (2.0f*(float)i)/(float)D;
            float denom = powf(10000.0f,exponent);
            float ang   = (float)t/denom;
            sin_out[t*H+i]=sinf(ang);
            cos_out[t*H+i]=cosf(ang);
        }
    }
}

//tensor struct
struct Tensor{ int rows; int cols; float *data; };

struct Tensor* tensor_create(int r,int c){
    struct Tensor *t=malloc(sizeof(struct Tensor));
    t->rows=r; t->cols=c;
    t->data=nano2_malloc_host(sizeof(float)*r*c);
    nano2_fill_zeros(t->data,(size_t)r*c);
    return t;
}

void tensor_fill(struct Tensor *t,float v){
    nano2_fill_constant(t->data, (size_t)t->rows*t->cols, v);
}

void tensor_fill_random(struct Tensor *t){
    int n=t->rows*t->cols;
    for(int i=0;i<n;i++) t->data[i] = (float)(rand()%100)/50.0f;
}

void tensor_show(struct Tensor *t){
    int n=t->rows*t->cols; if(n>16)n=16;
    printf("tensor_show %dx%d:\n",t->rows,t->cols);
    for(int i=0;i<n;i++) printf("%.4f ", t->data[i]);
    printf("\n");
}

void tensor_free(struct Tensor *t){
    nano2_free_host(t->data);
    free(t);
}
