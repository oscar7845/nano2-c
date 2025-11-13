// add RNG (xorshift32 + basic gaussian)
// still missing sin/cos tables

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime_api.h>

void* nano2_malloc_host(size_t b){ return b? malloc(b):NULL; }
void nano2_free_host(void *p){ free(p); }

void* nano2_malloc_device(size_t b){ void* p=NULL; if(b) cudaMalloc(&p,b); return p; }
void nano2_free_device(void *p){ if(p) cudaFree(p); }

void nano2_copy_host_to_device(void *d,const void *s,size_t b){
    if(b) cudaMemcpy(d,s,b,cudaMemcpyHostToDevice);
}
void nano2_copy_device_to_host(void *d,const void *s,size_t b){
    if(b) cudaMemcpy(d,s,b,cudaMemcpyDeviceToHost);
}

//RNG
struct Nano2RNG{
    uint32_t s;
    int have_spare;
    float spare;
};

static inline uint32_t nano2_xs(uint32_t x){
    x ^= x<<13; x ^= x>>17; x ^= x<<5;
    return x;
}

void nano2_rng_seed(struct Nano2RNG *r,uint32_t seed){
    r->s = seed? seed: 0x12345678u;
    r->s = nano2_xs(r->s);
    r->have_spare=0;
}

float nano2_rand_uniform(struct Nano2RNG *r){
    r->s = nano2_xs(r->s);
    const float scale = 1.0f / 4294967296.0f;
    float u = (r->s+1u)*scale;
    if(u>=1.0f) u=0.9999999f;
    return u;
}

float nano2_randn(struct Nano2RNG *r){
    if(r->have_spare){
        r->have_spare=0;
        return r->spare;
    }
    float u1 = nano2_rand_uniform(r);
    float u2 = nano2_rand_uniform(r);
    float m = sqrtf(-2*logf(u1));
    float z0 = m*cosf(2*M_PI*u2);
    float z1 = m*sinf(2*M_PI*u2);
    r->spare = z1; r->have_spare=1;
    return z0;
}

//tensor struct
struct Tensor{ int rows; int cols; float *data; };

struct Tensor* tensor_create(int r,int c){
    struct Tensor *t = malloc(sizeof(struct Tensor));
    t->rows=r; t->cols=c;
    t->data=nano2_malloc_host(sizeof(float)*r*c);
    for(int i=0;i<r*c;i++) t->data[i]=0;
    return t;
}

void tensor_fill(struct Tensor *t,float v){
    int n=t->rows*t->cols;
    for(int i=0;i<n;i++) t->data[i]=v;
}

void tensor_fill_random(struct Tensor *t){
    int n=t->rows*t->cols;
    for(int i=0;i<n;i++) t->data[i] = (float)(rand()%100)/30.0f;
}

void tensor_fill_gaussian(struct Tensor *t,float std,struct Nano2RNG *rng){
    int n=t->rows*t->cols;
    for(int i=0;i<n;i++) t->data[i] = std * nano2_randn(rng);
}

void tensor_show(struct Tensor *t){
    int n=t->rows*t->cols; if(n>16)n=16;
    printf("tensor_show %dx%d:\n",t->rows,t->cols);
    for(int i=0;i<n;i++) printf("%.3f ",t->data[i]);
    printf("\n");
}

void tensor_free(struct Tensor *t){
    nano2_free_host(t->data);
    free(t);
}
