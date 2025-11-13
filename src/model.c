//move toward flat param layout, still CPU-only MLP
//adds: param carve helper, seedable RNG, debug flag

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int MODEL_DEBUG = 1;

//tiny RNG (not same as big version, but closer)
struct RNG {
    unsigned s;
};
static unsigned xr(unsigned x){ x ^= x<<13; x ^= x>>17; x ^= x<<5; return x; }
static void rng_seed(struct RNG *r, unsigned seed){ r->s = seed?seed:1234567u; }
static float rng_uniform(struct RNG *r){
    r->s = xr(r->s);
    return (r->s & 0xffffff) / 16777216.0f;
}
static float rng_norm(struct RNG *r){
    float u1 = rng_uniform(r);
    float u2 = rng_uniform(r);
    float m = sqrtf(-2.f * logf(u1 + 1e-9f));
    return m * cosf(6.283185f*u2);
}

//flat params (host only)
struct Flat {
    float *p;
    size_t n;
};

//simple MLP dims
struct Model {
    int d_model;
    int hidden;

    //carved pointers
    float *W1, *b1;
    float *W2, *b2;

    struct Flat flat;
};

//carver helper
static float* carve(float **base, size_t n){
    float *p = *base;
    *base += n;
    return p;
}

static size_t count_params(int D){
    int H = 4*D;
    return (size_t)D*H + H + (size_t)H*D + D;
}

struct Model* model_new(int D){
    int H = 4*D;
    struct Model *m = malloc(sizeof(*m));
    memset(m,0,sizeof(*m));
    m->d_model = D;
    m->hidden = H;

    m->flat.n = count_params(D);
    m->flat.p = (float*)malloc(m->flat.n * sizeof(float));

    //carve
    float *base = m->flat.p;
    m->W1 = carve(&base, (size_t)D*H);
    m->b1 = carve(&base, H);
    m->W2 = carve(&base, (size_t)H*D);
    m->b2 = carve(&base, D);

    //init
    struct RNG r; rng_seed(&r, 123);
    float *p = m->flat.p;
    for(size_t i=0;i<m->flat.n;i++){
        p[i] = 0.05f * rng_norm(&r);
    }

    if(MODEL_DEBUG)
        printf("model_new v1.5: D=%d H=%d params=%zu\n",D,H,m->flat.n);

    return m;
}

void model_free(struct Model *m){
    if(!m) return;
    free(m->flat.p);
    free(m);
    if(MODEL_DEBUG) printf("model_free v1.5\n");
}
