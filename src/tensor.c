//host/device mem helpers
//RNG
//fillers
//sinusoidal table
//checks (i assume valid inputs; maybe change later)
//deterministic seed
//TODO: remove debug prints
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>

//mem helps 
//(host/device)
void* nano2_malloc_host(size_t bytes){
  return bytes ? malloc(bytes) : NULL;
}
void nano2_free_host(void* p){
  free(p);
}
void* nano2_malloc_pinned(size_t bytes){
  void* p= NULL; if(bytes) cudaHostAlloc(&p, bytes, cudaHostAllocDefault); return p;
}
void nano2_free_pinned(void* p){
  if(p) cudaFreeHost(p);
}
void* nano2_malloc_device(size_t bytes){
  void* p= NULL; if(bytes) cudaMalloc(&p, bytes); return p;
}
void nano2_free_device(void* p){
  if(p) cudaFree(p);
}
void nano2_copy_host_to_device(void* dst_dev, const void* src_host, size_t bytes){
  if(bytes) cudaMemcpy(dst_dev, src_host, bytes, cudaMemcpyHostToDevice);
}
void nano2_copy_device_to_host(void* dst_host, const void* src_dev, size_t bytes){
  if(bytes) cudaMemcpy(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost);
}
void nano2_memset_device(void* dst_dev, int value, size_t bytes){
  if(bytes) cudaMemset(dst_dev, value, bytes);
}

//RNG (xorshift32 + Box-Muller)
struct Nano2RNG{
  uint32_t s;
  int have_spare;
  float spare;
};

static inline uint32_t xorshift32(uint32_t x){
  x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x;
}

void nano2_rng_seed(struct Nano2RNG* r, uint32_t seed){
  //to avoid zero lock-up. mix a bit
  r->s = seed ? seed : 0x9e3779b9u;
  r->s ^= 0x85ebca6bu; r->s = xorshift32(r->s);
  r->have_spare = 0; r->spare = 0.0f;
}

float nano2_rand_uniform(struct Nano2RNG* r){
  r->s = xorshift32(r->s);
  //map to (0,1); to avoid exact 0 to keep Box-Muller stable
  const float scale = 1.0f / 4294967296.0f; //2^32
  float u= (r->s + 1u) * scale; //(0,1]
  if(u >= 1.0f) u= 0.99999994f; //clamp
  return u;
}

float nano2_randn(struct Nano2RNG* r){
  if(r->have_spare){ 
    r->have_spare= 0; 
    return r->spare; 
  }
  float u1= nano2_rand_uniform(r);
  float u2= nano2_rand_uniform(r);
  float m= sqrtf(-2.0f * logf(u1));
  float z0= m * cosf(2.0f * (float)M_PI * u2);
  float z1= m * sinf(2.0f * (float)M_PI * u2);
  r->spare= z1; r->have_spare = 1; return z0;
}


//fill helps
void nano2_fill_gaussian(float* dst, size_t n, float std, struct Nano2RNG* rng){
  for(size_t i=0;i<n;++i) dst[i] = std * nano2_randn(rng);
}
void nano2_fill_constant(float* dst, size_t n, float v){
  for(size_t i=0;i<n;++i) dst[i] = v;
}
void nano2_fill_zeros(float* dst, size_t n){ nano2_fill_constant(dst,n,0.0f); }
void nano2_fill_ones (float* dst, size_t n){ nano2_fill_constant(dst,n,1.0f); }

//sinusoidal position tables
//gives half-dim sin/cos tables so the model can mix them in the embed step
//sin_out[t, i] = sin( t / 10000^{2i/D} ), cos_out[t, i] = cos( … ), for i in [0, D/2)
void nano2_make_sincos_tables(int T, int D, float* sin_out, float* cos_out){
  const int H = D/2; //i assume even D
  for(int t=0;t<T;++t){
    for(int i=0;i<H;++i){
      float exponent= (2.0f * (float)i) / (float)D;
      float denom= powf(10000.0f, exponent);
      float ang= (float)t / denom;
      sin_out[t*H +i] = sinf(ang);
      cos_out[t*H +i] = cosf(ang);
    }
  }
}

