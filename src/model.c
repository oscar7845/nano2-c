//model parameter/buffer structs 
//and alloc/init 
//and sinusoidal tables on device
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime_api.h>

//(config.c)
struct Config{
  char train_path[512];
  char val_path[512];
  int seq_len;
  int batch_size;
  int vocab_size;
  int d_model;
  int ffn_mult;
  double lr;
  double weight_decay;
  double clip_grad_norm;
  int seed;
  int top_k;
};


//fw decls for tensor.c helpers
void* nano2_malloc_device(size_t bytes);
void nano2_free_device(void* p);
void nano2_copy_host_to_device(void* dst_dev, const void* src_host, size_t bytes);
void nano2_memset_device(void* dst_dev, int value, size_t bytes);
void* nano2_malloc_host(size_t bytes);
void nano2_free_host(void* p);

//must be complete here because we instantiate it below
struct Nano2RNG{
  uint32_t s;
  int have_spare;
  float spare;
};
void nano2_rng_seed(struct Nano2RNG* r, uint32_t seed);
float nano2_rand_uniform(struct Nano2RNG* r);
float nano2_randn(struct Nano2RNG* r);
void nano2_fill_gaussian(float* dst, size_t n, float std, struct Nano2RNG* rng);
void nano2_fill_zeros(float* dst, size_t n);
void nano2_fill_ones(float* dst, size_t n);
void nano2_make_sincos_tables(int T, int D, float* sin_out, float* cos_out);


//model params/buffer structs
struct Params {
  //tied token embedding
  float *E; // [V, D]
  //Block 1 (pre-LN)
  float *ln1_g, *ln1_b; // [D], [D]
  float *Wq, *Wk, *Wv, *Wo; // [D,D] each
  float *ln2_g, *ln2_b; // [D], [D]
  float *W1, *b1; // [D,F], [F]
  float *W2, *b2; // [F,D], [D]
};

struct Opt { // AdamW buffers (allocated now, used later)
  float *m; // [n_params]
  float *v; // [n_params]
};


struct Buffers { // forward workspaces
  float *x; // (B,T,D)
  float *x_ln1; // (B,T,D)
  float *q,*k,*v; // (B,T,D)
  float *scores; // (B,T,T)
  float *probs; // (B,T,T)
  float *attn_out; // (B,T,D)
  float *x_res1; // (B,T,D)
  float *x_ln2; // (B,T,D)
  float *ff1; // (B,T,F)
  float *ff2; // (B,T,D)
  float *x_res2; // (B,T,D)
  float *logits; // (B,T,V)
  float *rowmax; // (B,T)
  //saved stats/activations for backwards
  float *ln1_mean, *ln1_invstd; // [B*T]
  float *ln2_mean, *ln2_invstd; // [B*T]
  float *ff1_pre;               // [B*T, F] (pre-GELU)
};


struct Model {
  // dims (cached)
  int B,T,D,V,F;
  // params & optimizer & grads
  struct Params p;
  struct Opt opt;
  float *flat_params; // device contiguous params
  float *flat_grads; // device contiguous grads
  size_t n_params; // number of floats
  // positional tables (device)
  float *pos_sin; // [T, D/2]
  float *pos_cos; // [T, D/2]
  // forward buffers
  struct Buffers buf;
};


//helpers to carve flat buffer
static inline float* carve(float** base, size_t count){ float* p=*base; *base += count; return p; }


static size_t count_params(const struct Config* c){
  const int D = c->d_model;
  const int V = c->vocab_size;
  const int F = c->ffn_mult * D;
  size_t n = 0;
  n += (size_t)V * D; // E
  n += 2*(size_t)D; // ln1_g, ln1_b
  n += 4*(size_t)D * D; // Wq,Wk,Wv,Wo
  n += 2*(size_t)D; // ln2_g, ln2_b
  n += (size_t)D * F + F; // W1, b1
  n += (size_t)F * D + D; // W2, b2
  return n;
}

//allocate & initialize
void model_init(struct Model* M, const struct Config* c){
  memset(M, 0, sizeof(*M));
  M->B = c->batch_size; M->T = c->seq_len; M->D = c->d_model; M->V = c->vocab_size; M->F = c->ffn_mult * c->d_model;

  //allocate flat params on device
  M->n_params = count_params(c);
  size_t nbytes = M->n_params * sizeof(float);
  M->flat_params = (float*)nano2_malloc_device(nbytes);
  M->flat_grads = (float*)nano2_malloc_device(nbytes);
  nano2_memset_device(M->flat_grads, 0, nbytes);

  //Adam buffers (zero init)
  M->opt.m = (float*)nano2_malloc_device(nbytes);
  M->opt.v = (float*)nano2_malloc_device(nbytes);
  nano2_memset_device(M->opt.m, 0, nbytes);
  nano2_memset_device(M->opt.v, 0, nbytes);

  //carve pointers
  float* base = M->flat_params;
  const size_t VD = (size_t)M->V * (size_t)M->D;
  const size_t DD = (size_t)M->D * (size_t)M->D;
  const size_t DF = (size_t)M->D * (size_t)M->F;
  const size_t FD = (size_t)M->F * (size_t)M->D;

  M->p.E = carve(&base, VD);
  M->p.ln1_g = carve(&base, M->D);
  M->p.ln1_b = carve(&base, M->D);
  M->p.Wq = carve(&base, DD);
  M->p.Wk = carve(&base, DD);
  M->p.Wv = carve(&base, DD);
  M->p.Wo = carve(&base, DD);
  M->p.ln2_g = carve(&base, M->D); 
  M->p.ln2_b = carve(&base, M->D);
  M->p.W1 = carve(&base, DF);
  M->p.b1 = carve(&base, M->F);
  M->p.W2 = carve(&base, FD);
  M->p.b2 = carve(&base, M->D);

  //host init then copy once to device flat
  float* h = (float*)nano2_malloc_host(nbytes);

  //set up a mirror of the carving on host memory
  float* hbase = h;
  float *hE = carve(&hbase, VD);
  float *hln1_g = carve(&hbase, M->D);
  float *hln1_b = carve(&hbase, M->D);
  float *hWq = carve(&hbase, DD);
  float *hWk = carve(&hbase, DD);
  float *hWv = carve(&hbase, DD);
  float *hWo = carve(&hbase, DD);
  float *hln2_g = carve(&hbase, M->D);
  float *hln2_b = carve(&hbase, M->D);
  float *hW1 = carve(&hbase, DF);
  float *hb1 = carve(&hbase, M->F);
  float *hW2 = carve(&hbase, FD);
  float *hb2 = carve(&hbase, M->D);

  struct Nano2RNG rng; nano2_rng_seed(&rng, (uint32_t)c->seed);

  //inits
  //Embedding: N(0, 0.02)
  nano2_fill_gaussian(hE, VD, 0.02f, &rng);

  //LayerNorm scales 1, biases 0
  nano2_fill_ones (hln1_g, M->D); nano2_fill_zeros(hln1_b, M->D);
  nano2_fill_ones (hln2_g, M->D); nano2_fill_zeros(hln2_b, M->D);

  //Linear weights: N(0, 1/sqrt(in_dim))
  float std_D = 1.0f / sqrtf((float)M->D);
  float std_F = 1.0f / sqrtf((float)M->F);
  nano2_fill_gaussian(hWq, DD, std_D, &rng);
  nano2_fill_gaussian(hWk, DD, std_D, &rng);
  nano2_fill_gaussian(hWv, DD, std_D, &rng);
  nano2_fill_gaussian(hWo, DD, std_D, &rng);
  nano2_fill_gaussian(hW1, DF, std_D, &rng); // in_dim=D
  nano2_fill_gaussian(hW2, FD, std_F, &rng); // in_dim=F

  nano2_fill_zeros(hb1, M->F);
  nano2_fill_zeros(hb2, M->D);

  //copy once to device
  nano2_copy_host_to_device(M->flat_params, h, nbytes);
  nano2_free_host(h);

  //positional tables on device
  const int H = M->D/2;
  float* hsin = (float*)nano2_malloc_host((size_t)M->T * H * sizeof(float));
  float* hcos = (float*)nano2_malloc_host((size_t)M->T * H * sizeof(float));
  nano2_make_sincos_tables(M->T, M->D, hsin, hcos);
  M->pos_sin = (float*)nano2_malloc_device((size_t)M->T * H * sizeof(float));
  M->pos_cos = (float*)nano2_malloc_device((size_t)M->T * H * sizeof(float));
  nano2_copy_host_to_device(M->pos_sin, hsin, (size_t)M->T * H * sizeof(float));
  nano2_copy_host_to_device(M->pos_cos, hcos, (size_t)M->T * H * sizeof(float));
  nano2_free_host(hsin); nano2_free_host(hcos);

  //forward buffers
  const size_t BT = (size_t)M->B * (size_t)M->T;
  const size_t BTD = BT * (size_t)M->D;
  const size_t BTT = (size_t)M->B * (size_t)M->T * (size_t)M->T;
  const size_t BTV = BT * (size_t)M->V;
  const size_t BTF = BT * (size_t)M->F;

  M->buf.x = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.x_ln1 = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.q = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.k = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.v = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.scores = (float*)nano2_malloc_device(BTT * sizeof(float));
  M->buf.probs = (float*)nano2_malloc_device(BTT * sizeof(float));
  M->buf.attn_out = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.x_res1 = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.x_ln2 = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.ff1 = (float*)nano2_malloc_device(BTF * sizeof(float));
  M->buf.ff2 = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.x_res2 = (float*)nano2_malloc_device(BTD * sizeof(float));
  M->buf.logits = (float*)nano2_malloc_device(BTV * sizeof(float));
  M->buf.rowmax = (float*)nano2_malloc_device(BT * sizeof(float));
  //backward bufers
  M->buf.ln1_mean   = (float*)nano2_malloc_device(BT * sizeof(float));
  M->buf.ln1_invstd = (float*)nano2_malloc_device(BT * sizeof(float));
  M->buf.ln2_mean   = (float*)nano2_malloc_device(BT * sizeof(float));
  M->buf.ln2_invstd = (float*)nano2_malloc_device(BT * sizeof(float));
  M->buf.ff1_pre    = (float*)nano2_malloc_device(BTF * sizeof(float));
}

static void free_buf(struct Buffers* b){
  if(!b) return;
  if(b->x)        nano2_free_device(b->x),        b->x = NULL;
  if(b->x_ln1)    nano2_free_device(b->x_ln1),    b->x_ln1 = NULL;
  if(b->q)        nano2_free_device(b->q),        b->q = NULL;
  if(b->k)        nano2_free_device(b->k),        b->k = NULL;
  if(b->v)        nano2_free_device(b->v),        b->v = NULL;
  if(b->scores)   nano2_free_device(b->scores),   b->scores = NULL;
  if(b->probs)    nano2_free_device(b->probs),    b->probs = NULL;
  if(b->attn_out) nano2_free_device(b->attn_out), b->attn_out = NULL;
  if(b->x_res1)   nano2_free_device(b->x_res1),   b->x_res1 = NULL;
  if(b->x_ln2)    nano2_free_device(b->x_ln2),    b->x_ln2 = NULL;
  if(b->ff1)      nano2_free_device(b->ff1),      b->ff1 = NULL;
  if(b->ff2)      nano2_free_device(b->ff2),      b->ff2 = NULL;
  if(b->x_res2)   nano2_free_device(b->x_res2),   b->x_res2 = NULL;
  if(b->logits)   nano2_free_device(b->logits),   b->logits = NULL;
  if(b->rowmax)   nano2_free_device(b->rowmax),   b->rowmax = NULL;
  //backward
  if(b->ln1_mean)   nano2_free_device(b->ln1_mean),   b->ln1_mean=NULL;
  if(b->ln1_invstd) nano2_free_device(b->ln1_invstd), b->ln1_invstd=NULL;
  if(b->ln2_mean)   nano2_free_device(b->ln2_mean),   b->ln2_mean=NULL;
  if(b->ln2_invstd) nano2_free_device(b->ln2_invstd), b->ln2_invstd=NULL;
  if(b->ff1_pre)    nano2_free_device(b->ff1_pre),    b->ff1_pre=NULL;
}

void model_free(struct Model* M){
  if(!M) return;

  //params & grads
  if(M->flat_params) nano2_free_device(M->flat_params), M->flat_params = NULL;
  if(M->flat_grads)  nano2_free_device(M->flat_grads),  M->flat_grads  = NULL;

  //optimizer
  if(M->opt.m) nano2_free_device(M->opt.m), M->opt.m = NULL;
  if(M->opt.v) nano2_free_device(M->opt.v), M->opt.v = NULL;

  //positional tables
  if(M->pos_sin) nano2_free_device(M->pos_sin), M->pos_sin = NULL;
  if(M->pos_cos) nano2_free_device(M->pos_cos), M->pos_cos = NULL;

  //forward buffers
  free_buf(&M->buf);

  //clearer counters
  M->n_params = 0;
  M->B = M->T = M->D = M->V = M->F = 0;
}

void model_log_summary(const struct Model* M, const struct Config* c){
  (void)c; //not strictly needed, but keep to match prototype
  if(!M){ printf("Model: (null)\n"); return; }

  double nparams = (double)M->n_params;
  double bytes   = nparams * sizeof(float);
  double mib     = bytes / (1024.0 * 1024.0);

  printf("model summary:\n");
  printf("  dims: B=%d T=%d D=%d V=%d F=%d\n", M->B, M->T, M->D, M->V, M->F);
  printf("  parameters: %.0f (%.2f MiB as fp32)\n", nparams, mib);

  //quick layout log (matches carving order)
  printf("  tensors:\n");
  printf("    E           : [%d, %d]\n", M->V, M->D);
  printf("    ln1_g/ln1_b : [%d] / [%d]\n", M->D, M->D);
  printf("    Wq/Wk/Wv/Wo : 4 x [%d, %d]\n", M->D, M->D);
  printf("    ln2_g/ln2_b : [%d] / [%d]\n", M->D, M->D);
  printf("    W1/b1       : [%d, %d] / [%d]\n", M->D, M->F, M->F);
  printf("    W2/b2       : [%d, %d] / [%d]\n", M->F, M->D, M->D);
  printf("  pos tables    : sin/cos [%d, %d]\n", M->T, M->D/2);
}

